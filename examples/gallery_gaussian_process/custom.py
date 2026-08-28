"""A Gaussian-process reaction module.

A closed-form GP posterior (mean and variance, via a Cholesky solve) occupies
the same slot a neural network normally would. ``centers`` and ``targets``
hold real data: every real measured state and a real rate estimate at that
state, built from ``hybrax.format.splines``'s pseudobatch machinery. Only the
kernel hyperparameters are fit, by a genuine marginal-likelihood loss term
(``build_loss_module`` below) running alongside the trajectory loss inside
``hybrax.train``'s own training loop.

The predictive variance goes into ``ReactionOutputs.auxiliary``, which
hybrax.train threads straight into ``predictions.csv`` (see ``estimate_all_scales``
below for the rest: verbatim from Tutorial 4).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from hybrax.format.mechanistic import build_rhs_ode
from hybrax.format.splines import (
    build_backtransform_spline,
    build_pseudobatch_transform,
)
from hybrax.train import (
    DefaultLossModule,
    EstimatedScales,
    LossInputs,
    LossOutputs,
    ReactionInputs,
    ReactionOutputs,
    UserLossModule,
    RateModule,
    frozen_field,
    trainable_field,
)


class GPReactionModule(RateModule):
    centers: jax.Array = frozen_field()  # Z, real states, (n_points, n_features)
    targets: jax.Array = frozen_field()  # y, real rate estimates, (n_points, n_rates)
    log_lengthscale: jax.Array = trainable_field()  # (n_features,), ARD
    log_output_scale: jax.Array = trainable_field()  # scalar kernel amplitude
    log_noise: jax.Array = trainable_field()  # scalar jitter

    def __init__(self, *, centers, targets, **scale_kwargs):
        super().__init__(**scale_kwargs)
        n_features = self.n_modeled_RMCs
        self.centers = centers
        self.targets = targets
        self.log_lengthscale = jnp.zeros(n_features)
        self.log_output_scale = jnp.array(0.0)
        self.log_noise = jnp.array(-2.0)

    def _kernel(self, a, b):
        diff = (a[:, None, :] - b[None, :, :]) / jnp.exp(self.log_lengthscale)
        sq_dist = jnp.sum(diff**2, axis=-1)
        return jnp.exp(self.log_output_scale) * jnp.exp(-0.5 * sq_dist)

    def _chol(self):
        k_zz = self._kernel(self.centers, self.centers)
        k_zz = k_zz + jnp.exp(self.log_noise) * jnp.eye(self.centers.shape[0])
        return jsl.cho_factor(k_zz, lower=True)

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        x = inputs.SCL_modeled_RMCs[None, :]
        chol = self._chol()
        k_xz = self._kernel(x, self.centers)
        mean = (k_xz @ jsl.cho_solve(chol, self.targets))[0]
        v = jsl.cho_solve(chol, k_xz[0])
        var = jnp.exp(self.log_output_scale) - k_xz[0] @ v
        SCL_rate_std = jnp.sqrt(jnp.clip(var, 1e-12)) * jnp.ones_like(mean)
        # predictions.csv reports q_* in RAW units (RAW_modeled_ReactionOde_rates),
        # but auxiliary values pass through unscaled: convert here so rate_std is
        # comparable to q_* in the same file, not left in SCL units next to RAW ones.
        RAW_rate_std = self.unscale_modeled_ReactionOde_rates(SCL_rate_std)
        return ReactionOutputs(
            SCL_modeled_ReactionOde_rates=mean,
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            SCL_modeled_Inflows_rates=jnp.zeros(0),
            auxiliary={"rate_std": RAW_rate_std},
        )

    def marginal_nll(self) -> jax.Array:
        """Real GP negative log marginal likelihood over centers/targets.

        Fits ``log_lengthscale``/``log_output_scale``/``log_noise`` the way a
        textbook GP does: maximizing the probability of the real training
        pairs, not the downstream trajectory.
        """
        chol = self._chol()
        alpha = jsl.cho_solve(chol, self.targets)  # (n_points, n_rates)
        n_points, n_rates = self.targets.shape
        data_fit = 0.5 * jnp.sum(self.targets * alpha)
        log_det = n_rates * jnp.sum(jnp.log(jnp.diagonal(chol[0])))
        n_terms = n_points * n_rates
        return (data_fit + log_det + 0.5 * n_terms * jnp.log(2 * jnp.pi)) / n_terms


def build_reaction_module(*, seed, training_parent_collection, **kwargs):
    del seed
    scale_kwargs = {k: v for k, v in kwargs.items() if k.startswith("SCALE_")}
    first_process = next(iter(training_parent_collection.processes.values()))
    rhs = build_rhs_ode(first_process)
    rmc_names = list(rhs.name_modeled_RMCs)
    rmc_scaler = scale_kwargs["SCALE_modeled_RMCs"]
    rate_scaler = scale_kwargs["SCALE_modeled_ReactionOde_rates"]

    centers_list = []
    targets_list = []
    for process in training_parent_collection.processes.values():
        process.pseudobatch_transform = build_pseudobatch_transform(process)
        meas_times = np.asarray(
            process.reactor_medium.components["biomass"].concentration.times
        )
        splines = {
            name: build_backtransform_spline(process, name) for name in rmc_names
        }
        values = {name: np.asarray(splines[name](meas_times)) for name in rmc_names}
        derivatives = {
            name: np.asarray(splines[name].derivative()(meas_times))
            for name in rmc_names
        }
        biomass = values["biomass"]

        # q_biomass/q_glucose/q_product are all specific rates: the declared
        # ODE is "<species>' = q_<species> * biomass" for every one of them.
        raw_state = np.stack([values[name] for name in rmc_names], axis=1)
        raw_rate = np.stack([derivatives[name] / biomass for name in rmc_names], axis=1)

        centers_list.append(np.asarray(rmc_scaler.scale_value(jnp.asarray(raw_state))))
        # A rate is a derivative, not a value: scale_derivative, not
        # scale_value, so an affine scaler's offset (if any) is never
        # subtracted from it.
        targets_list.append(
            np.asarray(rate_scaler.scale_derivative(jnp.asarray(raw_rate)))
        )

    centers = jnp.asarray(np.concatenate(centers_list, axis=0))
    targets = jnp.asarray(np.concatenate(targets_list, axis=0))

    return GPReactionModule(centers=centers, targets=targets, **scale_kwargs)


class GPLossModule(DefaultLossModule):
    """The usual per-target trajectory loss, plus the GP's own marginal
    likelihood: both drive the same `hybrax.train` gradient step.
    """

    nll_weight: float = eqx.field(static=True)

    def __init__(self, *, target_names, nll_weight=0.0005):
        super().__init__(target_names=target_names)
        self.nll_weight = nll_weight

    @property
    def loss_names(self):
        return (*self.target_names, "gp_nll")

    def __call__(self, inputs: LossInputs) -> LossOutputs:
        base = super().__call__(inputs)
        gp_nll = inputs.reaction_module.marginal_nll()
        return LossOutputs(
            named_losses={**base.named_losses, "gp_nll": self.nll_weight * gp_nll}
        )


def build_loss_module(
    *, target_names, process_names, config, seed, training_parent_collection
):
    del process_names, config, seed, training_parent_collection
    return GPLossModule(target_names=tuple(target_names))


def estimate_all_scales(runtime_data, target_names, config):
    del target_names, config
    rhs = runtime_data.rhs_ode
    n_processes = len(runtime_data.process_order)

    def max_abs_state(name):
        best = 0.0
        for i in range(n_processes):
            _, values = runtime_data.raw_state_trace(i, name)
            if values.size:
                best = max(best, float(np.max(np.abs(values))))
        return max(best, 1e-6)

    rmc_scale = {name: max_abs_state(name) for name in rhs.name_modeled_RMCs}

    def rate_scale_for(species):
        per_process = []
        for i in range(n_processes):
            c_times, c_values = runtime_data.raw_state_trace(i, species)
            x_times, x_values = runtime_data.raw_state_trace(i, "biomass")
            exposure = np.trapezoid(x_values, x_times)
            per_process.append(abs(c_values[-1] - c_values[0]) / max(exposure, 1e-9))
        return max(max(per_process), 1e-9)

    rate_scale = [rate_scale_for(name[2:]) for name in rhs.name_modeled_rates]

    empty = jnp.zeros(0)
    return EstimatedScales(
        SCALE_modeled_RMCs=jnp.asarray([rmc_scale[n] for n in rhs.name_modeled_RMCs]),
        SCALE_modeled_ReactionOde_rates=jnp.asarray(rate_scale),
        SCALE_V_in_cumulative=jnp.asarray(
            max(runtime_data.initial_volume(i) for i in range(n_processes))
        ),
        SCALE_modeled_Inflows_cumulative=empty,
        SCALE_modeled_Inflows_rates=empty,
        SCALE_modeled_Outflows_cumulative=empty,
        SCALE_modeled_Outflows_rates=empty,
        SCALE_controlled_Inflows_cumulative=empty,
        SCALE_controlled_Inflows_rates=empty,
        SCALE_controlled_Outflows_cumulative=empty,
        SCALE_controlled_Outflows_rates=empty,
        SCALE_controlled_PVs=empty,
        SCALE_controlled_Inflows_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_controlled_Inflows)), 1.0
        ),
        SCALE_modeled_Inflows_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_modeled_Inflows)), 1.0
        ),
    )
