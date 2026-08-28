"""Pooling data across products: an ensemble of GPs, each anchored to a real
subsample of training data, reading a product-identity controlled PV.

Builds on gaussian_process.md's GPReactionModule, sharing its two real-data
grounding pieces: ``centers`` are real measured states (plus, here,
``is_new_product``), and ``targets`` are real rate estimates from
hybrax.format.splines's pseudobatch machinery, not learned. Two changes on
top of the single-GP page:

1. K independent GP heads instead of one, each anchored to a bootstrap
   subsample of REAL (state, product-identity) -> (real rate) pairs, not
   free trainable vectors. Final prediction is the mean across heads; the
   spread across heads stands in for predictive uncertainty. This mirrors
   Helleckes et al. 2024's "mean averaging ensemble... 30 GP models, each
   subsampling 50% of the training data experiments" (scaled down: K heads
   / n_anchors here, not 30, and subsampled at the point level rather than
   the experiment level -- both simplifications made for tractability
   inside hybrax.train's per-solver-step reaction-module call, not hidden).
2. A constant-valued controlled process variable, `is_new_product`, encodes
   which product a given process belongs to: a one-hot identity feature,
   using only existing hybrax machinery (no framework change).
   Shipped as a StaticVariable-valued ProcessVariable directly in data.json,
   baked in by the generator's build_demo_products().
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from hybrax.format.mechanistic import build_rhs_ode
from hybrax.format.splines import build_backtransform_spline, build_pseudobatch_transform
from hybrax.train import (
    DefaultLossModule,
    EstimatedScales,
    LossInputs,
    LossOutputs,
    ReactionInputs,
    ReactionOutputs,
    UserReactionModule,
    frozen_field,
    trainable_field,
)


class EnsembleGPReactionModule(UserReactionModule):
    centers: jax.Array = frozen_field()  # (K, M, n_features), REAL states + is_new_product
    targets: jax.Array = frozen_field()  # (K, M, n_rates), REAL rate estimates
    log_lengthscale: jax.Array = trainable_field()  # (K, n_features)
    log_output_scale: jax.Array = trainable_field()  # (K,)
    log_noise: jax.Array = trainable_field()  # (K,)

    def __init__(
        self, *, key, centers_pool, targets_pool, n_members=5, n_anchors=15, **scale_kwargs
    ):
        super().__init__(**scale_kwargs)
        n_features = centers_pool.shape[1]

        seed_val = int(jax.random.randint(key, (), 0, 2**31 - 1))
        rng = np.random.default_rng(seed_val)
        n_total = centers_pool.shape[0]
        centers_list, targets_list = [], []
        for _ in range(n_members):
            half = rng.choice(n_total, size=max(n_total // 2, 1), replace=False)
            m = min(n_anchors, half.size)
            chosen = rng.choice(half, size=m, replace=False)
            centers_list.append(centers_pool[chosen])
            targets_list.append(targets_pool[chosen])
        # Pad every member to the same M (repeat last row) so they stack into one array.
        max_m = max(c.shape[0] for c in centers_list)

        def pad(arrs):
            return [
                np.concatenate([a, np.repeat(a[-1:], max_m - a.shape[0], axis=0)], axis=0)
                for a in arrs
            ]

        self.centers = jnp.asarray(np.stack(pad(centers_list), axis=0))
        self.targets = jnp.asarray(np.stack(pad(targets_list), axis=0))

        self.log_lengthscale = jnp.zeros((n_members, n_features))
        self.log_output_scale = jnp.zeros(n_members)
        self.log_noise = jnp.full((n_members,), -2.0)

    @staticmethod
    def _kernel(a, b, log_ls, log_out):
        diff = (a[:, None, :] - b[None, :, :]) / jnp.exp(log_ls)
        return jnp.exp(log_out) * jnp.exp(-0.5 * jnp.sum(diff**2, axis=-1))

    def _chol(self, centers_k, log_ls_k, log_out_k, log_noise_k):
        k_zz = self._kernel(centers_k, centers_k, log_ls_k, log_out_k)
        k_zz = k_zz + jnp.exp(log_noise_k) * jnp.eye(centers_k.shape[0])
        return jsl.cho_factor(k_zz, lower=True)

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        x_phys = jnp.concatenate([inputs.SCL_modeled_RMCs, inputs.SCL_controlled_PVs])[None, :]

        def per_member(centers_k, log_ls_k, log_out_k, log_noise_k, targets_k):
            chol = self._chol(centers_k, log_ls_k, log_out_k, log_noise_k)
            k_xz = self._kernel(x_phys, centers_k, log_ls_k, log_out_k)
            return (k_xz @ jsl.cho_solve(chol, targets_k))[0]

        means = jax.vmap(per_member)(
            self.centers,
            self.log_lengthscale,
            self.log_output_scale,
            self.log_noise,
            self.targets,
        )
        mean = jnp.mean(means, axis=0)
        rate_std = jnp.std(means, axis=0)
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=mean,
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            SCL_modeled_Inflows_rates=jnp.zeros(0),
            auxiliary={"rate_std": rate_std},
        )

    def marginal_nll(self) -> jax.Array:
        """Real GP negative log marginal likelihood, averaged across heads.

        Fits each head's kernel hyperparameters the way a textbook GP does:
        maximizing the probability of that head's real (centers, targets)
        pairs, not the downstream trajectory.
        """

        def per_member_nll(centers_k, log_ls_k, log_out_k, log_noise_k, targets_k):
            chol = self._chol(centers_k, log_ls_k, log_out_k, log_noise_k)
            alpha = jsl.cho_solve(chol, targets_k)
            n_points, n_rates = targets_k.shape
            data_fit = 0.5 * jnp.sum(targets_k * alpha)
            log_det = n_rates * jnp.sum(jnp.log(jnp.diagonal(chol[0])))
            n_terms = n_points * n_rates
            return (data_fit + log_det + 0.5 * n_terms * jnp.log(2 * jnp.pi)) / n_terms

        nlls = jax.vmap(per_member_nll)(
            self.centers, self.log_lengthscale, self.log_output_scale, self.log_noise, self.targets
        )
        return jnp.mean(nlls)


def build_reaction_module(*, seed, training_parent_collection, **kwargs):
    scale_kwargs = {k: v for k, v in kwargs.items() if k.startswith("SCALE_")}
    first_process = next(iter(training_parent_collection.processes.values()))
    rhs = build_rhs_ode(first_process)
    rmc_names = list(rhs.name_modeled_RMCs)
    rmc_scaler = scale_kwargs["SCALE_modeled_RMCs"]
    rate_scaler = scale_kwargs["SCALE_modeled_BiologicalOde_rates"]

    centers_list, targets_list = [], []
    for process in training_parent_collection.processes.values():
        process.pseudobatch_transform = build_pseudobatch_transform(process)
        meas_times = np.asarray(
            process.reactor_medium.components["biomass"].concentration.times
        )
        splines = {name: build_backtransform_spline(process, name) for name in rmc_names}
        values = {name: np.asarray(splines[name](meas_times)) for name in rmc_names}
        derivatives = {
            name: np.asarray(splines[name].derivative()(meas_times)) for name in rmc_names
        }
        biomass = values["biomass"]

        # Specific rates: "<species>' = q_<species> * biomass". Drop the first 3
        # (of 17) samples: near the small inoculum, a derivative over a near-zero
        # biomass is fragile (rate std drops 8.3 -> 2.5 dropping these 3, vs. 4.4
        # dropping just the first).
        n_drop = 3
        raw_state = np.stack([values[name][n_drop:] for name in rmc_names], axis=1)
        raw_rate = np.stack(
            [derivatives[name][n_drop:] / biomass[n_drop:] for name in rmc_names], axis=1
        )

        scl_state = np.asarray(rmc_scaler.scale_value(jnp.asarray(raw_state)))
        is_new = float(process.process_variables["is_new_product"].values.value)
        pv_col = np.full((scl_state.shape[0], 1), is_new)
        centers_list.append(np.concatenate([scl_state, pv_col], axis=1))

        # A rate is a derivative, so it uses scale_derivative: an affine
        # scaler's offset (if any) is never subtracted from it.
        targets_list.append(np.asarray(rate_scaler.scale_derivative(jnp.asarray(raw_rate))))

    centers_pool = np.concatenate(centers_list, axis=0)
    targets_pool = np.concatenate(targets_list, axis=0)

    return EnsembleGPReactionModule(
        key=jax.random.key(seed),
        centers_pool=centers_pool,
        targets_pool=targets_pool,
        **scale_kwargs,
    )


class EnsembleGPLossModule(DefaultLossModule):
    """The usual per-target trajectory loss, plus the ensemble's own mean
    marginal likelihood across heads: both drive the same `hybrax.train`
    gradient step.
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


def build_loss_module(*, target_names, process_names, config, seed, training_parent_collection):
    del process_names, config, seed, training_parent_collection
    return EnsembleGPLossModule(target_names=tuple(target_names))


def estimate_all_scales(runtime_data, target_names, config):
    del target_names, config
    rhs = runtime_data.rhs_ode
    controls_store = runtime_data.controls_store
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

    n_PV = len(controls_store.name_controlled_PVs)
    empty = jnp.zeros(0)
    return EstimatedScales(
        SCALE_modeled_RMCs=jnp.asarray([rmc_scale[n] for n in rhs.name_modeled_RMCs]),
        SCALE_modeled_BiologicalOde_rates=jnp.asarray(rate_scale),
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
        SCALE_controlled_PVs=jnp.ones(n_PV),
        SCALE_controlled_Inflows_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_controlled_Inflows)), 1.0
        ),
        SCALE_modeled_Inflows_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_modeled_Inflows)), 1.0
        ),
    )
