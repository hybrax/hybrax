"""A Gaussian-process reaction module.

A closed-form sparse-GP posterior (mean AND variance, via a Cholesky solve
over a small set of trainable inducing points) occupies the same slot a
neural network normally would. Trained end-to-end by hybrax.train's own gradient
descent through the whole trajectory, unlike a standard GP's marginal-
likelihood fit: see the page for exactly how these differ, and why.

The predictive variance goes into ``ReactionOutputs.auxiliary``, which
hybrax.train threads straight into ``predictions.csv`` (see ``estimate_all_scales``
below for the rest: verbatim from Tutorial 4).
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from hybrax.train import (
    EstimatedScales,
    ReactionInputs,
    ReactionOutputs,
    UserReactionModule,
    trainable_field,
)


class GPReactionModule(UserReactionModule):
    centers: jax.Array = trainable_field()  # Z, (n_inducing, n_features)
    log_lengthscale: jax.Array = trainable_field()  # (n_features,), ARD
    log_output_scale: jax.Array = trainable_field()  # scalar kernel amplitude
    log_noise: jax.Array = trainable_field()  # scalar jitter
    pseudo_targets: jax.Array = trainable_field()  # y, (n_inducing, n_rates)

    def __init__(self, *, key, n_inducing=12, **scale_kwargs):
        super().__init__(**scale_kwargs)
        n_features = self.n_modeled_RMCs
        n_rates = self.n_modeled_BiologicalOde_rates
        key_c, key_y = jax.random.split(key)
        self.centers = jax.random.normal(key_c, (n_inducing, n_features))
        self.log_lengthscale = jnp.zeros(n_features)
        self.log_output_scale = jnp.array(0.0)
        self.log_noise = jnp.array(-2.0)
        self.pseudo_targets = jax.random.normal(key_y, (n_inducing, n_rates)) * 0.1

    def _kernel(self, a, b):
        diff = (a[:, None, :] - b[None, :, :]) / jnp.exp(self.log_lengthscale)
        sq_dist = jnp.sum(diff**2, axis=-1)
        return jnp.exp(self.log_output_scale) * jnp.exp(-0.5 * sq_dist)

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        x = inputs.SCL_modeled_RMCs[None, :]
        k_zz = self._kernel(self.centers, self.centers)
        k_zz = k_zz + jnp.exp(self.log_noise) * jnp.eye(self.centers.shape[0])
        k_xz = self._kernel(x, self.centers)
        chol = jsl.cho_factor(k_zz, lower=True)
        mean = (k_xz @ jsl.cho_solve(chol, self.pseudo_targets))[0]
        v = jsl.cho_solve(chol, k_xz[0])
        var = jnp.exp(self.log_output_scale) - k_xz[0] @ v
        rate_std = jnp.sqrt(jnp.clip(var, 1e-12)) * jnp.ones_like(mean)
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=mean,
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            SCL_modeled_Inflows_rates=jnp.zeros(0),
            auxiliary={"rate_std": rate_std},
        )


def build_reaction_module(*, seed, **kwargs):
    scale_kwargs = {k: v for k, v in kwargs.items() if k.startswith("SCALE_")}
    return GPReactionModule(key=jax.random.key(seed), **scale_kwargs)


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
        SCALE_controlled_PVs=empty,
        SCALE_controlled_Inflows_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_controlled_Inflows)), 1.0
        ),
        SCALE_modeled_Inflows_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_modeled_Inflows)), 1.0
        ),
    )
