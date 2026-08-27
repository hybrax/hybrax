"""Pooling data across products: an ensemble of GPs, each anchored to a real
subsample of training data, reading a product-identity controlled PV.

Builds on gaussian_process.md's GPReactionModule. Two changes:

1. K independent GP heads instead of one, each anchored to a bootstrap
   subsample of REAL training (state, product-identity) points, not free
   trainable vectors. Final prediction is the mean across heads; the spread
   across heads stands in for predictive uncertainty. This mirrors Helleckes
   et al. 2024's "mean averaging ensemble... 30 GP models, each subsampling
   50% of the training data experiments" (scaled down: K heads / n_anchors
   here, not 30, and subsampled at the point level rather than the
   experiment level -- both simplifications made for tractability inside
   hybrax.train's per-solver-step reaction-module call, not hidden).
2. A constant-valued controlled process variable, `is_new_product`, encodes
   which product a given process belongs to: a one-hot identity feature,
   using only existing hybrax machinery (no framework change).
   Shipped as a StaticVariable-valued ProcessVariable directly in data.json,
   baked in by the generator's build_demo_products().
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from hybrax.format.mechanistic import build_rhs_ode
from hybrax.train import (
    EstimatedScales,
    ReactionInputs,
    ReactionOutputs,
    UserReactionModule,
    frozen_field,
    trainable_field,
)


class EnsembleGPReactionModule(UserReactionModule):
    centers: jax.Array = frozen_field()  # (K, M, n_features), REAL data
    log_lengthscale: jax.Array = trainable_field()  # (K, n_features)
    log_output_scale: jax.Array = trainable_field()  # (K,)
    log_noise: jax.Array = trainable_field()  # (K,)
    pseudo_targets: jax.Array = trainable_field()  # (K, M, n_rates), still learned:
    #   rates are never directly observed, only inferred through the ODE fit,
    #   unlike the real X locations, which are real measured states.

    def __init__(self, *, key, data_pool, n_members=5, n_anchors=15, **scale_kwargs):
        super().__init__(**scale_kwargs)
        n_features = data_pool.shape[1]
        n_rates = self.n_modeled_BiologicalOde_rates

        seed_val = int(jax.random.randint(key, (), 0, 2**31 - 1))
        rng = np.random.default_rng(seed_val)
        n_total = data_pool.shape[0]
        centers_list = []
        for _ in range(n_members):
            half = rng.choice(n_total, size=max(n_total // 2, 1), replace=False)
            m = min(n_anchors, half.size)
            chosen = rng.choice(half, size=m, replace=False)
            centers_list.append(data_pool[chosen])
        # Pad every member to the same M (repeat last row) so they stack into one array.
        max_m = max(c.shape[0] for c in centers_list)
        padded = [
            np.concatenate([c, np.repeat(c[-1:], max_m - c.shape[0], axis=0)], axis=0)
            for c in centers_list
        ]
        self.centers = jnp.asarray(np.stack(padded, axis=0))

        key_y = jax.random.split(key, 1)[0]
        self.log_lengthscale = jnp.zeros((n_members, n_features))
        self.log_output_scale = jnp.zeros(n_members)
        self.log_noise = jnp.full((n_members,), -2.0)
        self.pseudo_targets = (
            jax.random.normal(key_y, (n_members, max_m, n_rates)) * 0.1
        )

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        x_phys = jnp.concatenate([inputs.SCL_modeled_RMCs, inputs.SCL_controlled_PVs])

        def per_member(centers_k, log_ls_k, log_out_k, log_noise_k, targets_k):
            diff_zz = (centers_k[:, None, :] - centers_k[None, :, :]) / jnp.exp(
                log_ls_k
            )
            k_zz = jnp.exp(log_out_k) * jnp.exp(-0.5 * jnp.sum(diff_zz**2, axis=-1))
            k_zz = k_zz + jnp.exp(log_noise_k) * jnp.eye(centers_k.shape[0])
            diff_xz = (x_phys[None, :] - centers_k) / jnp.exp(log_ls_k)
            k_xz = jnp.exp(log_out_k) * jnp.exp(-0.5 * jnp.sum(diff_xz**2, axis=-1))
            chol = jsl.cho_factor(k_zz, lower=True)
            return k_xz @ jsl.cho_solve(chol, targets_k)

        means = jax.vmap(per_member)(
            self.centers,
            self.log_lengthscale,
            self.log_output_scale,
            self.log_noise,
            self.pseudo_targets,
        )
        mean = jnp.mean(means, axis=0)
        rate_std = jnp.std(means, axis=0)
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=mean,
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            SCL_modeled_Inflows_rates=jnp.zeros(0),
            auxiliary={"rate_std": rate_std},
        )


def build_reaction_module(*, seed, training_parent_collection, **kwargs):
    scale_kwargs = {k: v for k, v in kwargs.items() if k.startswith("SCALE_")}
    first_process = next(iter(training_parent_collection.processes.values()))
    rhs = build_rhs_ode(first_process)
    rmc_names = list(rhs.name_modeled_RMCs)
    scaler = scale_kwargs["SCALE_modeled_RMCs"]

    pool = []
    for name, process in training_parent_collection.processes.items():
        traces = [
            (
                np.asarray(process.reactor_medium.components[n].concentration.times),
                np.asarray(process.reactor_medium.components[n].concentration.values),
            )
            for n in rmc_names
        ]
        values = np.stack([tr[1] for tr in traces], axis=1)  # RAW, (n_t, n_rmc)
        scl_values = np.asarray(scaler.scale_value(jnp.asarray(values)))
        is_new = float(process.process_variables["is_new_product"].values.value)
        pv_col = np.full((scl_values.shape[0], 1), is_new)
        pool.append(np.concatenate([scl_values, pv_col], axis=1))
    pool = np.concatenate(pool, axis=0)

    return EnsembleGPReactionModule(
        key=jax.random.key(seed), data_pool=pool, **scale_kwargs
    )


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
