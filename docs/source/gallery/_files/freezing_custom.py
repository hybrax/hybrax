"""Freezing part of a reaction module: a fixed encoder, a trainable head.

Split one MLP into two fields and tag them differently: `frozen_field()` for
the part you do not want the optimizer to touch, `trainable_field()` for the
part you do. `partition_trainable` (what the optimizer actually sees) reads
these tags directly off the module, and `print_trainable_structure` lets you
check the split before committing to a long run.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bp_train import (
    EstimatedScales,
    ReactionInputs,
    ReactionOutputs,
    UserReactionModule,
    frozen_field,
    trainable_field,
)


class FrozenEncoderReactionModule(UserReactionModule):
    """A fixed feature encoder feeding a small trainable readout head."""

    encoder: eqx.nn.MLP = frozen_field()
    head: eqx.nn.Linear = trainable_field()

    def __init__(self, *, key, n_hidden=16, **scale_kwargs):
        super().__init__(**scale_kwargs)
        key_enc, key_head = jax.random.split(key)
        self.encoder = eqx.nn.MLP(
            in_size=self.n_modeled_RMCs, out_size=n_hidden, width_size=n_hidden,
            depth=2, key=key_enc)
        self.head = eqx.nn.Linear(
            in_features=n_hidden, out_features=self.n_modeled_BiologicalOde_rates,
            key=key_head)

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        features = self.encoder(inputs.SCL_modeled_RMCs)
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=self.head(features),
            SCL_modeled_FVCs_rates=jnp.zeros(0),
        )


def build_reaction_module(*, seed, **kwargs):
    scale_kwargs = {k: v for k, v in kwargs.items() if k.startswith("SCALE_")}
    return FrozenEncoderReactionModule(key=jax.random.key(seed), **scale_kwargs)


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
            max(runtime_data.initial_volume(i) for i in range(n_processes))),
        SCALE_modeled_FVCs_cumulative=empty,
        SCALE_modeled_FVCs_rates=empty,
        SCALE_controlled_FVCs_cumulative=empty,
        SCALE_controlled_FVCs_rates=empty,
        SCALE_controlled_PVs=empty,
        SCALE_controlled_FVCs_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_controlled_FVCs)), 1.0),
        SCALE_modeled_FVCs_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_modeled_FVCs)), 1.0),
    )
