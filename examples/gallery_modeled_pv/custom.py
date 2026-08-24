"""Modeled process variable: two constant specific rates, no kinetic
structure at all. The point of this page is entirely in the dataset's own
declaration of glyco_frac as a modeled (uncontrolled) process variable, not in
this reaction module.
"""

import jax
import jax.numpy as jnp
import numpy as np

from hybrax.train import (
    EstimatedScales,
    ReactionOutputs,
    UserReactionModule,
    trainable_field,
)


class ModeledPVModule(UserReactionModule):
    log_q_biomass: jax.Array = trainable_field()
    log_r_glyco_frac: jax.Array = trainable_field()

    def __init__(self, **scale_kwargs):
        super().__init__(**scale_kwargs)
        # Deliberately mediocre starting guesses: the point is that they move.
        self.log_q_biomass = jnp.log(jnp.asarray(0.05))
        self.log_r_glyco_frac = jnp.log(jnp.asarray(0.1))

    def __call__(self, t, inputs) -> ReactionOutputs:
        del t, inputs   # every rate here is a plain constant, no state read
        RAW_rates = jnp.array([
            jnp.exp(self.log_q_biomass),
            jnp.exp(self.log_r_glyco_frac),
        ])
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=self.scale_modeled_BiologicalOde_rates(
                RAW_rates),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            SCL_modeled_Inflows_rates=jnp.zeros(0),
        )


def build_reaction_module(**kwargs):
    return ModeledPVModule(
        **{k: v for k, v in kwargs.items() if k.startswith("SCALE_")})


def estimate_all_scales(runtime_data, target_names, config):
    del target_names, config
    rhs = runtime_data.rhs_ode
    n_processes = len(runtime_data.process_order)
    span = max(end - start for start, end in
              (runtime_data.time_bounds(i) for i in range(n_processes)))

    def max_abs_state(name):
        best = 0.0
        for i in range(n_processes):
            _, values = runtime_data.raw_state_trace(i, name)
            if values.size:
                best = max(best, float(np.max(np.abs(values))))
        return max(best, 1e-6)

    rmc_scale = {name: max_abs_state(name) for name in rhs.name_modeled_RMCs}
    pv_scale = {name: max_abs_state(name) for name in rhs.name_modeled_PVs}
    biomass = rmc_scale["biomass"]
    empty = jnp.zeros(0)

    def rate_scale(rate_name):
        state_name = rate_name[2:]   # strip "q_" or "r_"
        if state_name in rmc_scale:
            return rmc_scale[state_name] / (biomass * span)
        return pv_scale[state_name] / span

    return EstimatedScales(
        SCALE_modeled_RMCs=jnp.asarray([rmc_scale[n] for n in rhs.name_modeled_RMCs]),
        SCALE_modeled_PVs=jnp.asarray([pv_scale[n] for n in rhs.name_modeled_PVs]),
        SCALE_modeled_BiologicalOde_rates=jnp.asarray(
            [rate_scale(n) for n in rhs.name_modeled_rates]),
        SCALE_V_in_cumulative=jnp.asarray(
            max(runtime_data.initial_volume(i) for i in range(n_processes))),
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
            jnp.abs(jnp.asarray(rhs.Cin_controlled_Inflows)), 1.0),
        SCALE_modeled_Inflows_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_modeled_Inflows)), 1.0),
    )
