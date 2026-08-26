"""Glutamine degradation rate law: three constant specific rates, no kinetic
structure at all. The point of this page is entirely in the dataset's own
biological_ode block (one shared rate feeding two derivatives), not in this
reaction module.
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


class GlutamineDecayModule(UserReactionModule):
    log_q_biomass: jax.Array = trainable_field()
    log_q_Gln: jax.Array = trainable_field()
    log_r_Gln: jax.Array = trainable_field()

    def __init__(self, **scale_kwargs):
        super().__init__(**scale_kwargs)
        # Deliberately mediocre starting guesses: the point is that they move.
        self.log_q_biomass = jnp.log(jnp.asarray(0.02))
        self.log_q_Gln = jnp.log(jnp.asarray(1e-4))
        self.log_r_Gln = jnp.log(jnp.asarray(0.01))

    def __call__(self, t, inputs) -> ReactionOutputs:
        del t, inputs  # every rate here is a plain constant, no state read
        RAW_rates = jnp.array(
            [
                jnp.exp(self.log_q_biomass),
                jnp.exp(self.log_q_Gln),
                jnp.exp(self.log_r_Gln),
            ]
        )
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=self.scale_modeled_BiologicalOde_rates(
                RAW_rates
            ),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            SCL_modeled_Inflows_rates=jnp.zeros(0),
        )


def build_reaction_module(**kwargs):
    return GlutamineDecayModule(
        **{k: v for k, v in kwargs.items() if k.startswith("SCALE_")}
    )


def estimate_all_scales(runtime_data, target_names, config):
    del target_names, config
    rhs = runtime_data.rhs_ode
    n_processes = len(runtime_data.process_order)
    span = max(
        end - start
        for start, end in (runtime_data.time_bounds(i) for i in range(n_processes))
    )

    def max_abs_state(name):
        best = 0.0
        for i in range(n_processes):
            _, values = runtime_data.raw_state_trace(i, name)
            if values.size:
                best = max(best, float(np.max(np.abs(values))))
        return max(best, 1e-6)

    rmc_scale = {name: max_abs_state(name) for name in rhs.name_modeled_RMCs}
    biomass = rmc_scale["biomass"]
    empty = jnp.zeros(0)
    return EstimatedScales(
        SCALE_modeled_RMCs=jnp.asarray([rmc_scale[n] for n in rhs.name_modeled_RMCs]),
        SCALE_modeled_BiologicalOde_rates=jnp.asarray(
            [rmc_scale[n[2:]] / (biomass * span) for n in rhs.name_modeled_rates]
        ),
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
