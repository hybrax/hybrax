"""Tutorial 4: a reaction module and scale estimation for demo_batch.

Two hooks, nothing else. Everything not defined here keeps its default.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hybrax.train import (
    EstimatedScales,
    ReactionInputs,
    ReactionOutputs,
    RateModule,
    trainable_field,
)


# --------------------------------------------------------------------------
# 1. The reaction module: modeled state -> specific rates
# --------------------------------------------------------------------------
class BatchReactionModule(RateModule):
    """One MLP mapping the modeled state to the three specific rates."""

    # `trainable_field()` is what makes these weights visible to the optimizer.
    # Untagged array leaves default to FROZEN.
    mlp: eqx.nn.MLP = trainable_field()

    def __init__(self, *, key, **scale_kwargs):
        # The base class stores every SCALE_* axis; always forward them.
        super().__init__(**scale_kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=self.n_modeled_RMCs,  # biomass, glucose, product
            out_size=self.n_modeled_ReactionOde_rates,  # q_biomass, q_glucose,
            # q_product
            width_size=32,
            depth=3,
            # Use a SMOOTH activation. eqx.nn.MLP defaults to relu, which makes
            # the predicted rates piecewise linear: kinks the solver has to
            # chase, and a derivative that jumps. tanh is what the built-in
            # default module uses, for the same reason.
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t  # this model has no explicit time dependence
        # The network reads SCL inputs, so its output is ALREADY in SCL space.
        # Emit it directly: do not re-apply scale_*, or it cancels on the
        # wrapper's unscale round trip. See the note in the tutorial text.
        return ReactionOutputs(
            SCL_modeled_ReactionOde_rates=self.mlp(inputs.SCL_modeled_RMCs),
            # demo_batch has no modeled feeds, but the field is still required.
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            SCL_modeled_Inflows_rates=jnp.zeros(0),
        )


def build_reaction_module(*, seed, **kwargs):
    scale_kwargs = {k: v for k, v in kwargs.items() if k.startswith("SCALE_")}
    return BatchReactionModule(key=jax.random.key(seed), **scale_kwargs)


# --------------------------------------------------------------------------
# 2. Scale estimation: make every axis O(1)
# --------------------------------------------------------------------------
def estimate_all_scales(runtime_data, target_names, config):
    del target_names, config
    rhs = runtime_data.rhs_ode
    n_processes = len(runtime_data.process_order)

    # A state scale is just "how big does this species get, anywhere in the data".
    def max_abs_state(name):
        best = 0.0
        for i in range(n_processes):
            _, values = runtime_data.raw_state_trace(i, name)
            if values.size:
                best = max(best, float(np.max(np.abs(values))))
        return max(best, 1e-6)  # floor: never divide by zero

    rmc_scale = {name: max_abs_state(name) for name in rhs.name_modeled_RMCs}

    # Rate scales are estimated from the data too. The average specific rate of a
    # species is its total change divided by the integrated biomass exposure,
    # which is exactly what "specific" means. Do NOT use max(biomass): biomass
    # grows by two orders of magnitude over a batch, so dividing by its peak
    # underestimates the rate several-fold.
    def rate_scale_for(species):
        per_process = []
        for i in range(n_processes):
            c_times, c_values = runtime_data.raw_state_trace(i, species)
            x_times, x_values = runtime_data.raw_state_trace(i, "biomass")
            exposure = np.trapezoid(x_values, x_times)
            per_process.append(abs(c_values[-1] - c_values[0]) / max(exposure, 1e-9))
        return max(max(per_process), 1e-9)

    rate_scale = [rate_scale_for(name[2:]) for name in rhs.name_modeled_rates]

    empty = jnp.zeros(0)  # demo_batch has no feeds and no process variables
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
