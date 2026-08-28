"""A stateful reaction module: a continuous-time LSTM.

Every reaction module so far has been memoryless: its rates depend only on
the CURRENT state. A stateful module (``n_latent > 0``) adds its own hidden
state, integrated as extra ODE dimensions alongside the physical ones, so the
rates can depend on the process's recent history too.

Use ``DefaultLstmReactionModule`` for this standard LSTM. Implement a custom
stateful reaction module only when a different recurrent architecture or input
layout is required.
"""

import jax
import jax.numpy as jnp
import numpy as np

from hybrax.train import DefaultLstmReactionModule, EstimatedScales


def build_reaction_module(*, seed, **kwargs):
    scale_kwargs = {k: v for k, v in kwargs.items() if k.startswith("SCALE_")}
    return DefaultLstmReactionModule(
        key=jax.random.key(seed), hidden_width=4, **scale_kwargs
    )


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

    # Note: no SCALE_latent here. DefaultLstmReactionModule sizes it from
    # hidden_width; latent state has no physical scale to estimate.
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
