"""Fed-batch: a reaction module that reads the feed and a controlled PV.

Unlike the batch tutorials, this process has a continuous feed, two boluses, a
controlled process variable (dissolved oxygen), and sampling events. None of
that changes how you write the reaction module in principle, but the module
now has real inputs beyond the state, and the scale hook has real controlled
axes to estimate.
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


class FedBatchModule(RateModule):
    """MLP over [modeled state | controlled feed rate | controlled PVs]."""

    mlp: eqx.nn.MLP = trainable_field()

    def __init__(self, *, key, **scale_kwargs):
        super().__init__(**scale_kwargs)
        n_in = self.n_modeled_RMCs + self.n_controlled_Inflows + self.n_controlled_PVs
        self.mlp = eqx.nn.MLP(
            in_size=n_in,
            out_size=self.n_modeled_ReactionOde_rates,
            width_size=32,
            depth=3,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        # The feed rate and DO are real biological inputs here, not just
        # transport bookkeeping: the model is allowed to respond to them.
        features = jnp.concatenate(
            [
                inputs.SCL_modeled_RMCs,
                inputs.SCL_controlled_Inflows_rates,
                inputs.SCL_controlled_PVs,
            ]
        )
        return ReactionOutputs(
            SCL_modeled_ReactionOde_rates=self.mlp(features),
            SCL_modeled_Inflows_rates=jnp.zeros(0),  # no MODELED feeds here
            SCL_modeled_Outflows_rates=jnp.zeros(0),  # no MODELED outflows here
        )


def build_reaction_module(*, seed, **kwargs):
    scale_kwargs = {k: v for k, v in kwargs.items() if k.startswith("SCALE_")}
    return FedBatchModule(key=jax.random.key(seed), **scale_kwargs)


def estimate_all_scales(runtime_data, target_names, config):
    """`runtime_data.controls_store` is always available: unlike the batch
    tutorials, this process has a real controlled feed and a real controlled
    PV to scale."""
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

    # Controlled axes: sample the fitted control splines over each run and
    # take the per-axis max-abs. This is the same recipe as the state scales,
    # just evaluated through the controls store instead of read off the data.
    n_inflows = len(controls_store.name_controlled_Inflows)
    n_outflows = len(controls_store.name_controlled_Outflows)
    n_PV = len(controls_store.name_controlled_PVs)
    inflow_rate_samples, outflow_rate_samples, pv_samples = [], [], []
    for i, process_name in enumerate(runtime_data.process_order):
        per_process = controls_store.get_controls(process_name)
        t_start, t_end = runtime_data.time_bounds(i)
        for t in np.linspace(t_start + 1e-3, t_end - 1e-3, 50):
            inflow_rate_samples.append(
                np.asarray(per_process.eval_controlled_Inflows_rates(float(t), None))
            )
            outflow_rate_samples.append(
                np.asarray(per_process.eval_controlled_Outflows_rates(float(t), None))
            )
            pv_samples.append(
                np.asarray(per_process.eval_controlled_PVs(float(t), None))
            )

    def axis_scale(samples, n_axis):
        arr = np.stack(samples, axis=0) if samples else np.ones((1, n_axis))
        return np.maximum(np.max(np.abs(arr), axis=0), 1e-2)

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
        SCALE_controlled_Inflows_cumulative=jnp.ones(n_inflows),
        SCALE_controlled_Inflows_rates=jnp.asarray(
            axis_scale(inflow_rate_samples, n_inflows)
        ),
        SCALE_controlled_Outflows_cumulative=jnp.ones(n_outflows),
        SCALE_controlled_Outflows_rates=jnp.asarray(
            axis_scale(outflow_rate_samples, n_outflows)
        ),
        SCALE_controlled_PVs=jnp.asarray(axis_scale(pv_samples, n_PV)),
        SCALE_controlled_Inflows_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_controlled_Inflows)), 1.0
        ),
        SCALE_modeled_Inflows_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_modeled_Inflows)), 1.0
        ),
    )
