"""Mechanistic and neural growth laws for the continuous-overflow gallery."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hybrax.format.mechanistic import build_rhs_ode
from hybrax.train import (
    EstimatedScales,
    ReactionInputs,
    ReactionOutputs,
    RateModule,
    trainable_field,
)


class FittedMonodModule(RateModule):
    """Monod growth with positive, trainable ``mu_max`` and ``Ks``."""

    log_mu_max: jax.Array = trainable_field()
    log_ks: jax.Array = trainable_field()
    i_glucose: int = eqx.field(static=True)

    def __init__(self, *, i_glucose, **scale_kwargs):
        super().__init__(**scale_kwargs)
        self.i_glucose = i_glucose
        self.log_mu_max = jnp.log(jnp.asarray(1.0))
        self.log_ks = jnp.log(jnp.asarray(1.0))

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        states = self.unscale_modeled_RMCs(inputs.SCL_modeled_RMCs)
        glucose = jnp.clip(states[self.i_glucose], 0.0, None)
        mu = jnp.exp(self.log_mu_max) * glucose / (jnp.exp(self.log_ks) + glucose)
        return ReactionOutputs(
            SCL_modeled_ReactionOde_rates=(
                self.scale_modeled_ReactionOde_rates(jnp.asarray([mu]))
            ),
            SCL_modeled_Inflows_rates=jnp.zeros(0),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
        )


class AnnGrowthModule(RateModule):
    """A 33-parameter ``1 → 4 → 4 → 1`` glucose-to-growth network."""

    mlp: eqx.nn.MLP = trainable_field()
    i_glucose: int = eqx.field(static=True)

    def __init__(self, *, i_glucose, key, **scale_kwargs):
        super().__init__(**scale_kwargs)
        self.i_glucose = i_glucose
        self.mlp = eqx.nn.MLP(
            in_size=1,
            out_size=1,
            width_size=4,
            depth=2,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        glucose = inputs.SCL_modeled_RMCs[self.i_glucose]
        mu = self.mlp(jnp.asarray([glucose]))
        return ReactionOutputs(
            SCL_modeled_ReactionOde_rates=mu,
            SCL_modeled_Inflows_rates=jnp.zeros(0),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
        )


def build_reaction_module(*, config, seed, training_parent_collection, **kwargs):
    process = next(iter(training_parent_collection.processes.values()))
    names = list(build_rhs_ode(process).name_modeled_RMCs)
    common = {
        "i_glucose": names.index("glucose"),
        **{key: value for key, value in kwargs.items() if key.startswith("SCALE_")},
    }
    # Both candidates occupy the same reaction-module slot and emit the same
    # declared biological rate. Only their parameterization differs.
    if config.custom.model == "monod":
        return FittedMonodModule(**common)
    if config.custom.model == "ann":
        return AnnGrowthModule(key=jax.random.key(seed), **common)
    raise ValueError(f"unknown model: {config.custom.model!r}")


def estimate_all_scales(runtime_data, target_names, config):
    """Scale states and controls from their observed ranges."""
    del target_names, config
    rhs = runtime_data.rhs_ode
    controls_store = runtime_data.controls_store

    state_scales = []
    for name in rhs.name_modeled_RMCs:
        values = [
            runtime_data.raw_state_trace(i, name)[1]
            for i in range(len(runtime_data.process_order))
        ]
        observed = np.concatenate(values)
        state_scales.append(max(float(np.median(np.abs(observed))), 1.0))

    n_inflows = len(controls_store.name_controlled_Inflows)
    n_outflows = len(controls_store.name_controlled_Outflows)
    inflow_cumulative = []
    inflow_rates = []
    outflow_cumulative = []
    outflow_rates = []
    for i, process_name in enumerate(runtime_data.process_order):
        controls = controls_store.get_controls(process_name)
        start, end = runtime_data.time_bounds(i)
        for time in np.linspace(start, end, 100):
            inflow_cumulative.append(
                np.asarray(controls.eval_controlled_Inflows_cumulative(time, None))
            )
            inflow_rates.append(
                np.asarray(controls.eval_controlled_Inflows_rates(time, None))
            )
            outflow_cumulative.append(
                np.asarray(controls.eval_controlled_Outflows_cumulative(time, None))
            )
            outflow_rates.append(
                np.asarray(controls.eval_controlled_Outflows_rates(time, None))
            )

    def axis_scale(samples, size):
        if not size:
            return jnp.zeros(0)
        return jnp.asarray(np.maximum(np.max(np.abs(np.stack(samples)), axis=0), 1e-6))

    empty = jnp.zeros(0)
    return EstimatedScales(
        SCALE_modeled_RMCs=jnp.asarray(state_scales),
        SCALE_modeled_ReactionOde_rates=jnp.ones(1),
        SCALE_V_in_cumulative=jnp.asarray(
            max(
                runtime_data.initial_volume(i)
                for i in range(len(runtime_data.process_order))
            )
        ),
        SCALE_modeled_Inflows_cumulative=empty,
        SCALE_modeled_Inflows_rates=empty,
        SCALE_modeled_Outflows_cumulative=empty,
        SCALE_modeled_Outflows_rates=empty,
        SCALE_controlled_Inflows_cumulative=axis_scale(inflow_cumulative, n_inflows),
        SCALE_controlled_Inflows_rates=axis_scale(inflow_rates, n_inflows),
        SCALE_controlled_Outflows_cumulative=axis_scale(outflow_cumulative, n_outflows),
        SCALE_controlled_Outflows_rates=axis_scale(outflow_rates, n_outflows),
        SCALE_controlled_PVs=empty,
        SCALE_controlled_Inflows_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_controlled_Inflows)), 1.0
        ),
        SCALE_modeled_Inflows_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_modeled_Inflows)), 1.0
        ),
    )
