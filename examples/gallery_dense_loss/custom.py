"""A loss module that constrains the trajectory BETWEEN measurements.

Builds on the reaction module and scale estimation from Tutorial 4: those two
hooks are unchanged. This file adds three more things, all evaluated on a
dense grid rather than only at the sparse measurement times:

1. A hinge penalty on every declared STATE bound (from
   ``ReactorMediumComponent.bounds``, already present in the data).
2. A hinge penalty on every declared RATE bound (from
   ``ReactionOde.rates``, attached here via ``transform_process_collection``
   since the auto-generated ODE leaves rates unbounded).
3. A smoothness penalty: the sum of squared second time-derivatives
   ("curvature") of each rate trajectory, masked away from genuine
   discontinuities (bolus/sample jumps) with hybrax.train's own helper.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hybrax.format.mechanistic import build_rhs_ode
from hybrax.train import (
    EstimatedScales,
    LossInputs,
    LossOutputs,
    ReactionInputs,
    ReactionOutputs,
    RateModule,
    dense_triple_mask_away_from_jumps,
    trainable_field,
)
from hybrax.train.dense import all_triple
from hybrax.train.defaults import DefaultLossModule


# --------------------------------------------------------------------------
# 1 & 2. Reaction module and scale estimation: identical to Tutorial 4.
# --------------------------------------------------------------------------
class BatchReactionModule(RateModule):
    mlp: eqx.nn.MLP = trainable_field()

    def __init__(self, *, key, **scale_kwargs):
        super().__init__(**scale_kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=self.n_modeled_RMCs,
            out_size=self.n_modeled_ReactionOde_rates,
            width_size=32,
            depth=3,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        return ReactionOutputs(
            SCL_modeled_ReactionOde_rates=self.mlp(inputs.SCL_modeled_RMCs),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            SCL_modeled_Inflows_rates=jnp.zeros(0),
        )


def build_reaction_module(*, seed, **kwargs):
    scale_kwargs = {k: v for k, v in kwargs.items() if k.startswith("SCALE_")}
    return BatchReactionModule(key=jax.random.key(seed), **scale_kwargs)


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


# --------------------------------------------------------------------------
# 3. Attach rate bounds. hybrax.format's auto-generated ReactionOde leaves every
#    rate unbounded (Bounds = (None, None)); here we declare what we actually
#    know about the biology, so the loss below has something to read.
# --------------------------------------------------------------------------
def transform_process_collection(collection, config):
    del config
    for process in collection.processes.values():
        # Uptake cannot be positive; formation cannot be wildly negative.
        process.reaction_ode.rates["q_biomass"] = (-0.05, 1.0)
        process.reaction_ode.rates["q_glucose"] = (-3.0, 0.0)
        process.reaction_ode.rates["q_product"] = (0.0, 0.3)
    return collection


# --------------------------------------------------------------------------
# 4. The loss module.
# --------------------------------------------------------------------------
class PhysicalConstraintsLoss(DefaultLossModule):
    """Per-target MSE (inherited) + a bounds hinge + a rate-smoothness term,
    all evaluated on a 200-point dense grid rather than only at measurements.
    """

    state_lo: jax.Array = eqx.field(static=True)
    state_hi: jax.Array = eqx.field(static=True)
    rate_lo: jax.Array = eqx.field(static=True)
    rate_hi: jax.Array = eqx.field(static=True)
    bounds_weight: float = eqx.field(static=True, default=20.0)
    smoothness_weight: float = eqx.field(static=True, default=1e-2)

    def __init__(self, *, target_names, state_lo, state_hi, rate_lo, rate_hi):
        super().__init__(target_names=target_names)
        self.state_lo, self.state_hi = state_lo, state_hi
        self.rate_lo, self.rate_hi = rate_lo, rate_hi

    @property
    def loss_names(self):
        return (*self.target_names, "bounds", "smoothness")

    @property
    def dense_grid_n(self):
        return 200  # opt in: populates every dense_* field on LossInputs

    def __call__(self, inputs: LossInputs) -> LossOutputs:
        residual = inputs.SCL_target_pred - jnp.where(
            inputs.mask_measured, inputs.SCL_target_measured, 0.0
        )
        per_target = self.residual_reduction(residual, inputs.mask_measured)

        valid = inputs.dense_valid_time  # post-solver-failure mask

        def hinge(value, lo, hi):
            # -inf / +inf bounds fall out of the clip naturally: no branching.
            return (
                jnp.clip(lo - value, min=0.0) ** 2 + jnp.clip(value - hi, min=0.0) ** 2
            )

        # State bounds: dense_RAW_states is [modeled RMCs | modeled PVs | V];
        # slice to just the RMC axes our bounds were built for.
        n_state_bounds = self.state_lo.shape[0]
        state_penalty = hinge(
            inputs.dense_RAW_states[:, :n_state_bounds], self.state_lo, self.state_hi
        )
        rate_penalty = hinge(
            inputs.dense_RAW_modeled_ReactionOde_rates, self.rate_lo, self.rate_hi
        )
        bounds = (
            jnp.sum(state_penalty * valid[:, None])
            + jnp.sum(rate_penalty * valid[:, None])
        ) / jnp.maximum(jnp.sum(valid), 1.0)

        # Smoothness: central second difference of each rate trajectory.
        # dense_t is a uniform linspace, so a fixed dt is valid.
        dt = inputs.dense_t[1] - inputs.dense_t[0]
        rates = inputs.dense_RAW_modeled_ReactionOde_rates
        curvature = (rates[2:] - 2.0 * rates[1:-1] + rates[:-2]) / (dt**2)
        # A bolus/sample creates a REAL kink; do not penalise curvature there.
        # hybrax.train ships this exact helper for that purpose.
        triple_mask = all_triple(valid) & dense_triple_mask_away_from_jumps(
            inputs.dense_t, inputs.jump_ts, jump_epsilon_h=2.0 * dt
        )
        smoothness = jnp.sum(
            jnp.square(curvature) * triple_mask[:, None]
        ) / jnp.maximum(jnp.sum(triple_mask), 1.0)

        return LossOutputs(
            named_losses={
                **{name: per_target[i] for i, name in enumerate(self.target_names)},
                "bounds": self.bounds_weight * bounds,
                "smoothness": self.smoothness_weight * smoothness,
            }
        )


def build_loss_module(*, target_names, training_parent_collection, **kwargs):
    # Bounds are read from the FIRST process. hybrax.format's cross-process
    # consistency check guarantees every process shares the same structure.
    process = next(iter(training_parent_collection.processes.values()))
    rhs = build_rhs_ode(process)

    def as_pair(bounds):
        lo, hi = bounds
        return (-jnp.inf if lo is None else lo), (jnp.inf if hi is None else hi)

    state_bounds = [
        as_pair(process.reactor_medium.components[n].bounds)
        for n in rhs.name_modeled_RMCs
    ]
    rate_bounds = [
        as_pair(process.reaction_ode.rates[n]) for n in rhs.name_modeled_rates
    ]

    return PhysicalConstraintsLoss(
        target_names=list(target_names),
        state_lo=jnp.array([b[0] for b in state_bounds]),
        state_hi=jnp.array([b[1] for b in state_bounds]),
        rate_lo=jnp.array([b[0] for b in rate_bounds]),
        rate_hi=jnp.array([b[1] for b in rate_bounds]),
    )
