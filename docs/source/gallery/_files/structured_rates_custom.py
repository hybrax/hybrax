"""Structured rate laws: Monod growth + Luedeking-Piret product formation.

Instead of an MLP mapping state to rates, every rate is an explicit kinetic
expression with named, physically meaningful, trainable constants.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bp_train import (
    EstimatedScales,
    ReactionOutputs,
    UserReactionModule,
    trainable_field,
)


class MonodModule(UserReactionModule):
    """mu = mu_max·S/(Ks+S);  q_S = -(mu/Yxs + ms·sigma);  q_P = alpha·mu + beta·sigma."""

    # Log-parameterised so every constant stays strictly positive under an
    # unconstrained optimizer. This is the cheapest way to impose positivity.
    log_mu_max: jax.Array = trainable_field()
    log_Ks: jax.Array = trainable_field()
    log_Yxs: jax.Array = trainable_field()
    log_ms: jax.Array = trainable_field()
    log_alpha: jax.Array = trainable_field()
    log_beta: jax.Array = trainable_field()

    # Index of each species in the state vector. Static: not arrays, not trained.
    i_biomass: int = eqx.field(static=True)
    i_glucose: int = eqx.field(static=True)

    def __init__(self, *, i_biomass, i_glucose, **scale_kwargs):
        super().__init__(**scale_kwargs)
        self.i_biomass, self.i_glucose = i_biomass, i_glucose
        # Deliberately mediocre starting guesses: the point is that they move.
        self.log_mu_max = jnp.log(jnp.asarray(0.20))
        self.log_Ks = jnp.log(jnp.asarray(0.50))
        self.log_Yxs = jnp.log(jnp.asarray(0.30))
        self.log_ms = jnp.log(jnp.asarray(0.05))
        self.log_alpha = jnp.log(jnp.asarray(0.03))
        self.log_beta = jnp.log(jnp.asarray(0.02))

    def __call__(self, t, inputs) -> ReactionOutputs:
        del t
        # Kinetics are written in PHYSICAL units, so unscale on the way in and
        # scale on the way out. Ks in g/L only means something in RAW space.
        RAW = self.unscale_modeled_RMCs(inputs.SCL_modeled_RMCs)
        S = jnp.clip(RAW[self.i_glucose], 0.0, None)   # guard tiny negative excursions

        sigma = S / (jnp.exp(self.log_Ks) + S)          # Monod saturation term
        mu = jnp.exp(self.log_mu_max) * sigma
        # Uptake is gated by the SAME saturation term, so it tapers at depletion
        # instead of being clipped afterwards.
        q_glucose = -(mu / jnp.exp(self.log_Yxs) + jnp.exp(self.log_ms) * sigma)
        q_product = jnp.exp(self.log_alpha) * mu + jnp.exp(self.log_beta) * sigma

        RAW_rates = jnp.array([mu, q_glucose, q_product])
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=self.scale_modeled_BiologicalOde_rates(
                RAW_rates),
            SCL_modeled_FVCs_rates=jnp.zeros(0),
        )


def build_reaction_module(*, runtime_context, **kwargs):
    # Never hard-code state indices: read them off the assembled ODE.
    rhs = runtime_context.training_data.rhs_ode
    names = list(rhs.name_modeled_RMCs)
    return MonodModule(
        i_biomass=names.index("biomass"),
        i_glucose=names.index("glucose"),
        **{k: v for k, v in kwargs.items() if k.startswith("SCALE_")},
    )


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
    biomass = rmc_scale["biomass"]
    empty = jnp.zeros(0)
    return EstimatedScales(
        SCALE_modeled_RMCs=jnp.asarray([rmc_scale[n] for n in rhs.name_modeled_RMCs]),
        SCALE_modeled_BiologicalOde_rates=jnp.asarray(
            [rmc_scale[n[2:]] / (biomass * span) for n in rhs.name_modeled_rates]),
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
