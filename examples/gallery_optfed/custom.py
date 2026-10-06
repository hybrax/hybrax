"""Reduced OptFed kinetics for direct concentration-trajectory fitting."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from hybrax.format.mechanistic import build_rhs_ode
from hybrax.train import EstimatedScales, RateModule, ReactionOutputs, trainable_field

R = 8.314
K_B = 1.38e-23
H = 6.63e-34
Y_XG = 0.627
Y_PG = 0.652
K_GLYCEROL = 0.001

# Bounds keep every fitted physical parameter in a meaningful range. Parameters
# are optimized in unconstrained raw space and mapped through a sigmoid.
BOUNDS = {
    "uptake_max": ((1e-12, 1.0),),
    "glycerol_inhibition": ((1e-5, 1e4),),
    "maintenance": ((1e-12, 1.0),),
    "uptake_maintenance": ((1e-12, 1e6),),
    "product_maintenance": ((1e-12, 1e6),),
    "production_max": ((1e-12, 1e-5),),
    "production_half_saturation": ((1e-12, 1e2),),
    "generation_inhibition": ((1e-12, 1e4),),
    "temperature": ((4e4, 1e5), (1e5, 1e7), (300.0, 315.0)),
}
NEUTRAL_INITIAL = {
    "uptake_max": (0.25,),
    "glycerol_inhibition": (100.0,),
    "maintenance": (1e-3,),
    "uptake_maintenance": (1.0,),
    "product_maintenance": (1.0,),
    # Start below the substrate cap so production parameters receive gradients.
    "production_max": (1e-9,),
    "production_half_saturation": (0.1,),
    "generation_inhibition": (10.0,),
    "temperature": (5.2e4, 1e6, 307.5),
}


def _to_raw(value, bounds, *, linear=False):
    low, high = bounds
    if linear:
        fraction = (value - low) / (high - low)
    else:
        fraction = (jnp.log(value) - jnp.log(low)) / (jnp.log(high) - jnp.log(low))
    return jnp.log(fraction) - jnp.log1p(-fraction)


def _from_raw(raw, bounds, *, linear=False):
    low, high = bounds
    fraction = jax.nn.sigmoid(raw)
    if linear:
        return low + fraction * (high - low)
    return jnp.exp(jnp.log(low) + fraction * (jnp.log(high) - jnp.log(low)))


def _neutral_raw_parameters():
    return {
        name: jnp.asarray(
            [
                _to_raw(value, bound, linear=name == "temperature" and i == 2)
                for i, (value, bound) in enumerate(
                    zip(NEUTRAL_INITIAL[name], bounds, strict=True)
                )
            ]
        )
        for name, bounds in BOUNDS.items()
    }


def _temperature_ceiling(temperature, constant, parameters):
    activation_energy, enthalpy, equilibrium_temperature = parameters
    return (
        constant
        * K_B
        * temperature
        * jnp.exp(-activation_energy / (R * temperature))
        / (
            H
            * (
                1.0
                + jnp.exp(
                    enthalpy * (1.0 / equilibrium_temperature - 1.0 / temperature) / R
                )
            )
        )
        * 3600.0
    )


def _inhibition(value, constant):
    return 1.0 / (1.0 + value / constant)


class OptFedReactionModule(RateModule):
    """Eleven bounded kinetic parameters and one latent generation state."""

    raw_parameters: dict = trainable_field()
    i_biomass: int = eqx.field(static=True)
    i_glycerol: int = eqx.field(static=True)
    i_product: int = eqx.field(static=True)
    i_temperature: int = eqx.field(static=True)
    i_supply_limited: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        raw_parameters,
        i_biomass,
        i_glycerol,
        i_product,
        i_temperature,
        i_supply_limited,
        **scale_kwargs,
    ):
        super().__init__(SCALE_latent=jnp.ones(1), **scale_kwargs)
        self.raw_parameters = raw_parameters
        self.i_biomass = i_biomass
        self.i_glycerol = i_glycerol
        self.i_product = i_product
        self.i_temperature = i_temperature
        self.i_supply_limited = i_supply_limited

    @property
    def parameters(self):
        return {
            name: jnp.asarray(
                [
                    _from_raw(raw, bound, linear=name == "temperature" and i == 2)
                    for i, (raw, bound) in enumerate(
                        zip(self.raw_parameters[name], bounds, strict=True)
                    )
                ]
            )
            for name, bounds in BOUNDS.items()
        }

    def __call__(self, t, inputs):
        del t
        states = self.unscale_modeled_RMCs(inputs.SCL_modeled_RMCs)
        biomass = jnp.maximum(states[self.i_biomass], 1e-8)
        glycerol = jnp.maximum(states[self.i_glycerol], 0.0)
        product = jnp.maximum(states[self.i_product], 0.0)
        active_biomass = jnp.maximum(biomass - product, 0.0)
        product_biomass = product / biomass
        generations = jnp.maximum(self.unscale_latent(inputs.SCL_latent)[0], 0.0)

        controls = self.unscale_controlled_PVs(inputs.SCL_controlled_PVs)
        temperature = controls[self.i_temperature]
        p = self.parameters

        kinetic_uptake = p["uptake_max"][0]
        kinetic_uptake *= glycerol / (K_GLYCEROL + glycerol)
        kinetic_uptake *= _inhibition(glycerol, p["glycerol_inhibition"][0])

        volume = self.unscale_modeled_V(inputs.SCL_modeled_V)
        feed_rates = self.unscale_controlled_Inflows_rates(
            inputs.SCL_controlled_Inflows_rates
        )
        feed_compositions = self.unscale_controlled_Inflows_Cin(
            inputs.SCL_controlled_Inflows_Cin
        )
        glycerol_feed = jnp.sum(feed_rates * feed_compositions[:, self.i_glycerol])
        supply_uptake = glycerol_feed / (volume * jnp.maximum(active_biomass, 1e-8))
        uptake = jax.lax.cond(
            controls[self.i_supply_limited] > 0.5,
            lambda: supply_uptake,
            lambda: kinetic_uptake,
        )

        maintenance = p["maintenance"][0]
        maintenance *= 1.0 + uptake * p["uptake_maintenance"][0]
        maintenance *= 1.0 + product_biomass * p["product_maintenance"][0]
        maintenance = jnp.minimum(maintenance, uptake)

        available = jnp.maximum(uptake - maintenance, 0.0)
        production = _temperature_ceiling(
            temperature, p["production_max"][0], p["temperature"]
        )
        production *= available / (p["production_half_saturation"][0] + available)
        production *= _inhibition(generations, p["generation_inhibition"][0])
        production = jnp.minimum(production, available)

        q_biomass = jnp.maximum(available - production, 0.0) * Y_XG
        q_product = production * Y_PG
        growth_rate = (q_biomass + q_product) * active_biomass / biomass
        return ReactionOutputs(
            SCL_modeled_ReactionOde_rates=self.scale_modeled_ReactionOde_rates(
                jnp.asarray([q_biomass, uptake, q_product])
            ),
            SCL_modeled_Inflows_rates=jnp.zeros(0),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            SCL_latent_derivative=jnp.asarray([growth_rate / jnp.log(2.0)]),
            auxiliary={
                "uptake": uptake,
                "maintenance": maintenance,
                "production": production,
                "generations": generations,
            },
        )


def build_reaction_module(
    *,
    training_parent_collection,
    config,
    target_names,
    process_names,
    seed,
    **scale_kwargs,
):
    """Build the eleven-parameter model from neutral physical values."""
    del target_names, process_names, seed
    process = next(iter(training_parent_collection.processes.values()))
    rhs = build_rhs_ode(process)
    states = list(rhs.name_modeled_RMCs)
    controls = list(rhs.name_controlled_PVs)
    raw = _neutral_raw_parameters()
    return OptFedReactionModule(
        raw_parameters=raw,
        i_biomass=states.index("biomass"),
        i_glycerol=states.index("glycerol"),
        i_product=states.index("product"),
        i_temperature=controls.index("temperature"),
        i_supply_limited=controls.index("supply_limited_uptake"),
        **scale_kwargs,
    )


def build_learning_rate(config, train_cfg, total_updates):
    """Warm from 10% to the configured peak, then cosine-decay to 5%."""
    return optax.warmup_cosine_decay_schedule(
        init_value=train_cfg.learning_rate * 0.1,
        peak_value=train_cfg.learning_rate,
        warmup_steps=int(total_updates * 0.05),
        decay_steps=total_updates,
        end_value=train_cfg.learning_rate * 0.05,
    )


def estimate_all_scales(runtime_data, target_names, config):
    """Scale states and measured controls to order-one solver values."""
    del target_names, config
    evidence = runtime_data.control_scale_evidence()
    indices = range(len(runtime_data.process_order))

    def axis_scale(traces):
        return jnp.asarray(
            [max(float(np.max(np.abs(values))), 1e-2) for values in traces]
        )

    return EstimatedScales(
        SCALE_modeled_RMCs=jnp.asarray(
            [
                max(
                    np.abs(runtime_data.raw_state_trace(i, name)[1]).max()
                    for i in indices
                )
                for name in runtime_data.rhs_ode.name_modeled_RMCs
            ]
        ),
        SCALE_V_in_cumulative=jnp.asarray(
            max(runtime_data.initial_volume(i) for i in indices)
        ),
        SCALE_modeled_Inflows_cumulative=jnp.zeros(0),
        SCALE_modeled_Outflows_cumulative=jnp.zeros(0),
        SCALE_controlled_Inflows_cumulative=axis_scale(evidence.cumulative_Inflows),
        SCALE_controlled_Inflows_rates=axis_scale(evidence.Inflow_rates),
        SCALE_controlled_Inflows_Cin=jnp.maximum(
            jnp.max(jnp.abs(jnp.asarray(evidence.controlled_Inflow_Cin)), axis=0), 1.0
        ),
        SCALE_controlled_Outflows_cumulative=jnp.zeros(0),
        SCALE_controlled_Outflows_rates=jnp.zeros(0),
        SCALE_controlled_PVs=axis_scale(evidence.PVs),
        SCALE_modeled_Inflows_Cin=jnp.zeros((0, 3)),
        SCALE_modeled_ReactionOde_rates=jnp.ones(3),
        SCALE_modeled_Inflows_rates=jnp.zeros(0),
        SCALE_modeled_Outflows_rates=jnp.zeros(0),
    )
