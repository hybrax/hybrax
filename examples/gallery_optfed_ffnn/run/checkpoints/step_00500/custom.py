"""Replace OptFed's kinetic rate laws with a small neural network."""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hybrax.format.mechanistic import build_rhs_ode
from hybrax.train import EstimatedScales, RateModule, ReactionOutputs, trainable_field


class NeuralReactionModule(RateModule):
    """Learn uptake capacity and sequential glycerol allocation fractions."""

    network: eqx.nn.MLP = trainable_field()  # Let Hybrax optimize the MLP weights.
    glycerol_index: int = eqx.field(static=True)
    biomass_index: int = eqx.field(static=True)
    product_index: int = eqx.field(static=True)

    def __init__(self, *, key, glycerol_index, biomass_index, product_index, **scales):
        # One latent state: OptFed's generation count, already of order one.
        super().__init__(SCALE_latent=jnp.ones(1), **scales)
        self.biomass_index = biomass_index
        self.product_index = product_index
        self.glycerol_index = glycerol_index
        self.network = eqx.nn.MLP(
            self.n_modeled_RMCs + self.n_controlled_PVs + 1,
            3,
            width_size=4,
            depth=2,
            activation=jax.nn.tanh,
            key=key,
        )
        # Start with nearly constant outputs: an uptake capacity of
        # softplus(bias) = 0.1 g glycerol/(g active biomass h); both fractions 0.5.
        self.network = eqx.tree_at(
            lambda net: (net.layers[-1].weight, net.layers[-1].bias),
            self.network,
            (
                self.network.layers[-1].weight * 0.01,
                jnp.asarray([jnp.log(jnp.expm1(0.1)), 0.0, 0.0]),
            ),
        )

    def __call__(self, _t, inputs):
        # Concentrations, temperature, and OptFed's generation count.
        features = jnp.concatenate(
            [
                inputs.SCL_modeled_RMCs,
                inputs.SCL_controlled_PVs,
                inputs.SCL_latent,
            ]
        )
        raw = self.network(features)
        q_max = jax.nn.softplus(raw[0])
        maintenance_fraction, product_fraction = jax.nn.sigmoid(raw[1:])
        states = self.unscale_modeled_RMCs(inputs.SCL_modeled_RMCs)
        glycerol = jnp.maximum(states[self.glycerol_index], 0.0)

        # OptFed's fixed Monod constant, in g/kg. Uptake stops at zero glycerol.
        uptake = q_max * glycerol / (1e-3 + glycerol)
        remaining = uptake * (1.0 - maintenance_fraction)
        production = remaining * product_fraction
        growth = remaining - production

        # OptFed's fixed yields convert glycerol allocations to specific rates.
        rates = jnp.asarray([0.627 * growth, uptake, 0.652 * production])
        # Same generation-count ODE as the mechanistic OptFed module.
        biomass = jnp.maximum(states[self.biomass_index], 1e-8)
        product = jnp.maximum(states[self.product_index], 0.0)
        active_biomass = jnp.maximum(biomass - product, 0.0)
        growth_rate = (rates[0] + rates[2]) * active_biomass / biomass
        return ReactionOutputs(
            SCL_modeled_ReactionOde_rates=self.scale_modeled_ReactionOde_rates(rates),
            # Feeds and sampling stay controlled by the imported dataset.
            SCL_modeled_Inflows_rates=jnp.zeros(0),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            SCL_latent_derivative=jnp.asarray([growth_rate / jnp.log(2.0)]),
        )


def build_reaction_module(*, seed, training_parent_collection, **kwargs):
    """Keep OptFed's scales and replace only the rate calculation."""
    process = next(iter(training_parent_collection.processes.values()))
    rhs = build_rhs_ode(process)
    return NeuralReactionModule(
        key=jax.random.key(seed),
        glycerol_index=list(rhs.name_modeled_RMCs).index("glycerol"),
        biomass_index=list(rhs.name_modeled_RMCs).index("biomass"),
        product_index=list(rhs.name_modeled_RMCs).index("product"),
        **{name: value for name, value in kwargs.items() if name.startswith("SCALE_")},
    )


def build_solver(config):
    """Reuse the OptFed gallery's solver."""
    del config
    return diffrax.Kvaerno5()


def estimate_all_scales(runtime_data, target_names, config):
    """Reuse OptFed's maximum-based scales, which also weight the default loss."""
    del target_names, config
    evidence = runtime_data.control_scale_evidence()
    indices = range(len(runtime_data.process_order))

    def axis_scale(traces):
        return jnp.asarray(
            [max(float(np.max(np.abs(values))), 1e-2) for values in traces]
        )

    # Empty arrays describe quantities absent from this dataset/model.
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
