from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from bp_format.dataclasses import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    FeedMedium,
    FeedMediumComponent,
    Inflow,
    Outflow,
    ReactorMedium,
    ReactorMediumComponent,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)
from bp_format.mechanistic import build_rhs_ode

from bp_train.controls_store import ControlsStore
from bp_train.defaults import DefaultStatefulReactionModule
from bp_train.model_api import (
    LinearScaler,
    ReactionInputs,
    ReactionOutputs,
    UserReactionModule,
    trainable_field,
)
from bp_train.physical_solve import solve_physical_states
from bp_train.wrapper import HybridOdeWrapper


def default_stateful_scale_kwargs(
    *,
    n_rmcs: int = 1,
    n_modeled_inflows: int = 0,
    n_modeled_outflows: int = 0,
    n_controlled_inflows: int = 1,
    n_controlled_outflows: int = 0,
    n_controlled_pvs: int = 0,
):
    return {
        "SCALE_modeled_RMCs": jnp.ones(n_rmcs),
        "SCALE_V_in_cumulative": jnp.asarray(1.0),
        "SCALE_modeled_Inflows_cumulative": jnp.ones(n_modeled_inflows),
        "SCALE_modeled_Outflows_cumulative": jnp.ones(n_modeled_outflows),
        "SCALE_controlled_Inflows_cumulative": jnp.ones(n_controlled_inflows),
        "SCALE_controlled_Inflows_rates": jnp.ones(n_controlled_inflows),
        "SCALE_controlled_Inflows_Cin": jnp.ones((n_controlled_inflows, n_rmcs)),
        "SCALE_controlled_Outflows_cumulative": jnp.ones(n_controlled_outflows),
        "SCALE_controlled_Outflows_rates": jnp.ones(n_controlled_outflows),
        "SCALE_controlled_PVs": jnp.ones(n_controlled_pvs),
        "SCALE_modeled_Inflows_Cin": jnp.ones((n_modeled_inflows, n_rmcs)),
        "SCALE_modeled_BiologicalOde_rates": jnp.ones(1),
        "SCALE_modeled_Inflows_rates": jnp.ones(n_modeled_inflows),
        "SCALE_modeled_Outflows_rates": jnp.ones(n_modeled_outflows),
    }


class ZeroLatentDerivativeModule(UserReactionModule):
    h0: jax.Array

    def __init__(self, h0: jax.Array):
        super().__init__()
        self.h0 = h0

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.zeros(1, dtype=inputs.SCL_latent.dtype),
            SCL_modeled_Inflows_rates=jnp.zeros(0, dtype=inputs.SCL_latent.dtype),
            SCL_modeled_Outflows_rates=jnp.zeros(0, dtype=inputs.SCL_latent.dtype),
            SCL_latent_derivative=jnp.zeros_like(inputs.SCL_latent),
        )

    def initial_latent(self, RAW_phys_y0):
        del RAW_phys_y0
        return self.h0


class TrainableH0Module(UserReactionModule):
    h0: jax.Array = trainable_field()

    def __init__(self, h0: jax.Array):
        super().__init__()
        self.h0 = h0

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.zeros(1, dtype=inputs.SCL_latent.dtype),
            SCL_modeled_Inflows_rates=jnp.zeros(0, dtype=inputs.SCL_latent.dtype),
            SCL_modeled_Outflows_rates=jnp.zeros(0, dtype=inputs.SCL_latent.dtype),
            SCL_latent_derivative=inputs.SCL_latent,
        )

    def initial_latent(self, RAW_phys_y0):
        del RAW_phys_y0
        return self.h0


class TrainableH0DefaultStateful(DefaultStatefulReactionModule):
    """``DefaultStatefulReactionModule`` with a trainable initial latent state."""

    h0: jax.Array = trainable_field()

    def __init__(self, *, key, h0: jax.Array, **scale_kwargs):
        super().__init__(key=key, n_latent=h0.shape[0], **scale_kwargs)
        self.h0 = h0

    def initial_latent(self, RAW_phys_y0):
        del RAW_phys_y0
        return self.h0


def make_process(*, feed_rate: float = 0.0, jump: bool = False) -> BioProcess:
    volume_changes = {}
    if feed_rate:
        feed_medium = FeedMedium(
            name="feed",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": FeedMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(0.0),
                    is_controlled=False,
                )
            },
        )
        volume_changes["feed_A"] = Inflow(
            name="feed_A",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            values=TimeSeries(
                times=jnp.asarray([0.0, 2.0]),
                values=jnp.asarray([0.0, 2.0 * feed_rate]),
            ),
            feed_medium=feed_medium,
        )
    if jump:
        bolus_medium = FeedMedium(
            name="bolus",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": FeedMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(2.0),
                    is_controlled=False,
                )
            },
        )
        volume_changes["sample_1"] = Outflow(
            name="sample_1",
            unit="L",
            is_controlled=False,
            is_continuous=False,
            values=TimeSeries(times=jnp.asarray([1.0]), values=jnp.asarray([-0.2])),
        )
        volume_changes["bolus_A"] = Inflow(
            name="bolus_A",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=TimeSeries(times=jnp.asarray([1.0]), values=jnp.asarray([0.3])),
            feed_medium=bolus_medium,
        )

    return BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes=volume_changes,
        ),
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray([0.0, 2.0]),
                        values=jnp.asarray([1.0, 1.0]),
                    ),
                )
            },
        ),
        process_variables={},
    )


def build_stateful_wrapper(process: BioProcess, module: UserReactionModule):
    collection = BioProcessCollection(processes={"p1": process}, metadata={})
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    rhs = build_rhs_ode(process)
    n_controlled_fvcs = len(controls.name_controlled_FVCs)
    n_latent = module.h0.shape[0] if hasattr(module, "h0") else module.n_latent
    # A module that already sized a rate head (e.g. DefaultStatefulReactionModule)
    # must have been built with the rhs rate count; otherwise the overridden scale
    # below would silently disagree with the head width.
    n_rates = len(rhs.name_modeled_rates)
    assert module.n_modeled_BiologicalOde_rates in (0, n_rates), (
        "build_stateful_wrapper cannot re-shape a module whose rate head was "
        f"sized to {module.n_modeled_BiologicalOde_rates}, not the rhs count {n_rates}"
    )
    scale_kwargs = {
        **default_stateful_scale_kwargs(n_controlled_inflows=n_controlled_fvcs),
        "SCALE_modeled_BiologicalOde_rates": jnp.ones(len(rhs.name_modeled_rates)),
        "SCALE_latent": jnp.ones(n_latent),
    }
    module = eqx.tree_at(
        lambda m: tuple(getattr(m, name) for name in scale_kwargs),
        module,
        tuple(LinearScaler(v) for v in scale_kwargs.values()),
    )
    return HybridOdeWrapper.from_process(
        reaction_module=module,
        process=process,
        controls=controls,
    )


def solve(wrapper, t_eval, y0, *, rtol=1e-6, atol=1e-8, max_steps=10_000):
    """Run the callbacks solve over ``t_eval`` with the shared test defaults."""
    return solve_physical_states(
        wrapper,
        t_eval=t_eval,
        n_measured=t_eval.shape[0],
        RAW_y0=y0,
        max_steps=max_steps,
        rtol=rtol,
        atol=atol,
    )
