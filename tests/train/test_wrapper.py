from __future__ import annotations

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from bp_format.dataclasses import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    FeedMedium,
    FeedMediumComponent,
    FeedVolumeChange,
    ReactorMedium,
    ReactorMediumComponent,
    SampleVolumeChange,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)
from bp_format.mechanistic import build_rhs_ode

from bp_train.controls import EVENT_RUN_MIN_DT_CONFIG_KEY
from bp_train.controls_store import ControlsStore
from bp_train.model_api import (
    ReactionInputs,
    ReactionOutputs,
    UserReactionModule,
)
from bp_train.wrapper import (
    HybridOdeWrapper,
    validate_rhs_ode_compatibility,
)


def _unit_scale_kwargs(
    *,
    n_species: int,
    n_rates: int,
    n_modeled_VCs: int,
    controls: ControlsStore | None = None,
    n_controlled_FVCs: int | None = None,
    n_controlled_PVs: int | None = None,
    n_controlled_FVCs_bolus: int | None = None,
) -> dict[str, jnp.ndarray]:
    """All-ones SCALE_* kwargs sized to a layout. Pass either ``controls`` or
    explicit per-axis sizes."""
    if controls is not None:
        n_controlled_FVCs = len(controls.name_controlled_FVCs)
        n_controlled_PVs = len(controls.name_controlled_PVs)
        n_controlled_FVCs_bolus = len(controls.name_extras) - 1
    assert n_controlled_FVCs is not None
    assert n_controlled_PVs is not None
    assert n_controlled_FVCs_bolus is not None
    f32 = jnp.float32
    return {
        "SCALE_modeled_RMCs": jnp.ones(n_species, dtype=f32),
        "SCALE_V_in_cumulative": jnp.asarray(1.0, dtype=f32),
        "SCALE_modeled_FVCs_cumulative": jnp.ones(n_modeled_VCs, dtype=f32),
        "SCALE_controlled_FVCs_cumulative": jnp.ones(n_controlled_FVCs, dtype=f32),
        "SCALE_controlled_FVCs_rates": jnp.ones(n_controlled_FVCs, dtype=f32),
        "SCALE_controlled_FVCs_Cin": jnp.ones(
            (n_controlled_FVCs, n_species), dtype=f32
        ),
        "SCALE_controlled_FVCs_bolus_rates": jnp.ones(
            n_controlled_FVCs_bolus, dtype=f32
        ),
        "SCALE_controlled_PVs": jnp.ones(n_controlled_PVs, dtype=f32),
        "SCALE_modeled_FVCs_Cin": jnp.ones((n_modeled_VCs, n_species), dtype=f32),
        "SCALE_modeled_BiologicalOde_rates": jnp.ones(n_rates, dtype=f32),
        "SCALE_modeled_FVCs_rates": jnp.ones(n_modeled_VCs, dtype=f32),
    }


_PLACEHOLDER_SCALES: dict[str, jnp.ndarray] = {
    "SCALE_modeled_RMCs": jnp.zeros(0, dtype=jnp.float32),
    "SCALE_V_in_cumulative": jnp.asarray(1.0, dtype=jnp.float32),
    "SCALE_modeled_FVCs_cumulative": jnp.zeros(0, dtype=jnp.float32),
    "SCALE_controlled_FVCs_cumulative": jnp.zeros(0, dtype=jnp.float32),
    "SCALE_controlled_FVCs_rates": jnp.zeros(0, dtype=jnp.float32),
    "SCALE_controlled_FVCs_Cin": jnp.zeros((0, 0), dtype=jnp.float32),
    "SCALE_controlled_FVCs_bolus_rates": jnp.zeros(0, dtype=jnp.float32),
    "SCALE_controlled_PVs": jnp.zeros(0, dtype=jnp.float32),
    "SCALE_modeled_FVCs_Cin": jnp.zeros((0, 0), dtype=jnp.float32),
    "SCALE_modeled_BiologicalOde_rates": jnp.zeros(0, dtype=jnp.float32),
    "SCALE_modeled_FVCs_rates": jnp.zeros(0, dtype=jnp.float32),
}


class ConstantReactionModule(UserReactionModule):
    """Test reaction module returning fixed rates (in SCL space).

    Tests construct without scale kwargs (placeholder zeros fill in); the
    test ``_build_wrapper`` helper injects correctly-sized all-ones SCALE_*
    via ``eqx.tree_at`` before constructing the wrapper.
    """

    SCL_specific_rates: jnp.ndarray
    SCL_feed_rates: jnp.ndarray
    aux: dict[str, jnp.ndarray] | None

    def __init__(
        self,
        specific_rates: jnp.ndarray,
        modeled_feed_rates: jnp.ndarray,
        auxiliary: dict[str, jnp.ndarray] | None = None,
        **scale_kwargs,
    ):
        scale_kwargs = {**_PLACEHOLDER_SCALES, **scale_kwargs}
        super().__init__(**scale_kwargs)
        self.SCL_specific_rates = specific_rates
        self.SCL_feed_rates = modeled_feed_rates
        self.aux = auxiliary

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=self.SCL_specific_rates,
            SCL_modeled_FVCs_rates=self.SCL_feed_rates,
            auxiliary=self.aux,
        )


class InvalidReactionShapeModule(UserReactionModule):
    """Test reaction module returning malformed output ranks."""

    def __init__(self, **scale_kwargs):
        scale_kwargs = {**_PLACEHOLDER_SCALES, **scale_kwargs}
        super().__init__(**scale_kwargs)

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray([[0.1]], dtype=jnp.float32),
            SCL_modeled_FVCs_rates=jnp.zeros((0,), dtype=jnp.float32),
        )


class VolumeFeatureEchoReactionModule(UserReactionModule):
    """Reaction module that echoes ``SCL_V`` into the rates output."""

    n_species: int = eqx.field(static=True)
    n_modeled: int = eqx.field(static=True)

    def __init__(self, *, n_species: int, n_modeled: int = 0, **scale_kwargs):
        scale_kwargs = {**_PLACEHOLDER_SCALES, **scale_kwargs}
        super().__init__(**scale_kwargs)
        self.n_species = n_species
        self.n_modeled = n_modeled

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        v_feature = jnp.asarray(inputs.SCL_modeled_V, dtype=jnp.float32)
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.full(
                (self.n_species,), v_feature, dtype=jnp.float32
            ),
            SCL_modeled_FVCs_rates=jnp.zeros((self.n_modeled,), dtype=jnp.float32),
        )


def _make_single_species_process(
    *,
    feed_rate: float = 0.2,
    feed_biomass_concentration: float = 0.0,
) -> BioProcess:
    """Process with biomass in reactor_medium and one controlled feed."""
    feed_medium = FeedMedium(
        name="feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": FeedMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=StaticVariable(feed_biomass_concentration),
                is_controlled=False,
            )
        },
    )
    return BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "feed_A": FeedVolumeChange(
                    name="feed_A",
                    unit="L/h",
                    is_controlled=True,
                    is_continuous=True,
                    values=TimeSeries(
                        times=jnp.asarray([0.0, 2.0]),
                        values=jnp.asarray([feed_rate, feed_rate]),
                    ),
                    feed_medium=feed_medium,
                ),
                "sample_1": SampleVolumeChange(
                    name="sample_1",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([1.0]),
                        values=jnp.asarray([-0.1]),
                    ),
                ),
            },
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
                ),
            },
        ),
        process_variables={},
    )


def _make_single_species_collection(**kwargs) -> BioProcessCollection:
    process = _make_single_species_process(**kwargs)
    return BioProcessCollection(processes={"p1": process}, metadata={})


def _make_two_species_two_feed_process() -> BioProcess:
    """Process with biomass+product and two controlled feeds."""
    feed_a = FeedMedium(
        name="feed_a",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": FeedMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=StaticVariable(10.0),
                is_controlled=False,
            ),
            "product": FeedMediumComponent(
                name="product",
                unit="g/L",
                concentration=StaticVariable(0.0),
                is_controlled=False,
            ),
        },
    )
    feed_b = FeedMedium(
        name="feed_b",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": FeedMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=StaticVariable(0.0),
                is_controlled=False,
            ),
            "product": FeedMediumComponent(
                name="product",
                unit="g/L",
                concentration=StaticVariable(5.0),
                is_controlled=False,
            ),
        },
    )

    return BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "feed_A": FeedVolumeChange(
                    name="feed_A",
                    unit="L/h",
                    is_controlled=True,
                    is_continuous=True,
                    values=TimeSeries(
                        times=jnp.asarray([0.0, 2.0]),
                        values=jnp.asarray([0.2, 0.2]),
                    ),
                    feed_medium=feed_a,
                ),
                "feed_B": FeedVolumeChange(
                    name="feed_B",
                    unit="L/h",
                    is_controlled=True,
                    is_continuous=True,
                    values=TimeSeries(
                        times=jnp.asarray([0.0, 2.0]),
                        values=jnp.asarray([0.3, 0.3]),
                    ),
                    feed_medium=feed_b,
                ),
                "sample_1": SampleVolumeChange(
                    name="sample_1",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([1.0]),
                        values=jnp.asarray([-0.1]),
                    ),
                ),
            },
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
                ),
                "product": ReactorMediumComponent(
                    name="product",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray([0.0, 2.0]),
                        values=jnp.asarray([2.0, 2.0]),
                    ),
                ),
            },
        ),
        process_variables={},
    )


def _derive_unit_scale_kwargs(process, controls) -> dict[str, jnp.ndarray]:
    """Build all-ones SCALE_* kwargs sized to the process / controls layout."""
    rhs_ode = build_rhs_ode(process)
    return _unit_scale_kwargs(
        n_species=len(rhs_ode.name_modeled_RMCs),
        n_rates=len(rhs_ode.name_modeled_rates),
        n_modeled_VCs=len(rhs_ode.name_modeled_FVCs),
        controls=controls,
    )


def _inject_scales(
    reaction_module: UserReactionModule, scale_kwargs: dict[str, jnp.ndarray]
) -> UserReactionModule:
    """Replace placeholder SCALE_* fields with correctly-sized values."""
    return eqx.tree_at(
        lambda m: tuple(getattr(m, name) for name in scale_kwargs.keys()),
        reaction_module,
        tuple(scale_kwargs.values()),
    )


def _build_wrapper(process, controls, reaction_module, **kwargs):
    """Test helper: derives unit SCALE_* from layout, injects them into the
    module, and constructs the HybridOdeWrapper."""
    scale_kwargs = _derive_unit_scale_kwargs(process, controls)
    reaction_module = _inject_scales(reaction_module, scale_kwargs)
    return HybridOdeWrapper.from_process(
        reaction_module=reaction_module,
        process=process,
        controls=controls,
        **kwargs,
    )


def test_validate_rhs_ode_compatibility_rejects_different_species():
    process_a = _make_single_species_process()
    process_b = _make_two_species_two_feed_process()
    rhs_a = build_rhs_ode(process_a)
    rhs_b = build_rhs_ode(process_b)

    with pytest.raises(ValueError, match="name_modeled_RMCs differ"):
        validate_rhs_ode_compatibility("a", rhs_a, "b", rhs_b)


def _make_bolus_ramp_process(*, bolus_time: float = 10.0) -> BioProcess:
    bolus_medium = FeedMedium(
        name="bolus",
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
    process = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=100.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "bolus_feed": FeedVolumeChange(
                    name="bolus_feed",
                    unit="L",
                    is_controlled=True,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([bolus_time]),
                        values=jnp.asarray([2.0]),
                    ),
                    feed_medium=bolus_medium,
                ),
            },
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
                        # Data-derived min_dt is 0.01 h, so the 0.05 h
                        # config value proves runtime config wins.
                        times=jnp.asarray([0.0, 0.01, 100.0]),
                        values=jnp.asarray([1.0, 1.0, 1.0]),
                    ),
                ),
            },
        ),
        process_variables={},
    )
    return process
