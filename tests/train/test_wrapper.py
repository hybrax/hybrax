from __future__ import annotations

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from hybrax.format.dataclasses import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    FeedMedium,
    FeedMediumComponent,
    Inflow,
    ProcessVariable,
    ReactorMedium,
    ReactorMediumComponent,
    Outflow,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)
from hybrax.format.mechanistic import build_rhs_ode

from hybrax.train.controls_store import ControlsStore
from hybrax.train.model_api import (
    AffineScaler,
    LinearScaler,
    ReactionInputs,
    Scaler,
    ReactionOutputs,
    UserReactionModule,
)
from hybrax.train.wrapper import (
    HybridOdeWrapper,
    validate_rhs_ode_compatibility,
)


def _unit_scale_kwargs(
    *,
    n_species: int,
    n_rates: int,
    n_modeled_Inflows: int,
    n_modeled_Outflows: int = 0,
    n_modeled_PVs: int = 0,
    controls: ControlsStore | None = None,
    n_controlled_Inflows: int | None = None,
    n_controlled_Outflows: int = 0,
    n_controlled_PVs: int | None = None,
) -> dict[str, jnp.ndarray]:
    """All-ones SCALE_* kwargs sized to a layout. Pass either ``controls`` or
    explicit per-axis sizes."""
    if controls is not None:
        n_controlled_Inflows = len(controls.name_controlled_Inflows)
        n_controlled_Outflows = len(controls.name_controlled_Outflows)
        n_controlled_PVs = len(controls.name_controlled_PVs)
    assert n_controlled_Inflows is not None
    assert n_controlled_Outflows is not None
    assert n_controlled_PVs is not None
    return {
        "SCALE_modeled_RMCs": jnp.ones(n_species),
        "SCALE_modeled_PVs": jnp.ones(n_modeled_PVs),
        "SCALE_V_in_cumulative": jnp.asarray(1.0),
        "SCALE_modeled_Inflows_cumulative": jnp.ones(n_modeled_Inflows),
        "SCALE_modeled_Outflows_cumulative": jnp.ones(n_modeled_Outflows),
        "SCALE_controlled_Inflows_cumulative": jnp.ones(n_controlled_Inflows),
        "SCALE_controlled_Inflows_rates": jnp.ones(n_controlled_Inflows),
        "SCALE_controlled_Outflows_cumulative": jnp.ones(n_controlled_Outflows),
        "SCALE_controlled_Outflows_rates": jnp.ones(n_controlled_Outflows),
        "SCALE_controlled_Inflows_Cin": jnp.ones((n_controlled_Inflows, n_species)),
        "SCALE_controlled_PVs": jnp.ones(n_controlled_PVs),
        "SCALE_modeled_Inflows_Cin": jnp.ones((n_modeled_Inflows, n_species)),
        "SCALE_modeled_BiologicalOde_rates": jnp.ones(n_rates),
        "SCALE_modeled_Inflows_rates": jnp.ones(n_modeled_Inflows),
        "SCALE_modeled_Outflows_rates": jnp.ones(n_modeled_Outflows),
    }


_PLACEHOLDER_SCALES: dict[str, jnp.ndarray] = {
    "SCALE_modeled_RMCs": jnp.zeros(0),
    "SCALE_V_in_cumulative": jnp.asarray(1.0),
    "SCALE_modeled_Inflows_cumulative": jnp.zeros(0),
    "SCALE_modeled_Outflows_cumulative": jnp.zeros(0),
    "SCALE_controlled_Inflows_cumulative": jnp.zeros(0),
    "SCALE_controlled_Inflows_rates": jnp.zeros(0),
    "SCALE_controlled_Outflows_cumulative": jnp.zeros(0),
    "SCALE_controlled_Outflows_rates": jnp.zeros(0),
    "SCALE_controlled_Inflows_Cin": jnp.zeros((0, 0)),
    "SCALE_controlled_PVs": jnp.zeros(0),
    "SCALE_modeled_Inflows_Cin": jnp.zeros((0, 0)),
    "SCALE_modeled_BiologicalOde_rates": jnp.zeros(0),
    "SCALE_modeled_Inflows_rates": jnp.zeros(0),
    "SCALE_modeled_Outflows_rates": jnp.zeros(0),
    "SCALE_latent": jnp.zeros(0),
}


class ConstantReactionModule(UserReactionModule):
    """Test reaction module returning fixed rates (in SCL space).

    Tests construct without scale kwargs (placeholder zeros fill in); the
    test ``_build_wrapper`` helper injects correctly-sized all-ones SCALE_*
    via ``eqx.tree_at`` before constructing the wrapper.
    """

    SCL_specific_rates: jnp.ndarray
    SCL_Inflow_rates: jnp.ndarray
    SCL_outflow_rates: jnp.ndarray
    aux: dict[str, jnp.ndarray] | None

    def __init__(
        self,
        specific_rates: jnp.ndarray,
        modeled_Inflows_rates: jnp.ndarray,
        modeled_outflow_rates: jnp.ndarray | None = None,
        auxiliary: dict[str, jnp.ndarray] | None = None,
        **scale_kwargs,
    ):
        scale_kwargs = {**_PLACEHOLDER_SCALES, **scale_kwargs}
        super().__init__(**scale_kwargs)
        self.SCL_specific_rates = specific_rates
        self.SCL_Inflow_rates = modeled_Inflows_rates
        self.SCL_outflow_rates = (
            jnp.zeros(0) if modeled_outflow_rates is None else modeled_outflow_rates
        )
        self.aux = auxiliary

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=self.SCL_specific_rates,
            SCL_modeled_Inflows_rates=self.SCL_Inflow_rates,
            SCL_modeled_Outflows_rates=self.SCL_outflow_rates,
            auxiliary=self.aux,
        )


class InvalidReactionShapeModule(UserReactionModule):
    """Return configurable outputs so each malformed field is tested in isolation."""

    biological_rates: jnp.ndarray
    feed_rates: jnp.ndarray
    latent_derivative: jnp.ndarray

    def __init__(
        self,
        biological_rates,
        feed_rates,
        latent_derivative,
        **scale_kwargs,
    ):
        scale_kwargs = {**_PLACEHOLDER_SCALES, **scale_kwargs}
        super().__init__(**scale_kwargs)
        self.biological_rates = jnp.asarray(biological_rates)
        self.feed_rates = jnp.asarray(feed_rates)
        self.latent_derivative = jnp.asarray(latent_derivative)

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=self.biological_rates,
            SCL_modeled_Inflows_rates=self.feed_rates,
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            SCL_latent_derivative=self.latent_derivative,
        )


class LatentEchoReactionModule(UserReactionModule):
    initial_h0: jnp.ndarray

    def __init__(self, *, initial_h0, **scale_kwargs):
        scale_kwargs = {**_PLACEHOLDER_SCALES, **scale_kwargs}
        super().__init__(**scale_kwargs)
        self.initial_h0 = jnp.asarray(initial_h0)

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.zeros(1),
            SCL_modeled_Inflows_rates=jnp.zeros(0),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            SCL_latent_derivative=inputs.SCL_latent
            + jnp.asarray([1.0, 2.0], dtype=inputs.SCL_latent.dtype),
            auxiliary={"SCL_latent": inputs.SCL_latent},
        )

    def initial_latent(self, RAW_phys_y0):
        del RAW_phys_y0
        return self.initial_h0


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
        v_feature = jnp.asarray(inputs.SCL_modeled_V)
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.full((self.n_species,), v_feature),
            SCL_modeled_Inflows_rates=jnp.zeros((self.n_modeled,)),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
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
                "feed_A": Inflow(
                    name="feed_A",
                    unit="L",
                    is_controlled=True,
                    is_continuous=True,
                    values=TimeSeries(
                        times=jnp.asarray([0.0, 2.0]),
                        values=jnp.asarray([0.0, 2.0 * feed_rate]),
                    ),
                    feed_medium=feed_medium,
                ),
                "sample_1": Outflow(
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


def _make_mixed_flow_process() -> BioProcess:
    process = _make_single_species_process(feed_rate=0.2)
    controlled_feed = process.volume.volume_changes["feed_A"]
    process.volume.volume_changes["modeled_feed"] = replace(
        controlled_feed,
        name="modeled_feed",
        is_controlled=False,
        values=TimeSeries(times=[0.0, 2.0], values=[0.0, 0.8]),
    )
    process.volume.volume_changes["harvest"] = Outflow(
        name="harvest",
        unit="L",
        is_controlled=True,
        is_continuous=True,
        values=TimeSeries(times=[0.0, 2.0], values=[0.0, -0.2]),
        retention={"biomass": 0.0},
    )
    process.volume.volume_changes["perfusion"] = Outflow(
        name="perfusion",
        unit="L",
        is_controlled=False,
        is_continuous=True,
        values=TimeSeries(times=[0.0, 2.0], values=[0.0, -0.4]),
        retention={"biomass": 0.5},
    )
    process.volume.volume_changes["evaporation"] = Outflow(
        name="evaporation",
        unit="L",
        is_controlled=True,
        is_continuous=True,
        values=TimeSeries(times=[0.0, 2.0], values=[0.0, -0.6]),
        retention={"biomass": 1.0},
    )
    return process


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
                "feed_A": Inflow(
                    name="feed_A",
                    unit="L",
                    is_controlled=True,
                    is_continuous=True,
                    values=TimeSeries(
                        times=jnp.asarray([0.0, 2.0]),
                        values=jnp.asarray([0.0, 0.4]),
                    ),
                    feed_medium=feed_a,
                ),
                "feed_B": Inflow(
                    name="feed_B",
                    unit="L",
                    is_controlled=True,
                    is_continuous=True,
                    values=TimeSeries(
                        times=jnp.asarray([0.0, 2.0]),
                        values=jnp.asarray([0.0, 0.6]),
                    ),
                    feed_medium=feed_b,
                ),
                "sample_1": Outflow(
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
        n_modeled_Inflows=len(rhs_ode.name_modeled_Inflows),
        n_modeled_Outflows=len(rhs_ode.name_modeled_Outflows),
        n_modeled_PVs=len(rhs_ode.name_modeled_PVs),
        controls=controls,
    )


def _inject_scales(
    reaction_module: UserReactionModule, scale_kwargs: dict[str, jnp.ndarray]
) -> UserReactionModule:
    """Replace placeholder SCALE_* fields with correctly-sized values.

    Wraps bare arrays in ``LinearScaler`` to match the scaler-typed fields
    (``tree_at`` bypasses the module ``__init__`` promotion).
    """
    return eqx.tree_at(
        lambda m: tuple(getattr(m, name) for name in scale_kwargs.keys()),
        reaction_module,
        tuple(
            v if isinstance(v, Scaler) else LinearScaler(v)
            for v in scale_kwargs.values()
        ),
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


@pytest.mark.parametrize(
    ("field", "modeled_feed", "n_latent", "method", "expected_shape"),
    [
        (
            "SCL_modeled_BiologicalOde_rates",
            False,
            0,
            "physical_rhs",
            (1,),
        ),
        ("SCL_modeled_Inflows_rates", True, 0, "physical_rhs", (1,)),
        ("SCL_latent_derivative", False, 2, "physical_rhs", (2,)),
        (
            "SCL_modeled_BiologicalOde_rates",
            False,
            0,
            "physical_save_outputs",
            (1,),
        ),
    ],
)
def test_wrapper_rejects_broadcastable_scalar_reaction_output(
    field, modeled_feed, n_latent, method, expected_shape
):
    """Reject scalars that downstream scaler operations would silently broadcast.

    Every fixed-layout output is covered through the RHS, and the duplicate module
    invocation in the save path gets an independent regression case.
    """
    process = _make_single_species_process()
    if modeled_feed:
        process.volume.volume_changes["feed_A"].is_controlled = False
    controls = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": process}, metadata={})
    ).get_controls("p1")
    outputs = {
        "biological_rates": jnp.zeros(1),
        "feed_rates": jnp.zeros(1 if modeled_feed else 0),
        "latent_derivative": jnp.zeros(n_latent),
    }
    output_names = {
        "SCL_modeled_BiologicalOde_rates": "biological_rates",
        "SCL_modeled_Inflows_rates": "feed_rates",
        "SCL_latent_derivative": "latent_derivative",
    }
    outputs[output_names[field]] = jnp.asarray(0.0)
    module = InvalidReactionShapeModule(
        **outputs,
        SCALE_latent=jnp.ones(n_latent),
    )
    wrapper = _build_wrapper(process, controls, module)
    y_phys = jnp.ones(2 + int(modeled_feed) + n_latent)

    with pytest.raises(ValueError) as exc_info:
        getattr(wrapper, method)(0.0, y_phys)

    assert str(exc_info.value) == (
        f"ReactionOutputs.{field} has shape (), expected {expected_shape}"
    )


@pytest.mark.parametrize(
    ("rate_field", "modeled_feed"),
    [
        ("SCALE_controlled_Inflows_rates", False),
        ("SCALE_modeled_BiologicalOde_rates", False),
        ("SCALE_modeled_Inflows_rates", True),
    ],
)
def test_wrapper_rejects_nonzero_rate_offset_for_direct_module(
    rate_field, modeled_feed
):
    # Common wrapper boundary must catch direct/custom module construction,
    # not only the estimate_all_scales hook resolver.
    process = _make_single_species_process()
    if modeled_feed:
        process.volume.volume_changes["feed_A"].is_controlled = False
    collection = BioProcessCollection(processes={"p1": process}, metadata={})
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    n_modeled_feeds = 1 if modeled_feed else 0
    module = ConstantReactionModule(
        specific_rates=jnp.zeros(1),
        modeled_Inflows_rates=jnp.zeros(n_modeled_feeds),
    )
    scales = _derive_unit_scale_kwargs(process, controls)
    scales[rate_field] = AffineScaler(
        jnp.ones(1),
        jnp.ones(1),
    )
    module = _inject_scales(module, scales)
    with pytest.raises(ValueError, match=f"{rate_field} is a rate axis"):
        HybridOdeWrapper.from_process(
            reaction_module=module,
            process=process,
            controls=controls,
        )


def test_physical_rhs_uses_custom_rate_derivative_semantics():
    class DivergentRateScaler(Scaler):
        scale: jnp.ndarray

        def __init__(self):
            self.scale = jnp.ones(1)

        def __rtruediv__(self, raw):
            return raw * 10.0

        def __rmul__(self, scl):
            return scl * 10.0

        def scale_derivative(self, rate):
            return rate / 3.0

        def unscale_derivative(self, rate):
            return rate * 3.0

        @property
        def shape(self):
            return self.scale.shape

        def astype(self, dtype):
            return self

        def __getitem__(self, idx):
            return self

    process = _make_single_species_process(feed_rate=0.0)
    collection = BioProcessCollection(processes={"p1": process}, metadata={})
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    module = ConstantReactionModule(
        specific_rates=jnp.asarray([2.0]),
        modeled_Inflows_rates=jnp.zeros(0),
    )
    scales = _derive_unit_scale_kwargs(process, controls)
    scales["SCALE_modeled_BiologicalOde_rates"] = DivergentRateScaler()
    wrapper = HybridOdeWrapper.from_process(
        reaction_module=_inject_scales(module, scales),
        process=process,
        controls=controls,
    )

    derivative = wrapper.physical_rhs(0.0, jnp.asarray([1.0, 1.0]))

    assert jnp.array_equal(derivative, jnp.asarray([6.0, 0.0]))


def test_wrapper_accepts_custom_offset_free_rate_scaler_without_offset_metadata():
    class UnitRateScaler(Scaler):
        scale: jnp.ndarray

        def __init__(self, scale):
            self.scale = scale

        def __rtruediv__(self, raw):
            return raw / self.scale

        def __rmul__(self, scl):
            return scl * self.scale

        def scale_derivative(self, rate):
            return rate / self.scale

        def unscale_derivative(self, rate):
            return rate * self.scale

        @property
        def shape(self):
            return self.scale.shape

        def astype(self, dtype):
            return UnitRateScaler(self.scale.astype(dtype))

        def __getitem__(self, idx):
            return UnitRateScaler(self.scale[idx])

    process = _make_single_species_process()
    collection = BioProcessCollection(processes={"p1": process}, metadata={})
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    module = ConstantReactionModule(
        specific_rates=jnp.zeros(1),
        modeled_Inflows_rates=jnp.zeros(0),
    )
    scales = _derive_unit_scale_kwargs(process, controls)
    scales["SCALE_modeled_BiologicalOde_rates"] = UnitRateScaler(jnp.ones(1))
    module = _inject_scales(module, scales)
    wrapper = HybridOdeWrapper.from_process(
        reaction_module=module,
        process=process,
        controls=controls,
    )
    assert wrapper.reaction_module is module


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
                "bolus_feed": Inflow(
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


def test_wrapper_appends_initial_latent_to_physical_state():
    process = _make_single_species_process()
    controls = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": process}, metadata={})
    ).get_controls("p1")
    module = LatentEchoReactionModule(
        initial_h0=jnp.asarray([7.0, 8.0]),
        SCALE_latent=jnp.asarray([2.0, 4.0]),
    )
    wrapper = _build_wrapper(process, controls, module)

    y0 = jnp.asarray([1.0, 2.0])

    assert jnp.array_equal(
        wrapper.initial_physical_state_from_raw(y0),
        jnp.asarray([1.0, 2.0, 7.0, 8.0]),
    )


def test_wrapper_rhs_appends_latent_derivative_without_physical_transport():
    process = _make_single_species_process(feed_rate=0.2)
    controls = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": process}, metadata={})
    ).get_controls("p1")
    module = LatentEchoReactionModule(
        initial_h0=jnp.zeros(2),
        SCALE_latent=jnp.asarray([2.0, 4.0]),
    )
    wrapper = _build_wrapper(process, controls, module)

    y = jnp.asarray([1.0, 1.0, 6.0, 20.0])
    dy = wrapper.physical_rhs(0.5, y)

    assert dy.shape == y.shape
    # `LatentEchoReactionModule` returns SCL dh/dt = [6/2, 20/4] + [1, 2].
    # The wrapper unscales that derivative and does not apply dilution/clamping.
    assert jnp.array_equal(dy[-2:], jnp.asarray([8.0, 28.0]))


def test_wrapper_save_outputs_passes_latent_but_saves_physical_state_only():
    process = _make_single_species_process()
    controls = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": process}, metadata={})
    ).get_controls("p1")
    module = LatentEchoReactionModule(
        initial_h0=jnp.zeros(2),
        SCALE_latent=jnp.asarray([2.0, 4.0]),
    )
    wrapper = _build_wrapper(process, controls, module)

    outputs = wrapper.physical_save_outputs(0.5, jnp.asarray([1.0, 1.0, 6.0, 20.0]))

    assert outputs.SCL_states.shape == (2,)
    assert jnp.array_equal(outputs.SCL_states, jnp.asarray([1.0, 1.0]))
    assert outputs.auxiliary is not None
    assert jnp.array_equal(outputs.auxiliary["SCL_latent"], jnp.asarray([3.0, 5.0]))


def test_physical_rhs_passes_process_minimum_volume_to_bp_format():
    process = _make_single_species_process(feed_rate=0.0)
    controls = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": process}, metadata={})
    ).get_controls("p1")
    module = ConstantReactionModule(
        specific_rates=jnp.zeros((1,)), modeled_Inflows_rates=jnp.zeros((0,))
    )
    wrapper = _build_wrapper(process, controls, module)

    with pytest.raises(Exception, match="minimum reactor volume"):
        wrapper.physical_rhs(0.0, jnp.asarray([1.0, controls.min_V]))


@pytest.mark.parametrize("initial_volume", [0.001, 0.0005])
def test_solve_rejects_initial_volume_at_or_below_minimum(initial_volume):
    from hybrax.train.physical_solve import solve_physical_states

    process = _make_single_species_process()
    controls = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": process}, metadata={})
    ).get_controls("p1")
    module = ConstantReactionModule(
        specific_rates=jnp.zeros((1,)), modeled_Inflows_rates=jnp.zeros((0,))
    )
    wrapper = _build_wrapper(process, controls, module)

    with pytest.raises(Exception, match="initial state reached minimum reactor volume"):
        solve_physical_states(
            wrapper,
            t_eval=jnp.asarray([0.0, 1.0]),
            n_measured=2,
            RAW_y0=jnp.asarray([1.0, initial_volume]),
            max_steps=10_000,
            rtol=1e-6,
            atol=1e-8,
        )


@pytest.mark.parametrize("event_time", [0.0, 1.0])
@pytest.mark.parametrize("sample_volume", [0.999, 1.0])
def test_solve_rejects_sample_volume_at_or_below_minimum(sample_volume, event_time):
    from hybrax.train.physical_solve import solve_physical_states

    process = _make_single_species_process(feed_rate=0.0)
    controls = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": process}, metadata={})
    ).get_controls("p1")
    controls = eqx.tree_at(
        lambda c: (
            c.sample_event_times,
            c.sample_event_volumes,
            c.sample_event_mask,
            c.min_V,
        ),
        controls,
        (
            jnp.asarray([event_time]),
            jnp.asarray([sample_volume]),
            jnp.asarray([True]),
            jnp.asarray(1.0) - jnp.asarray(sample_volume),
        ),
    )
    module = ConstantReactionModule(
        specific_rates=jnp.zeros((1,)), modeled_Inflows_rates=jnp.zeros((0,))
    )
    wrapper = _build_wrapper(process, controls, module)

    with pytest.raises(Exception, match="sample reached minimum reactor volume"):
        solve_physical_states(
            wrapper,
            t_eval=jnp.asarray([0.0, 1.0, 2.0]),
            n_measured=3,
            RAW_y0=jnp.asarray([1.0, 1.0]),
            max_steps=10_000,
            rtol=1e-6,
            atol=1e-8,
        )


@pytest.mark.parametrize("batched", [False, True])
def test_valid_sample_is_not_speculatively_reapplied(batched):
    from hybrax.train.physical_solve import solve_physical_states

    process = _make_single_species_process(feed_rate=0.0)
    controls = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": process}, metadata={})
    ).get_controls("p1")
    controls = eqx.tree_at(
        lambda c: (
            c.sample_event_times,
            c.sample_event_volumes,
            c.sample_event_mask,
        ),
        controls,
        (jnp.asarray([1.0]), jnp.asarray([0.998]), jnp.asarray([True])),
    )
    module = ConstantReactionModule(
        specific_rates=jnp.zeros((1,)), modeled_Inflows_rates=jnp.zeros((0,))
    )
    wrapper = _build_wrapper(process, controls, module)

    def solve(y0):
        return solve_physical_states(
            wrapper,
            t_eval=jnp.asarray([0.0, 1.0, 2.0]),
            n_measured=3,
            RAW_y0=y0,
            max_steps=10_000,
            rtol=1e-6,
            atol=1e-8,
        )

    states = (
        eqx.filter_jit(jax.vmap(solve))(jnp.asarray([[1.0, 1.0]]))[0]
        if batched
        else solve(jnp.asarray([1.0, 1.0]))
    )

    assert states[-1, 1] == pytest.approx(0.002, abs=1e-6)


def test_vmap_masks_preset_affect_for_lane_without_trigger():
    from hybrax.train.physical_solve import solve_physical_states

    process = _make_single_species_process(feed_rate=0.0)
    controls = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": process}, metadata={})
    ).get_controls("p1")
    module = ConstantReactionModule(
        specific_rates=jnp.zeros((1,)), modeled_Inflows_rates=jnp.zeros((0,))
    )
    wrapper = _build_wrapper(process, controls, module)

    def solve(event_time):
        lane_controls = eqx.tree_at(
            lambda c: (
                c.sample_event_times,
                c.sample_event_volumes,
                c.sample_event_mask,
            ),
            controls,
            (event_time[None], jnp.asarray([0.998]), jnp.asarray([True])),
        )
        lane_wrapper = eqx.tree_at(lambda w: w.controls, wrapper, lane_controls)
        return solve_physical_states(
            lane_wrapper,
            t_eval=jnp.asarray([0.0, 1.0, 2.0]),
            n_measured=3,
            RAW_y0=jnp.asarray([1.0, 1.0]),
            max_steps=10_000,
            rtol=1e-6,
            atol=1e-8,
        )

    states = eqx.filter_jit(jax.vmap(solve))(jnp.asarray([1.0, 2.0]))

    assert jnp.allclose(states[:, -1, 1], jnp.asarray([0.002, 0.002]), atol=1e-6)


def test_start_time_sample_and_bolus_are_applied_once():
    from hybrax.train.physical_solve import solve_physical_states

    process = _make_single_species_process(feed_rate=0.0)
    controls = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": process}, metadata={})
    ).get_controls("p1")
    controls = eqx.tree_at(
        lambda c: (
            c.sample_event_times,
            c.sample_event_volumes,
            c.sample_event_mask,
            c.bolus_event_times,
            c.bolus_event_volumes,
            c.bolus_event_Cin,
            c.bolus_event_mask,
        ),
        controls,
        (
            jnp.asarray([0.0]),
            jnp.asarray([0.2]),
            jnp.asarray([True]),
            jnp.asarray([0.0]),
            jnp.asarray([0.1]),
            jnp.asarray([[3.0]]),
            jnp.asarray([True]),
        ),
    )
    module = ConstantReactionModule(
        specific_rates=jnp.zeros((1,)), modeled_Inflows_rates=jnp.zeros((0,))
    )
    wrapper = _build_wrapper(process, controls, module)

    states = solve_physical_states(
        wrapper,
        t_eval=jnp.asarray([0.0, 0.0, 1.0]),
        n_measured=3,
        RAW_y0=jnp.asarray([1.0, 1.0]),
        max_steps=10_000,
        rtol=1e-6,
        atol=1e-8,
    )

    assert jnp.allclose(states[:2], jnp.asarray([[1.0, 0.8], [1.0, 0.8]]))
    assert jnp.allclose(states[2], jnp.asarray([1.1 / 0.9, 0.9]), atol=1e-5)


def test_rhs_compatibility_checks_outflow_retention_shapes():
    rhs_ode = build_rhs_ode(_make_mixed_flow_process())
    mismatched = eqx.tree_at(
        lambda rhs: rhs.retention_modeled_Outflows,
        rhs_ode,
        jnp.zeros((2, 1)),
    )

    with pytest.raises(ValueError, match="retention_modeled_Outflows shapes differ"):
        validate_rhs_ode_compatibility("reference", rhs_ode, "candidate", mismatched)


def test_mixed_flow_rhs_matches_direct_rhs_and_hand_mass_balance():
    process = _make_mixed_flow_process()
    controls = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": process}, metadata={})
    ).get_controls("p1")
    module = ConstantReactionModule(
        specific_rates=jnp.zeros(1),
        modeled_Inflows_rates=jnp.asarray([0.4]),
        modeled_outflow_rates=jnp.asarray([-0.2]),
    )
    wrapper = _build_wrapper(process, controls, module)

    # [biomass, V, modeled Inflow cumulative, modeled Outflow cumulative]
    state = jnp.asarray([10.0, 2.0, 0.0, 0.0])
    derivative = wrapper.physical_rhs(0.0, state)

    direct_rhs = build_rhs_ode(process)
    controlled_outflow_rates = {
        "harvest": -0.1,
        "evaporation": -0.3,
    }
    direct_controls = jnp.asarray(
        [0.2]
        + [
            controlled_outflow_rates[name]
            for name in direct_rhs.name_controlled_Outflows
        ]
    )
    direct_derivative = direct_rhs(
        state[:2],
        jnp.zeros(1),
        direct_controls,
        jnp.asarray([0.4]),
        jnp.asarray([-0.2]),
        controls.min_V,
    )
    assert jnp.allclose(derivative[:2], direct_derivative)

    # Component-mass removal by harvest/perfusion/evaporation at retention
    # 0/0.5/1 is -(1-retention) * removal_rate * concentration.
    mass_derivatives = jnp.asarray(
        [-(1.0 - 0.0) * 0.1 * 10.0, -(1.0 - 0.5) * 0.2 * 10.0, 0.0]
    )
    expected_mass_derivative = jnp.sum(mass_derivatives)
    expected_volume_derivative = 0.2 + 0.4 - 0.1 - 0.2 - 0.3
    expected_concentration_derivative = (
        expected_mass_derivative - 10.0 * expected_volume_derivative
    ) / 2.0
    assert jnp.allclose(mass_derivatives, jnp.asarray([-1.0, -1.0, 0.0]))
    assert expected_mass_derivative == -2.0
    assert expected_volume_derivative == pytest.approx(0.0)
    assert expected_concentration_derivative == pytest.approx(-1.0)
    assert jnp.allclose(derivative, jnp.asarray([-1.0, 0.0, 0.4, -0.2]))

    assert jnp.array_equal(wrapper.rhs_ode.Cin_controlled_Inflows, jnp.asarray([[0.0]]))
    assert jnp.array_equal(wrapper.rhs_ode.Cin_modeled_Inflows, jnp.asarray([[0.0]]))
    controlled_retention = dict(
        zip(
            wrapper.rhs_ode.name_controlled_Outflows,
            wrapper.rhs_ode.retention_controlled_Outflows[:, 0].tolist(),
            strict=True,
        )
    )
    assert controlled_retention == {"harvest": 0.0, "evaporation": 1.0}
    assert jnp.array_equal(
        wrapper.rhs_ode.retention_modeled_Outflows, jnp.asarray([[0.5]])
    )


def test_continuous_feed_transport_volume_and_dilution():
    """Self-contained continuous-feed volume regression (no ``examples/`` dependency).

    Single biomass species, biomass-free continuous feed (0.2 L/h) plus a 0.1 L sample
    at t=1, zero reaction (pure transport). Guards:
      * the t=0 export equals y0 (initial-state correctness),
      * V(t) tracks ``v0 + ∫feed − samples`` with the post-sample drop applied,
      * biomass amount ``X·V`` is conserved by the biomass-free feed and only drops at
        the sample (well-mixed removal),
      * a duplicated-t0 grid (the measurement/dense/prediction union can carry t0 at
        several indices) returns y0 at *every* t0 row (V0 dense-export boundary fix).
    """
    from hybrax.train.physical_solve import solve_physical_states

    # Biomass-free continuous feed stored as CUMULATIVE volume (0 -> 0.4 over [0, 2] =
    # 0.2 L/h) plus a 0.1 L sample at t=1; built inline so the feed is a real flow (a
    # constant-rate fixture reads as a constant cumulative, i.e. zero flow).
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
    process = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "feed_A": Inflow(
                    name="feed_A",
                    unit="L",
                    is_controlled=True,
                    is_continuous=True,
                    values=TimeSeries(
                        times=jnp.asarray([0.0, 2.0]), values=jnp.asarray([0.0, 0.4])
                    ),
                    feed_medium=feed_medium,
                ),
                "sample_1": Outflow(
                    name="sample_1",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([1.0]), values=jnp.asarray([-0.1])
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
                        times=jnp.asarray([0.0, 2.0]), values=jnp.asarray([1.0, 1.0])
                    ),
                )
            },
        ),
        process_variables={},
    )
    collection = BioProcessCollection(processes={"p1": process}, metadata={})
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    module = ConstantReactionModule(
        specific_rates=jnp.zeros((1,)),  # zero reaction
        modeled_Inflows_rates=jnp.zeros((0,)),
    )
    wrapper = _build_wrapper(process, controls, module)

    # state layout is [biomass, V]; initial biomass=1.0, V0=1.0
    y0 = jnp.asarray([1.0, 1.0])
    t_eval = jnp.linspace(0.0, 2.0, 21)
    states = solve_physical_states(
        wrapper,
        t_eval=t_eval,
        n_measured=t_eval.shape[0],
        RAW_y0=y0,
        max_steps=100_000,
        rtol=1e-8,
        atol=1e-10,
    )
    biomass, volume = states[:, 0], states[:, 1]

    assert jnp.allclose(states[0], y0)  # t=0 export == initial state

    # V(t) = v0 + 0.2 t, with a 0.1 L post-sample drop from t=1 onward.
    expected_V = jnp.where(t_eval < 1.0, 1.0 + 0.2 * t_eval, 1.0 + 0.2 * t_eval - 0.1)
    assert jnp.allclose(volume, expected_V, atol=1e-3)

    # Biomass-free feed conserves X·V; the sample removes biomass proportionally so
    # X·V drops once: 1.0 -> (1.0/1.2)*1.1 = 0.9167 from the sample onward.
    xv = biomass * volume
    assert jnp.allclose(xv[t_eval < 1.0], 1.0, atol=2e-3)
    assert jnp.allclose(xv[t_eval >= 1.0], (1.0 / 1.2) * 1.1, atol=2e-3)

    # Regression: t0 repeated across the grid must all return y0 (not the gather
    # boundary value with the first feed interval already integrated in).
    dup = jnp.asarray([0.0, 0.0, 0.0, 1.0, 2.0])
    dup_states = solve_physical_states(
        wrapper,
        t_eval=dup,
        n_measured=dup.shape[0],
        RAW_y0=y0,
        max_steps=100_000,
        rtol=1e-8,
        atol=1e-10,
    )
    assert jnp.allclose(dup_states[:3], y0[None, :])


def _make_modeled_pv_process() -> BioProcess:
    """biomass RMC + one uncontrolled (modeled) process variable.

    With no user biological_ode, auto-generation yields rates
    ``(q_biomass, r_<pv>)`` and a PV derivative ``r_<pv>`` — i.e. the PV is an
    MLP-predicted, integrated state alongside the RMCs.
    """
    return BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes={}),
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
        process_variables={
            "ratio": ProcessVariable(
                name="ratio",
                unit="-",
                is_controlled=False,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 2.0]),
                    values=jnp.asarray([0.0, 1.0]),
                ),
            ),
        },
    )


def test_wrapper_supports_modeled_pv():
    from hybrax.train.physical_solve import solve_physical_states

    process = _make_modeled_pv_process()
    rhs_ode = build_rhs_ode(process)
    # Auto-generated ODE: a biomass rate plus an r_<pv> rate; the PV is a state.
    assert rhs_ode.name_modeled_PVs == ("ratio",)
    assert rhs_ode.name_modeled_rates == ("q_biomass", "r_ratio")

    collection = BioProcessCollection(processes={"p1": process}, metadata={})
    controls = ControlsStore.from_collection(collection).get_controls("p1")

    # q_biomass = 0 (biomass constant); r_ratio = 0.5 (PV grows 0.5 / h).
    module = ConstantReactionModule(
        specific_rates=jnp.asarray([0.0, 0.5]),
        modeled_Inflows_rates=jnp.zeros((0,)),
    )
    wrapper = _build_wrapper(process, controls, module)  # must NOT raise
    assert wrapper.modeled_PV_names == ("ratio",)

    # State layout is [biomass | ratio | V]; the PV slot is present.
    n_state = (
        len(rhs_ode.name_modeled_RMCs)
        + len(rhs_ode.name_modeled_PVs)
        + 1
        + len(rhs_ode.name_modeled_Inflows)
        + len(rhs_ode.name_modeled_Outflows)
    )
    assert n_state == 3
    y0 = jnp.asarray([1.0, 0.0, 1.0])  # biomass, ratio, V

    # RHS at t=0: [d_biomass=0 | d_ratio=0.5 (biological-only) | d_V=0].
    d0 = wrapper.physical_rhs(0.0, y0)
    assert d0.shape == (3,)
    assert jnp.allclose(d0, jnp.asarray([0.0, 0.5, 0.0]), atol=1e-6)

    # Integrate: biomass + V constant (no feed), ratio(t) = 0.5 t.
    t_eval = jnp.linspace(0.0, 2.0, 11)
    states = solve_physical_states(
        wrapper,
        t_eval=t_eval,
        n_measured=t_eval.shape[0],
        RAW_y0=y0,
        max_steps=100_000,
        rtol=1e-8,
        atol=1e-10,
    )
    biomass, ratio, volume = states[:, 0], states[:, 1], states[:, 2]
    assert jnp.allclose(biomass, 1.0, atol=1e-4)
    assert jnp.allclose(volume, 1.0, atol=1e-4)
    assert jnp.allclose(ratio, 0.5 * t_eval, atol=1e-3)
