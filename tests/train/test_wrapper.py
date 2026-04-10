from __future__ import annotations

import jax.numpy as jnp
import pytest
from bpbench.dataclasses import (
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

from bp_train.controls_store import ControlsStore
from bp_train.model_api import ReactionOutputs, UserReactionModule
from bp_train.wrapper import (
    HybridOdeWrapper,
    validate_rhs_ode_compatibility,
)


class ConstantReactionModule(UserReactionModule):
    """Test reaction module returning fixed `ReactionOutputs`."""

    specific_rates: jnp.ndarray
    modeled_feed_rates: jnp.ndarray

    def __init__(self, specific_rates: jnp.ndarray, modeled_feed_rates: jnp.ndarray):
        self.specific_rates = specific_rates
        self.modeled_feed_rates = modeled_feed_rates

    def __call__(self, t, c_species, controls_vector):
        del t, c_species, controls_vector
        return ReactionOutputs(
            specific_rates=self.specific_rates,
            modeled_feed_rates=self.modeled_feed_rates,
        )


class InvalidReactionShapeModule(UserReactionModule):
    """Test reaction module returning malformed output ranks."""

    def __call__(self, t, c_species, controls_vector):
        del t, c_species, controls_vector
        return ReactionOutputs(
            specific_rates=jnp.asarray([[0.1]], dtype=jnp.float32),
            modeled_feed_rates=jnp.zeros((0,), dtype=jnp.float32),
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
                    is_intracellular=False,
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
                    is_intracellular=False,
                ),
                "product": ReactorMediumComponent(
                    name="product",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray([0.0, 2.0]),
                        values=jnp.asarray([2.0, 2.0]),
                    ),
                    is_intracellular=False,
                ),
            },
        ),
        process_variables={},
    )


def _build_wrapper(process, controls, reaction_module):
    return HybridOdeWrapper.from_process(
        reaction_module=reaction_module,
        process=process,
        controls=controls,
    )


def test_wrapper_produces_finite_state_derivative():
    process = _make_single_species_process(
        feed_rate=0.2,
        feed_biomass_concentration=0.0,
    )
    collection = BioProcessCollection(processes={"p1": process}, metadata={})
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    reaction_module = ConstantReactionModule(
        specific_rates=jnp.asarray([0.3], dtype=jnp.float32),
        modeled_feed_rates=jnp.zeros((0,), dtype=jnp.float32),
    )
    wrapper = _build_wrapper(process, controls, reaction_module)

    y = jnp.asarray([1.0, 1.2], dtype=jnp.float32)
    dy = wrapper(2.0, y)

    assert dy.shape == (2,)
    assert jnp.all(jnp.isfinite(dy))


def test_wrapper_with_modeled_feed_produces_finite_derivative():
    """Process with uncontrolled feed (base_feed) modeled by the MLP."""
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
    base_feed_medium = FeedMedium(
        name="base_feed",
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
    process = BioProcess(
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
                        values=jnp.asarray([0.1, 0.1]),
                    ),
                    feed_medium=feed_medium,
                ),
                "base_feed": FeedVolumeChange(
                    name="base_feed",
                    unit="L/h",
                    is_controlled=False,
                    is_continuous=True,
                    values=TimeSeries(
                        times=jnp.asarray([0.0, 2.0]),
                        values=jnp.asarray([0.3, 0.3]),
                    ),
                    feed_medium=base_feed_medium,
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
                    is_intracellular=False,
                ),
            },
        ),
        process_variables={},
    )
    collection = BioProcessCollection(processes={"p1": process}, metadata={})
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    reaction_module = ConstantReactionModule(
        specific_rates=jnp.asarray([0.0], dtype=jnp.float32),
        modeled_feed_rates=jnp.asarray([0.3], dtype=jnp.float32),
    )
    wrapper = _build_wrapper(process, controls, reaction_module)

    # State layout: [biomass, V_cont, B_base_feed_cum]
    y = jnp.asarray([1.0, 1.0, 0.0], dtype=jnp.float32)
    dy = wrapper(0.0, y)

    assert dy.shape == (3,)
    assert jnp.all(jnp.isfinite(dy))


def test_wrapper_multiple_controlled_feeds():
    process = _make_two_species_two_feed_process()
    collection = BioProcessCollection(processes={"p1": process}, metadata={})
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    reaction_module = ConstantReactionModule(
        specific_rates=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        modeled_feed_rates=jnp.zeros((0,), dtype=jnp.float32),
    )
    wrapper = _build_wrapper(process, controls, reaction_module)

    y = jnp.asarray([1.0, 2.0, 1.2], dtype=jnp.float32)
    dy = wrapper(2.0, y)

    assert dy.shape == (3,)
    assert jnp.all(jnp.isfinite(dy))


def test_wrapper_rejects_modeled_rate_shape_mismatch():
    """base_feed requires 1 modeled rate but module returns 2."""
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
    base_feed_medium = FeedMedium(
        name="base_feed",
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
                "feed_A": FeedVolumeChange(
                    name="feed_A",
                    unit="L/h",
                    is_controlled=True,
                    is_continuous=True,
                    values=TimeSeries(
                        times=jnp.asarray([0.0, 2.0]),
                        values=jnp.asarray([0.2, 0.2]),
                    ),
                    feed_medium=feed_medium,
                ),
                "base_feed": FeedVolumeChange(
                    name="base_feed",
                    unit="L/h",
                    is_controlled=False,
                    is_continuous=True,
                    values=TimeSeries(
                        times=jnp.asarray([0.0, 2.0]),
                        values=jnp.asarray([0.1, 0.1]),
                    ),
                    feed_medium=base_feed_medium,
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
                    is_intracellular=False,
                ),
            },
        ),
        process_variables={},
    )
    collection = BioProcessCollection(processes={"p1": process}, metadata={})
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    # base_feed is modeled (1 flow) but module returns 2 rates
    reaction_module = ConstantReactionModule(
        specific_rates=jnp.asarray([0.0], dtype=jnp.float32),
        modeled_feed_rates=jnp.asarray([0.2, 0.3], dtype=jnp.float32),
    )
    wrapper = _build_wrapper(process, controls, reaction_module)

    with pytest.raises(ValueError, match="modeled_feed_rates must have shape"):
        # State layout: [biomass, V_cont, B_base_feed_cum]
        wrapper(0.0, jnp.asarray([1.0, 1.0, 0.0], dtype=jnp.float32))


def test_wrapper_rejects_invalid_state_vector_shape():
    process = _make_single_species_process()
    collection = _make_single_species_collection()
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    reaction_module = ConstantReactionModule(
        specific_rates=jnp.asarray([0.0], dtype=jnp.float32),
        modeled_feed_rates=jnp.zeros((0,), dtype=jnp.float32),
    )
    wrapper = _build_wrapper(process, controls, reaction_module)

    with pytest.raises(ValueError, match="state vector y must have shape"):
        wrapper(0.0, jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32))


def test_wrapper_augmented_controls_names_includes_cin():
    process = _make_single_species_process()
    collection = _make_single_species_collection()
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    reaction_module = ConstantReactionModule(
        specific_rates=jnp.asarray([0.0], dtype=jnp.float32),
        modeled_feed_rates=jnp.zeros((0,), dtype=jnp.float32),
    )
    wrapper = _build_wrapper(process, controls, reaction_module)

    assert "cin:feed_A:biomass" in wrapper.augmented_controls_names
    assert len(wrapper.augmented_controls_units) == len(
        wrapper.augmented_controls_names
    )


def test_validate_rhs_ode_compatibility_rejects_different_species():
    from bpbench.mechanistic import get_rhs_ode

    process_a = _make_single_species_process()
    process_b = _make_two_species_two_feed_process()
    rhs_a = get_rhs_ode(process_a)
    rhs_b = get_rhs_ode(process_b)

    with pytest.raises(ValueError, match="reactor_component_state_names differ"):
        validate_rhs_ode_compatibility("a", rhs_a, "b", rhs_b)


def test_wrapper_constant_feed_rate_integrates_volume_correctly():
    """Regression test for the U_flow units bug.

    Build a single-process collection with one species (biomass), one
    controlled feed `feed_A` with cumulative values [0.0, 1.0] at times
    [0.0, 10.0] (i.e. constant flow rate of 0.1 kg/h), V0=1.0, no sampling,
    no modeled feeds, and a zero reaction module.  Integrate from t=0 to
    t=10 and assert the final V_cont ~= 2.0 (V0 + 1.0 of feed).

    Before the fix this would integrate ~6.0 (because the wrapper was
    treating the cumulative value as a flow rate).
    """
    import diffrax

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
        time_axis=TimeAxis(unit="h", start=0.0, end=10.0, time_reference="start"),
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
                        times=jnp.asarray([0.0, 10.0]),
                        values=jnp.asarray([0.0, 1.0]),  # cumulative kg
                    ),
                    feed_medium=feed_medium,
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
                        times=jnp.asarray([0.0, 10.0]),
                        values=jnp.asarray([1.0, 1.0]),
                    ),
                    is_intracellular=False,
                ),
            },
        ),
        process_variables={},
    )
    collection = BioProcessCollection(processes={"p1": process}, metadata={})
    controls = ControlsStore.from_collection(collection).get_controls("p1")

    reaction_module = ConstantReactionModule(
        specific_rates=jnp.asarray([0.0], dtype=jnp.float32),
        modeled_feed_rates=jnp.zeros((0,), dtype=jnp.float32),
    )
    wrapper = HybridOdeWrapper.from_process(
        reaction_module=reaction_module,
        process=process,
        controls=controls,
    )

    # State layout: [biomass, V_cont] (no modeled feeds → state size = 2)
    y0_physical = jnp.asarray([1.0, 1.0], dtype=jnp.float32)
    y0_scaled = wrapper.scale_state(y0_physical)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: wrapper(t, y)),
        diffrax.Tsit5(),
        t0=0.0,
        t1=10.0,
        dt0=None,
        y0=y0_scaled,
        saveat=diffrax.SaveAt(ts=jnp.asarray([10.0])),
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
        max_steps=4096,
        throw=False,
    )
    final_physical = wrapper.unscale_state(sol.ys[0])
    final_v_cont = float(final_physical[-1])

    # V_cont(10) = V0 + cumulative_feed(10) = 1.0 + 1.0 = 2.0
    assert final_v_cont == pytest.approx(2.0, rel=1e-3, abs=1e-3), (
        f"final V_cont = {final_v_cont}, expected ~2.0. "
        "If this is ~6.0 the U_flow units bug regressed: the wrapper is "
        "passing controls.eval (cumulative volume) instead of "
        "controls.eval_derivative (flow rate) to RhsOde."
    )
