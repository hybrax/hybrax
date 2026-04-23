from __future__ import annotations

import diffrax
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
from bp_format.mechanistic import get_rhs_ode

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


class VolumeFeatureEchoReactionModule(UserReactionModule):
    """Reaction module that echoes the last ANN control feature into q."""

    n_species: int
    n_modeled: int
    expects_v_real_feature: bool

    def __init__(
        self,
        *,
        n_species: int,
        n_modeled: int = 0,
        expects_v_real_feature: bool = False,
    ):
        self.n_species = n_species
        self.n_modeled = n_modeled
        self.expects_v_real_feature = expects_v_real_feature

    def __call__(self, t, c_species, controls_vector):
        del t, c_species
        v_feature = jnp.asarray(controls_vector[-1], dtype=jnp.float32)
        return ReactionOutputs(
            specific_rates=jnp.full((self.n_species,), v_feature, dtype=jnp.float32),
            modeled_feed_rates=jnp.zeros((self.n_modeled,), dtype=jnp.float32),
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


def test_wrapper_optional_ann_volume_feature_uses_v_real():
    process = _make_single_species_process()
    collection = _make_single_species_collection()
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    reaction_module = VolumeFeatureEchoReactionModule(
        n_species=1,
        expects_v_real_feature=True,
    )
    wrapper = _build_wrapper(process, controls, reaction_module)

    assert wrapper.include_v_real_feature is True
    assert wrapper.augmented_controls_names[-1] == "v_real"

    # t=2.0 is after the sample event at t=1.0; V_sample_acc ~= 0.1 L.
    y_physical = jnp.asarray([1.0, 1.2], dtype=jnp.float32)
    y_scaled = wrapper.scale_state(y_physical)
    dy_scaled = wrapper(2.0, y_scaled)
    dy_physical = dy_scaled * wrapper.state_scale
    assert float(dy_physical[0]) == pytest.approx(1.1, rel=1e-6, abs=1e-6)


def test_wrapper_save_outputs_splits_export_and_runtime_v_real():
    process = _make_single_species_process(feed_rate=0.0)
    collection = _make_single_species_collection(feed_rate=0.0)
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    reaction_module = VolumeFeatureEchoReactionModule(
        n_species=1,
        expects_v_real_feature=True,
    )
    wrapper = HybridOdeWrapper.from_process(
        reaction_module=reaction_module,
        process=process,
        controls=controls,
        min_real_volume=0.02,
    )

    # At t=2.0 the accumulated sample volume is 0.1 L.
    y_physical = jnp.asarray([1.0, 0.05], dtype=jnp.float32)
    saved = wrapper.save_outputs(2.0, wrapper.scale_state(y_physical))

    assert jnp.allclose(saved.states_physical, y_physical)
    assert float(saved.v_real_export) == pytest.approx(-0.05, abs=1e-6)
    assert float(saved.v_real_runtime) == pytest.approx(0.02, abs=1e-6)
    # The optional ANN feature must use runtime semantics, not export semantics.
    assert float(saved.specific_rates_physical[0]) == pytest.approx(0.02, abs=1e-6)
    assert saved.modeled_feed_rates_physical.shape == (0,)


def test_wrapper_save_outputs_returns_physical_specific_and_modeled_feed_rates():
    """Save payload should expose unscaled physical q and modeled feed rates."""
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
    wrapper = HybridOdeWrapper.from_process(
        reaction_module=ConstantReactionModule(
            specific_rates=jnp.asarray([0.25], dtype=jnp.float32),
            modeled_feed_rates=jnp.asarray([0.3], dtype=jnp.float32),
        ),
        process=process,
        controls=controls,
        q_scale=jnp.asarray([4.0], dtype=jnp.float32),
        f_scale=jnp.asarray([2.5], dtype=jnp.float32),
    )

    y_physical = jnp.asarray([1.0, 1.0, 0.7], dtype=jnp.float32)
    saved = wrapper.save_outputs(0.0, wrapper.scale_state(y_physical))

    assert jnp.allclose(saved.states_physical, y_physical)
    assert jnp.allclose(
        saved.specific_rates_physical,
        jnp.asarray([1.0], dtype=jnp.float32),
    )
    assert jnp.allclose(
        saved.modeled_feed_rates_physical,
        jnp.asarray([jax.nn.softplus(0.3) * 2.5], dtype=jnp.float32),
    )


def test_wrapper_save_outputs_works_with_diffrax_saveat_fn():
    process = _make_single_species_process(feed_rate=0.0)
    collection = _make_single_species_collection(feed_rate=0.0)
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    wrapper = _build_wrapper(
        process,
        controls,
        ConstantReactionModule(
            specific_rates=jnp.asarray([0.4], dtype=jnp.float32),
            modeled_feed_rates=jnp.zeros((0,), dtype=jnp.float32),
        ),
    )

    y0_physical = jnp.asarray([1.0, 1.2], dtype=jnp.float32)
    y0_scaled = wrapper.scale_state(y0_physical)
    save_ts = jnp.asarray([2.0], dtype=jnp.float32)
    solve_kwargs = dict(
        solver=diffrax.Tsit5(),
        t0=0.0,
        t1=2.0,
        dt0=None,
        y0=y0_scaled,
        stepsize_controller=diffrax.PIDController(
            rtol=1e-6,
            atol=1e-8,
            jump_ts=controls.active_step_ts,
        ),
        max_steps=4096,
        throw=False,
    )

    state_sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: wrapper(t, y)),
        saveat=diffrax.SaveAt(ts=save_ts),
        **solve_kwargs,
    )
    save_sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: wrapper(t, y)),
        saveat=diffrax.SaveAt(ts=save_ts, fn=wrapper.save_outputs),
        **solve_kwargs,
    )

    assert state_sol.result == diffrax.RESULTS.successful
    assert save_sol.result == diffrax.RESULTS.successful

    expected = wrapper.save_outputs(2.0, state_sol.ys[0])
    assert jnp.allclose(save_sol.ys.states_physical[0], expected.states_physical)
    assert float(save_sol.ys.v_real_export[0]) == pytest.approx(
        float(expected.v_real_export),
        abs=1e-6,
    )
    assert float(save_sol.ys.v_real_runtime[0]) == pytest.approx(
        float(expected.v_real_runtime),
        abs=1e-6,
    )
    assert jnp.allclose(
        save_sol.ys.specific_rates_physical[0],
        expected.specific_rates_physical,
    )
    assert jnp.allclose(
        save_sol.ys.modeled_feed_rates_physical[0],
        expected.modeled_feed_rates_physical,
    )


def test_validate_rhs_ode_compatibility_rejects_different_species():
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


def test_wrapper_bolus_feed_integrates_v_cont():
    """Regression: non-continuous controlled bolus contributes to dV_cont/dt."""
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
        time_axis=TimeAxis(unit="h", start=0.0, end=10.0, time_reference="start"),
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
                        times=jnp.asarray([2.0]),
                        values=jnp.asarray([2.0]),  # 2 L bolus amount
                    ),
                    feed_medium=bolus_medium,
                ),
                "sample_dummy": SampleVolumeChange(
                    name="sample_dummy",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
                        values=jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0]),
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
                        times=jnp.asarray([0.0, 4.0]),
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
    wrapper = _build_wrapper(process, controls, reaction_module)

    y0_physical = jnp.asarray([1.0, 1.0], dtype=jnp.float32)
    y0_scaled = wrapper.scale_state(y0_physical)
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: wrapper(t, y)),
        solver=diffrax.Tsit5(),
        t0=0.0,
        t1=4.0,
        dt0=None,
        y0=y0_scaled,
        saveat=diffrax.SaveAt(ts=jnp.asarray([4.0], dtype=jnp.float32)),
        stepsize_controller=diffrax.PIDController(
            rtol=1e-6,
            atol=1e-8,
            jump_ts=controls.active_step_ts,
        ),
        max_steps=4096,
        throw=False,
    )
    assert solution.result == diffrax.RESULTS.successful
    final_physical = wrapper.unscale_state(solution.ys[0])
    final_v_cont = float(final_physical[-1])

    # V_cont must increase by the specified bolus volume (2.0 L).
    # (initial V_cont = 1.0, bolus = 2.0 => final ~= 3.0)
    assert final_v_cont == pytest.approx(1.0 + 2.0, abs=2e-3)


def test_wrapper_bolus_transport_only_for_present_species():
    """Regression: bolus Cin for missing species is zero in extra-feed path."""
    bolus_medium = FeedMedium(
        name="bolus",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": FeedMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=StaticVariable(10.0),
                is_controlled=False,
            ),
        },
    )
    process = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=10.0, time_reference="start"),
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
                        times=jnp.asarray([2.0]),
                        values=jnp.asarray([2.0]),
                    ),
                    feed_medium=bolus_medium,
                ),
                "sample_dummy": SampleVolumeChange(
                    name="sample_dummy",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([0.0, 1.0, 2.0, 3.0, 10.0]),
                        values=jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0]),
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
                        times=jnp.asarray([0.0, 10.0]),
                        values=jnp.asarray([1.0, 1.0]),
                    ),
                    is_intracellular=False,
                ),
                "product": ReactorMediumComponent(
                    name="product",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray([0.0, 10.0]),
                        values=jnp.asarray([0.0, 0.0]),
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
        specific_rates=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        modeled_feed_rates=jnp.zeros((0,), dtype=jnp.float32),
    )
    wrapper = _build_wrapper(process, controls, reaction_module)

    triangle_width = float(controls.control_metadata["bolus_feed"]["triangle_width"])
    t_in_ramp = 2.0 + 0.5 * triangle_width

    # Evaluate inside the synthetic bolus ramp.
    y_physical = jnp.asarray([1.0, 0.0, 1.0], dtype=jnp.float32)
    y_scaled = wrapper.scale_state(y_physical)
    dy_scaled = wrapper(t_in_ramp, y_scaled)
    dy_physical = dy_scaled * wrapper.state_scale

    # biomass present in feed medium -> non-zero bolus transport contribution.
    assert float(dy_physical[0]) != pytest.approx(0.0, abs=1e-9)
    # product absent in feed medium and C_product==0 -> zero bolus contribution.
    assert float(dy_physical[1]) == pytest.approx(0.0, abs=1e-9)
    # dV_cont/dt receives bolus rate during the ramp.
    assert float(dy_physical[2]) > 0.0


def test_wrapper_multi_bolus_final_v_cont_invariant():
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
    bolus_deltas = jnp.asarray([0.2, 0.15, 0.3], dtype=jnp.float32)
    process = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=8.0, time_reference="start"),
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
                        times=jnp.asarray([1.0, 3.0, 5.0], dtype=jnp.float32),
                        values=bolus_deltas,
                    ),
                    feed_medium=bolus_medium,
                ),
                "sample_dummy": SampleVolumeChange(
                    name="sample_dummy",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray(
                            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                        ),
                        values=jnp.asarray(
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        ),
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
                        times=jnp.asarray([0.0, 8.0], dtype=jnp.float32),
                        values=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
                    ),
                    is_intracellular=False,
                )
            },
        ),
        process_variables={},
    )
    controls = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": process}, metadata={})
    ).get_controls("p1")
    wrapper = _build_wrapper(
        process,
        controls,
        ConstantReactionModule(
            specific_rates=jnp.asarray([0.0], dtype=jnp.float32),
            modeled_feed_rates=jnp.zeros((0,), dtype=jnp.float32),
        ),
    )

    y0_scaled = wrapper.scale_state(jnp.asarray([1.0, 1.0], dtype=jnp.float32))
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: wrapper(t, y)),
        solver=diffrax.Tsit5(),
        t0=0.0,
        t1=8.0,
        dt0=None,
        y0=y0_scaled,
        saveat=diffrax.SaveAt(ts=jnp.asarray([8.0], dtype=jnp.float32)),
        stepsize_controller=diffrax.PIDController(
            rtol=1e-6,
            atol=1e-8,
            jump_ts=controls.active_step_ts,
        ),
        max_steps=4096,
        throw=False,
    )
    assert solution.result == diffrax.RESULTS.successful
    final_state = wrapper.unscale_state(solution.ys[0])
    final_v_cont = float(final_state[-1])
    expected_final_v = 1.0 + float(jnp.sum(bolus_deltas))
    assert final_v_cont == pytest.approx(expected_final_v, abs=2e-3)
