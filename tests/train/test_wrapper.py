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
    ProcessVariable,
    ReactorMedium,
    SampleVolumeChange,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)

from bp_train.controls_store import ControlsStore
from bp_train.model_api import ReactionOutputs, UserReactionModule
from bp_train.wrapper import LibraryRhsWrapper, ModeledFeedSpec


class ConstantReactionModule(UserReactionModule):
    """Test reaction module returning fixed `ReactionOutputs`."""

    reaction_terms: jnp.ndarray
    modeled_feed_rates: jnp.ndarray

    def __init__(self, reaction_terms: jnp.ndarray, modeled_feed_rates: jnp.ndarray):
        self.reaction_terms = reaction_terms
        self.modeled_feed_rates = modeled_feed_rates

    def __call__(self, t, c_species, controls_vector):
        del t, c_species, controls_vector
        return ReactionOutputs(
            reaction_terms=self.reaction_terms,
            modeled_feed_rates=self.modeled_feed_rates,
        )


class InvalidReactionShapeModule(UserReactionModule):
    """Test reaction module returning malformed output ranks."""

    def __call__(self, t, c_species, controls_vector):
        del t, c_species, controls_vector
        return ReactionOutputs(
            reaction_terms=jnp.asarray([[0.1]], dtype=jnp.float32),
            modeled_feed_rates=jnp.zeros((0,), dtype=jnp.float32),
        )


class _NonEqxReactionModule:
    def __call__(self, t, c_species, controls_vector):
        del t, c_species, controls_vector
        return ReactionOutputs(
            reaction_terms=jnp.asarray([0.0], dtype=jnp.float32),
            modeled_feed_rates=jnp.zeros((0,), dtype=jnp.float32),
        )


def _make_single_process_collection(
    *,
    feed_rate: float = 0.2,
    feed_x_concentration: float = 0.0,
) -> BioProcessCollection:
    feed_medium = FeedMedium(
        name="feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "X": FeedMediumComponent(
                name="X",
                unit="g/L",
                concentration=StaticVariable(feed_x_concentration),
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
                        timepoints=jnp.asarray([0.0, 2.0]),
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
                        timepoints=jnp.asarray([1.0]),
                        values=jnp.asarray([-0.1]),
                    ),
                ),
            },
        ),
        reactor_medium=ReactorMedium(name="rm", density=1.0, density_unit="kg/L"),
        process_variables={
            "X": ProcessVariable(
                name="X",
                unit="g/L",
                is_controlled=False,
                values=TimeSeries(
                    timepoints=jnp.asarray([0.0, 2.0]),
                    values=jnp.asarray([1.0, 1.0]),
                ),
            )
        },
    )
    return BioProcessCollection(processes={"p1": process}, metadata={})


def _make_multi_feed_two_species_collection() -> BioProcessCollection:
    feed_a = FeedMedium(
        name="feed_a",
        density=1.0,
        density_unit="kg/L",
        components={
            "X": FeedMediumComponent(
                name="X",
                unit="g/L",
                concentration=StaticVariable(10.0),
                is_controlled=False,
            ),
        },
    )
    feed_b = FeedMedium(
        name="feed_b",
        density=1.0,
        density_unit="kg/L",
        components={
            "P": FeedMediumComponent(
                name="P",
                unit="g/L",
                concentration=StaticVariable(5.0),
                is_controlled=False,
            ),
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
                        timepoints=jnp.asarray([0.0, 2.0]),
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
                        timepoints=jnp.asarray([0.0, 2.0]),
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
                        timepoints=jnp.asarray([1.0]),
                        values=jnp.asarray([-0.1]),
                    ),
                ),
            },
        ),
        reactor_medium=ReactorMedium(name="rm", density=1.0, density_unit="kg/L"),
        process_variables={
            "X": ProcessVariable(
                name="X",
                unit="g/L",
                is_controlled=False,
                values=TimeSeries(
                    timepoints=jnp.asarray([0.0, 2.0]),
                    values=jnp.asarray([1.0, 1.0]),
                ),
            ),
            "P": ProcessVariable(
                name="P",
                unit="g/L",
                is_controlled=False,
                values=TimeSeries(
                    timepoints=jnp.asarray([0.0, 2.0]),
                    values=jnp.asarray([2.0, 2.0]),
                ),
            ),
        },
    )
    return BioProcessCollection(processes={"p1": process}, metadata={})


def test_wrapper_reconstructs_real_volume_and_merges_reaction_with_transport():
    collection = _make_single_process_collection(
        feed_rate=0.2,
        feed_x_concentration=0.0,
    )
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    reaction_module = ConstantReactionModule(
        reaction_terms=jnp.asarray([0.3], dtype=jnp.float32),
        modeled_feed_rates=jnp.zeros((0,), dtype=jnp.float32),
    )
    wrapper = LibraryRhsWrapper.from_process_controls(
        reaction_module=reaction_module,
        controls=controls,
        species_names=["X"],
    )

    y = jnp.asarray([1.0, 1.2], dtype=jnp.float32)
    dy = wrapper(2.0, y)

    # V_real = V_cont - V_sample_acc = 1.2 - 0.1 = 1.1
    # transport = 0.2 * (0.0 - 1.0) / 1.1 = -0.181818...
    # dX = reaction + transport = 0.3 - 0.181818...
    assert dy.shape == (2,)
    assert dy[0] == pytest.approx(0.1181818, rel=1e-5)
    assert dy[1] == pytest.approx(0.2, rel=1e-6)


def test_wrapper_merges_controlled_and_modeled_feed_transport():
    collection = _make_single_process_collection(
        feed_rate=0.1,
        feed_x_concentration=0.0,
    )
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    reaction_module = ConstantReactionModule(
        reaction_terms=jnp.asarray([0.0], dtype=jnp.float32),
        modeled_feed_rates=jnp.asarray([0.3], dtype=jnp.float32),
    )
    wrapper = LibraryRhsWrapper.from_process_controls(
        reaction_module=reaction_module,
        controls=controls,
        species_names=["X"],
        modeled_feeds=[
            ModeledFeedSpec(name="modeled_base", component_concentrations={"X": 2.0})
        ],
    )

    y = jnp.asarray([1.0, 1.0], dtype=jnp.float32)
    dy = wrapper(0.0, y)

    # controlled: 0.1 * (0 - 1) / 1 = -0.1
    # modeled:    0.3 * (2 - 1) / 1 = +0.3
    assert dy[0] == pytest.approx(0.2, rel=1e-6)
    assert dy[1] == pytest.approx(0.4, rel=1e-6)


def test_wrapper_multiple_controlled_feeds_sum_and_apply_composition_per_species():
    collection = _make_multi_feed_two_species_collection()
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    reaction_module = ConstantReactionModule(
        reaction_terms=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        modeled_feed_rates=jnp.zeros((0,), dtype=jnp.float32),
    )
    wrapper = LibraryRhsWrapper.from_process_controls(
        reaction_module=reaction_module,
        controls=controls,
        species_names=["X", "P"],
    )

    y = jnp.asarray([1.0, 2.0, 1.2], dtype=jnp.float32)
    dy = wrapper(2.0, y)

    # At t=2.0, sample_acc=0.1 => V_real = V_cont - V_sample_acc = 1.2 - 0.1 = 1.1
    # X numerator: 0.2 * (10 - 1) + 0.3 * (0 - 1) = 1.5 -> dX = 1.5 / 1.1
    # P numerator: 0.2 * (0 - 2) + 0.3 * (5 - 2) = 0.5 -> dP = 0.5 / 1.1
    # dV_cont: 0.2 + 0.3 = 0.5
    assert dy.shape == (3,)
    assert dy[0] == pytest.approx(1.3636364, rel=1e-6)
    assert dy[1] == pytest.approx(0.45454547, rel=1e-6)
    assert dy[2] == pytest.approx(0.5, rel=1e-6)


def test_wrapper_rejects_modeled_rate_shape_mismatch():
    collection = _make_single_process_collection()
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    reaction_module = ConstantReactionModule(
        reaction_terms=jnp.asarray([0.0], dtype=jnp.float32),
        modeled_feed_rates=jnp.asarray([0.2, 0.3], dtype=jnp.float32),
    )
    wrapper = LibraryRhsWrapper.from_process_controls(
        reaction_module=reaction_module,
        controls=controls,
        species_names=["X"],
        modeled_feeds=[
            ModeledFeedSpec(name="modeled_1", component_concentrations={"X": 0.0})
        ],
    )

    with pytest.raises(ValueError, match="modeled_feed_rates must match"):
        wrapper(0.0, jnp.asarray([1.0, 1.0], dtype=jnp.float32))


def test_wrapper_rejects_invalid_reaction_output_rank():
    collection = _make_single_process_collection()
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    wrapper = LibraryRhsWrapper.from_process_controls(
        reaction_module=InvalidReactionShapeModule(),
        controls=controls,
        species_names=["X"],
    )

    with pytest.raises(ValueError, match="reaction_terms must be a rank-1 vector"):
        wrapper(0.0, jnp.asarray([1.0, 1.0], dtype=jnp.float32))


def test_wrapper_rejects_invalid_state_vector_shape():
    collection = _make_single_process_collection()
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    wrapper = LibraryRhsWrapper.from_process_controls(
        reaction_module=ConstantReactionModule(
            reaction_terms=jnp.asarray([0.0], dtype=jnp.float32),
            modeled_feed_rates=jnp.zeros((0,), dtype=jnp.float32),
        ),
        controls=controls,
        species_names=["X"],
    )

    with pytest.raises(ValueError, match="state vector y must have shape"):
        wrapper(0.0, jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32))


def test_wrapper_rejects_duplicate_modeled_feed_names():
    collection = _make_single_process_collection()
    controls = ControlsStore.from_collection(collection).get_controls("p1")
    reaction_module = ConstantReactionModule(
        reaction_terms=jnp.asarray([0.0], dtype=jnp.float32),
        modeled_feed_rates=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
    )

    with pytest.raises(ValueError, match="modeled feed names must be unique"):
        LibraryRhsWrapper.from_process_controls(
            reaction_module=reaction_module,
            controls=controls,
            species_names=["X"],
            modeled_feeds=[
                ModeledFeedSpec(name="dup", component_concentrations={"X": 0.0}),
                ModeledFeedSpec(name="dup", component_concentrations={"X": 1.0}),
            ],
        )


def test_wrapper_rejects_non_eqx_reaction_module():
    collection = _make_single_process_collection()
    controls = ControlsStore.from_collection(collection).get_controls("p1")

    with pytest.raises(TypeError, match="must be an `eqx.Module` instance"):
        LibraryRhsWrapper.from_process_controls(
            reaction_module=_NonEqxReactionModule(),
            controls=controls,
            species_names=["X"],
        )
