from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from bpbench.dataclasses import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    ProcessVariable,
    ReactorMedium,
    SampleVolumeChange,
    TimeAxis,
    TimeSeries,
    Volume,
)

from bp_train.harness import TrainHarnessConfig, train_collection
from bp_train.model_api import ReactionOutputs, UserReactionModule
from bp_train.training_data import TrainingDataStore


class _LinearReactionModule(UserReactionModule):
    model: eqx.nn.Linear
    non_model_bias: jax.Array

    def __init__(self):
        self.model = eqx.nn.Linear(1, 1, key=jax.random.key(42))
        self.non_model_bias = jnp.asarray([0.05], dtype=jnp.float32)

    def __call__(self, t, c_species, controls_vector):
        del t, controls_vector
        reaction = self.model(c_species)[0] + self.non_model_bias[0]
        return ReactionOutputs(
            reaction_terms=jnp.asarray([reaction], dtype=c_species.dtype),
            modeled_feed_rates=jnp.zeros((0,), dtype=c_species.dtype),
        )


def _make_collection() -> BioProcessCollection:
    p1 = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "sample_1": SampleVolumeChange(
                    name="sample_1",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([1.0]),
                        values=jnp.asarray([-0.1]),
                    ),
                )
            },
        ),
        reactor_medium=ReactorMedium(name="rm", density=1.0, density_unit="kg/L"),
        process_variables={
            "X": ProcessVariable(
                name="X",
                unit="g/L",
                is_controlled=False,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 1.0, 2.0]),
                    values=jnp.asarray([1.0, 0.8, 0.64]),
                ),
            )
        },
    )
    p2 = BioProcess(
        metadata=BioProcessMetadata(name="p2", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.1,
            unit="L",
            volume_changes={
                "sample_1": SampleVolumeChange(
                    name="sample_1",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([1.0]),
                        values=jnp.asarray([-0.1]),
                    ),
                )
            },
        ),
        reactor_medium=ReactorMedium(name="rm", density=1.0, density_unit="kg/L"),
        process_variables={
            "X": ProcessVariable(
                name="X",
                unit="g/L",
                is_controlled=False,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 1.0, 2.0]),
                    values=jnp.asarray([0.9, 0.72, 0.58]),
                ),
            )
        },
    )
    return BioProcessCollection(processes={"p1": p1, "p2": p2}, metadata={})


def test_train_collection_single_process_loss_decreases():
    store = TrainingDataStore.from_collection(
        _make_collection(), target_variable_order=["X"]
    )
    result = train_collection(
        store,
        reaction_module=_LinearReactionModule(),
        config=TrainHarnessConfig(
            process_names=("p1",),
            steps=8,
            learning_rate=5e-2,
            log_every=1,
        ),
    )

    assert result.process_names == ("p1",)
    assert len(result.mean_loss_by_step) == 8
    assert result.mean_loss_by_step[-1] < result.mean_loss_by_step[0]
    assert result.compile_count_by_process["p1"] == 1
    assert result.compile_time_seconds_by_process["p1"] > 0.0
    assert len(result.step_time_seconds_by_process["p1"]) == 8
    assert result.total_compile_count == 1
    assert result.total_compile_seconds >= result.compile_time_seconds_by_process["p1"]
    assert result.total_step_seconds > 0.0
    assert result.suspicious_step_spikes_by_process["p1"] >= 0


def test_train_collection_multi_process_tracks_per_process_histories():
    store = TrainingDataStore.from_collection(
        _make_collection(), target_variable_order=["X"]
    )
    result = train_collection(
        store,
        reaction_module=_LinearReactionModule(),
        config=TrainHarnessConfig(
            process_names=("p1", "p2"),
            steps=5,
            learning_rate=2e-2,
            log_every=1,
        ),
    )

    assert result.process_names == ("p1", "p2")
    assert len(result.mean_loss_by_step) == 5
    assert set(result.loss_by_process.keys()) == {"p1", "p2"}
    assert len(result.loss_by_process["p1"]) == 5
    assert len(result.loss_by_process["p2"]) == 5
    assert result.compile_count_by_process == {"p1": 1, "p2": 1}
    assert result.compile_time_seconds_by_process["p1"] > 0.0
    assert result.compile_time_seconds_by_process["p2"] > 0.0
    assert all(jnp.isfinite(jnp.asarray(result.mean_loss_by_step)))
    assert result.total_compile_count == 2
    assert result.total_compile_seconds > 0.0
    assert result.total_step_seconds > 0.0


def test_train_collection_rejects_unknown_process_selection():
    store = TrainingDataStore.from_collection(
        _make_collection(), target_variable_order=["X"]
    )
    with pytest.raises(ValueError, match="requested process names not found"):
        train_collection(
            store,
            reaction_module=_LinearReactionModule(),
            config=TrainHarnessConfig(process_names=("unknown",), steps=2),
        )


def test_train_collection_rejects_nonpositive_solver_max_steps():
    store = TrainingDataStore.from_collection(
        _make_collection(),
        target_variable_order=["X"],
    )
    with pytest.raises(ValueError, match="solver_max_steps must be positive"):
        train_collection(
            store,
            reaction_module=_LinearReactionModule(),
            config=TrainHarnessConfig(
                process_names=("p1",),
                steps=2,
                solver_max_steps=0,
            ),
        )


def test_train_collection_rejects_nonpositive_solver_tolerances():
    store = TrainingDataStore.from_collection(
        _make_collection(),
        target_variable_order=["X"],
    )
    with pytest.raises(ValueError, match="solver_rtol must be positive"):
        train_collection(
            store,
            reaction_module=_LinearReactionModule(),
            config=TrainHarnessConfig(
                process_names=("p1",),
                steps=2,
                solver_rtol=0.0,
            ),
        )
    with pytest.raises(ValueError, match="solver_atol must be positive"):
        train_collection(
            store,
            reaction_module=_LinearReactionModule(),
            config=TrainHarnessConfig(
                process_names=("p1",),
                steps=2,
                solver_atol=0.0,
            ),
        )
