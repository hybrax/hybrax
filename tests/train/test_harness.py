from __future__ import annotations

import logging

import equinox as eqx
import jax
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

from bp_train.harness import (
    TrainHarnessConfig,
    _build_batch_index_stream,
    _ensure_process_names,
    _validate_batching_config,
    train_collection,
)
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


def _make_feed_mismatch_collection() -> BioProcessCollection:
    def _make_process(name: str, feed_x_concentration: float) -> BioProcess:
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
        return BioProcess(
            metadata=BioProcessMetadata(name=name, process_type="fed_batch"),
            time_axis=TimeAxis(
                unit="h",
                start=0.0,
                end=2.0,
                time_reference="start",
            ),
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
            reactor_medium=ReactorMedium(name="rm", density=1.0, density_unit="kg/L"),
            process_variables={
                "X": ProcessVariable(
                    name="X",
                    unit="g/L",
                    is_controlled=False,
                    values=TimeSeries(
                        times=jnp.asarray([0.0, 2.0]),
                        values=jnp.asarray([1.0, 1.0]),
                    ),
                )
            },
        )

    return BioProcessCollection(
        processes={
            "p1": _make_process("p1", 0.0),
            "p2": _make_process("p2", 1.0),
        },
        metadata={},
    )


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
            batch_size=2,
            optimizer_name="adam",
            learning_rate=5e-2,
            log_every=1,
        ),
    )

    assert len(result.mean_loss_by_step) == 8
    assert result.mean_loss_by_step[-1] < result.mean_loss_by_step[0]
    assert result.compile_warmup_seconds > 0.0
    assert len(result.step_time_seconds) == 8
    assert all(dt > 0.0 for dt in result.step_time_seconds)
    assert len(result.batch_process_names_by_step) == 8
    assert all(names == ("p1", "p1") for names in result.batch_process_names_by_step)
    assert set(result.sampled_loss_by_process_at_log_steps.keys()) == set(range(1, 9))
    assert all(
        len(sampled) == 2 and all(name == "p1" for name, _loss in sampled)
        for sampled in result.sampled_loss_by_process_at_log_steps.values()
    )
    assert result.train_step_rebuild_count == 0
    assert len(result.train_step_input_signature) > 0


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
            optimizer_name="sgd",
            learning_rate=2e-2,
            log_every=1,
        ),
    )

    assert len(result.mean_loss_by_step) == 5
    assert all(jnp.isfinite(jnp.asarray(result.mean_loss_by_step)))
    assert len(result.batch_process_names_by_step) == 5
    assert result.compile_warmup_seconds > 0.0
    assert len(result.step_time_seconds) == 5
    assert set(result.sampled_loss_by_process_at_log_steps.keys()) == {1, 2, 3, 4, 5}
    for step, sampled in result.sampled_loss_by_process_at_log_steps.items():
        assert step >= 1
        assert len(sampled) == 2
        assert all(name in {"p1", "p2"} for name, _loss in sampled)
    assert result.train_step_rebuild_count == 0


def test_train_collection_rejects_unknown_process_selection():
    store = TrainingDataStore.from_collection(
        _make_collection(), target_variable_order=["X"]
    )
    with pytest.raises(ValueError, match="unknown process names"):
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


def test_train_collection_rejects_unsupported_optimizer_name():
    store = TrainingDataStore.from_collection(
        _make_collection(),
        target_variable_order=["X"],
    )
    with pytest.raises(ValueError, match="optimizer_name"):
        train_collection(
            store,
            reaction_module=_LinearReactionModule(),
            config=TrainHarnessConfig(
                process_names=("p1",),
                steps=2,
                optimizer_name="rmsprop",
            ),
        )


def test_train_collection_rejects_mismatched_feed_semantics_across_selected_processes():
    store = TrainingDataStore.from_collection(
        _make_feed_mismatch_collection(),
        target_variable_order=["X"],
    )
    with pytest.raises(ValueError, match="incompatible wrapper feed semantics"):
        train_collection(
            store,
            reaction_module=_LinearReactionModule(),
            config=TrainHarnessConfig(
                process_names=("p1", "p2"),
                steps=2,
                batch_size=2,
                optimizer_name="adam",
                learning_rate=2e-2,
                log_every=1,
            ),
        )


def test_harness_process_name_validation_rejects_duplicates_and_empty():
    store = TrainingDataStore.from_collection(
        _make_collection(),
        target_variable_order=["X"],
    )
    with pytest.raises(ValueError, match="duplicate entries in process_names"):
        _ensure_process_names(store, ("p1", "p1"))
    with pytest.raises(ValueError, match="non-empty"):
        _ensure_process_names(store, ())


def test_harness_phase1_batching_validation_checks_basics():
    cfg = TrainHarnessConfig(
        steps=3,
        batch_size=None,
        optimizer_name="adam",
        learning_rate=1e-3,
    )
    assert _validate_batching_config(cfg, selected_process_count=2) == 2

    with pytest.raises(ValueError, match="steps"):
        _validate_batching_config(
            TrainHarnessConfig(steps=0),
            selected_process_count=2,
        )
    with pytest.raises(ValueError, match="learning_rate"):
        _validate_batching_config(
            TrainHarnessConfig(steps=1, learning_rate=0.0),
            selected_process_count=2,
        )
    with pytest.raises(ValueError, match="log_every"):
        _validate_batching_config(
            TrainHarnessConfig(steps=1, log_every=0),
            selected_process_count=2,
        )
    with pytest.raises(ValueError, match="optimizer_name"):
        _validate_batching_config(
            TrainHarnessConfig(steps=1, learning_rate=1e-3, optimizer_name="rmsprop"),
            selected_process_count=2,
        )
    with pytest.raises(ValueError, match="effective batch_size"):
        _validate_batching_config(
            TrainHarnessConfig(steps=1, batch_size=0),
            selected_process_count=2,
        )
    with pytest.raises(ValueError, match="effective batch_size"):
        _validate_batching_config(
            TrainHarnessConfig(steps=1, batch_size=-1),
            selected_process_count=2,
        )


def test_harness_batch_stream_round_robin_without_shuffle():
    stream = _build_batch_index_stream(
        selected_process_indices=jnp.asarray([5, 7], dtype=jnp.int32),
        steps=3,
        batch_size=3,
        shuffle_batches=False,
        batch_seed=None,
        seed=123,
    )
    assert tuple(stream.shape) == (3, 3)
    assert stream.tolist() == [[5, 7, 5], [7, 5, 7], [5, 7, 5]]


def test_harness_batch_stream_shuffle_is_deterministic_and_seeded():
    kwargs = dict(
        selected_process_indices=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        steps=4,
        batch_size=2,
        shuffle_batches=True,
    )
    s1 = _build_batch_index_stream(batch_seed=999, seed=123, **kwargs)
    s2 = _build_batch_index_stream(batch_seed=999, seed=777, **kwargs)
    s3 = _build_batch_index_stream(batch_seed=1000, seed=123, **kwargs)
    assert s1.tolist() == s2.tolist()
    assert s1.tolist() != s3.tolist()


def test_harness_batch_stream_falls_back_to_seed_when_batch_seed_none():
    kwargs = dict(
        selected_process_indices=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        steps=3,
        batch_size=3,
        shuffle_batches=True,
        batch_seed=None,
    )
    s1 = _build_batch_index_stream(seed=42, **kwargs)
    s2 = _build_batch_index_stream(seed=42, **kwargs)
    s3 = _build_batch_index_stream(seed=43, **kwargs)
    assert s1.tolist() == s2.tolist()
    assert s1.tolist() != s3.tolist()


def test_harness_config_has_no_drop_last_batch_field():
    assert "drop_last_batch" not in TrainHarnessConfig.__dataclass_fields__


def test_train_collection_signature_is_stable_and_no_rebuilds():
    store = TrainingDataStore.from_collection(
        _make_collection(), target_variable_order=["X"]
    )
    result = train_collection(
        store,
        reaction_module=_LinearReactionModule(),
        config=TrainHarnessConfig(
            process_names=("p1", "p2"),
            steps=4,
            batch_size=2,
            optimizer_name="adam",
            learning_rate=2e-2,
            log_every=2,
        ),
    )

    assert len(result.train_step_input_signature) > 0
    assert result.train_step_rebuild_count == 0
    assert set(result.sampled_loss_by_process_at_log_steps.keys()) == {2, 4}


def test_train_collection_logs_sampled_losses_only_at_log_steps(caplog):
    store = TrainingDataStore.from_collection(
        _make_collection(), target_variable_order=["X"]
    )
    caplog.set_level(logging.INFO, logger="bp_train.harness")
    train_collection(
        store,
        reaction_module=_LinearReactionModule(),
        config=TrainHarnessConfig(
            process_names=("p1", "p2"),
            steps=4,
            batch_size=2,
            optimizer_name="adam",
            learning_rate=2e-2,
            log_every=2,
        ),
    )

    step_logs = [
        record.message
        for record in caplog.records
        if record.message.startswith("step ")
    ]
    assert len(step_logs) == 2
    assert step_logs[0].startswith("step 2/4 ")
    assert step_logs[1].startswith("step 4/4 ")
    assert all("sampled=" in msg for msg in step_logs)
    assert all("mean_loss=" not in msg for msg in step_logs)
