from __future__ import annotations

import logging
import warnings

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

from bp_train.harness import (
    TrainHarnessConfig,
    _build_batch_index_stream,
    _ensure_process_names,
    _validate_batching_config,
    train_from_collection,
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
        rate = self.model(c_species)[0] + self.non_model_bias[0]
        return ReactionOutputs(
            specific_rates=jnp.asarray([rate], dtype=c_species.dtype),
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
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray([0.0, 1.0, 2.0]),
                        values=jnp.asarray([1.0, 0.8, 0.64]),
                    ),
                    is_intracellular=False,
                ),
            },
        ),
        process_variables={},
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
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray([0.0, 1.0, 2.0]),
                        values=jnp.asarray([0.9, 0.72, 0.58]),
                    ),
                    is_intracellular=False,
                ),
            },
        ),
        process_variables={},
    )
    return BioProcessCollection(processes={"p1": p1, "p2": p2}, metadata={})


def _make_feed_mismatch_collection() -> BioProcessCollection:
    """Two processes with different feed compositions (Cin values differ)."""

    def _make_process(name: str, feed_biomass_concentration: float) -> BioProcess:
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

    return BioProcessCollection(
        processes={
            "p1": _make_process("p1", 0.0),
            "p2": _make_process("p2", 1.0),
        },
        metadata={},
    )


def test_train_collection_single_process_loss_decreases():
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    result = train_collection(
        store,
        reaction_module=_LinearReactionModule(),
        collection=collection,
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
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    result = train_collection(
        store,
        reaction_module=_LinearReactionModule(),
        collection=collection,
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


def test_train_collection_with_different_cin_per_process():
    """Processes with different feed compositions should train without error."""
    collection = _make_feed_mismatch_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    result = train_collection(
        store,
        reaction_module=_LinearReactionModule(),
        collection=collection,
        config=TrainHarnessConfig(
            process_names=("p1", "p2"),
            steps=4,
            batch_size=2,
            optimizer_name="adam",
            learning_rate=2e-2,
            log_every=2,
        ),
    )

    assert len(result.mean_loss_by_step) == 4
    assert all(jnp.isfinite(jnp.asarray(result.mean_loss_by_step)))


def test_train_collection_rejects_unknown_process_selection():
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    with pytest.raises(ValueError, match="unknown process names"):
        train_collection(
            store,
            reaction_module=_LinearReactionModule(),
            collection=collection,
            config=TrainHarnessConfig(process_names=("unknown",), steps=2),
        )


def test_train_collection_rejects_nonpositive_solver_max_steps():
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    with pytest.raises(ValueError, match="solver_max_steps must be positive"):
        train_collection(
            store,
            reaction_module=_LinearReactionModule(),
            collection=collection,
            config=TrainHarnessConfig(
                process_names=("p1",),
                steps=2,
                solver_max_steps=0,
            ),
        )


def test_train_collection_rejects_nonpositive_solver_tolerances():
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    with pytest.raises(ValueError, match="solver_rtol must be positive"):
        train_collection(
            store,
            reaction_module=_LinearReactionModule(),
            collection=collection,
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
            collection=collection,
            config=TrainHarnessConfig(
                process_names=("p1",),
                steps=2,
                solver_atol=0.0,
            ),
        )


def test_train_collection_rejects_unsupported_optimizer_name():
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    with pytest.raises(ValueError, match="optimizer_name"):
        train_collection(
            store,
            reaction_module=_LinearReactionModule(),
            collection=collection,
            config=TrainHarnessConfig(
                process_names=("p1",),
                steps=2,
                optimizer_name="rmsprop",
            ),
        )


def test_harness_process_name_validation_rejects_duplicates_and_empty():
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
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
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    result = train_collection(
        store,
        reaction_module=_LinearReactionModule(),
        collection=collection,
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
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    caplog.set_level(logging.INFO, logger="bp_train.harness")
    train_collection(
        store,
        reaction_module=_LinearReactionModule(),
        collection=collection,
        config=TrainHarnessConfig(
            process_names=("p1", "p2"),
            steps=4,
            batch_size=2,
            optimizer_name="adam",
            learning_rate=2e-2,
            log_every=2,
        ),
    )

    # The new tabular layout emits one row per step in the form
    # " HH:MM:SS | <step> | <loss> | <tgt> | <dt> ", surrounded by an
    # initial header (column-name row + separator row) that re-prints every
    # `header_every` steps. Filter to data rows by matching the leading
    # whitespace + clock pattern.
    import re

    row_re = re.compile(r"^\s\d{2}:\d{2}:\d{2}\s\|")
    step_rows = [
        record.message for record in caplog.records if row_re.match(record.message)
    ]
    assert len(step_rows) == 4
    # Per-process indented line is emitted only at log_every cadence.
    sampled_logs = [
        record.message for record in caplog.records if "per-process:" in record.message
    ]
    assert len(sampled_logs) == 2
    # Sampled-loss history dict still tracks the same log-step keys.
    # (Verified above by the previous test; here we only re-check the count.)


def test_train_from_collection_warns_and_logs_when_targets_default(monkeypatch, caplog):
    collection = _make_collection()

    class _DummyStore:
        target_names = ("biomass",)
        target_source = "reactor_components"
        process_order = ("p1", "p2")

    def fake_from_collection(collection, *, target_variable_order, target_source):
        del collection, target_variable_order, target_source
        return _DummyStore()

    monkeypatch.setattr(
        "bp_train.harness.TrainingDataStore.from_collection",
        fake_from_collection,
    )
    monkeypatch.setattr("bp_train.harness.load_custom_module", lambda _p: object())
    monkeypatch.setattr("bp_train.harness.resolve_config", lambda _m, _r: {})
    monkeypatch.setattr(
        "bp_train.harness._ensure_process_names", lambda _s, _n: ("p1",)
    )
    monkeypatch.setattr(
        "bp_train.harness._build_reaction_module", lambda **_kw: object()
    )
    monkeypatch.setattr(
        "bp_train.harness.train_collection",
        lambda *args, **kwargs: "train-result",
    )

    caplog.set_level(logging.INFO, logger="bp_train.harness")
    with pytest.warns(UserWarning, match="No target_variable_order specified"):
        result = train_from_collection(
            collection,
            config=TrainHarnessConfig(target_variable_order=None, steps=1),
            custom_py=None,
            runtime_config=None,
        )

    assert result == "train-result"
    assert "Training targets: ('biomass',)" in caplog.text


def test_train_from_collection_uses_custom_config_targets_without_warning(
    monkeypatch, caplog
):
    collection = _make_collection()
    captured: dict[str, object] = {}

    class _DummyStore:
        target_names = ("cfg_biomass",)
        target_source = "reactor_components"
        process_order = ("p1", "p2")

    def fake_from_collection(collection, *, target_variable_order, target_source):
        del collection, target_source
        captured["target_variable_order"] = target_variable_order
        return _DummyStore()

    monkeypatch.setattr(
        "bp_train.harness.TrainingDataStore.from_collection",
        fake_from_collection,
    )
    monkeypatch.setattr("bp_train.harness.load_custom_module", lambda _p: object())
    monkeypatch.setattr(
        "bp_train.harness.resolve_config",
        lambda _m, _r: {"target_variable_order": ["cfg_biomass"]},
    )
    monkeypatch.setattr(
        "bp_train.harness._ensure_process_names", lambda _s, _n: ("p1",)
    )
    monkeypatch.setattr(
        "bp_train.harness._build_reaction_module", lambda **_kw: object()
    )
    monkeypatch.setattr(
        "bp_train.harness.train_collection",
        lambda *args, **kwargs: "train-result",
    )

    caplog.set_level(logging.INFO, logger="bp_train.harness")
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("always")
        result = train_from_collection(
            collection,
            config=TrainHarnessConfig(target_variable_order=None, steps=1),
            custom_py="custom.py",
            runtime_config=None,
        )

    assert result == "train-result"
    assert len(warns) == 0
    assert captured["target_variable_order"] == ("cfg_biomass",)
    assert "Training targets: ('cfg_biomass',)" in caplog.text
