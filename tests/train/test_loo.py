"""Tests for the config-driven Leave-one/some-process-out CV orchestrator."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import pandas as pd
import pytest
from hybrax.format.dataclasses import (
    AugmentedBioProcess,
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    ReactorMedium,
    ReactorMediumComponent,
    Outflow,
    TimeAxis,
    TimeSeries,
    Volume,
)

from hybrax.train import cli
from hybrax.train.harness import PreparedTraining, TrainHarnessResult
from hybrax.train import loo as loo_mod
import hybrax.train.serialization as serialization
from hybrax.train.loo import (
    Fold,
    FoldResult,
    LOOResult,
    _build_fold_groups,
    _read_final_train_loss,
    compute_parallel_split,
    resolve_folds,
    run_loo_cv,
    run_single_fold,
    _write_summary_and_aggregate,
)
from hybrax.train.run_config import (
    DataConfig,
    HoldoutSet,
    LooConfig,
    RunConfig,
    TrainConfig,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_process(name: str, biomass_values=(1.0, 0.8, 0.64)) -> BioProcess:
    return BioProcess(
        metadata=BioProcessMetadata(name=name, process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "sample_1": Outflow(
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
                        values=jnp.asarray(biomass_values),
                    ),
                ),
            },
        ),
        process_variables={},
    )


def _make_augmented(name: str, parent: str) -> AugmentedBioProcess:
    base = _make_process(name)
    return AugmentedBioProcess(
        metadata=base.metadata,
        time_axis=base.time_axis,
        volume=base.volume,
        reactor_medium=base.reactor_medium,
        process_variables=base.process_variables,
        parent_process=parent,
    )


def _three_parent_collection() -> BioProcessCollection:
    return BioProcessCollection(
        processes={
            "p1": _make_process("p1"),
            "p2": _make_process("p2", biomass_values=(0.9, 0.72, 0.58)),
            "p3": _make_process("p3", biomass_values=(1.1, 0.88, 0.70)),
        },
        metadata={},
    )


def _augmented_collection() -> BioProcessCollection:
    return BioProcessCollection(
        processes={
            "P0": _make_process("P0"),
            "P0_aug": _make_augmented("P0_aug", "P0"),
            "P1": _make_process("P1", biomass_values=(0.5, 0.4, 0.32)),
        },
        metadata={},
    )


def _run_config(seed: int = 10) -> RunConfig:
    return RunConfig(
        data=DataConfig(prepared=Path("prepared.json")),
        train=TrainConfig(epochs=2, seed=seed),
    )


# ---------------------------------------------------------------------------
# _build_fold_groups (unchanged behaviour, drives the auto-LOO fallback)
# ---------------------------------------------------------------------------


def test_build_fold_groups_returns_one_group_per_parent():
    groups = _build_fold_groups(_three_parent_collection())
    assert groups == (("p1", ("p1",)), ("p2", ("p2",)), ("p3", ("p3",)))


def test_build_fold_groups_attaches_augmented_children_to_parent():
    groups = _build_fold_groups(_augmented_collection())
    assert groups == (("P0", ("P0", "P0_aug")), ("P1", ("P1",)))


def test_build_fold_groups_rejects_orphan_augmented_child():
    collection = BioProcessCollection(
        processes={
            "P0": _make_process("P0"),
            "ghost_aug": _make_augmented("ghost_aug", "ghost"),
        },
        metadata={},
    )
    with pytest.raises(ValueError, match="references parent_process 'ghost'"):
        _build_fold_groups(collection)


# ---------------------------------------------------------------------------
# resolve_folds — auto-LOO fallback
# ---------------------------------------------------------------------------


def test_resolve_folds_auto_one_per_parent():
    folds = resolve_folds(_three_parent_collection(), None, 10)
    assert [f.slug for f in folds] == ["p1", "p2", "p3"]
    assert folds[0].test == ("p1",)
    assert folds[0].train == ("p2", "p3")
    assert [fold.seed for fold in folds] == [10, 11, 12]
    assert all(set(f.test).isdisjoint(f.train) for f in folds)


def test_resolve_folds_auto_groups_augmented_with_parent():
    folds = resolve_folds(_augmented_collection(), None, 10)
    assert [f.slug for f in folds] == ["P0", "P1"]
    # Holding out P0 also holds out its augmented child; train is only P1.
    assert folds[0].test == ("P0", "P0_aug")
    assert folds[0].train == ("P1",)
    # Holding out P1 keeps P0 + its augmented child together in train.
    assert folds[1].test == ("P1",)
    assert set(folds[1].train) == {"P0", "P0_aug"}


def test_resolve_folds_auto_requires_two_parents():
    collection = BioProcessCollection(processes={"only": _make_process("only")})
    with pytest.raises(ValueError, match="requires >= 2 parent processes"):
        resolve_folds(collection, None, 10)


# ---------------------------------------------------------------------------
# resolve_folds — explicit per_fold_holdout_sets
# ---------------------------------------------------------------------------


def test_resolve_folds_explicit_default_train():
    loo_cfg = LooConfig(
        per_fold_holdout_sets=(HoldoutSet(test=("p1",)), HoldoutSet(test=("p2", "p3")))
    )
    folds = resolve_folds(_three_parent_collection(), loo_cfg, 10)
    assert folds[0].test == ("p1",)
    assert folds[0].train == ("p2", "p3")
    assert folds[1].test == ("p2", "p3")
    assert folds[1].train == ("p1",)
    assert folds[1].slug == "p2+p3"


def test_resolve_folds_explicit_pinned_train():
    loo_cfg = LooConfig(
        per_fold_holdout_sets=(HoldoutSet(test=("p2", "p3"), train=("p1",)),)
    )
    folds = resolve_folds(_three_parent_collection(), loo_cfg, 10)
    assert folds[0].train == ("p1",)


def test_resolve_folds_explicit_default_train_excludes_augmented_child():
    # test=[P0]; default train = everything not in test, but P0_aug must drop out
    # because its parent P0 is held out (no leak).
    loo_cfg = LooConfig(per_fold_holdout_sets=(HoldoutSet(test=("P0",)),))
    folds = resolve_folds(_augmented_collection(), loo_cfg, 10)
    assert folds[0].train == ("P1",)


def test_resolve_folds_explicit_train_leak_raises():
    # Pinning P0_aug into train while its parent P0 is held out is a leak.
    loo_cfg = LooConfig(
        per_fold_holdout_sets=(HoldoutSet(test=("P0",), train=("P1", "P0_aug")),)
    )
    with pytest.raises(ValueError, match="leaks augmentation-group"):
        resolve_folds(_augmented_collection(), loo_cfg, 10)


def _two_child_collection() -> BioProcessCollection:
    return BioProcessCollection(
        processes={
            "P0": _make_process("P0"),
            "C1": _make_augmented("C1", "P0"),
            "C2": _make_augmented("C2", "P0"),
            "P1": _make_process("P1", biomass_values=(0.5, 0.4, 0.32)),
        },
        metadata={},
    )


def test_resolve_folds_child_held_out_alone_excludes_whole_group():
    # Holding out only the augmented child must exclude its parent AND siblings
    # from train (the held-out child is a synthetic variant of P0).
    loo_cfg = LooConfig(per_fold_holdout_sets=(HoldoutSet(test=("C1",)),))
    folds = resolve_folds(_two_child_collection(), loo_cfg, 10)
    assert folds[0].train == ("P1",)  # P0 and C2 excluded, not just C1


def test_resolve_folds_explicit_train_parent_of_held_out_child_raises():
    # test=child, train pins the parent -> parent leaks the held-out variant.
    loo_cfg = LooConfig(
        per_fold_holdout_sets=(HoldoutSet(test=("C1",), train=("P0", "P1")),)
    )
    with pytest.raises(ValueError, match="leaks augmentation-group"):
        resolve_folds(_two_child_collection(), loo_cfg, 10)


def test_resolve_folds_named_fold_slug():
    loo_cfg = LooConfig(
        per_fold_holdout_sets=(
            HoldoutSet(name="control, high S", test=("p1", "p2")),
            HoldoutSet(name="1003 47µLS@5h", test=("p3",)),
        )
    )
    folds = resolve_folds(_three_parent_collection(), loo_cfg, 10)
    assert folds[0].slug == "control_high_S"
    assert folds[1].slug == "1003_47µLS_5h"


def test_resolve_folds_duplicate_slug_raises():
    loo_cfg = LooConfig(
        per_fold_holdout_sets=(
            HoldoutSet(name="dup", test=("p1",)),
            HoldoutSet(name="dup", test=("p2",)),
        )
    )
    with pytest.raises(ValueError, match="same output directory slug"):
        resolve_folds(_three_parent_collection(), loo_cfg, 10)


def test_resolve_folds_explicit_unknown_test_raises():
    loo_cfg = LooConfig(per_fold_holdout_sets=(HoldoutSet(test=("ghost",)),))
    with pytest.raises(ValueError, match="unknown process name"):
        resolve_folds(_three_parent_collection(), loo_cfg, 10)


def test_resolve_folds_explicit_overlap_raises():
    loo_cfg = LooConfig(
        per_fold_holdout_sets=(HoldoutSet(test=("p1",), train=("p1", "p2")),)
    )
    with pytest.raises(ValueError, match="both test and train"):
        resolve_folds(_three_parent_collection(), loo_cfg, 10)


# ---------------------------------------------------------------------------
# resolve_folds — data_processes restriction
# ---------------------------------------------------------------------------


def test_resolve_folds_classic_respects_data_processes():
    folds = resolve_folds(
        _three_parent_collection(), None, 0, data_processes=("p1", "p2")
    )
    assert [f.slug for f in folds] == ["p1", "p2"]
    assert all("p3" not in (*f.test, *f.train) for f in folds)


def test_resolve_folds_classic_data_processes_default_train_excludes_restricted():
    folds = resolve_folds(
        _three_parent_collection(), None, 0, data_processes=("p1", "p2")
    )
    assert folds[0].train == ("p2",)


def test_resolve_folds_data_processes_child_without_parent_raises():
    with pytest.raises(ValueError, match="excludes its parent process 'P0'"):
        resolve_folds(_augmented_collection(), None, 0, data_processes=("P0_aug", "P1"))


def test_resolve_folds_data_processes_parent_without_child_is_allowed():
    folds = resolve_folds(_augmented_collection(), None, 0, data_processes=("P0", "P1"))
    assert [f.slug for f in folds] == ["P0", "P1"]
    assert folds[0].test == ("P0",)
    assert folds[0].train == ("P1",)


def test_resolve_folds_data_processes_unknown_name_raises():
    with pytest.raises(ValueError, match="data.processes.*unknown process name"):
        resolve_folds(
            _three_parent_collection(), None, 0, data_processes=("p1", "ghost")
        )


def test_resolve_folds_per_fold_test_outside_data_processes_raises():
    loo_cfg = LooConfig(per_fold_holdout_sets=(HoldoutSet(test=("p3",)),))
    with pytest.raises(ValueError, match="excluded by data.processes"):
        resolve_folds(
            _three_parent_collection(), loo_cfg, 0, data_processes=("p1", "p2")
        )


def test_resolve_folds_per_fold_train_outside_data_processes_raises():
    loo_cfg = LooConfig(
        per_fold_holdout_sets=(HoldoutSet(test=("p1",), train=("p3",)),)
    )
    with pytest.raises(ValueError, match="excluded by data.processes"):
        resolve_folds(
            _three_parent_collection(), loo_cfg, 0, data_processes=("p1", "p2")
        )


def test_resolve_folds_per_fold_default_train_restricted_by_data_processes():
    loo_cfg = LooConfig(per_fold_holdout_sets=(HoldoutSet(test=("p1",)),))
    folds = resolve_folds(
        _three_parent_collection(), loo_cfg, 0, data_processes=("p1", "p2")
    )
    assert folds[0].train == ("p2",)


def test_resolve_folds_data_processes_none_matches_unrestricted():
    collection = _three_parent_collection()
    assert resolve_folds(collection, None, 0, data_processes=None) == resolve_folds(
        collection, None, 0
    )


# ---------------------------------------------------------------------------
# compute_parallel_split
# ---------------------------------------------------------------------------


def test_split_user_picks_parallel_device_budget():
    # 4 folds at once on 16 CPUs -> 4 JAX devices each; product <= CPUs.
    parallel, devices = compute_parallel_split(9, 16, 4)
    assert (parallel, devices) == (4, 4)
    assert parallel * devices <= 16


def test_split_sequential_uses_all_device_budget():
    parallel, devices = compute_parallel_split(12, 16, 1)
    assert (parallel, devices) == (1, 16)


def test_split_clamped_to_fold_and_cpu_count():
    # asking for 20-wide with only 9 folds / 8 cpus -> 8 parallel.
    parallel, devices = compute_parallel_split(9, 8, 20)
    assert parallel == 8 and devices == 1


def test_split_devices_capped_by_batch():
    # 2 folds, 16 cores -> 8 devices each, but batch caps at 3.
    parallel, devices = compute_parallel_split(2, 16, 2, max_devices_per_fold=3)
    assert (parallel, devices) == (2, 3)


def test_split_uses_configured_devices_per_fold():
    parallel, devices = compute_parallel_split(9, 16, 4, devices_per_fold=2)
    assert (parallel, devices) == (4, 2)


def test_split_configured_devices_clamps_parallel_to_cpu_count():
    parallel, devices = compute_parallel_split(9, 16, 9, devices_per_fold=4)
    assert (parallel, devices) == (4, 4)


def test_split_never_below_one():
    parallel, devices = compute_parallel_split(0, 1, 1)
    assert parallel == 1 and devices == 1


# ---------------------------------------------------------------------------
# _write_summary_and_aggregate (reads fold losses back from disk)
# ---------------------------------------------------------------------------


def _write_stub_fold(output_dir: Path, fold: Fold, *, target: str = "biomass") -> None:
    fold_dir = output_dir / "folds" / fold.slug
    fold_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"process": n, "total": 0.5, target: 0.5, "split": "holdout"} for n in fold.test
    ] + [
        {"process": n, "total": 0.1, target: 0.1, "split": "train"} for n in fold.train
    ]
    pd.DataFrame(rows).to_csv(fold_dir / "losses.csv", index=False)
    (fold_dir / "trained_wrapper.meta.json").write_text(
        json.dumps({"training": {"final_mean_loss": 0.2}})
    )


def test_summary_and_aggregate_from_disk(tmp_path):
    folds = resolve_folds(_three_parent_collection(), None, 10)
    for fold in folds:
        _write_stub_fold(tmp_path, fold)

    aggregate = _write_summary_and_aggregate(
        folds=folds,
        output_dir=tmp_path,
        summary_csv_path=tmp_path / "loo_summary.csv",
        aggregate_json_path=tmp_path / "loo_aggregate.json",
        base_seed=10,
    )

    df = pd.read_csv(tmp_path / "loo_summary.csv")
    assert len(df) == len(folds) + 1  # one row per fold + mean row
    assert set(df.columns) >= {
        "fold_idx",
        "fold_slug",
        "test",
        "fold_seed",
        "holdout_total",
        "holdout_biomass",
        "train_mean_total",
        "final_train_loss",
    }
    # Each fold's holdout loss is the stub 0.5; train mean is 0.1.
    per_fold = df[df["fold_slug"] != "mean"]
    assert (per_fold["holdout_total"] == 0.5).all()
    assert (per_fold["train_mean_total"] == 0.1).all()
    assert df.iloc[-1]["fold_slug"] == "mean"

    assert aggregate["n_folds"] == 3
    assert aggregate["base_seed"] == 10
    assert aggregate["holdout_total_mean"] == pytest.approx(0.5)


def test_null_final_train_loss_remains_missing_not_finite(tmp_path):
    fold_dir = tmp_path / "fold"
    fold_dir.mkdir()
    (fold_dir / "trained_wrapper.meta.json").write_text(
        '{"training": {"final_mean_loss": null}}', encoding="utf-8"
    )

    assert math.isnan(_read_final_train_loss(fold_dir))


# ---------------------------------------------------------------------------
# run_single_fold (worker) with mocked training/forward
# ---------------------------------------------------------------------------


def _stub_train_result() -> TrainHarnessResult:
    return TrainHarnessResult(
        trained_wrapper=object(),
        mean_loss_by_step=(1.0, 0.5),
        batch_process_names_by_step=(),
        per_process_loss_by_step=(),
        compile_warmup_seconds=0.0,
        step_time_seconds=(),
        train_step_input_signature=(),
        train_step_rebuild_count=0,
    )


def _patch_worker_internals(monkeypatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    reloaded_wrapper = object()

    def fake_prepare(coll, *, config, custom_module, run_config):
        captured["process_names"] = config.process_names
        captured["seed"] = config.seed
        captured["holdout"] = config.holdout_processes
        captured["run_config"] = run_config
        return PreparedTraining(
            store=object(),
            reaction_module=object(),
            loss_module=SimpleNamespace(loss_names=("X",)),
            config=config,
            optimizer=object(),
            prediction_parent_process_names=("p1", "p2", "p3"),
        )

    def fake_evaluate(
        wrapper,
        store,
        *,
        config,
        training_process_names,
        prediction_process_names,
        **_kw,
    ):
        captured["evaluation_wrapper"] = wrapper
        captured["training_process_names"] = training_process_names
        captured["eval_process_names"] = config.process_names
        captured["prediction_process_names"] = prediction_process_names
        return SimpleNamespace()

    def fake_write(*, output_dir, **_kw):
        captured["fold_dir"] = Path(output_dir)

    monkeypatch.setattr("hybrax.train.loo.prepare_training", fake_prepare)
    monkeypatch.setattr(
        "hybrax.train.loo.train_collection", lambda *_a, **_k: _stub_train_result()
    )
    monkeypatch.setattr("hybrax.train.loo.evaluate_trained_wrapper", fake_evaluate)
    monkeypatch.setattr("hybrax.train.cli._write_train_results", fake_write)
    monkeypatch.setattr(loo_mod, "save_model", lambda *_a, **_k: None)
    monkeypatch.setattr(
        loo_mod,
        "load_trained_wrapper",
        lambda _path, *, template: (
            captured.update(reload_template=template) or reloaded_wrapper
        ),
    )
    monkeypatch.setattr(loo_mod, "save_model_metadata", lambda *_a, **_k: None)
    captured["reloaded_wrapper"] = reloaded_wrapper
    return captured


def test_run_single_fold_trains_excluding_holdout(monkeypatch, tmp_path):
    collection = _three_parent_collection()
    captured = _patch_worker_internals(monkeypatch)
    cfg = _run_config(seed=10)
    cfg = cfg.model_copy(
        update={"output": cfg.output.model_copy(update={"predictions": "parents"})}
    )
    custom_py = tmp_path / "shared-custom.py"
    custom_py.write_text("VALUE = 1\n")

    result = run_single_fold(
        collection,
        cfg=cfg,
        custom_module=None,
        output_dir=tmp_path,
        fold_idx=1,
        custom_py=custom_py,
    )

    assert isinstance(result, FoldResult)
    assert result.fold.test == ("p2",)
    assert captured["process_names"] == ("p1", "p3")
    assert captured["holdout"] == ("p2",)
    assert captured["prediction_process_names"] == ("p1", "p3", "p2")
    assert captured["evaluation_wrapper"] is captured["reloaded_wrapper"]
    assert result.train_result.trained_wrapper is captured["reloaded_wrapper"]
    assert captured["reload_template"] is not result.train_result.trained_wrapper
    assert result.fold_seed == 10 + 1  # base seed + fold idx
    assert result.fold_dir == tmp_path / "folds" / "p2"
    effective = captured["run_config"]
    assert effective.data.processes == ("p1", "p3")
    assert effective.train.seed == 11
    assert effective.output.dir == result.fold_dir.resolve()
    document = json.loads((result.fold_dir / "config.json").read_text())
    assert document["status"] == "complete"
    # Every fold config pins the input its model trained on: the shared
    # reconstruction path refuses to rebuild a model without it.
    assert document["inputs"]["prepared_input"] == {
        "path": str(cfg.data.prepared),
        "content_hash": serialization.content_hash(collection),
    }
    assert document["updates_completed"] == 0
    assert document["final_mean_loss"] == 0.5
    bundled = document["config"]
    assert bundled["data"]["processes"] == ["p1", "p3"]
    assert bundled["train"]["seed"] == 11
    assert bundled["custom_py"] == str(result.fold_dir / "custom.py")
    assert (result.fold_dir / "custom.py").read_text() == "VALUE = 1\n"


def test_holdout_is_test_set_only_not_all_nontrain(monkeypatch, tmp_path):
    # 4 processes; explicit fold pins train=[p1], test=[p2]; p3 and p4 are in
    # NEITHER. The holdout must be EXACTLY the test set, and the fold must only
    # evaluate train ∪ test (p3/p4 are excluded, not silently labelled holdout).
    collection = BioProcessCollection(
        processes={
            "p1": _make_process("p1"),
            "p2": _make_process("p2"),
            "p3": _make_process("p3"),
            "p4": _make_process("p4"),
        },
        metadata={},
    )
    captured = _patch_worker_internals(monkeypatch)
    cfg = RunConfig(
        data=DataConfig(prepared=Path("prepared.json")),
        train=TrainConfig(epochs=2, seed=0),
        loo=LooConfig(
            per_fold_holdout_sets=(HoldoutSet(test=("p2",), train=("p1",)),),
        ),
    )

    result = run_single_fold(
        collection, cfg=cfg, custom_module=None, output_dir=tmp_path, fold_idx=0
    )

    assert result.fold.train == ("p1",)
    assert result.fold.test == ("p2",)
    assert captured["training_process_names"] == ("p1",)
    assert captured["holdout"] == ("p2",)  # holdout loss = TEST set only
    assert set(captured["eval_process_names"]) == {"p1", "p2"}  # p3/p4 excluded


def test_run_single_fold_out_of_range(monkeypatch, tmp_path):
    collection = _three_parent_collection()
    _patch_worker_internals(monkeypatch)
    with pytest.raises(ValueError, match="out of range"):
        run_single_fold(
            collection,
            cfg=_run_config(),
            custom_module=None,
            output_dir=tmp_path,
            fold_idx=9,
        )


def test_run_single_fold_respects_data_processes_restriction(monkeypatch, tmp_path):
    collection = _three_parent_collection()
    captured = _patch_worker_internals(monkeypatch)
    cfg = RunConfig(
        data=DataConfig(prepared=Path("prepared.json"), processes=("p1", "p2")),
        train=TrainConfig(epochs=2, seed=10),
    )

    result = run_single_fold(
        collection, cfg=cfg, custom_module=None, output_dir=tmp_path, fold_idx=1
    )

    assert result.fold.test == ("p2",)
    assert captured["process_names"] == ("p1",)
    assert "p3" not in captured["process_names"]

    with pytest.raises(ValueError, match="out of range"):
        run_single_fold(
            collection, cfg=cfg, custom_module=None, output_dir=tmp_path, fold_idx=2
        )


def test_prepare_single_fold_preserves_prediction_parent_names(monkeypatch, tmp_path):
    def fake_prepare(*_args, config, **_kwargs):
        return PreparedTraining(
            store=object(),
            reaction_module=object(),
            loss_module=SimpleNamespace(loss_names=("X",)),
            config=config,
            optimizer=object(),
            prediction_parent_process_names=("p1", "p2", "p3"),
        )

    monkeypatch.setattr(loo_mod, "prepare_training", fake_prepare)
    prepared = loo_mod.prepare_single_fold(
        _three_parent_collection(),
        cfg=_run_config(),
        custom_module=None,
        output_dir=tmp_path,
        fold_idx=1,
    )

    assert prepared.training.prediction_parent_process_names == ("p1", "p2", "p3")


# ---------------------------------------------------------------------------
# Artifact-backed orchestration and internal CLI modes
# ---------------------------------------------------------------------------


def test_produce_runtime_artifact_respects_data_processes(monkeypatch, tmp_path):
    collection = _three_parent_collection()
    store = SimpleNamespace(rhs_ode=object())

    selected_scale_processes = []
    validation_calls = []

    class _ProducerData:
        process_order = ("p1", "p2", "p3")
        augmentation_parents = (None, None, None)

        def select_training_parents(self, _collection, process_names):
            selected_scale_processes.append(tuple(process_names))
            validation_calls.append(("scale", tuple(process_names)))
            return self

    producer_data = _ProducerData()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "hybrax.format.serialization.load_process_collection", lambda _path: collection
    )
    monkeypatch.setattr(
        loo_mod,
        "ensure_prepared_training_semantics",
        lambda candidate: validation_calls.append(("semantics", candidate)),
    )

    def validate_parents(candidate):
        validation_calls.append(("parents", candidate))
        return True, ()

    monkeypatch.setattr(loo_mod, "validate_augmented_parent_refs", validate_parents)

    def validate_training(candidate, **kwargs):
        validation_calls.append(("training", candidate, kwargs))

    monkeypatch.setattr(loo_mod, "validate_for_training", validate_training)
    monkeypatch.setattr(
        loo_mod.TrainingDataStore,
        "from_collection",
        lambda *_args, **_kwargs: store,
    )
    monkeypatch.setattr(
        loo_mod.ProducerCollectionData,
        "from_collection",
        lambda *_args: producer_data,
    )
    monkeypatch.setattr(
        loo_mod, "_resolve_estimated_scales", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        loo_mod.RhsNames, "from_rhs_ode", classmethod(lambda _cls, _rhs: object())
    )
    monkeypatch.setattr(loo_mod, "content_hash", lambda _collection: "sha256:data")

    def fake_write(path, **kwargs):
        captured.update(path=path, **kwargs)
        return "sha256:artifact"

    monkeypatch.setattr(loo_mod, "write_runtime_artifact", fake_write)
    bundle = tmp_path / "loo-config.json"
    bundle.write_text("{}", encoding="utf-8")
    cfg = _run_config(seed=37)
    cfg = cfg.model_copy(
        update={
            "data": cfg.data.model_copy(update={"processes": ("p1", "p2")}),
            "output": cfg.output.model_copy(update={"predictions": "none"}),
        }
    )

    identity = loo_mod.produce_runtime_artifact(
        cfg=cfg,
        custom_module=None,
        output_dir=tmp_path,
        bundle_path=bundle,
    )

    assert identity == "sha256:artifact"
    assert validation_calls[0] == ("parents", collection)
    assert validation_calls[1][0] == "semantics"
    assert validation_calls[2] == (
        "training",
        validation_calls[1][1],
        {"strict": True, "require_biological_ode": True},
    )
    assert [call[0] for call in validation_calls] == [
        "parents",
        "semantics",
        "training",
        "scale",
        "scale",
    ]
    validated_parent_collection = validation_calls[1][1]
    assert validated_parent_collection is captured["parent_collection"]
    assert validated_parent_collection is not collection
    assert tuple(validated_parent_collection.processes) == ("p1", "p2", "p3")
    folds = tuple(record for record, _scales in captured["folds"])
    assert [(fold.test, fold.train, fold.seed) for fold in folds] == [
        (("p1",), ("p2",), 37),
        (("p2",), ("p1",), 38),
    ]
    assert selected_scale_processes == [("p2",), ("p1",)]
    assert tuple(captured["parent_collection"].processes) == ("p1", "p2", "p3")

    cfg = cfg.model_copy(
        update={"loo": LooConfig(per_fold_holdout_sets=(HoldoutSet(test=("p3",)),))}
    )
    with pytest.raises(ValueError, match="excluded by data.processes"):
        loo_mod.produce_runtime_artifact(
            cfg=cfg,
            custom_module=None,
            output_dir=tmp_path,
            bundle_path=bundle,
        )


def test_produce_runtime_artifact_validates_augmented_parent_refs(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "hybrax.format.serialization.load_process_collection",
        lambda _path: _three_parent_collection(),
    )
    monkeypatch.setattr(
        loo_mod,
        "validate_augmented_parent_refs",
        lambda _collection: (False, ("bad parent",)),
    )
    bundle = tmp_path / "loo-config.json"
    bundle.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="bad parent"):
        loo_mod.produce_runtime_artifact(
            cfg=_run_config(),
            custom_module=None,
            output_dir=tmp_path,
            bundle_path=bundle,
        )


def _runtime_metadata(collection, seed: int = 10):
    folds = resolve_folds(collection, None, seed)
    records = tuple(loo_mod._fold_record(fold) for fold in folds)
    return loo_mod.RuntimeArtifactMetadata(
        identity="sha256:artifact",
        identity_inputs={
            "run_fingerprint": "sha256:fingerprint",
            "prepared_content_hash": "sha256:prepared",
        },
        folds=records,
    )


def _patch_runtime_metadata(monkeypatch, tmp_path, metadata):
    artifact_path = tmp_path / "runtime-artifact"
    monkeypatch.setattr(
        loo_mod,
        "_runtime_metadata",
        lambda *_args, **_kwargs: (artifact_path, metadata),
    )
    return artifact_path


def test_runtime_metadata_rejects_replacement_artifact(monkeypatch, tmp_path):
    bundle = tmp_path / "loo-config.json"
    bundle.write_text("{}", encoding="utf-8")
    metadata = loo_mod.RuntimeArtifactMetadata(
        identity="sha256:replacement",
        identity_inputs={
            "run_fingerprint": loo_mod._fingerprint(bundle, None),
            "prepared_content_hash": "sha256:prepared",
        },
        folds=_runtime_metadata(_three_parent_collection()).folds,
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "runtime_artifact": {
                    "format_version": loo_mod.FORMAT_VERSION,
                    "identity": "sha256:original",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        loo_mod, "read_runtime_artifact_metadata", lambda _path: metadata
    )

    with pytest.raises(ValueError, match="does not match run config anchor"):
        loo_mod._runtime_metadata(tmp_path, bundle_path=bundle, custom_path=None)


@pytest.mark.parametrize(
    "anchor",
    [
        {"format_version": True, "identity": "sha256:artifact"},
        {"format_version": loo_mod.FORMAT_VERSION, "identity": 1},
        {
            "format_version": loo_mod.FORMAT_VERSION,
            "identity": "sha256:artifact",
            "extra": 1,
        },
    ],
)
def test_runtime_metadata_requires_exact_typed_anchor(monkeypatch, tmp_path, anchor):
    bundle = tmp_path / "loo-config.json"
    bundle.write_text("{}", encoding="utf-8")
    (tmp_path / "config.json").write_text(
        json.dumps({"runtime_artifact": anchor}), encoding="utf-8"
    )
    monkeypatch.setattr(
        loo_mod,
        "read_runtime_artifact_metadata",
        lambda _path: pytest.fail("manifest read for invalid anchor"),
    )

    with pytest.raises(ValueError, match="invalid runtime artifact anchor"):
        loo_mod._runtime_metadata(tmp_path, bundle_path=bundle, custom_path=None)


def test_runtime_metadata_rejects_config_fingerprint_mismatch(monkeypatch, tmp_path):
    bundle = tmp_path / "loo-config.json"
    bundle.write_text("{}", encoding="utf-8")
    metadata = loo_mod.RuntimeArtifactMetadata(
        identity="sha256:artifact",
        identity_inputs={
            "run_fingerprint": "sha256:wrong",
            "prepared_content_hash": "sha256:prepared",
        },
        folds=_runtime_metadata(_three_parent_collection()).folds,
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "runtime_artifact": {
                    "format_version": loo_mod.FORMAT_VERSION,
                    "identity": metadata.identity,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        loo_mod, "read_runtime_artifact_metadata", lambda _path: metadata
    )

    with pytest.raises(ValueError, match="fingerprint does not match"):
        loo_mod._runtime_metadata(tmp_path, bundle_path=bundle, custom_path=None)


@pytest.mark.parametrize(
    "runtime",
    [
        None,
        {
            "artifact_format_version": loo_mod.FORMAT_VERSION,
            "artifact_identity": "sha256:artifact",
        },
        {
            "artifact_format_version": loo_mod.FORMAT_VERSION,
            "artifact_identity": "sha256:artifact",
            "fold_id": 0,
            "extra": True,
        },
        {
            "artifact_format_version": True,
            "artifact_identity": "sha256:artifact",
            "fold_id": 0,
        },
        {
            "artifact_format_version": loo_mod.FORMAT_VERSION,
            "artifact_identity": "sha256:artifact",
            "fold_id": False,
        },
        {
            "artifact_format_version": loo_mod.FORMAT_VERSION + 1,
            "artifact_identity": "sha256:artifact",
            "fold_id": 0,
        },
        {
            "artifact_format_version": loo_mod.FORMAT_VERSION,
            "artifact_identity": "sha256:other",
            "fold_id": 0,
        },
        {
            "artifact_format_version": loo_mod.FORMAT_VERSION,
            "artifact_identity": "sha256:artifact",
            "fold_id": 1,
        },
    ],
)
def test_fold_complete_rejects_malformed_binding_without_deleting(tmp_path, runtime):
    metadata = _runtime_metadata(_three_parent_collection())
    record = metadata.folds[0]
    fold_dir = tmp_path / record.slug
    fold_dir.mkdir()
    (fold_dir / "config.json").write_text(
        json.dumps({"status": "complete", "loo_runtime": runtime}),
        encoding="utf-8",
    )
    (fold_dir / "losses.csv").write_text("loss", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match manifest"):
        loo_mod._fold_complete(fold_dir, metadata.identity, record)

    assert fold_dir.is_dir()
    assert (fold_dir / "config.json").is_file()


@pytest.mark.parametrize(
    ("status", "runtime", "losses"),
    [
        ("complete", None, True),
        ("running", "valid", True),
        ("complete", "valid", False),
    ],
)
def test_fold_complete_reruns_interrupted_records(tmp_path, status, runtime, losses):
    metadata = _runtime_metadata(_three_parent_collection())
    record = metadata.folds[0]
    fold_dir = tmp_path / record.slug
    fold_dir.mkdir()
    document = {"status": status}
    if runtime == "valid":
        document.update(loo_mod._fold_runtime_metadata(metadata.identity, record.idx))
    (fold_dir / "config.json").write_text(json.dumps(document), encoding="utf-8")
    if losses:
        (fold_dir / "losses.csv").write_text("loss", encoding="utf-8")

    assert not loo_mod._fold_complete(fold_dir, metadata.identity, record)


def test_run_loo_cv_resume_validates_all_folds_before_deleting(monkeypatch, tmp_path):
    metadata = _runtime_metadata(_three_parent_collection())
    _patch_runtime_metadata(monkeypatch, tmp_path, metadata)
    first, second = metadata.folds[:2]
    first_dir = tmp_path / "folds" / first.slug
    first_dir.mkdir(parents=True)
    sentinel = first_dir / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    second_dir = tmp_path / "folds" / second.slug
    second_dir.mkdir()
    (second_dir / "config.json").write_text(
        json.dumps(
            {
                "status": "complete",
                **loo_mod._fold_runtime_metadata("sha256:other", second.idx),
            }
        ),
        encoding="utf-8",
    )
    (second_dir / "losses.csv").write_text("loss", encoding="utf-8")
    monkeypatch.setattr(
        loo_mod,
        "_dispatch_pool",
        lambda *_args: pytest.fail("workers dispatched after invalid completion"),
    )
    cfg = RunConfig(
        data=DataConfig(prepared=Path("prepared.json")),
        train=TrainConfig(epochs=2, seed=10),
    )

    with pytest.raises(ValueError, match="does not match manifest"):
        run_loo_cv(
            cfg=cfg,
            config_path=tmp_path / "loo-config.json",
            output_dir=tmp_path,
            resume=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def _patch_dispatch(monkeypatch) -> dict[str, Any]:
    """Replace the subprocess pool with a stub-fold writer (no real training)."""
    seen: dict[str, Any] = {}

    def fake_pool(config_path, output_dir, artifact_path, folds, parallel, devices):
        seen["pool_folds"] = [f.slug for f in folds]
        seen["artifact_path"] = Path(artifact_path)
        seen["parallel"] = parallel
        seen["devices"] = devices
        for fold in folds:
            _write_stub_fold(Path(output_dir), fold)

    monkeypatch.setattr(loo_mod, "_dispatch_pool", fake_pool)
    return seen


def test_run_loo_cv_dispatches_only_manifest_folds(monkeypatch, tmp_path):
    metadata = _runtime_metadata(_three_parent_collection())
    artifact_path = _patch_runtime_metadata(monkeypatch, tmp_path, metadata)
    seen = _patch_dispatch(monkeypatch)
    cfg = RunConfig(
        data=DataConfig(prepared=Path("prepared.json")),
        train=TrainConfig(epochs=2, seed=10),
        loo=LooConfig(parallel_folds=2),
    )

    result = run_loo_cv(
        cfg=cfg, config_path=tmp_path / "loo-config.json", output_dir=tmp_path
    )

    assert isinstance(result, LOOResult)
    assert seen["pool_folds"] == ["p1", "p2", "p3"]
    assert seen["artifact_path"] == artifact_path
    assert seen["parallel"] == 2
    assert result.parallel_folds == 2
    assert result.aggregate["n_folds"] == 3


def test_run_loo_cv_uses_configured_devices_per_fold(monkeypatch, tmp_path):
    monkeypatch.setattr(loo_mod.os, "cpu_count", lambda: 16)
    metadata = _runtime_metadata(_three_parent_collection())
    _patch_runtime_metadata(monkeypatch, tmp_path, metadata)
    seen = _patch_dispatch(monkeypatch)
    cfg = RunConfig(
        data=DataConfig(prepared=Path("prepared.json")),
        train=TrainConfig(epochs=2, seed=10),
        loo=LooConfig(parallel_folds=2, devices_per_fold=2),
    )

    result = run_loo_cv(
        cfg=cfg,
        config_path=tmp_path / "loo-config.json",
        output_dir=tmp_path,
    )

    assert seen["parallel"] == 2
    assert seen["devices"] == 2
    assert result.devices_per_fold == 2


def test_run_loo_cv_resume_requires_identity_bound_completion(monkeypatch, tmp_path):
    metadata = _runtime_metadata(_three_parent_collection())
    _patch_runtime_metadata(monkeypatch, tmp_path, metadata)
    seen = _patch_dispatch(monkeypatch)
    first = loo_mod._fold_from_record(metadata.folds[0])
    _write_stub_fold(tmp_path, first)
    # losses.csv alone is incomplete: it must be cleared and rerun.
    cfg = RunConfig(
        data=DataConfig(prepared=Path("prepared.json")),
        train=TrainConfig(epochs=2, seed=10),
        loo=LooConfig(parallel_folds=1),
    )

    run_loo_cv(
        cfg=cfg,
        config_path=tmp_path / "loo-config.json",
        output_dir=tmp_path,
        resume=True,
    )

    assert seen["pool_folds"] == ["p1", "p2", "p3"]


def test_run_loo_cv_resume_skips_matching_completed_fold(monkeypatch, tmp_path):
    metadata = _runtime_metadata(_three_parent_collection())
    _patch_runtime_metadata(monkeypatch, tmp_path, metadata)
    seen = _patch_dispatch(monkeypatch)
    fold = loo_mod._fold_from_record(metadata.folds[0])
    _write_stub_fold(tmp_path, fold)
    (tmp_path / "folds" / fold.slug / "config.json").write_text(
        json.dumps(
            {
                "status": "complete",
                **loo_mod._fold_runtime_metadata(metadata.identity, fold.idx),
            }
        )
    )
    cfg = RunConfig(
        data=DataConfig(prepared=Path("prepared.json")),
        train=TrainConfig(epochs=2, seed=10),
        loo=LooConfig(parallel_folds=1),
    )

    run_loo_cv(
        cfg=cfg,
        config_path=tmp_path / "loo-config.json",
        output_dir=tmp_path,
        resume=True,
    )

    assert seen["pool_folds"] == ["p2", "p3"]


def test_manifest_seed_reaches_effective_config_and_provenance(monkeypatch, tmp_path):
    metadata = _runtime_metadata(_three_parent_collection())
    record = loo_mod.RuntimeArtifactFold(
        idx=metadata.folds[0].idx,
        test=metadata.folds[0].test,
        train=metadata.folds[0].train,
        slug=metadata.folds[0].slug,
        seed=901,
    )
    metadata = loo_mod.RuntimeArtifactMetadata(
        metadata.identity,
        metadata.identity_inputs,
        (record,),
    )
    artifact_path = _patch_runtime_metadata(monkeypatch, tmp_path, metadata)
    monkeypatch.setattr(
        loo_mod,
        "load_runtime_artifact",
        lambda *_args, **_kwargs: SimpleNamespace(
            identity=metadata.identity,
            fold=record,
            training_data=object(),
            scales=object(),
            training_parent_collection=object(),
            augmentation_parents=(None, None, None),
        ),
    )
    captured = {}

    def fake_prepare(_artifact, *, config, **_kwargs):
        captured["harness_seed"] = config.seed
        return PreparedTraining(
            store=object(),
            reaction_module=object(),
            loss_module=SimpleNamespace(loss_names=("X",)),
            config=config,
            optimizer=object(),
            prediction_parent_process_names=("p1", "p2", "p3"),
        )

    monkeypatch.setattr(loo_mod, "prepare_training_from_runtime_artifact", fake_prepare)
    prepared = loo_mod.prepare_single_fold_from_runtime_artifact(
        cfg=_run_config(seed=10),
        custom_module=None,
        output_dir=tmp_path,
        bundle_path=tmp_path / "loo-config.json",
        artifact_path=artifact_path,
        fold_idx=record.idx,
    )

    assert prepared.fold.seed == 901
    assert prepared.fold_seed == 901
    assert prepared.effective_cfg.train.seed == 901
    assert captured["harness_seed"] == 901
    fold_document = json.loads(prepared.config_json.read_text())
    assert fold_document["config"]["train"]["seed"] == 901
    # The fold inherits the producer-validated prepared hash, so the fold model is
    # loadable on the shared reconstruction path.
    recorded_hash = fold_document["inputs"]["prepared_input"]["content_hash"]
    assert recorded_hash == metadata.identity_inputs["prepared_content_hash"]

    monkeypatch.setattr(
        loo_mod, "train_collection", lambda *_args, **_kwargs: _stub_train_result()
    )
    monkeypatch.setattr(loo_mod, "save_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        loo_mod, "load_trained_wrapper", lambda _path, *, template: template
    )
    monkeypatch.setattr(
        loo_mod,
        "save_model_metadata",
        lambda _path, value: captured.update(model_metadata=value),
    )
    trained = loo_mod.train_prepared_fold(prepared)

    assert trained.fold_seed == 901
    assert captured["model_metadata"]["fold_seed"] == 901
    assert captured["model_metadata"]["training"]["seed"] == 901

    _write_stub_fold(tmp_path, prepared.fold)
    loo_mod._write_summary_and_aggregate(
        folds=(prepared.fold,),
        output_dir=tmp_path,
        summary_csv_path=tmp_path / "loo_summary.csv",
        aggregate_json_path=tmp_path / "loo_aggregate.json",
        base_seed=10,
    )
    summary = pd.read_csv(tmp_path / "loo_summary.csv")
    assert summary.iloc[0]["fold_seed"] == 901


def test_loo_survives_cross_fold_loss_plot_failure(monkeypatch, tmp_path, caplog):
    fold = Fold(idx=0, test=("p1",), train=("p2",), slug="p1", seed=1)
    fold_dir = tmp_path / "folds" / fold.slug
    fold_dir.mkdir(parents=True)
    pd.DataFrame({"step": [1], "mean_loss": [0.5]}).to_csv(
        fold_dir / "metrics.csv", index=False
    )

    def fail_plot(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        "hybrax.train.postprocessing.plot_cross_fold_loss_curves", fail_plot
    )

    loo_mod._plot_cross_fold_losses(folds=(fold,), output_dir=tmp_path)

    assert "failed to write cross-fold loss curve" in caplog.text


def test_worker_env_strips_inherited_host_device_pin(monkeypatch):
    monkeypatch.setenv(
        "XLA_FLAGS", "--xla_force_host_platform_device_count=32 --xla_cpu_foo=1"
    )
    env = loo_mod._worker_env(3)
    assert env["HYBRAX_TRAIN_DEVICES"] == "3"
    assert "xla_force_host_platform_device_count" not in env.get("XLA_FLAGS", "")
    assert "--xla_cpu_foo=1" in env["XLA_FLAGS"]


def test_worker_env_drops_xla_flags_when_only_pin(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    assert "XLA_FLAGS" not in loo_mod._worker_env(2)


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


def _write_min_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "data": {"prepared": "prepared.json"},
                "train": {"epochs": 2, "seed": 7},
                "loo": {"parallel_folds": 1},
            }
        )
    )


def test_loo_cli_worker_is_collection_free(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    _write_min_config(cfg_path)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        cli,
        "load_process_collection",
        lambda *_a: pytest.fail("worker loaded prepared collection"),
    )
    monkeypatch.setattr(
        cli,
        "prepare_single_fold_from_runtime_artifact",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(cli, "train_prepared_fold", lambda _prepared: object())
    monkeypatch.setattr(cli, "execute_trained_fold", lambda _trained: None)

    assert (
        cli.main(
            [
                "loo",
                "--config",
                str(cfg_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--runtime-artifact",
                str(tmp_path / "artifact"),
                "--fold",
                "2",
            ]
        )
        == 0
    )
    assert captured["fold_idx"] == 2
    assert captured["artifact_path"] == tmp_path / "artifact"


def test_loo_cli_producer_is_the_only_collection_owner(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    _write_min_config(cfg_path)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        cli,
        "produce_runtime_artifact",
        lambda **kwargs: captured.update(kwargs),
    )

    assert (
        cli.main(
            [
                "loo",
                "--config",
                str(cfg_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--produce-runtime",
            ]
        )
        == 0
    )
    assert captured["bundle_path"] == cfg_path


def test_loo_cli_orchestrator_produces_before_dispatch(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    _write_min_config(cfg_path)
    (tmp_path / "prepared.json").write_text("{}")
    out_dir = tmp_path / "out"
    events: list[str] = []
    monkeypatch.setattr(
        cli,
        "load_process_collection",
        lambda *_a: pytest.fail("orchestrator loaded collection"),
    )
    metadata = _runtime_metadata(_three_parent_collection())
    monkeypatch.setattr(
        cli, "_dispatch_producer", lambda *_a: events.append("producer") or 0
    )
    monkeypatch.setattr(cli, "_validated_runtime_metadata", lambda *_a, **_k: metadata)

    def fake_run(**_kwargs):
        events.append("workers")
        document = json.loads((out_dir / "config.json").read_text())
        assert document["runtime_artifact"] == {
            "format_version": loo_mod.FORMAT_VERSION,
            "identity": metadata.identity,
        }
        return LOOResult(
            fold_dirs=(out_dir / "folds" / "p1",),
            parallel_folds=1,
            devices_per_fold=1,
            summary_csv_path=out_dir / "loo_summary.csv",
            aggregate_json_path=out_dir / "loo_aggregate.json",
            aggregate={"n_folds": 1},
        )

    monkeypatch.setattr(cli, "run_loo_cv", fake_run)

    assert (
        cli.main(
            [
                "loo",
                "--config",
                str(cfg_path),
                "--output-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    assert events == ["producer", "workers"]
    assert (out_dir / "loo-config.json").is_file()
    assert (out_dir / "prepared.json").is_file()


def _stub_loo_result(output_dir: Path) -> LOOResult:
    return LOOResult(
        fold_dirs=(),
        parallel_folds=1,
        devices_per_fold=1,
        summary_csv_path=output_dir / "loo_summary.csv",
        aggregate_json_path=output_dir / "loo_aggregate.json",
        aggregate={"n_folds": 0},
    )


def _patch_successful_orchestration(monkeypatch, output_dir: Path):
    metadata = _runtime_metadata(_three_parent_collection())
    monkeypatch.setattr(cli, "_dispatch_producer", lambda *_args: 0)
    monkeypatch.setattr(
        cli, "_validated_runtime_metadata", lambda *_args, **_kwargs: metadata
    )
    monkeypatch.setattr(
        cli, "run_loo_cv", lambda **_kwargs: _stub_loo_result(output_dir)
    )
    return metadata


@pytest.mark.parametrize("existing", ["empty", "config", "artifact"])
def test_loo_cli_fresh_rejects_every_existing_output_before_changes(
    monkeypatch, tmp_path, existing
):
    cfg_path = tmp_path / "config.json"
    _write_min_config(cfg_path)
    (tmp_path / "prepared.json").write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    if existing == "config":
        (output_dir / "config.json").write_text("sentinel", encoding="utf-8")
    elif existing == "artifact":
        artifact = output_dir / "runtime-artifact"
        artifact.mkdir()
        (artifact / "sentinel").write_text("keep", encoding="utf-8")
    before = {
        path.relative_to(output_dir): (None if path.is_dir() else path.read_bytes())
        for path in output_dir.rglob("*")
    }
    monkeypatch.setattr(
        cli,
        "_dispatch_producer",
        lambda *_args: pytest.fail("producer ran for an existing output"),
    )

    assert (
        cli.main(["loo", "--config", str(cfg_path), "--output-dir", str(output_dir)])
        == 1
    )

    after = {
        path.relative_to(output_dir): (None if path.is_dir() else path.read_bytes())
        for path in output_dir.rglob("*")
    }
    assert after == before


def test_loo_cli_overwrite_removes_and_reclaims_output(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    _write_min_config(cfg_path)
    (tmp_path / "prepared.json").write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "sentinel").write_text("old", encoding="utf-8")
    _patch_successful_orchestration(monkeypatch, output_dir)

    assert (
        cli.main(
            [
                "loo",
                "--config",
                str(cfg_path),
                "--output-dir",
                str(output_dir),
                "--overwrite",
            ]
        )
        == 0
    )

    assert not (output_dir / "sentinel").exists()
    assert json.loads((output_dir / "config.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("cli_override", [False, True])
def test_loo_cli_overwrite_replaces_symlink_without_touching_target(
    monkeypatch, tmp_path, cli_override
):
    cfg_path = tmp_path / "config.json"
    _write_min_config(cfg_path)
    config = json.loads(cfg_path.read_text())
    config["data"]["prepared"] = "target/prepared.json"
    if not cli_override:
        config["output"] = {"dir": "out"}
    cfg_path.write_text(json.dumps(config))
    target = tmp_path / "target"
    target.mkdir()
    prepared = target / "prepared.json"
    prepared.write_text("{}", encoding="utf-8")
    sentinel = target / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.symlink_to(target, target_is_directory=True)
    _patch_successful_orchestration(monkeypatch, output_dir)
    argv = ["loo", "--config", str(cfg_path), "--overwrite"]
    if cli_override:
        argv.extend(["--output-dir", str(output_dir)])

    assert cli.main(argv) == 0

    assert output_dir.is_dir()
    assert not output_dir.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert prepared.read_text(encoding="utf-8") == "{}"
    assert json.loads((output_dir / "config.json").read_text())["status"] == "complete"


def test_loo_cli_claims_missing_output_below_existing_parent(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    _write_min_config(cfg_path)
    (tmp_path / "prepared.json").write_text("{}", encoding="utf-8")
    parent = tmp_path / "parent"
    parent.mkdir()
    output_dir = parent / "out"
    _patch_successful_orchestration(monkeypatch, output_dir)

    assert (
        cli.main(["loo", "--config", str(cfg_path), "--output-dir", str(output_dir)])
        == 0
    )
    assert output_dir.is_dir()


def test_loo_cli_fresh_claim_race_has_one_unmodified_loser(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    _write_min_config(cfg_path)
    (tmp_path / "prepared.json").write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "out"
    _patch_successful_orchestration(monkeypatch, output_dir)

    argv = ["loo", "--config", str(cfg_path), "--output-dir", str(output_dir)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(cli.main, (argv, argv)))

    assert sorted(results) == [0, 1]
    assert json.loads((output_dir / "config.json").read_text())["status"] == "complete"


def test_loo_cli_producer_failure_leaves_no_commit_marker(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    _write_min_config(cfg_path)
    (tmp_path / "prepared.json").write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "out"
    monkeypatch.setattr(cli, "_dispatch_producer", lambda *_args: 17)
    monkeypatch.setattr(
        cli,
        "run_loo_cv",
        lambda **_kwargs: pytest.fail("workers dispatched after producer failure"),
    )

    with pytest.raises(RuntimeError, match="exit 17"):
        cli.main(["loo", "--config", str(cfg_path), "--output-dir", str(output_dir)])

    assert not (output_dir / "config.json").exists()


def test_loo_cli_manifest_failure_preserves_error_and_cleans_artifact(
    monkeypatch, tmp_path
):
    cfg_path = tmp_path / "config.json"
    _write_min_config(cfg_path)
    (tmp_path / "prepared.json").write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "out"

    def produce(*_args):
        artifact = output_dir / "runtime-artifact"
        artifact.mkdir()
        (artifact / "sentinel").write_text("published", encoding="utf-8")
        return 0

    monkeypatch.setattr(cli, "_dispatch_producer", produce)
    monkeypatch.setattr(
        cli,
        "_validated_runtime_metadata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("manifest sentinel")
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_loo_cv",
        lambda **_kwargs: pytest.fail("workers dispatched after manifest failure"),
    )

    with pytest.raises(ValueError, match="manifest sentinel"):
        cli.main(["loo", "--config", str(cfg_path), "--output-dir", str(output_dir)])

    assert not (output_dir / "config.json").exists()
    assert not (output_dir / "runtime-artifact").exists()


def test_loo_cli_config_publication_failure_cleans_unanchored_artifact(
    monkeypatch, tmp_path
):
    cfg_path = tmp_path / "config.json"
    _write_min_config(cfg_path)
    (tmp_path / "prepared.json").write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "out"
    metadata = _runtime_metadata(_three_parent_collection())

    def produce(*_args):
        artifact = output_dir / "runtime-artifact"
        artifact.mkdir()
        return 0

    monkeypatch.setattr(cli, "_dispatch_producer", produce)
    monkeypatch.setattr(
        cli, "_validated_runtime_metadata", lambda *_args, **_kwargs: metadata
    )
    real_replace = serialization.os.replace

    def fail_config_publication(source, destination):
        if Path(destination).name == "config.json":
            raise OSError("config publication sentinel")
        real_replace(source, destination)

    monkeypatch.setattr(serialization.os, "replace", fail_config_publication)
    monkeypatch.setattr(
        cli,
        "run_loo_cv",
        lambda **_kwargs: pytest.fail("workers dispatched after config failure"),
    )

    with pytest.raises(OSError, match="config publication sentinel"):
        cli.main(["loo", "--config", str(cfg_path), "--output-dir", str(output_dir)])

    assert not (output_dir / "config.json").exists()
    assert not (output_dir / "runtime-artifact").exists()
    assert list(output_dir.glob(".config.json.*.tmp")) == []


def test_loo_cli_resume_does_not_load_collection(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_min_config(run_dir / "loo-config.json")
    (run_dir / "config.json").write_text(json.dumps({"status": "running"}))
    monkeypatch.setattr(
        cli,
        "load_process_collection",
        lambda *_a: pytest.fail("resume loaded collection"),
    )
    monkeypatch.setattr(
        cli,
        "_runtime_metadata",
        lambda *_a, **_kw: (run_dir / "runtime-artifact", None),
    )
    monkeypatch.setattr(
        cli,
        "run_loo_cv",
        lambda **kwargs: LOOResult(
            fold_dirs=(),
            parallel_folds=1,
            devices_per_fold=1,
            summary_csv_path=run_dir / "loo_summary.csv",
            aggregate_json_path=run_dir / "loo_aggregate.json",
            aggregate={"n_folds": 2},
        ),
    )

    assert cli.main(["loo", "--resume", str(run_dir)]) == 0


def test_loo_cli_resume_rejects_uninitialized_run_dir(capsys, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_min_config(run_dir / "loo-config.json")

    assert cli.main(["loo", "--resume", str(run_dir)]) == 1
    # cli.main's logging.basicConfig(..., force=True) replaces root's handlers
    # (including pytest's caplog one) with its own real stderr handler, so this
    # asserts on captured stderr rather than caplog.text.
    stderr = capsys.readouterr().err
    assert "initialization did not complete" in stderr
    assert "--overwrite" in stderr


def test_loo_cli_rejects_config_and_resume_together(tmp_path):
    assert cli.main(["loo", "--config", "x.json", "--resume", str(tmp_path)]) == 1
