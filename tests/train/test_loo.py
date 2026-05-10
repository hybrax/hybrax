"""Tests for the Leave-One-Process-Out cross-validation orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pandas as pd
import pytest
from bp_format.dataclasses import (
    AugmentedBioProcess,
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    ReactorMedium,
    ReactorMediumComponent,
    SampleVolumeChange,
    TimeAxis,
    TimeSeries,
    Volume,
)

from bp_train import cli
from bp_train.harness import (
    ForwardConfig,
    ForwardResult,
    TrainHarnessConfig,
    TrainHarnessResult,
)
from bp_train.loo import (
    FoldResult,
    LOOConfig,
    LOOResult,
    _build_fold_groups,
    _resolve_selected_folds,
    run_loo_cv,
    run_loo_fold,
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


# ---------------------------------------------------------------------------
# _build_fold_groups
# ---------------------------------------------------------------------------


def test_build_fold_groups_returns_one_group_per_parent():
    collection = _three_parent_collection()
    groups = _build_fold_groups(collection)
    assert groups == (
        ("p1", ("p1",)),
        ("p2", ("p2",)),
        ("p3", ("p3",)),
    )


def test_build_fold_groups_attaches_augmented_children_to_parent():
    collection = _augmented_collection()
    groups = _build_fold_groups(collection)
    assert groups == (
        ("P0", ("P0", "P0_aug")),
        ("P1", ("P1",)),
    )


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
# _resolve_selected_folds
# ---------------------------------------------------------------------------


def test_resolve_selected_folds_runs_all_when_none():
    collection = _three_parent_collection()
    groups = _build_fold_groups(collection)
    resolved = _resolve_selected_folds(groups, None, collection)
    assert tuple(idx for idx, _ in resolved) == (0, 1, 2)


def test_resolve_selected_folds_handles_subset():
    collection = _three_parent_collection()
    groups = _build_fold_groups(collection)
    resolved = _resolve_selected_folds(groups, ("p3", "p1"), collection)
    assert [idx for idx, _ in resolved] == [2, 0]


def test_resolve_selected_folds_rejects_unknown():
    collection = _three_parent_collection()
    groups = _build_fold_groups(collection)
    with pytest.raises(ValueError, match="unknown process name 'ghost'"):
        _resolve_selected_folds(groups, ("ghost",), collection)


def test_resolve_selected_folds_rejects_augmented_holdout():
    collection = _augmented_collection()
    groups = _build_fold_groups(collection)
    with pytest.raises(ValueError, match="must reference parent processes"):
        _resolve_selected_folds(groups, ("P0_aug",), collection)


def test_resolve_selected_folds_rejects_duplicates():
    collection = _three_parent_collection()
    groups = _build_fold_groups(collection)
    with pytest.raises(ValueError, match="duplicate entry"):
        _resolve_selected_folds(groups, ("p1", "p1"), collection)


# ---------------------------------------------------------------------------
# run_loo_cv with mocked training/forward
# ---------------------------------------------------------------------------


def _stub_train_result() -> TrainHarnessResult:
    return TrainHarnessResult(
        trained_wrapper=object(),
        mean_loss_by_step=(1.0, 0.5),
        sampled_loss_by_process_at_log_steps={1: (("p1", 1.0),)},
        batch_process_names_by_step=(("p1",),),
        per_process_loss_by_step=((1.0,),),
        compile_warmup_seconds=0.0,
        step_time_seconds=(0.0,),
        train_step_input_signature=(),
        train_step_rebuild_count=0,
    )


def _stub_forward_result_for_collection(
    collection: BioProcessCollection,
    training_processes: tuple[str, ...],
) -> ForwardResult:
    process_names = tuple(collection.processes.keys())
    per_total = {name: 0.1 if name in training_processes else 0.5 for name in process_names}
    per_target = {
        name: (per_total[name],) for name in process_names
    }

    class _DummyStore:
        process_order = process_names

    return ForwardResult(
        trained_wrapper=None,
        store=_DummyStore(),
        process_names=process_names,
        target_names=("biomass",),
        name_modeled_FVCs=(),
        name_modeled_SVCs=(),
        training_process_names=training_processes,
        per_process_total_loss=per_total,
        per_process_per_target_loss=per_target,
    )


def _patch_loo_internals(monkeypatch, collection):
    """Mock train_from_collection and _write_train_results for fast LOO tests."""
    captured: dict[str, list[Any]] = {"train_calls": [], "fold_dirs": []}

    def fake_train_from_collection(coll, *, config, custom_py, runtime_config):
        captured["train_calls"].append(
            {
                "process_names": config.process_names,
                "seed": config.seed,
                "custom_py": custom_py,
                "runtime_config": runtime_config,
            }
        )
        return _stub_train_result()

    def fake_write_train_results(
        *,
        output_dir,
        collection,
        trained_wrapper,
        train_result,
        config,
        runtime_config,
        custom_py,
        training_process_names,
        render_plots,
        eval_process_names=None,
    ):
        captured["fold_dirs"].append(Path(output_dir))
        return _stub_forward_result_for_collection(
            collection,
            training_processes=training_process_names,
        )

    monkeypatch.setattr(
        "bp_train.loo.train_from_collection",
        fake_train_from_collection,
    )
    # _write_train_results is imported lazily inside _execute_fold;
    # patch on cli where it lives.
    monkeypatch.setattr(
        "bp_train.cli._write_train_results",
        fake_write_train_results,
    )
    monkeypatch.setattr(
        "bp_train.loo.save_model",
        lambda *_a, **_k: None,
        raising=False,
    )
    # save_model lives in postprocessing; patch imported alias at use site:
    import bp_train.postprocessing as postprocessing

    monkeypatch.setattr(postprocessing, "save_model", lambda *_a, **_k: None)
    monkeypatch.setattr(
        postprocessing, "save_model_metadata", lambda *_a, **_k: None
    )
    return captured


def _make_loo_config(tmp_path: Path, *, selected=None, render_plots=False) -> LOOConfig:
    base = TrainHarnessConfig(
        steps=2,
        log_every=1,
        seed=10,
        checkpoint_dir=None,  # disable per-fold checkpoints in tests
    )
    return LOOConfig(
        base_train_config=base,
        output_dir=tmp_path,
        selected_holdouts=selected,
        render_plots=render_plots,
    )


def test_run_loo_cv_produces_one_fold_per_parent(monkeypatch, tmp_path):
    collection = _three_parent_collection()
    captured = _patch_loo_internals(monkeypatch, collection)
    cfg = _make_loo_config(tmp_path)

    result = run_loo_cv(collection, config=cfg)

    assert isinstance(result, LOOResult)
    assert len(result.folds) == 3
    parents = [fold.holdout_parent for fold in result.folds]
    assert parents == ["p1", "p2", "p3"]
    for fold in result.folds:
        assert fold.holdout_parent not in fold.train_processes
        assert set(fold.train_processes).isdisjoint(fold.holdout_group)

    # 3 train calls, each excluding the held-out parent
    excluded = [
        set(collection.processes) - set(call["process_names"])
        for call in captured["train_calls"]
    ]
    assert excluded == [{"p1"}, {"p2"}, {"p3"}]


def test_run_loo_cv_groups_augmented_children_with_parent(monkeypatch, tmp_path):
    collection = _augmented_collection()
    captured = _patch_loo_internals(monkeypatch, collection)
    cfg = _make_loo_config(tmp_path)

    result = run_loo_cv(collection, config=cfg)

    assert [fold.holdout_parent for fold in result.folds] == ["P0", "P1"]
    p0_fold = result.folds[0]
    assert p0_fold.holdout_group == ("P0", "P0_aug")
    assert p0_fold.train_processes == ("P1",)
    # Anti-leakage: when P0 is held out, P0_aug must NOT train.
    assert "P0_aug" not in p0_fold.train_processes
    # When P1 is held out, P0 (and its augmented child P0_aug) train together.
    p1_fold = result.folds[1]
    assert p1_fold.holdout_group == ("P1",)
    assert set(p1_fold.train_processes) == {"P0", "P0_aug"}


def test_run_loo_fold_uses_seed_plus_fold_idx(monkeypatch, tmp_path):
    collection = _three_parent_collection()
    captured = _patch_loo_internals(monkeypatch, collection)
    cfg = _make_loo_config(tmp_path)

    fold = run_loo_fold(collection, holdout_parent="p2", config=cfg)

    assert fold.fold_idx == 1
    assert fold.fold_seed == cfg.base_train_config.seed + 1
    train_call = captured["train_calls"][0]
    assert train_call["seed"] == cfg.base_train_config.seed + 1
    assert train_call["process_names"] == ("p1", "p3")


def test_run_loo_cv_writes_summary_and_aggregate(monkeypatch, tmp_path):
    collection = _three_parent_collection()
    _patch_loo_internals(monkeypatch, collection)
    cfg = _make_loo_config(tmp_path)

    result = run_loo_cv(collection, config=cfg)

    assert result.summary_csv_path is not None
    assert result.summary_csv_path.exists()
    assert result.aggregate_json_path is not None
    df = pd.read_csv(result.summary_csv_path)
    # one row per fold + one aggregate row at the end
    assert len(df) == 4
    assert set(df.columns) >= {
        "fold_idx",
        "holdout_parent",
        "holdout_group",
        "fold_seed",
        "holdout_total",
        "holdout_biomass",
        "train_mean_total",
        "final_train_loss",
    }
    # Last row is the mean aggregate
    last = df.iloc[-1]
    assert last["holdout_parent"] == "mean"

    aggregate = json.loads(result.aggregate_json_path.read_text(encoding="utf-8"))
    assert aggregate["n_folds"] == 3
    assert aggregate["base_seed"] == cfg.base_train_config.seed
    assert "holdout_total_mean" in aggregate
    assert "holdout_total_std" in aggregate


def test_run_loo_cv_skips_summary_for_subset(monkeypatch, tmp_path):
    collection = _three_parent_collection()
    _patch_loo_internals(monkeypatch, collection)
    cfg = _make_loo_config(tmp_path, selected=("p2",))

    result = run_loo_cv(collection, config=cfg)

    assert len(result.folds) == 1
    assert result.folds[0].holdout_parent == "p2"
    assert result.summary_csv_path is None
    assert result.aggregate_json_path is None
    assert not (tmp_path / "loo_summary.csv").exists()


def test_run_loo_cv_writes_fold_artifact_layout(monkeypatch, tmp_path):
    collection = _three_parent_collection()
    _patch_loo_internals(monkeypatch, collection)
    cfg = _make_loo_config(tmp_path)

    result = run_loo_cv(collection, config=cfg)

    for fold in result.folds:
        assert fold.fold_dir == tmp_path / "folds" / fold.holdout_parent
        assert fold.fold_dir.exists()


def test_run_loo_cv_fails_fast_on_single_parent(monkeypatch, tmp_path):
    collection = BioProcessCollection(processes={"p1": _make_process("p1")})
    cfg = _make_loo_config(tmp_path)
    with pytest.raises(ValueError, match="LOO-CV requires at least 2 parent processes"):
        run_loo_cv(collection, config=cfg)


def test_run_loo_fold_rejects_augmented_holdout(monkeypatch, tmp_path):
    collection = _augmented_collection()
    cfg = _make_loo_config(tmp_path)
    with pytest.raises(ValueError, match="AugmentedBioProcess"):
        run_loo_fold(collection, holdout_parent="P0_aug", config=cfg)


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


def test_loo_cli_dispatches_to_run_loo_cv(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    sentinel_collection = _three_parent_collection()

    def fake_load(json_path):
        captured["loaded_path"] = Path(json_path)
        return sentinel_collection

    def fake_run_loo_cv(collection, *, config, custom_py, runtime_config):
        captured["collection"] = collection
        captured["config"] = config
        captured["custom_py"] = custom_py
        captured["runtime_config"] = runtime_config
        return LOOResult(
            folds=(),
            summary_csv_path=None,
            aggregate_json_path=None,
            aggregate={},
        )

    monkeypatch.setattr(cli, "load_process_collection_json", fake_load)
    monkeypatch.setattr(cli, "run_loo_cv", fake_run_loo_cv)
    monkeypatch.setattr(cli, "load_custom_module", lambda _path: object())
    monkeypatch.setattr(cli, "resolve_config", lambda _module, _config: {})
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(
        [
            "loo",
            "--input",
            "prepared.json",
            "--custom",
            "custom.py",
            "--holdouts",
            "p1,p2",
            "--steps",
            "3",
            "--seed",
            "7",
            "--output-dir",
            "out_loo",
            "--no-plot",
        ]
    )

    assert exit_code == 0
    cfg = captured["config"]
    assert isinstance(cfg, LOOConfig)
    assert cfg.selected_holdouts == ("p1", "p2")
    assert cfg.render_plots is False
    assert cfg.output_dir == Path("out_loo")
    assert cfg.base_train_config.seed == 7
    assert cfg.base_train_config.steps == 3
