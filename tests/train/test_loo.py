"""Tests for the config-driven Leave-one/some-process-out CV orchestrator."""

from __future__ import annotations

import json
import math
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
from bp_train.harness import TrainHarnessResult
from bp_train import loo as loo_mod
from bp_train.loo import (
    Fold,
    FoldResult,
    LOOResult,
    _build_fold_groups,
    compute_parallel_split,
    resolve_folds,
    run_loo_cv,
    run_single_fold,
    _write_summary_and_aggregate,
)
from bp_train.run_config import (
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
    folds = resolve_folds(_three_parent_collection(), None)
    assert [f.slug for f in folds] == ["p1", "p2", "p3"]
    assert folds[0].test == ("p1",)
    assert folds[0].train == ("p2", "p3")
    assert all(set(f.test).isdisjoint(f.train) for f in folds)


def test_resolve_folds_auto_groups_augmented_with_parent():
    folds = resolve_folds(_augmented_collection(), None)
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
        resolve_folds(collection, None)


# ---------------------------------------------------------------------------
# resolve_folds — explicit per_fold_holdout_sets
# ---------------------------------------------------------------------------


def test_resolve_folds_explicit_default_train():
    loo_cfg = LooConfig(
        per_fold_holdout_sets=(HoldoutSet(test=("p1",)), HoldoutSet(test=("p2", "p3")))
    )
    folds = resolve_folds(_three_parent_collection(), loo_cfg)
    assert folds[0].test == ("p1",)
    assert folds[0].train == ("p2", "p3")
    assert folds[1].test == ("p2", "p3")
    assert folds[1].train == ("p1",)
    assert folds[1].slug == "p2+p3"


def test_resolve_folds_explicit_pinned_train():
    loo_cfg = LooConfig(
        per_fold_holdout_sets=(HoldoutSet(test=("p2", "p3"), train=("p1",)),)
    )
    folds = resolve_folds(_three_parent_collection(), loo_cfg)
    assert folds[0].train == ("p1",)


def test_resolve_folds_explicit_default_train_excludes_augmented_child():
    # test=[P0]; default train = everything not in test, but P0_aug must drop out
    # because its parent P0 is held out (no leak).
    loo_cfg = LooConfig(per_fold_holdout_sets=(HoldoutSet(test=("P0",)),))
    folds = resolve_folds(_augmented_collection(), loo_cfg)
    assert folds[0].train == ("P1",)


def test_resolve_folds_explicit_train_leak_raises():
    # Pinning P0_aug into train while its parent P0 is held out is a leak.
    loo_cfg = LooConfig(
        per_fold_holdout_sets=(HoldoutSet(test=("P0",), train=("P1", "P0_aug")),)
    )
    with pytest.raises(ValueError, match="leaks augmentation-group"):
        resolve_folds(_augmented_collection(), loo_cfg)


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
    folds = resolve_folds(_two_child_collection(), loo_cfg)
    assert folds[0].train == ("P1",)  # P0 and C2 excluded, not just C1


def test_resolve_folds_explicit_train_parent_of_held_out_child_raises():
    # test=child, train pins the parent -> parent leaks the held-out variant.
    loo_cfg = LooConfig(
        per_fold_holdout_sets=(HoldoutSet(test=("C1",), train=("P0", "P1")),)
    )
    with pytest.raises(ValueError, match="leaks augmentation-group"):
        resolve_folds(_two_child_collection(), loo_cfg)


def test_resolve_folds_named_fold_slug():
    loo_cfg = LooConfig(
        per_fold_holdout_sets=(
            HoldoutSet(name="control, high S", test=("p1", "p2")),
            HoldoutSet(name="1003 47µLS@5h", test=("p3",)),
        )
    )
    folds = resolve_folds(_three_parent_collection(), loo_cfg)
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
        resolve_folds(_three_parent_collection(), loo_cfg)


def test_resolve_folds_explicit_unknown_test_raises():
    loo_cfg = LooConfig(per_fold_holdout_sets=(HoldoutSet(test=("ghost",)),))
    with pytest.raises(ValueError, match="unknown process name"):
        resolve_folds(_three_parent_collection(), loo_cfg)


def test_resolve_folds_explicit_overlap_raises():
    loo_cfg = LooConfig(
        per_fold_holdout_sets=(HoldoutSet(test=("p1",), train=("p1", "p2")),)
    )
    with pytest.raises(ValueError, match="both test and train"):
        resolve_folds(_three_parent_collection(), loo_cfg)


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
    folds = resolve_folds(_three_parent_collection(), None)
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

    def fake_train(coll, *, config, custom_module, run_config):
        captured["process_names"] = config.process_names
        captured["seed"] = config.seed
        captured["holdout"] = config.holdout_processes
        captured["run_config"] = run_config
        return _stub_train_result()

    def fake_write(
        *, output_dir, training_process_names, eval_process_names=None, **_kw
    ):
        captured["fold_dir"] = Path(output_dir)
        captured["training_process_names"] = training_process_names
        captured["eval_process_names"] = eval_process_names

        class _Dummy:
            pass

        return _Dummy()

    monkeypatch.setattr("bp_train.loo.train_from_collection", fake_train)
    monkeypatch.setattr("bp_train.cli._write_train_results", fake_write)
    import bp_train.postprocessing as pp
    import bp_train.serialization as ser

    monkeypatch.setattr(ser, "save_model", lambda *_a, **_k: None)
    monkeypatch.setattr(pp, "save_model_metadata", lambda *_a, **_k: None)
    return captured


def test_run_single_fold_trains_excluding_holdout(monkeypatch, tmp_path):
    collection = _three_parent_collection()
    captured = _patch_worker_internals(monkeypatch)
    cfg = _run_config(seed=10)
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
    assert result.fold_seed == 10 + 1  # base seed + fold idx
    assert result.fold_dir == tmp_path / "folds" / "p2"
    effective = captured["run_config"]
    assert effective.data.processes == ("p1", "p3")
    assert effective.train.seed == 11
    assert effective.output.dir == result.fold_dir.resolve()
    document = json.loads((result.fold_dir / "config.json").read_text())
    assert document["status"] == "complete"
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


# ---------------------------------------------------------------------------
# run_loo_cv (orchestrator) with mocked subprocess dispatch
# ---------------------------------------------------------------------------


def _patch_dispatch(monkeypatch) -> dict[str, Any]:
    """Replace the subprocess pool with a stub-fold writer (no real training)."""
    seen: dict[str, Any] = {}

    def fake_pool(config_path, output_dir, folds, parallel, devices):
        seen["pool_folds"] = [f.slug for f in folds]
        seen["parallel"] = parallel
        seen["devices"] = devices
        for fold in folds:
            _write_stub_fold(Path(output_dir), fold)

    monkeypatch.setattr(loo_mod, "_dispatch_pool", fake_pool)
    return seen


def test_run_loo_cv_runs_all_folds_and_aggregates(monkeypatch, tmp_path):
    collection = _three_parent_collection()
    seen = _patch_dispatch(monkeypatch)
    cfg = RunConfig(
        data=DataConfig(prepared=Path("prepared.json")),
        train=TrainConfig(epochs=2, seed=10),
        loo=LooConfig(parallel_folds=2),
    )

    result = run_loo_cv(
        collection,
        cfg=cfg,
        config_path=tmp_path / "loo-config.json",
        output_dir=tmp_path,
    )

    assert isinstance(result, LOOResult)
    assert seen["pool_folds"] == ["p1", "p2", "p3"]  # all folds dispatched
    assert seen["parallel"] == 2  # user-set parallel_folds
    assert result.parallel_folds == 2
    assert (tmp_path / "loo_summary.csv").exists()
    assert result.aggregate["n_folds"] == 3


def test_run_loo_cv_uses_configured_devices_per_fold(monkeypatch, tmp_path):
    monkeypatch.setattr(loo_mod.os, "cpu_count", lambda: 16)
    collection = _three_parent_collection()
    seen = _patch_dispatch(monkeypatch)
    cfg = RunConfig(
        data=DataConfig(prepared=Path("prepared.json")),
        train=TrainConfig(epochs=2, seed=10),
        loo=LooConfig(parallel_folds=2, devices_per_fold=2),
    )

    result = run_loo_cv(
        collection,
        cfg=cfg,
        config_path=tmp_path / "loo-config.json",
        output_dir=tmp_path,
    )

    assert seen["parallel"] == 2
    assert seen["devices"] == 2
    assert result.devices_per_fold == 2


def test_run_loo_cv_resume_skips_completed_folds(monkeypatch, tmp_path):
    collection = _three_parent_collection()
    seen = _patch_dispatch(monkeypatch)
    folds = resolve_folds(collection, None)
    _write_stub_fold(tmp_path, folds[0])  # pretend fold "p1" already finished
    cfg = RunConfig(
        data=DataConfig(prepared=Path("prepared.json")),
        train=TrainConfig(epochs=2, seed=10),
        loo=LooConfig(parallel_folds=1),
    )

    result = run_loo_cv(
        collection,
        cfg=cfg,
        config_path=tmp_path / "loo-config.json",
        output_dir=tmp_path,
        resume=True,
    )

    assert seen["pool_folds"] == ["p2", "p3"]  # p1 skipped (already complete)
    assert result.aggregate["n_folds"] == 3  # aggregate still spans all folds


def test_worker_env_strips_inherited_host_device_pin(monkeypatch):
    monkeypatch.setenv(
        "XLA_FLAGS", "--xla_force_host_platform_device_count=32 --xla_cpu_foo=1"
    )
    env = loo_mod._worker_env(3)
    assert env["BP_TRAIN_DEVICES"] == "3"
    assert "xla_force_host_platform_device_count" not in env.get("XLA_FLAGS", "")
    assert "--xla_cpu_foo=1" in env["XLA_FLAGS"]


def test_worker_env_drops_xla_flags_when_only_pin(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    env = loo_mod._worker_env(2)
    assert "XLA_FLAGS" not in env


def test_single_fold_std_is_nan(tmp_path):
    collection = _three_parent_collection()
    loo_cfg = LooConfig(per_fold_holdout_sets=(HoldoutSet(test=("p1",)),))
    folds = resolve_folds(collection, loo_cfg)
    _write_stub_fold(tmp_path, folds[0])
    agg = loo_mod._write_summary_and_aggregate(
        folds=folds,
        output_dir=tmp_path,
        summary_csv_path=tmp_path / "s.csv",
        aggregate_json_path=tmp_path / "a.json",
        base_seed=0,
    )
    assert math.isnan(agg["holdout_total_std"])


def test_read_final_train_loss_accepts_comments_and_malformed_is_nan(tmp_path):
    sidecar = tmp_path / "trained_wrapper.meta.json"
    sidecar.write_text(
        '// fit result\n{"training": {"final_mean_loss": 1.25}}',
        encoding="utf-8",
    )
    assert loo_mod._read_final_train_loss(tmp_path) == pytest.approx(1.25)

    sidecar.write_text("// comment only", encoding="utf-8")
    assert math.isnan(loo_mod._read_final_train_loss(tmp_path))


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


def test_loo_cli_worker_mode_calls_run_single_fold(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}
    cfg_path = tmp_path / "config.json"
    _write_min_config(cfg_path)

    monkeypatch.setattr(cli, "load_process_collection", lambda _p: object())

    def fake_single(collection, *, cfg, custom_module, output_dir, fold_idx, custom_py):
        captured["fold_idx"] = fold_idx
        captured["output_dir"] = Path(output_dir)
        return None

    monkeypatch.setattr(cli, "run_single_fold", fake_single)

    rc = cli.main(
        [
            "loo",
            "--config",
            str(cfg_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--fold",
            "2",
        ]
    )
    assert rc == 0
    assert captured["fold_idx"] == 2
    assert captured["output_dir"] == tmp_path / "out"


def test_loo_cli_orchestrator_bundles_and_calls_cv(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}
    cfg_path = tmp_path / "config.json"
    _write_min_config(cfg_path)
    cfg_path.write_text(
        "// source config comment\n" + cfg_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "prepared.json").write_text("{}")  # real file -> bundle copies it
    out_dir = tmp_path / "out"
    stale_fold = out_dir / "folds" / "old-fold"
    stale_fold.mkdir(parents=True)
    (stale_fold / "checkpoint.eqx").write_text("stale", encoding="utf-8")
    (out_dir / "obsolete.txt").write_text("old run", encoding="utf-8")

    monkeypatch.setattr(cli, "load_process_collection", lambda _p: object())
    monkeypatch.setattr(cli, "content_hash", lambda _c: "sha256:stub")

    def fake_cv(collection, *, cfg, config_path, output_dir, custom_py, resume=False):
        captured["config_path"] = Path(config_path)
        captured["output_dir"] = Path(output_dir)
        captured["resume"] = resume
        return LOOResult(
            fold_dirs=(Path(output_dir) / "folds" / "p1",),
            parallel_folds=1,
            devices_per_fold=1,
            summary_csv_path=Path(output_dir) / "loo_summary.csv",
            aggregate_json_path=Path(output_dir) / "loo_aggregate.json",
            aggregate={"n_folds": 1},
        )

    monkeypatch.setattr(cli, "run_loo_cv", fake_cv)

    rc = cli.main(
        [
            "loo",
            "--config",
            str(cfg_path),
            "--output-dir",
            str(out_dir),
            "--overwrite",
        ]
    )
    assert rc == 0
    assert not (out_dir / "obsolete.txt").exists()
    assert not stale_fold.exists()
    # Workers are pointed at the bundled, self-contained config — not the source.
    assert captured["config_path"] == out_dir / "loo-config.json"
    assert captured["resume"] is False
    # The run dir is self-contained: bundled loadable config + copied prepared.
    assert (out_dir / "loo-config.json").is_file()
    assert (out_dir / "prepared.json").is_file()
    bundled = json.loads((out_dir / "loo-config.json").read_text())
    assert bundled["data"]["prepared"] == "prepared.json"  # relative -> local copy
    assert bundled["output"]["dir"] == "."
    document = json.loads((out_dir / "config.json").read_text())
    assert document["status"] == "complete"
    assert document["aggregate"] == {"n_folds": 1}


def test_loo_cli_resume_reloads_bundle(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "loo-config.json").write_text(
        json.dumps(
            {
                "data": {"prepared": "prepared.json"},
                "train": {"epochs": 2, "seed": 7},
                "output": {"dir": "."},
                "loo": {"parallel_folds": 1},
            }
        )
    )
    (run_dir / "prepared.json").write_text("{}")
    (run_dir / "config.json").write_text(json.dumps({"status": "running"}))

    monkeypatch.setattr(cli, "load_process_collection", lambda _p: object())

    def fake_cv(collection, *, cfg, config_path, output_dir, custom_py, resume=False):
        captured["resume"] = resume
        captured["config_path"] = Path(config_path)
        captured["output_dir"] = Path(output_dir)
        return LOOResult(
            fold_dirs=(),
            parallel_folds=1,
            devices_per_fold=1,
            summary_csv_path=run_dir / "loo_summary.csv",
            aggregate_json_path=run_dir / "loo_aggregate.json",
            aggregate={"n_folds": 2},
        )

    monkeypatch.setattr(cli, "run_loo_cv", fake_cv)

    rc = cli.main(["loo", "--resume", str(run_dir)])
    assert rc == 0
    assert captured["resume"] is True
    assert captured["config_path"] == run_dir / "loo-config.json"
    assert captured["output_dir"] == run_dir
    document = json.loads((run_dir / "config.json").read_text())
    assert document["status"] == "complete"


def test_loo_cli_rejects_config_and_resume_together(tmp_path):
    rc = cli.main(["loo", "--config", "x.json", "--resume", str(tmp_path)])
    assert rc == 1
