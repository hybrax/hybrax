"""Tests for the config-driven Leave-one/some-process-out CV orchestrator."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
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
from bp_train.harness import PreparedTraining, TrainHarnessResult
from bp_train import loo as loo_mod
from bp_train.loo import (
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
# resolve_folds — data_processes restriction
# ---------------------------------------------------------------------------


def test_resolve_folds_classic_respects_data_processes():
    folds = resolve_folds(_three_parent_collection(), None, data_processes=("p1", "p2"))
    assert [f.slug for f in folds] == ["p1", "p2"]
    assert all("p3" not in (*f.test, *f.train) for f in folds)


def test_resolve_folds_classic_data_processes_default_train_excludes_restricted():
    folds = resolve_folds(_three_parent_collection(), None, data_processes=("p1", "p2"))
    assert folds[0].train == ("p2",)


def test_resolve_folds_data_processes_child_without_parent_raises():
    with pytest.raises(ValueError, match="excludes its parent process 'P0'"):
        resolve_folds(_augmented_collection(), None, data_processes=("P0_aug", "P1"))


def test_resolve_folds_data_processes_parent_without_child_is_allowed():
    folds = resolve_folds(_augmented_collection(), None, data_processes=("P0", "P1"))
    assert [f.slug for f in folds] == ["P0", "P1"]
    assert folds[0].test == ("P0",)
    assert folds[0].train == ("P1",)


def test_resolve_folds_data_processes_unknown_name_raises():
    with pytest.raises(ValueError, match="data.processes.*unknown process name"):
        resolve_folds(_three_parent_collection(), None, data_processes=("p1", "ghost"))


def test_resolve_folds_per_fold_test_outside_data_processes_raises():
    loo_cfg = LooConfig(per_fold_holdout_sets=(HoldoutSet(test=("p3",)),))
    with pytest.raises(ValueError, match="excluded by data.processes"):
        resolve_folds(_three_parent_collection(), loo_cfg, data_processes=("p1", "p2"))


def test_resolve_folds_per_fold_train_outside_data_processes_raises():
    loo_cfg = LooConfig(
        per_fold_holdout_sets=(HoldoutSet(test=("p1",), train=("p3",)),)
    )
    with pytest.raises(ValueError, match="excluded by data.processes"):
        resolve_folds(_three_parent_collection(), loo_cfg, data_processes=("p1", "p2"))


def test_resolve_folds_per_fold_default_train_restricted_by_data_processes():
    loo_cfg = LooConfig(per_fold_holdout_sets=(HoldoutSet(test=("p1",)),))
    folds = resolve_folds(
        _three_parent_collection(), loo_cfg, data_processes=("p1", "p2")
    )
    assert folds[0].train == ("p2",)


def test_resolve_folds_data_processes_none_matches_unrestricted():
    collection = _three_parent_collection()
    assert resolve_folds(collection, None, data_processes=None) == resolve_folds(
        collection, None
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
            plot_sources=None,
        )

    def fake_evaluate(wrapper, store, *, config, training_process_names, **_kw):
        captured["evaluation_wrapper"] = wrapper
        captured["training_process_names"] = training_process_names
        captured["eval_process_names"] = config.process_names
        return SimpleNamespace()

    def fake_write(*, output_dir, **_kw):
        captured["fold_dir"] = Path(output_dir)

    monkeypatch.setattr("bp_train.loo.prepare_training", fake_prepare)
    monkeypatch.setattr(
        "bp_train.loo.train_collection", lambda *_a, **_k: _stub_train_result()
    )
    monkeypatch.setattr("bp_train.loo.evaluate_trained_wrapper", fake_evaluate)
    monkeypatch.setattr("bp_train.cli._write_train_results", fake_write)
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


def test_prepare_single_fold_merges_only_missing_holdout_plot_sources(
    monkeypatch, tmp_path
):
    collection = _three_parent_collection()
    extracted = []

    def fake_prepare(*_args, config, **_kwargs):
        assert config.plots is True
        return PreparedTraining(
            store=SimpleNamespace(rhs_ode=object()),
            reaction_module=object(),
            loss_module=SimpleNamespace(loss_names=("X",)),
            config=config,
            optimizer=object(),
            plot_sources={"p1": "train-1", "p3": "train-3"},
        )

    def fake_extract(_collection, _rhs, process_names):
        extracted.append(process_names)
        return {name: f"holdout-{name}" for name in process_names}

    monkeypatch.setattr(loo_mod, "prepare_training", fake_prepare)
    monkeypatch.setattr(loo_mod, "extract_process_plot_sources", fake_extract)
    prepared = loo_mod.prepare_single_fold(
        collection,
        cfg=_run_config(),
        custom_module=None,
        output_dir=tmp_path,
        fold_idx=1,
    )

    assert extracted == [("p2",)]
    assert prepared.training.plot_sources == {
        "p1": "train-1",
        "p3": "train-3",
        "p2": "holdout-p2",
    }


def test_prepare_single_fold_skips_holdout_sources_when_plots_are_off(
    monkeypatch, tmp_path
):
    def fake_prepare(*_args, config, **_kwargs):
        assert config.plots is False
        return PreparedTraining(
            store=object(),
            reaction_module=object(),
            loss_module=SimpleNamespace(loss_names=("X",)),
            config=config,
            optimizer=object(),
            plot_sources=None,
        )

    monkeypatch.setattr(loo_mod, "prepare_training", fake_prepare)
    monkeypatch.setattr(
        loo_mod,
        "extract_process_plot_sources",
        lambda *_a, **_k: pytest.fail("holdout sources extracted with plots off"),
    )
    cfg = _run_config()
    cfg = cfg.model_copy(
        update={"output": cfg.output.model_copy(update={"plots": False})}
    )
    prepared = loo_mod.prepare_single_fold(
        _three_parent_collection(),
        cfg=cfg,
        custom_module=None,
        output_dir=tmp_path,
        fold_idx=1,
    )

    assert prepared.training.plot_sources is None


# ---------------------------------------------------------------------------
# Artifact-backed orchestration and internal CLI modes
# ---------------------------------------------------------------------------


def _runtime_state(collection, output_dir: Path, seed: int = 10):
    folds = resolve_folds(collection, None)
    records = tuple(
        loo_mod.RuntimeArtifactFold(
            idx=fold.idx,
            test=fold.test,
            train=fold.train,
            slug=fold.slug,
            seed=seed + fold.idx,
        )
        for fold in folds
    )
    return loo_mod.LooRuntimeState(
        "sha256:fingerprint",
        "sha256:prepared",
        loo_mod.FORMAT_VERSION,
        output_dir / "runtime-artifact",
        "sha256:artifact",
        records,
    )


def test_loo_runtime_state_rejects_nonlocal_artifact_path(tmp_path):
    state = _runtime_state(_three_parent_collection(), Path("."))
    loo_mod._write_loo_state(tmp_path / "loo-runtime.json", state)
    raw = json.loads((tmp_path / "loo-runtime.json").read_text())
    raw["artifact_path"] = "../runtime-artifact"
    (tmp_path / "loo-runtime.json").write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="invalid LOO runtime state"):
        loo_mod._read_loo_state(tmp_path)


def test_loo_runtime_state_rejects_nonlist_fold_membership(tmp_path):
    state = _runtime_state(_three_parent_collection(), Path("."))
    loo_mod._write_loo_state(tmp_path / "loo-runtime.json", state)
    raw = json.loads((tmp_path / "loo-runtime.json").read_text())
    raw["folds"][0]["test"] = "p1"
    (tmp_path / "loo-runtime.json").write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="invalid LOO runtime state"):
        loo_mod._read_loo_state(tmp_path)


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


def test_run_loo_cv_dispatches_only_state_folds(monkeypatch, tmp_path):
    collection = _three_parent_collection()
    state = _runtime_state(collection, tmp_path)
    monkeypatch.setattr(loo_mod, "_validate_state", lambda *_a, **_k: state)
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
    assert seen["artifact_path"] == state.artifact_path
    assert seen["parallel"] == 2
    assert result.parallel_folds == 2
    assert result.aggregate["n_folds"] == 3


def test_run_loo_cv_uses_configured_devices_per_fold(monkeypatch, tmp_path):
    monkeypatch.setattr(loo_mod.os, "cpu_count", lambda: 16)
    collection = _three_parent_collection()
    state = _runtime_state(collection, tmp_path)
    monkeypatch.setattr(loo_mod, "_validate_state", lambda *_a, **_k: state)
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
    collection = _three_parent_collection()
    state = _runtime_state(collection, tmp_path)
    monkeypatch.setattr(loo_mod, "_validate_state", lambda *_a, **_k: state)
    seen = _patch_dispatch(monkeypatch)
    first = loo_mod._fold_from_record(state.folds[0])
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
    collection = _three_parent_collection()
    state = _runtime_state(collection, tmp_path)
    monkeypatch.setattr(loo_mod, "_validate_state", lambda *_a, **_k: state)
    seen = _patch_dispatch(monkeypatch)
    fold = loo_mod._fold_from_record(state.folds[0])
    _write_stub_fold(tmp_path, fold)
    (tmp_path / "folds" / fold.slug / "config.json").write_text(
        json.dumps(
            {
                "status": "complete",
                **loo_mod._fold_runtime_metadata(state, state.folds[0]),
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
    monkeypatch.setattr(
        cli, "_dispatch_producer", lambda *_a: events.append("producer") or 0
    )
    monkeypatch.setattr(
        cli,
        "run_loo_cv",
        lambda **kwargs: (
            events.append("workers")
            or LOOResult(
                fold_dirs=(out_dir / "folds" / "p1",),
                parallel_folds=1,
                devices_per_fold=1,
                summary_csv_path=out_dir / "loo_summary.csv",
                aggregate_json_path=out_dir / "loo_aggregate.json",
                aggregate={"n_folds": 1},
            )
        ),
    )

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


def test_loo_cli_rejects_config_and_resume_together(tmp_path):
    assert cli.main(["loo", "--config", "x.json", "--resume", str(tmp_path)]) == 1
