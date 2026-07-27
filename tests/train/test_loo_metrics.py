"""Tests for the post-hoc CV-metric API in ``bp_train.loo_metrics``.

These tests synthesise tiny LOO output dirs on disk (sidecars +
predictions.csv files) and a matching prepared collection, then call the
public ``compute_per_process_metrics`` / ``compute_aggregated_metrics``
functions. No real training, no JAX in the metric path.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
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
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)

from bp_train.loo_metrics import (
    DEFAULT_METRICS,
    _read_fold_sidecar,
    compute_aggregated_metrics,
    compute_per_process_metrics,
    format_incompleteness_banner,
)


def test_read_fold_sidecar_accepts_whole_line_comments(tmp_path: Path):
    (tmp_path / "trained_wrapper.meta.json").write_text(
        '// fold identity\n{"holdout_processes": ["p1"]}',
        encoding="utf-8",
    )

    assert _read_fold_sidecar(tmp_path)["holdout_processes"] == ["p1"]


# ---------------------------------------------------------------------------
# Fixtures: collections, prediction-csv writers, sidecar writers
# ---------------------------------------------------------------------------


def _make_process(
    name: str,
    *,
    biomass_times=(0.0, 1.0, 2.0),
    biomass_values=(1.0, 2.0, 3.0),
    feed_times=(0.0, 1.0, 2.0),
    feed_values=(0.0, 0.1, 0.2),
    include_feed: bool = True,
) -> BioProcess:
    components = {
        "biomass": ReactorMediumComponent(
            name="biomass",
            unit="g/L",
            concentration=TimeSeries(
                times=jnp.asarray(biomass_times),
                values=jnp.asarray(biomass_values),
            ),
        ),
    }
    volume_changes = {}
    if include_feed:
        volume_changes["base_feed"] = FeedVolumeChange(
            name="base_feed",
            unit="L",
            is_controlled=False,
            is_continuous=True,
            values=TimeSeries(
                times=jnp.asarray(feed_times),
                values=jnp.asarray(feed_values),
            ),
            feed_medium=FeedMedium(
                name="base",
                density=1.0,
                density_unit="kg/L",
                components={
                    "biomass": FeedMediumComponent(
                        name="biomass",
                        unit="g/L",
                        concentration=StaticVariable(value=0.0),
                        is_controlled=False,
                    )
                },
            ),
        )
    return BioProcess(
        metadata=BioProcessMetadata(name=name, process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes=volume_changes,
        ),
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components=components,
        ),
        process_variables={},
    )


def _two_process_collection() -> BioProcessCollection:
    return BioProcessCollection(
        processes={
            "p1": _make_process(
                "p1",
                biomass_values=(1.0, 2.0, 3.0),
                feed_values=(0.0, 0.1, 0.2),
            ),
            "p2": _make_process(
                "p2",
                biomass_values=(0.5, 1.0, 1.5),
                feed_values=(0.0, 0.05, 0.1),
            ),
        },
        metadata={},
    )


def _three_process_collection() -> BioProcessCollection:
    return BioProcessCollection(
        processes={
            "p1": _make_process("p1", biomass_values=(1.0, 2.0, 3.0)),
            "p2": _make_process("p2", biomass_values=(0.5, 1.0, 1.5)),
            "p3": _make_process("p3", biomass_values=(0.8, 1.6, 2.4)),
        },
        metadata={},
    )


def _write_fold(
    fold_dir: Path,
    *,
    fold_idx: int,
    holdout_parent: str,
    holdout_group: tuple[str, ...],
    targets: tuple[str, ...],
    process_predictions: dict[str, dict[str, np.ndarray]],
    training_processes: tuple[str, ...] | None = None,
) -> None:
    """Write a synthetic fold dir with sidecar + predictions.csv.

    process_predictions: {process_name: {column_name: dense_array}}
                         column_name e.g. "c_biomass" or "B_base_feed_cum".
                         All arrays must share the same "t" axis.
    """
    fold_dir.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "fold_idx": fold_idx,
        "holdout_parent": holdout_parent,
        "holdout_group": list(holdout_group),
        "training_processes": list(training_processes or ()),
        "targets": list(targets),
    }
    (fold_dir / "trained_wrapper.meta.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    rows = []
    for proc, cols in process_predictions.items():
        t = cols["t"]
        n = len(t)
        for i in range(n):
            row = {"process": proc, "t": float(t[i])}
            for col_name, arr in cols.items():
                if col_name == "t":
                    continue
                row[col_name] = float(arr[i])
            rows.append(row)
    pd.DataFrame(rows).to_csv(fold_dir / "predictions.csv", index=False)


def _build_loo_dir(
    root: Path,
    *,
    folds: list[dict],
    name: str = "output_loo",
) -> Path:
    out_dir = root / name
    folds_root = out_dir / "folds"
    folds_root.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        _write_fold(folds_root / fold["holdout_parent"], **fold)
    return out_dir


# Predictions that hit the truth exactly at the measurement timestamps.
def _perfect_pred_for(process: BioProcess) -> dict[str, np.ndarray]:
    biomass = process.reactor_medium.components["biomass"].concentration
    times = np.asarray(biomass.times, dtype=float)
    biomass_vals = np.asarray(biomass.values, dtype=float)
    base_feed = process.volume.volume_changes.get("base_feed")
    cols = {
        "t": times,
        "c_biomass": biomass_vals,
    }
    if base_feed is not None:
        cols["B_base_feed_cum"] = np.asarray(base_feed.values.values, dtype=float)
    return cols


def _perfect_loo_dir(
    root: Path, name: str = "output_loo"
) -> tuple[Path, BioProcessCollection]:
    collection = _two_process_collection()
    folds = []
    for fold_idx, parent in enumerate(("p1", "p2")):
        train_proc = next(p for p in collection.processes if p != parent)
        # Predictions for both processes are perfect.
        process_preds = {
            proc_name: _perfect_pred_for(collection.processes[proc_name])
            for proc_name in collection.processes
        }
        folds.append(
            {
                "fold_idx": fold_idx,
                "holdout_parent": parent,
                "holdout_group": (parent,),
                "training_processes": (train_proc,),
                "targets": ("biomass",),
                "process_predictions": process_preds,
            }
        )
    return _build_loo_dir(root, folds=folds, name=name), collection


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_per_process_single_dir_default_metrics(tmp_path):
    out_dir, collection = _perfect_loo_dir(tmp_path)
    df = compute_per_process_metrics(out_dir, collection)

    # Two folds × one holdout × one reactor-component target ("biomass") +
    # one volume-change target ("base_feed") = 4 rows.
    assert len(df) == 4
    expected_cols = {
        "run_dir",
        "fold_idx",
        "holdout_parent",
        "holdout_process",
        "target_kind",
        "target_name",
        "n_measured",
        "r2",
        "nmae",
        "mae",
        "rmse",
    }
    assert expected_cols.issubset(df.columns)
    # Perfect predictions: r2 == 1, others == 0.
    assert df["r2"].to_numpy() == pytest.approx(1.0)
    assert df["nmae"].to_numpy() == pytest.approx(0.0)
    assert df["mae"].to_numpy() == pytest.approx(0.0)
    assert df["rmse"].to_numpy() == pytest.approx(0.0)


def test_aggregated_pooled_matches_handcomputed(tmp_path):
    """Pooled metric = metric on concatenated y_true/y_pred per target."""
    out_dir, collection = _perfect_loo_dir(tmp_path)
    df = compute_aggregated_metrics(out_dir, collection)

    # One row per (target_kind, target_name): biomass + base_feed.
    assert len(df) == 2
    rows_by_target = {row["target_name"]: row for _, row in df.iterrows()}
    assert set(rows_by_target) == {"biomass", "base_feed"}
    # biomass: 3 measurement points × 2 holdout processes (p1, p2) = 6 obs.
    assert int(rows_by_target["biomass"]["n_obs"]) == 6
    assert int(rows_by_target["biomass"]["n_processes"]) == 2
    assert rows_by_target["biomass"]["r2"] == pytest.approx(1.0)
    assert rows_by_target["biomass"]["mae"] == pytest.approx(0.0)


def test_pooled_vs_per_process_differ_when_unbalanced(tmp_path):
    """Per-process mean of RMSE differs from pooled RMSE on uneven processes."""
    # p1 has 2 measurements, p2 has 20. Make their per-process errors very
    # different so the unweighted per-process mean ≠ point-pooled RMSE.
    p1 = _make_process(
        "p1",
        biomass_times=(0.0, 2.0),
        biomass_values=(1.0, 1.0),
        include_feed=False,
    )
    p2 = _make_process(
        "p2",
        biomass_times=tuple(np.linspace(0.0, 2.0, 20).tolist()),
        biomass_values=tuple(np.full(20, 5.0).tolist()),
        include_feed=False,
    )
    collection = BioProcessCollection(processes={"p1": p1, "p2": p2}, metadata={})

    folds = []
    for fold_idx, parent in enumerate(("p1", "p2")):
        train_proc = next(p for p in collection.processes if p != parent)
        # p1 prediction is wildly wrong; p2 prediction is perfect.
        p1_truth = np.asarray(
            collection.processes["p1"]
            .reactor_medium.components["biomass"]
            .concentration.values,
            dtype=float,
        )
        p2_truth = np.asarray(
            collection.processes["p2"]
            .reactor_medium.components["biomass"]
            .concentration.values,
            dtype=float,
        )
        p1_times = np.asarray(
            collection.processes["p1"]
            .reactor_medium.components["biomass"]
            .concentration.times,
            dtype=float,
        )
        p2_times = np.asarray(
            collection.processes["p2"]
            .reactor_medium.components["biomass"]
            .concentration.times,
            dtype=float,
        )
        process_preds = {
            "p1": {
                "t": p1_times,
                "c_biomass": p1_truth + 10.0,  # error of 10 at every point
            },
            "p2": {
                "t": p2_times,
                "c_biomass": p2_truth,  # perfect
            },
        }
        folds.append(
            {
                "fold_idx": fold_idx,
                "holdout_parent": parent,
                "holdout_group": (parent,),
                "training_processes": (train_proc,),
                "targets": ("biomass",),
                "process_predictions": process_preds,
            }
        )
    out_dir = _build_loo_dir(tmp_path, folds=folds)

    pp = compute_per_process_metrics(out_dir, collection)
    agg = compute_aggregated_metrics(out_dir, collection)

    # Per-process RMSE: p1 row = 10, p2 row = 0; mean ≈ 5.
    pp_biomass = pp[pp["target_name"] == "biomass"]
    assert pp_biomass["rmse"].mean() == pytest.approx(5.0)

    # Pooled RMSE: 2 errors of 10, 20 errors of 0 → sqrt(2*100/22) ≈ 3.015.
    agg_biomass = agg[agg["target_name"] == "biomass"].iloc[0]
    pooled_rmse = math.sqrt(2 * 100.0 / 22)
    assert agg_biomass["rmse"] == pytest.approx(pooled_rmse, rel=1e-9)
    assert pp_biomass["rmse"].mean() != pytest.approx(pooled_rmse, rel=1e-3)


def test_extra_metrics_appears_in_both_functions(tmp_path):
    out_dir, collection = _perfect_loo_dir(tmp_path)
    extra = {"max_abs_err": lambda yt, yp: float(np.max(np.abs(yt - yp)))}

    pp = compute_per_process_metrics(out_dir, collection, extra_metrics=extra)
    agg = compute_aggregated_metrics(out_dir, collection, extra_metrics=extra)

    assert "max_abs_err" in pp.columns
    assert "max_abs_err" in agg.columns
    assert pp["max_abs_err"].to_numpy() == pytest.approx(0.0)


def test_metrics_replace_drops_defaults(tmp_path):
    out_dir, collection = _perfect_loo_dir(tmp_path)
    just_r2 = {"r2": DEFAULT_METRICS["r2"]}

    pp = compute_per_process_metrics(out_dir, collection, metrics=just_r2)
    agg = compute_aggregated_metrics(out_dir, collection, metrics=just_r2)

    assert "r2" in pp.columns
    for unused in ("nmae", "mae", "rmse"):
        assert unused not in pp.columns
        assert unused not in agg.columns


def test_includes_volume_changes_by_default(tmp_path):
    out_dir, collection = _perfect_loo_dir(tmp_path)
    df = compute_per_process_metrics(out_dir, collection)
    kinds = set(df["target_kind"].tolist())
    assert "reactor" in kinds
    assert "volume_change" in kinds
    vc_rows = df[df["target_kind"] == "volume_change"]
    assert set(vc_rows["target_name"].tolist()) == {"base_feed"}


def test_include_volume_changes_false(tmp_path):
    out_dir, collection = _perfect_loo_dir(tmp_path)
    df = compute_per_process_metrics(out_dir, collection, include_volume_changes=False)
    assert "volume_change" not in set(df["target_kind"].tolist())


def test_skips_train_by_default(tmp_path):
    out_dir, collection = _perfect_loo_dir(tmp_path)
    df = compute_per_process_metrics(out_dir, collection)
    # Only holdout-side rows: each fold's holdout_process == its holdout_parent.
    assert (df["holdout_process"] == df["holdout_parent"]).all()


def test_include_train_adds_rows(tmp_path):
    out_dir, collection = _perfect_loo_dir(tmp_path)
    df = compute_per_process_metrics(out_dir, collection, include_train=True)
    # Each fold has 1 holdout + 1 train process, both scored on biomass+feed.
    # 2 folds × 2 processes × 2 targets = 8 rows.
    assert len(df) == 8
    assert (df["holdout_process"] != df["holdout_parent"]).any()


def test_handles_missing_measurements(tmp_path):
    """Target declared in sidecar but absent on a process → row skipped."""
    p1 = _make_process("p1", biomass_values=(1.0, 2.0, 3.0))
    p2 = _make_process("p2", biomass_values=(0.5, 1.0, 1.5))
    # Strip the biomass component from p2 to simulate a missing measurement.
    p2.reactor_medium.components.pop("biomass")
    collection = BioProcessCollection(processes={"p1": p1, "p2": p2}, metadata={})

    folds = []
    for fold_idx, parent in enumerate(("p1", "p2")):
        train_proc = next(p for p in collection.processes if p != parent)
        process_preds = {
            "p1": _perfect_pred_for(p1),
            "p2": {
                "t": np.array([0.0, 1.0, 2.0]),
                "c_biomass": np.array([0.5, 1.0, 1.5]),
                "B_base_feed_cum": np.array([0.0, 0.05, 0.1]),
            },
        }
        folds.append(
            {
                "fold_idx": fold_idx,
                "holdout_parent": parent,
                "holdout_group": (parent,),
                "training_processes": (train_proc,),
                "targets": ("biomass",),
                "process_predictions": process_preds,
            }
        )
    out_dir = _build_loo_dir(tmp_path, folds=folds)

    df = compute_per_process_metrics(out_dir, collection)
    # p1 holdout: biomass + base_feed = 2 rows.
    # p2 holdout: biomass missing, only base_feed = 1 row.
    assert len(df) == 3
    p2_rows = df[df["holdout_process"] == "p2"]
    assert set(p2_rows["target_name"].tolist()) == {"base_feed"}


def test_rejects_reserved_metric_name(tmp_path):
    out_dir, collection = _perfect_loo_dir(tmp_path)
    with pytest.raises(ValueError, match="reserved column"):
        compute_per_process_metrics(
            out_dir, collection, extra_metrics={"n_measured": lambda yt, yp: 1.0}
        )


def test_user_metric_error_yields_nan_with_warning(tmp_path, caplog):
    out_dir, collection = _perfect_loo_dir(tmp_path)

    def boom(yt, yp):
        raise RuntimeError("intentional")

    with caplog.at_level(logging.WARNING, logger="bp_train.loo_metrics"):
        df = compute_per_process_metrics(
            out_dir, collection, extra_metrics={"boom": boom}
        )
    assert df["boom"].isna().all()
    assert any("boom" in rec.message for rec in caplog.records)
    # Defaults still work.
    assert df["r2"].to_numpy() == pytest.approx(1.0)


def test_incomplete_loo_attrs_and_warning(tmp_path, caplog):
    """Output dir with fewer folds than expected sets all_runs_complete=False."""
    collection = _three_process_collection()
    # Only p1 fold present; p2, p3 missing.
    process_preds = {
        proc_name: _perfect_pred_for(collection.processes[proc_name])
        for proc_name in collection.processes
    }
    folds = [
        {
            "fold_idx": 0,
            "holdout_parent": "p1",
            "holdout_group": ("p1",),
            "training_processes": ("p2", "p3"),
            "targets": ("biomass",),
            "process_predictions": process_preds,
        }
    ]
    out_dir = _build_loo_dir(tmp_path, folds=folds)

    with caplog.at_level(logging.WARNING, logger="bp_train.loo_metrics"):
        df = compute_per_process_metrics(out_dir, collection)

    assert df.attrs["all_runs_complete"] is False
    assert len(df.attrs["incomplete_runs"]) == 1
    entry = df.attrs["incomplete_runs"][0]
    assert entry["n_actual"] == 1
    assert entry["n_expected"] == 3
    assert set(entry["missing_holdout_processes"]) == {"p2", "p3"}
    assert any("incomplete LOO" in rec.message for rec in caplog.records)
    banner = format_incompleteness_banner(df)
    assert banner is not None
    assert "INCOMPLETE LOO" in banner


def test_complete_loo_attrs_clean(tmp_path):
    out_dir, collection = _perfect_loo_dir(tmp_path)
    df = compute_per_process_metrics(out_dir, collection)
    assert df.attrs["all_runs_complete"] is True
    assert df.attrs["incomplete_runs"] == []
    assert format_incompleteness_banner(df) is None


def test_equal_comparison_restricts_to_intersection(tmp_path):
    """Two dirs, A:{p1,p2,p3}, B:{p2,p3,p4} → keep {p2,p3}."""
    p1 = _make_process("p1", biomass_values=(1.0, 2.0, 3.0))
    p2 = _make_process("p2", biomass_values=(0.5, 1.0, 1.5))
    p3 = _make_process("p3", biomass_values=(0.8, 1.6, 2.4))
    p4 = _make_process("p4", biomass_values=(0.2, 0.4, 0.6))
    collection = BioProcessCollection(
        processes={"p1": p1, "p2": p2, "p3": p3, "p4": p4}, metadata={}
    )

    def make_dir(name: str, holdouts: tuple[str, ...]) -> Path:
        folds = []
        for fold_idx, parent in enumerate(holdouts):
            others = [p for p in collection.processes if p != parent]
            process_preds = {
                proc_name: _perfect_pred_for(collection.processes[proc_name])
                for proc_name in collection.processes
            }
            folds.append(
                {
                    "fold_idx": fold_idx,
                    "holdout_parent": parent,
                    "holdout_group": (parent,),
                    "training_processes": tuple(others),
                    "targets": ("biomass",),
                    "process_predictions": process_preds,
                }
            )
        return _build_loo_dir(tmp_path, folds=folds, name=name)

    dir_a = make_dir("run_a", ("p1", "p2", "p3"))
    dir_b = make_dir("run_b", ("p2", "p3", "p4"))

    df = compute_per_process_metrics([dir_a, dir_b], collection, equal_comparison=True)
    holdouts = set(df["holdout_parent"].tolist())
    assert holdouts == {"p2", "p3"}
    assert df.attrs["intersection_holdout_processes"] == ("p2", "p3")
    drops = df.attrs["dropped_for_equal_comparison"]
    assert drops[str(dir_a)] == ["p1"]
    assert drops[str(dir_b)] == ["p4"]


def test_equal_comparison_false_keeps_per_dir_full(tmp_path):
    p1 = _make_process("p1", biomass_values=(1.0, 2.0, 3.0))
    p2 = _make_process("p2", biomass_values=(0.5, 1.0, 1.5))
    p3 = _make_process("p3", biomass_values=(0.8, 1.6, 2.4))
    p4 = _make_process("p4", biomass_values=(0.2, 0.4, 0.6))
    collection = BioProcessCollection(
        processes={"p1": p1, "p2": p2, "p3": p3, "p4": p4}, metadata={}
    )

    def make_dir(name: str, holdouts: tuple[str, ...]) -> Path:
        folds = []
        for fold_idx, parent in enumerate(holdouts):
            others = [p for p in collection.processes if p != parent]
            process_preds = {
                proc_name: _perfect_pred_for(collection.processes[proc_name])
                for proc_name in collection.processes
            }
            folds.append(
                {
                    "fold_idx": fold_idx,
                    "holdout_parent": parent,
                    "holdout_group": (parent,),
                    "training_processes": tuple(others),
                    "targets": ("biomass",),
                    "process_predictions": process_preds,
                }
            )
        return _build_loo_dir(tmp_path, folds=folds, name=name)

    dir_a = make_dir("run_a", ("p1", "p2", "p3"))
    dir_b = make_dir("run_b", ("p2", "p3", "p4"))

    df = compute_per_process_metrics([dir_a, dir_b], collection, equal_comparison=False)
    # 4 distinct holdout parents observed across the two dirs.
    holdouts = set(df["holdout_parent"].tolist())
    assert holdouts == {"p1", "p2", "p3", "p4"}
    assert df.attrs["intersection_holdout_processes"] == ()
    assert df.attrs["dropped_for_equal_comparison"] == {}


def test_equal_comparison_noop_for_single_dir(tmp_path):
    out_dir, collection = _perfect_loo_dir(tmp_path)
    df = compute_per_process_metrics(out_dir, collection, equal_comparison=True)
    assert df.attrs["dropped_for_equal_comparison"] == {}
    # Single-dir intersection is a no-op; provenance reflects no trimming.
    assert df.attrs["intersection_holdout_processes"] == ()


def test_require_measurement_nodes_fail_fast():
    """The scorer must refuse to silently interpolate: a measurement time that is
    not an exact node of the prediction grid raises (guards the jump-blind bug).
    """
    from bp_train.loo_metrics import _require_measurement_nodes

    uniform = np.linspace(0.0, 2.0, 11)  # no node at 0.7
    with pytest.raises(ValueError, match="no grid node"):
        _require_measurement_nodes(
            uniform, np.array([0.0, 0.7, 2.0]), process="p1", target="biomass"
        )
    # a measurement-inclusive grid passes (float32 round-trip tolerated)
    inclusive = np.unique(np.concatenate([uniform, [0.7]]))
    _require_measurement_nodes(
        inclusive,
        np.array([0.0, 0.7, 2.0], dtype=np.float32).astype(float),
        process="p1",
        target="biomass",
    )
