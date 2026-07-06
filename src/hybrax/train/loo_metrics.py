"""Post-hoc LOO-CV goodness-of-fit metrics from per-fold predictions.csv.

Hijacks the dense ``predictions.csv`` written by every fold (and by every
``bp-train train`` run) so R², NMAE, MAE, and RMSE can be computed against
the original measurements *without* reloading any model or rerunning the
solver. Inputs:

- ``<loo_output_dir>/folds/<parent>/predictions.csv`` — dense simulated
  trajectory of each process for that fold.
- ``<loo_output_dir>/folds/<parent>/trained_wrapper.meta.json`` — sidecar
  identifying which processes formed the holdout group for that fold.
- ``prepared.json`` — source of truth for measurement timestamps and
  measured values per target.

Predictions are linearly interpolated to measurement timestamps before the
metrics are computed.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from bp_format.dataclasses import (
    BioProcess,
    BioProcessCollection,
    FeedVolumeChange,
    ReactorMediumComponent,
    SampleVolumeChange,
    StaticVariable,
    TimeSeries,
)
from bp_format.serialization import load_process_collection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_pred - y_true
    ss_res = float(np.sum(diff * diff))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_pred - y_true
    return float(np.sqrt(np.mean(diff * diff)))


def _nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    abs_mean = float(np.mean(np.abs(y_true)))
    if abs_mean <= 0:
        return float("nan")
    return _mae(y_true, y_pred) / abs_mean


DEFAULT_METRICS: dict[str, MetricFn] = {
    "r2": _r2,
    "nmae": _nmae,
    "mae": _mae,
    "rmse": _rmse,
}

# Identifier columns reserved on the result DataFrames; metric names that
# collide with these are rejected at entry to keep CSV/groupby logic stable.
_RESERVED_COLUMN_NAMES = frozenset(
    {
        "run_dir",
        "fold_idx",
        "holdout_parent",
        "holdout_process",
        "target_kind",
        "target_name",
        "n_measured",
        "n_obs",
        "n_processes",
    }
)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LOOMetricsResult:
    """Outputs of :func:`compute_loo_metrics`."""

    per_fold_target: pd.DataFrame  # columns: fold_idx, holdout_parent, holdout_process, target, n_measured, r2, nmae, mae, rmse
    aggregate: dict[str, Any]      # per-target mean/std/median across folds + overall summary
    metrics_csv_path: Path | None
    aggregate_json_path: Path | None


# ---------------------------------------------------------------------------
# Per-process measurement extraction
# ---------------------------------------------------------------------------


def _extract_measurements(
    process: BioProcess,
    target_names: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``{target_name: (times, values)}`` for measured reactor components.

    Skips :class:`StaticVariable` entries and any target that is missing
    from the process. Missing targets simply do not appear in the result.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if process.reactor_medium is None:
        return out
    components = process.reactor_medium.components or {}
    for name in target_names:
        comp = components.get(name)
        if not isinstance(comp, ReactorMediumComponent):
            continue
        conc = comp.concentration
        if isinstance(conc, StaticVariable):
            continue
        if not isinstance(conc, TimeSeries):
            continue
        times = np.asarray(conc.times, dtype=np.float64)
        values = np.asarray(conc.values, dtype=np.float64)
        if times.size == 0:
            continue
        out[name] = (times, values)
    return out


def _extract_volume_change_measurements(
    process: BioProcess,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``{volume_change_name: (times, values)}`` for measured changes.

    Pairs ``B_<name>_cum`` predictions with the cumulative
    ``vc.values`` time series stored in ``prepared.json``. Skips static
    or empty entries (no truth data to pair against).
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if process.volume is None or not process.volume.volume_changes:
        return out
    for name, vc in process.volume.volume_changes.items():
        if not isinstance(vc, (FeedVolumeChange, SampleVolumeChange)):
            continue
        values = vc.values
        if values is None:
            continue
        if not isinstance(values, TimeSeries):
            continue
        times = np.asarray(values.times, dtype=np.float64)
        vals = np.asarray(values.values, dtype=np.float64)
        if times.size == 0:
            continue
        out[name] = (times, vals)
    return out


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute R², NMAE, MAE, RMSE on aligned 1-D arrays.

    NMAE normalises by ``mean(|y_true|)`` to keep it dimensionless and
    well-defined when the mean of ``y_true`` itself is near zero (e.g.
    centred residuals).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    n = int(y_true.size)
    if n == 0:
        return {
            "n_measured": 0,
            "r2": float("nan"),
            "nmae": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
        }
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    abs_mean = float(np.mean(np.abs(y_true)))
    nmae = mae / abs_mean if abs_mean > 0 else float("nan")
    ss_res = float(np.sum(diff * diff))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        # constant ground truth: R² is undefined; report NaN rather than ±inf.
        r2 = float("nan")
    return {
        "n_measured": n,
        "r2": r2,
        "nmae": nmae,
        "mae": mae,
        "rmse": rmse,
    }


# ---------------------------------------------------------------------------
# Per-fold and per-process evaluation
# ---------------------------------------------------------------------------


def _require_measurement_nodes(
    pred_t: np.ndarray,
    meas_t: np.ndarray,
    *,
    process: str,
    target: str,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> None:
    """Fail-fast: every measurement time must be an exact node of the (sorted)
    prediction grid ``pred_t``.

    Otherwise ``np.interp`` would silently draw a straight ramp across any
    bolus/feed discontinuity that falls between two grid points. bp-train's export
    splices the measurement grid into predictions.csv, so a violation means the file
    was produced by an older bp-train and must be regenerated.
    """
    pred_t = np.asarray(pred_t, dtype=float)
    meas_t = np.asarray(meas_t, dtype=float)
    if meas_t.size == 0:
        return
    if pred_t.size == 0:
        raise ValueError(
            f"empty prediction grid for process {process!r} target {target!r}"
        )
    idx = np.clip(np.searchsorted(pred_t, meas_t), 0, pred_t.size - 1)
    left = np.clip(idx - 1, 0, pred_t.size - 1)
    nearest = np.where(
        np.abs(pred_t[idx] - meas_t) <= np.abs(pred_t[left] - meas_t),
        pred_t[idx],
        pred_t[left],
    )
    off = ~np.isclose(nearest, meas_t, rtol=rtol, atol=atol)
    if np.any(off):
        raise ValueError(
            f"predictions.csv has no grid node at measurement time(s) "
            f"{meas_t[off].tolist()} for process {process!r} target {target!r}; "
            f"regenerate predictions.csv with the current bp-train (its export grid "
            f"must include the measurement times)."
        )


def _prediction_unscoreable(
    pred_t: np.ndarray, pred_y: np.ndarray, meas_t: np.ndarray
) -> bool:
    """A diverged / truncated prediction that must be scored NaN — NOT interpolated or node-guarded.

    True iff the prediction has any non-finite value, or a measurement time falls outside the
    prediction grid's range (a solve that blew up and stopped early). Such a fold cannot be scored,
    so it becomes NaN and is skipped (never a crash, and never a clamped-``np.interp`` endpoint that
    would read as a misleading finite score). A finite, full-range prediction returns ``False`` so
    :func:`_require_measurement_nodes` still fails loudly on a genuinely-old node-omitting file.
    """
    pred_t = np.asarray(pred_t, dtype=float)
    pred_y = np.asarray(pred_y, dtype=float)
    meas_t = np.asarray(meas_t, dtype=float)
    if pred_t.size == 0 or not bool(np.all(np.isfinite(pred_y))):
        return True
    if meas_t.size and (meas_t.min() < pred_t.min() or meas_t.max() > pred_t.max()):
        return True
    return False


def _evaluate_predictions_for_process(
    *,
    pred_t: np.ndarray,
    pred_columns: dict[str, np.ndarray],
    measurements: dict[str, tuple[np.ndarray, np.ndarray]],
    process: str = "?",
) -> dict[str, dict[str, float]]:
    """Read predictions at measurement times (exact nodes) and compute metrics.

    A diverged/truncated prediction (see :func:`_prediction_unscoreable`) scores NaN and is skipped.
    """
    out: dict[str, dict[str, float]] = {}
    empty = np.asarray([], dtype=float)
    for target, (meas_t, meas_y) in measurements.items():
        col_name = f"c_{target}"
        pred_y = pred_columns.get(col_name)
        if pred_y is None:
            continue
        if _prediction_unscoreable(pred_t, pred_y, meas_t):
            out[target] = _compute_metrics(empty, empty)  # all-NaN, n_measured=0
            continue
        _require_measurement_nodes(pred_t, meas_t, process=process, target=target)
        pred_at_meas = np.interp(meas_t, pred_t, pred_y)
        out[target] = _compute_metrics(meas_y, pred_at_meas)
    return out


def _read_fold_sidecar(fold_dir: Path) -> dict[str, Any]:
    sidecar = fold_dir / "trained_wrapper.meta.json"
    if not sidecar.exists():
        raise FileNotFoundError(
            f"missing sidecar at {sidecar}; cannot identify holdout group"
        )
    return json.loads(sidecar.read_text(encoding="utf-8"))


def _read_predictions_csv(fold_dir: Path) -> pd.DataFrame:
    pred_path = fold_dir / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"missing predictions.csv at {pred_path}"
        )
    return pd.read_csv(pred_path)


def _numeric_pred_columns(sub: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return ``{col: float64 array}`` for every numeric column in ``sub``.

    Skips non-numeric columns like ``process`` so callers can do
    ``pred_columns.get("c_biomass")`` without tripping over string casts.
    """
    out: dict[str, np.ndarray] = {}
    for col in sub.columns:
        if pd.api.types.is_numeric_dtype(sub[col]):
            out[col] = sub[col].to_numpy(dtype=np.float64)
    return out


def _resolve_target_names(
    sidecar: dict[str, Any],
    pred_df: pd.DataFrame,
    target_override: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if target_override is not None:
        return tuple(target_override)
    sidecar_targets = sidecar.get("targets")
    if sidecar_targets:
        return tuple(sidecar_targets)
    # Fall back to any "c_<name>" column in predictions.csv.
    return tuple(
        col[len("c_"):]
        for col in pred_df.columns
        if col.startswith("c_") and col != "c_modeled"
    )


def _iter_fold_dirs(loo_output_dir: Path) -> list[Path]:
    folds_root = loo_output_dir / "folds"
    if not folds_root.exists():
        raise FileNotFoundError(
            f"no 'folds/' directory under {loo_output_dir}; not a LOO output dir"
        )
    return sorted(p for p in folds_root.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_loo_metrics(
    loo_output_dir: str | Path,
    prepared_json: str | Path | BioProcessCollection,
    *,
    target_names: Iterable[str] | None = None,
    write_outputs: bool = True,
) -> LOOMetricsResult:
    """Compute per-fold/per-process/per-target R², NMAE, MAE, RMSE.

    Args:
        loo_output_dir: directory containing ``folds/<parent>/`` produced
            by ``bp-train loo``.
        prepared_json: path to ``prepared.json`` *or* an already-loaded
            :class:`BioProcessCollection`. Source of measured values.
        target_names: explicit subset of targets to score. Defaults to the
            sidecar's ``targets`` field, then to any ``c_<name>`` column
            present in ``predictions.csv``.
        write_outputs: when ``True``, writes ``loo_metrics.csv`` and
            ``loo_metrics_aggregate.json`` next to ``folds/``.

    Returns:
        :class:`LOOMetricsResult` with the per-(fold, holdout_process,
        target) DataFrame and a per-target aggregate dict.
    """
    loo_dir = Path(loo_output_dir)
    if isinstance(prepared_json, BioProcessCollection):
        collection = prepared_json
    else:
        collection = load_process_collection(Path(prepared_json))

    target_override = tuple(target_names) if target_names is not None else None

    rows: list[dict[str, Any]] = []
    for fold_dir in _iter_fold_dirs(loo_dir):
        sidecar = _read_fold_sidecar(fold_dir)
        holdout_group = tuple(sidecar.get("test") or ())
        holdout_parent = fold_dir.name
        fold_idx = int(sidecar.get("fold_idx", -1))
        if not holdout_group:
            logger.warning(
                "fold '%s' has no test set in sidecar; skipping",
                fold_dir,
            )
            continue
        try:
            pred_df = _read_predictions_csv(fold_dir)
        except FileNotFoundError as exc:
            # A diverged fold whose forward produced no predictions.csv: skip it (its rows become
            # absent → NaN in the aggregate), never abort the whole dataset. Unexpected errors still raise.
            logger.warning("fold '%s': %s; skipping", holdout_parent, exc)
            continue
        targets = _resolve_target_names(sidecar, pred_df, target_override)
        for proc_name in holdout_group:
            process = collection.processes.get(proc_name)
            if process is None:
                logger.warning(
                    "holdout process '%s' not found in collection; "
                    "skipping fold %s",
                    proc_name,
                    holdout_parent,
                )
                continue
            sub = pred_df.loc[pred_df["process"] == proc_name]
            if sub.empty:
                logger.warning(
                    "predictions.csv for fold '%s' has no rows for "
                    "holdout process '%s'; skipping",
                    holdout_parent,
                    proc_name,
                )
                continue
            sub = sub.sort_values("t")
            pred_t = sub["t"].to_numpy(dtype=np.float64)
            pred_columns = _numeric_pred_columns(sub)
            measurements = _extract_measurements(process, targets)
            metrics_per_target = _evaluate_predictions_for_process(
                pred_t=pred_t,
                pred_columns=pred_columns,
                measurements=measurements,
                process=proc_name,
            )
            for target, m in metrics_per_target.items():
                rows.append(
                    {
                        "fold_idx": fold_idx,
                        "holdout_parent": holdout_parent,
                        "holdout_process": proc_name,
                        "target": target,
                        **m,
                    }
                )

    if not rows:
        raise RuntimeError(
            "no holdout metrics computed; check that LOO folds, predictions, "
            "and prepared collection are consistent"
        )

    per_fold_target = pd.DataFrame(rows)
    aggregate = _aggregate_metrics(per_fold_target)

    metrics_csv_path: Path | None = None
    aggregate_json_path: Path | None = None
    if write_outputs:
        metrics_csv_path = loo_dir / "loo_metrics.csv"
        aggregate_json_path = loo_dir / "loo_metrics_aggregate.json"
        per_fold_target.to_csv(metrics_csv_path, index=False)
        aggregate_json_path.write_text(
            json.dumps(aggregate, indent=2), encoding="utf-8"
        )
        logger.info(
            "LOO metrics written to %s; aggregate to %s",
            metrics_csv_path,
            aggregate_json_path,
        )

    return LOOMetricsResult(
        per_fold_target=per_fold_target,
        aggregate=aggregate,
        metrics_csv_path=metrics_csv_path,
        aggregate_json_path=aggregate_json_path,
    )


def compute_metrics_from_predictions_csv(
    predictions_csv: str | Path,
    prepared_json: str | Path | BioProcessCollection,
    *,
    target_names: Iterable[str] | None = None,
    process_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute the same metrics from a single ``predictions.csv``.

    Useful for non-LOO ``bp-train train`` runs (one model, every process)
    so the same metric definitions apply uniformly.
    """
    if isinstance(prepared_json, BioProcessCollection):
        collection = prepared_json
    else:
        collection = load_process_collection(Path(prepared_json))
    pred_df = pd.read_csv(predictions_csv)
    targets = _resolve_target_names({}, pred_df, tuple(target_names) if target_names else None)
    selected_processes = (
        tuple(process_names)
        if process_names is not None
        else tuple(pd.unique(pred_df["process"]))
    )
    rows: list[dict[str, Any]] = []
    for proc_name in selected_processes:
        process = collection.processes.get(proc_name)
        if process is None:
            continue
        sub = pred_df.loc[pred_df["process"] == proc_name].sort_values("t")
        if sub.empty:
            continue
        pred_t = sub["t"].to_numpy(dtype=np.float64)
        pred_columns = {col: sub[col].to_numpy(dtype=np.float64) for col in sub.columns}
        measurements = _extract_measurements(process, targets)
        metrics_per_target = _evaluate_predictions_for_process(
            pred_t=pred_t,
            pred_columns=pred_columns,
            measurements=measurements,
        )
        for target, m in metrics_per_target.items():
            rows.append({"process": proc_name, "target": target, **m})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_metrics(per_fold_target: pd.DataFrame) -> dict[str, Any]:
    """Aggregate per-(fold, target) metrics into per-target + overall stats."""
    out: dict[str, Any] = {
        "n_folds": int(per_fold_target["fold_idx"].nunique()),
        "n_holdout_processes": int(per_fold_target["holdout_process"].nunique()),
        "per_target": {},
    }
    for target, sub in per_fold_target.groupby("target", sort=False):
        target_stats: dict[str, Any] = {"n_observations": int(sub["n_measured"].sum())}
        for metric in ("r2", "nmae", "mae", "rmse"):
            vals = [float(v) for v in sub[metric].tolist() if _is_finite(v)]
            target_stats[f"{metric}_mean"] = _safe_mean(vals)
            target_stats[f"{metric}_std"] = _safe_std(vals)
            target_stats[f"{metric}_median"] = _safe_median(vals)
        out["per_target"][str(target)] = target_stats

    # Overall pooled aggregate (unweighted across all rows).
    for metric in ("r2", "nmae", "mae", "rmse"):
        vals = [float(v) for v in per_fold_target[metric].tolist() if _is_finite(v)]
        out[f"{metric}_mean"] = _safe_mean(vals)
        out[f"{metric}_std"] = _safe_std(vals)
        out[f"{metric}_median"] = _safe_median(vals)
    return out


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0 if len(values) == 1 else float("nan")
    return float(statistics.stdev(values))


def _safe_median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


# ---------------------------------------------------------------------------
# Shared loader: paired (y_true, y_pred) records across runs/folds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PairedRecord:
    run_dir: str
    fold_idx: int
    holdout_parent: str
    holdout_process: str
    target_kind: str  # "reactor" | "volume_change"
    target_name: str
    y_true: np.ndarray
    y_pred: np.ndarray


def _normalize_output_dirs(
    output_dirs: str | Path | Sequence[str | Path],
) -> list[Path]:
    if isinstance(output_dirs, (str, Path)):
        return [Path(output_dirs)]
    dirs = [Path(d) for d in output_dirs]
    if not dirs:
        raise ValueError("output_dirs must contain at least one path")
    return dirs


def _resolve_metric_registry(
    metrics: dict[str, MetricFn] | None,
    extra_metrics: dict[str, MetricFn] | None,
) -> dict[str, MetricFn]:
    base = dict(DEFAULT_METRICS) if metrics is None else dict(metrics)
    if extra_metrics:
        for name, fn in extra_metrics.items():
            if name in base:
                raise ValueError(
                    f"extra_metrics overrides existing metric '{name}'; "
                    "pass it via metrics=... if intentional"
                )
            base[name] = fn
    for name in base:
        if name in _RESERVED_COLUMN_NAMES:
            raise ValueError(
                f"metric name '{name}' collides with a reserved column "
                f"({sorted(_RESERVED_COLUMN_NAMES)})"
            )
        if not callable(base[name]):
            raise TypeError(f"metric '{name}' is not callable")
    return base


def _gather_paired_arrays(
    output_dirs: list[Path],
    collection: BioProcessCollection,
    *,
    target_names: tuple[str, ...] | None,
    include_volume_changes: bool,
    include_train: bool,
    equal_comparison: bool,
) -> tuple[list[_PairedRecord], dict[str, Any]]:
    """Walk every output dir, pair predictions against truth.

    Returns the list of paired records plus a provenance dict suitable
    for attachment to ``df.attrs``.
    """
    expected_holdouts = tuple(
        name
        for name, proc in collection.processes.items()
        if not _is_augmented_process(proc)
    )

    per_dir_records: dict[str, list[_PairedRecord]] = {}
    per_dir_actual_holdouts: dict[str, set[str]] = {}
    per_dir_n_actual: dict[str, int] = {}

    for out_dir in output_dirs:
        run_key = str(out_dir)
        records: list[_PairedRecord] = []
        actual_holdouts: set[str] = set()

        fold_dirs = _iter_fold_dirs(out_dir)
        per_dir_n_actual[run_key] = len(fold_dirs)

        for fold_dir in fold_dirs:
            sidecar = _read_fold_sidecar(fold_dir)
            holdout_group = tuple(sidecar.get("holdout_group") or ())
            if not holdout_group:
                logger.warning(
                    "fold '%s': no holdout_group in sidecar; skipping",
                    fold_dir,
                )
                continue
            holdout_parent = sidecar.get("holdout_parent") or fold_dir.name
            fold_idx = int(sidecar.get("fold_idx", -1))
            actual_holdouts.update(holdout_group)

            try:
                pred_df = _read_predictions_csv(fold_dir)
            except FileNotFoundError as exc:
                logger.warning("fold '%s': %s; skipping", fold_dir, exc)
                continue

            targets = _resolve_target_names(sidecar, pred_df, target_names)

            # Process set to score for this fold.
            if include_train:
                training_processes = tuple(sidecar.get("training_processes") or ())
                fold_processes = tuple(holdout_group) + training_processes
            else:
                fold_processes = tuple(holdout_group)

            for proc_name in fold_processes:
                process = collection.processes.get(proc_name)
                if process is None:
                    logger.warning(
                        "fold '%s': process '%s' not in collection; skipping",
                        holdout_parent,
                        proc_name,
                    )
                    continue
                sub = pred_df.loc[pred_df["process"] == proc_name]
                if sub.empty:
                    logger.warning(
                        "fold '%s': no predictions for process '%s'; skipping",
                        holdout_parent,
                        proc_name,
                    )
                    continue
                sub = sub.sort_values("t")
                pred_t = sub["t"].to_numpy(dtype=np.float64)
                pred_columns = _numeric_pred_columns(sub)

                # Reactor-component pairing (c_<target>).
                reactor_meas = _extract_measurements(process, targets)
                for tname, (meas_t, meas_y) in reactor_meas.items():
                    pred_y = pred_columns.get(f"c_{tname}")
                    if pred_y is None:
                        continue
                    if _prediction_unscoreable(pred_t, pred_y, meas_t):
                        pred_at_meas = np.full(np.asarray(meas_t).shape, np.nan)
                    else:
                        _require_measurement_nodes(
                            pred_t, meas_t, process=proc_name, target=tname
                        )
                        pred_at_meas = np.interp(meas_t, pred_t, pred_y)
                    records.append(
                        _PairedRecord(
                            run_dir=run_key,
                            fold_idx=fold_idx,
                            holdout_parent=holdout_parent,
                            holdout_process=proc_name,
                            target_kind="reactor",
                            target_name=tname,
                            y_true=meas_y,
                            y_pred=pred_at_meas,
                        )
                    )

                # Volume-change pairing (B_<name>_cum).
                if include_volume_changes:
                    vc_meas = _extract_volume_change_measurements(process)
                    for vc_name, (meas_t, meas_y) in vc_meas.items():
                        pred_col = f"B_{vc_name}_cum"
                        pred_y = pred_columns.get(pred_col)
                        if pred_y is None:
                            continue
                        if _prediction_unscoreable(pred_t, pred_y, meas_t):
                            pred_at_meas = np.full(np.asarray(meas_t).shape, np.nan)
                        else:
                            _require_measurement_nodes(
                                pred_t, meas_t, process=proc_name, target=vc_name
                            )
                            pred_at_meas = np.interp(meas_t, pred_t, pred_y)
                        records.append(
                            _PairedRecord(
                                run_dir=run_key,
                                fold_idx=fold_idx,
                                holdout_parent=holdout_parent,
                                holdout_process=proc_name,
                                target_kind="volume_change",
                                target_name=vc_name,
                                y_true=meas_y,
                                y_pred=pred_at_meas,
                            )
                        )

        per_dir_records[run_key] = records
        per_dir_actual_holdouts[run_key] = actual_holdouts

    # Per-dir completeness.
    expected_set = set(expected_holdouts)
    incomplete_runs: list[dict[str, Any]] = []
    for run_key, actuals in per_dir_actual_holdouts.items():
        missing = sorted(expected_set - actuals)
        if missing:
            incomplete_runs.append(
                {
                    "run_dir": run_key,
                    "n_expected": len(expected_set),
                    "n_actual": len(actuals),
                    "missing_holdout_processes": missing,
                }
            )
            logger.warning(
                "%s: incomplete LOO — %d/%d folds present; missing holdouts: %s",
                run_key,
                len(actuals),
                len(expected_set),
                missing,
            )

    # Cross-dir intersection (equal_comparison).
    intersection: tuple[str, ...] = ()
    intersection_set: set[str] = set()
    dropped_for_equal: dict[str, list[str]] = {}
    apply_intersection = equal_comparison and len(output_dirs) > 1
    if apply_intersection:
        sets = [per_dir_actual_holdouts[str(d)] for d in output_dirs]
        intersection_set = set.intersection(*sets) if sets else set()
        intersection = tuple(sorted(intersection_set))
        for run_key, actuals in per_dir_actual_holdouts.items():
            dropped = sorted(actuals - intersection_set)
            if dropped:
                dropped_for_equal[run_key] = dropped
        if dropped_for_equal:
            logger.info(
                "equal_comparison=True dropped %d holdouts: %s",
                sum(len(v) for v in dropped_for_equal.values()),
                dropped_for_equal,
            )

    # Apply intersection filter: drop any record whose holdout_parent is
    # not in the cross-dir intersection. Train-side rows for an *included*
    # parent stay (they're not part of the LOO comparison axis but are
    # consistent across runs).
    flat_records: list[_PairedRecord] = []
    for records in per_dir_records.values():
        for rec in records:
            if apply_intersection and rec.holdout_parent not in intersection_set:
                continue
            flat_records.append(rec)

    provenance = {
        "output_dirs": tuple(str(d) for d in output_dirs),
        "include_train": bool(include_train),
        "include_volume_changes": bool(include_volume_changes),
        "equal_comparison": bool(equal_comparison),
        "intersection_holdout_processes": intersection,
        "dropped_for_equal_comparison": dropped_for_equal,
        "all_runs_complete": len(incomplete_runs) == 0,
        "incomplete_runs": incomplete_runs,
    }
    return flat_records, provenance


def _is_augmented_process(proc: Any) -> bool:
    """Detect AugmentedBioProcess without forcing the import at module top.

    bp_format ships AugmentedBioProcess as a BioProcess subclass with a
    ``parent_process`` attribute; we only need the structural test.
    """
    try:
        from bp_format.dataclasses import AugmentedBioProcess
    except ImportError:
        return False
    return isinstance(proc, AugmentedBioProcess)


# ---------------------------------------------------------------------------
# Public API: per-process and aggregated metrics across LOO output dirs
# ---------------------------------------------------------------------------


def compute_per_process_metrics(
    output_dirs: str | Path | Sequence[str | Path],
    prepared_json: str | Path | BioProcessCollection,
    *,
    metrics: dict[str, MetricFn] | None = None,
    extra_metrics: dict[str, MetricFn] | None = None,
    target_names: Iterable[str] | None = None,
    include_volume_changes: bool = True,
    include_train: bool = False,
    equal_comparison: bool = True,
) -> pd.DataFrame:
    """Per-(run, fold, holdout_process, target) goodness-of-fit metrics.

    One row per process scored on its own ``(y_true, y_pred)`` pair —
    equal weight per process. Useful for spot-checking which fold
    drove the mean. For pooled-within-target metrics (equal weight per
    measurement point) use :func:`compute_aggregated_metrics`.

    Provenance — including any incompleteness flags or
    ``equal_comparison`` drops — is attached to ``df.attrs``.
    """
    dirs = _normalize_output_dirs(output_dirs)
    collection = _load_collection(prepared_json)
    metric_registry = _resolve_metric_registry(metrics, extra_metrics)
    target_override = tuple(target_names) if target_names is not None else None

    records, provenance = _gather_paired_arrays(
        dirs,
        collection,
        target_names=target_override,
        include_volume_changes=include_volume_changes,
        include_train=include_train,
        equal_comparison=equal_comparison,
    )

    rows: list[dict[str, Any]] = []
    for rec in records:
        y_true, y_pred = _filter_finite(rec.y_true, rec.y_pred)
        row: dict[str, Any] = {
            "run_dir": rec.run_dir,
            "fold_idx": rec.fold_idx,
            "holdout_parent": rec.holdout_parent,
            "holdout_process": rec.holdout_process,
            "target_kind": rec.target_kind,
            "target_name": rec.target_name,
            "n_measured": int(y_true.size),
        }
        for metric_name, metric_fn in metric_registry.items():
            row[metric_name] = _safe_call_metric(
                metric_fn, metric_name, y_true, y_pred
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "no per-process metrics computed. Most common cause: each fold's "
            "predictions.csv is missing rows for the holdout process(es). "
            "Re-run training (cli._write_train_results was fixed to include "
            "the eval set in predictions.csv) or run 'bp-train forward' per "
            "fold to regenerate predictions covering every process."
        )
    df.attrs.update(provenance)
    df.attrs["metrics_used"] = tuple(metric_registry.keys())
    return df


def compute_aggregated_metrics(
    output_dirs: str | Path | Sequence[str | Path],
    prepared_json: str | Path | BioProcessCollection,
    *,
    metrics: dict[str, MetricFn] | None = None,
    extra_metrics: dict[str, MetricFn] | None = None,
    target_names: Iterable[str] | None = None,
    include_volume_changes: bool = True,
    include_train: bool = False,
    equal_comparison: bool = True,
) -> pd.DataFrame:
    """Per-(target_kind, target_name) metrics, pooled within target.

    For each target, ``(y_true, y_pred)`` arrays from every (run, fold,
    holdout process) tuple are concatenated *within target only* — so
    biomass [g/L] never gets mixed with cumulative feed [L]. Each
    metric is then computed once on the concatenated array. Equal
    weight per measurement point.

    Mirrors ``MPMs/ANA_functions_02.NEW_compare_versions`` semantics:
    ``y_pooled = US_true[:,:,j].flatten()`` ⇔ "concat across (process,
    time) for target j".

    Provenance is attached to ``df.attrs`` (same keys as
    :func:`compute_per_process_metrics`).
    """
    dirs = _normalize_output_dirs(output_dirs)
    collection = _load_collection(prepared_json)
    metric_registry = _resolve_metric_registry(metrics, extra_metrics)
    target_override = tuple(target_names) if target_names is not None else None

    records, provenance = _gather_paired_arrays(
        dirs,
        collection,
        target_names=target_override,
        include_volume_changes=include_volume_changes,
        include_train=include_train,
        equal_comparison=equal_comparison,
    )

    # Group by (target_kind, target_name); concat within each group.
    grouped: dict[tuple[str, str], list[_PairedRecord]] = {}
    for rec in records:
        grouped.setdefault((rec.target_kind, rec.target_name), []).append(rec)

    rows: list[dict[str, Any]] = []
    for (kind, name), recs in grouped.items():
        y_true_all = np.concatenate([r.y_true for r in recs])
        y_pred_all = np.concatenate([r.y_pred for r in recs])
        y_true_all, y_pred_all = _filter_finite(y_true_all, y_pred_all)
        n_processes = len({r.holdout_process for r in recs})
        row: dict[str, Any] = {
            "target_kind": kind,
            "target_name": name,
            "n_obs": int(y_true_all.size),
            "n_processes": n_processes,
        }
        for metric_name, metric_fn in metric_registry.items():
            row[metric_name] = _safe_call_metric(
                metric_fn, metric_name, y_true_all, y_pred_all
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "no aggregated metrics computed. Most common cause: each fold's "
            "predictions.csv is missing rows for the holdout process(es). "
            "Re-run training (cli._write_train_results was fixed to include "
            "the eval set in predictions.csv) or run 'bp-train forward' per "
            "fold to regenerate predictions covering every process."
        )
    df.attrs.update(provenance)
    df.attrs["metrics_used"] = tuple(metric_registry.keys())
    return df


def _load_collection(
    prepared_json: str | Path | BioProcessCollection,
) -> BioProcessCollection:
    if isinstance(prepared_json, BioProcessCollection):
        return prepared_json
    return load_process_collection(Path(prepared_json))


def _filter_finite(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[finite].astype(np.float64), y_pred[finite].astype(np.float64)


def _safe_call_metric(
    fn: MetricFn, name: str, y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Run a metric callable; on error or empty input, return NaN with a warning."""
    if y_true.size == 0:
        return float("nan")
    try:
        return float(fn(y_true, y_pred))
    except Exception as exc:  # noqa: BLE001 — user metrics are arbitrary
        logger.warning("metric '%s' raised: %s; reporting NaN", name, exc)
        return float("nan")


# ---------------------------------------------------------------------------
# Pretty-print helpers (used by demo scripts; not strictly required)
# ---------------------------------------------------------------------------


def format_incompleteness_banner(df: pd.DataFrame) -> str | None:
    """Return a multi-line banner string when ``df`` carries incomplete-LOO
    flags, else None. Banner is suitable for ``print()``.
    """
    incomplete = df.attrs.get("incomplete_runs") or []
    if not incomplete:
        return None
    lines = ["!!! INCOMPLETE LOO !!!"]
    for entry in incomplete:
        lines.append(
            f"  {entry['run_dir']}: {entry['n_actual']} of "
            f"{entry['n_expected']} folds present"
        )
        miss = entry.get("missing_holdout_processes") or []
        if miss:
            lines.append(f"    missing: {', '.join(miss)}")
    lines.append("Metrics below are computed only on the folds available.")
    return "\n".join(lines)
