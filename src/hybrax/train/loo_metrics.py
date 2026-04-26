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
from typing import Any, Iterable

import numpy as np
import pandas as pd
from bp_format.dataclasses import (
    BioProcess,
    BioProcessCollection,
    ReactorMediumComponent,
    StaticVariable,
    TimeSeries,
)
from bp_format.serialization import load_process_collection_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LOOMetricsResult:
    """Outputs of :func:`compute_loo_metrics`."""

    per_fold_target: pd.DataFrame  # columns: fold_idx, holdout_parent, holdout_process, target, n_meas, r2, nmae, mae, rmse
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
            "n_meas": 0,
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
        "n_meas": n,
        "r2": r2,
        "nmae": nmae,
        "mae": mae,
        "rmse": rmse,
    }


# ---------------------------------------------------------------------------
# Per-fold and per-process evaluation
# ---------------------------------------------------------------------------


def _evaluate_predictions_for_process(
    *,
    pred_t: np.ndarray,
    pred_columns: dict[str, np.ndarray],
    measurements: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, dict[str, float]]:
    """Interpolate predictions to measurement times and compute metrics."""
    out: dict[str, dict[str, float]] = {}
    for target, (meas_t, meas_y) in measurements.items():
        col_name = f"c_{target}"
        pred_y = pred_columns.get(col_name)
        if pred_y is None:
            continue
        # numpy.interp clamps to first/last sample for out-of-range times,
        # which matches the solver's behaviour at t < t0 and t > t_end.
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
        collection = load_process_collection_json(Path(prepared_json))

    target_override = tuple(target_names) if target_names is not None else None

    rows: list[dict[str, Any]] = []
    for fold_dir in _iter_fold_dirs(loo_dir):
        sidecar = _read_fold_sidecar(fold_dir)
        holdout_group = tuple(sidecar.get("holdout_group") or ())
        holdout_parent = sidecar.get("holdout_parent") or fold_dir.name
        fold_idx = int(sidecar.get("fold_idx", -1))
        if not holdout_group:
            logger.warning(
                "fold '%s' has no holdout_group in sidecar; skipping",
                fold_dir,
            )
            continue
        pred_df = _read_predictions_csv(fold_dir)
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
            pred_columns = {
                col: sub[col].to_numpy(dtype=np.float64) for col in sub.columns
            }
            measurements = _extract_measurements(process, targets)
            metrics_per_target = _evaluate_predictions_for_process(
                pred_t=pred_t,
                pred_columns=pred_columns,
                measurements=measurements,
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
        collection = load_process_collection_json(Path(prepared_json))
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
        target_stats: dict[str, Any] = {"n_observations": int(sub["n_meas"].sum())}
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
