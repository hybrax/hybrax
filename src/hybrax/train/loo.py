"""Leave-One-Process-Out cross-validation orchestration for ``bp-train``.

Given a prepared ``BioProcessCollection``, run one training fold per parent
process group: train on N-1 groups, evaluate forward on every process so
the held-out parent (and any of its augmented children) appear as
``holdout`` in the per-fold loss table. Aggregate results across folds.

This module is a thin orchestrator on top of
:func:`bp_train.harness.train_from_collection` and
:func:`bp_train.harness.forward_from_collection`. It does not reimplement
training, batching, or loss evaluation.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from bp_format.dataclasses import (
    AugmentedBioProcess,
    BioProcessCollection,
)
from bp_format.serialization import load_process_collection_json

from .harness import (
    ForwardResult,
    TrainHarnessConfig,
    TrainHarnessResult,
    train_from_collection,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LOOConfig:
    """Configuration for a Leave-One-Process-Out cross-validation run."""

    base_train_config: TrainHarnessConfig
    output_dir: Path
    selected_holdouts: tuple[str, ...] | None = None
    """Parent process names to run as holdouts. ``None`` runs all parents.

    A single name selects exactly that fold (cluster-friendly). Multiple
    names select a subset. Augmented child names are rejected.
    """
    render_plots: bool = True
    write_per_fold_predictions: bool = True


@dataclass(frozen=True)
class FoldResult:
    """Outputs of a single LOO fold."""

    holdout_parent: str
    holdout_group: tuple[str, ...]
    fold_idx: int
    fold_seed: int
    train_processes: tuple[str, ...]
    train_result: TrainHarnessResult
    forward_result: ForwardResult
    fold_dir: Path


@dataclass(frozen=True)
class LOOResult:
    """Outputs of a full LOO sweep across all selected folds."""

    folds: tuple[FoldResult, ...]
    summary_csv_path: Path | None
    aggregate_json_path: Path | None
    aggregate: dict[str, Any]


# ---------------------------------------------------------------------------
# Fold-group construction
# ---------------------------------------------------------------------------


FoldGroup = tuple[str, tuple[str, ...]]


def _build_fold_groups(collection: BioProcessCollection) -> tuple[FoldGroup, ...]:
    """Return ``(parent_name, group_member_names)`` tuples in canonical order.

    Each non-augmented :class:`BioProcess` becomes a fold group; every
    :class:`AugmentedBioProcess` is appended to its parent's group.
    Augmented processes never form their own fold, so they are never
    held out alone — they always travel with their parent.

    Raises:
        ValueError: if any augmented child references a parent that does
            not exist in the collection.
    """
    parent_groups: dict[str, list[str]] = {}
    augmented_children: list[tuple[str, str]] = []
    for name, process in collection.processes.items():
        if isinstance(process, AugmentedBioProcess):
            augmented_children.append((name, process.parent_process))
        else:
            parent_groups[name] = [name]

    for child_name, parent_name in augmented_children:
        if parent_name not in parent_groups:
            raise ValueError(
                f"AugmentedBioProcess '{child_name}' references "
                f"parent_process '{parent_name}', which is not a "
                "non-augmented BioProcess in the collection"
            )
        parent_groups[parent_name].append(child_name)

    return tuple(
        (parent, tuple(members)) for parent, members in parent_groups.items()
    )


def _resolve_selected_folds(
    fold_groups: tuple[FoldGroup, ...],
    selected_holdouts: tuple[str, ...] | None,
    collection: BioProcessCollection,
) -> tuple[tuple[int, FoldGroup], ...]:
    """Resolve ``selected_holdouts`` to ``(fold_idx, group)`` pairs."""
    parent_to_idx = {parent: idx for idx, (parent, _) in enumerate(fold_groups)}
    if selected_holdouts is None:
        return tuple(enumerate(fold_groups))

    seen: set[str] = set()
    resolved: list[tuple[int, FoldGroup]] = []
    for name in selected_holdouts:
        if name in seen:
            raise ValueError(
                f"--holdouts contains duplicate entry '{name}'"
            )
        seen.add(name)
        if name in parent_to_idx:
            idx = parent_to_idx[name]
            resolved.append((idx, fold_groups[idx]))
            continue

        process = collection.processes.get(name)
        if isinstance(process, AugmentedBioProcess):
            raise ValueError(
                f"--holdouts must reference parent processes; "
                f"'{name}' is augmented (parent='{process.parent_process}')"
            )
        raise ValueError(
            f"--holdouts contains unknown process name '{name}'; "
            f"available parents={tuple(parent_to_idx)}"
        )
    return tuple(resolved)


# ---------------------------------------------------------------------------
# Per-fold execution
# ---------------------------------------------------------------------------


def run_loo_fold(
    collection: BioProcessCollection,
    *,
    holdout_parent: str,
    config: LOOConfig,
    custom_py: str | Path | None = None,
    runtime_config: dict[str, Any] | None = None,
) -> FoldResult:
    """Train one LOO fold (holdout = ``holdout_parent`` and its augmented children).

    The fold writes ``trained_wrapper.eqx``, sidecar metadata, ``losses.csv``,
    ``predictions.csv``, optional plots, and (if ``checkpoint_dir`` is set
    on the base train config) per-fold checkpoints under
    ``<output_dir>/folds/<holdout_parent>/``.
    """
    fold_groups = _build_fold_groups(collection)
    parent_to_idx = {parent: idx for idx, (parent, _) in enumerate(fold_groups)}
    if holdout_parent not in parent_to_idx:
        process = collection.processes.get(holdout_parent)
        if isinstance(process, AugmentedBioProcess):
            raise ValueError(
                f"holdout_parent='{holdout_parent}' is an "
                f"AugmentedBioProcess (parent='{process.parent_process}'); "
                "only non-augmented parent processes can be held out"
            )
        raise ValueError(
            f"holdout_parent='{holdout_parent}' is not in the collection; "
            f"available parents={tuple(parent_to_idx)}"
        )

    fold_idx = parent_to_idx[holdout_parent]
    holdout_group = fold_groups[fold_idx][1]

    return _execute_fold(
        collection=collection,
        fold_idx=fold_idx,
        holdout_parent=holdout_parent,
        holdout_group=holdout_group,
        config=config,
        custom_py=custom_py,
        runtime_config=runtime_config,
    )


def _execute_fold(
    *,
    collection: BioProcessCollection,
    fold_idx: int,
    holdout_parent: str,
    holdout_group: tuple[str, ...],
    config: LOOConfig,
    custom_py: str | Path | None,
    runtime_config: dict[str, Any] | None,
) -> FoldResult:
    # Local import: keeps loo.py free of cli.py at module-load time.
    # _write_train_results owns: forward eval + losses.csv + predictions.csv
    # + optional plots, exactly like the post-train block in _handle_train.
    from .cli import _write_train_results

    base_cfg = config.base_train_config
    process_order = tuple(collection.processes.keys())
    train_processes = tuple(p for p in process_order if p not in holdout_group)
    if not train_processes:
        raise ValueError(
            f"fold '{holdout_parent}' has no train processes after "
            "removing the holdout group"
        )

    fold_dir = Path(config.output_dir) / "folds" / holdout_parent
    fold_dir.mkdir(parents=True, exist_ok=True)

    fold_seed = int(base_cfg.seed) + fold_idx

    if base_cfg.checkpoint_dir is None:
        # base config disables checkpointing entirely; preserve.
        fold_checkpoint_dir: Path | None = None
    else:
        fold_checkpoint_dir = fold_dir / "checkpoints"

    fold_cfg = dataclasses.replace(
        base_cfg,
        process_names=train_processes,
        seed=fold_seed,
        checkpoint_dir=fold_checkpoint_dir,
        # Monitor (validation) loss = holdout group, evaluated at log-step
        # cadence. Diagnostic only — never drives optimizer updates.
        monitor_processes=holdout_group,
        monitor_label="holdout",
    )

    logger.info(
        "LOO fold %d/%d: holdout_parent=%s holdout_group=%s "
        "train_processes=%s seed=%d",
        fold_idx + 1,
        len(_build_fold_groups(collection)),
        holdout_parent,
        list(holdout_group),
        list(train_processes),
        fold_seed,
    )

    train_result = train_from_collection(
        collection,
        config=fold_cfg,
        custom_py=custom_py,
        runtime_config=runtime_config,
    )

    # Save trained wrapper + sidecar (mirrors _handle_train post-train block).
    from .postprocessing import save_model, save_model_metadata

    model_path = fold_dir / "trained_wrapper.eqx"
    save_model(train_result.trained_wrapper, model_path)

    last_loss = float(train_result.mean_loss_by_step[-1])
    sidecar_dir = fold_dir.resolve()
    custom_py_rel = (
        os.path.relpath(Path(custom_py).resolve(), sidecar_dir)
        if custom_py is not None
        else None
    )
    meta = {
        "prepared_input": None,  # caller-known; LOO does not load from a path
        "custom_py": custom_py_rel,
        "training_processes": list(train_processes),
        "holdout_parent": holdout_parent,
        "holdout_group": list(holdout_group),
        "fold_idx": fold_idx,
        "fold_seed": fold_seed,
        "targets": (
            list(fold_cfg.target_variable_order)
            if fold_cfg.target_variable_order is not None
            else None
        ),
        "target_source": fold_cfg.target_source,
        "solver": {
            "max_steps": int(fold_cfg.solver_max_steps),
            "rtol": float(fold_cfg.solver_rtol),
            "atol": float(fold_cfg.solver_atol),
            "use_jump_ts": bool(fold_cfg.solver_use_jump_ts),
        },
        "training": {
            "steps": int(fold_cfg.steps),
            "batch_size": fold_cfg.batch_size,
            "seed": fold_seed,
            "final_mean_loss": last_loss,
        },
    }
    save_model_metadata(fold_dir / "trained_wrapper.meta.json", meta)

    # Forward over the full collection so train/holdout split is recorded.
    forward_result = _write_train_results(
        output_dir=fold_dir,
        collection=collection,
        trained_wrapper=train_result.trained_wrapper,
        train_result=train_result,
        config=fold_cfg,
        runtime_config=runtime_config,
        custom_py=str(custom_py) if custom_py is not None else None,
        training_process_names=train_processes,
        render_plots=config.render_plots,
        eval_process_names=process_order,
    )

    return FoldResult(
        holdout_parent=holdout_parent,
        holdout_group=holdout_group,
        fold_idx=fold_idx,
        fold_seed=fold_seed,
        train_processes=train_processes,
        train_result=train_result,
        forward_result=forward_result,
        fold_dir=fold_dir,
    )


# ---------------------------------------------------------------------------
# Cross-fold orchestration & aggregation
# ---------------------------------------------------------------------------


def run_loo_cv(
    collection: BioProcessCollection,
    *,
    config: LOOConfig,
    custom_py: str | Path | None = None,
    runtime_config: dict[str, Any] | None = None,
) -> LOOResult:
    """Run all selected LOO folds and (when running >1 fold) aggregate.

    Single-fold invocations skip the top-level summary / aggregate so
    cluster-parallel runs don't race. Aggregate artifacts are only
    written when ``selected_holdouts is None`` (i.e. the full sweep).
    """
    fold_groups = _build_fold_groups(collection)
    if len(fold_groups) < 2:
        raise ValueError(
            "LOO-CV requires at least 2 parent processes; "
            f"got {len(fold_groups)}"
        )

    selected = _resolve_selected_folds(
        fold_groups, config.selected_holdouts, collection
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds: list[FoldResult] = []
    for fold_idx, (parent_name, holdout_group) in selected:
        fold = _execute_fold(
            collection=collection,
            fold_idx=fold_idx,
            holdout_parent=parent_name,
            holdout_group=holdout_group,
            config=config,
            custom_py=custom_py,
            runtime_config=runtime_config,
        )
        folds.append(fold)

    # Only write summary/aggregate for full sweeps. Subset/single-fold runs
    # leave aggregation to a later --aggregate-only invocation.
    write_summary = config.selected_holdouts is None
    summary_csv_path: Path | None = None
    aggregate_json_path: Path | None = None
    aggregate: dict[str, Any] = {}
    if write_summary:
        summary_csv_path = output_dir / "loo_summary.csv"
        aggregate_json_path = output_dir / "loo_aggregate.json"
        aggregate = _write_summary_and_aggregate(
            folds=tuple(folds),
            summary_csv_path=summary_csv_path,
            aggregate_json_path=aggregate_json_path,
            base_seed=int(config.base_train_config.seed),
        )

    return LOOResult(
        folds=tuple(folds),
        summary_csv_path=summary_csv_path,
        aggregate_json_path=aggregate_json_path,
        aggregate=aggregate,
    )


def run_loo_from_prepared_json(
    prepared_json: str | Path,
    *,
    config: LOOConfig,
    custom_py: str | Path | None = None,
    runtime_config: dict[str, Any] | None = None,
) -> LOOResult:
    """Path-based wrapper around :func:`run_loo_cv`."""
    collection = load_process_collection_json(Path(prepared_json))
    return run_loo_cv(
        collection,
        config=config,
        custom_py=custom_py,
        runtime_config=runtime_config,
    )


# ---------------------------------------------------------------------------
# Summary / aggregate helpers
# ---------------------------------------------------------------------------


def _write_summary_and_aggregate(
    *,
    folds: tuple[FoldResult, ...],
    summary_csv_path: Path,
    aggregate_json_path: Path,
    base_seed: int,
) -> dict[str, Any]:
    """Write ``loo_summary.csv`` and ``loo_aggregate.json``.

    Returns the aggregate dict so callers can attach it to ``LOOResult``.
    """
    if not folds:
        raise ValueError("cannot aggregate an empty list of folds")

    target_names = folds[0].forward_result.target_names

    summary_rows: list[dict[str, Any]] = []
    holdout_totals: list[float] = []
    holdout_per_target: dict[str, list[float]] = {n: [] for n in target_names}

    for fold in folds:
        fwd = fold.forward_result
        # Holdout numbers: averaged across the holdout group (parent + any
        # augmented children) so single-process and augmented runs are
        # comparable.
        ht_values: list[float] = []
        ht_per_target: dict[str, list[float]] = {n: [] for n in target_names}
        train_totals: list[float] = []
        train_per_target: dict[str, list[float]] = {n: [] for n in target_names}
        for name in fwd.process_names:
            total = fwd.per_process_total_loss[name]
            per_target = fwd.per_process_per_target_loss[name]
            if name in fold.holdout_group:
                ht_values.append(total)
                for tname, v in zip(target_names, per_target):
                    ht_per_target[tname].append(v)
            else:
                train_totals.append(total)
                for tname, v in zip(target_names, per_target):
                    train_per_target[tname].append(v)

        if not ht_values:
            raise RuntimeError(
                f"fold '{fold.holdout_parent}' produced no holdout losses"
            )

        holdout_total = sum(ht_values) / len(ht_values)
        holdout_totals.append(holdout_total)

        row: dict[str, Any] = {
            "fold_idx": fold.fold_idx,
            "holdout_parent": fold.holdout_parent,
            "holdout_group": ";".join(fold.holdout_group),
            "fold_seed": fold.fold_seed,
            "holdout_total": holdout_total,
        }
        for tname in target_names:
            mean_v = (
                sum(ht_per_target[tname]) / len(ht_per_target[tname])
                if ht_per_target[tname]
                else float("nan")
            )
            row[f"holdout_{tname}"] = mean_v
            holdout_per_target[tname].append(mean_v)

        row["train_mean_total"] = (
            sum(train_totals) / len(train_totals) if train_totals else float("nan")
        )
        for tname in target_names:
            row[f"train_mean_{tname}"] = (
                sum(train_per_target[tname]) / len(train_per_target[tname])
                if train_per_target[tname]
                else float("nan")
            )
        row["final_train_loss"] = float(fold.train_result.mean_loss_by_step[-1])
        summary_rows.append(row)

    # Append a final aggregate (mean across folds) row for human inspection.
    aggregate: dict[str, Any] = {
        "base_seed": base_seed,
        "n_folds": len(folds),
        "holdout_total_mean": _mean(holdout_totals),
        "holdout_total_std": _std(holdout_totals),
        "holdout_total_median": _median(holdout_totals),
    }
    for tname in target_names:
        vals = holdout_per_target[tname]
        aggregate[f"holdout_{tname}_mean"] = _mean(vals)
        aggregate[f"holdout_{tname}_std"] = _std(vals)
        aggregate[f"holdout_{tname}_median"] = _median(vals)

    mean_row: dict[str, Any] = {
        "fold_idx": "mean",
        "holdout_parent": "mean",
        "holdout_group": "",
        "fold_seed": "",
        "holdout_total": aggregate["holdout_total_mean"],
    }
    for tname in target_names:
        mean_row[f"holdout_{tname}"] = aggregate[f"holdout_{tname}_mean"]
    mean_row["train_mean_total"] = _mean(
        [r["train_mean_total"] for r in summary_rows]
    )
    for tname in target_names:
        mean_row[f"train_mean_{tname}"] = _mean(
            [r[f"train_mean_{tname}"] for r in summary_rows]
        )
    mean_row["final_train_loss"] = _mean(
        [r["final_train_loss"] for r in summary_rows]
    )
    summary_rows.append(mean_row)

    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    aggregate_json_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_json_path.write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    logger.info(
        "LOO summary saved to %s; aggregate to %s",
        summary_csv_path,
        aggregate_json_path,
    )
    return aggregate


def _mean(values: list[float]) -> float:
    cleaned = [v for v in values if v == v]  # drop NaNs
    return float(statistics.fmean(cleaned)) if cleaned else float("nan")


def _std(values: list[float]) -> float:
    cleaned = [v for v in values if v == v]
    if len(cleaned) < 2:
        return 0.0 if len(cleaned) == 1 else float("nan")
    return float(statistics.stdev(cleaned))


def _median(values: list[float]) -> float:
    cleaned = [v for v in values if v == v]
    return float(statistics.median(cleaned)) if cleaned else float("nan")
