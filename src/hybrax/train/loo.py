"""Leave-one/some-process-out cross-validation for ``bp-train``.

Config-driven, mirroring ``train``: a ``loo`` section in the run config defines
the folds (:class:`~bp_train.run_config.HoldoutSet` entries in
``per_fold_holdout_sets``, or classic leave-one-out when omitted) and the
fold-level parallelism.

Each fold trains as **its own subprocess** so it can own a private slice of the
CPU device pool — the JAX host-device count is fixed per process at import, so
concurrent in-process folds cannot get separate shards. The user picks how many
folds run at once (``loo.parallel_folds``); the orchestrator splits the cores
across them (``devices_per_fold = n_cpu // parallel_folds``), dispatches the
workers pinned to disjoint core blocks, then aggregates their on-disk losses
into ``loo_summary.csv`` / ``loo_aggregate.json``. ``run_loo_cv(resume=True)``
skips folds that already wrote ``losses.csv`` and re-runs only the rest.

This module stays a thin orchestrator on top of
:func:`bp_train.harness.train_from_collection`; it does not reimplement
training, batching, or loss evaluation.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import queue
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from bp_format.dataclasses import AugmentedBioProcess, BioProcessCollection

from .harness import train_from_collection, train_harness_config_from_run_config
from .run_config import LooConfig, RunConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    """One resolved cross-validation fold.

    ``test`` is the held-out evaluation set; ``train`` is the
    (augmentation-corrected) set actually trained on; ``slug`` is the on-disk
    directory name under ``<output>/folds/``.
    """

    idx: int
    test: tuple[str, ...]
    train: tuple[str, ...]
    slug: str


@dataclass(frozen=True)
class FoldResult:
    """Outputs of a single executed fold (returned by :func:`run_single_fold`)."""

    fold: Fold
    fold_seed: int
    train_result: Any
    forward_result: Any
    fold_dir: Path


@dataclass(frozen=True)
class LOOResult:
    """Outputs of a full LOO sweep (returned by :func:`run_loo_cv`)."""

    fold_dirs: tuple[Path, ...]
    parallel_folds: int
    devices_per_fold: int
    summary_csv_path: Path
    aggregate_json_path: Path
    aggregate: dict[str, Any]


# ---------------------------------------------------------------------------
# Fold-group construction & resolution
# ---------------------------------------------------------------------------


FoldGroup = tuple[str, tuple[str, ...]]


def _build_fold_groups(collection: BioProcessCollection) -> tuple[FoldGroup, ...]:
    """Return ``(parent_name, group_member_names)`` tuples in canonical order.

    Each non-augmented :class:`~bp_format.dataclasses.BioProcess` becomes a fold
    group; every :class:`~bp_format.dataclasses.AugmentedBioProcess` is appended
    to its parent's group. Augmented processes never form their own fold — they
    always travel with their parent.
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

    return tuple((parent, tuple(members)) for parent, members in parent_groups.items())


def _augmented_parent_map(collection: BioProcessCollection) -> dict[str, str]:
    """Map every augmented process name -> its parent process name (validated)."""
    parents = {
        name
        for name, p in collection.processes.items()
        if not isinstance(p, AugmentedBioProcess)
    }
    out: dict[str, str] = {}
    for name, process in collection.processes.items():
        if isinstance(process, AugmentedBioProcess):
            if process.parent_process not in parents:
                raise ValueError(
                    f"AugmentedBioProcess '{name}' references parent_process "
                    f"'{process.parent_process}', which is not a non-augmented "
                    "BioProcess in the collection"
                )
            out[name] = process.parent_process
    return out


def _augmentation_group_of(
    collection: BioProcessCollection,
) -> dict[str, frozenset[str]]:
    """Map every process to its augmentation group's members.

    A group is a non-augmented parent plus all of its augmented children; a
    plain process with no children is a singleton. Holding out **any** member of
    a group taints the whole group for train/eval splits, because each augmented
    child is a synthetic variant of the same parent — so training on the parent
    or a sibling would leak the held-out sample into the fold.
    """
    parent_of = _augmented_parent_map(collection)  # child -> parent (validated)
    children: dict[str, list[str]] = {}
    for child, parent in parent_of.items():
        children.setdefault(parent, []).append(child)
    groups: dict[str, frozenset[str]] = {}
    for name in collection.processes:
        parent = parent_of.get(name, name)
        groups[name] = frozenset({parent, *children.get(parent, [])})
    return groups


def _resolve_train(
    group_of: dict[str, frozenset[str]],
    process_order: tuple[str, ...],
    test: tuple[str, ...],
    explicit_train: tuple[str, ...] | None,
    idx: int,
) -> tuple[str, ...]:
    """Resolve the train set, excluding the augmentation group of every held-out
    process. Default train (``explicit_train is None``) = every process not in a
    tainted group. A user-pinned ``train`` that lists a tainted group member of a
    held-out process is a leak and raises.
    """
    test_set = set(test)
    tainted: set[str] = set()
    for t in test:
        tainted |= group_of[t]
    if explicit_train is None:
        return tuple(p for p in process_order if p not in tainted)
    leaks = sorted(p for p in explicit_train if p in tainted and p not in test_set)
    if leaks:
        raise ValueError(
            f"loo fold {idx}: 'train' leaks augmentation-group member(s) of a "
            f"held-out process: {leaks}. Holding out a process holds out its "
            "whole augmentation group (parent + all children); remove these "
            "from train."
        )
    return tuple(explicit_train)


def _slug_from_name(name: str, idx: int) -> str:
    """Filesystem-safe fold directory name from a user-supplied fold name."""
    safe = "".join(c if (c.isalnum() or c in "+-.") else "_" for c in name)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or f"fold_{idx:03d}"


def _fold_slug(test: tuple[str, ...], idx: int) -> str:
    """Filesystem-safe fold directory name derived from the test set."""
    raw = "+".join(test)
    safe = "".join(c if (c.isalnum() or c in "+-_.") else "_" for c in raw)
    if not safe or len(safe) > 80:
        return f"fold_{idx:03d}"
    return safe


def _check_unique_slugs(folds: list[Fold]) -> None:
    seen: dict[str, int] = {}
    for fold in folds:
        if fold.slug in seen:
            raise ValueError(
                f"loo folds {seen[fold.slug]} and {fold.idx} resolve to the same "
                f"output directory slug '{fold.slug}'; give them distinct `name`s "
                "or test sets."
            )
        seen[fold.slug] = fold.idx


def _require_known(
    names: tuple[str, ...], known: set[str], *, what: str, idx: int
) -> None:
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(
            f"loo fold {idx}: '{what}' contains unknown process name(s) "
            f"{unknown}; available={sorted(known)}"
        )


def resolve_folds(
    collection: BioProcessCollection, loo_cfg: LooConfig | None
) -> tuple[Fold, ...]:
    """Resolve config into concrete folds.

    ``per_fold_holdout_sets is None`` → classic leave-one-out (one fold per
    parent group). Otherwise each :class:`HoldoutSet` becomes a fold:
    ``test = entry.test``; ``train = entry.train`` or everything not in ``test``.
    Augmentation is corrected in both modes (see :func:`_apply_augmentation`).
    """
    process_order = tuple(collection.processes.keys())
    known = set(process_order)
    group_of = _augmentation_group_of(collection)
    folds: list[Fold] = []

    if loo_cfg is None or loo_cfg.per_fold_holdout_sets is None:
        groups = _build_fold_groups(collection)
        if len(groups) < 2:
            raise ValueError(
                f"leave-one-out requires >= 2 parent processes; got {len(groups)}"
            )
        for idx, (parent, members) in enumerate(groups):
            train = _resolve_train(group_of, process_order, members, None, idx)
            if not train:
                raise ValueError(f"fold '{parent}' has no train processes")
            folds.append(Fold(idx=idx, test=members, train=train, slug=parent))
        _check_unique_slugs(folds)
        return tuple(folds)

    if not loo_cfg.per_fold_holdout_sets:
        raise ValueError("loo.per_fold_holdout_sets is empty")

    for idx, hs in enumerate(loo_cfg.per_fold_holdout_sets):
        test = tuple(hs.test)
        _require_known(test, known, what="test", idx=idx)
        explicit_train = None
        if hs.train is not None:
            explicit_train = tuple(hs.train)
            _require_known(explicit_train, known, what="train", idx=idx)
            overlap = sorted(set(test) & set(explicit_train))
            if overlap:
                raise ValueError(
                    f"loo fold {idx}: process(es) in both test and train: {overlap}"
                )
        train = _resolve_train(group_of, process_order, test, explicit_train, idx)
        if not train:
            raise ValueError(
                f"loo fold {idx} (test={list(test)}) has no train processes"
            )
        slug = _slug_from_name(hs.name, idx) if hs.name else _fold_slug(test, idx)
        folds.append(Fold(idx=idx, test=test, train=train, slug=slug))
    _check_unique_slugs(folds)
    return tuple(folds)


# ---------------------------------------------------------------------------
# Parallelism sizing
# ---------------------------------------------------------------------------


def compute_parallel_split(
    n_folds: int,
    n_cpu: int,
    parallel_folds: int,
    *,
    devices_per_fold: int | None = None,
    max_devices_per_fold: int | None = None,
) -> tuple[int, int]:
    """Return ``(parallel_folds, devices_per_fold)`` for the fold pool.

    ``parallel_folds`` is the user-chosen fold concurrency. If
    ``devices_per_fold`` is omitted, leftover cores are split across concurrent
    folds: ``devices_per_fold = n_cpu // parallel`` (so a single fold soaks up
    the cores, many folds run one core each). Devices never exceed
    ``max_devices_per_fold`` (the smallest fold's batch — exposing more host
    devices than the batch only deadlocks the pmap collective).

    Invariant: ``parallel_folds * devices_per_fold <= n_cpu``. There is no RAM
    sizing here — the user owns the memory call via ``parallel_folds``.
    """
    n_cpu = max(1, int(n_cpu))
    n_folds = max(1, int(n_folds))
    requested_parallel = max(1, min(int(parallel_folds), n_folds, n_cpu))
    if devices_per_fold is None:
        devices = max(1, n_cpu // requested_parallel)
    else:
        devices = max(1, int(devices_per_fold))
    devices = min(devices, n_cpu)
    if max_devices_per_fold is not None:
        devices = max(1, min(devices, int(max_devices_per_fold)))
    effective_parallel = max(1, min(requested_parallel, n_cpu // devices))
    return effective_parallel, devices


# ---------------------------------------------------------------------------
# Subprocess dispatch
# ---------------------------------------------------------------------------


def _worker_cmd(config_path: Path, output_dir: Path, fold_idx: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "bp_train.cli",
        "loo",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--fold",
        str(fold_idx),
    ]


def _worker_env(devices: int) -> dict[str, str]:
    env = dict(os.environ)
    env["BP_TRAIN_DEVICES"] = str(int(devices))
    env.setdefault("JAX_PLATFORMS", "cpu")
    # Strip any inherited host-device pin so the worker's import-time bootstrap
    # re-derives the device count from BP_TRAIN_DEVICES. Otherwise a pre-set
    # XLA_FLAGS=--xla_force_host_platform_device_count=K silently overrides the
    # per-fold count (every fold would expose K devices, reintroducing the
    # over-exposure deadlock the device cap exists to prevent).
    xla = env.get("XLA_FLAGS")
    if xla and "xla_force_host_platform_device_count" in xla:
        stripped = " ".join(
            tok
            for tok in xla.split()
            if not tok.startswith("--xla_force_host_platform_device_count")
        ).strip()
        if stripped:
            env["XLA_FLAGS"] = stripped
        else:
            env.pop("XLA_FLAGS", None)
    return env


def _set_affinity(pid: int, cores: list[int]) -> None:
    if not cores:
        return
    try:
        os.sched_setaffinity(pid, set(cores))
    except (AttributeError, OSError):  # non-Linux / restricted: best-effort
        pass


def _dispatch_worker(
    config_path: Path,
    output_dir: Path,
    fold_idx: int,
    devices: int,
    *,
    cores: list[int] | None = None,
) -> int:
    """Run one fold subprocess, pinned to ``cores``. Returns its exit code."""
    proc = subprocess.Popen(
        _worker_cmd(config_path, output_dir, fold_idx), env=_worker_env(devices)
    )
    if cores:
        _set_affinity(proc.pid, cores)
    return proc.wait()


def _dispatch_pool(
    config_path: Path,
    output_dir: Path,
    folds: list[Fold],
    parallel: int,
    devices: int,
) -> None:
    """Run ``folds`` in a pool of ``parallel`` workers pinned to disjoint cores."""
    n_cpu = os.cpu_count() or 1
    blocks: queue.Queue[list[int]] = queue.Queue()
    for i in range(parallel):
        lo = i * devices
        blocks.put(list(range(lo, min(lo + devices, n_cpu))))

    failures: list[tuple[str, str]] = []

    def _run(fold: Fold) -> None:
        block = blocks.get()
        try:
            rc = _dispatch_worker(
                config_path, output_dir, fold.idx, devices, cores=block or None
            )
            if rc != 0:
                failures.append((fold.slug, f"exit {rc}"))
        except Exception as exc:  # noqa: BLE001 - record, don't abort other folds
            failures.append((fold.slug, f"{type(exc).__name__}: {exc}"))
        finally:
            blocks.put(block)

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        list(pool.map(_run, folds))

    if failures:
        detail = ", ".join(f"{slug} ({why})" for slug, why in failures)
        raise RuntimeError(f"LOO fold(s) failed: {detail}")


# ---------------------------------------------------------------------------
# Per-fold execution (worker)
# ---------------------------------------------------------------------------


def _execute_fold(
    *,
    collection: BioProcessCollection,
    fold: Fold,
    cfg: RunConfig,
    custom_module: Any | None,
    output_dir: Path,
    custom_py: str | Path | None,
) -> FoldResult:
    # Local import keeps loo.py free of cli.py at module-load time.
    # _write_train_results owns: forward eval + losses.csv + predictions.csv +
    # optional plots, exactly like the post-train block in _handle_train.
    from .cli import _write_train_results
    from .postprocessing import save_model_metadata
    from .serialization import save_model

    base_seed = int(cfg.train.seed)
    fold_seed = base_seed + fold.idx
    fold_dir = Path(output_dir) / "folds" / fold.slug
    fold_dir.mkdir(parents=True, exist_ok=True)

    harness_cfg = train_harness_config_from_run_config(cfg, run_dir=fold_dir)
    harness_cfg = dataclasses.replace(
        harness_cfg,
        process_names=fold.train,
        seed=fold_seed,
        # Monitor loss = the held-out TEST set only (never the full
        # not-in-train set). Diagnostic — never drives optimizer updates.
        # Evaluated at `loo.monitor_every` cadence (None -> the logging cadence).
        monitor_processes=fold.test,
        monitor_label="holdout",
        monitor_every=cfg.loo.monitor_every if cfg.loo is not None else None,
    )

    logger.info(
        "LOO fold %d: slug=%s test=%s n_train=%d seed=%d",
        fold.idx,
        fold.slug,
        list(fold.test),
        len(fold.train),
        fold_seed,
    )

    train_result = train_from_collection(
        collection,
        config=harness_cfg,
        custom_module=custom_module,
        run_config=cfg,
    )

    save_model(train_result.trained_wrapper, fold_dir / "trained_wrapper.eqx")

    last_loss = float(train_result.mean_loss_by_step[-1])
    sidecar_dir = fold_dir.resolve()
    custom_py_rel = (
        os.path.relpath(Path(custom_py).resolve(), sidecar_dir)
        if custom_py is not None
        else None
    )
    meta = {
        "custom_py": custom_py_rel,
        "test": list(fold.test),
        "train": list(fold.train),
        "fold_idx": fold.idx,
        "fold_seed": fold_seed,
        "targets": (
            list(harness_cfg.target_variable_order)
            if harness_cfg.target_variable_order is not None
            else None
        ),
        "target_source": harness_cfg.target_source,
        "solver": {
            "max_steps": int(harness_cfg.solver_max_steps),
            "rtol": float(harness_cfg.solver_rtol),
            "atol": float(harness_cfg.solver_atol),
            "use_jump_ts": bool(harness_cfg.solver_use_jump_ts),
        },
        "training": {
            "steps": int(harness_cfg.steps),
            "batch_size": harness_cfg.batch_size,
            "seed": fold_seed,
            "final_mean_loss": last_loss,
        },
    }
    save_model_metadata(fold_dir / "trained_wrapper.meta.json", meta)

    # Evaluate exactly this fold's processes: the train set (labelled "train")
    # plus the held-out test set (labelled "holdout"). Restricting to train ∪
    # test — rather than every process — keeps the per-fold losses.csv consistent
    # with the orchestrator's loo_summary (which averages the holdout over
    # `test`) and avoids labelling processes that are in neither set (e.g.
    # augmentation siblings excluded from train) as misleading "holdout" rows.
    eval_processes = tuple(dict.fromkeys((*fold.train, *fold.test)))
    forward_result = _write_train_results(
        output_dir=fold_dir,
        collection=collection,
        trained_wrapper=train_result.trained_wrapper,
        train_result=train_result,
        config=harness_cfg,
        runtime_config=None,
        custom_py=str(custom_py) if custom_py is not None else None,
        training_process_names=fold.train,
        render_plots=cfg.output.plots,
        eval_process_names=eval_processes,
        run_config=cfg,
        custom_module=custom_module,
    )

    return FoldResult(
        fold=fold,
        fold_seed=fold_seed,
        train_result=train_result,
        forward_result=forward_result,
        fold_dir=fold_dir,
    )


def run_single_fold(
    collection: BioProcessCollection,
    *,
    cfg: RunConfig,
    custom_module: Any | None,
    output_dir: str | Path,
    fold_idx: int,
    custom_py: str | Path | None = None,
) -> FoldResult:
    """Worker entry point: resolve folds and execute exactly one by index.

    Writes ``<output_dir>/folds/<slug>/`` and does **not** aggregate (the
    orchestrator does that after all workers finish).
    """
    folds = resolve_folds(collection, cfg.loo)
    if not 0 <= fold_idx < len(folds):
        raise ValueError(
            f"--fold {fold_idx} out of range; {len(folds)} fold(s) resolved"
        )
    return _execute_fold(
        collection=collection,
        fold=folds[fold_idx],
        cfg=cfg,
        custom_module=custom_module,
        output_dir=Path(output_dir),
        custom_py=custom_py,
    )


# ---------------------------------------------------------------------------
# Cross-fold orchestration (orchestrator)
# ---------------------------------------------------------------------------


def run_loo_cv(
    collection: BioProcessCollection,
    *,
    cfg: RunConfig,
    config_path: str | Path,
    output_dir: str | Path,
    custom_py: str | Path | None = None,
    resume: bool = False,
) -> LOOResult:
    """Resolve folds, size + dispatch the subprocess pool, then aggregate.

    ``config_path`` is the run-config JSON re-passed verbatim to each worker
    subprocess (``loo --config <config_path> --fold <i>``). With ``resume=True``
    folds that already wrote ``folds/<slug>/losses.csv`` are skipped and only the
    missing/partial ones are re-run.
    """
    folds = resolve_folds(collection, cfg.loo)
    n_folds = len(folds)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path).resolve()

    loo_cfg = cfg.loo or LooConfig()
    n_cpu = os.cpu_count() or 1

    # A fold's effective batch is its train-set size, capped by an explicit
    # train.batch_size. Exposing more host devices than the batch only deadlocks
    # the pmap collective, so the per-fold device count is capped at the
    # SMALLEST fold's effective batch.
    batch_size = cfg.train.batch_size

    def _eff_batch(fold: Fold) -> int:
        return min(len(fold.train), batch_size) if batch_size else len(fold.train)

    min_batch = min(_eff_batch(f) for f in folds)

    parallel, devices = compute_parallel_split(
        n_folds,
        n_cpu,
        loo_cfg.parallel_folds,
        devices_per_fold=loo_cfg.devices_per_fold,
        max_devices_per_fold=min_batch,
    )
    if loo_cfg.parallel_folds > n_folds:
        logger.info(
            "LOO: parallel_folds=%d exceeds the %d resolved fold(s); clamped to %d.",
            loo_cfg.parallel_folds,
            n_folds,
            parallel,
        )

    # On resume, skip folds whose losses.csv already exists (the artifact the
    # aggregation reads); re-run the rest (overwriting any partial output).
    if resume:
        pending = [
            f
            for f in folds
            if not (output_dir / "folds" / f.slug / "losses.csv").is_file()
        ]
        logger.info(
            "LOO resume: %d/%d fold(s) already complete, running %d remaining",
            n_folds - len(pending),
            n_folds,
            len(pending),
        )
    else:
        pending = list(folds)

    logger.info(
        "LOO: %d fold(s), %d cpu(s) -> %d parallel fold(s) x %d device(s) each",
        n_folds,
        n_cpu,
        parallel,
        devices,
    )

    if pending:
        _dispatch_pool(config_path, output_dir, pending, parallel, devices)

    # --- aggregate on-disk fold losses ---
    summary_csv_path = output_dir / "loo_summary.csv"
    aggregate_json_path = output_dir / "loo_aggregate.json"
    aggregate = _write_summary_and_aggregate(
        folds=folds,
        output_dir=output_dir,
        summary_csv_path=summary_csv_path,
        aggregate_json_path=aggregate_json_path,
        base_seed=int(cfg.train.seed),
    )

    if cfg.output.plots:
        _plot_cross_fold_losses(folds=folds, output_dir=output_dir)

    return LOOResult(
        fold_dirs=tuple(output_dir / "folds" / f.slug for f in folds),
        parallel_folds=parallel,
        devices_per_fold=devices,
        summary_csv_path=summary_csv_path,
        aggregate_json_path=aggregate_json_path,
        aggregate=aggregate,
    )


# ---------------------------------------------------------------------------
# Summary / aggregate (read fold losses back from disk)
# ---------------------------------------------------------------------------


def _read_fold_losses(
    fold_dir: Path,
) -> tuple[tuple[str, ...], dict[str, tuple[float, tuple[float, ...]]]]:
    """Parse ``losses.csv`` into ``(target_names, {process: (total, per_target)})``.

    Only real per-process rows are returned (the appended ``* (mean)`` summary
    rows are looked up by process name and never collide with them).
    """
    df = pd.read_csv(fold_dir / "losses.csv")
    target_names = tuple(
        c for c in df.columns if c not in ("process", "total", "split")
    )
    out: dict[str, tuple[float, tuple[float, ...]]] = {}
    for _, row in df.iterrows():
        name = str(row["process"])
        out[name] = (
            float(row["total"]),
            tuple(float(row[c]) for c in target_names),
        )
    return target_names, out


def _read_final_train_loss(fold_dir: Path) -> float:
    try:
        meta = json.loads((fold_dir / "trained_wrapper.meta.json").read_text())
        return float(meta["training"]["final_mean_loss"])
    except (OSError, KeyError, ValueError, TypeError):
        return float("nan")


def _read_fold_loss_history(
    fold_dir: Path,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Read per-step train and monitor loss from a fold's ``metrics.csv``.

    Returns `(train_steps, train_loss, monitor_steps, monitor_loss)`. The
    monitor (holdout) series is sparse -- only steps where it was evaluated are
    kept. Missing/unreadable files yield empty lists.
    """
    try:
        df = pd.read_csv(fold_dir / "metrics.csv")
        tr = df[["step", "mean_loss"]].dropna()
        train_steps = tr["step"].tolist()
        train_loss = tr["mean_loss"].tolist()
        if "monitor_loss" in df.columns:
            mon = df[["step", "monitor_loss"]].dropna()
            monitor_steps = mon["step"].tolist()
            monitor_loss = mon["monitor_loss"].tolist()
        else:
            monitor_steps, monitor_loss = [], []
    except (OSError, KeyError, ValueError, pd.errors.EmptyDataError) as exc:
        # An optional end-of-run plot must never sink a completed LOO run, so a
        # missing or malformed metrics.csv just drops that fold from the figure.
        logger.warning("skipping loss history for %s: %s", fold_dir, exc)
        return [], [], [], []
    return train_steps, train_loss, monitor_steps, monitor_loss


def _plot_cross_fold_losses(*, folds: tuple[Fold, ...], output_dir: Path) -> None:
    """Render one figure overlaying every fold's train + holdout loss curves."""
    # Deferred like the other postprocessing uses here, to keep loo.py's
    # module import free of matplotlib/jax-heavy postprocessing.
    from .postprocessing import plot_cross_fold_loss_curves

    fold_curves = []
    for f in folds:
        tr_steps, tr_loss, mon_steps, mon_loss = _read_fold_loss_history(
            output_dir / "folds" / f.slug
        )
        if tr_loss or mon_loss:
            fold_curves.append((f.slug, tr_steps, tr_loss, mon_steps, mon_loss))

    if not fold_curves:
        logger.warning("no fold loss history found; skipping cross-fold loss plot")
        return
    plot_cross_fold_loss_curves(fold_curves, output_dir / "loo_loss_curves.png")


def _write_summary_and_aggregate(
    *,
    folds: tuple[Fold, ...],
    output_dir: Path,
    summary_csv_path: Path,
    aggregate_json_path: Path,
    base_seed: int,
) -> dict[str, Any]:
    """Write ``loo_summary.csv`` + ``loo_aggregate.json`` from on-disk fold losses.

    Holdout metrics average over each fold's ``test`` set; train metrics average
    over its ``train`` set.
    """
    if not folds:
        raise ValueError("cannot aggregate an empty list of folds")

    target_names: tuple[str, ...] | None = None
    summary_rows: list[dict[str, Any]] = []
    holdout_totals: list[float] = []
    holdout_per_target: dict[str, list[float]] = {}

    for fold in folds:
        fold_dir = output_dir / "folds" / fold.slug
        loss_csv = fold_dir / "losses.csv"
        if not loss_csv.is_file():
            raise FileNotFoundError(
                f"fold '{fold.slug}' did not write its losses: {loss_csv}"
            )
        tnames, losses = _read_fold_losses(fold_dir)
        if target_names is None:
            target_names = tnames
            holdout_per_target = {n: [] for n in target_names}

        def _avg(names: tuple[str, ...]) -> tuple[float, dict[str, float]]:
            totals = [losses[n][0] for n in names if n in losses]
            per_t: dict[str, float] = {}
            for ti, tname in enumerate(target_names):
                vals = [losses[n][1][ti] for n in names if n in losses]
                per_t[tname] = sum(vals) / len(vals) if vals else float("nan")
            total = sum(totals) / len(totals) if totals else float("nan")
            return total, per_t

        holdout_total, holdout_targets = _avg(fold.test)
        train_total, train_targets = _avg(fold.train)
        holdout_totals.append(holdout_total)

        row: dict[str, Any] = {
            "fold_idx": fold.idx,
            "fold_slug": fold.slug,
            "test": ";".join(fold.test),
            "fold_seed": base_seed + fold.idx,
            "holdout_total": holdout_total,
        }
        for tname in target_names:
            row[f"holdout_{tname}"] = holdout_targets[tname]
            holdout_per_target[tname].append(holdout_targets[tname])
        row["train_mean_total"] = train_total
        for tname in target_names:
            row[f"train_mean_{tname}"] = train_targets[tname]
        row["final_train_loss"] = _read_final_train_loss(fold_dir)
        summary_rows.append(row)

    assert target_names is not None
    nan_folds = [
        folds[i].slug
        for i, v in enumerate(holdout_totals)
        if v != v  # NaN
    ]
    if nan_folds:
        logger.warning(
            "LOO: %d fold(s) contributed no holdout loss and are excluded from "
            "the aggregate (no test process found in losses.csv): %s",
            len(nan_folds),
            nan_folds,
        )
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
        "fold_slug": "mean",
        "test": "",
        "fold_seed": "",
        "holdout_total": aggregate["holdout_total_mean"],
    }
    for tname in target_names:
        mean_row[f"holdout_{tname}"] = aggregate[f"holdout_{tname}_mean"]
    mean_row["train_mean_total"] = _mean([r["train_mean_total"] for r in summary_rows])
    for tname in target_names:
        mean_row[f"train_mean_{tname}"] = _mean(
            [r[f"train_mean_{tname}"] for r in summary_rows]
        )
    mean_row["final_train_loss"] = _mean([r["final_train_loss"] for r in summary_rows])
    summary_rows.append(mean_row)

    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    aggregate_json_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_json_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
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
    # Spread is undefined with fewer than two folds -> NaN (0.0 would read as
    # "no variance" rather than "not enough folds").
    cleaned = [v for v in values if v == v]
    if len(cleaned) < 2:
        return float("nan")
    return float(statistics.stdev(cleaned))


def _median(values: list[float]) -> float:
    cleaned = [v for v in values if v == v]
    return float(statistics.median(cleaned)) if cleaned else float("nan")
