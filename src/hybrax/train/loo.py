"""Leave-one/some-process-out cross-validation for ``hybrax.train``.

Config-driven, mirroring ``train``: a ``loo`` section in the run config defines
the folds (:class:`~hybrax.train.run_config.HoldoutSet` entries in
``per_fold_holdout_sets``, or classic leave-one-out when omitted) and the
fold-level parallelism.

Each fold trains as **its own subprocess** so it can expose a private JAX CPU
device count — the JAX host-device count is fixed per process at import, so
concurrent in-process folds cannot get separate shards. The user picks how many
folds run at once (``loo.parallel_folds``); the orchestrator chooses the per-fold
JAX device count (``devices_per_fold = n_cpu // parallel_folds`` by default),
lets the OS schedule the worker processes, then aggregates their on-disk losses
into ``loo_summary.csv`` / ``loo_aggregate.json``. ``run_loo_cv(resume=True)``
skips only identity-matched, complete folds and re-runs the rest.

This module stays a thin orchestrator on top of the harness preparation,
training, and evaluation entry points; it does not reimplement batching or loss
calculation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from hybrax.format import validate_augmented_parent_refs
from hybrax.format.dataclasses import AugmentedBioProcess, BioProcessCollection
from hybrax.format.json_io import load_json

from .harness import (
    ForwardConfig,
    ForwardResult,
    PreparedTraining,
    TrainHarnessConfig,
    TrainHarnessResult,
    _resolve_estimated_scales,
    evaluate_trained_wrapper,
    prepare_training,
    prepare_training_from_runtime_artifact,
    train_collection,
    train_harness_config_from_run_config,
)
from .postprocessing import save_model_metadata
from .runtime_artifact import (
    FORMAT_VERSION,
    RhsNames,
    RuntimeArtifactFold,
    RuntimeArtifactMetadata,
    load_runtime_artifact,
    read_runtime_artifact_metadata,
    write_runtime_artifact,
)
from .runtime_context import (
    ProducerCollectionData,
    original_parent_processes,
    select_parent_collection,
)
from .run_config import LooConfig, PredictionScope, RunConfig
from .serialization import (
    content_hash,
    load_trained_wrapper,
    run_config_to_jsonable,
    save_model,
    update_json,
    write_json,
)
from .validate import ensure_prepared_training_semantics, validate_for_training
from .training_data import TrainingDataStore

logger = logging.getLogger(__name__)

_RUNTIME_ARTIFACT_NAME = "runtime-artifact"


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
    seed: int


@dataclass(frozen=True)
class FoldResult:
    """Outputs of a single executed fold (returned by :func:`run_single_fold`)."""

    fold: Fold
    fold_seed: int
    train_result: TrainHarnessResult
    forward_result: ForwardResult
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


def _fold_record(fold: Fold) -> RuntimeArtifactFold:
    return RuntimeArtifactFold(
        idx=fold.idx,
        test=fold.test,
        train=fold.train,
        slug=fold.slug,
        seed=fold.seed,
    )


def _fingerprint(bundle_path: Path, custom_path: Path | None) -> str:
    raw = load_json(bundle_path)
    if not isinstance(raw, dict):
        raise ValueError("bundled LOO config must be a JSON object")
    custom_hash = (
        "sha256:" + hashlib.sha256(custom_path.read_bytes()).hexdigest()
        if custom_path is not None
        else None
    )
    payload = {"config": raw, "custom_hash": custom_hash}
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def _validated_runtime_metadata(
    artifact_path: Path, *, bundle_path: Path, custom_path: Path | None
) -> RuntimeArtifactMetadata:
    metadata = read_runtime_artifact_metadata(artifact_path)
    if set(metadata.identity_inputs) != {
        "run_fingerprint",
        "prepared_content_hash",
    }:
        raise ValueError("LOO runtime artifact has invalid identity inputs")
    if metadata.identity_inputs["run_fingerprint"] != _fingerprint(
        bundle_path, custom_path
    ):
        raise ValueError(
            "LOO runtime artifact fingerprint does not match bundled config"
        )
    return metadata


def _runtime_metadata(
    output_dir: Path, *, bundle_path: Path, custom_path: Path | None
) -> tuple[Path, RuntimeArtifactMetadata]:
    try:
        document = load_json(output_dir / "config.json")
    except Exception as error:
        raise ValueError("invalid LOO run config") from error
    anchor = document.get("runtime_artifact") if isinstance(document, dict) else None
    if (
        not isinstance(anchor, dict)
        or set(anchor) != {"format_version", "identity"}
        or type(anchor["format_version"]) is not int
        or type(anchor["identity"]) is not str
    ):
        raise ValueError("invalid runtime artifact anchor")
    artifact_path = output_dir / _RUNTIME_ARTIFACT_NAME
    metadata = _validated_runtime_metadata(
        artifact_path, bundle_path=bundle_path, custom_path=custom_path
    )
    if anchor["format_version"] != FORMAT_VERSION:
        raise ValueError("runtime artifact anchor has wrong format version")
    if anchor["identity"] != metadata.identity:
        raise ValueError("runtime artifact identity does not match run config anchor")
    return artifact_path, metadata


# ---------------------------------------------------------------------------
# Fold-group construction & resolution
# ---------------------------------------------------------------------------


FoldGroup = tuple[str, tuple[str, ...]]


def _build_fold_groups(collection: BioProcessCollection) -> tuple[FoldGroup, ...]:
    """Return ``(parent_name, group_member_names)`` tuples in canonical order.

    Each non-augmented :class:`~hybrax.format.dataclasses.BioProcess` becomes a fold
    group; every :class:`~hybrax.format.dataclasses.AugmentedBioProcess` is appended
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
    names: tuple[str, ...],
    known: set[str],
    *,
    what: str,
    idx: int | None = None,
    full: set[str] | None = None,
) -> None:
    unknown = [n for n in names if n not in known]
    if not unknown:
        return
    prefix = f"loo fold {idx}: " if idx is not None else ""
    if full is not None:
        excluded = sorted(n for n in unknown if n in full)
        absent = sorted(n for n in unknown if n not in full)
        parts = []
        if excluded:
            parts.append(
                f"{excluded} present in the collection but excluded by data.processes"
            )
        if absent:
            parts.append(f"{absent} not present in the collection at all")
        raise ValueError(
            f"{prefix}'{what}' contains unknown process name(s): "
            + "; ".join(parts)
            + f"; available={sorted(known)}"
        )
    raise ValueError(
        f"{prefix}'{what}' contains unknown process name(s) {unknown}; "
        f"available={sorted(known)}"
    )


def resolve_folds(
    collection: BioProcessCollection,
    loo_cfg: LooConfig | None,
    base_seed: int,
    *,
    data_processes: tuple[str, ...] | None = None,
) -> tuple[Fold, ...]:
    """Resolve config into concrete folds.

    ``per_fold_holdout_sets is None`` → classic leave-one-out (one fold per
    parent group). Otherwise each :class:`HoldoutSet` becomes a fold:
    ``test = entry.test``; ``train = entry.train`` or everything not in ``test``.
    Augmentation is corrected in both modes: holding out any member of an
    augmentation group excludes the whole group from train (see
    ``_augmentation_group_of``/``_resolve_train``).

    ``data_processes`` (from ``DataConfig.processes``), when given, restricts
    the process universe available to fold construction *before*
    ``per_fold_holdout_sets``/classic-LOO see it: names outside it can never
    end up in a fold's ``test``/``train``, and ``per_fold_holdout_sets``
    entries that reference such a name raise instead of silently including
    it.
    """
    process_order_full = tuple(collection.processes.keys())
    full_names = set(process_order_full)

    if data_processes is not None:
        _require_known(tuple(data_processes), full_names, what="data.processes")
        restrict = set(data_processes)
    else:
        restrict = full_names

    # Parent resolution is always derived from the full collection — an
    # augmented child's parent must resolve even if only one of the two
    # survives the data_processes restriction.
    parent_of = _augmented_parent_map(collection)
    for child, parent in parent_of.items():
        if child in restrict and parent not in restrict:
            raise ValueError(
                f"data.processes includes augmented process '{child}' but "
                f"excludes its parent process '{parent}'; augmented "
                "processes must be restricted together with their "
                f"non-augmented parent (add '{parent}' to data.processes, "
                f"or remove '{child}')."
            )

    process_order = tuple(p for p in process_order_full if p in restrict)
    known = restrict

    loo_collection = collection
    if data_processes is not None:
        loo_collection = BioProcessCollection(
            processes={
                name: proc
                for name, proc in collection.processes.items()
                if name in restrict
            },
            metadata=collection.metadata,
        )

    group_of = _augmentation_group_of(collection)
    folds: list[Fold] = []

    if loo_cfg is None or loo_cfg.per_fold_holdout_sets is None:
        groups = _build_fold_groups(loo_collection)
        if len(groups) < 2:
            raise ValueError(
                f"leave-one-out requires >= 2 parent processes; got {len(groups)}"
            )
        for idx, (parent, members) in enumerate(groups):
            train = _resolve_train(group_of, process_order, members, None, idx)
            if not train:
                raise ValueError(f"fold '{parent}' has no train processes")
            folds.append(
                Fold(
                    idx=idx,
                    test=members,
                    train=train,
                    slug=parent,
                    seed=base_seed + idx,
                )
            )
        _check_unique_slugs(folds)
        return tuple(folds)

    if not loo_cfg.per_fold_holdout_sets:
        raise ValueError("loo.per_fold_holdout_sets is empty")

    for idx, hs in enumerate(loo_cfg.per_fold_holdout_sets):
        test = tuple(hs.test)
        _require_known(test, known, what="test", idx=idx, full=full_names)
        explicit_train = None
        if hs.train is not None:
            explicit_train = tuple(hs.train)
            _require_known(
                explicit_train, known, what="train", idx=idx, full=full_names
            )
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
        folds.append(
            Fold(
                idx=idx,
                test=test,
                train=train,
                slug=slug,
                seed=base_seed + idx,
            )
        )
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
    ``devices_per_fold`` is omitted, the per-fold JAX CPU device budget is split
    across concurrent folds: ``devices_per_fold = n_cpu // parallel`` (so a
    single fold exposes many devices, many folds expose one device each).
    Devices never exceed ``max_devices_per_fold`` (the smallest fold's batch —
    exposing more host devices than the batch only deadlocks the pmap collective).

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


def _producer_cmd(config_path: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "hybrax.train.cli",
        "loo",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--produce-runtime",
    ]


def _worker_cmd(
    config_path: Path, output_dir: Path, artifact_path: Path, fold_idx: int
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "hybrax.train.cli",
        "loo",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--runtime-artifact",
        str(artifact_path),
        "--fold",
        str(fold_idx),
    ]


def _worker_env(devices: int) -> dict[str, str]:
    env = dict(os.environ)
    env["HYBRAX_TRAIN_DEVICES"] = str(int(devices))
    env.setdefault("JAX_PLATFORMS", "cpu")
    # Strip any inherited host-device pin so the worker's import-time bootstrap
    # re-derives the device count from HYBRAX_TRAIN_DEVICES. Otherwise a pre-set
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


def _dispatch_producer(config_path: Path, output_dir: Path) -> int:
    """Run the collection-owning producer before any fold worker starts."""
    return subprocess.run(
        _producer_cmd(config_path, output_dir), check=False
    ).returncode


def _dispatch_worker(
    config_path: Path,
    output_dir: Path,
    artifact_path: Path,
    fold_idx: int,
    devices: int,
) -> int:
    """Run one fold subprocess. Returns its exit code."""
    proc = subprocess.Popen(
        _worker_cmd(config_path, output_dir, artifact_path, fold_idx),
        env=_worker_env(devices),
    )
    return proc.wait()


def _dispatch_pool(
    config_path: Path,
    output_dir: Path,
    artifact_path: Path,
    folds: list[Fold],
    parallel: int,
    devices: int,
) -> None:
    """Run ``folds`` in a pool of ``parallel`` worker subprocesses."""
    failures: list[tuple[str, str]] = []

    def _run(fold: Fold) -> None:
        try:
            rc = _dispatch_worker(
                config_path, output_dir, artifact_path, fold.idx, devices
            )
            if rc != 0:
                failures.append((fold.slug, f"exit {rc}"))
        except Exception as exc:  # noqa: BLE001 - record, don't abort other folds
            failures.append((fold.slug, f"{type(exc).__name__}: {exc}"))

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        list(pool.map(_run, folds))

    if failures:
        detail = ", ".join(f"{slug} ({why})" for slug, why in failures)
        raise RuntimeError(f"LOO fold(s) failed: {detail}")


# ---------------------------------------------------------------------------
# Per-fold execution (worker)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedFold:
    """Collection-free inputs for one fold's training."""

    fold: Fold
    fold_seed: int
    fold_dir: Path
    fold_custom: Path | None
    config_json: Path
    effective_cfg: RunConfig
    training: PreparedTraining


@dataclass(frozen=True)
class TrainedFold:
    """Lean inputs retained for one fold's final evaluation."""

    fold: Fold
    fold_seed: int
    fold_dir: Path
    config_json: Path
    config: TrainHarnessConfig
    store: TrainingDataStore
    target_names: tuple[str, ...]
    prediction_parent_process_names: tuple[str, ...]
    output_predictions: PredictionScope
    train_result: TrainHarnessResult


def _effective_fold_config(
    cfg: RunConfig, fold: Fold, output_dir: Path, custom_py: Path | None
) -> tuple[RunConfig, Path, Path | None]:
    fold_dir = output_dir / "folds" / fold.slug
    fold_custom = fold_dir / "custom.py" if custom_py is not None else None
    effective_cfg = cfg.model_copy(
        update={
            "data": cfg.data.model_copy(update={"processes": fold.train}),
            "train": cfg.train.model_copy(update={"seed": fold.seed}),
            "output": cfg.output.model_copy(update={"dir": fold_dir.resolve()}),
            "custom_py": fold_custom,
        }
    )
    return effective_cfg, fold_dir, fold_custom


def produce_runtime_artifact(
    *, cfg: RunConfig, custom_module: Any, output_dir: Path, bundle_path: Path
) -> str:
    """Collection-owning, short-lived producer for all fold runtime inputs."""
    if cfg.data is None:
        raise ValueError("LOO requires data")
    from hybrax.format.serialization import load_process_collection

    collection = load_process_collection(cfg.data.prepared)
    augmented_parents_ok, augmented_parent_messages = validate_augmented_parent_refs(
        collection
    )
    if not augmented_parents_ok:
        raise ValueError(
            "augmented parent validation failed:\n"
            + "\n".join(augmented_parent_messages)
        )
    folds = resolve_folds(
        collection,
        cfg.loo,
        int(cfg.train.seed),
        data_processes=cfg.data.processes,
    )
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=cfg.data.targets,
        target_source=cfg.data.target_source,
    )
    producer_data = ProducerCollectionData.from_collection(store, collection)
    parent_names = original_parent_processes(
        producer_data.process_order, producer_data.augmentation_parents
    )
    training_parent_collection = select_parent_collection(collection, parent_names)
    ensure_prepared_training_semantics(training_parent_collection)
    validate_for_training(
        training_parent_collection,
        strict=True,
        require_reaction_ode=True,
    )
    records = []
    for fold in folds:
        effective, _dir, _custom = _effective_fold_config(
            cfg, fold, output_dir, cfg.custom_py
        )
        scale_data = producer_data.select_training_parents(collection, fold.train)
        scales = _resolve_estimated_scales(
            custom_module=custom_module,
            runtime_data=scale_data,
            custom_cfg=effective,
        )
        records.append((_fold_record(fold), scales))
    return write_runtime_artifact(
        output_dir / _RUNTIME_ARTIFACT_NAME,
        training_data=store,
        parent_collection=training_parent_collection,
        augmentation_parents=producer_data.augmentation_parents,
        folds=tuple(records),
        rhs_names=RhsNames.from_rhs_ode(store.rhs_ode),
        identity_inputs={
            "run_fingerprint": _fingerprint(bundle_path, cfg.custom_py),
            "prepared_content_hash": content_hash(collection),
        },
    )


def _fold_from_record(fold: RuntimeArtifactFold) -> Fold:
    return Fold(fold.idx, fold.test, fold.train, fold.slug, fold.seed)


def _fold_harness_config(effective_cfg: RunConfig, fold: Fold, fold_dir: Path):
    return dataclasses.replace(
        train_harness_config_from_run_config(effective_cfg, run_dir=fold_dir),
        holdout_processes=fold.test,
        holdout_label="holdout",
    )


def _fold_inputs(
    effective_cfg: RunConfig, prepared_content_hash: str
) -> dict[str, Any]:
    """The fold's ``inputs`` block, mirroring a normal training run's.

    Every loadable model record must pin the input it was trained on: the shared
    reconstruction path requires ``inputs.prepared_input.content_hash`` and refuses
    to build hooks without it. A fold's ``data.prepared`` is the self-contained
    prepared copy at the LOO run root, and the hash is the producer-validated one,
    so a fold model loads exactly like a ``train`` run's.
    """
    return {
        "inputs": {
            "prepared_input": {
                "path": str(effective_cfg.data.prepared),
                "content_hash": prepared_content_hash,
            }
        }
    }


def _fold_runtime_metadata(identity: str, fold_idx: int) -> dict[str, Any]:
    return {
        "loo_runtime": {
            "artifact_format_version": FORMAT_VERSION,
            "artifact_identity": identity,
            "fold_id": fold_idx,
        }
    }


def prepare_single_fold_from_runtime_artifact(
    *,
    cfg: RunConfig,
    custom_module: Any | None,
    output_dir: Path,
    bundle_path: Path,
    artifact_path: Path,
    fold_idx: int,
) -> PreparedFold:
    """Prepare one fold solely from its validated runtime artifact."""
    expected_path, metadata = _runtime_metadata(
        output_dir, bundle_path=bundle_path, custom_path=cfg.custom_py
    )
    if artifact_path.resolve() != expected_path.resolve():
        raise ValueError("runtime artifact path does not match LOO run")
    artifact = load_runtime_artifact(artifact_path, fold_id=fold_idx)
    try:
        record = next(fold for fold in metadata.folds if fold.idx == fold_idx)
    except StopIteration as error:
        raise ValueError(f"runtime artifact has no fold {fold_idx}") from error
    if artifact.identity != metadata.identity or artifact.fold != record:
        raise ValueError("runtime artifact selected fold does not match manifest")
    fold = _fold_from_record(record)
    effective_cfg, fold_dir, fold_custom = _effective_fold_config(
        cfg, fold, output_dir, cfg.custom_py
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    if fold_custom is not None:
        shutil.copyfile(cfg.custom_py, fold_custom)
    config_json = fold_dir / "config.json"
    write_json(
        config_json,
        {
            "status": "running",
            "config": run_config_to_jsonable(effective_cfg),
            **_fold_inputs(
                effective_cfg, metadata.identity_inputs["prepared_content_hash"]
            ),
            **_fold_runtime_metadata(metadata.identity, record.idx),
        },
    )
    return PreparedFold(
        fold=fold,
        fold_seed=record.seed,
        fold_dir=fold_dir,
        fold_custom=fold_custom,
        config_json=config_json,
        effective_cfg=effective_cfg,
        training=prepare_training_from_runtime_artifact(
            artifact,
            config=_fold_harness_config(effective_cfg, fold, fold_dir),
            custom_module=custom_module,
            custom_cfg=effective_cfg,
        ),
    )


def prepare_single_fold(
    collection: BioProcessCollection,
    *,
    cfg: RunConfig,
    custom_module: Any | None,
    output_dir: str | Path,
    fold_idx: int,
    custom_py: str | Path | None = None,
) -> PreparedFold:
    """Resolve and prepare one fold without starting its training."""
    folds = resolve_folds(
        collection,
        cfg.loo,
        int(cfg.train.seed),
        data_processes=cfg.data.processes,
    )
    if not 0 <= fold_idx < len(folds):
        raise ValueError(
            f"--fold {fold_idx} out of range; {len(folds)} fold(s) resolved"
        )
    fold = folds[fold_idx]
    fold_seed = fold.seed
    effective_cfg, fold_dir, fold_custom = _effective_fold_config(
        cfg, fold, Path(output_dir), Path(custom_py) if custom_py is not None else None
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    if fold_custom is not None:
        shutil.copyfile(custom_py, fold_custom)
    config_json = fold_dir / "config.json"
    write_json(
        config_json,
        {
            "status": "running",
            "config": run_config_to_jsonable(effective_cfg),
            **_fold_inputs(effective_cfg, content_hash(collection)),
        },
    )
    harness_cfg = _fold_harness_config(effective_cfg, fold, fold_dir)
    logger.info(
        "LOO fold %d: slug=%s test=%s n_train=%d seed=%d",
        fold.idx,
        fold.slug,
        list(fold.test),
        len(fold.train),
        fold_seed,
    )
    training = prepare_training(
        collection,
        config=harness_cfg,
        custom_module=custom_module,
        run_config=effective_cfg,
    )
    return PreparedFold(
        fold=fold,
        fold_seed=fold_seed,
        fold_dir=fold_dir,
        fold_custom=fold_custom,
        config_json=config_json,
        effective_cfg=effective_cfg,
        training=training,
    )


def train_prepared_fold(prepared: PreparedFold) -> TrainedFold:
    """Train one prepared fold and retain only final-evaluation inputs."""
    fold = prepared.fold
    fold_seed = prepared.fold_seed
    fold_dir = prepared.fold_dir
    harness_cfg = prepared.training.config
    store = prepared.training.store
    target_names = tuple(prepared.training.loss_module.loss_names)
    train_result = train_collection(
        store,
        reaction_module=prepared.training.reaction_module,
        loss_module=prepared.training.loss_module,
        config=harness_cfg,
        optimizer=prepared.training.optimizer,
    )

    model_path = fold_dir / "trained_wrapper.eqx"
    save_model(train_result.trained_wrapper, model_path)
    train_result = dataclasses.replace(
        train_result,
        trained_wrapper=load_trained_wrapper(
            model_path,
            template=train_result.trained_wrapper,
        ),
    )
    last_loss = float(train_result.mean_loss_by_step[-1])
    meta = {
        "custom_py": "custom.py" if prepared.fold_custom is not None else None,
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
        "output_predictions": prepared.effective_cfg.output.predictions,
        "solver": {
            "max_steps": int(harness_cfg.solver_max_steps),
            "rtol": float(harness_cfg.solver_rtol),
            "atol": float(harness_cfg.solver_atol),
            "use_jump_ts": bool(harness_cfg.solver_use_jump_ts),
        },
        "training": {
            "epochs": int(harness_cfg.epochs),
            "batch_size": harness_cfg.batch_size,
            "seed": fold_seed,
            "final_mean_loss": last_loss,
        },
    }
    save_model_metadata(fold_dir / "trained_wrapper.meta.json", meta)
    return TrainedFold(
        fold=fold,
        fold_seed=fold_seed,
        fold_dir=fold_dir,
        config_json=prepared.config_json,
        config=harness_cfg,
        store=store,
        target_names=target_names,
        prediction_parent_process_names=(
            prepared.training.prediction_parent_process_names
        ),
        output_predictions=prepared.effective_cfg.output.predictions,
        train_result=train_result,
    )


def execute_trained_fold(trained: TrainedFold) -> FoldResult:
    """Evaluate and write one trained fold from its lean runtime inputs."""
    from .cli import _now_iso, _select_prediction_processes, _write_train_results

    fold = trained.fold
    config = trained.config
    eval_processes = tuple(dict.fromkeys((*fold.train, *fold.test)))
    prediction_processes = _select_prediction_processes(
        trained.output_predictions,
        eval_processes,
        trained.prediction_parent_process_names,
    )
    forward_result = evaluate_trained_wrapper(
        trained.train_result.trained_wrapper,
        trained.store,
        config=ForwardConfig(
            process_names=eval_processes,
            target_variable_order=config.target_variable_order,
            target_source=config.target_source,
            solver_max_steps=config.solver_max_steps,
            solver_rtol=config.solver_rtol,
            solver_atol=config.solver_atol,
            solver_use_jump_ts=config.solver_use_jump_ts,
        ),
        target_names=trained.target_names,
        training_process_names=fold.train,
        prediction_process_names=prediction_processes,
    )
    _write_train_results(
        output_dir=trained.fold_dir,
        forward_result=forward_result,
        prediction_processes=prediction_processes,
    )
    last_loss = float(trained.train_result.mean_loss_by_step[-1])
    update_json(
        trained.config_json,
        status="complete",
        finished_at=_now_iso(),
        updates_completed=trained.train_result.updates_completed,
        final_mean_loss=last_loss,
    )
    return FoldResult(
        fold=fold,
        fold_seed=trained.fold_seed,
        train_result=trained.train_result,
        forward_result=forward_result,
        fold_dir=trained.fold_dir,
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
    """Prepare and execute one fold for direct Python callers.

    The CLI uses the two-stage entry points directly so its collection-owning
    frame returns to collection-free execution before training starts.
    """
    prepared = prepare_single_fold(
        collection,
        cfg=cfg,
        custom_module=custom_module,
        output_dir=output_dir,
        fold_idx=fold_idx,
        custom_py=custom_py,
    )
    del collection
    trained = train_prepared_fold(prepared)
    del prepared
    return execute_trained_fold(trained)


# ---------------------------------------------------------------------------
# Cross-fold orchestration (orchestrator)
# ---------------------------------------------------------------------------


def _fold_complete(
    fold_dir: Path, artifact_identity: str, fold: RuntimeArtifactFold
) -> bool:
    config_path = fold_dir / "config.json"
    if not config_path.exists():
        return False
    try:
        document = load_json(config_path)
    except Exception as error:
        raise ValueError(f"invalid fold config for {fold.slug}") from error
    if not isinstance(document, dict):
        raise ValueError(f"invalid fold config for {fold.slug}")
    if "loo_runtime" not in document:
        return False
    runtime = document["loo_runtime"]
    expected = _fold_runtime_metadata(artifact_identity, fold.idx)["loo_runtime"]
    if (
        not isinstance(runtime, dict)
        or set(runtime) != set(expected)
        or type(runtime.get("artifact_format_version")) is not int
        or type(runtime.get("artifact_identity")) is not str
        or type(runtime.get("fold_id")) is not int
        or runtime["fold_id"] < 0
        or runtime != expected
    ):
        raise ValueError(f"fold runtime identity does not match manifest: {fold.slug}")
    return document.get("status") == "complete" and (fold_dir / "losses.csv").is_file()


def run_loo_cv(
    *,
    cfg: RunConfig,
    config_path: str | Path,
    output_dir: str | Path,
    resume: bool = False,
) -> LOOResult:
    """Dispatch artifact-backed workers and aggregate their fold outputs."""
    output_dir = Path(output_dir)
    config_path = Path(config_path).resolve()
    artifact_path, metadata = _runtime_metadata(
        output_dir, bundle_path=config_path, custom_path=cfg.custom_py
    )
    folds = tuple(_fold_from_record(fold) for fold in metadata.folds)
    n_folds = len(folds)
    loo_cfg = cfg.loo or LooConfig()
    n_cpu = os.cpu_count() or 1
    batch_size = cfg.train.batch_size

    def _eff_batch(fold: Fold) -> int:
        return min(len(fold.train), batch_size) if batch_size else len(fold.train)

    parallel, devices = compute_parallel_split(
        n_folds,
        n_cpu,
        loo_cfg.parallel_folds,
        devices_per_fold=loo_cfg.devices_per_fold,
        max_devices_per_fold=min(_eff_batch(fold) for fold in folds),
    )
    if loo_cfg.parallel_folds > n_folds:
        logger.info(
            "LOO: parallel_folds=%d exceeds the %d resolved fold(s); clamped to %d.",
            loo_cfg.parallel_folds,
            n_folds,
            parallel,
        )
    if resume:
        complete = tuple(
            _fold_complete(output_dir / "folds" / fold.slug, metadata.identity, record)
            for fold, record in zip(folds, metadata.folds, strict=True)
        )
        pending = [fold for fold, done in zip(folds, complete, strict=True) if not done]
        for fold in pending:
            fold_dir = output_dir / "folds" / fold.slug
            if fold_dir.exists():
                shutil.rmtree(fold_dir)
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
        _dispatch_pool(
            config_path, output_dir, artifact_path, pending, parallel, devices
        )

    summary_csv_path = output_dir / "loo_summary.csv"
    aggregate_json_path = output_dir / "loo_aggregate.json"
    aggregate = _write_summary_and_aggregate(
        folds=folds,
        output_dir=output_dir,
        summary_csv_path=summary_csv_path,
        aggregate_json_path=aggregate_json_path,
        base_seed=int(cfg.train.seed),
    )
    _plot_cross_fold_losses(folds=folds, output_dir=output_dir)
    return LOOResult(
        fold_dirs=tuple(output_dir / "folds" / fold.slug for fold in folds),
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
        meta = load_json(fold_dir / "trained_wrapper.meta.json")
        return float(meta["training"]["final_mean_loss"])
    except (OSError, KeyError, ValueError, TypeError):
        return float("nan")


def _read_fold_loss_history(
    fold_dir: Path,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Read per-step train and holdout loss from a fold's ``metrics.csv``.

    Returns ``(train_steps, train_loss, holdout_steps, holdout_loss)``. The
    holdout series is sparse -- only checkpoint steps are
    kept. Missing/unreadable files yield empty lists.
    """
    try:
        df = pd.read_csv(fold_dir / "metrics.csv")
        tr = df[["step", "mean_loss"]].dropna()
        train_steps = tr["step"].tolist()
        train_loss = tr["mean_loss"].tolist()
        if "holdout_loss" in df.columns:
            holdout = df[["step", "holdout_loss"]].dropna()
            holdout_steps = holdout["step"].tolist()
            holdout_loss = holdout["holdout_loss"].tolist()
        else:
            holdout_steps, holdout_loss = [], []
    except (OSError, KeyError, ValueError, pd.errors.EmptyDataError) as exc:
        # An optional end-of-run plot must never sink a completed LOO run, so a
        # missing or malformed metrics.csv just drops that fold from the figure.
        logger.warning("skipping loss history for %s: %s", fold_dir, exc)
        return [], [], [], []
    return train_steps, train_loss, holdout_steps, holdout_loss


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
    try:
        plot_cross_fold_loss_curves(fold_curves, output_dir / "loo_loss_curves.png")
    except Exception:
        # Fold results are complete; an optional PNG must not fail the LOO run.
        logger.exception("failed to write cross-fold loss curve")


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
            "fold_seed": fold.seed,
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
    write_json(aggregate_json_path, aggregate)
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
