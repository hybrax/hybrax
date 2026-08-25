"""The ``prepare`` command: raw data in, a validated ``prepared.json`` artifact out.

``prepare_artifact`` is the entry point; everything else in this module
supports it (raw-collection loading, semantics-diff provenance, and the
prepared-control-contract validation shared by every process).
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from hybrax.format import validate_augmented_parent_refs
from hybrax.format.dataclasses import (
    BioProcess,
    BioProcessCollection,
    StaticVariable,
    TimeSeries,
)
from hybrax.format.serialization import (
    load_process_collection,
    save_process_collection,
)

from .augmentation import augment_process_collection
from .augmentation_plot import AUGMENTATION_PLOT_FILENAME, render_augmentation_plot
from .constants import METADATA_NAMESPACE
from .controls import ControlSourceBundle, select_control_sources
from .defaults import default_transform_process_collection
from .run_config import LoadedRunConfig
from .serialization import content_hash, environment_versions, write_json
from .utils import get_hook, split_hooks_by_customization
from .validate import (
    ensure_prepared_training_semantics,
    ensure_required_controls,
    summarize_process_semantics,
    validate_for_training,
)

logger = logging.getLogger(__name__)


def _read_bytes(path: str | Path | None) -> bytes | None:
    if path is None:
        return None
    return Path(path).read_bytes()


def _sha256_hex(data: bytes | None) -> str | None:
    if data is None:
        return None
    return sha256(data).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _portable_input_path(input_path: Path | None, output_path: Path) -> str | None:
    if input_path is None:
        return None
    if not input_path.is_absolute():
        return str(input_path)
    output_dir = output_path.parent.resolve()
    return os.path.relpath(input_path, output_dir)


def load_raw_collection(
    input_json: str | Path | BioProcessCollection,
) -> BioProcessCollection:
    """Load a BioProcessCollection from a file or object.

    Accepts a ``BioProcessCollection`` (returned as-is) or a path to a JSON
    file holding one.
    """
    if isinstance(input_json, BioProcessCollection):
        return input_json
    return load_process_collection(Path(input_json))


def _warn_on_validation_report(validation_report: dict[str, dict[str, object]]) -> None:
    failed = [name for name, entry in validation_report.items() if not entry["ok"]]
    if failed:
        warnings.warn(
            "hybrax.format validation reported non-OK status for "
            f"{len(failed)} process(es); "
            f"see metadata[{METADATA_NAMESPACE!r}]['format_validation_raw'] "
            "for details",
            stacklevel=2,
        )


def _added_names(before: list[str], after: list[str]) -> list[str]:
    before_set = set(before)
    return [name for name in after if name not in before_set]


def _modified_names(
    before: dict[str, dict[str, object]],
    after: dict[str, dict[str, object]],
) -> list[str]:
    shared = set(before) & set(after)
    return sorted(name for name in shared if before[name] != after[name])


def _build_semantics_provenance(
    raw_snapshots: dict[str, dict[str, object]],
    prepared_snapshots: dict[str, dict[str, object]],
    reverse_rename_map: dict[str, str],
    augmentation_created_names: set[str],
) -> dict[str, dict[str, object]]:
    provenance: dict[str, dict[str, object]] = {}

    for process_name, prepared_summary in prepared_snapshots.items():
        old_name = reverse_rename_map.get(process_name, process_name)
        raw_summary = raw_snapshots.get(old_name)
        if raw_summary is None:
            changed_by_hooks = [
                "augmentation"
                if process_name in augmentation_created_names
                else "transform_process_collection"
            ]
            provenance[process_name] = {
                "raw": None,
                "prepared": prepared_summary,
                "changed_by_hooks": changed_by_hooks,
                "reactor_components_added": prepared_summary["reactor_component_names"],
                "reactor_components_modified": [],
                "feed_components_added": prepared_summary[
                    "feed_component_names_by_change"
                ],
                "feed_components_modified": {},
            }
            continue
        changed_by_hooks: list[str] = []
        # A pure rename (no semantic changes) should still be flagged.
        was_renamed = process_name in reverse_rename_map
        if was_renamed or prepared_summary != raw_summary:
            changed_by_hooks.append("transform_process_collection")

        raw_feed = raw_summary["feed_component_names_by_change"]
        prepared_feed = prepared_summary["feed_component_names_by_change"]
        feed_components_added = {
            change_name: _added_names(
                raw_feed.get(change_name, []),
                prepared_feed.get(change_name, []),
            )
            for change_name in sorted(prepared_feed)
        }
        feed_components_added = {
            change_name: names
            for change_name, names in feed_components_added.items()
            if names
        }
        raw_feed_details = raw_summary["feed_component_details_by_change"]
        prepared_feed_details = prepared_summary["feed_component_details_by_change"]
        feed_components_modified = {
            change_name: _modified_names(
                raw_feed_details.get(change_name, {}),
                prepared_feed_details.get(change_name, {}),
            )
            for change_name in sorted(prepared_feed_details)
        }
        feed_components_modified = {
            change_name: names
            for change_name, names in feed_components_modified.items()
            if names
        }

        provenance[process_name] = {
            "raw": raw_summary,
            "prepared": prepared_summary,
            "changed_by_hooks": changed_by_hooks,
            "reactor_components_added": _added_names(
                raw_summary["reactor_component_names"],
                prepared_summary["reactor_component_names"],
            ),
            "reactor_components_modified": _modified_names(
                raw_summary["reactor_component_details"],
                prepared_summary["reactor_component_details"],
            ),
            "feed_components_added": feed_components_added,
            "feed_components_modified": feed_components_modified,
        }

    return provenance


def _validate_prepared_control_contract(
    process_bundles: dict[str, ControlSourceBundle],
    *,
    require_consistent_controls: bool,
) -> None:
    reference_categorised: (
        tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None
    ) = None

    for process_name, bundle in process_bundles.items():
        control_names = list(bundle.all_names)
        if len(control_names) != len(set(control_names)):
            raise ValueError(
                f"{process_name}: duplicate control names after transforms"
            )

        if require_consistent_controls:
            categorised = (
                bundle.name_controlled_Inflows,
                bundle.name_controlled_Outflows,
                bundle.name_controlled_PVs,
            )
            if reference_categorised is None:
                reference_categorised = categorised
            elif categorised != reference_categorised:
                raise ValueError(
                    f"{process_name}: categorised control layout differs across "
                    "processes; either make hooks consistent or disable "
                    "require_consistent_controls"
                )


def prepare_artifact(
    loaded_config: LoadedRunConfig,
    output_dir: str | Path,
) -> BioProcessCollection:
    """Run the full ``prepare`` pipeline and write ``output_dir/prepared.json``.

    Loads the raw collection at ``config.prepare.raw_input``, validates it,
    applies the ``transform_process_collection`` hook (default: identity),
    generates augmented children via
    :func:`~hybrax.train.augmentation.augment_process_collection`,
    re-validates the result under strict training semantics, extracts and
    validates each process's control sources, and assembles the
    ``hybrax_train`` metadata block (provenance, semantics diffs vs. the raw
    input, per-process control names, a stable ``content_hash``). Writes the
    prepared collection to ``prepared.json``, a standalone
    ``prepare_config.json`` record of the run, and (unless disabled) a
    per-process control diagnostics plot and, if augmentation ran, an
    augmentation preview plot.

    Args:
        loaded_config: A :func:`~hybrax.train.run_config.load_prepare_config`
            result; ``loaded_config.config.prepare`` must be set.
        output_dir: Directory ``prepared.json`` and the other prepare-owned
            files are written into. The CLI owns all ``--overwrite`` handling
            (both guarding an existing ``prepared.json`` and clearing
            ``output_dir``) before calling this function.

    Returns:
        The prepared :class:`~hybrax.format.dataclasses.BioProcessCollection`
        (the same object written to ``prepared.json``).

    Raises:
        ValueError: If ``loaded_config.config.prepare`` is unset, a
            transform/augmentation hook produces an ambiguous rename, or the
            prepared collection fails required-control or augmented-parent
            validation.
    """
    config = loaded_config.config
    custom_module = loaded_config.custom_module
    custom_hash = loaded_config.custom_py_sha256
    prepare = config.prepare
    if prepare is None:
        raise ValueError("prepare_artifact requires a prepare config section")

    input_path = prepare.raw_input
    # `--output-dir` is prepare's own exclusively-owned directory (the CLI
    # deletes it wholesale on `--overwrite`, like train/loo/forward): prepared.json
    # + prepare_config.json + optional augmented-data.png + prepare_diagnostics/.
    output_dir = Path(output_dir)
    output_path = output_dir / "prepared.json"

    raw_collection = load_raw_collection(input_path)
    validation_report = validate_for_training(
        raw_collection,
        strict=prepare.strict_format_validation,
    )
    if not prepare.strict_format_validation:
        _warn_on_validation_report(validation_report)

    collection = deepcopy(raw_collection)

    transform_process_collection = get_hook(
        custom_module,
        "transform_process_collection",
        default_transform_process_collection,
    )
    augment_state_values = get_hook(custom_module, "augment_state_values", None)
    customized_hooks, default_hooks = split_hooks_by_customization(
        custom_module,
        ("transform_process_collection", "augment_state_values"),
    )
    logger.info("prepare hooks detected: %s", ", ".join(customized_hooks) or "none")
    logger.info("prepare hooks default: %s", ", ".join(default_hooks) or "none")
    raw_semantics = {
        process_name: summarize_process_semantics(process)
        for process_name, process in collection.processes.items()
    }
    for process_name, process in collection.processes.items():
        process.metadata._pre_transform_key = process_name
    collection = transform_process_collection(collection, config)
    transformed_process_names = set(collection.processes)
    collection = augment_process_collection(
        collection,
        config,
        augment_state_values,
    )
    augmentation_created_names = set(collection.processes) - transformed_process_names
    tagged_processes: list[tuple[str, str]] = []
    for process_name, process in collection.processes.items():
        old_name = getattr(process.metadata, "_pre_transform_key", None)
        if old_name is not None:
            del process.metadata._pre_transform_key
            tagged_processes.append((process_name, old_name))

    tag_claims_by_old_name: dict[str, list[str]] = {}
    for process_name, old_name in tagged_processes:
        tag_claims_by_old_name.setdefault(old_name, []).append(process_name)

    tag_owner_by_old_name: dict[str, str] = {}
    for old_name, process_names in tag_claims_by_old_name.items():
        if old_name in process_names:
            tag_owner_by_old_name[old_name] = old_name
        elif len(process_names) == 1:
            tag_owner_by_old_name[old_name] = process_names[0]
        else:
            raise ValueError(
                f"ambiguous pre-transform provenance tag {old_name!r}: "
                f"claimed by {process_names}"
            )
    reverse_rename_map = {
        process_name: old_name
        for old_name, process_name in tag_owner_by_old_name.items()
        if old_name != process_name
    }

    prepared_semantics: dict[str, dict[str, object]] = {}
    for process_name, process in collection.processes.items():
        prepared_semantics[process_name] = summarize_process_semantics(process)

    semantics_validation_report = ensure_prepared_training_semantics(collection)
    augmented_parents_ok, augmented_parent_results = validate_augmented_parent_refs(
        collection
    )
    if not augmented_parents_ok:
        raise ValueError(
            "augmented parent validation failed:\n"
            + "\n".join(message for _, message in augmented_parent_results)
        )
    prepared_validation_report = validate_for_training(
        collection,
        strict=True,
        require_biological_ode=True,
    )
    semantics_provenance = _build_semantics_provenance(
        raw_snapshots=raw_semantics,
        prepared_snapshots=prepared_semantics,
        reverse_rename_map=reverse_rename_map,
        augmentation_created_names=augmentation_created_names,
    )

    process_bundles: dict[str, ControlSourceBundle] = {}

    required_control_names = prepare.required_control_names
    if isinstance(required_control_names, dict):
        required_control_names_by_process = required_control_names
    else:
        required_control_names_by_process = {
            name: list(required_control_names) for name in collection.processes
        }

    for process_name, process in collection.processes.items():
        bundle = select_control_sources(process)
        ensure_required_controls(
            process_name=process_name,
            available_control_names=list(bundle.all_names),
            required_control_names=required_control_names_by_process.get(
                process_name, []
            ),
        )
        process_bundles[process_name] = bundle

    _validate_prepared_control_contract(
        process_bundles=process_bundles,
        require_consistent_controls=prepare.require_consistent_controls,
    )

    process_order = list(collection.processes.keys())

    source_hash = _sha256_hex(_read_bytes(input_path))

    existing_metadata = dict(collection.metadata or {})
    transform_hooks = {
        "transform_process_collection": getattr(
            transform_process_collection,
            "__name__",
            str(transform_process_collection),
        ),
    }
    if augment_state_values is not None:
        transform_hooks["augment_state_values"] = getattr(
            augment_state_values,
            "__name__",
            str(augment_state_values),
        )

    hybrax_train_metadata: dict[str, Any] = {
        "prepared_at": _utc_now_iso(),
        "source_input_path": _portable_input_path(input_path, output_path),
        "source_input_sha256": source_hash,
        "custom_py_sha256": custom_hash,
        "transform_hooks": transform_hooks,
        "dynamic_volume": True,
        "format_validation": prepared_validation_report,
        "format_validation_raw": validation_report,
        "format_validation_prepared": prepared_validation_report,
        "prepared_semantics_validation": semantics_validation_report,
        "semantics_provenance": {
            "processes": semantics_provenance,
        },
        "process_order": process_order,
        "processes": {},
    }

    for process_name, process in collection.processes.items():
        bundle = process_bundles[process_name]
        hybrax_train_metadata["processes"][process_name] = {
            "name_controlled_Inflows": list(bundle.name_controlled_Inflows),
            "name_controlled_Outflows": list(bundle.name_controlled_Outflows),
            "name_controlled_PVs": list(bundle.name_controlled_PVs),
            "control_metadata": {
                source.name: source.metadata for source in bundle.all_sources
            },
        }

    # FAIR provenance sub-block — excluded from content_hash (see
    # serialization._VOLATILE_NS_KEYS), so re-preparing identical data yields an
    # identical content_hash. Carries the resolved PrepareConfig, the custom.py
    # file hash, the raw-input hash, package versions, and a timestamp.
    hybrax_train_metadata["provenance"] = {
        "prepared_at": _utc_now_iso(),
        "prepare_config": prepare.model_dump(mode="json"),
        "custom_py_file_hash": (f"sha256:{custom_hash}" if custom_hash else None),
        "raw_input_sha256": (f"sha256:{source_hash}" if source_hash else None),
        "environment": environment_versions(),
    }

    existing_metadata[METADATA_NAMESPACE] = hybrax_train_metadata
    collection.metadata = existing_metadata

    # Record the prepared collection's own stable content_hash (computed with the
    # provenance block excluded, so it is self-consistent and re-prepare-stable).
    hybrax_train_metadata["provenance"]["content_hash"] = content_hash(collection)

    output_dir.mkdir(parents=True, exist_ok=True)
    augmentation_plot_path = output_dir / AUGMENTATION_PLOT_FILENAME
    if prepare.augmentation is not None:
        render_augmentation_plot(
            collection,
            prepare.augmentation,
            augmentation_plot_path,
        )
    elif augmentation_plot_path.exists():
        augmentation_plot_path.unlink()

    save_process_collection(collection, output_path)
    # Standalone, inspectable record of how this prepare ran (clash-free with
    # train's config.json) — the hybrax.train provenance/metadata block, without the
    # bulk collection.
    write_json(output_dir / "prepare_config.json", hybrax_train_metadata, default=str)

    if prepare.diagnostics:
        try:
            _render_control_diagnostics(collection, process_bundles, output_dir)
        except Exception as exc:  # noqa: BLE001 — diagnostics are auxiliary
            logger.warning("control diagnostics rendering failed: %r", exc)

    return collection


def _raw_control_samples(
    process: BioProcess, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Raw measured samples for a control (TimeSeries knots or a static level)."""
    if name in process.process_variables:
        value = process.process_variables[name].values
    else:
        value = process.volume.volume_changes[name].values
    if isinstance(value, TimeSeries) and value.times is not None:
        return np.asarray(value.times, dtype=float), np.asarray(
            value.values, dtype=float
        )
    if isinstance(value, StaticVariable):
        t0, t1 = float(process.time_axis.start), float(process.time_axis.end)
        return np.asarray([t0, t1]), np.asarray([float(value.value)] * 2)
    return np.asarray([]), np.asarray([])


def _control_unit(process: BioProcess, name: str) -> str:
    if name in process.process_variables:
        return str(getattr(process.process_variables[name], "unit", ""))
    vc = process.volume.volume_changes.get(name)
    return str(getattr(vc, "unit", "")) if vc is not None else ""


def _render_control_diagnostics(
    collection: BioProcessCollection,
    process_bundles: dict[str, ControlSourceBundle],
    output_dir: Path,
) -> None:
    """Build per-process control diagnostics and render them (prepare is one-shot)."""
    import shutil

    from .controls_store import ControlsStore
    from .postprocessing import (
        ControlDiagnostic,
        ProcessControlDiagnostics,
        render_control_diagnostics,
    )

    # Idempotent re-run into a directory the CLI's --overwrite guard did not clear
    # (a first-ever run landing on stray content from a previous failed attempt,
    # one that crashed before writing prepared.json): clear stale plots (e.g.
    # fewer processes than the last attempt) before rebuilding them.
    diag_dir = output_dir / "prepare_diagnostics"
    if diag_dir.exists():
        shutil.rmtree(diag_dir)
    store = ControlsStore.from_collection(collection)
    for name, process in collection.processes.items():
        bundle = process_bundles[name]
        per = store.get_controls(name)
        spline_grid = np.asarray(per.spline_breaks, dtype=float)
        grid = np.unique(
            np.concatenate(
                [
                    spline_grid[np.isfinite(spline_grid)],
                    np.asarray(per.active_linear_grid, dtype=float),
                ]
            )
        )
        t0 = float(process.time_axis.start)
        t1 = float(process.time_axis.end)
        fine = np.linspace(t0, t1, 600)
        diags = []
        for cname in bundle.all_names:
            src = bundle.sources_by_name[cname]
            curve = np.asarray(src.evaluator(fine), dtype=float).reshape(-1)
            raw_t, raw_v = _raw_control_samples(process, cname)
            if raw_t.size:
                ref = np.asarray(src.evaluator(raw_t), dtype=float).reshape(-1)
                max_rel_dev = float(
                    np.max(np.abs(ref - raw_v) / np.maximum(np.abs(raw_v), 1e-9))
                )
            else:
                max_rel_dev = 0.0
            diags.append(
                ControlDiagnostic(
                    name=cname,
                    unit=_control_unit(process, cname),
                    raw_times=raw_t,
                    raw_values=raw_v,
                    curve_t=fine,
                    curve_values=curve,
                    grid_t=grid,
                    is_spline=src.metadata.get("source") == "spline",
                    max_rel_dev=max_rel_dev,
                )
            )
        render_control_diagnostics(
            ProcessControlDiagnostics(
                process_name=name,
                time_unit=str(process.time_axis.unit),
                controls=tuple(diags),
            ),
            diag_dir,
        )
