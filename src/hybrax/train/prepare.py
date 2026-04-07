from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
import warnings

from bpbench.dataclasses import BioProcessCollection
from bpbench.serialization import (
    load_process_collection_json,
    save_process_collection_json,
)

from .controls import (
    BP_TRAIN_SAMPLE_ACC_NAME,
    select_control_sources,
)
from .defaults import (
    default_build_sample_acc_series,
    default_transform_process_collection,
)
from .utils import get_hook, load_custom_module, resolve_config
from .validation import (
    ensure_prepared_training_semantics,
    ensure_required_controls,
    summarize_process_semantics,
    validate_collection,
    validate_raw_collection,
)


@dataclass
class PrepareConfig:
    metadata_namespace: str = "bp_train"
    initial_grid_points: int = 16
    max_rel_error: float = 1e-4
    max_refinement_rounds: int = 8


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


def load_raw_collection(
    input_json: str | Path | BioProcessCollection,
    *,
    case_study: str | None = None,
) -> BioProcessCollection:
    """Load a BioProcessCollection from a file, object, or BenchmarkDataset.

    If the input is a BenchmarkDataset (contains ``case_studies``), the named
    case study is extracted.  When *case_study* is ``None`` the first case
    study is used.
    """
    if isinstance(input_json, BioProcessCollection):
        return input_json

    path = Path(input_json)
    collection = load_process_collection_json(path)
    if collection.processes:
        return collection

    # Try as BenchmarkDataset
    from bpbench.serialization import load_dataset

    dataset = load_dataset(path)
    if not dataset.case_studies:
        raise ValueError(f"No processes or case studies found in {path}")

    name = case_study or next(iter(dataset.case_studies))
    if name not in dataset.case_studies:
        available = list(dataset.case_studies.keys())
        raise ValueError(
            f"Case study {name!r} not found; available: {available}"
        )
    cs = dataset.case_studies[name]
    return BioProcessCollection(
        processes=cs.processes,
        metadata=dataset.metadata,
    )


def _warn_on_validation_report(validation_report: dict[str, dict[str, object]]) -> None:
    failed = [name for name, entry in validation_report.items() if not entry["ok"]]
    if failed:
        warnings.warn(
            f"bpbench validation reported non-OK status for {len(failed)} process(es); "
            "see metadata['bp_train']['bpbench_validation_raw'] for details",
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
) -> dict[str, dict[str, object]]:
    provenance: dict[str, dict[str, object]] = {}

    for process_name, prepared_summary in prepared_snapshots.items():
        old_name = reverse_rename_map.get(process_name, process_name)
        raw_summary = raw_snapshots.get(old_name, prepared_summary)
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
    process_sources: dict[str, list[Any]],
    sample_sources: dict[str, Any],
    *,
    require_consistent_controls: bool,
) -> None:
    reference_names: list[str] | None = None

    for process_name, sources in process_sources.items():
        control_names = [source.name for source in sources]
        if BP_TRAIN_SAMPLE_ACC_NAME in control_names:
            raise ValueError(
                f"{process_name}: reserved control name {BP_TRAIN_SAMPLE_ACC_NAME} "
                "may not be produced by transform_process_collection"
            )
        if len(control_names) != len(set(control_names)):
            raise ValueError(
                f"{process_name}: duplicate control names after transforms"
            )
        if sample_sources[process_name].name != BP_TRAIN_SAMPLE_ACC_NAME:
            raise ValueError(
                f"{process_name}: sample-acc source "
                f"must be named {BP_TRAIN_SAMPLE_ACC_NAME}"
            )

        if require_consistent_controls:
            if reference_names is None:
                reference_names = control_names
            elif control_names != reference_names:
                raise ValueError(
                    f"{process_name}: control names/order differ across processes; "
                    "either make hooks consistent or disable "
                    "require_consistent_controls"
                )


def prepare_artifact(
    input_json: str | Path | BioProcessCollection,
    output_json: str | Path,
    *,
    custom_py: str | Path | None = None,
    config: dict[str, Any] | None = None,
    case_study: str | None = None,
) -> BioProcessCollection:
    input_path = (
        None if isinstance(input_json, BioProcessCollection) else Path(input_json)
    )
    output_path = Path(output_json)

    custom_module = load_custom_module(custom_py)
    user_config = resolve_config(custom_module, config)

    defaults = asdict(PrepareConfig())
    defaults.update(user_config)
    resolved_config = defaults

    raw_collection = load_raw_collection(input_json, case_study=case_study)
    validation_report = validate_raw_collection(
        raw_collection,
        strict=bool(resolved_config.get("strict_bpbench_validation", False)),
    )
    if not bool(resolved_config.get("strict_bpbench_validation", False)):
        _warn_on_validation_report(validation_report)

    collection = deepcopy(raw_collection)

    transform_process_collection = get_hook(
        custom_module,
        "transform_process_collection",
        default_transform_process_collection,
    )
    build_sample_acc = get_hook(
        custom_module,
        "build_sample_acc_series",
        default_build_sample_acc_series,
    )
    raw_semantics = {
        process_name: summarize_process_semantics(process)
        for process_name, process in collection.processes.items()
    }
    for process_name, process in collection.processes.items():
        process.metadata._pre_transform_key = process_name
    collection = transform_process_collection(collection, resolved_config)
    reverse_rename_map = {}
    for process_name, process in collection.processes.items():
        old_name = process.metadata._pre_transform_key
        del process.metadata._pre_transform_key
        if old_name != process_name:
            reverse_rename_map[process_name] = old_name

    prepared_semantics: dict[str, dict[str, object]] = {}
    for process_name, process in collection.processes.items():
        prepared_semantics[process_name] = summarize_process_semantics(process)

    semantics_validation_report = ensure_prepared_training_semantics(collection)
    prepared_validation_report = validate_collection(collection, strict=True)
    semantics_provenance = _build_semantics_provenance(
        raw_snapshots=raw_semantics,
        prepared_snapshots=prepared_semantics,
        reverse_rename_map=reverse_rename_map,
    )

    process_sources: dict[str, list[Any]] = {}
    sample_sources: dict[str, Any] = {}

    required_control_names = resolved_config.get("required_control_names", [])
    if isinstance(required_control_names, dict):
        required_control_names_by_process = required_control_names
    else:
        required_control_names_by_process = {
            name: list(required_control_names) for name in collection.processes
        }

    for process_name, process in collection.processes.items():
        control_sources = select_control_sources(
            process_name=process_name,
            process=process,
            config=resolved_config,
        )
        ensure_required_controls(
            process_name=process_name,
            available_control_names=[source.name for source in control_sources],
            required_control_names=required_control_names_by_process.get(
                process_name, []
            ),
        )
        sample_source = build_sample_acc(
            process,
            process_name,
            collection.metadata or {},
            resolved_config,
        )
        process_sources[process_name] = control_sources
        sample_sources[process_name] = sample_source

    _validate_prepared_control_contract(
        process_sources=process_sources,
        sample_sources=sample_sources,
        require_consistent_controls=bool(
            resolved_config.get("require_consistent_controls", True)
        ),
    )

    process_order = list(collection.processes.keys())

    source_hash = _sha256_hex(_read_bytes(input_path))
    custom_hash = _sha256_hex(_read_bytes(custom_py))

    existing_metadata = dict(collection.metadata or {})
    bp_train_metadata: dict[str, Any] = {
        "prepared_at": _utc_now_iso(),
        "source_input_path": None if input_path is None else str(input_path),
        "source_input_sha256": source_hash,
        "custom_py_sha256": custom_hash,
        "transform_hooks": {
            "transform_process_collection": getattr(
                transform_process_collection,
                "__name__",
                str(transform_process_collection),
            ),
            "build_sample_acc_series": getattr(
                build_sample_acc, "__name__", str(build_sample_acc)
            ),
        },
        "dynamic_volume": True,
        "bpbench_validation": prepared_validation_report,
        "bpbench_validation_raw": validation_report,
        "bpbench_validation_prepared": prepared_validation_report,
        "prepared_semantics_validation": semantics_validation_report,
        "semantics_provenance": {
            "processes": semantics_provenance,
        },
        "process_order": process_order,
        "runtime_controls_config": {
            "initial_grid_points": int(resolved_config["initial_grid_points"]),
            "max_rel_error": float(resolved_config["max_rel_error"]),
            "max_refinement_rounds": int(resolved_config["max_refinement_rounds"]),
            "require_consistent_controls": bool(
                resolved_config.get("require_consistent_controls", True)
            ),
        },
        "processes": {},
    }

    for process_name, process in collection.processes.items():
        control_sources = process_sources[process_name]
        sample_source = sample_sources[process_name]
        local_control_names = [source.name for source in control_sources] + [
            sample_source.name
        ]
        bp_train_metadata["processes"][process_name] = {
            "local_control_names": local_control_names,
            "control_metadata": {
                source.name: source.metadata
                for source in [*control_sources, sample_source]
            },
            "sample_acc_name": BP_TRAIN_SAMPLE_ACC_NAME,
            "sample_acc_source": {
                "times": [float(v) for v in sample_source.times.tolist()],
                "values": [float(v) for v in sample_source.values.tolist()],
                "step_ts": [float(v) for v in sample_source.step_ts],
                "metadata": dict(sample_source.metadata),
            },
        }

    existing_metadata[resolved_config["metadata_namespace"]] = bp_train_metadata
    collection.metadata = existing_metadata

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_process_collection_json(collection, output_path)
    return collection
