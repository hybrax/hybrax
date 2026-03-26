from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
import warnings

from bpbench.dataclasses import BioProcessCollection
from bpbench.serialization import load_process_collection_json, save_process_collection_json

from .controls import (
    BP_TRAIN_SAMPLE_ACC_NAME,
    build_dense_payload,
    build_sample_acc_source_default,
    compute_signal_spreads,
    select_control_sources,
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


def load_raw_collection(input_json: str | Path | BioProcessCollection) -> BioProcessCollection:
    if isinstance(input_json, BioProcessCollection):
        return input_json
    return load_process_collection_json(Path(input_json))


def _default_transform_controls(process, config):
    return process


def _default_transform_states(process, config):
    return process


def _default_build_sample_acc(process, process_name, collection_metadata, config):
    return build_sample_acc_source_default(process)


def _pad_process_payload(
    payload: dict[str, Any],
    max_grid_length: int,
    global_control_names: list[str],
    max_step_ts_length: int,
) -> dict[str, Any]:
    grid = list(payload["grid"])
    values = [list(row) for row in payload["values"]]
    derivatives = [list(row) for row in payload["derivatives"]]
    step_ts = list(payload["step_ts"])

    grid_length = len(grid)
    local_control_names = list(payload["control_names"])
    control_count = len(local_control_names)
    global_index = {name: idx for idx, name in enumerate(global_control_names)}
    max_controls = len(global_control_names)

    padded_grid = grid + [0.0] * (max_grid_length - grid_length)
    grid_mask = [True] * grid_length + [False] * (max_grid_length - grid_length)

    padded_values = []
    padded_derivatives = []
    for row, deriv_row in zip(values, derivatives, strict=False):
        global_row = [0.0] * max_controls
        global_deriv_row = [0.0] * max_controls
        for local_idx, control_name in enumerate(local_control_names):
            target_idx = global_index[control_name]
            global_row[target_idx] = row[local_idx]
            global_deriv_row[target_idx] = deriv_row[local_idx]
        padded_values.append(global_row)
        padded_derivatives.append(global_deriv_row)

    zero_row = [0.0] * max_controls
    for _ in range(max_grid_length - grid_length):
        padded_values.append(list(zero_row))
        padded_derivatives.append(list(zero_row))

    control_mask = [False] * max_controls
    for control_name in local_control_names:
        control_mask[global_index[control_name]] = True

    padded_step_ts = step_ts + [0.0] * (max_step_ts_length - len(step_ts))
    step_ts_mask = [True] * len(step_ts) + [False] * (max_step_ts_length - len(step_ts))

    return {
        "dense_grid": padded_grid,
        "dense_grid_mask": grid_mask,
        "control_values": padded_values,
        "control_derivatives": padded_derivatives,
        "control_mask": control_mask,
        "step_ts": padded_step_ts,
        "step_ts_mask": step_ts_mask,
        "grid_length": grid_length,
        "control_count": control_count,
    }


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
    controls_snapshots: dict[str, dict[str, object]],
    prepared_snapshots: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    provenance: dict[str, dict[str, object]] = {}

    for process_name, raw_summary in raw_snapshots.items():
        after_controls = controls_snapshots[process_name]
        prepared_summary = prepared_snapshots[process_name]
        changed_by_hooks: list[str] = []
        if after_controls != raw_summary:
            changed_by_hooks.append("transform_controls")
        if prepared_summary != after_controls:
            changed_by_hooks.append("transform_states")

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
                "may not be produced by transform_controls"
            )
        if len(control_names) != len(set(control_names)):
            raise ValueError(f"{process_name}: duplicate control names after transforms")
        if sample_sources[process_name].name != BP_TRAIN_SAMPLE_ACC_NAME:
            raise ValueError(f"{process_name}: sample-acc source must be named {BP_TRAIN_SAMPLE_ACC_NAME}")

        if require_consistent_controls:
            if reference_names is None:
                reference_names = control_names
            elif control_names != reference_names:
                raise ValueError(
                    f"{process_name}: control names/order differ across processes; "
                    "either make hooks consistent or disable require_consistent_controls"
                )


def _build_global_control_axis(
    process_order: list[str],
    process_sources: dict[str, list[Any]],
) -> list[str]:
    seen: set[str] = set()
    global_names: list[str] = []

    for process_name in process_order:
        for source in process_sources[process_name]:
            if source.name in seen:
                continue
            seen.add(source.name)
            global_names.append(source.name)

    global_names.append(BP_TRAIN_SAMPLE_ACC_NAME)
    return global_names


def prepare_artifact(
    input_json: str | Path | BioProcessCollection,
    output_json: str | Path,
    *,
    custom_py: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> BioProcessCollection:
    input_path = None if isinstance(input_json, BioProcessCollection) else Path(input_json)
    output_path = Path(output_json)

    custom_module = load_custom_module(custom_py)
    user_config = resolve_config(custom_module, config)

    defaults = asdict(PrepareConfig())
    defaults.update(user_config)
    resolved_config = defaults

    raw_collection = load_raw_collection(input_json)
    validation_report = validate_raw_collection(
        raw_collection,
        strict=bool(resolved_config.get("strict_bpbench_validation", False)),
    )
    if not bool(resolved_config.get("strict_bpbench_validation", False)):
        _warn_on_validation_report(validation_report)

    collection = deepcopy(raw_collection)

    transform_controls = get_hook(custom_module, "transform_controls", _default_transform_controls)
    transform_states = get_hook(custom_module, "transform_states", _default_transform_states)
    build_sample_acc = get_hook(
        custom_module,
        "build_sample_acc_series",
        _default_build_sample_acc,
    )

    raw_semantics = {
        process_name: summarize_process_semantics(process)
        for process_name, process in collection.processes.items()
    }
    controls_semantics: dict[str, dict[str, object]] = {}
    prepared_semantics: dict[str, dict[str, object]] = {}

    for process_name, process in list(collection.processes.items()):
        process = transform_controls(process, resolved_config)
        controls_semantics[process_name] = summarize_process_semantics(process)
        process = transform_states(process, resolved_config)
        prepared_semantics[process_name] = summarize_process_semantics(process)
        collection.processes[process_name] = process

    semantics_validation_report = ensure_prepared_training_semantics(collection)
    prepared_validation_report = validate_collection(collection, strict=True)
    semantics_provenance = _build_semantics_provenance(
        raw_snapshots=raw_semantics,
        controls_snapshots=controls_semantics,
        prepared_snapshots=prepared_semantics,
    )

    process_sources: dict[str, list[Any]] = {}
    sample_sources: dict[str, Any] = {}

    required_control_names = resolved_config.get("required_control_names", [])
    if isinstance(required_control_names, dict):
        required_control_names_by_process = required_control_names
    else:
        required_control_names_by_process = {
            name: list(required_control_names)
            for name in collection.processes
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
            required_control_names=required_control_names_by_process.get(process_name, []),
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
        require_consistent_controls=bool(resolved_config.get("require_consistent_controls", True)),
    )

    spread_inputs = {
        process_name: sources + [sample_sources[process_name]]
        for process_name, sources in process_sources.items()
    }
    signal_spreads = compute_signal_spreads(spread_inputs)

    process_order = list(collection.processes.keys())
    global_control_names = _build_global_control_axis(
        process_order=process_order,
        process_sources=process_sources,
    )

    process_payloads: dict[str, dict[str, Any]] = {}
    max_grid_length = 0
    max_step_ts_length = 0

    for process_name, process in collection.processes.items():
        sources = list(process_sources[process_name]) + [sample_sources[process_name]]
        payload = build_dense_payload(
            process=process,
            sources=sources,
            spreads=signal_spreads,
            config=resolved_config,
        )
        process_payloads[process_name] = {
            "control_names": [source.name for source in sources],
            "control_metadata": {source.name: source.metadata for source in sources},
            "sample_acc_index": global_control_names.index(BP_TRAIN_SAMPLE_ACC_NAME),
            "step_ts": payload["step_ts"],
            "grid": payload["grid"],
            "values": payload["values"],
            "derivatives": payload["derivatives"],
        }
        max_grid_length = max(max_grid_length, len(payload["grid"]))
        max_step_ts_length = max(max_step_ts_length, len(payload["step_ts"]))

    source_hash = _sha256_hex(_read_bytes(input_path))
    custom_hash = _sha256_hex(_read_bytes(custom_py))

    existing_metadata = dict(collection.metadata or {})
    bp_train_metadata: dict[str, Any] = {
        "prepared_at": _utc_now_iso(),
        "source_input_path": None if input_path is None else str(input_path),
        "source_input_sha256": source_hash,
        "custom_py_sha256": custom_hash,
        "transform_hooks": {
            "transform_controls": getattr(transform_controls, "__name__", str(transform_controls)),
            "transform_states": getattr(transform_states, "__name__", str(transform_states)),
            "build_sample_acc_series": getattr(build_sample_acc, "__name__", str(build_sample_acc)),
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
        "global_control_names": global_control_names,
        "global_control_name_to_index": {
            name: idx for idx, name in enumerate(global_control_names)
        },
        "shape_metadata": {
            "n_processes": len(collection.processes),
            "max_grid_length": max_grid_length,
            "max_controls": len(global_control_names),
            "max_step_ts_length": max_step_ts_length,
        },
        "processes": {},
    }

    for process_name, payload in process_payloads.items():
        padded = _pad_process_payload(
            payload=payload,
            max_grid_length=max_grid_length,
            global_control_names=global_control_names,
            max_step_ts_length=max_step_ts_length,
        )
        bp_train_metadata["processes"][process_name] = {
            "local_control_names": payload["control_names"],
            "control_name_to_index": {
                name: idx for idx, name in enumerate(global_control_names)
            },
            "local_to_global_index": {
                name: global_control_names.index(name) for name in payload["control_names"]
            },
            "control_metadata": payload["control_metadata"],
            "sample_acc_index": payload["sample_acc_index"],
            "sample_acc_name": BP_TRAIN_SAMPLE_ACC_NAME,
            "step_ts": padded["step_ts"],
            "step_ts_mask": padded["step_ts_mask"],
            "grid_length": padded["grid_length"],
            "control_count": padded["control_count"],
            "dense_grid": padded["dense_grid"],
            "dense_grid_mask": padded["dense_grid_mask"],
            "control_values": padded["control_values"],
            "control_derivatives": padded["control_derivatives"],
            "control_mask": padded["control_mask"],
        }

    existing_metadata[resolved_config["metadata_namespace"]] = bp_train_metadata
    collection.metadata = existing_metadata

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_process_collection_json(collection, output_path)
    return collection
