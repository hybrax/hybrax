from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from bpbench.dataclasses import BioProcessCollection
from bpbench.serialization import load_process_collection_json, save_process_collection_json

from .controls import (
    BP_TRAIN_SAMPLE_ACC_NAME,
    build_dense_payload,
    build_sample_acc_source_default,
    compute_signal_spreads,
    select_control_sources,
)
from .custom import get_hook, load_custom_module, resolve_config
from .validation import ensure_required_controls, validate_raw_collection


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


def load_raw_collection(input_json: str | Path) -> BioProcessCollection:
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
    max_controls: int,
) -> dict[str, Any]:
    grid = list(payload["grid"])
    values = [list(row) for row in payload["values"]]
    derivatives = [list(row) for row in payload["derivatives"]]

    grid_length = len(grid)
    control_count = len(values[0]) if values else 0

    padded_grid = grid + [0.0] * (max_grid_length - grid_length)
    grid_mask = [True] * grid_length + [False] * (max_grid_length - grid_length)

    padded_values = []
    padded_derivatives = []
    for row, deriv_row in zip(values, derivatives, strict=False):
        padded_values.append(row + [0.0] * (max_controls - control_count))
        padded_derivatives.append(deriv_row + [0.0] * (max_controls - control_count))

    zero_row = [0.0] * max_controls
    for _ in range(max_grid_length - grid_length):
        padded_values.append(list(zero_row))
        padded_derivatives.append(list(zero_row))

    control_mask = [True] * control_count + [False] * (max_controls - control_count)

    return {
        "dense_grid": padded_grid,
        "dense_grid_mask": grid_mask,
        "control_values": padded_values,
        "control_derivatives": padded_derivatives,
        "control_mask": control_mask,
        "grid_length": grid_length,
        "control_count": control_count,
    }


def prepare_artifact(
    input_json: str | Path,
    output_json: str | Path,
    *,
    custom_py: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> BioProcessCollection:
    input_path = Path(input_json)
    output_path = Path(output_json)

    custom_module = load_custom_module(custom_py)
    user_config = resolve_config(custom_module, config)

    defaults = asdict(PrepareConfig())
    defaults.update(user_config)
    resolved_config = defaults

    raw_collection = load_raw_collection(input_path)
    validation_report = validate_raw_collection(
        raw_collection,
        strict=bool(resolved_config.get("strict_bpbench_validation", False)),
    )

    collection = deepcopy(raw_collection)

    transform_controls = get_hook(custom_module, "transform_controls", _default_transform_controls)
    transform_states = get_hook(custom_module, "transform_states", _default_transform_states)
    build_sample_acc = get_hook(
        custom_module,
        "build_sample_acc_series",
        _default_build_sample_acc,
    )

    for process_name, process in list(collection.processes.items()):
        process = transform_controls(process, resolved_config)
        process = transform_states(process, resolved_config)
        collection.processes[process_name] = process

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

    spread_inputs = {
        process_name: sources + [sample_sources[process_name]]
        for process_name, sources in process_sources.items()
    }
    signal_spreads = compute_signal_spreads(spread_inputs)

    process_payloads: dict[str, dict[str, Any]] = {}
    max_grid_length = 0
    max_controls = 0

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
            "sample_acc_index": len(sources) - 1,
            "step_ts": payload["step_ts"],
            "grid": payload["grid"],
            "values": payload["values"],
            "derivatives": payload["derivatives"],
        }
        max_grid_length = max(max_grid_length, len(payload["grid"]))
        max_controls = max(max_controls, len(sources))

    source_hash = _sha256_hex(_read_bytes(input_path))
    custom_hash = _sha256_hex(_read_bytes(custom_py))

    existing_metadata = dict(collection.metadata or {})
    bp_train_metadata: dict[str, Any] = {
        "prepared_at": _utc_now_iso(),
        "source_input_path": str(input_path),
        "source_input_sha256": source_hash,
        "custom_py_sha256": custom_hash,
        "transform_hooks": {
            "transform_controls": getattr(transform_controls, "__name__", str(transform_controls)),
            "transform_states": getattr(transform_states, "__name__", str(transform_states)),
            "build_sample_acc_series": getattr(build_sample_acc, "__name__", str(build_sample_acc)),
        },
        "dynamic_volume": True,
        "bpbench_validation": validation_report,
        "process_order": list(collection.processes.keys()),
        "shape_metadata": {
            "n_processes": len(collection.processes),
            "max_grid_length": max_grid_length,
            "max_controls": max_controls,
        },
        "processes": {},
    }

    for process_name, payload in process_payloads.items():
        padded = _pad_process_payload(
            payload=payload,
            max_grid_length=max_grid_length,
            max_controls=max_controls,
        )
        bp_train_metadata["processes"][process_name] = {
            "control_names": payload["control_names"],
            "control_name_to_index": {
                name: idx for idx, name in enumerate(payload["control_names"])
            },
            "control_metadata": payload["control_metadata"],
            "sample_acc_index": payload["sample_acc_index"],
            "sample_acc_name": BP_TRAIN_SAMPLE_ACC_NAME,
            "step_ts": payload["step_ts"],
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
