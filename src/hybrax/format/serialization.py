"""Serialization utilities for bioprocess benchmarking dataset."""

import gzip
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Union

import jax.numpy as jnp
import numpy as np
from ijson.common import ObjectBuilder

from .dataclasses import (
    AugmentedBioProcess,
    BiologicalOde,
    BioProcessCollection,
    BioProcess,
    Bounds,
    _DEFAULT_RMC_BOUNDS,
    TimeSeries,
    TimeAxis,
    DiscreteEvents,
    FeedMedium,
    FeedMediumComponent,
    StaticVariable,
    BioProcessMetadata,
    Volume,
    FeedVolumeChange,
    SampleVolumeChange,
    PseudobatchTransform,
    ReactorMedium,
    ReactorMediumComponent,
    ProcessVariable,
)
from .json_io import _kvitems, _parse, load_json


def _bounds_to_dict(bounds: Bounds) -> Optional[Dict]:
    """Serialize bounds; return ``None`` when both sides are unbounded so the
    JSON stays clean for the common default."""
    if bounds is None or (bounds[0] is None and bounds[1] is None):
        return None
    return {"lower": bounds[0], "upper": bounds[1]}


def _dict_to_bounds(data: Optional[Dict]) -> Bounds:
    """Deserialize bounds; missing or null → ``(None, None)``."""
    if not data:
        return (None, None)
    return (data.get("lower"), data.get("upper"))


def _biological_ode_to_dict(ode: BiologicalOde) -> Dict:
    return {
        "algebraic": dict(ode.algebraic),
        "rates": {name: _bounds_to_dict(bounds) for name, bounds in ode.rates.items()},
        "derivatives": dict(ode.derivatives),
    }


def _dict_to_biological_ode(data: Dict) -> BiologicalOde:
    return BiologicalOde(
        algebraic=dict(data.get("algebraic", {})),
        rates={name: _dict_to_bounds(rd) for name, rd in data.get("rates", {}).items()},
        derivatives=dict(data.get("derivatives", {})),
    )


DEFAULT_JSON_FILENAME = "data.json"
DEFAULT_JSON_GZ_FILENAME = "data.json.gz"


def _is_json_gz_path(path: Path) -> bool:
    """Return whether the path ends in `.json.gz`."""
    return Path(path).suffixes[-2:] == [".json", ".gz"]


def _is_supported_json_file_path(path: Path) -> bool:
    """Return whether the path is an explicit supported JSON file path."""
    path = Path(path)
    return path.suffix == ".json" or _is_json_gz_path(path)


def _resolve_json_path(path: Path) -> Path:
    """Return the JSON path used by the serializer."""
    path = Path(path)
    if _is_supported_json_file_path(path):
        return path
    return path / DEFAULT_JSON_FILENAME


def _resolve_existing_json_path(path: Path) -> Path:
    """Resolve a load path and enforce JSON-only inputs."""
    path = Path(path)
    if _is_supported_json_file_path(path):
        if path.exists():
            return path
    else:
        json_path = path / DEFAULT_JSON_FILENAME
        if json_path.exists():
            return json_path

        json_gz_path = path / DEFAULT_JSON_GZ_FILENAME
        if json_gz_path.exists():
            return json_gz_path

    if path.suffix:
        raise FileNotFoundError(
            "Only JSON serialization is supported. "
            f"Expected a '.json' or '.json.gz' file, got '{path}'."
        )

    raise FileNotFoundError(
        f"Expected JSON dataset at '{path / DEFAULT_JSON_FILENAME}' "
        f"or '{path / DEFAULT_JSON_GZ_FILENAME}'."
    )


def _open_json_file(path: Path, mode: str):
    """Open a JSON or JSON.GZ file."""
    path = Path(path)
    kwargs = {} if "b" in mode else {"encoding": "utf-8"}
    if _is_json_gz_path(path):
        return gzip.open(path, mode, **kwargs)
    return open(path, mode, **kwargs)


def _save_json(data_dict: Dict, json_path: Path) -> None:
    """Write a serialized data dict to a `.json` / `.json.gz` file."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _normalize_nonfinite(data_dict)
    with _open_json_file(json_path, "wt") as f:
        json.dump(data_dict, f, indent=2, cls=NumpyEncoder, allow_nan=False)
    print(f"✓ Saved to {json_path}")


def _restore_arrays(obj):
    """Recursively rebuild JAX arrays from the `__ndarray__` JSON encoding."""
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            # Floating data is float64 (x64 pipeline). Legacy payloads may store
            # float32; load them straight as float64 (int/bool dtypes preserved).
            stored = np.dtype(obj["dtype"])
            payload = obj["__ndarray__"]
            if np.issubdtype(stored, np.floating):
                return jnp.array(np.asarray(payload, dtype=np.float64))
            if _contains_none(payload):
                raise ValueError(
                    f"null is invalid in a typed {stored} __ndarray__ payload"
                )
            return jnp.array(payload, dtype=stored)
        return {k: _restore_arrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_arrays(item) for item in obj]
    return obj


def _load_json(json_path: Path) -> Dict:
    """Read a `.json` / `.json.gz` file and restore JAX arrays."""
    json_path = Path(json_path)
    return _restore_arrays(load_json(json_path))


def save_process_collection(collection: BioProcessCollection, path: Path) -> None:
    """Save a BioProcessCollection as JSON.

    `path` may be a JSON file path or a directory, in which case `data.json`
    is written inside it.
    """
    _save_json(_process_collection_to_dict(collection), _resolve_json_path(path))


def _read_collection_header(json_path: Path) -> Dict:
    """Stream-read the top-level scalar/metadata fields of a collection JSON
    (case_id, organism, citation, metadata) without materializing `processes`."""
    header: Dict = {"case_id": None, "organism": None, "citation": None, "metadata": None}
    seen = {"case_id": False, "organism": False, "citation": False, "metadata": False}
    metadata_builder = None
    metadata_depth = 0
    pending_key = None
    root_seen = False
    processes_seen = False

    def _all_seen() -> bool:
        return processes_seen and all(seen.values())

    with _open_json_file(json_path, "rb") as f:
        for prefix, event, value in _parse(f, source=json_path):
            if not root_seen:
                if prefix != "" or event != "start_map":
                    raise ValueError(f"{json_path}: collection root must be an object")
                root_seen = True
                continue

            if prefix == "" and event == "map_key":
                pending_key = value
                continue

            if pending_key == "processes" and prefix == "processes":
                if event != "start_map":
                    raise ValueError(
                        f"{json_path}: collection processes must be an object"
                    )
                processes_seen = True
                pending_key = None
                if _all_seen():
                    return header
                continue

            if pending_key in ("case_id", "organism", "citation") and prefix == pending_key:
                if event == "string":
                    header[pending_key] = value
                elif event == "null":
                    header[pending_key] = None
                else:
                    raise ValueError(
                        f"{json_path}: collection {pending_key} must be a string or null"
                    )
                seen[pending_key] = True
                pending_key = None
                if _all_seen():
                    return header
                continue

            if pending_key == "metadata" and prefix == "metadata":
                if event == "null":
                    header["metadata"] = None
                    seen["metadata"] = True
                    if _all_seen():
                        return header
                elif event == "start_map":
                    metadata_builder = ObjectBuilder()
                    metadata_builder.event(event, value)
                    metadata_depth = 1
                else:
                    raise ValueError(
                        f"{json_path}: collection metadata must be an object or null"
                    )
                pending_key = None
                continue

            if metadata_builder is not None and (
                prefix == "metadata" or prefix.startswith("metadata.")
            ):
                metadata_builder.event(event, value)
                if event in ("start_map", "start_array"):
                    metadata_depth += 1
                elif event in ("end_map", "end_array"):
                    metadata_depth -= 1
                    if metadata_depth == 0:
                        header["metadata"] = _restore_arrays(metadata_builder.value)
                        metadata_builder = None
                        seen["metadata"] = True
                        if _all_seen():
                            return header

    if not root_seen:
        raise ValueError(f"{json_path}: collection root must be an object")
    if not processes_seen:
        raise ValueError(f"{json_path}: collection must contain a processes object")
    return header


def _stream_process_collection(json_path: Path) -> BioProcessCollection:
    header = _read_collection_header(json_path)
    processes = {}
    with _open_json_file(json_path, "rb") as f:
        for process_id, process_data in _kvitems(f, "processes", source=json_path):
            processes[process_id] = _dict_to_process(_restore_arrays(process_data))
    return BioProcessCollection(
        case_id=header["case_id"],
        organism=header["organism"],
        citation=header["citation"],
        metadata=header["metadata"],
        processes=processes,
    )


def load_process_collection(path: Path) -> BioProcessCollection:
    """Load a BioProcessCollection incrementally from JSON.

    `path` may be a JSON file path or a directory containing `data.json`.
    The YAJL-backed parser also accepts ``//`` and ``/* ... */`` comments.
    """
    return _stream_process_collection(_resolve_existing_json_path(path))


# ============================================================
# Helper Functions
# ============================================================


def _process_collection_to_dict(collection: BioProcessCollection) -> Dict:
    """Convert BioProcessCollection to nested dictionary."""
    result: Dict = {}
    if collection.case_id is not None:
        result["case_id"] = collection.case_id
    if collection.organism is not None:
        result["organism"] = collection.organism
    if collection.citation is not None:
        result["citation"] = collection.citation
    result["metadata"] = deepcopy(collection.metadata)
    result["processes"] = {
        p_id: _process_to_dict(process) for p_id, process in collection.processes.items()
    }
    return result


def _process_to_dict(process: BioProcess) -> Dict:
    """Convert BioProcess to dictionary"""
    result = {
        "metadata": _process_metadata_to_dict(process.metadata),
        "time_axis": {
            "unit": process.time_axis.unit,
            "start": process.time_axis.start,
            "end": process.time_axis.end,
            "time_reference": process.time_axis.time_reference,
        }
        if process.time_axis
        else None,
        "reactor_medium": _reactor_medium_to_dict(process.reactor_medium)
        if process.reactor_medium
        else None,
        "process_variables": {
            name: _process_variable_to_dict(pv)
            for name, pv in process.process_variables.items()
        },
    }

    # Add volume if present
    if process.volume is not None:
        result["volume"] = _volume_to_dict(process.volume)

    # Add discrete events if present
    if process.discrete_events is not None:
        result["discrete_events"] = _discrete_events_to_dict(process.discrete_events)

    if process.biological_ode is not None:
        result["biological_ode"] = _biological_ode_to_dict(process.biological_ode)

    if process.pseudobatch_transform is not None:
        result["pseudobatch_transform"] = _pseudobatch_transform_to_dict(
            process.pseudobatch_transform
        )

    if isinstance(process, AugmentedBioProcess):
        result["__type__"] = "AugmentedBioProcess"
        result["parent_process"] = process.parent_process

    return result


def _process_metadata_to_dict(metadata: Optional[BioProcessMetadata]) -> Optional[Dict]:
    """Convert BioProcessMetadata to dictionary."""
    if metadata is None:
        return None
    return {
        "name": metadata.name,
        "process_type": metadata.process_type,
        "notes": metadata.notes,
    }


def _reactor_medium_to_dict(reactor_medium: ReactorMedium) -> Dict:
    """Convert ReactorMedium to dictionary"""
    return {
        "name": reactor_medium.name,
        "density": reactor_medium.density,
        "density_unit": reactor_medium.density_unit,
        "components": {
            name: _reactor_component_to_dict(comp)
            for name, comp in reactor_medium.components.items()
        },
    }


def _reactor_component_to_dict(comp: ReactorMediumComponent) -> Dict:
    """Convert ReactorMediumComponent to dictionary"""
    result = {
        "name": comp.name,
        "unit": comp.unit,
        "concentration": _timeseries_or_static_to_dict(comp.concentration),
    }
    if comp.c_star_concentration is not None:
        result["c_star_concentration"] = _timeseries_or_static_to_dict(
            comp.c_star_concentration
        )
    if comp.bounds != _DEFAULT_RMC_BOUNDS:
        # Preserve explicit unbounded bounds instead of reloading the RMC default.
        result["bounds"] = _bounds_to_dict(comp.bounds)
    return result


def _process_variable_to_dict(pv: ProcessVariable) -> Dict:
    """Convert ProcessVariable to dictionary"""
    result = {
        "name": pv.name,
        "unit": pv.unit,
        "is_controlled": pv.is_controlled,
        "values": _timeseries_or_static_to_dict(pv.values),
    }
    bounds_dict = _bounds_to_dict(pv.bounds)
    if bounds_dict is not None:
        result["bounds"] = bounds_dict
    return result


def _pseudobatch_transform_to_dict(transform: PseudobatchTransform) -> Dict:
    """Convert PseudobatchTransform to dictionary."""
    result = {
        "adf": _timeseries_to_dict_payload(transform.adf, include_type=False),
        "feed_corrections": {
            name: _timeseries_to_dict_payload(ts, include_type=False)
            for name, ts in transform.feed_corrections.items()
        },
        "accumulated_feeds": {
            name: _timeseries_to_dict_payload(ts, include_type=False)
            for name, ts in transform.accumulated_feeds.items()
        },
    }
    if transform.sample_compensation is not None:
        result["sample_compensation"] = _timeseries_to_dict_payload(
            transform.sample_compensation,
            include_type=False,
        )
    return result


def _timeseries_to_dict_payload(
    value: TimeSeries, *, include_type: bool = True
) -> Dict:
    """Serialize TimeSeries using canonical keys."""
    times = getattr(value, "times", None)

    payload = {
        "times": times,
        "values": value.values,
    }
    if include_type:
        payload["type"] = "TimeSeries"

    if hasattr(value, "derived"):
        payload["derived"] = bool(value.derived)
    if getattr(value, "jump_times", None) is not None:
        payload["jump_times"] = value.jump_times
    if getattr(value, "breaks", None) is not None:
        payload["breaks"] = value.breaks
    if getattr(value, "coeffs", None) is not None:
        payload["coeffs"] = value.coeffs
    if getattr(value, "segment_start_piece_idx", None) is not None:
        payload["segment_start_piece_idx"] = value.segment_start_piece_idx
    if hasattr(value, "continuity_side"):
        payload["continuity_side"] = value.continuity_side
    if getattr(value, "metadata", None) is not None:
        payload["metadata"] = deepcopy(value.metadata)
    return payload


def _timeseries_or_static_to_dict(value: Union[TimeSeries, StaticVariable]) -> Dict:
    """Convert TimeSeries or StaticVariable to dictionary"""
    if isinstance(value, TimeSeries):
        return _timeseries_to_dict_payload(value, include_type=True)
    elif isinstance(value, StaticVariable):
        return {"type": "StaticVariable", "value": value.value}
    else:
        raise ValueError(f"Unknown value type: {type(value)}")


def _volume_to_dict(volume: Volume) -> Dict:
    """Convert Volume to dictionary"""
    result = {
        "initial_volume": volume.initial_volume,
        "unit": volume.unit,
        "volume_changes": {
            name: _volume_change_to_dict(vc)
            for name, vc in volume.volume_changes.items()
        },
    }
    if volume.total_volume is not None:
        result["total_volume"] = _timeseries_to_dict_payload(
            volume.total_volume,
            include_type=False,
        )
    bounds_dict = _bounds_to_dict(volume.bounds)
    if bounds_dict is not None:
        result["bounds"] = bounds_dict
    return result


def _volume_change_to_dict(vc) -> Dict:
    """Convert FeedVolumeChange or SampleVolumeChange to dictionary"""
    result = {
        "name": vc.name,
        "unit": vc.unit,
        "is_controlled": vc.is_controlled,
        "is_continuous": vc.is_continuous,
        "values": (
            _timeseries_to_dict_payload(vc.values, include_type=False)
            if vc.values
            else None
        ),
    }
    if isinstance(vc, FeedVolumeChange):
        result["type"] = "FeedVolumeChange"
        result["feed_medium"] = (
            _feed_medium_to_dict(vc.feed_medium) if vc.feed_medium else None
        )
    elif isinstance(vc, SampleVolumeChange):
        result["type"] = "SampleVolumeChange"
    else:
        raise ValueError(f"Unknown volume change type: {type(vc)}")

    return result


def _feed_medium_to_dict(feed: FeedMedium) -> Dict:
    """Convert FeedMedium to dictionary"""
    return {
        "name": feed.name,
        "density": feed.density,
        "density_unit": feed.density_unit,
        "components": {
            name: _feed_component_to_dict(comp)
            for name, comp in feed.components.items()
        },
    }


def _feed_component_to_dict(comp: FeedMediumComponent) -> Dict:
    """Convert FeedMediumComponent to dictionary"""
    return {
        "name": comp.name,
        "unit": comp.unit,
        "is_controlled": comp.is_controlled,
        "concentration": _timeseries_or_static_to_dict(comp.concentration),
    }


def _dict_to_process_collection(data: Dict) -> BioProcessCollection:
    """Reconstruct BioProcessCollection from dictionary."""
    return BioProcessCollection(
        case_id=data.get("case_id"),
        organism=data.get("organism"),
        citation=data.get("citation"),
        metadata=data.get("metadata"),
        processes={
            p_id: _dict_to_process(p_data)
            for p_id, p_data in data.get("processes", {}).items()
        },
    )


def _dict_to_process(p_data: Dict) -> BioProcess:
    """Reconstruct BioProcess from dictionary"""
    # Reconstruct metadata
    metadata = None
    if p_data.get("metadata") is not None:
        metadata = BioProcessMetadata(
            name=p_data["metadata"]["name"],
            process_type=p_data["metadata"]["process_type"],
            notes=p_data["metadata"].get("notes"),
        )

    # Reconstruct time axis
    time_axis = None
    if p_data.get("time_axis"):
        time_axis = TimeAxis(
            unit=p_data["time_axis"]["unit"],
            start=p_data["time_axis"]["start"],
            end=p_data["time_axis"]["end"],
            time_reference=p_data["time_axis"]["time_reference"],
        )

    # Reconstruct reactor medium
    reactor_medium = None
    if p_data.get("reactor_medium"):
        reactor_medium = _dict_to_reactor_medium(p_data["reactor_medium"])

    # Reconstruct process variables
    process_variables = {
        name: _dict_to_process_variable(pv_data)
        for name, pv_data in p_data.get("process_variables", {}).items()
    }

    # Reconstruct volume
    volume = None
    if p_data.get("volume"):
        volume = _dict_to_volume(p_data["volume"])

    # Reconstruct discrete events
    discrete_events = None
    if p_data.get("discrete_events"):
        discrete_events = _dict_to_discrete_events(p_data["discrete_events"])

    biological_ode = None
    if p_data.get("biological_ode") is not None:
        biological_ode = _dict_to_biological_ode(p_data["biological_ode"])

    # Absent key means no pseudobatch transform. Present malformed entries fail
    # fast so old or partial bundle payloads do not silently load.
    pseudobatch_transform = None
    if "pseudobatch_transform" in p_data:
        transform_data = p_data["pseudobatch_transform"]
        if transform_data is not None:
            pseudobatch_transform = _dict_to_pseudobatch_transform(transform_data)

    if p_data.get("__type__") == "AugmentedBioProcess":
        parent = p_data.get("parent_process")
        if not isinstance(parent, str) or not parent:
            raise ValueError(
                "AugmentedBioProcess payload missing required 'parent_process' string"
            )
        return AugmentedBioProcess(
            metadata=metadata,
            time_axis=time_axis,
            volume=volume,
            reactor_medium=reactor_medium,
            process_variables=process_variables,
            discrete_events=discrete_events,
            biological_ode=biological_ode,
            pseudobatch_transform=pseudobatch_transform,
            parent_process=parent,
        )

    return BioProcess(
        metadata=metadata,
        time_axis=time_axis,
        volume=volume,
        reactor_medium=reactor_medium,
        process_variables=process_variables,
        discrete_events=discrete_events,
        biological_ode=biological_ode,
        pseudobatch_transform=pseudobatch_transform,
    )


def _require_mapping(value, context: str) -> Dict:
    """Return a dict-like payload or raise a clear loader error."""
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a dictionary.")
    return value


def _require_keys(data: Dict, required: tuple[str, ...], context: str) -> None:
    """Validate required keys for a strict serialized payload."""
    missing = [key for key in required if key not in data]
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(f"{context} missing required key(s): {missing_keys}.")


def _dict_to_pseudobatch_transform(data: Dict) -> PseudobatchTransform:
    """Reconstruct PseudobatchTransform from dictionary."""
    data = _require_mapping(data, "pseudobatch_transform")
    _require_keys(
        data,
        (
            "adf",
            "feed_corrections",
        ),
        "pseudobatch_transform",
    )

    feed_corrections_data = _require_mapping(
        data["feed_corrections"],
        "pseudobatch_transform.feed_corrections",
    )
    accumulated_feeds_data = _require_mapping(
        data.get("accumulated_feeds", {}),
        "pseudobatch_transform.accumulated_feeds",
    )

    sample_compensation = None
    if data.get("sample_compensation") is not None:
        sample_compensation = _dict_to_pseudobatch_timeseries(
            data["sample_compensation"],
            "pseudobatch_transform.sample_compensation",
        )

    return PseudobatchTransform(
        adf=_dict_to_pseudobatch_timeseries(data["adf"], "pseudobatch_transform.adf"),
        feed_corrections={
            name: _dict_to_pseudobatch_timeseries(
                ts_data,
                f"pseudobatch_transform.feed_corrections.{name}",
            )
            for name, ts_data in feed_corrections_data.items()
        },
        sample_compensation=sample_compensation,
        accumulated_feeds={
            name: _dict_to_pseudobatch_timeseries(
                ts_data,
                f"pseudobatch_transform.accumulated_feeds.{name}",
            )
            for name, ts_data in accumulated_feeds_data.items()
        },
    )


def _dict_to_pseudobatch_timeseries(data: Dict, context: str) -> TimeSeries:
    """Reconstruct one strict TimeSeries payload in a pseudobatch bundle."""
    data = _require_mapping(data, context)
    return _timeseries_from_dict_payload(data)


def _dict_to_reactor_medium(rm_data: Dict) -> ReactorMedium:
    """Reconstruct ReactorMedium from dictionary"""
    components = {
        name: _dict_to_reactor_component(comp_data)
        for name, comp_data in rm_data.get("components", {}).items()
    }

    return ReactorMedium(
        name=rm_data["name"],
        density=rm_data["density"],
        density_unit=rm_data["density_unit"],
        components=components,
    )


def _dict_to_reactor_component(comp_data: Dict) -> ReactorMediumComponent:
    """Reconstruct ReactorMediumComponent from dictionary"""
    _reject_legacy_interpolator_payload(
        comp_data.get("interpolator"), "ReactorMediumComponent"
    )
    c_star_concentration = None
    if comp_data.get("c_star_concentration") is not None:
        c_star_concentration = _dict_to_timeseries_or_static(
            comp_data["c_star_concentration"]
        )
    kwargs = {}
    if "bounds" in comp_data:
        kwargs["bounds"] = _dict_to_bounds(comp_data["bounds"])
    return ReactorMediumComponent(
        name=comp_data["name"],
        unit=comp_data["unit"],
        concentration=_dict_to_timeseries_or_static(comp_data["concentration"]),
        c_star_concentration=c_star_concentration,
        **kwargs,
    )


def _dict_to_process_variable(pv_data: Dict) -> ProcessVariable:
    """Reconstruct ProcessVariable from dictionary"""
    _reject_legacy_interpolator_payload(pv_data.get("interpolator"), "ProcessVariable")
    return ProcessVariable(
        name=pv_data["name"],
        unit=pv_data["unit"],
        is_controlled=pv_data["is_controlled"],
        values=_dict_to_timeseries_or_static(pv_data["values"]),
        bounds=_dict_to_bounds(pv_data.get("bounds")),
    )


def _timeseries_from_dict_payload(value_data: Dict) -> TimeSeries:
    """Reconstruct TimeSeries from a typed or untyped serialized payload."""
    times = value_data.get("times")
    values = value_data.get("values")
    metadata = value_data.get("metadata")
    _reject_nested_pseudobatch_metadata(metadata)

    kwargs: Dict = {"values": values}
    if values is not None:
        if times is None:
            raise ValueError(
                "TimeSeries payload with discrete values must include 'times'."
            )
        kwargs["times"] = times

    if "derived" in value_data:
        kwargs["derived"] = bool(value_data["derived"])
    if "jump_times" in value_data:
        kwargs["jump_times"] = value_data["jump_times"]
    if "breaks" in value_data:
        kwargs["breaks"] = value_data["breaks"]
    if "coeffs" in value_data:
        kwargs["coeffs"] = value_data["coeffs"]
    if "segment_start_piece_idx" in value_data:
        kwargs["segment_start_piece_idx"] = value_data["segment_start_piece_idx"]
    if "continuity_side" in value_data:
        kwargs["continuity_side"] = value_data["continuity_side"]
    if "metadata" in value_data:
        kwargs["metadata"] = metadata

    return TimeSeries(**kwargs)


def _dict_to_timeseries_or_static(
    value_data: Dict,
) -> Union[TimeSeries, StaticVariable]:
    """Reconstruct TimeSeries or StaticVariable from dictionary"""
    if value_data["type"] == "TimeSeries":
        return _timeseries_from_dict_payload(value_data)
    elif value_data["type"] == "StaticVariable":
        return StaticVariable(value=value_data["value"])
    else:
        raise ValueError(f"Unknown value type: {value_data['type']}")


def _dict_to_volume(vol_data: Dict) -> Volume:
    """Reconstruct Volume from dictionary"""
    volume_changes = {
        name: _dict_to_volume_change(vc_data)
        for name, vc_data in vol_data.get("volume_changes", {}).items()
    }

    total_volume = None
    if vol_data.get("total_volume") is not None:
        total_volume = _timeseries_from_dict_payload(vol_data["total_volume"])

    return Volume(
        initial_volume=vol_data["initial_volume"],
        unit=vol_data["unit"],
        volume_changes=volume_changes,
        total_volume=total_volume,
        bounds=_dict_to_bounds(vol_data.get("bounds")),
    )


def _dict_to_volume_change(vc_data: Dict):
    """Reconstruct FeedVolumeChange or SampleVolumeChange from dictionary"""
    vc_type = vc_data.get("type")
    if vc_type is None:
        raise ValueError(
            "Old VolumeChange schema detected (missing 'type'). "
            "Please regenerate datasets by running "
            "examples/*/01_load_single_process.ipynb and "
            "examples/*/02_load_all_processes.ipynb."
        )

    values = None
    if vc_data.get("values"):
        values = _timeseries_from_dict_payload(vc_data["values"])

    common = dict(
        name=vc_data["name"],
        unit=vc_data["unit"],
        is_controlled=vc_data["is_controlled"],
        is_continuous=vc_data["is_continuous"],
        values=values,
    )

    _reject_legacy_interpolator_payload(vc_data.get("interpolator"), "VolumeChange")

    if vc_type == "FeedVolumeChange":
        feed_medium = None
        if vc_data.get("feed_medium"):
            feed_medium = _dict_to_feed_medium(vc_data["feed_medium"])
        return FeedVolumeChange(**common, feed_medium=feed_medium)
    elif vc_type == "SampleVolumeChange":
        return SampleVolumeChange(**common)
    else:
        raise ValueError(f"Unknown volume change type: {vc_type}")


def _reject_legacy_interpolator_payload(interpolator_data: Dict | None, owner: str):
    """Reject legacy sibling ``interpolator`` payloads loudly."""
    if interpolator_data is not None:
        raise ValueError(
            "Legacy sibling 'interpolator' payloads are no longer supported for "
            f"{owner}. Regenerate datasets with TimeSeries-only spline storage."
        )


def _reject_nested_pseudobatch_metadata(metadata) -> None:
    """Reject executable pseudobatch transform payloads embedded in metadata."""
    if not isinstance(metadata, dict):
        return
    transform = metadata.get("transform")
    if not isinstance(transform, dict):
        return
    if "series" in transform:
        raise ValueError(
            "TimeSeries metadata contains nested executable pseudobatch transform "
            "series. Store pseudobatch state in process.pseudobatch_transform."
        )


def _dict_to_feed_medium(feed_data: Dict) -> FeedMedium:
    """Reconstruct FeedMedium from dictionary"""
    components = {
        name: _dict_to_feed_component(comp_data)
        for name, comp_data in feed_data.get("components", {}).items()
    }

    return FeedMedium(
        name=feed_data["name"],
        density=feed_data["density"],
        density_unit=feed_data["density_unit"],
        components=components,
    )


def _dict_to_feed_component(comp_data: Dict) -> FeedMediumComponent:
    """Reconstruct FeedMediumComponent from dictionary"""
    return FeedMediumComponent(
        name=comp_data["name"],
        unit=comp_data["unit"],
        is_controlled=comp_data["is_controlled"],
        concentration=_dict_to_timeseries_or_static(comp_data["concentration"]),
    )


# ============================================================
# DiscreteEvents serialization helpers
# ============================================================


def _discrete_events_to_dict(de: DiscreteEvents) -> Dict:
    """Convert DiscreteEvents to dictionary"""
    return {
        "times": de.times,
        "labels": deepcopy(de.labels),
        "metadata": deepcopy(de.metadata),
    }


def _dict_to_discrete_events(data: Dict) -> DiscreteEvents:
    """Reconstruct DiscreteEvents from dictionary"""
    times = data["times"]
    if not isinstance(times, jnp.ndarray):
        times = jnp.array(times)
    return DiscreteEvents(
        times=times,
        labels=data.get("labels"),
        metadata=data.get("metadata"),
    )


# ============================================================
# JSON Encoder for numpy/JAX arrays
# ============================================================


def _contains_none(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_none(item) for item in value)
    return False


def _normalize_nonfinite(value):
    """Replace non-finite floats in temporary serialization data with null."""
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = _normalize_nonfinite(item)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _normalize_nonfinite(item)
    elif isinstance(value, tuple):
        return tuple(_normalize_nonfinite(item) for item in value)
    return value


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/JAX arrays."""

    def default(self, obj):
        if isinstance(obj, (jnp.ndarray, np.ndarray)):
            return {
                "__ndarray__": _normalize_nonfinite(obj.tolist()),
                "dtype": str(obj.dtype),
            }
        if isinstance(obj, np.floating):
            return _normalize_nonfinite(obj)
        return super().default(obj)
