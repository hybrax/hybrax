"""Serialization utilities for bioprocess benchmarking dataset."""

import gzip
import json
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
from .dataclasses import (
    AugmentedBioProcess,
    BenchmarkDataset,
    BioProcessCollection,
    CaseStudy,
    BioProcess,
    TimeSeries,
    TimeAxis,
    Interpolator,
    DiscreteEvents,
    FeedMedium,
    FeedMediumComponent,
    StaticVariable,
    BioProcessMetadata,
    Volume,
    VolumeChange,
    FeedVolumeChange,
    SampleVolumeChange,
    ReactorMedium,
    ReactorMediumComponent,
    ProcessVariable,
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
    """Open a JSON or JSON.GZ file in text mode."""
    path = Path(path)
    if _is_json_gz_path(path):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def save_dataset(dataset: BenchmarkDataset, path: Path) -> None:
    """Save a dataset as JSON.

    `path` may be a JSON file path or a directory, in which case `data.json`
    is written inside it.
    """
    save_dataset_json(dataset, _resolve_json_path(path))


def save_process_collection(collection: BioProcessCollection, path: Path) -> None:
    """Save a process collection as JSON.

    `path` may be a JSON file path or a directory, in which case `data.json`
    is written inside it.
    """
    save_process_collection_json(collection, _resolve_json_path(path))


def load_dataset(path: Path) -> BenchmarkDataset:
    """Load a dataset from JSON.

    `path` may be a JSON file path or a directory containing `data.json`.
    """
    return load_dataset_json(_resolve_existing_json_path(path))


def load_process_collection(path: Path) -> BioProcessCollection:
    """Load a process collection from JSON.

    `path` may be a JSON file path or a directory containing `data.json`.
    """
    return load_process_collection_json(_resolve_existing_json_path(path))


def save_dataset_json(dataset: BenchmarkDataset, json_path: Path) -> None:
    """
    Save dataset as single JSON file (human-readable but larger)

    Args:
        dataset: BenchmarkDataset to save
        json_path: File path where JSON will be saved
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    data_dict = _dataset_to_dict(dataset)

    with _open_json_file(json_path, "wt") as f:
        json.dump(data_dict, f, indent=2, cls=NumpyEncoder)

    print(f"✓ Dataset saved to {json_path}")


def save_process_collection_json(
    collection: BioProcessCollection, json_path: Path
) -> None:
    """
    Save a BioProcessCollection as a single JSON file.

    Args:
        collection: BioProcessCollection to save
        json_path: File path where JSON will be saved
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    data_dict = _process_collection_to_dict(collection)

    with _open_json_file(json_path, "wt") as f:
        json.dump(data_dict, f, indent=2, cls=NumpyEncoder)

    print(f"✓ Process collection saved to {json_path}")


def load_dataset_json(json_path: Path) -> BenchmarkDataset:
    """
    Load dataset from JSON

    Args:
        json_path: Path to JSON file

    Returns:
        Reconstructed BenchmarkDataset
    """
    json_path = Path(json_path)

    with _open_json_file(json_path, "rt") as f:
        data_dict = json.load(f)

    # Restore arrays
    def restore_arrays(obj):
        if isinstance(obj, dict):
            if "__ndarray__" in obj:
                return jnp.array(obj["__ndarray__"], dtype=obj["dtype"])
            return {k: restore_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [restore_arrays(item) for item in obj]
        return obj

    data_dict = restore_arrays(data_dict)
    dataset = _dict_to_dataset(data_dict)

    print(f"✓ Dataset loaded from {json_path}")
    return dataset


def load_process_collection_json(json_path: Path) -> BioProcessCollection:
    """
    Load a BioProcessCollection from JSON.

    Args:
        json_path: Path to JSON file

    Returns:
        Reconstructed BioProcessCollection
    """
    json_path = Path(json_path)

    with _open_json_file(json_path, "rt") as f:
        data_dict = json.load(f)

    def restore_arrays(obj):
        if isinstance(obj, dict):
            if "__ndarray__" in obj:
                return jnp.array(obj["__ndarray__"], dtype=obj["dtype"])
            return {k: restore_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [restore_arrays(item) for item in obj]
        return obj

    data_dict = restore_arrays(data_dict)
    collection = _dict_to_process_collection(data_dict)

    print(f"✓ Process collection loaded from {json_path}")
    return collection


# ============================================================
# Helper Functions
# ============================================================


def _dataset_to_dict(dataset: BenchmarkDataset) -> Dict:
    """Convert dataset to nested dictionary"""
    return {
        "metadata": dataset.metadata,
        "case_studies": {
            cs_id: _case_study_to_dict(cs) for cs_id, cs in dataset.case_studies.items()
        },
    }


def _process_collection_to_dict(collection: BioProcessCollection) -> Dict:
    """Convert BioProcessCollection to nested dictionary."""
    return {
        "metadata": collection.metadata,
        "processes": {
            p_id: _process_to_dict(process)
            for p_id, process in collection.processes.items()
        },
    }


def _case_study_to_dict(case_study: CaseStudy) -> Dict:
    """Convert CaseStudy to nested dictionary."""
    return {
        "case_id": case_study.case_id,
        "organism": case_study.organism,
        "citation": case_study.citation,
        "processes": {
            p_id: _process_to_dict(process)
            for p_id, process in case_study.processes.items()
        },
    }


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
    return {
        "name": comp.name,
        "unit": comp.unit,
        "is_intracellular": comp.is_intracellular,
        "concentration": _timeseries_or_static_to_dict(comp.concentration),
        "interpolator": _interpolator_to_dict(comp.interpolator)
        if comp.interpolator is not None
        else None,
    }


def _process_variable_to_dict(pv: ProcessVariable) -> Dict:
    """Convert ProcessVariable to dictionary"""
    return {
        "name": pv.name,
        "unit": pv.unit,
        "is_controlled": pv.is_controlled,
        "values": _timeseries_or_static_to_dict(pv.values),
        "interpolator": _interpolator_to_dict(pv.interpolator)
        if pv.interpolator is not None
        else None,
    }


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
        payload["metadata"] = value.metadata
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
    return {
        "initial_volume": volume.initial_volume,
        "unit": volume.unit,
        "volume_changes": {
            name: _volume_change_to_dict(vc)
            for name, vc in volume.volume_changes.items()
        },
    }


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

    result["interpolator"] = (
        _interpolator_to_dict(vc.interpolator)
        if getattr(vc, "interpolator", None) is not None
        else None
    )
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


def _dict_to_dataset(data: Dict) -> BenchmarkDataset:
    """Reconstruct BenchmarkDataset from dictionary"""
    case_studies = {}

    for cs_id, cs_data in data.get("case_studies", {}).items():
        case_studies[cs_id] = _dict_to_case_study(cs_data)

    return BenchmarkDataset(
        metadata=data.get("metadata", {}), case_studies=case_studies
    )


def _dict_to_process_collection(data: Dict) -> BioProcessCollection:
    """Reconstruct BioProcessCollection from dictionary."""
    return BioProcessCollection(
        metadata=data.get("metadata"),
        processes={
            p_id: _dict_to_process(p_data)
            for p_id, p_data in data.get("processes", {}).items()
        },
    )


def _dict_to_case_study(data: Dict) -> CaseStudy:
    """Reconstruct CaseStudy from dictionary."""
    return CaseStudy(
        case_id=data["case_id"],
        organism=data["organism"],
        citation=data["citation"],
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

    if p_data.get("__type__") == "AugmentedBioProcess":
        parent = p_data.get("parent_process")
        if not isinstance(parent, str) or not parent:
            raise ValueError(
                "AugmentedBioProcess payload missing required "
                "'parent_process' string"
            )
        return AugmentedBioProcess(
            metadata=metadata,
            time_axis=time_axis,
            volume=volume,
            reactor_medium=reactor_medium,
            process_variables=process_variables,
            discrete_events=discrete_events,
            parent_process=parent,
        )

    return BioProcess(
        metadata=metadata,
        time_axis=time_axis,
        volume=volume,
        reactor_medium=reactor_medium,
        process_variables=process_variables,
        discrete_events=discrete_events,
    )


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
    interpolator = None
    interpolator_data = comp_data.get("interpolator")
    if interpolator_data is not None:
        interpolator = _dict_to_interpolator(interpolator_data)
    return ReactorMediumComponent(
        name=comp_data["name"],
        unit=comp_data["unit"],
        is_intracellular=comp_data["is_intracellular"],
        concentration=_dict_to_timeseries_or_static(comp_data["concentration"]),
        interpolator=interpolator,
    )


def _dict_to_process_variable(pv_data: Dict) -> ProcessVariable:
    """Reconstruct ProcessVariable from dictionary"""
    interpolator = None
    interpolator_data = pv_data.get("interpolator")
    if interpolator_data is not None:
        interpolator = _dict_to_interpolator(interpolator_data)
    return ProcessVariable(
        name=pv_data["name"],
        unit=pv_data["unit"],
        is_controlled=pv_data["is_controlled"],
        values=_dict_to_timeseries_or_static(pv_data["values"]),
        interpolator=interpolator,
    )


def _timeseries_from_dict_payload(value_data: Dict) -> TimeSeries:
    """Reconstruct TimeSeries from a typed or untyped serialized payload."""
    times = value_data.get("times")
    values = value_data.get("values")

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
        kwargs["metadata"] = value_data["metadata"]

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

    return Volume(
        initial_volume=vol_data["initial_volume"],
        unit=vol_data["unit"],
        volume_changes=volume_changes,
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

    interpolator = None
    interpolator_data = vc_data.get("interpolator")
    if interpolator_data is not None:
        interpolator = _dict_to_interpolator(interpolator_data)

    if vc_type == "FeedVolumeChange":
        feed_medium = None
        if vc_data.get("feed_medium"):
            feed_medium = _dict_to_feed_medium(vc_data["feed_medium"])
        return FeedVolumeChange(
            **common, feed_medium=feed_medium, interpolator=interpolator
        )
    elif vc_type == "SampleVolumeChange":
        return SampleVolumeChange(**common, interpolator=interpolator)
    else:
        raise ValueError(f"Unknown volume change type: {vc_type}")


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
# Interpolator and DiscreteEvents serialization helpers
# ============================================================

_SEGMENTED_INTERPOLATOR_KINDS = {
    "interpax_cubic",
    "interpax_linear",
}


def _interpolator_to_dict(interpolator: Interpolator) -> Dict:
    """Convert Interpolator to dictionary (compact JSON form)."""
    result = {"kind": interpolator.kind}

    if interpolator.kind in _SEGMENTED_INTERPOLATOR_KINDS:
        n_seg = int(interpolator.n_segments or 0)
        n_per_seg = [int(interpolator.n[i]) for i in range(n_seg)]
        result.update(
            {
                "x": [
                    np.asarray(interpolator.x[i, : n_per_seg[i]]).tolist()
                    for i in range(n_seg)
                ],
                "y": [
                    np.asarray(interpolator.y[i, : n_per_seg[i]]).tolist()
                    for i in range(n_seg)
                ],
                "n": n_per_seg,
                "n_segments": n_seg,
                "segment_boundaries": np.asarray(
                    interpolator.segment_boundaries[: n_seg + 1]
                ).tolist(),
                "bc_type": interpolator.bc_type,
            }
        )
    elif interpolator.kind == "interpax_ppoly":
        result.update(
            {
                "x": np.asarray(interpolator.x).tolist(),
                "coefficients": np.asarray(interpolator.coefficients).tolist(),
                "extrapolate": interpolator.extrapolate,
            }
        )
    else:
        raise ValueError(
            f"Unsupported interpolator kind for serialization: {interpolator.kind}"
        )

    result["interpolator_metadata"] = interpolator.interpolator_metadata
    return result


def _dict_to_interpolator(data: Dict) -> Interpolator:
    """Reconstruct Interpolator from compact dictionary formats."""
    kind = data["kind"]
    metadata = data.get("interpolator_metadata")

    if kind == "interpax_ppoly":
        x_raw = data["x"]
        coeff_raw = data["coefficients"]
        x = x_raw if isinstance(x_raw, jnp.ndarray) else jnp.array(x_raw)
        coefficients = (
            coeff_raw if isinstance(coeff_raw, jnp.ndarray) else jnp.array(coeff_raw)
        )
        return Interpolator(
            kind=kind,
            x=x,
            coefficients=coefficients,
            extrapolate=data.get("extrapolate", True),
            interpolator_metadata=metadata,
        )

    n_segments = data["n_segments"]
    x_raw = data["x"]
    y_raw = data["y"]
    n_raw = data["n"]
    seg_b_raw = data["segment_boundaries"]

    is_unpadded = (
        isinstance(x_raw, list) and len(x_raw) > 0 and isinstance(x_raw[0], list)
    )

    if is_unpadded:
        n_per_seg = [int(v) for v in n_raw]
        max_ctrl = max(len(row) for row in x_raw)

        x = np.zeros((n_segments, max_ctrl))
        y = np.zeros((n_segments, max_ctrl))
        for i in range(n_segments):
            ni = len(x_raw[i])
            x[i, :ni] = x_raw[i]
            y[i, :ni] = y_raw[i]

        n_arr = np.array(n_per_seg, dtype=int)
        seg_b = np.zeros(n_segments + 1)
        for i, v in enumerate(seg_b_raw):
            seg_b[i] = v

        x = jnp.array(x)
        y = jnp.array(y)
        n_arr = jnp.array(n_arr)
        seg_b = jnp.array(seg_b)
    else:
        x = x_raw if isinstance(x_raw, jnp.ndarray) else jnp.array(x_raw)
        y = y_raw if isinstance(y_raw, jnp.ndarray) else jnp.array(y_raw)
        n_arr = n_raw if isinstance(n_raw, jnp.ndarray) else jnp.array(n_raw)
        seg_b = (
            seg_b_raw if isinstance(seg_b_raw, jnp.ndarray) else jnp.array(seg_b_raw)
        )

    return Interpolator(
        kind=kind,
        x=x,
        y=y,
        n=n_arr,
        n_segments=n_segments,
        segment_boundaries=seg_b,
        bc_type=data.get("bc_type", "natural"),
        interpolator_metadata=metadata,
    )


def _discrete_events_to_dict(de: DiscreteEvents) -> Dict:
    """Convert DiscreteEvents to dictionary"""
    return {
        "times": de.times,
        "labels": de.labels,
        "metadata": de.metadata,
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


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/JAX arrays"""

    def default(self, obj):
        if isinstance(obj, (jnp.ndarray, np.ndarray)):
            return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype)}
        return super().default(obj)
