"""
Serialization utilities for bioprocess benchmarking dataset
Hybrid approach: YAML for metadata + HDF5 for arrays
"""

import yaml
import h5py
import json
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Union
from .dataclasses import (
    BenchmarkDataset, CaseStudy, BioProcess, TimeSeries, TimeAxis,
    SplineRepresentation, DiscreteEvents, FeedMedium, FeedMediumComponent,
    StaticVariable, BioProcessMetadata, Volume, BaseVolumeChange,
    FeedVolumeChange, SampleVolumeChange,
    ReactorMedium, ReactorMediumComponent, ProcessVariable
)


# ============================================================
# Serialization to YAML + HDF5
# ============================================================

def save_dataset(dataset: BenchmarkDataset, base_path: Path) -> None:
    """
    Save dataset using hybrid approach:
    - metadata.yaml: human-readable structure
    - arrays.h5: efficient binary storage for JAX arrays
    
    Args:
        dataset: BenchmarkDataset to save
        base_path: Directory path where files will be saved
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Convert dataset to nested dict
    metadata_dict = _dataset_to_dict(dataset)
    
    # Extract arrays and replace with references
    arrays_store = {}
    _extract_arrays(metadata_dict, arrays_store, prefix="")
    
    # Save human-readable metadata
    with open(base_path / "metadata.yaml", "w") as f:
        yaml.dump(metadata_dict, f, default_flow_style=False, sort_keys=False)
    
    # Save arrays efficiently
    with h5py.File(base_path / "arrays.h5", "w") as f:
        for key, array in arrays_store.items():
            f.create_dataset(key, data=array)
    
    print(f"✓ Dataset saved to {base_path}")


def load_dataset(base_path: Path) -> BenchmarkDataset:
    """
    Load dataset from YAML + HDF5
    
    Args:
        base_path: Directory path where files are stored
        
    Returns:
        Reconstructed BenchmarkDataset
    """
    base_path = Path(base_path)
    
    # Load metadata
    with open(base_path / "metadata.yaml", "r") as f:
        metadata_dict = yaml.safe_load(f)
    
    # Load arrays
    arrays_store = {}
    with h5py.File(base_path / "arrays.h5", "r") as f:
        def load_datasets(group, prefix=""):
            """Recursively load all datasets from HDF5"""
            for key in group.keys():
                item = group[key]
                full_key = f"{prefix}/{key}" if prefix else key
                if isinstance(item, h5py.Dataset):
                    arrays_store[full_key] = jnp.array(item[:])
                elif isinstance(item, h5py.Group):
                    load_datasets(item, full_key)
        load_datasets(f)
    
    # Reconstruct arrays in metadata
    _restore_arrays(metadata_dict, arrays_store)
    
    # Reconstruct dataclasses
    dataset = _dict_to_dataset(metadata_dict)
    
    print(f"✓ Dataset loaded from {base_path}")
    return dataset


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
    
    with open(json_path, "w") as f:
        json.dump(data_dict, f, indent=2, cls=NumpyEncoder)
    
    print(f"✓ Dataset saved to {json_path}")


def load_dataset_json(json_path: Path) -> BenchmarkDataset:
    """
    Load dataset from JSON
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Reconstructed BenchmarkDataset
    """
    json_path = Path(json_path)
    
    with open(json_path, "r") as f:
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


# ============================================================
# Helper Functions for YAML + HDF5
# ============================================================

def _dataset_to_dict(dataset: BenchmarkDataset) -> Dict:
    """Convert dataset to nested dictionary"""
    return {
        "metadata": dataset.metadata,
        "case_studies": {
            cs_id: {
                "case_id": cs.case_id,
                "organism": cs.organism,
                "citation": cs.citation,
                "processes": {
                    p_id: _process_to_dict(p)
                    for p_id, p in cs.processes.items()
                }
            }
            for cs_id, cs in dataset.case_studies.items()
        }
    }


def _process_to_dict(process: BioProcess) -> Dict:
    """Convert BioProcess to dictionary"""
    result = {
        "metadata": {
            "name": process.metadata.name,
            "process_type": process.metadata.process_type,
            "notes": process.metadata.notes
        },
        "time_axis": {
            "unit": process.time_axis.unit,
            "start": process.time_axis.start,
            "end": process.time_axis.end,
            "time_reference": process.time_axis.time_reference
        } if process.time_axis else None,
        "reactor_medium": _reactor_medium_to_dict(process.reactor_medium) if process.reactor_medium else None,
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
    
    return result


def _reactor_medium_to_dict(reactor_medium: ReactorMedium) -> Dict:
    """Convert ReactorMedium to dictionary"""
    return {
        "name": reactor_medium.name,
        "density": reactor_medium.density,
        "density_unit": reactor_medium.density_unit,
        "components": {
            name: _reactor_component_to_dict(comp)
            for name, comp in reactor_medium.components.items()
        }
    }


def _reactor_component_to_dict(comp: ReactorMediumComponent) -> Dict:
    """Convert ReactorMediumComponent to dictionary"""
    return {
        "name": comp.name,
        "unit": comp.unit,
        "is_intracellular": comp.is_intracellular,
        "concentration": _timeseries_or_static_to_dict(comp.concentration),
        "spline": _spline_to_dict(comp.spline) if comp.spline is not None else None,
    }


def _process_variable_to_dict(pv: ProcessVariable) -> Dict:
    """Convert ProcessVariable to dictionary"""
    return {
        "name": pv.name,
        "unit": pv.unit,
        "is_controlled": pv.is_controlled,
        "values": _timeseries_or_static_to_dict(pv.values),
        "spline": _spline_to_dict(pv.spline) if pv.spline is not None else None,
    }


def _timeseries_or_static_to_dict(value: Union[TimeSeries, StaticVariable]) -> Dict:
    """Convert TimeSeries or StaticVariable to dictionary"""
    if isinstance(value, TimeSeries):
        return {
            "type": "TimeSeries",
            "timepoints": value.timepoints,
            "values": value.values
        }
    elif isinstance(value, StaticVariable):
        return {
            "type": "StaticVariable",
            "value": value.value
        }
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
        }
    }


def _volume_change_to_dict(vc) -> Dict:
    """Convert FeedVolumeChange or SampleVolumeChange to dictionary"""
    result = {
        "name": vc.name,
        "unit": vc.unit,
        "is_controlled": vc.is_controlled,
        "is_continuous": vc.is_continuous,
        "values": {
            "timepoints": vc.values.timepoints,
            "values": vc.values.values
        } if vc.values else None
    }
    if isinstance(vc, FeedVolumeChange):
        result["type"] = "FeedVolumeChange"
        result["feed_medium"] = _feed_medium_to_dict(vc.feed_medium) if vc.feed_medium else None
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
        }
    }


def _feed_component_to_dict(comp: FeedMediumComponent) -> Dict:
    """Convert FeedMediumComponent to dictionary"""
    return {
        "name": comp.name,
        "unit": comp.unit,
        "is_controlled": comp.is_controlled,
        "concentration": _timeseries_or_static_to_dict(comp.concentration)
    }


def _extract_arrays(obj: Any, store: Dict, prefix: str) -> None:
    """Recursively extract JAX arrays and replace with references"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}/{key}" if prefix else key
            if isinstance(value, (jnp.ndarray, np.ndarray)):
                store[new_prefix] = value
                obj[key] = f"@array:{new_prefix}"
            else:
                _extract_arrays(value, store, new_prefix)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _extract_arrays(item, store, f"{prefix}/{i}")


def _restore_arrays(obj: Any, store: Dict) -> None:
    """Recursively restore JAX arrays from references"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and value.startswith("@array:"):
                array_key = value[7:]  # remove "@array:" prefix
                obj[key] = store[array_key]
            else:
                _restore_arrays(value, store)
    elif isinstance(obj, list):
        for item in obj:
            _restore_arrays(item, store)


def _dict_to_dataset(data: Dict) -> BenchmarkDataset:
    """Reconstruct BenchmarkDataset from dictionary"""
    case_studies = {}
    
    for cs_id, cs_data in data.get("case_studies", {}).items():
        processes = {}
        
        for p_id, p_data in cs_data.get("processes", {}).items():
            processes[p_id] = _dict_to_process(p_data)
        
        case_studies[cs_id] = CaseStudy(
            case_id=cs_data["case_id"],
            organism=cs_data["organism"],
            citation=cs_data["citation"],
            processes=processes
        )
    
    return BenchmarkDataset(
        metadata=data.get("metadata", {}),
        case_studies=case_studies
    )


def _dict_to_process(p_data: Dict) -> BioProcess:
    """Reconstruct BioProcess from dictionary"""
    # Reconstruct metadata
    metadata = BioProcessMetadata(
        name=p_data["metadata"]["name"],
        process_type=p_data["metadata"]["process_type"],
        notes=p_data["metadata"].get("notes")
    )
    
    # Reconstruct time axis
    time_axis = None
    if p_data.get("time_axis"):
        time_axis = TimeAxis(
            unit=p_data["time_axis"]["unit"],
            start=p_data["time_axis"]["start"],
            end=p_data["time_axis"]["end"],
            time_reference=p_data["time_axis"]["time_reference"]
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
        components=components
    )


def _dict_to_reactor_component(comp_data: Dict) -> ReactorMediumComponent:
    """Reconstruct ReactorMediumComponent from dictionary"""
    spline = None
    if comp_data.get("spline") is not None:
        spline = _dict_to_spline(comp_data["spline"])
    return ReactorMediumComponent(
        name=comp_data["name"],
        unit=comp_data["unit"],
        is_intracellular=comp_data["is_intracellular"],
        concentration=_dict_to_timeseries_or_static(comp_data["concentration"]),
        spline=spline,
    )


def _dict_to_process_variable(pv_data: Dict) -> ProcessVariable:
    """Reconstruct ProcessVariable from dictionary"""
    spline = None
    if pv_data.get("spline") is not None:
        spline = _dict_to_spline(pv_data["spline"])
    return ProcessVariable(
        name=pv_data["name"],
        unit=pv_data["unit"],
        is_controlled=pv_data["is_controlled"],
        values=_dict_to_timeseries_or_static(pv_data["values"]),
        spline=spline,
    )


def _dict_to_timeseries_or_static(value_data: Dict) -> Union[TimeSeries, StaticVariable]:
    """Reconstruct TimeSeries or StaticVariable from dictionary"""
    if value_data["type"] == "TimeSeries":
        return TimeSeries(
            timepoints=value_data["timepoints"],
            values=value_data["values"]
        )
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
        volume_changes=volume_changes
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
        values = TimeSeries(
            timepoints=vc_data["values"]["timepoints"],
            values=vc_data["values"]["values"]
        )

    common = dict(
        name=vc_data["name"],
        unit=vc_data["unit"],
        is_controlled=vc_data["is_controlled"],
        is_continuous=vc_data["is_continuous"],
        values=values,
    )

    if vc_type == "FeedVolumeChange":
        feed_medium = None
        if vc_data.get("feed_medium"):
            feed_medium = _dict_to_feed_medium(vc_data["feed_medium"])
        return FeedVolumeChange(**common, feed_medium=feed_medium)
    elif vc_type == "SampleVolumeChange":
        return SampleVolumeChange(**common)
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
        components=components
    )


def _dict_to_feed_component(comp_data: Dict) -> FeedMediumComponent:
    """Reconstruct FeedMediumComponent from dictionary"""
    return FeedMediumComponent(
        name=comp_data["name"],
        unit=comp_data["unit"],
        is_controlled=comp_data["is_controlled"],
        concentration=_dict_to_timeseries_or_static(comp_data["concentration"])
    )


# ============================================================
# SplineRepresentation and DiscreteEvents serialization helpers
# ============================================================

def _spline_to_dict(spline: SplineRepresentation) -> Dict:
    """Convert SplineRepresentation to dictionary (unpadded for compact JSON)."""
    n_seg = spline.n_segments
    n_per_seg = [int(spline.n[i]) for i in range(n_seg)]
    return {
        "kind": spline.kind,
        "x": [np.asarray(spline.x[i, :n_per_seg[i]]).tolist() for i in range(n_seg)],
        "y": [np.asarray(spline.y[i, :n_per_seg[i]]).tolist() for i in range(n_seg)],
        "n": n_per_seg,
        "n_segments": n_seg,
        "segment_boundaries": np.asarray(spline.segment_boundaries[:n_seg + 1]).tolist(),
        "bc_type": spline.bc_type,
        "spline_metadata": spline.spline_metadata,
    }


def _dict_to_spline(data: Dict) -> SplineRepresentation:
    """Reconstruct SplineRepresentation from dictionary.

    Handles both the compact unpadded format (list-of-lists) and the legacy
    padded format (``__ndarray__`` dicts or pre-converted jnp arrays).
    """
    n_segments = data["n_segments"]
    x_raw = data["x"]
    y_raw = data["y"]
    n_raw = data["n"]
    seg_b_raw = data["segment_boundaries"]

    # Detect format: new unpadded (list-of-lists) vs old padded (ndarray)
    is_unpadded = (
        isinstance(x_raw, list)
        and len(x_raw) > 0
        and isinstance(x_raw[0], list)
    )

    if is_unpadded:
        # New compact format: re-pad to exact size
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
        # Legacy padded format
        x = x_raw if isinstance(x_raw, jnp.ndarray) else jnp.array(x_raw)
        y = y_raw if isinstance(y_raw, jnp.ndarray) else jnp.array(y_raw)
        n_arr = n_raw if isinstance(n_raw, jnp.ndarray) else jnp.array(n_raw)
        seg_b = seg_b_raw if isinstance(seg_b_raw, jnp.ndarray) else jnp.array(seg_b_raw)

    return SplineRepresentation(
        kind=data["kind"],
        x=x,
        y=y,
        n=n_arr,
        n_segments=n_segments,
        segment_boundaries=seg_b,
        bc_type=data.get("bc_type", "natural"),
        spline_metadata=data.get("spline_metadata"),
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
