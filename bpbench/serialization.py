# """
# Serialization utilities for bioprocess benchmarking dataset
# Hybrid approach: YAML for metadata + HDF5 for arrays
# """

# import yaml
# import h5py
# import json
# import jax.numpy as jnp
# import numpy as np
# from pathlib import Path
# from typing import Any, Dict, Optional
# from .dataclasses import (
#     BenchmarkDataset, CaseStudy, Process, TimeSeries, TimeAxis,
#     RawTimeSeries, SplineRepresentation, Feed, FeedComponent,
#     StaticVariable, ReactorProperties, Volume, VolumeChange
# )


# # ============================================================
# # Serialization to YAML + HDF5
# # ============================================================

# def save_dataset(dataset: BenchmarkDataset, base_path: Path) -> None:
#     """
#     Save dataset using hybrid approach:
#     - metadata.yaml: human-readable structure
#     - arrays.h5: efficient binary storage for JAX arrays
    
#     Args:
#         dataset: BenchmarkDataset to save
#         base_path: Directory path where files will be saved
#     """
#     base_path = Path(base_path)
#     base_path.mkdir(parents=True, exist_ok=True)
    
#     # Convert dataset to nested dict
#     metadata_dict = _dataset_to_dict(dataset)
    
#     # Extract arrays and replace with references
#     arrays_store = {}
#     _extract_arrays(metadata_dict, arrays_store, prefix="")
    
#     # Save human-readable metadata
#     with open(base_path / "metadata.yaml", "w") as f:
#         yaml.dump(metadata_dict, f, default_flow_style=False, sort_keys=False)
    
#     # Save arrays efficiently
#     with h5py.File(base_path / "arrays.h5", "w") as f:
#         for key, array in arrays_store.items():
#             f.create_dataset(key, data=array)
    
#     print(f"✓ Dataset saved to {base_path}")


# def load_dataset(base_path: Path) -> BenchmarkDataset:
#     """
#     Load dataset from YAML + HDF5
    
#     Args:
#         base_path: Directory path where files are stored
        
#     Returns:
#         Reconstructed BenchmarkDataset
#     """
#     base_path = Path(base_path)
    
#     # Load metadata
#     with open(base_path / "metadata.yaml", "r") as f:
#         metadata_dict = yaml.safe_load(f)
    
#     # Load arrays
#     arrays_store = {}
#     with h5py.File(base_path / "arrays.h5", "r") as f:
#         def load_datasets(group, prefix=""):
#             """Recursively load all datasets from HDF5"""
#             for key in group.keys():
#                 item = group[key]
#                 full_key = f"{prefix}/{key}" if prefix else key
#                 if isinstance(item, h5py.Dataset):
#                     arrays_store[full_key] = jnp.array(item[:])
#                 elif isinstance(item, h5py.Group):
#                     load_datasets(item, full_key)
#         load_datasets(f)
    
#     # Reconstruct arrays in metadata
#     _restore_arrays(metadata_dict, arrays_store)
    
#     # Reconstruct dataclasses
#     dataset = _dict_to_dataset(metadata_dict)
    
#     print(f"✓ Dataset loaded from {base_path}")
#     return dataset


# # ============================================================
# # Helper Functions for YAML + HDF5
# # ============================================================

# def _dataset_to_dict(dataset: BenchmarkDataset) -> Dict:
#     """Convert dataset to nested dictionary"""
#     return {
#         "metadata": dataset.metadata,
#         "case_studies": {
#             cs_id: {
#                 "case_id": cs.case_id,
#                 "organism": cs.organism,
#                 "citation": cs.citation,
#                 "processes": {
#                     p_id: _process_to_dict(p)
#                     for p_id, p in cs.processes.items()
#                 }
#             }
#             for cs_id, cs in dataset.case_studies.items()
#         }
#     }


# def _volume_change_to_dict(vc: VolumeChange) -> Dict:
#     """Convert VolumeChange to dictionary"""
#     return {
#         "name": vc.name,
#         "controlled": vc.controlled,
#         "continuous": vc.continuous,
#         "unit": vc.unit,
#         "feed_medium": vc.feed_medium,
#         "timeseries": _timeseries_to_dict(vc.timeseries) if vc.timeseries else None,
#         "timepoints": vc.timepoints,  # will be extracted for HDF5
#         "values": vc.values  # will be extracted for HDF5
#     }


# def _volume_to_dict(volume: Volume) -> Dict:
#     """Convert Volume to dictionary"""
#     return {
#         "volume_changes": {
#             name: _volume_change_to_dict(vc)
#             for name, vc in volume.volume_changes.items()
#         },
#         "initial_volume": volume.initial_volume,
#         "volume_unit": volume.volume_unit
#     }


# def _process_to_dict(process: Process) -> Dict:
#     """Convert process to dictionary"""
#     result = {
#         "process_id": process.process_id,
#         "process_type": process.process_type,
#         "replicate_id": process.replicate_id,
#         "time": {
#             "unit": process.time.unit,
#             "start": process.time.start,
#             "end": process.time.end,
#             "time_reference": process.time.time_reference
#         } if process.time else None,
#         "dynamic_variables": {
#             name: _timeseries_to_dict(ts)
#             for name, ts in process.dynamic_variables.items()
#         },
#         "static_variables": {
#             name: {"value": sv.value, "unit": sv.unit}
#             for name, sv in process.static_variables.items()
#         },
#         "feeds": {
#             name: {
#                 "name": feed.name,
#                 "density": feed.density,
#                 "density_unit": feed.density_unit,
#                 "components": {
#                     comp_name: {
#                         "concentration": comp.concentration,
#                         "unit": comp.unit
#                     }
#                     for comp_name, comp in feed.components.items()
#                 }
#             }
#             for name, feed in process.feeds.items()
#         },
#         "event_times": process.event_times,  # will be extracted
#         "reactor": {
#             "working_volume": process.reactor.working_volume,
#             "volume_unit": process.reactor.volume_unit,
#             "density": process.reactor.density
#         } if process.reactor else None
#     }
    
#     # Add volume if present
#     if process.volume is not None:
#         result["volume"] = _volume_to_dict(process.volume)
    
#     return result


# def _timeseries_to_dict(ts: TimeSeries) -> Dict:
#     """Convert TimeSeries to dictionary"""
#     return {
#         "name": ts.name,
#         "canonical_name": ts.canonical_name,
#         "unit": ts.unit,
#         "role": ts.role,
#         "raw": {
#             "timepoints": ts.raw.timepoints,
#             "values": ts.raw.values,
#             "measurement_std": ts.raw.measurement_std
#         } if ts.raw else None,
#         "spline": {
#             "type": ts.spline.type,
#             "breakpoints": ts.spline.breakpoints,
#             "coefficients": ts.spline.coefficients,
#             "discontinuous": ts.spline.discontinuous,
#             "fit_residual_std": ts.spline.fit_residual_std,
#             "notes": ts.spline.notes
#         } if ts.spline else None
#     }


# def _extract_arrays(obj: Any, store: Dict, prefix: str) -> None:
#     """Recursively extract JAX arrays and replace with references"""
#     if isinstance(obj, dict):
#         for key, value in obj.items():
#             new_prefix = f"{prefix}/{key}" if prefix else key
#             if isinstance(value, jnp.ndarray):
#                 store[new_prefix] = value
#                 obj[key] = f"@array:{new_prefix}"
#             else:
#                 _extract_arrays(value, store, new_prefix)
#     elif isinstance(obj, list):
#         for i, item in enumerate(obj):
#             _extract_arrays(item, store, f"{prefix}/{i}")


# def _restore_arrays(obj: Any, store: Dict) -> None:
#     """Recursively restore JAX arrays from references"""
#     if isinstance(obj, dict):
#         for key, value in obj.items():
#             if isinstance(value, str) and value.startswith("@array:"):
#                 array_key = value[7:]  # remove "@array:" prefix
#                 obj[key] = store[array_key]
#             else:
#                 _restore_arrays(value, store)
#     elif isinstance(obj, list):
#         for item in obj:
#             _restore_arrays(item, store)


# def _dict_to_dataset(data: Dict) -> BenchmarkDataset:
#     """Reconstruct BenchmarkDataset from dictionary"""
#     case_studies = {}
    
#     for cs_id, cs_data in data.get("case_studies", {}).items():
#         processes = {}
        
#         for p_id, p_data in cs_data.get("processes", {}).items():
#             # Reconstruct time axis
#             time = None
#             if p_data.get("time"):
#                 time = TimeAxis(
#                     unit=p_data["time"]["unit"],
#                     start=p_data["time"]["start"],
#                     end=p_data["time"]["end"],
#                     time_reference=p_data["time"]["time_reference"]
#                 )
            
#             # Reconstruct dynamic variables (combine old dynamic_states and dynamic_controls)
#             dynamic_variables = {}
            
#             # Try new unified field first
#             if "dynamic_variables" in p_data:
#                 dynamic_variables = {
#                     name: _dict_to_timeseries(ts_data)
#                     for name, ts_data in p_data["dynamic_variables"].items()
#                 }
#             else:
#                 # Backward compatibility: merge dynamic_states and dynamic_controls
#                 if "dynamic_states" in p_data:
#                     dynamic_variables.update({
#                         name: _dict_to_timeseries(ts_data)
#                         for name, ts_data in p_data["dynamic_states"].items()
#                     })
#                 elif "states" in p_data:  # Even older compatibility
#                     dynamic_variables.update({
#                         name: _dict_to_timeseries(ts_data)
#                         for name, ts_data in p_data["states"].items()
#                     })
                
#                 if "dynamic_controls" in p_data:
#                     dynamic_variables.update({
#                         name: _dict_to_timeseries(ts_data)
#                         for name, ts_data in p_data["dynamic_controls"].items()
#                     })
#                 elif "controls" in p_data:  # Even older compatibility
#                     dynamic_variables.update({
#                         name: _dict_to_timeseries(ts_data)
#                         for name, ts_data in p_data["controls"].items()
#                     })
            
#             # Reconstruct static variables (combine old static_controls and static_parameters)
#             static_variables = {}
            
#             # Try new unified field first
#             if "static_variables" in p_data:
#                 static_variables = {
#                     name: StaticVariable(
#                         value=sv_data["value"],
#                         unit=sv_data["unit"]
#                     )
#                     for name, sv_data in p_data["static_variables"].items()
#                 }
#             else:
#                 # Backward compatibility: merge static_controls and static_parameters
#                 if "static_controls" in p_data:
#                     static_variables.update({
#                         name: StaticVariable(
#                             value=sc_data["value"],
#                             unit=sc_data["unit"]
#                         )
#                         for name, sc_data in p_data["static_controls"].items()
#                     })
                
#                 if "static_parameters" in p_data:
#                     static_variables.update({
#                         name: StaticVariable(
#                             value=sp_data["value"],
#                             unit=sp_data["unit"]
#                         )
#                         for name, sp_data in p_data["static_parameters"].items()
#                     })
            
            
#             # Reconstruct feeds
#             feeds = {}
#             for feed_name, feed_data in p_data.get("feeds", {}).items():
#                 components = {
#                     comp_name: FeedComponent(
#                         concentration=comp_data["concentration"],
#                         unit=comp_data["unit"]
#                     )
#                     for comp_name, comp_data in feed_data.get("components", {}).items()
#                 }
#                 feeds[feed_name] = Feed(
#                     name=feed_data["name"],
#                     density=feed_data["density"],
#                     density_unit=feed_data["density_unit"],
#                     components=components
#                 )
            
#             # Reconstruct reactor
#             reactor = None
#             if p_data.get("reactor"):
#                 reactor = ReactorProperties(
#                     working_volume=p_data["reactor"]["working_volume"],
#                     volume_unit=p_data["reactor"]["volume_unit"],
#                     density=p_data["reactor"].get("density")
#                 )
            
#             # Reconstruct volume
#             volume = None
#             if p_data.get("volume"):
#                 volume = _dict_to_volume(p_data["volume"])
            
#             # Reconstruct process
#             processes[p_id] = Process(
#                 process_id=p_data["process_id"],
#                 process_type=p_data["process_type"],
#                 replicate_id=p_data.get("replicate_id"),
#                 time=time,
#                 dynamic_variables=dynamic_variables,
#                 static_variables=static_variables,
#                 volume=volume,
#                 feeds=feeds,
#                 event_times=p_data.get("event_times"),
#                 reactor=reactor
#             )
        
#         # Reconstruct case study
#         case_studies[cs_id] = CaseStudy(
#             case_id=cs_data["case_id"],
#             organism=cs_data["organism"],
#             citation=cs_data["citation"],
#             processes=processes
#         )
    
#     # Reconstruct dataset
#     return BenchmarkDataset(
#         metadata=data.get("metadata", {}),
#         case_studies=case_studies
#     )


# def _dict_to_volume(vol_data: Dict) -> Volume:
#     """Reconstruct Volume from dictionary"""
#     volume_changes = {}
#     for name, vc_data in vol_data.get("volume_changes", {}).items():
#         timeseries = None
#         if vc_data.get("timeseries"):
#             timeseries = _dict_to_timeseries(vc_data["timeseries"])
        
#         volume_changes[name] = VolumeChange(
#             name=vc_data["name"],
#             controlled=vc_data["controlled"],
#             continuous=vc_data["continuous"],
#             unit=vc_data["unit"],
#             feed_medium=vc_data.get("feed_medium"),
#             timeseries=timeseries,
#             timepoints=vc_data.get("timepoints"),
#             values=vc_data.get("values")
#         )
    
#     return Volume(
#         volume_changes=volume_changes,
#         initial_volume=vol_data.get("initial_volume"),
#         volume_unit=vol_data["volume_unit"]
#     )


# def _dict_to_timeseries(ts_data: Dict) -> TimeSeries:
#     """Reconstruct TimeSeries from dictionary"""
#     raw = None
#     if ts_data.get("raw"):
#         raw = RawTimeSeries(
#             timepoints=ts_data["raw"]["timepoints"],
#             values=ts_data["raw"]["values"],
#             measurement_std=ts_data["raw"].get("measurement_std")
#         )
    
#     spline = None
#     if ts_data.get("spline"):
#         spline = SplineRepresentation(
#             type=ts_data["spline"]["type"],
#             breakpoints=ts_data["spline"]["breakpoints"],
#             coefficients=ts_data["spline"]["coefficients"],
#             discontinuous=ts_data["spline"]["discontinuous"],
#             fit_residual_std=ts_data["spline"].get("fit_residual_std"),
#             notes=ts_data["spline"].get("notes")
#         )
    
#     return TimeSeries(
#         name=ts_data["name"],
#         canonical_name=ts_data.get("canonical_name"),
#         unit=ts_data.get("unit", ""),
#         role=ts_data.get("role", "state"),
#         raw=raw,
#         spline=spline
#     )


# # ============================================================
# # Alternative: Pure JSON (simpler but less efficient)
# # ============================================================

# class NumpyEncoder(json.JSONEncoder):
#     """JSON encoder that handles numpy/JAX arrays"""
#     def default(self, obj):
#         if isinstance(obj, (jnp.ndarray, np.ndarray)):
#             return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype)}
#         return super().default(obj)


# def save_dataset_json(dataset: BenchmarkDataset, filepath: Path) -> None:
#     """
#     Save dataset as single JSON file (human-readable but larger)
    
#     Args:
#         dataset: BenchmarkDataset to save
#         filepath: File path where JSON will be saved
#     """
#     filepath = Path(filepath)
#     filepath.parent.mkdir(parents=True, exist_ok=True)
    
#     data_dict = _dataset_to_dict(dataset)
    
#     with open(filepath, "w") as f:
#         json.dump(data_dict, f, indent=2, cls=NumpyEncoder)
    
#     print(f"✓ Dataset saved to {filepath}")


# def load_dataset_json(filepath: Path) -> BenchmarkDataset:
#     """
#     Load dataset from JSON
    
#     Args:
#         filepath: Path to JSON file
        
#     Returns:
#         Reconstructed BenchmarkDataset
#     """
#     filepath = Path(filepath)
    
#     with open(filepath, "r") as f:
#         data_dict = json.load(f)
    
#     # Restore arrays
#     def restore_arrays(obj):
#         if isinstance(obj, dict):
#             if "__ndarray__" in obj:
#                 return jnp.array(obj["__ndarray__"], dtype=obj["dtype"])
#             return {k: restore_arrays(v) for k, v in obj.items()}
#         elif isinstance(obj, list):
#             return [restore_arrays(item) for item in obj]
#         return obj
    
#     data_dict = restore_arrays(data_dict)
#     dataset = _dict_to_dataset(data_dict)
    
#     print(f"✓ Dataset loaded from {filepath}")
#     return dataset
