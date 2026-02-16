import jax.numpy as jnp
from typing import Tuple, List, Generator
from .dataclasses import BioProcess, CaseStudy, BenchmarkDataset, TimeSeries, VolumeChange


def print_structure(process: BioProcess, indent: int = 0, show_values: bool = False) -> None:
    """
    Print a hierarchical view of the BioProcess object structure.
    
    This function displays the complete structure of a BioProcess object in a 
    human-readable format, showing all fields, their types, and sizes.
    
    Args:
        process: BioProcess object to inspect
        indent: Starting indentation level (used internally for recursion)
        show_values: If True, show sample values for arrays (first few elements)
        
    Example:
        >>> from bpbench import load_dataset
        >>> dataset = load_dataset("path/to/dataset")
        >>> process = dataset.case_studies["case1"].processes["proc1"]
        >>> print_structure(process)
    """
    prefix = "  " * indent
    
    # Header
    if indent == 0:
        print("=" * 80)
        print("BioProcess Structure")
        print("=" * 80)
    
    # Basic information - BioProcess uses metadata instead of direct fields
    print(f"{prefix}Process Name: {process.metadata.name}")
    print(f"{prefix}Process Type: {process.metadata.process_type}")
    if process.metadata.notes:
        print(f"{prefix}Notes: {process.metadata.notes}")
    
    # Time axis
    if process.time is not None:
        print(f"\n{prefix}Time:")
        print(f"{prefix}  Range: {process.time.start:.2f} to {process.time.end:.2f} {process.time.unit}")
        print(f"{prefix}  Reference: {process.time.time_reference}")
    
    # Dynamic variables (combines states and controls)
    if process.dynamic_variables:
        print(f"\n{prefix}Dynamic Variables: ({len(process.dynamic_variables)} total)")
        for name, ts in process.dynamic_variables.items():
            _print_timeseries_info(ts, prefix + "  ", show_values)
    
    # Static variables (combines static controls and parameters)
    if process.static_variables:
        print(f"\n{prefix}Static Variables: ({len(process.static_variables)} total)")
        for name, var in process.static_variables.items():
            print(f"{prefix}  {name}: {var.value} {var.unit}")
    
    # Volume information
    if process.volume is not None:
        print(f"\n{prefix}Volume:")
        print(f"{prefix}  Initial: {process.volume.initial_volume} {process.volume.volume_unit}")
        print(f"{prefix}  Density: {process.volume.density} {process.volume.density_unit}")
        if process.volume.volume_changes:
            print(f"{prefix}  Volume Changes: ({len(process.volume.volume_changes)} total)")
            for name, change in process.volume.volume_changes.items():
                _print_volume_change_info(change, prefix + "    ", show_values)
    
    if indent == 0:
        print("=" * 80)



def _print_timeseries_info(ts: TimeSeries, prefix: str, show_values: bool = False) -> None:
    """Helper function to print TimeSeries information."""
    print(f"{prefix}{ts.name}")
    print(f"{prefix}  Unit: {ts.unit}")
    print(f"{prefix}  Controlled: {ts.controlled}")
    
    if ts.raw is not None:
        n_points = len(ts.raw.timepoints)
        print(f"{prefix}  Raw Data: {n_points} points")
        if n_points > 0:
            t_range = (float(ts.raw.timepoints[0]), float(ts.raw.timepoints[-1]))
            v_range = (float(jnp.min(ts.raw.values)), float(jnp.max(ts.raw.values)))
            print(f"{prefix}    Time range: {t_range[0]:.2f} to {t_range[1]:.2f}")
            print(f"{prefix}    Value range: {v_range[0]:.4f} to {v_range[1]:.4f}")
            
            if show_values and n_points <= 5:
                print(f"{prefix}    Values: {ts.raw.values}")
            elif show_values:
                print(f"{prefix}    First 3: {ts.raw.values[:3]}")
        
        if ts.raw.measurement_std is not None:
            print(f"{prefix}    Measurement std: provided")
    
    if ts.spline is not None:
        print(f"{prefix}  Spline: available")


def _print_volume_change_info(change: VolumeChange, prefix: str, show_values: bool = False) -> None:
    """Helper function to print VolumeChange information."""
    print(f"{prefix}{change.name}:")
    print(f"{prefix}  Type: {'Controlled' if change.controlled else 'Modeled'}, "
          f"{'Continuous' if change.continuous else 'Discrete'}")
    print(f"{prefix}  Unit: {change.unit}")
    
    if change.feed_medium:
        print(f"{prefix}  Feed Medium: {change.feed_medium}")
    
    if change.timeseries is not None:
        print(f"{prefix}  TimeSeries: {change.timeseries.name}")
        if change.timeseries.raw is not None:
            n_points = len(change.timeseries.raw.timepoints)
            print(f"{prefix}    Points: {n_points}")
            if n_points > 0:
                v_range = (float(jnp.min(change.timeseries.raw.values)), 
                          float(jnp.max(change.timeseries.raw.values)))
                print(f"{prefix}    Value range: {v_range[0]:.4f} to {v_range[1]:.4f} {change.unit}")
                if change.continuous:
                    total_change = float(change.timeseries.raw.values[-1] - change.timeseries.raw.values[0])
                    print(f"{prefix}    Total change: {total_change:.2f} {change.unit}")
                else:
                    total_change = float(jnp.sum(change.timeseries.raw.values))
                    print(f"{prefix}    Total change: {total_change:.2f} {change.unit}")
