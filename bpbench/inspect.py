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
    if process.time_axis is not None:
        print(f"\n{prefix}Time:")
        print(f"{prefix}  Range: {process.time_axis.start:.2f} to {process.time_axis.end:.2f} {process.time_axis.unit}")
        print(f"{prefix}  Reference: {process.time_axis.time_reference}")
    
    # Reactor medium
    if process.reactor_medium:
        print(f"\n{prefix}Reactor Medium: ({len(process.reactor_medium)} total)")
        for name, medium in process.reactor_medium.items():
            print(f"{prefix}  {name}:")
            print(f"{prefix}    Density: {medium.density} {medium.density_unit}")
            if medium.components:
                print(f"{prefix}    Components: ({len(medium.components)} total)")
                for comp_name, comp in medium.components.items():
                    _print_reactor_component_info(comp, prefix + "      ", show_values)
    
    # Process variables (pH, temperature, etc.)
    if process.process_variables:
        print(f"\n{prefix}Process Variables: ({len(process.process_variables)} total)")
        for name, pv in process.process_variables.items():
            _print_process_variable_info(pv, prefix + "  ", show_values)
    
    # Volume information
    if process.volume is not None:
        print(f"\n{prefix}Volume:")
        print(f"{prefix}  Initial: {process.volume.initial_volume} {process.volume.unit}")
        if process.volume.volume_changes:
            print(f"{prefix}  Volume Changes: ({len(process.volume.volume_changes)} total)")
            for name, change in process.volume.volume_changes.items():
                _print_volume_change_info(change, prefix + "    ", show_values)
    
    if indent == 0:
        print("=" * 80)



def _print_process_variable_info(pv, prefix: str, show_values: bool = False) -> None:
    """Helper function to print ProcessVariable information."""
    print(f"{prefix}{pv.name}")
    print(f"{prefix}  Unit: {pv.unit}")
    print(f"{prefix}  Controlled: {pv.is_controlled}")
    
    # Check if values is TimeSeries or StaticVariable
    if hasattr(pv.values, 'timepoints'):  # TimeSeries
        ts = pv.values
        n_points = len(ts.timepoints)
        print(f"{prefix}  TimeSeries Data: {n_points} points")
        if n_points > 0:
            t_range = (float(ts.timepoints[0]), float(ts.timepoints[-1]))
            v_range = (float(jnp.min(ts.values)), float(jnp.max(ts.values)))
            print(f"{prefix}    Time range: {t_range[0]:.2f} to {t_range[1]:.2f}")
            print(f"{prefix}    Value range: {v_range[0]:.4f} to {v_range[1]:.4f}")
            
            if show_values and n_points <= 5:
                print(f"{prefix}    Values: {ts.values}")
            elif show_values:
                print(f"{prefix}    First 3: {ts.values[:3]}")
    elif hasattr(pv.values, 'value'):  # StaticVariable
        print(f"{prefix}  Static Value: {pv.values.value}")
    
    if pv.spline is not None:
        print(f"{prefix}  Spline: available")


def _print_reactor_component_info(comp, prefix: str, show_values: bool = False) -> None:
    """Helper function to print ReactorMediumComponent information."""
    print(f"{prefix}{comp.name}")
    print(f"{prefix}  Unit: {comp.unit}")
    print(f"{prefix}  Intracellular: {comp.is_intracellular}")
    
    # Check if concentration is TimeSeries or StaticVariable
    if hasattr(comp.concentration, 'timepoints'):  # TimeSeries
        ts = comp.concentration
        n_points = len(ts.timepoints)
        print(f"{prefix}  TimeSeries Data: {n_points} points")
        if n_points > 0:
            t_range = (float(ts.timepoints[0]), float(ts.timepoints[-1]))
            v_range = (float(jnp.min(ts.values)), float(jnp.max(ts.values)))
            print(f"{prefix}    Time range: {t_range[0]:.2f} to {t_range[1]:.2f}")
            print(f"{prefix}    Value range: {v_range[0]:.4f} to {v_range[1]:.4f}")
            
            if show_values and n_points <= 5:
                print(f"{prefix}    Values: {ts.values}")
            elif show_values:
                print(f"{prefix}    First 3: {ts.values[:3]}")
    elif hasattr(comp.concentration, 'value'):  # StaticVariable
        print(f"{prefix}  Static Concentration: {comp.concentration.value}")


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
    print(f"{prefix}  Type: {'Controlled' if change.is_controlled else 'Modeled'}, "
          f"{'Continuous' if change.is_continuous else 'Discrete'}")
    print(f"{prefix}  Unit: {change.unit}")
    
    if change.feed_medium:
        print(f"{prefix}  Feed Medium: {change.feed_medium.name}")
    
    if change.values is not None:
        n_points = len(change.values.timepoints)
        print(f"{prefix}  TimeSeries Points: {n_points}")
        if n_points > 0:
            v_range = (float(jnp.min(change.values.values)), 
                      float(jnp.max(change.values.values)))
            print(f"{prefix}    Value range: {v_range[0]:.4f} to {v_range[1]:.4f} {change.unit}")
            if change.is_continuous:
                total_change = float(change.values.values[-1] - change.values.values[0])
                print(f"{prefix}    Total change: {total_change:.2f} {change.unit}")
            else:
                total_change = float(jnp.sum(change.values.values))
                print(f"{prefix}    Total change: {total_change:.2f} {change.unit}")
