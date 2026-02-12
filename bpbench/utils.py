"""
Utility functions for bioprocess benchmarking
"""

import jax.numpy as jnp
from typing import Tuple, List, Generator
from .dataclasses import Process, CaseStudy, BenchmarkDataset, TimeSeries, VolumeChange


def get_event_times(process: Process) -> jnp.ndarray:
    """
    Extract event times for diffrax solver
    
    Args:
        process: Process object containing event times
        
    Returns:
        Array of event times, or empty array if None
    """
    return process.event_times if process.event_times is not None else jnp.array([])


def leave_one_process_out(case_study: CaseStudy) -> Generator[Tuple[List[str], str], None, None]:
    """
    Generator for leave-one-process-out cross-validation.
    
    Args:
        case_study: CaseStudy object containing multiple processes
        
    Yields:
        (train_process_ids, test_process_id) tuples
    """
    process_ids = list(case_study.processes.keys())
    for i, test_id in enumerate(process_ids):
        train_ids = [pid for j, pid in enumerate(process_ids) if j != i]
        yield train_ids, test_id


def iter_loocv(dataset: BenchmarkDataset) -> Generator[Tuple[str, List[str], str], None, None]:
    """
    Iterator for leave-one-process-out cross-validation across all case studies.
    
    Args:
        dataset: BenchmarkDataset containing multiple case studies
        
    Yields:
        (case_id, train_process_ids, test_process_id) tuples
    """
    for case_id, case_study in dataset.case_studies.items():
        for train_ids, test_id in leave_one_process_out(case_study):
            yield case_id, train_ids, test_id


def print_structure(process: Process, indent: int = 0, show_values: bool = False) -> None:
    """
    Print a hierarchical view of the Process object structure.
    
    This function displays the complete structure of a Process object in a 
    human-readable format, showing all fields, their types, and sizes.
    
    Args:
        process: Process object to inspect
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
        print("Process Structure")
        print("=" * 80)
    
    # Basic information
    print(f"{prefix}Process ID: {process.process_id}")
    print(f"{prefix}Process Type: {process.process_type}")
    if process.replicate_id:
        print(f"{prefix}Replicate ID: {process.replicate_id}")
    
    # Time axis
    if process.time is not None:
        print(f"\n{prefix}Time:")
        print(f"{prefix}  Range: {process.time.start:.2f} to {process.time.end:.2f} {process.time.unit}")
        print(f"{prefix}  Reference: {process.time.time_reference}")
    
    # Dynamic states
    if process.dynamic_states:
        print(f"\n{prefix}Dynamic States: ({len(process.dynamic_states)} total)")
        for name, ts in process.dynamic_states.items():
            _print_timeseries_info(ts, prefix + "  ", show_values)
    
    # Dynamic controls
    if process.dynamic_controls:
        print(f"\n{prefix}Dynamic Controls: ({len(process.dynamic_controls)} total)")
        for name, ts in process.dynamic_controls.items():
            _print_timeseries_info(ts, prefix + "  ", show_values)
    
    # Static controls
    if process.static_controls:
        print(f"\n{prefix}Static Controls: ({len(process.static_controls)} total)")
        for name, var in process.static_controls.items():
            print(f"{prefix}  {name}: {var.value} {var.unit}")
    
    # Volume information
    if process.volume is not None:
        print(f"\n{prefix}Volume:")
        print(f"{prefix}  Initial: {process.volume.initial_volume} {process.volume.volume_unit}")
        if process.volume.volume_changes:
            print(f"{prefix}  Volume Changes: ({len(process.volume.volume_changes)} total)")
            for name, change in process.volume.volume_changes.items():
                _print_volume_change_info(change, prefix + "    ", show_values)
    
    # Feeds
    if process.feeds:
        print(f"\n{prefix}Feeds: ({len(process.feeds)} total)")
        for name, feed in process.feeds.items():
            print(f"{prefix}  {name}:")
            print(f"{prefix}    Density: {feed.density} {feed.density_unit}")
            if feed.components:
                print(f"{prefix}    Components: ({len(feed.components)} total)")
                for comp_name, comp in feed.components.items():
                    print(f"{prefix}      {comp_name}: {comp.concentration} {comp.unit}")
    
    # Static parameters
    if process.static_parameters:
        print(f"\n{prefix}Static Parameters: ({len(process.static_parameters)} total)")
        for name, var in process.static_parameters.items():
            print(f"{prefix}  {name}: {var.value} {var.unit}")
    
    # Event times
    if process.event_times is not None and len(process.event_times) > 0:
        print(f"\n{prefix}Event Times: ({len(process.event_times)} total)")
        if len(process.event_times) <= 10:
            print(f"{prefix}  {list(process.event_times)}")
        else:
            print(f"{prefix}  First 5: {list(process.event_times[:5])}")
            print(f"{prefix}  Last 5: {list(process.event_times[-5:])}")
    
    # Reactor properties
    if process.reactor is not None:
        print(f"\n{prefix}Reactor:")
        print(f"{prefix}  Working Volume: {process.reactor.working_volume} {process.reactor.volume_unit}")
        if process.reactor.density is not None:
            print(f"{prefix}  Density: {process.reactor.density}")
    
    if indent == 0:
        print("=" * 80)


def _print_timeseries_info(ts: TimeSeries, prefix: str, show_values: bool = False) -> None:
    """Helper function to print TimeSeries information."""
    print(f"{prefix}{ts.name} (canonical: {ts.canonical_name or 'N/A'})")
    print(f"{prefix}  Unit: {ts.unit}")
    print(f"{prefix}  Role: {ts.role}")
    
    if ts.raw is not None:
        n_points = len(ts.raw.timepoints)
        print(f"{prefix}  Raw Data: {n_points} points")
        if n_points > 0:
            t_range = (float(ts.raw.timepoints[0]), float(ts.raw.timepoints[-1]))
            v_range = (float(jnp.min(ts.raw.values)), float(jnp.max(ts.raw.values)))
            print(f"{prefix}    Time range: {t_range[0]:.2f} to {t_range[1]:.2f}")
            print(f"{prefix}    Value range: {v_range[0]:.4f} to {v_range[1]:.4f}")
            
            if show_values and n_points <= 5:
                print(f"{prefix}    Values: {list(ts.raw.values)}")
            elif show_values:
                print(f"{prefix}    First 3: {list(ts.raw.values[:3])}")
        
        if ts.raw.measurement_std is not None:
            print(f"{prefix}    Measurement std: provided")
    
    if ts.spline is not None:
        print(f"{prefix}  Spline: {ts.spline.type}")
        print(f"{prefix}    Breakpoints: {len(ts.spline.breakpoints)}")
        print(f"{prefix}    Discontinuous: {ts.spline.discontinuous}")
        if ts.spline.fit_residual_std is not None:
            print(f"{prefix}    Fit residual std: {ts.spline.fit_residual_std:.4f}")


def _print_volume_change_info(change: VolumeChange, prefix: str, show_values: bool = False) -> None:
    """Helper function to print VolumeChange information."""
    print(f"{prefix}{change.name}:")
    print(f"{prefix}  Type: {'Controlled' if change.controlled else 'Modeled'}, "
          f"{'Continuous' if change.continuous else 'Discrete'}")
    print(f"{prefix}  Unit: {change.unit}")
    
    if change.feed_medium:
        print(f"{prefix}  Feed Medium: {change.feed_medium}")
    
    if change.continuous and change.timeseries is not None:
        print(f"{prefix}  TimeSeries: {change.timeseries.name}")
        if change.timeseries.raw is not None:
            n_points = len(change.timeseries.raw.timepoints)
            print(f"{prefix}    Points: {n_points}")
            if n_points > 0:
                v_range = (float(jnp.min(change.timeseries.raw.values)), 
                          float(jnp.max(change.timeseries.raw.values)))
                print(f"{prefix}    Value range: {v_range[0]:.4f} to {v_range[1]:.4f} {change.unit}")
    
    if not change.continuous and change.timepoints is not None:
        print(f"{prefix}  Discrete Events: {len(change.timepoints)}")
        if change.values is not None:
            total_vol = float(jnp.sum(change.values))
            print(f"{prefix}    Total Volume: {total_vol:.2f} {change.unit}")
