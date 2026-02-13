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
            print(f"{prefix}  {process.event_times}")
        else:
            print(f"{prefix}  First 5: {process.event_times[:5]}")
            print(f"{prefix}  Last 5: {process.event_times[-5:]}")
    
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
                print(f"{prefix}    Values: {ts.raw.values}")
            elif show_values:
                print(f"{prefix}    First 3: {ts.raw.values[:3]}")
        
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


def fit_cubic_spline(timepoints: jnp.ndarray, values: jnp.ndarray, 
                     discontinuities: jnp.ndarray = None) -> 'SplineRepresentation':
    """
    Fit a cubic spline to time series data, handling discontinuities.
    
    This function fits piecewise cubic splines between discontinuity points.
    For cumulative data (like cumulative volumes), this preserves monotonicity
    and allows computation of rates via differentiation.
    
    Args:
        timepoints: Time points of measurements (shape: N,)
        values: Measured values at timepoints (shape: N,)
        discontinuities: Optional array of time points where discontinuities occur
        
    Returns:
        SplineRepresentation object with fitted spline coefficients
        
    Example:
        >>> times = jnp.array([0., 1., 2., 3., 4.])
        >>> cumulative_vol = jnp.array([0., 0.1, 0.3, 0.6, 1.0])
        >>> spline = fit_cubic_spline(times, cumulative_vol)
    """
    from .dataclasses import SplineRepresentation
    from scipy import interpolate
    import numpy as np
    
    # Convert JAX arrays to numpy for scipy
    t_np = np.array(timepoints)
    v_np = np.array(values)
    
    if discontinuities is not None and len(discontinuities) > 0:
        # Split data at discontinuities and fit separate splines
        disc_np = np.array(discontinuities)
        
        # Find segments between discontinuities
        segments = []
        breakpoints = [t_np[0]]
        
        for disc_time in disc_np:
            # Find indices before and after discontinuity
            mask_before = t_np <= disc_time
            if np.any(mask_before):
                idx = np.where(mask_before)[0][-1]
                if idx < len(t_np) - 1:
                    breakpoints.append(disc_time)
        
        breakpoints.append(t_np[-1])
        breakpoints = np.unique(np.array(breakpoints))
        
        # Use linear interpolation for simplicity when discontinuities present
        # (cubic splines across discontinuities would require more sophisticated handling)
        coeffs = []
        for i in range(len(breakpoints) - 1):
            mask = (t_np >= breakpoints[i]) & (t_np <= breakpoints[i+1])
            t_seg = t_np[mask]
            v_seg = v_np[mask]
            
            if len(t_seg) >= 2:
                # Linear fit for this segment
                slope = (v_seg[-1] - v_seg[0]) / (t_seg[-1] - t_seg[0]) if t_seg[-1] > t_seg[0] else 0
                intercept = v_seg[0] - slope * t_seg[0]
                coeffs.append([intercept, slope])
            else:
                coeffs.append([v_seg[0], 0.0])
        
        coeffs_array = jnp.array(coeffs)
        fit_type = "linear"
        is_discontinuous = True
        
        # Evaluate for residuals
        predicted = np.zeros_like(v_np)
        for i, t in enumerate(t_np):
            # Find segment
            seg_idx = np.searchsorted(np.array(breakpoints[:-1]), t, side='right') - 1
            seg_idx = max(0, min(seg_idx, len(coeffs_array) - 1))
            
            if coeffs_array.shape[1] == 2:
                predicted[i] = coeffs_array[seg_idx, 0] + coeffs_array[seg_idx, 1] * t
            else:
                predicted[i] = v_np[i]
        
    else:
        # Fit cubic spline to entire dataset
        if len(t_np) >= 4:
            # Use cubic spline
            cs = interpolate.CubicSpline(t_np, v_np, bc_type='natural')
            
            # Extract breakpoints and coefficients
            breakpoints = jnp.array(cs.x)
            # CubicSpline stores coefficients in shape (n_segments, 4) for [c3, c2, c1, c0]
            coeffs_array = jnp.array(cs.c.T)  # Transpose to (n_segments, 4)
            fit_type = "cubic_hermite"
            is_discontinuous = False
            
            # Calculate fit residual using the fitted spline
            predicted = cs(t_np)
        else:
            # Fall back to linear interpolation for few points
            breakpoints = jnp.array(t_np)
            coeffs = []
            for i in range(len(t_np) - 1):
                slope = (v_np[i+1] - v_np[i]) / (t_np[i+1] - t_np[i]) if t_np[i+1] > t_np[i] else 0
                intercept = v_np[i] - slope * t_np[i]
                coeffs.append([intercept, slope])
            coeffs_array = jnp.array(coeffs)
            fit_type = "linear"
            is_discontinuous = False
            
            # Linear evaluation
            predicted = np.zeros_like(v_np)
            for i, t in enumerate(t_np):
                # Find segment
                seg_idx = np.searchsorted(np.array(breakpoints[:-1]), t, side='right') - 1
                seg_idx = max(0, min(seg_idx, len(coeffs_array) - 1))
                
                if coeffs_array.shape[1] == 2:
                    predicted[i] = coeffs_array[seg_idx, 0] + coeffs_array[seg_idx, 1] * t
                else:
                    predicted[i] = v_np[i]  # Fallback
    
    # Calculate fit residual
    residuals = v_np - predicted
    fit_residual_std = float(np.std(residuals))
    
    return SplineRepresentation(
        type=fit_type,
        breakpoints=breakpoints,
        coefficients=coeffs_array,
        discontinuous=is_discontinuous,
        fit_residual_std=fit_residual_std,
        notes=f"Fitted with {len(breakpoints)} breakpoints"
    )


def compute_rate_from_cumulative(spline: 'SplineRepresentation', 
                                 eval_times: jnp.ndarray) -> jnp.ndarray:
    """
    Compute rate (derivative) from cumulative volume spline.
    
    For cumulative data stored as splines, this computes the instantaneous
    rate at specified time points.
    
    Args:
        spline: SplineRepresentation of cumulative data
        eval_times: Times at which to evaluate the rate
        
    Returns:
        Array of rate values at eval_times
        
    Example:
        >>> # spline represents cumulative volume
        >>> times = jnp.array([0., 1., 2., 3.])
        >>> rates = compute_rate_from_cumulative(spline, times)
        >>> # rates[i] is the instantaneous feed rate at times[i]
    """
    import numpy as np
    from scipy import interpolate
    
    eval_times_np = np.array(eval_times)
    
    if spline.type == "cubic_hermite":
        # For cubic splines, compute analytical derivative
        # Coefficients are [c3, c2, c1, c0] for c3*t^3 + c2*t^2 + c1*t + c0
        # Derivative is 3*c3*t^2 + 2*c2*t + c1
        
        rates = np.zeros_like(eval_times_np)
        breakpoints_np = np.array(spline.breakpoints)
        coeffs_np = np.array(spline.coefficients)
        
        for i, t in enumerate(eval_times_np):
            # Find which segment this time belongs to
            seg_idx = np.searchsorted(breakpoints_np[:-1], t, side='right') - 1
            seg_idx = max(0, min(seg_idx, len(coeffs_np) - 1))
            
            # Get segment start time
            t0 = breakpoints_np[seg_idx]
            dt = t - t0
            
            # Compute derivative: d/dt[c3*dt^3 + c2*dt^2 + c1*dt + c0]
            if coeffs_np.shape[1] >= 4:
                c3, c2, c1, c0 = coeffs_np[seg_idx]
                rates[i] = 3*c3*dt**2 + 2*c2*dt + c1
            else:
                # Fallback to numerical differentiation
                rates[i] = 0.0
                
    elif spline.type == "linear":
        # For linear splines, derivative is simply the slope
        rates = np.zeros_like(eval_times_np)
        breakpoints_np = np.array(spline.breakpoints)
        coeffs_np = np.array(spline.coefficients)
        
        for i, t in enumerate(eval_times_np):
            seg_idx = np.searchsorted(breakpoints_np[:-1], t, side='right') - 1
            seg_idx = max(0, min(seg_idx, len(coeffs_np) - 1))
            
            if coeffs_np.shape[1] >= 2:
                # Linear: intercept + slope*t, so derivative is slope
                rates[i] = coeffs_np[seg_idx, 1]
            else:
                rates[i] = 0.0
    else:
        # Unknown spline type, return zeros
        rates = np.zeros_like(eval_times_np)
    
    return jnp.array(rates)
