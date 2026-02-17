"""
Validation utilities for bioprocess data
"""

import jax.numpy as jnp
from typing import Optional, Tuple, Dict, Union
from .dataclasses import Volume, TimeAxis, VolumeChange, FeedMedium, TimeSeries, BioProcess


def validate_volume_consistency(volume: Volume,
                                time_axis: Optional[TimeAxis] = None,
                                final_volume: Optional[float] = None,
                                plot: bool = False) -> Tuple[bool, str]:
    """
    Validate that volume changes sum to expected final volume.
    
    This function checks whether the sum of all volume changes (feeds, sampling, etc.)
    is consistent with the expected final volume. It handles both continuous 
    (cumulative time series) and discrete volume changes.
    
    Args:
        volume: Volume object containing initial volume and volume changes
        time_axis: Optional TimeAxis for validation context
        final_volume: Optional expected final volume for comparison
        plot: If True, creates a plot showing measured volumes over time
        
    Returns:
        (is_valid, message): Tuple of validation result and descriptive message
    """
    
    if volume.initial_volume is None:
        return (True, "No initial volume specified, skipping validation")
    
    if not volume.volume_changes:
        return (True, "No volume changes to validate")
    
    # Calculate total volume change and collect data for plotting
    total_change = 0.0
    messages = []
    volume_data = {}  # For plotting: {name: (times, values)}
    
    for name, change in volume.volume_changes.items():
        if change.is_continuous and change.values is not None:
            # For continuous changes, data should be cumulative
            times = change.values.timepoints
            values = change.values.values
            
            if len(times) > 1:
                # Check unit to determine if cumulative or rate
                if "/" not in change.unit:
                    # Cumulative volume: final - initial
                    change_vol = float(values[-1] - values[0])
                    volume_data[name] = (times, values)
                else:
                    # Rate (e.g., "L/h"): integrate using trapezoidal rule
                    dt = jnp.diff(times)
                    avg_rates = (values[:-1] + values[1:]) / 2.0
                    change_vol = float(jnp.sum(dt * avg_rates))
                    # Calculate cumulative for plotting
                    cumulative = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dt * avg_rates)])
                    volume_data[name] = (times, cumulative)
                
                total_change += change_vol
                messages.append(f"  {name}: +{change_vol:.2f} {volume.unit} (continuous)")
        elif not change.is_continuous and change.values is not None:
            # For discrete changes, sum all values from the timeseries
            times = change.values.timepoints
            values = change.values.values
            change_vol = float(jnp.sum(values))
            total_change += change_vol
            
            # For plotting discrete changes, show cumulative sum
            cumulative = jnp.cumsum(values)
            volume_data[name] = (times, cumulative)
            messages.append(f"  {name}: {change_vol:+.2f} {volume.unit} (discrete)")
    
    calculated_final = volume.initial_volume + total_change
    
    # Create plot if requested
    if plot and volume_data:
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot 1: Individual volume changes
            for name, (times, values) in volume_data.items():
                ax1.plot(times, values, marker='o', label=name, linewidth=2)
            
            ax1.set_xlabel('Time (h)' if time_axis is None else f'Time ({time_axis.unit})')
            ax1.set_ylabel(f'Volume Change ({volume.unit})')
            ax1.set_title('Individual Volume Changes Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Total volume over time
            if time_axis and volume_data:
                # Create a dense time grid for interpolation
                all_times = []
                for times, _ in volume_data.values():
                    all_times.extend([float(t) for t in times])
                all_times = sorted(set(all_times))
                
                # Calculate total volume at each time point
                total_volumes = []
                for t in all_times:
                    vol = volume.initial_volume
                    for name, (times, values) in volume_data.items():
                        # Find the value at or before time t
                        idx = jnp.searchsorted(times, t, side='right') - 1
                        if idx >= 0 and idx < len(values):
                            vol += float(values[idx])
                    total_volumes.append(vol)
                
                ax2.plot(all_times, total_volumes, 'b-', linewidth=2, label='Total Volume')
                ax2.axhline(y=volume.initial_volume, color='g', linestyle='--', 
                           label=f'Initial: {volume.initial_volume:.2f} {volume.unit}')
                if final_volume:
                    ax2.axhline(y=final_volume, color='r', linestyle='--', 
                               label=f'Expected Final: {final_volume:.2f} {volume.unit}')
                ax2.axhline(y=calculated_final, color='orange', linestyle='--', 
                           label=f'Calculated Final: {calculated_final:.2f} {volume.unit}')
            else:
                # Just show initial and final
                ax2.axhline(y=volume.initial_volume, color='g', linestyle='-', 
                           linewidth=2, label=f'Initial: {volume.initial_volume:.2f} {volume.unit}')
                if final_volume:
                    ax2.axhline(y=final_volume, color='r', linestyle='--', 
                               linewidth=2, label=f'Expected Final: {final_volume:.2f} {volume.unit}')
                ax2.axhline(y=calculated_final, color='orange', linestyle='--', 
                           linewidth=2, label=f'Calculated Final: {calculated_final:.2f} {volume.unit}')
            
            ax2.set_xlabel('Time (h)' if time_axis is None else f'Time ({time_axis.unit})')
            ax2.set_ylabel(f'Total Volume ({volume.unit})')
            ax2.set_title('Total Volume Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            messages.append("Note: matplotlib not available for plotting")
    
    if final_volume is not None:
        diff = abs(calculated_final - final_volume)
        rel_diff = diff / final_volume if final_volume > 0 else 0
        
        messages.insert(0, f"Initial volume: {volume.initial_volume:.2f} {volume.unit}")
        messages.append(f"Total change: {total_change:.2f} {volume.unit}")
        messages.append(f"Calculated final: {calculated_final:.2f} {volume.unit}")
        messages.append(f"Expected final: {final_volume:.2f} {volume.unit}")
        messages.append(f"Difference: {diff:.2f} {volume.unit} ({rel_diff*100:.1f}%)")
        
        if rel_diff > 0.05:  # More than 5% difference
            return (False, "Volume inconsistency detected:\n" + "\n".join(messages))
        else:
            return (True, "Volume balance OK:\n" + "\n".join(messages))
    else:
        messages.insert(0, f"Initial volume: {volume.initial_volume:.2f} {volume.unit}")
        messages.append(f"Calculated final: {calculated_final:.2f} {volume.unit}")
        return (True, "Volume changes calculated:\n" + "\n".join(messages))


def validate_feed_components(volume: Volume,
                            process_feeds: Dict[str, FeedMedium], 
                            dynamic_variables: Dict[str, TimeSeries]) -> Tuple[bool, str]:
    """
    DEPRECATED: Validate that feed compositions are properly defined for volume changes.
    
    NOTE: This function is deprecated as FeedMedium objects are now stored inline
    in VolumeChange.feed_medium, not in a separate Process.feeds dictionary.
    Kept for backward compatibility but may be removed in future versions.
    
    For each VolumeChange with a feed_medium reference:
    - The referenced feed must exist in process_feeds
    - Warning if feed components don't cover all dynamic variables
    
    Args:
        volume: Volume object containing volume changes
        process_feeds: Dictionary of FeedMedium objects (deprecated - no longer part of BioProcess)
        dynamic_variables: Dictionary of TimeSeries from Process.dynamic_variables
        
    Returns:
        (is_valid, message): Tuple of validation result and descriptive message
    """
    messages = []
    all_valid = True
    
    for vc_name, vc in volume.volume_changes.items():
        # Check if this volume change has a feed
        feed = None
        if vc.feed_medium is not None:
            if isinstance(vc.feed_medium, FeedMedium):
                # Feed is stored inline - use it directly
                feed = vc.feed_medium
            elif vc.feed_medium in process_feeds:
                # Feed is referenced by name in the process_feeds dict
                feed = process_feeds[vc.feed_medium]
            else:
                messages.append(f"ERROR: VolumeChange '{vc_name}' references feed '{vc.feed_medium}' "
                              f"which is not defined in Process.feeds")
                all_valid = False
        
        # If there's a feed, validate component coverage
        if feed is not None:
            missing_components = []
            for var_name in dynamic_variables.keys():
                if var_name not in feed.components:
                    missing_components.append(var_name)
            
            if missing_components:
                messages.append(f"WARNING: VolumeChange '{vc_name}' feed '{feed.name}' "
                              f"is missing concentrations for dynamic variables: {missing_components}")
    
    if not messages:
        return (True, "All feed components properly defined")
    
    return (all_valid, "\n".join(messages))
