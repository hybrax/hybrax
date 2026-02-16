"""
Validation utilities for bioprocess data
"""

import jax.numpy as jnp
from typing import Optional, Tuple, Dict, Union
from .dataclasses import Volume, TimeAxis, VolumeChange, FeedMedium, TimeSeries, BioProcess


def validate_volume_consistency(volume: Volume,
                                time_axis: Optional[TimeAxis] = None,
                                final_volume: Optional[float] = None) -> Tuple[bool, str]:
    """
    Validate that volume changes sum to expected final volume.
    
    This function checks whether the sum of all volume changes (feeds, sampling, etc.)
    is consistent with the expected final volume. It handles both continuous 
    (cumulative time series) and discrete volume changes.
    
    Args:
        volume: Volume object containing initial volume and volume changes
        time_axis: Optional TimeAxis for validation context (not currently used)
        final_volume: Optional expected final volume for comparison
        
    Returns:
        (is_valid, message): Tuple of validation result and descriptive message
    """
    
    if volume.initial_volume is None:
        return (True, "No initial volume specified, skipping validation")
    
    if not volume.volume_changes:
        return (True, "No volume changes to validate")
    
    # Calculate total volume change
    total_change = 0.0
    messages = []
    
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
                else:
                    # Rate (e.g., "L/h"): integrate using trapezoidal rule
                    dt = jnp.diff(times)
                    avg_rates = (values[:-1] + values[1:]) / 2.0
                    change_vol = float(jnp.sum(dt * avg_rates))
                
                total_change += change_vol
                messages.append(f"  {name}: +{change_vol:.2f} {volume.unit} (continuous)")
        elif not change.is_continuous and change.values is not None:
            # For discrete changes, sum all values from the timeseries
            values = change.values.values
            change_vol = float(jnp.sum(values))
            total_change += change_vol
            messages.append(f"  {name}: {change_vol:+.2f} {volume.unit} (discrete)")
    
    calculated_final = volume.initial_volume + total_change
    
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
