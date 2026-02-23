"""
Validation utilities for bioprocess data
"""

import jax.numpy as jnp
from typing import Optional, Tuple
from .dataclasses import BioProcess


def validate_volume_consistency(process: BioProcess,
                                final_volume: Optional[float] = None) -> Tuple[bool, str, float]:
    """
    Validate that volume changes sum to expected final volume.
    
    This function checks whether the sum of all volume changes (feeds, sampling, etc.)
    is consistent with the expected final volume. It handles both continuous 
    (cumulative time series) and discrete volume changes.
    
    Note: as these values may be on different time-scale and this check is supposed to be
    run _before_ any modeling or spline interpolation happens, here only the last time points 
    are considered.

    Args:
        TODO: these have to be redone
        
    Returns:
        TODO: these have to be redone
    """

    volume = process.volume
    
    # Calculate total volume change and collect data for plotting
    total_change = 0.0
    messages = []
    
    for name, change in volume.volume_changes.items():
        if change.is_continuous:
            # For continuous changes, data should be cumulative
            values = change.values.values
            # Cumulative volume: final - initial
            change_vol = float(values[-1] - values[0])
            total_change += change_vol
            messages.append(f"  {name:15}: {change_vol:+8.2f} {volume.unit} (continuous)")
        elif not change.is_continuous:
            # For discrete changes, sum all values from the timeseries
            values = change.values.values
            change_vol = float(jnp.sum(values))
            total_change += change_vol
            messages.append(f"  {name:15}: {change_vol:+8.2f} {volume.unit} (discrete)")
    
    calculated_final = volume.initial_volume + total_change
    
    diff = abs(calculated_final - final_volume)
    delta = total_change
    rel_diff = diff / final_volume if final_volume > 0 else 0
    
    messages.insert(0, f"Initial volume   : {volume.initial_volume:8.2f} {volume.unit}")
    messages.append(f"Total change     : {total_change:8.2f} {volume.unit}")
    messages.append(f"Calculated final : {calculated_final:8.2f} {volume.unit}")
    messages.append(f"Expected final   : {final_volume:8.2f} {volume.unit}")
    messages.append(f"Difference       : {diff:8.2f} {volume.unit} ({rel_diff*100:.1f}%)")
    
    if rel_diff > 0.05:  # More than 5% difference
        return (False, "Volume inconsistency detected:\n" + "\n".join(messages), delta)
    else:
        return (True, "Volume balance OK:\n" + "\n".join(messages), delta)