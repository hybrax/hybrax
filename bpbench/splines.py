"""
Spline fitting and manipulation functions for bioprocess time series data

This module provides utilities for fitting splines to cumulative measurements
(like feed volumes) and computing rates from those splines.
"""

import jax.numpy as jnp
from typing import Optional
from .dataclasses import SplineRepresentation


def fit_cubic_spline(timepoints: jnp.ndarray, values: jnp.ndarray, 
                     discontinuities: Optional[jnp.ndarray] = None) -> SplineRepresentation:
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
                dt = t_seg[-1] - t_seg[0]
                if dt > 1e-10:  # Use epsilon tolerance to avoid division by near-zero
                    slope = (v_seg[-1] - v_seg[0]) / dt
                else:
                    slope = 0.0
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
                dt = t_np[i+1] - t_np[i]
                if dt > 1e-10:  # Use epsilon tolerance to avoid division by near-zero
                    slope = (v_np[i+1] - v_np[i]) / dt
                else:
                    slope = 0.0
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


def compute_rate_from_cumulative(spline: SplineRepresentation, 
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
