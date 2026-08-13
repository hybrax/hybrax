"""
Spline-fitting and discrete-event infrastructure for bioprocess data.

This module provides:
- Discrete-event detection from non-continuous VolumeChanges
- Segment boundary construction and TimeSeries splitting around events
- Cubic spline fitting (interpolating and smoothing), stored as TimeSeries
  spline state (breaks/coeffs, power-basis)
- Small standalone spline builders (constant series, cubic PPoly from arrays)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from scipy import interpolate

from .dataclasses import BioProcess, DiscreteEvents, TimeSeries
from .time_series import PPoly, spline_ops


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONSTANT_SPLINE_DT = 1e-6
_MIN_REACTOR_VOLUME = 1e-10
_MIN_SMOOTHING_BSPLINE_SAMPLES = 4

DEFAULT_MAX_SEGMENTS = 16


# ---------------------------------------------------------------------------
# Discrete-event detection and segmentation
# ---------------------------------------------------------------------------


def _has_spline_state(ts: TimeSeries) -> bool:
    """Return True when a TimeSeries contains spline coefficients."""
    return (
        getattr(ts, "breaks", None) is not None
        and getattr(ts, "coeffs", None) is not None
    )


def detect_discrete_state_events(process: BioProcess) -> DiscreteEvents:
    """Detect discrete event times from non-continuous VolumeChanges.

    Parameters
    ----------
    process:
        A BioProcess whose ``volume.volume_changes`` may contain discrete
        (``is_continuous=False``) entries.

    Returns
    -------
    DiscreteEvents
        Sorted, unique event times and optional labels.
    """
    times: list = []
    labels: list = []
    for vc_name, vc in process.volume.volume_changes.items():
        if not vc.is_continuous:
            tp = jnp.asarray(vc.values.times).tolist()
            times.extend(tp)
            labels.extend([vc_name] * len(tp))

    if not times:
        return DiscreteEvents(times=jnp.zeros(0))

    order = jnp.argsort(jnp.array(times))
    sorted_times = jnp.array(times)[order]
    sorted_labels = [labels[int(i)] for i in order]
    unique_times, unique_idx = jnp.unique(sorted_times, return_index=True)
    unique_labels = [sorted_labels[int(i)] for i in unique_idx]

    return DiscreteEvents(
        times=jnp.array(unique_times),
        labels=unique_labels,
    )


def make_segment_boundaries(
    t_min: float, t_max: float, event_times: jnp.ndarray
) -> jnp.ndarray:
    """Return segment boundaries ``[t_min, ...events..., t_max]``.

    Only events strictly inside ``(t_min, t_max)`` are included.
    The result is a strictly-increasing array.
    """
    ev = jnp.asarray(event_times, dtype=float)
    interior = ev[(ev > t_min) & (ev < t_max)]
    boundaries = jnp.unique(
        jnp.concatenate([jnp.array([t_min]), interior, jnp.array([t_max])])
    )
    return boundaries


def split_timeseries(ts: TimeSeries, boundaries: jnp.ndarray) -> List[TimeSeries]:
    """Split *ts* into segments defined by *boundaries*.

    Points that fall exactly on a boundary belong to both adjacent segments
    (duplicated at split point) to ensure each segment covers its endpoints.
    """
    t = jnp.asarray(ts.times)
    v = jnp.asarray(ts.values)
    order = jnp.argsort(t)
    t = t[order]
    v = v[order]

    segments: List[TimeSeries] = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (t >= lo) & (t <= hi)
        seg_t = t[mask]
        seg_v = v[mask]
        _, idx = jnp.unique(seg_t, return_index=True)
        seg_t = seg_t[idx]
        seg_v = seg_v[idx]
        segments.append(
            TimeSeries(
                times=jnp.array(seg_t),
                values=jnp.array(seg_v),
            )
        )
    return segments


# ---------------------------------------------------------------------------
# Spline fitting
# ---------------------------------------------------------------------------


def _prepare_knots(t: jnp.ndarray, y: jnp.ndarray):
    """Sort and deduplicate knots. Returns (t, y) with at least 2 points."""
    t = jnp.asarray(t, dtype=float)
    y = jnp.asarray(y, dtype=float)
    order = jnp.argsort(t)
    t, y = t[order], y[order]
    _, idx = jnp.unique(t, return_index=True)
    t, y = t[idx], y[idx]
    if len(t) < 2:
        t = jnp.array([t[0], t[0] + _CONSTANT_SPLINE_DT])
        y = jnp.array([y[0], y[0]])
    return jnp.asarray(t), jnp.asarray(y)


def _fit_smoothing_segment(
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    s: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Fit a SciPy smoothing B-spline and return power-basis arrays."""

    if len(x) < _MIN_SMOOTHING_BSPLINE_SAMPLES:
        return _fit_interp_segment(x, y)

    degree = 3
    bspline = interpolate.make_splrep(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        s=s,
        k=degree,
    )
    poly = PPoly.from_scipy_ppoly(interpolate.PPoly.from_spline(bspline))
    return jnp.asarray(poly.breaks, dtype=float), jnp.asarray(poly.coeffs, dtype=float)


def _fit_interp_segment(
    x: jnp.ndarray, y: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Fit an interpolating natural cubic segment in power-basis form."""
    x_unique, y_unique = _prepare_knots(x, y)
    ppoly = interpolate.CubicSpline(
        np.asarray(x_unique, dtype=np.float64),
        np.asarray(y_unique, dtype=np.float64),
        bc_type="natural",
        extrapolate=True,
    )
    poly = PPoly.from_scipy_ppoly(ppoly)
    return jnp.asarray(poly.breaks, dtype=float), jnp.asarray(poly.coeffs, dtype=float)


def _combine_segment_splines(
    segment_splines: List[Tuple[jnp.ndarray, jnp.ndarray]],
    boundaries: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Flatten per-segment power-basis splines into TimeSeries arrays."""
    all_breaks: List[np.ndarray] = []
    all_coeffs: List[np.ndarray] = []
    segment_starts: List[int] = []
    piece_cursor = 0

    boundary_arr = jnp.asarray(boundaries, dtype=float)

    for seg_idx, (seg_breaks, seg_coeffs) in enumerate(segment_splines):
        if seg_coeffs.shape[0] == 0:
            continue

        seg_start = float(boundary_arr[seg_idx])
        seg_end = float(boundary_arr[seg_idx + 1])
        interior_breaks = [
            float(b)
            for b in np.asarray(seg_breaks[1:-1], dtype=np.float64)
            if seg_start < float(b) < seg_end
        ]
        target_breaks = jnp.asarray(
            [seg_start, *interior_breaks, seg_end],
            dtype=float,
        )
        target_coeffs = spline_ops.rebase_to_breaks(
            jnp.asarray(seg_breaks, dtype=float),
            jnp.asarray(seg_coeffs, dtype=float),
            target_breaks,
        )

        segment_starts.append(piece_cursor)
        piece_cursor += int(target_coeffs.shape[0])
        if not all_breaks:
            all_breaks.append(np.asarray(target_breaks, dtype=np.float64))
        else:
            if np.isclose(all_breaks[-1][-1], float(target_breaks[0])):
                all_breaks.append(np.asarray(target_breaks[1:], dtype=np.float64))
            else:
                all_breaks.append(np.asarray(target_breaks, dtype=np.float64))
        all_coeffs.append(np.asarray(target_coeffs, dtype=np.float64))

    if not all_coeffs:
        raise ValueError("No valid spline pieces found while fitting TimeSeries")

    return (
        jnp.asarray(np.concatenate(all_breaks, axis=0), dtype=float),
        jnp.asarray(np.concatenate(all_coeffs, axis=0), dtype=float),
        jnp.asarray(segment_starts, dtype=jnp.int32),
    )


def fit_timeseries_spline(
    ts: TimeSeries,
    *,
    boundaries: Optional[jnp.ndarray] = None,
    smoothing_s: float = 0.0,
) -> TimeSeries:
    """Fit segmented spline state onto a TimeSeries.

    Segments with at least four points use SciPy's cubic smoothing B-spline
    path. ``smoothing_s=0`` makes that path exact/interpolating. Shorter
    segments fall back to an interpolating natural CubicSpline because cubic
    smoothing B-splines require at least four samples.
    """
    t_arr = jnp.asarray(ts.times)
    v_arr = jnp.asarray(ts.values)

    order = jnp.argsort(t_arr)
    t_arr = t_arr[order]
    v_arr = v_arr[order]
    _, uidx = jnp.unique(t_arr, return_index=True)
    t_arr = t_arr[uidx]
    v_arr = v_arr[uidx]

    if boundaries is None:
        boundaries = jnp.array([t_arr[0], t_arr[-1]])

    segments = split_timeseries(
        TimeSeries(times=jnp.array(t_arr), values=jnp.array(v_arr)),
        boundaries,
    )
    actual_n_segments = len(segments)

    segment_splines: List[Tuple[jnp.ndarray, jnp.ndarray]] = []
    segment_fit_strategies: list[str] = []

    for seg in segments:
        seg_t = jnp.asarray(seg.times)
        seg_v = jnp.asarray(seg.values)
        n_pts = len(seg_t)

        if n_pts == 0:
            raise ValueError("Spline segment has no samples; adjust boundaries.")
        if n_pts == 1:
            seg_t = jnp.array([seg_t[0], seg_t[0] + _CONSTANT_SPLINE_DT])
            seg_v = jnp.array([seg_v[0], seg_v[0]])
            segment_splines.append(_fit_interp_segment(seg_t, seg_v))
            segment_fit_strategies.append("cubic_interp")
            continue

        if n_pts >= _MIN_SMOOTHING_BSPLINE_SAMPLES:
            segment_spline = _fit_smoothing_segment(seg_t, seg_v, s=smoothing_s)
            segment_fit_strategies.append("smoothing_bspline")
        else:
            segment_spline = _fit_interp_segment(seg_t, seg_v)
            segment_fit_strategies.append("cubic_interp")

        segment_splines.append(segment_spline)

    breaks, coeffs, segment_start_piece_idx = _combine_segment_splines(
        segment_splines, boundaries
    )

    unique_strategies = set(segment_fit_strategies)
    fit_strategy = segment_fit_strategies[0] if len(unique_strategies) == 1 else "mixed"
    used_smoothing_fit = "smoothing_bspline" in unique_strategies
    metadata = {
        "smoothing_s": float(smoothing_s),
        "actual_segments": int(actual_n_segments),
        "fit_strategy": fit_strategy,
        "fit_strategies": segment_fit_strategies,
        "segment_boundaries": np.asarray(boundaries, dtype=float).tolist(),
    }
    if used_smoothing_fit:
        metadata["smoothing_storage"] = "direct_power_basis"
        metadata["smoothing_bc_type"] = "not-a-knot"
        metadata["interp_segment_bc_type"] = "natural"

    return TimeSeries(
        times=jnp.asarray(t_arr, dtype=float),
        values=jnp.asarray(v_arr, dtype=float),
        breaks=breaks,
        coeffs=coeffs,
        segment_start_piece_idx=segment_start_piece_idx,
        continuity_side="right",
        metadata=metadata,
    )


def make_constant_spline(
    value: float,
    t_min: float,
    t_max: float,
) -> TimeSeries:
    """Create a constant spline-backed TimeSeries over [t_min, t_max]."""
    series = fit_timeseries_spline(
        TimeSeries(
            times=jnp.asarray([t_min, t_max], dtype=float),
            values=jnp.asarray([value, value], dtype=float),
        )
    )
    metadata = dict(series.metadata or {})
    metadata["is_constant"] = True
    metadata["constant_value"] = float(value)
    return TimeSeries(
        times=series.times,
        values=series.values,
        jump_times=series.jump_times,
        derived=series.derived,
        continuity_side=series.continuity_side,
        breaks=series.breaks,
        coeffs=series.coeffs,
        segment_start_piece_idx=series.segment_start_piece_idx,
        metadata=metadata,
    )


def make_cubic_ppoly(t: jnp.ndarray, y: jnp.ndarray, bc_type: str = "natural") -> PPoly:
    """Build a robust cubic PPoly from arrays. Ensures unique, sorted knots."""
    t, y = _prepare_knots(t, y)
    scipy_ppoly = interpolate.CubicSpline(
        np.asarray(t, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        bc_type=bc_type,
        extrapolate=True,
    )
    return PPoly.from_scipy_ppoly(scipy_ppoly)
