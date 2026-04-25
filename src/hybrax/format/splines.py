"""
Pseudobatch normalization utilities and spline storage for bioprocess data.

This module provides:
- Pseudobatch transform computation (c*, ADF, feed correction)
- Conversion of pseudobatch results to TimeSeries carriers for minimal storage
- Reconstruction of backtransformed concentrations from stored TimeSeries
- Core spline infrastructure (fitting, evaluation, segmentation)

Design goals:
- Always compute pseudobatch transform (c*) and correction terms from physical
  event semantics.
- Avoid dense-grid pseudo-events from interpolation artifacts.
- Interpolate:
    - c*: cubic/PCHIP (smooth pseudobatch space)
    - ADF: piecewise-linear baseline + instantaneous bolus jumps
    - feed_correction (discrete mode): linear baseline + instantaneous jumps
- Keep dense grid for robust event bookkeeping and plotting only.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from scipy import interpolate
import pseudobatch
import pseudobatch.data_correction

from .dataclasses import (
    BioProcess,
    DiscreteEvents,
    FeedVolumeChange,
    SampleVolumeChange,
    StaticVariable,
    TimeSeries,
)
from .time_series import spline_ops


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Small epsilon used when inserting pre-event points for discrete bolus events.
# Must be large enough to survive float32 quantization but tiny relative to true
# sampling intervals.
_EPS = 1e-4
_JUMP_DT_EPS_FACTOR = 20.0
_JUMP_DT_MEDIAN_FRACTION = 0.1
_CONSTANT_SPLINE_DT = 1e-6
_IS_CONSTANT_ABS_TOL = 1e-8
_IS_CONSTANT_REL_TOL = 1e-4

DEFAULT_MAX_SEGMENTS = 16
DEFAULT_MAX_CTRL_POINTS = 128
SMOOTHING_THRESHOLD = 100  # > 100 points -> smoothing spline


# ===========================================================================
# Core spline infrastructure (fitting, evaluation, segmentation)
# ===========================================================================


def _has_spline_state(ts: TimeSeries) -> bool:
    """Return True when a TimeSeries contains spline coefficients."""
    return (
        getattr(ts, "breaks", None) is not None
        and getattr(ts, "coeffs", None) is not None
    )


def _has_discrete_samples(ts: TimeSeries) -> bool:
    """Return True when a TimeSeries contains sample grids."""
    return (
        getattr(ts, "times", None) is not None
        and getattr(ts, "values", None) is not None
    )


def _series_reference_times(ts: TimeSeries) -> jnp.ndarray:
    """Return a representative grid for a TimeSeries."""
    if _has_discrete_samples(ts):
        return jnp.asarray(ts.times, dtype=float)
    if _has_spline_state(ts):
        return jnp.asarray(ts.breaks, dtype=float)
    raise ValueError("TimeSeries must provide discrete samples or spline state")


def _evaluate_timeseries_on_grid(ts: TimeSeries, grid: jnp.ndarray) -> jnp.ndarray:
    """Evaluate a TimeSeries on a target grid with spline-first semantics."""
    t_grid = jnp.asarray(grid, dtype=float)
    if _has_spline_state(ts) and hasattr(ts, "evaluate_many"):
        return jnp.asarray(ts.evaluate_many(t_grid), dtype=float)
    if _has_discrete_samples(ts):
        base_t = jnp.asarray(ts.times, dtype=float)
        base_v = jnp.asarray(ts.values, dtype=float)
        return jnp.interp(t_grid, base_t, base_v, left=base_v[0], right=base_v[-1])
    raise ValueError(
        "TimeSeries cannot be evaluated without spline or discrete samples"
    )


def evaluate_left_continuous_step(
    t: jnp.ndarray,
    step_times: jnp.ndarray,
    step_values: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate left-continuous piecewise-constant value at time(s) *t*.

    Supported encodings:
    - Canonical interval encoding: ``len(values) == len(times) + 1``.
    - Legacy knot encoding: ``len(values) == len(times)``.
    """
    t = jnp.asarray(t, dtype=float)
    step_times = jnp.asarray(step_times, dtype=float)
    step_values = jnp.asarray(step_values, dtype=float)
    n_times = int(step_times.shape[0])
    n_values = int(step_values.shape[0])

    if n_values == 0:
        raise ValueError("step_values must contain at least one element")

    if n_values == n_times + 1:
        idx = jnp.searchsorted(step_times, t, side="left")
        idx = jnp.clip(idx, 0, n_values - 1)
        return step_values[idx]

    if n_values == n_times:
        idx = jnp.searchsorted(step_times, t, side="left") - 1
        idx = jnp.clip(idx, 0, n_values - 1)
        return step_values[idx]

    raise ValueError(
        "Invalid step metadata shape: expected len(values)==len(times) "
        "or len(values)==len(times)+1."
    )


def _canonicalize_left_continuous_step_metadata(
    step_times: jnp.ndarray, step_values: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return canonical left-continuous step metadata.

    Canonical form stores interval values, so ``len(values) == len(times) + 1``.
    Legacy same-length metadata is compacted by change-point detection so that
    transitions happen at the original knot time, not one knot later.
    """
    step_times = jnp.asarray(step_times, dtype=float)
    step_values = jnp.asarray(step_values, dtype=float)
    n_times = int(step_times.shape[0])
    n_values = int(step_values.shape[0])

    if n_values == n_times + 1:
        return step_times, step_values

    if n_values == n_times:
        return _canonicalize_adf_step_table(step_times, step_values)

    raise ValueError(
        "Invalid step metadata shape: expected len(values)==len(times) "
        "or len(values)==len(times)+1."
    )


def _canonicalize_adf_step_table(
    dense_times: jnp.ndarray, dense_adf: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build canonical ADF step metadata from dense ADF samples."""
    dense_times = jnp.asarray(dense_times, dtype=float)
    dense_adf = jnp.asarray(dense_adf, dtype=float)
    n_dense = int(dense_adf.shape[0])
    if n_dense == 0:
        raise ValueError("dense_adf must contain at least one sample")
    if n_dense == 1:
        return jnp.zeros(0, dtype=float), dense_adf

    change_idx = [
        i
        for i in range(n_dense - 1)
        if abs(float(dense_adf[i + 1] - dense_adf[i])) > 1e-12
    ]
    if not change_idx:
        return jnp.zeros(0, dtype=float), dense_adf[:1]

    idx = jnp.asarray(change_idx, dtype=int)
    step_times = dense_times[idx]
    post_values = dense_adf[idx + 1]
    step_values = jnp.concatenate([dense_adf[:1], post_values])
    return step_times, step_values


def _jump_increments_to_step_values(jump_values: jnp.ndarray) -> jnp.ndarray:
    """Convert jump increments to canonical left-continuous step values."""
    jump_values = jnp.asarray(jump_values, dtype=float)
    if jump_values.size == 0:
        return jnp.asarray([0.0], dtype=float)
    return jnp.concatenate([jnp.asarray([0.0], dtype=float), jnp.cumsum(jump_values)])


def _ppoly_to_power_basis(ppoly) -> Tuple[np.ndarray, np.ndarray]:
    """Convert SciPy PPoly coefficients to local cubic power-basis arrays."""
    x = np.asarray(ppoly.x, dtype=np.float64)
    c = np.asarray(ppoly.c, dtype=np.float64)
    if c.shape[0] < 1 or c.shape[0] > 4:
        raise ValueError("unsupported polynomial degree")
    if c.shape[0] < 4:
        full = np.zeros((4, c.shape[1]), dtype=np.float64)
        full[4 - c.shape[0] :, :] = c
        c = full

    widths = np.diff(x)
    keep = widths > 0
    breaks = np.concatenate([x[:-1][keep], x[-1:]], axis=0)
    coeffs = np.stack([c[3], c[2], c[1], c[0]], axis=1)[keep]
    return breaks, coeffs


def _linear_plus_step_spline_state(
    base_times: jnp.ndarray,
    base_values: jnp.ndarray,
    jump_times: jnp.ndarray,
    jump_values: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build spline state for a linear baseline plus left-continuous jumps."""
    base_times, base_values = _prepare_knots(base_times, base_values)
    jump_times = jnp.asarray(jump_times, dtype=float)
    jump_values = jnp.asarray(jump_values, dtype=float)

    breaks = jnp.unique(jnp.concatenate([base_times, jump_times]))
    if breaks.size == 1:
        breaks = jnp.asarray(
            [float(breaks[0]), float(breaks[0]) + _CONSTANT_SPLINE_DT],
            dtype=float,
        )

    base_at_breaks = jnp.interp(breaks, base_times, base_values)
    jump_cumsum = jnp.concatenate(
        [jnp.asarray([0.0], dtype=float), jnp.cumsum(jump_values)]
    )
    offset_idx = jnp.searchsorted(jump_times, breaks[:-1], side="right")
    offsets = jump_cumsum[offset_idx]

    y0 = base_at_breaks[:-1] + offsets
    y1 = base_at_breaks[1:] + offsets
    dx = breaks[1:] - breaks[:-1]
    slope = (y1 - y0) / jnp.maximum(dx, _CONSTANT_SPLINE_DT)
    zeros = jnp.zeros_like(slope)
    coeffs = jnp.stack([y0, slope, zeros, zeros], axis=1)
    return jnp.asarray(breaks, dtype=float), jnp.asarray(coeffs, dtype=float)


def _fit_spline_timeseries(
    t: jnp.ndarray,
    y: jnp.ndarray,
    *,
    method: str = "cubic",
    continuity_side: str = "right",
    metadata: Optional[dict] = None,
) -> TimeSeries:
    """Build a spline-backed TimeSeries from discrete samples."""
    t, y = _prepare_knots(t, y)
    t_np = np.asarray(t, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    if method == "pchip":
        ppoly = interpolate.PchipInterpolator(t_np, y_np, extrapolate=True)
    elif method == "cubic":
        ppoly = interpolate.CubicSpline(t_np, y_np, bc_type="natural", extrapolate=True)
    else:
        raise ValueError(f"Unsupported spline method {method!r}")

    breaks, coeffs = _ppoly_to_power_basis(ppoly)
    return TimeSeries(
        times=t,
        values=y,
        breaks=jnp.asarray(breaks, dtype=jnp.float32),
        coeffs=jnp.asarray(coeffs, dtype=jnp.float32),
        segment_start_piece_idx=jnp.asarray([0], dtype=jnp.int32),
        continuity_side=continuity_side,
        metadata=metadata,
    )


def _build_linear_plus_step_timeseries(
    base_times: jnp.ndarray,
    base_values: jnp.ndarray,
    jump_times: jnp.ndarray,
    jump_values: jnp.ndarray,
    *,
    continuity_side: str = "left",
    metadata: Optional[dict] = None,
) -> TimeSeries:
    """Build a TimeSeries payload for linear baseline plus instantaneous jumps."""
    base_times, base_values = _prepare_knots(base_times, base_values)
    jump_times = jnp.asarray(jump_times, dtype=float)
    jump_values = jnp.asarray(jump_values, dtype=float)
    if jump_times.size != jump_values.size:
        raise ValueError("jump_times and jump_values must have the same length")
    if jump_times.size > 0:
        order = jnp.argsort(jump_times)
        jump_times = jump_times[order]
        jump_values = jump_values[order]
        unique_times, inverse = jnp.unique(jump_times, return_inverse=True)
        aggregated = jnp.zeros(unique_times.shape[0], dtype=jump_values.dtype)
        aggregated = aggregated.at[inverse].add(jump_values)
        jump_times = unique_times
        jump_values = aggregated

    ts_metadata = dict(metadata or {})
    ts_metadata["interp"] = "linear_plus_step"
    ts_metadata["jump_values"] = (
        np.asarray(jump_values, dtype=float).tolist() if jump_values is not None else []
    )
    breaks, coeffs = _linear_plus_step_spline_state(
        base_times, base_values, jump_times, jump_values
    )
    return TimeSeries(
        times=base_times,
        values=base_values,
        jump_times=jump_times,
        breaks=breaks,
        coeffs=coeffs,
        segment_start_piece_idx=jnp.asarray([0], dtype=jnp.int32),
        continuity_side=continuity_side,
        metadata=ts_metadata,
    )


def _timeseries_to_canonical_payload(series: TimeSeries) -> Dict[str, Any]:
    """Serialize a nested TimeSeries payload using canonical dict keys."""
    return dict(series.to_dict())


def _timeseries_from_canonical_payload(payload: Dict[str, Any]) -> TimeSeries:
    """Deserialize a canonical nested TimeSeries payload."""
    return TimeSeries.from_dict(payload)


def _timeseries_interp_mode(series: TimeSeries, default: str) -> str:
    """Return the interpolation mode encoded on a transform TimeSeries."""
    metadata = series.metadata if isinstance(series.metadata, dict) else {}
    return str(metadata.get("interp", default))


def _timeseries_jump_values(series: TimeSeries) -> jnp.ndarray:
    """Return jump magnitudes encoded on a transform TimeSeries."""
    metadata = series.metadata if isinstance(series.metadata, dict) else {}
    jump_values = metadata.get("jump_values", [])
    jump_values = jnp.asarray(jump_values, dtype=float)
    jump_times = jnp.asarray(series.jump_times, dtype=float)
    if jump_values.shape[0] != jump_times.shape[0]:
        raise ValueError(
            "Transform TimeSeries jump_values metadata must align with jump_times."
        )
    return jump_values


def evaluate_linear_plus_step(
    t: jnp.ndarray,
    base_times: jnp.ndarray,
    base_values: jnp.ndarray,
    jump_times: jnp.ndarray,
    jump_values: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate linear baseline plus instantaneous left-continuous jumps."""
    t = jnp.asarray(t, dtype=float)
    base = jnp.interp(t, base_times, base_values)
    jump_step_values = _jump_increments_to_step_values(jump_values)
    jump = evaluate_left_continuous_step(t, jump_times, jump_step_values)
    return base + jump


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


def choose_spline_kind(n_points: int) -> str:
    """Pick ``'smoothing_bspline'`` for large N, else ``'cubic_interp'``."""
    if n_points > SMOOTHING_THRESHOLD:
        return "smoothing_bspline"
    return "cubic_interp"


def _fit_smoothing_segment(
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    s: float,
    n_ctrl: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Fit a SciPy smoothing B-spline, then resample to *n_ctrl* control points."""

    if len(x) < 4:
        return x, y
    tck = interpolate.splrep(np.asarray(x), np.asarray(y), s=s, k=3)
    x_ctrl = jnp.linspace(float(x[0]), float(x[-1]), n_ctrl)
    y_ctrl = jnp.asarray(interpolate.splev(np.asarray(x_ctrl), tck))
    return x_ctrl, y_ctrl


def _fit_interp_segment(
    x: jnp.ndarray, y: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """For <= SMOOTHING_THRESHOLD points, store original sorted/unique data."""
    _, idx = jnp.unique(x, return_index=True)
    return x[idx], y[idx]


def _combine_segment_splines(
    segment_points: List[Tuple[jnp.ndarray, jnp.ndarray]],
    boundaries: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Flatten per-segment cubic fits into TimeSeries spline arrays."""
    all_breaks: List[np.ndarray] = []
    all_coeffs: List[np.ndarray] = []
    segment_starts: List[int] = []
    piece_cursor = 0

    boundary_arr = jnp.asarray(boundaries, dtype=float)

    for seg_idx, (seg_t, seg_v) in enumerate(segment_points):
        seg_t, seg_v = _prepare_knots(seg_t, seg_v)
        ppoly = interpolate.CubicSpline(
            np.asarray(seg_t, dtype=np.float64),
            np.asarray(seg_v, dtype=np.float64),
            bc_type="natural",
            extrapolate=True,
        )
        seg_breaks, seg_coeffs = _ppoly_to_power_basis(ppoly)
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
    n_ctrl: int = DEFAULT_MAX_CTRL_POINTS,
) -> TimeSeries:
    """Fit segmented cubic spline state onto a TimeSeries."""
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

    segment_points: List[Tuple[jnp.ndarray, jnp.ndarray]] = []
    kind = "interpax_cubic"
    used_smoothing_fit = False

    for seg in segments:
        seg_t = jnp.asarray(seg.times)
        seg_v = jnp.asarray(seg.values)
        n_pts = len(seg_t)

        if n_pts < 2:
            if n_pts == 1:
                seg_t = jnp.array([seg_t[0], seg_t[0] + 1e-6])
                seg_v = jnp.array([seg_v[0], seg_v[0]])
            else:
                seg_t = jnp.array([0.0, 1e-6])
                seg_v = jnp.array([0.0, 0.0])
            segment_points.append((seg_t, seg_v))
            continue

        strategy = choose_spline_kind(n_pts)
        if strategy == "smoothing_bspline":
            used_smoothing_fit = True
            xc, yc = _fit_smoothing_segment(seg_t, seg_v, s=smoothing_s, n_ctrl=n_ctrl)
        else:
            xc, yc = _fit_interp_segment(seg_t, seg_v)

        segment_points.append((xc, yc))

    breaks, coeffs, segment_start_piece_idx = _combine_segment_splines(
        segment_points, boundaries
    )

    metadata = {
        "smoothing_s": float(smoothing_s),
        "n_ctrl": int(n_ctrl),
        "actual_segments": int(actual_n_segments),
        "fit_strategy": "smoothing_bspline" if used_smoothing_fit else "cubic_interp",
        "kind": kind,
        "bc_type": "natural",
        "segment_boundaries": np.asarray(boundaries, dtype=float).tolist(),
    }

    return TimeSeries(
        times=jnp.asarray(t_arr, dtype=float),
        values=jnp.asarray(v_arr, dtype=float),
        breaks=breaks,
        coeffs=coeffs,
        segment_start_piece_idx=segment_start_piece_idx,
        continuity_side="right",
        metadata=metadata,
    )


def _segment_boundaries_from_series(rep: TimeSeries) -> jnp.ndarray:
    """Return explicit segment boundaries for a spline-backed TimeSeries."""
    metadata = rep.metadata if isinstance(rep.metadata, dict) else {}
    boundaries = metadata.get("segment_boundaries")
    if boundaries is not None:
        return jnp.asarray(boundaries, dtype=float)
    return jnp.asarray([rep.breaks[0], rep.breaks[-1]], dtype=float)


def build_interpax_spline(rep: TimeSeries):
    """Reconstruct per-segment interpax splines from a TimeSeries.

    Returns ``(splines, boundaries)`` where *splines* is a list of
    per-segment interpax spline objects (one per segment) and *boundaries*
    is a 1-D array of length ``n_segments + 1``.
    """
    boundaries = _segment_boundaries_from_series(rep)
    if _has_spline_state(rep):
        piece_starts = jnp.asarray(rep.segment_start_piece_idx, dtype=int)
        piece_starts = np.asarray(piece_starts, dtype=int)
        breaks = jnp.asarray(rep.breaks, dtype=float)
        coeffs = jnp.asarray(rep.coeffs, dtype=float)
        splines = []
        n_pieces = int(coeffs.shape[0])
        if len(piece_starts) + 1 != len(boundaries):
            raise ValueError(
                "Segment boundary metadata must align with segment_start_piece_idx."
            )
        for i, start in enumerate(piece_starts):
            end = piece_starts[i + 1] if i + 1 < len(piece_starts) else n_pieces
            seg_breaks = breaks[start : end + 1]
            if not (
                np.isclose(float(seg_breaks[0]), float(boundaries[i]))
                and np.isclose(float(seg_breaks[-1]), float(boundaries[i + 1]))
            ):
                raise ValueError(
                    "Spline segment breaks must match stored segment boundaries."
                )
            seg_coeffs = coeffs[start:end].T[::-1]
            splines.append(
                interpax.PPoly.construct_fast(
                    seg_coeffs,
                    seg_breaks,
                    extrapolate=True,
                )
            )
        return splines, boundaries

    metadata = rep.metadata if isinstance(rep.metadata, dict) else {}
    bc_type = metadata.get("bc_type", "natural")
    segments = split_timeseries(rep, boundaries)
    splines = []
    for seg in segments:
        xi = jnp.asarray(seg.times, dtype=float)
        yi = jnp.asarray(seg.values, dtype=float)
        sp = interpax.CubicSpline(xi, yi, bc_type=bc_type, check=False)
        splines.append(sp)
    return splines, boundaries


def evaluate_spline_at(rep: TimeSeries, t: float) -> float:
    """Evaluate a segmented cubic TimeSeries at scalar time *t* (not jitted)."""
    splines, boundaries = build_interpax_spline(rep)
    idx = int(jnp.searchsorted(boundaries[1:], float(t), side="right"))
    idx = max(0, min(idx, len(splines) - 1))
    return float(splines[idx](t))


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


def _is_near_constant(values: jnp.ndarray) -> bool:
    """Return True when the series is effectively constant."""
    values = jnp.asarray(values, dtype=float)
    span = float(jnp.max(values) - jnp.min(values))
    scale = max(1.0, float(jnp.max(jnp.abs(values))))
    return span <= max(_IS_CONSTANT_ABS_TOL, _IS_CONSTANT_REL_TOL * scale)


# ===========================================================================
# Pseudobatch transform pipeline
# ===========================================================================


def _build_dense_time_grid(
    process: BioProcess, meas_times: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build dense time grid covering all measurement times, all volume-change
    reference / event times, ± ``_EPS`` knots around discrete events, plus a
    background linspace to guarantee the gap between consecutive knots is
    small enough that piecewise-linear interpolation of ``reactor_volume``
    and ``ADF`` tracks continuous-feed growth faithfully.

    Without the background linspace, processes with sparse continuous-feed
    reference times (e.g. 10 points across 320 h) leave multi-hour gaps in
    the dense grid. Linear interpolation of ``adf_dense`` across such gaps
    would render the smooth continuous-feed growth as a big apparent jump
    right after each event's post-knot.
    """
    extra_times = set()

    t_start = float(process.time_axis.start)
    t_end = float(process.time_axis.end)

    for vc_name, vc in process.volume.volume_changes.items():
        ev_t = _series_reference_times(vc.values)
        for t in ev_t:
            extra_times.add(float(t))
            if not vc.is_continuous:
                t_pre = float(t) - _EPS
                if t_pre >= t_start:
                    extra_times.add(t_pre)
                t_post = float(t) + _EPS
                if t_post <= t_end:
                    extra_times.add(t_post)

    # Background densification: at least ~500 evenly-spaced knots across
    # [t_start, t_end]. Cheap (only hundreds of extra floats) and keeps
    # ADF interpolation faithful regardless of how sparse the user's
    # volume-change reference times are.
    n_background = max(500, 5 * int(len(meas_times)))
    bg = np.linspace(t_start, t_end, n_background, dtype=float)
    extra_times.update(bg.tolist())

    all_times = jnp.array(sorted(set(meas_times.tolist()) | extra_times), dtype=float)

    meas_indices = jnp.array(
        [int(jnp.argmin(jnp.abs(all_times - mt))) for mt in meas_times], dtype=int
    )
    assert jnp.allclose(all_times[meas_indices], meas_times, atol=_EPS / 2), (
        "Some measurement times not found in dense grid"
    )

    return all_times, meas_indices


def _compute_dense_volumes(
    process: BioProcess, dense_times: jnp.ndarray, species_name: str
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]:
    """
    Compute reactor_volume (dense), accumulated_feed (dense), sample_volume (dense)
    and concentration_in_feed for a given species on the dense grid.
    """
    n = len(dense_times)
    reactor_volume = jnp.full(n, float(process.volume.initial_volume))
    feed_streams = []
    sample_volume = jnp.zeros(n)

    for vc_name, vc in process.volume.volume_changes.items():
        if isinstance(vc, FeedVolumeChange):
            c_feed = 0.0
            if vc.feed_medium is not None and species_name in vc.feed_medium.components:
                fc = vc.feed_medium.components[species_name]
                if isinstance(fc.concentration, StaticVariable):
                    c_feed = float(fc.concentration.value)

            if vc.is_continuous:
                cum_feed = _evaluate_timeseries_on_grid(vc.values, dense_times)
            else:
                ev_times = jnp.asarray(vc.values.times, dtype=float)
                ev_vals = jnp.asarray(vc.values.values, dtype=float)
                cum_feed = jnp.zeros(n)
                for et, ev in zip(ev_times, ev_vals):
                    # Exact event timestamp remains pre-event; jump appears after.
                    cum_feed = cum_feed + jnp.where(dense_times > et, float(ev), 0.0)

            reactor_volume = reactor_volume + cum_feed
            feed_streams.append((cum_feed, c_feed))

        elif isinstance(vc, SampleVolumeChange):
            ev_times = jnp.asarray(vc.values.times, dtype=float)
            ev_vals = jnp.asarray(vc.values.values, dtype=float)
            for et, ev in zip(ev_times, ev_vals):
                idx = int(jnp.argmin(jnp.abs(dense_times - et)))
                if jnp.isclose(dense_times[idx], et, atol=_EPS / 2):
                    sample_volume = sample_volume.at[idx].add(abs(float(ev)))
                reactor_volume = reactor_volume + jnp.where(
                    dense_times > et, float(ev), 0.0
                )

    if len(feed_streams) == 1:
        accumulated_feed = feed_streams[0][0]
        concentration_in_feed = feed_streams[0][1]
    elif len(feed_streams) > 1:
        accumulated_feed = jnp.column_stack([fs[0] for fs in feed_streams])
        concentration_in_feed = jnp.array([fs[1] for fs in feed_streams])
    else:
        accumulated_feed = jnp.zeros(n)
        concentration_in_feed = 0.0

    return reactor_volume, accumulated_feed, sample_volume, concentration_in_feed


def _prepare_knots(t: jnp.ndarray, y: jnp.ndarray):
    """Sort and deduplicate knots. Returns (t, y) with at least 2 points."""
    t = jnp.asarray(t, dtype=float)
    y = jnp.asarray(y, dtype=float)
    order = jnp.argsort(t)
    t, y = t[order], y[order]
    _, idx = jnp.unique(t, return_index=True)
    t, y = t[idx], y[idx]
    if len(t) < 2:
        t = jnp.array([t[0], t[0] + 1e-6])
        y = jnp.array([y[0], y[0]])
    return jnp.asarray(t), jnp.asarray(y)


def make_interpax_spline(t: jnp.ndarray, y: jnp.ndarray, bc_type: str = "natural"):
    """
    Build a robust interpax.CubicSpline from arrays. Ensures unique, sorted knots.
    """
    t, y = _prepare_knots(t, y)
    return interpax.CubicSpline(t, y, bc_type=bc_type, check=False)


def make_pchip_spline(t: jnp.ndarray, y: jnp.ndarray):
    """
    Build an interpax.PchipInterpolator from arrays. Ensures unique, sorted knots.

    PCHIP preserves monotonicity between consecutive knots, preventing
    overshoot that can cause negative concentrations from sparse data.
    """
    t, y = _prepare_knots(t, y)
    return interpax.PchipInterpolator(t, y, check=False)


def build_pseudobatch_inputs(process: BioProcess, species_name: str) -> Dict[str, Any]:
    """
    Build canonical inputs for pseudobatch normalization.

    Returned dict contains:
      - meas_times, meas_conc, c_star (arrays at measurement times)
      - dense_times, meas_indices
      - reactor_volume_dense, sample_volume_dense, accumulated_feed_dense,
        concentration_in_feed
      - adf_at_meas, feed_corr_at_meas
      - has_discrete_feed (bool)

    Notes
    -----
    ADF and feed_correction are computed only at measurement times
    (this avoids treating continuous feeds as discrete dilutions).
    Discrete bolus events are treated as left-continuous in this transform:
    at exactly ``t_b`` values are pre-event, and the jump appears for ``t > t_b``.
    """
    # --- measurements
    comp = process.reactor_medium.components[species_name]
    ts = comp.concentration
    assert isinstance(ts, TimeSeries), f"{species_name} must be a TimeSeries"
    meas_times = _series_reference_times(ts)
    meas_conc = _evaluate_timeseries_on_grid(ts, meas_times)

    # --- dense grid and indices
    dense_times, meas_indices = _build_dense_time_grid(process, meas_times)

    # --- dense volumes / feeds / samples
    (
        reactor_volume_dense,
        accumulated_feed_dense,
        sample_volume_dense,
        concentration_in_feed,
    ) = _compute_dense_volumes(process, dense_times, species_name)

    # --- slice at measurement indices
    mi = meas_indices
    rv_at_meas = reactor_volume_dense[mi]
    sv_at_meas = sample_volume_dense[mi]

    af = accumulated_feed_dense
    if af.ndim == 1:
        af_at_meas = af[mi]
    else:
        af_at_meas = af[mi, :]

    # --- ADF via sample-compensation factor.
    # Physically: a sample removes liquid but does NOT change concentration,
    # so ADF must be held at sample events. Yet the reference volume for
    # subsequent bolus ratios must reflect the sample reduction so that
    # bolus dilution ratios at simultaneous sample+bolus events come out
    # physically correct.
    #
    # Define:
    #     S(t)     = product over samples at time ≤ t of V_before / V_after
    #     V_eff(t) = V_reactor_actual(t) · S(t)
    #     ADF(t)   = V_eff(t) / V_init
    #
    # Behaviour at each event type:
    #   - continuous feed : V_reactor grows, S unchanged → ADF grows by
    #                        V_post/V_pre (pseudobatch smoothness invariant)
    #   - sample only     : V_reactor drops by V_s, S multiplies by
    #                        V_pre/(V_pre−V_s) → V_eff and ADF unchanged
    #   - bolus only      : V_reactor grows by V_b, S unchanged → ADF ratio
    #                        = (V_pre+V_b)/V_pre (correct)
    #   - sample + bolus simultaneous (sample first): V_reactor net
    #                        V_b−V_s, S compensates the sample → ADF ratio
    #                        = (V_pre−V_s+V_b)/(V_pre−V_s) (correct)
    n_dense = len(dense_times)
    actual_V = np.asarray(reactor_volume_dense, dtype=float)
    dense_t_np = np.asarray(dense_times, dtype=float)
    V_init = float(process.volume.initial_volume)
    S = np.ones_like(actual_V)
    for _name, vc in process.volume.volume_changes.items():
        if not isinstance(vc, SampleVolumeChange) or vc.is_continuous:
            continue
        ev_times = np.asarray(vc.values.times, dtype=float)
        ev_vals = np.asarray(vc.values.values, dtype=float)
        for t_s, v_s in zip(ev_times, ev_vals):
            # Dense grid contains t_s ± _EPS knots. idx_post lands at
            # t_s + _EPS. V_before is the reactor volume just before the
            # sample (idx_post - 1). Compute V_after_sample EXPLICITLY from
            # the sample amount so that bolus additions coincident with
            # this sample do NOT collapse the compensation ratio.
            idx_post = int(np.searchsorted(dense_t_np, float(t_s) + _EPS / 2.0))
            if idx_post <= 0 or idx_post >= actual_V.size:
                continue
            V_before = float(actual_V[idx_post - 1])
            V_after_sample = V_before + float(v_s)  # v_s is negative
            if V_before <= 0.0 or V_after_sample <= 0.0:
                continue
            S[idx_post:] *= V_before / V_after_sample
    adf_dense = jnp.asarray(actual_V * S / V_init, dtype=float)
    adf_at_meas = adf_dense[mi]

    # --- pseudobatch transform using bolus-only ADF
    def _fed_species_term(accum_feed, conc_in_feed):
        feed_in_interval = jnp.diff(accum_feed, prepend=accum_feed[0])
        return adf_at_meas * feed_in_interval * conc_in_feed / rv_at_meas

    if af_at_meas.ndim == 1:
        c_star = meas_conc * adf_at_meas - jnp.cumsum(
            _fed_species_term(af_at_meas, concentration_in_feed)
        )
    else:
        fed_sum = jnp.zeros(len(meas_times))
        for i in range(af_at_meas.shape[1]):
            fed_sum = fed_sum + _fed_species_term(
                af_at_meas[:, i], concentration_in_feed[i]
            )
        c_star = meas_conc * adf_at_meas - jnp.cumsum(fed_sum)

    feed_corr_at_meas = meas_conc * adf_at_meas - c_star

    # --- dense feed-correction trajectory
    # Same cumulative-sum form as feed_corr_at_meas, but evaluated at every
    # dense grid point. This captures both continuous-feed growth AND
    # instantaneous bolus jumps faithfully, so build_splines can look up
    # fc_pre at a bolus event's t_pre with physics-level accuracy — critical
    # for dual-fed species (continuous + bolus sharing the species).
    def _fed_species_term_dense(accum_feed_dense, conc_in_feed):
        feed_in_interval = jnp.diff(accum_feed_dense, prepend=accum_feed_dense[0])
        return adf_dense * feed_in_interval * conc_in_feed / reactor_volume_dense

    if af.ndim == 1:
        feed_corr_dense = jnp.cumsum(_fed_species_term_dense(af, concentration_in_feed))
    else:
        fed_sum_dense = jnp.zeros(n_dense)
        for i in range(af.shape[1]):
            fed_sum_dense = fed_sum_dense + _fed_species_term_dense(
                af[:, i], concentration_in_feed[i]
            )
        feed_corr_dense = jnp.cumsum(fed_sum_dense)

    has_discrete_feed = any(
        not vc.is_continuous for vc in process.volume.volume_changes.values()
    )

    return {
        "meas_times": meas_times,
        "meas_conc": meas_conc,
        "c_star": jnp.asarray(c_star),
        "dense_times": dense_times,
        "meas_indices": meas_indices,
        "reactor_volume_dense": reactor_volume_dense,
        "sample_volume_dense": sample_volume_dense,
        "accumulated_feed_dense": accumulated_feed_dense,
        "concentration_in_feed": concentration_in_feed,
        "adf_dense": jnp.asarray(adf_dense),
        "adf_at_meas": jnp.asarray(adf_at_meas),
        "feed_corr_at_meas": jnp.asarray(feed_corr_at_meas),
        "feed_corr_dense": jnp.asarray(feed_corr_dense),
        "sample_compensation_dense": jnp.asarray(S),
        "has_discrete_feed": has_discrete_feed,
    }


def _locate_bolus_epsilon_pairs(dense_t: np.ndarray, bolus_times: list) -> list:
    """For each bolus event within the domain, locate its ``t_b ± _EPS`` knot
    indices on the dense grid and return ``(t_b, i_pre, i_post)`` tuples
    sorted by time. Events at or past ``dense_t[-1]`` are dropped.
    """
    out: list[tuple[float, int, int]] = []
    n = dense_t.size
    t_end = float(dense_t[-1])
    half_eps = _EPS / 2.0
    for t_b in sorted(float(t) for t in bolus_times):
        if t_b >= t_end:
            continue
        i_b = int(np.argmin(np.abs(dense_t - t_b)))
        i_pre = i_b
        while i_pre > 0 and dense_t[i_pre] > t_b - half_eps:
            i_pre -= 1
        i_post = i_b
        while i_post < n - 1 and dense_t[i_post] < t_b + half_eps:
            i_post += 1
        if i_post <= i_pre:
            continue
        out.append((t_b, i_pre, i_post))
    return out


def _split_dense_into_smooth_and_jumps(
    dense_t: np.ndarray,
    fc_dense_raw: np.ndarray,
    bolus_epsilon_pairs: list,
) -> tuple:
    """Decompose the raw dense feed-correction trajectory into a continuous
    smooth baseline plus instantaneous bolus jumps.

    Each bolus event contributes a jump of magnitude ``fc_dense_raw[i_post] -
    fc_dense_raw[i_pre]`` — this is the physics-correct jump set by the
    pseudobatch cumsum at the ε-pair knots. Subtracting the cumulative jumps
    from ``fc_dense_raw`` yields a trajectory that is continuous across every
    bolus event (no steps) while preserving all continuous-feed curvature.

    Returns ``(fc_dense_smooth, cumjump_at_dense, jump_times, jump_values)``.
    ``cumjump_at_dense[k]`` is the pre-event cumulative jump at ``dense_t[k]``
    (left-continuous: the jump at an event time is NOT included at t = t_b).
    """
    fc_smooth = fc_dense_raw.astype(float).copy()
    cumjump = np.zeros_like(fc_smooth)
    jump_times: list[float] = []
    jump_values: list[float] = []
    for t_b, i_pre, i_post in bolus_epsilon_pairs:
        delta = float(fc_dense_raw[i_post] - fc_dense_raw[i_pre]) - float(
            cumjump[i_post] - cumjump[i_pre]
        )
        # ``delta`` is the raw jump; subsequent boluses are already accounted
        # for by cumjump, so the physics jump of THIS event is the residual.
        if abs(delta) <= 1e-12:
            continue
        fc_smooth[i_post:] -= delta
        cumjump[i_post:] += delta
        jump_times.append(0.5 * (float(dense_t[i_pre]) + float(dense_t[i_post])))
        jump_values.append(delta)
    return fc_smooth, cumjump, jump_times, jump_values


def _calibrate_smooth_to_meas_anchors(
    dense_t: np.ndarray,
    fc_dense_smooth: np.ndarray,
    meas_idx: np.ndarray,
    fc_at_meas_smooth: np.ndarray,
) -> np.ndarray:
    """Rescale a jump-free dense trajectory so it anchors exactly at the
    provided smooth meas-level values at every measurement index.

    Within each inter-meas interval ``[meas_idx[i], meas_idx[i+1]]``:
        d_a, d_b = fc_dense_smooth[a], fc_dense_smooth[b]
        v_a, v_b = fc_at_meas_smooth[i], fc_at_meas_smooth[i+1]
        fc_calib[k] = v_a + (fc_dense_smooth[k] - d_a) * (v_b - v_a) / (d_b - d_a)

    Because ``fc_dense_smooth`` is continuous (no step discontinuities), the
    linear rescale preserves its curvature without distorting any bolus jump
    — there are none to distort. If ``|d_b - d_a|`` is at noise level, fall
    back to linear time-based interpolation between the two anchors.
    """
    fc_calib = np.zeros_like(fc_dense_smooth, dtype=float)
    n_meas = meas_idx.size
    if n_meas < 2:
        return fc_calib
    for i in range(n_meas - 1):
        a = int(meas_idx[i])
        b = int(meas_idx[i + 1])
        t_a, t_b = float(dense_t[a]), float(dense_t[b])
        v_a = float(fc_at_meas_smooth[i])
        v_b = float(fc_at_meas_smooth[i + 1])
        d_a = float(fc_dense_smooth[a])
        d_b = float(fc_dense_smooth[b])
        seg_slice = slice(a, b + 1)
        if abs(d_b - d_a) > 1e-12:
            scale = (v_b - v_a) / (d_b - d_a)
            fc_calib[seg_slice] = v_a + (fc_dense_smooth[seg_slice] - d_a) * scale
        else:
            denom_t = t_b - t_a if t_b > t_a else 1.0
            frac = (dense_t[seg_slice] - t_a) / denom_t
            fc_calib[seg_slice] = v_a + frac * (v_b - v_a)
    first = int(meas_idx[0])
    last = int(meas_idx[-1])
    if first > 0:
        fc_calib[:first] = float(fc_at_meas_smooth[0])
    if last < fc_calib.size - 1:
        fc_calib[last + 1 :] = float(fc_at_meas_smooth[-1])
    return fc_calib


def build_splines(
    inputs: Dict[str, Any],
    process: "BioProcess | None" = None,
    species_name: "str | None" = None,
) -> Dict[str, Any]:
    """
    Build the runtime pseudobatch spline payload from
    ``build_pseudobatch_inputs``.

    The feed-correction trajectory is represented on the dense time grid and
    calibrated so it anchors exactly at ``feed_corr_at_meas`` at every
    measurement. Between measurements the curve follows the physical dense
    cumulative-sum shape, eliminating the piecewise-linear zig-zag artefact
    that sparse meas-level representations produced for dual-fed species.
    Bolus jumps are extracted deterministically from the known event list.

    Returns a runtime payload dict consumed by
    ``evaluate_real_concentration``, ``to_timeseries``,
    ``BacktransformSpline``, and the mechanistic pseudobatch path.
    """
    spline_cstar = make_interpax_spline(inputs["meas_times"], inputs["c_star"])

    # Switch to PCHIP (monotonicity-preserving) if the cubic spline goes
    # negative OR overshoots the measured c* range significantly. The latter
    # catches near-stepwise c* trajectories that arise when discrete events
    # dominate (e.g. bolus-only processes with no continuous feed) — there,
    # c_star = meas × ADF is essentially piecewise-constant and a natural
    # cubic spline exhibits Gibbs-style oscillation between knots.
    c_star_vals = jnp.asarray(inputs["c_star"], dtype=float)
    if len(c_star_vals) >= 2:
        t_dense = jnp.linspace(
            float(inputs["meas_times"][0]),
            float(inputs["meas_times"][-1]),
            max(200, 10 * len(inputs["meas_times"])),
        )
        c_dense = jax.vmap(spline_cstar)(t_dense)
        data_min = float(jnp.min(c_star_vals))
        data_max = float(jnp.max(c_star_vals))
        data_range = max(data_max - data_min, 1.0)
        overshoot_tol = 0.05 * data_range
        dense_min = float(jnp.min(c_dense))
        dense_max = float(jnp.max(c_dense))
        negative_overshoot = data_min >= 0.0 and dense_min < -1e-8
        range_overshoot = (dense_min < data_min - overshoot_tol) or (
            dense_max > data_max + overshoot_tol
        )
        if negative_overshoot or range_overshoot:
            spline_cstar = make_pchip_spline(inputs["meas_times"], inputs["c_star"])
            inputs["cstar_interp"] = "pchip"

    meas_times = jnp.array(inputs["meas_times"])
    adf_at_meas = jnp.array(inputs["adf_at_meas"])
    fc_at_meas = jnp.array(inputs["feed_corr_at_meas"])

    has_discrete_feed = bool(inputs.get("has_discrete_feed", False))
    spline_feed_corr = None
    if not has_discrete_feed:
        spline_feed_corr = make_interpax_spline(
            inputs["meas_times"], inputs["feed_corr_at_meas"]
        )

    dense_t_np = np.asarray(inputs["dense_times"], dtype=float)
    fc_dense_np = np.asarray(inputs["feed_corr_dense"], dtype=float)
    meas_idx_np = np.asarray(inputs["meas_indices"], dtype=int)
    fc_at_meas_np = np.asarray(inputs["feed_corr_at_meas"], dtype=float)
    meas_t_np = np.asarray(inputs["meas_times"], dtype=float)

    # Known bolus event times. ADF no longer steps at sample events (ADF uses
    # a sample-free volume trajectory) so we only extract jumps at boluses.
    bolus_times: list[float] = []
    if process is not None:
        for _, vc in process.volume.volume_changes.items():
            if not isinstance(vc, FeedVolumeChange) or vc.is_continuous:
                continue
            bolus_times.extend(float(t) for t in np.asarray(vc.values.times))

    # Locate bolus ε-pairs; decompose fc_dense into smooth baseline + physics-
    # correct fc jumps (from the dense cumsum).
    bolus_pairs = _locate_bolus_epsilon_pairs(dense_t_np, bolus_times)
    fc_dense_smooth, cumjump_dense, jump_t_list, jump_v_list = (
        _split_dense_into_smooth_and_jumps(dense_t_np, fc_dense_np, bolus_pairs)
    )

    # Step 2: compute the smooth (jump-removed) meas anchors. Under left-
    # continuous semantics, a meas at exactly t_b sees only jumps at t < t_b.
    # cumjump_dense[idx_of_t_meas] gives exactly that pre-event cumulative
    # because we subtracted deltas starting at i_post, leaving the value at
    # the meas knot itself untouched when meas coincides with bolus time.
    fc_at_meas_smooth_np = fc_at_meas_np - cumjump_dense[meas_idx_np]

    # Step 3: calibrate the smooth baseline so it anchors exactly at
    # fc_at_meas_smooth at every measurement index. Because the trajectory
    # being calibrated has no step discontinuities, the rescaling preserves
    # curvature without distorting jumps (there are none).
    fc_base_np = _calibrate_smooth_to_meas_anchors(
        dense_t_np, fc_dense_smooth, meas_idx_np, fc_at_meas_smooth_np
    )

    fc_base_times = jnp.asarray(dense_t_np, dtype=float)
    fc_base_values = jnp.asarray(fc_base_np, dtype=float)
    fc_jump_times = jnp.asarray(jump_t_list, dtype=float)
    fc_jump_values = jnp.asarray(jump_v_list, dtype=float)

    # Build ADF as a smooth baseline plus instantaneous bolus jumps.
    # Sampling already cancels out of ADF, so only bolus events should create
    # jumps here. This removes the artificial epsilon-window ramp while
    # preserving continuous-feed smoothness.
    adf_dense_np = np.asarray(inputs["adf_dense"], dtype=float)
    adf_dense_smooth, _, adf_jump_t_list, adf_jump_v_list = (
        _split_dense_into_smooth_and_jumps(dense_t_np, adf_dense_np, bolus_pairs)
    )
    adf_base_times = jnp.asarray(dense_t_np, dtype=float)
    adf_base_values = jnp.asarray(adf_dense_smooth, dtype=float)
    adf_jump_times = jnp.asarray(adf_jump_t_list, dtype=float)
    adf_jump_values = jnp.asarray(adf_jump_v_list, dtype=float)

    return {
        "spline_cstar": spline_cstar,
        "spline_feed_corr": spline_feed_corr,
        "meas_times": meas_times,
        "adf_at_meas": adf_at_meas,
        "feed_corr_at_meas": fc_at_meas,
        "feed_corr_base_times": fc_base_times,
        "feed_corr_base_values": fc_base_values,
        "feed_corr_jump_times": fc_jump_times,
        "feed_corr_jump_values": fc_jump_values,
        "adf_base_times": adf_base_times,
        "adf_base_values": adf_base_values,
        "adf_jump_times": adf_jump_times,
        "adf_jump_values": adf_jump_values,
        # Legacy dense ADF arrays retained for compatibility with older
        # transform metadata payloads.
        "adf_times": jnp.asarray(inputs["dense_times"], dtype=float),
        "adf_values": jnp.asarray(inputs["adf_dense"], dtype=float),
        "dense_times": inputs["dense_times"],
        "adf_dense": inputs["adf_dense"],
    }


def evaluate_real_concentration(
    t_eval: jnp.ndarray, splines: Dict[str, Any]
) -> jnp.ndarray:
    """
    Backtransform c*(t) -> c(t) at arbitrary evaluation times t_eval.

    Uses:
      - c*(t) via interpax.CubicSpline (smooth)
      - ADF via dense grid (correct step-wise behaviour at bolus events)
      - feed_correction via jnp.interp (piecewise-linear from augmented grid)

    Returns:
      array of backtransformed concentrations evaluated at t_eval
    """
    t_eval = jnp.asarray(t_eval, dtype=float)
    cs = jnp.asarray(splines["spline_cstar"](jnp.asarray(t_eval)))

    if "adf_base_times" in splines and "adf_base_values" in splines:
        adf = evaluate_linear_plus_step(
            t_eval,
            splines["adf_base_times"],
            splines["adf_base_values"],
            splines.get("adf_jump_times", jnp.zeros(0, dtype=float)),
            splines.get("adf_jump_values", jnp.zeros(0, dtype=float)),
        )
    else:
        adf = jnp.interp(
            t_eval,
            jnp.asarray(splines["dense_times"], dtype=float),
            jnp.asarray(splines["adf_dense"], dtype=float),
        )
    if splines.get("spline_feed_corr") is not None:
        fc = jnp.asarray(splines["spline_feed_corr"](jnp.asarray(t_eval)))
    elif (
        splines.get("feed_corr_jump_times") is not None
        and splines.get("feed_corr_jump_values") is not None
    ):
        fc = evaluate_linear_plus_step(
            t_eval,
            splines.get("feed_corr_base_times", splines["meas_times"]),
            splines.get("feed_corr_base_values", splines["feed_corr_at_meas"]),
            splines["feed_corr_jump_times"],
            splines["feed_corr_jump_values"],
        )
    else:
        raise ValueError(
            "Discrete pseudobatch backtransform requires feed_corr jump metadata "
            "(linear_plus_step)."
        )

    adf = jnp.where(jnp.abs(adf) < 1e-12, 1e-12, adf)

    return (cs + fc) / adf


# ===========================================================================
# TimeSeries conversion and evaluation
# ===========================================================================


def to_timeseries(
    inputs: Dict[str, Any],
    splines: Dict[str, Any],
    species_name: str,
) -> TimeSeries:
    """Convert pseudobatch pipeline outputs to a TimeSeries-first carrier."""
    meas_conc = jnp.asarray(inputs["meas_conc"], dtype=float)
    has_discrete = bool(inputs.get("has_discrete_feed", False))
    is_constant = _is_near_constant(meas_conc) and not has_discrete
    cstar_method = inputs.get("cstar_interp", "cubic")

    adf_ts = _build_linear_plus_step_timeseries(
        splines.get("adf_base_times", inputs["dense_times"]),
        splines.get("adf_base_values", inputs["adf_dense"]),
        splines.get("adf_jump_times", jnp.zeros(0, dtype=float)),
        splines.get("adf_jump_values", jnp.zeros(0, dtype=float)),
        continuity_side="left",
    )

    feed_corr_interp = "linear_plus_step" if has_discrete else "cubic"
    if has_discrete:
        feed_corr_ts = _build_linear_plus_step_timeseries(
            splines.get("feed_corr_base_times", splines["meas_times"]),
            splines.get("feed_corr_base_values", splines["feed_corr_at_meas"]),
            splines.get("feed_corr_jump_times", jnp.zeros(0, dtype=float)),
            splines.get("feed_corr_jump_values", jnp.zeros(0, dtype=float)),
            continuity_side="left",
        )
    else:
        feed_corr_ts = _fit_spline_timeseries(
            splines["meas_times"],
            splines["feed_corr_at_meas"],
            method="cubic",
            continuity_side="right",
            metadata={"interp": "cubic"},
        )

    transform = {
        "name": "pseudo_batch",
        "species": species_name,
        "feed_corr_interp": feed_corr_interp,
        "cstar_interp": cstar_method,
        "is_constant": is_constant,
        "constant_value": float(jnp.mean(meas_conc)) if is_constant else None,
        "series": {
            "adf_ts": _timeseries_to_canonical_payload(adf_ts),
            "feed_corr_ts": _timeseries_to_canonical_payload(feed_corr_ts),
        },
    }

    series = _fit_spline_timeseries(
        inputs["meas_times"],
        inputs["c_star"],
        method="pchip" if cstar_method == "pchip" else "cubic",
        continuity_side="right",
        metadata={"transform": transform},
    )
    return series


class BacktransformSpline(eqx.Module):
    """JIT-compatible evaluation of backtransformed pseudobatch splines.

    Reconstructs the real concentration via the inverse pseudobatch transform:
        ``c(t) = (c*(t) + feed_corr(t)) / ADF(t)``

    For species with constant (or near-constant) measured concentrations,
    the backtransform is bypassed and the stored constant value is returned
    directly (avoiding cubic spline oscillation artifacts).

    All fields are JAX arrays or ``eqx.Module`` instances, so this object
    can be passed through ``eqx.filter_jit`` and used inside JIT-compiled
    functions (e.g. ODE right-hand sides).

    Build with :func:`build_backtransform_spline` from a stored transformed
    :class:`TimeSeries`.
    """

    c_star_spline: interpax.CubicSpline  # or PchipInterpolator (PPoly subclass)
    adf_times: jnp.ndarray
    adf_values: jnp.ndarray
    adf_jump_times: jnp.ndarray
    adf_jump_values: jnp.ndarray
    fc_spline: interpax.CubicSpline
    fc_times: jnp.ndarray
    fc_values: jnp.ndarray
    fc_jump_times: jnp.ndarray
    fc_jump_values: jnp.ndarray
    use_piecewise_adf: bool = eqx.field(static=True)
    use_cubic_fc: bool = eqx.field(static=True)
    is_constant: bool = eqx.field(static=True)
    constant_value: jnp.ndarray

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate backtransformed concentration at time(s) *t*."""
        if self.is_constant:
            return self.constant_value + t * 0.0  # keep JAX tracing happy
        cs = self.c_star_spline(t)
        if self.use_piecewise_adf:
            adf = evaluate_linear_plus_step(
                t,
                self.adf_times,
                self.adf_values,
                self.adf_jump_times,
                self.adf_jump_values,
            )
        else:
            adf = jnp.interp(t, self.adf_times, self.adf_values)
        if self.use_cubic_fc:
            fc = self.fc_spline(t)
        else:
            fc = evaluate_linear_plus_step(
                t,
                self.fc_times,
                self.fc_values,
                self.fc_jump_times,
                self.fc_jump_values,
            )
        adf = jnp.where(jnp.abs(adf) < 1e-12, 1e-12, adf)
        return (cs + fc) / adf

    def derivative(self):
        """Return a callable evaluating dc/dt at time *t*.

        With ``c(t) = (c*(t) + fc(t)) / ADF(t)``, the quotient rule gives

        .. math::
            \\frac{dc}{dt} =
                \\frac{dc^*/dt + dfc/dt}{\\text{ADF}}
                \\;-\\; \\frac{(c^* + fc)}{\\text{ADF}^2}\\,\\frac{d\\text{ADF}}{dt}
              = \\frac{dc^*/dt + dfc/dt - c\\cdot d\\text{ADF}/dt}{\\text{ADF}}

        The ``-c · dADF/dt / ADF`` term is non-zero whenever continuous feed
        (or the sample-compensation factor) makes ADF vary smoothly between
        events, so it MUST be included to avoid a systematic dc/dt bias in
        the mechanistic q-inversion.

        ``dc^*/dt`` uses the analytical cubic/PCHIP spline derivative.
        ``dfc/dt`` uses the spline derivative (cubic mode) or the exact
        piecewise-constant derivative of ``jnp.interp`` (linear_plus_step
        mode, with per-interval slopes pre-computed for the smooth base).
        ``d(ADF)/dt`` uses the exact piecewise-constant derivative of the
        dense linear ADF interpolation.
        """
        dc_star = self.c_star_spline.derivative()
        if self.use_cubic_fc:
            dfc_cubic = self.fc_spline.derivative()
        else:
            # Precompute slopes of piecewise-linear fc interpolation.
            dfc_cubic = None
            _fc_dt = jnp.diff(self.fc_times)
            _fc_slopes = jnp.diff(self.fc_values) / jnp.maximum(
                _fc_dt, jnp.array(1e-12)
            )
            _fc_times = self.fc_times

        # Pre-compute piecewise-constant slopes of the ADF linear interp.
        _adf_dt = jnp.diff(self.adf_times)
        _adf_slopes = jnp.diff(self.adf_values) / jnp.maximum(_adf_dt, jnp.array(1e-12))
        _adf_times = self.adf_times

        def _deriv(t):
            if self.is_constant:
                return t * 0.0
            if self.use_piecewise_adf:
                adf = evaluate_linear_plus_step(
                    t,
                    self.adf_times,
                    self.adf_values,
                    self.adf_jump_times,
                    self.adf_jump_values,
                )
            else:
                adf = jnp.interp(t, self.adf_times, self.adf_values)
            adf = jnp.where(jnp.abs(adf) < 1e-12, 1e-12, adf)
            dc_star_dt = dc_star(t)
            if dfc_cubic is not None:
                dfc_dt = dfc_cubic(t)
            else:
                # Exact derivative of jnp.interp (piecewise constant)
                idx = jnp.searchsorted(_fc_times, t) - 1
                idx = jnp.clip(idx, 0, len(_fc_slopes) - 1)
                dfc_dt = _fc_slopes[idx]
            adf_idx = jnp.searchsorted(_adf_times, t) - 1
            adf_idx = jnp.clip(adf_idx, 0, len(_adf_slopes) - 1)
            dadf_dt = _adf_slopes[adf_idx]
            # c(t) = (cs + fc) / adf; derive dc/dt from the evaluated c(t)
            if self.use_cubic_fc:
                fc = self.fc_spline(t)
            else:
                fc = evaluate_linear_plus_step(
                    t,
                    self.fc_times,
                    self.fc_values,
                    self.fc_jump_times,
                    self.fc_jump_values,
                )
            cs = self.c_star_spline(t)
            c_val = (cs + fc) / adf
            return (dc_star_dt + dfc_dt - c_val * dadf_dt) / adf

        return _deriv


def build_backtransform_spline(rep: TimeSeries) -> BacktransformSpline:
    """Build a JIT-compatible :class:`BacktransformSpline` from a stored
    pseudobatch carrier.

    This is meant to be called **once** (outside JIT).  The returned module
    can then be passed into ``eqx.filter_jit``-compiled functions.

    Parameters
    ----------
    rep:
        A pseudobatch TimeSeries with ``metadata["transform"]``.

    Returns
    -------
    BacktransformSpline
    """
    metadata = rep.metadata if isinstance(rep.metadata, dict) else {}
    if "transform" not in metadata:
        raise ValueError("Pseudobatch TimeSeries must provide metadata['transform'].")
    tr = metadata["transform"]
    xi = jnp.asarray(rep.times, dtype=float)
    yi = jnp.asarray(rep.values, dtype=float)

    is_constant = tr.get("is_constant", False)
    constant_value = jnp.array(tr.get("constant_value") or 0.0)

    cstar_method = tr.get("cstar_interp", "cubic")
    if cstar_method == "pchip":
        c_star_spline = interpax.PchipInterpolator(xi, yi, check=False)
    else:
        bc_type = metadata.get("bc_type", "natural")
        c_star_spline = interpax.CubicSpline(xi, yi, bc_type=bc_type, check=False)

    if "series" not in tr:
        raise ValueError(
            "Pseudobatch transform metadata must use nested "
            "'series.adf_ts'/'series.feed_corr_ts' payloads."
        )
    adf_ts = _timeseries_from_canonical_payload(tr["series"]["adf_ts"])
    feed_corr_ts = _timeseries_from_canonical_payload(tr["series"]["feed_corr_ts"])

    adf_times = jnp.asarray(adf_ts.times, dtype=float)
    adf_values = jnp.asarray(adf_ts.values, dtype=float)
    adf_jump_times = jnp.asarray(adf_ts.jump_times, dtype=float)
    adf_jump_values = _timeseries_jump_values(adf_ts)
    use_piecewise_adf = _timeseries_interp_mode(adf_ts, "linear_plus_step") == (
        "linear_plus_step"
    )

    fc_interp = _timeseries_interp_mode(
        feed_corr_ts, tr.get("feed_corr_interp", "cubic")
    )
    fc_times = jnp.asarray(feed_corr_ts.times, dtype=float)
    fc_values = jnp.asarray(feed_corr_ts.values, dtype=float)
    if fc_interp == "linear_plus_step":
        fc_jump_times = jnp.asarray(feed_corr_ts.jump_times, dtype=float)
        fc_jump_values = _timeseries_jump_values(feed_corr_ts)
    elif fc_interp == "cubic":
        fc_jump_times = jnp.zeros(0, dtype=float)
        fc_jump_values = jnp.zeros(0, dtype=float)
    elif fc_interp == "linear":
        raise ValueError(
            "Legacy feed_corr_interp='linear' unsupported for discrete "
            "pseudobatch backtransform; regenerate transformed TimeSeries "
            "payloads."
        )
    else:
        raise ValueError(
            f"Unknown feed_corr_interp={fc_interp!r}; expected 'cubic' or "
            "'linear_plus_step'."
        )
    use_cubic_fc = fc_interp == "cubic"

    # Build feed_corr spline (used when use_cubic_fc=True; dummy otherwise,
    # but must be a valid CubicSpline to keep the pytree structure fixed)
    fc_spline = make_interpax_spline(fc_times, fc_values)

    return BacktransformSpline(
        c_star_spline=c_star_spline,
        adf_times=adf_times,
        adf_values=adf_values,
        adf_jump_times=adf_jump_times,
        adf_jump_values=adf_jump_values,
        fc_spline=fc_spline,
        fc_times=fc_times,
        fc_values=fc_values,
        fc_jump_times=fc_jump_times,
        fc_jump_values=fc_jump_values,
        use_piecewise_adf=use_piecewise_adf,
        use_cubic_fc=use_cubic_fc,
        is_constant=is_constant,
        constant_value=constant_value,
    )


# =====================================================================
# Batched backtransform spline (vectorized N-species evaluation)
# =====================================================================

_DEFAULT_BATCH_KNOTS = 128


class BatchedBacktransformSpline(eqx.Module):
    """Vectorized evaluation of N concentration splines in a single call.

    Resamples all per-species ``c*`` and feed-correction splines onto a
    shared uniform knot grid and stacks their polynomial coefficients into
    batched ``interpax.PPoly`` objects.  A single call evaluates all N
    species simultaneously, replacing the N separate Python-loop calls
    that dominate ODE RHS cost.

    Build with :func:`build_batched_conc_splines`.
    """

    c_star_ppoly: interpax.PPoly  # coeff shape (4, m, n_sp)
    fc_ppoly: interpax.PPoly  # coeff shape (4, m, n_sp)
    fc_jump_times: jnp.ndarray  # (n_jump,)
    fc_jump_cumsum: jnp.ndarray  # (n_jump + 1, n_sp)
    fc_step_mask: jnp.ndarray  # (n_sp,) bool
    adf_times: jnp.ndarray  # (n_adf,)
    adf_values: jnp.ndarray  # (n_adf,)
    adf_jump_times: jnp.ndarray  # (n_adf_jump,)
    adf_jump_cumsum: jnp.ndarray  # (n_adf_jump + 1,)
    constant_mask: jnp.ndarray  # (n_sp,) bool
    constant_values: jnp.ndarray  # (n_sp,)
    n_species: int = eqx.field(static=True)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate all species concentrations at scalar time *t*.

        Returns
        -------
        jnp.ndarray
            Shape ``(n_sp,)``.
        """
        cs = self.c_star_ppoly(t)  # (n_sp,)
        fc = self.fc_ppoly(t)  # (n_sp,)
        if int(self.fc_jump_times.shape[0]) > 0:
            jump_idx = jnp.searchsorted(self.fc_jump_times, t, side="left")
            jump = self.fc_jump_cumsum[jump_idx]
            fc = fc + jnp.where(self.fc_step_mask, jump, 0.0)
        adf = jnp.interp(t, self.adf_times, self.adf_values)
        if int(self.adf_jump_times.shape[0]) > 0:
            adf_jump_idx = jnp.searchsorted(self.adf_jump_times, t, side="left")
            adf = adf + self.adf_jump_cumsum[adf_jump_idx]
        adf = jnp.where(jnp.abs(adf) < 1e-12, 1e-12, adf)
        result = (cs + fc) / adf
        return jnp.where(self.constant_mask, self.constant_values, result)

    def eval_derivative(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate dc/dt for all species at scalar time *t*.

        Applies the full quotient rule for ``c(t) = (c* + fc) / ADF(t)``:

        .. math::
            \\frac{dc}{dt} =
                \\frac{dc^*/dt + dfc/dt - c\\cdot d\\text{ADF}/dt}{\\text{ADF}}

        The ``c · dADF/dt / ADF`` term matters whenever continuous feed (or
        the sample-compensation factor) makes ADF vary smoothly between
        events — omitting it produces a systematic dc/dt bias.

        Returns
        -------
        jnp.ndarray
            Shape ``(n_sp,)``.
        """
        cs = self.c_star_ppoly(t)  # (n_sp,)
        fc = self.fc_ppoly(t)  # (n_sp,)
        if int(self.fc_jump_times.shape[0]) > 0:
            jump_idx = jnp.searchsorted(self.fc_jump_times, t, side="left")
            jump = self.fc_jump_cumsum[jump_idx]
            fc = fc + jnp.where(self.fc_step_mask, jump, 0.0)
        dc_star = self.c_star_ppoly(t, nu=1)  # (n_sp,)
        dfc = self.fc_ppoly(t, nu=1)  # (n_sp,)
        adf = jnp.interp(t, self.adf_times, self.adf_values)
        if int(self.adf_jump_times.shape[0]) > 0:
            adf_jump_idx = jnp.searchsorted(self.adf_jump_times, t, side="left")
            adf = adf + self.adf_jump_cumsum[adf_jump_idx]
        adf = jnp.where(jnp.abs(adf) < 1e-12, 1e-12, adf)
        # Derivative ignores instantaneous jumps and follows the smooth base.
        adf_dt = jnp.diff(self.adf_times)
        adf_slopes = jnp.diff(self.adf_values) / jnp.maximum(adf_dt, 1e-12)
        idx = jnp.searchsorted(self.adf_times, t) - 1
        idx = jnp.clip(idx, 0, adf_slopes.shape[0] - 1)
        dadf_dt = adf_slopes[idx]
        c_val = (cs + fc) / adf
        result = (dc_star + dfc - c_val * dadf_dt) / adf
        return jnp.where(self.constant_mask, 0.0, result)


def build_batched_conc_splines(
    conc_splines,
    species_names,
    t_start: float,
    t_end: float,
    n_knots: int = _DEFAULT_BATCH_KNOTS,
):
    """Build a :class:`BatchedBacktransformSpline` from individual splines.

    Handles mixed spline types: ``BacktransformSpline`` objects are
    decomposed into their ``c*`` and ``fc`` components; plain
    ``interpax.CubicSpline`` objects are treated as ``c* = spline``,
    ``fc = 0``, ``ADF = 1``.

    Parameters
    ----------
    conc_splines : dict
        Mapping species name → callable spline (BacktransformSpline or
        CubicSpline).
    species_names : list[str]
        Ordered species names (determines column order in batched arrays).
    t_start, t_end : float
        Time range for resampling.
    n_knots : int
        Number of uniformly-spaced knots for resampling (default 128).

    Returns
    -------
    BatchedBacktransformSpline
    """
    x_common = jnp.linspace(t_start, t_end, n_knots)
    n_sp = len(species_names)

    c_star_resampled = []
    fc_resampled = []
    fc_step_mask_list = []
    fc_jump_times_list = []
    fc_jump_values_list = []
    constant_mask_list = []
    constant_values_list = []
    adf_times = None
    adf_values = None
    adf_jump_times = None
    adf_jump_cumsum = None

    for sp_name in species_names:
        sp = conc_splines[sp_name]

        if isinstance(sp, BacktransformSpline):
            if adf_times is None:
                adf_times = sp.adf_times
                adf_values = sp.adf_values
                adf_jump_times = sp.adf_jump_times
                adf_jump_cumsum = jnp.cumsum(
                    jnp.concatenate(
                        [
                            jnp.asarray([0.0], dtype=float),
                            jnp.asarray(sp.adf_jump_values, dtype=float),
                        ]
                    )
                )
            else:
                if not (
                    jnp.array_equal(adf_times, sp.adf_times)
                    and jnp.allclose(adf_values, sp.adf_values, atol=1e-12)
                    and jnp.array_equal(adf_jump_times, sp.adf_jump_times)
                    and jnp.allclose(
                        adf_jump_cumsum,
                        jnp.cumsum(
                            jnp.concatenate(
                                [
                                    jnp.asarray([0.0], dtype=float),
                                    jnp.asarray(sp.adf_jump_values, dtype=float),
                                ]
                            )
                        ),
                        atol=1e-12,
                    )
                ):
                    raise ValueError(
                        "All BacktransformSpline instances in batched build must "
                        "share identical ADF metadata."
                    )

            constant_mask_list.append(sp.is_constant)
            constant_values_list.append(float(sp.constant_value))

            if sp.is_constant:
                # Dummy splines for constant species (masked out in eval)
                c_star_resampled.append(jnp.zeros(n_knots))
                fc_resampled.append(jnp.zeros(n_knots))
                fc_step_mask_list.append(False)
                fc_jump_times_list.append(jnp.zeros(0, dtype=float))
                fc_jump_values_list.append(jnp.zeros(0, dtype=float))
            else:
                c_star_resampled.append(sp.c_star_spline(x_common))
                if sp.use_cubic_fc:
                    fc_resampled.append(sp.fc_spline(x_common))
                    fc_step_mask_list.append(False)
                    fc_jump_times_list.append(jnp.zeros(0, dtype=float))
                    fc_jump_values_list.append(jnp.zeros(0, dtype=float))
                else:
                    # Piecewise-linear base fc + explicit step jump term.
                    fc_resampled.append(jnp.interp(x_common, sp.fc_times, sp.fc_values))
                    has_jump = int(sp.fc_jump_times.shape[0]) > 0
                    fc_step_mask_list.append(has_jump)
                    fc_jump_times_list.append(sp.fc_jump_times)
                    fc_jump_values_list.append(sp.fc_jump_values)
        else:
            # Plain CubicSpline or other callable: treat as c*=spline, fc=0, ADF=1
            constant_mask_list.append(False)
            constant_values_list.append(0.0)
            c_star_resampled.append(sp(x_common))
            fc_resampled.append(jnp.zeros(n_knots))
            fc_step_mask_list.append(False)
            fc_jump_times_list.append(jnp.zeros(0, dtype=float))
            fc_jump_values_list.append(jnp.zeros(0, dtype=float))

    # If no BacktransformSpline was found, use trivial ADF
    if adf_times is None:
        adf_times = jnp.array([t_start, t_end])
        adf_values = jnp.array([1.0, 1.0])
        adf_jump_times = jnp.zeros(0, dtype=float)
        adf_jump_cumsum = jnp.asarray([0.0], dtype=float)

    # Build batched PPoly for c* splines
    c_star_cubic = [
        interpax.CubicSpline(x_common, y, bc_type="natural", check=False)
        for y in c_star_resampled
    ]
    c_star_c = jnp.stack([s.c for s in c_star_cubic], axis=-1)  # (4, m, n_sp)
    c_star_ppoly = interpax.PPoly.construct_fast(c_star_c, x_common, extrapolate=True)

    # Build batched piecewise-linear PPoly for fc baseline.
    # Using linear (not cubic refit) preserves discrete-feed jump semantics.
    fc_mat = jnp.stack(fc_resampled, axis=1)  # (n_knots, n_sp)
    dx = jnp.diff(x_common)  # (m,)
    fc_slope = (fc_mat[1:, :] - fc_mat[:-1, :]) / jnp.maximum(dx[:, None], 1e-12)
    m = int(x_common.shape[0]) - 1
    fc_c = jnp.zeros((4, m, n_sp), dtype=float)
    fc_c = fc_c.at[2, :, :].set(fc_slope)
    fc_c = fc_c.at[3, :, :].set(fc_mat[:-1, :])
    fc_ppoly = interpax.PPoly.construct_fast(fc_c, x_common, extrapolate=True)

    all_jump_times = [
        np.asarray(jt, dtype=float)
        for jt in fc_jump_times_list
        if int(np.asarray(jt).shape[0]) > 0
    ]
    if all_jump_times:
        shared_jump_times = np.unique(np.concatenate(all_jump_times))
        jump_matrix = np.zeros((n_sp, shared_jump_times.shape[0]), dtype=float)
        for i, (jt, jv) in enumerate(zip(fc_jump_times_list, fc_jump_values_list)):
            jt_np = np.asarray(jt, dtype=float)
            jv_np = np.asarray(jv, dtype=float)
            if jt_np.size == 0:
                continue
            idx = np.searchsorted(shared_jump_times, jt_np)
            valid = idx < shared_jump_times.shape[0]
            valid &= np.isclose(shared_jump_times[idx], jt_np, atol=1e-12)
            jump_matrix[i, idx[valid]] = jv_np[valid]
        jump_cumsum = np.concatenate(
            [np.zeros((n_sp, 1), dtype=float), np.cumsum(jump_matrix, axis=1)],
            axis=1,
        )
        fc_jump_times = jnp.asarray(shared_jump_times, dtype=float)
        fc_jump_cumsum = jnp.asarray(jump_cumsum.T, dtype=float)
    else:
        fc_jump_times = jnp.zeros(0, dtype=float)
        fc_jump_cumsum = jnp.zeros((1, n_sp), dtype=float)

    return BatchedBacktransformSpline(
        c_star_ppoly=c_star_ppoly,
        fc_ppoly=fc_ppoly,
        fc_jump_times=fc_jump_times,
        fc_jump_cumsum=fc_jump_cumsum,
        fc_step_mask=jnp.asarray(fc_step_mask_list, dtype=bool),
        adf_times=adf_times,
        adf_values=adf_values,
        adf_jump_times=adf_jump_times,
        adf_jump_cumsum=adf_jump_cumsum,
        constant_mask=jnp.array(constant_mask_list, dtype=bool),
        constant_values=jnp.array(constant_values_list),
        n_species=n_sp,
    )
