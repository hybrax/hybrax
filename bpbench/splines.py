"""
Spline fitting, serialization and evaluation for bioprocess time series data.

This module is the canonical place for:
- Discrete event detection from VolumeChanges
- Segmented spline fitting (interpolating or smoothing)
- Conversion of SciPy smoothing B-spline → interpax-compatible parameters
- Reconstruction of interpax splines from stored SplineRepresentation
- Piecewise spline evaluation compatible with JAX JIT

All heavy arrays are stored in padded, fixed-shape format so that JAX
never recompiles due to shape changes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from scipy import interpolate

from .dataclasses import (
    BioProcess,
    DiscreteEvents,
    SplineRepresentation,
    TimeSeries,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_SEGMENTS = 16
DEFAULT_MAX_CTRL_POINTS = 128
SMOOTHING_THRESHOLD = 100  # > 100 points → smoothing spline


# ---------------------------------------------------------------------------
# 1) Discrete event detection
# ---------------------------------------------------------------------------

def detect_discrete_events(process: BioProcess) -> DiscreteEvents:
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
            tp = np.array(vc.values.timepoints).tolist()
            times.extend(tp)
            labels.extend([vc_name] * len(tp))

    if not times:
        return DiscreteEvents(times=jnp.zeros(0))

    # Sort and deduplicate
    order = np.argsort(times)
    sorted_times = np.array(times)[order]
    sorted_labels = [labels[i] for i in order]
    unique_times, unique_idx = np.unique(sorted_times, return_index=True)
    unique_labels = [sorted_labels[i] for i in unique_idx]

    return DiscreteEvents(
        times=jnp.array(unique_times),
        labels=unique_labels,
    )


# ---------------------------------------------------------------------------
# 2) Segment boundaries from events
# ---------------------------------------------------------------------------

def make_segment_boundaries(
    t_min: float, t_max: float, event_times: jnp.ndarray
) -> np.ndarray:
    """Return segment boundaries ``[t_min, ...events..., t_max]``.

    Only events strictly inside ``(t_min, t_max)`` are included.
    The result is a strictly-increasing numpy array.
    """
    ev = np.asarray(event_times, dtype=float)
    interior = ev[(ev > t_min) & (ev < t_max)]
    boundaries = np.unique(np.concatenate([[t_min], interior, [t_max]]))
    return boundaries


# ---------------------------------------------------------------------------
# 3) Split a TimeSeries into segments
# ---------------------------------------------------------------------------

def split_timeseries(
    ts: TimeSeries, boundaries: np.ndarray
) -> List[TimeSeries]:
    """Split *ts* into segments defined by *boundaries*.

    Points that fall exactly on a boundary belong to both adjacent segments
    (duplicated at split point) to ensure each segment covers its endpoints.
    Segments with fewer than 2 points are still returned.
    """
    t = np.asarray(ts.timepoints)
    v = np.asarray(ts.values)

    # Sort by time
    order = np.argsort(t)
    t = t[order]
    v = v[order]

    segments: List[TimeSeries] = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (t >= lo) & (t <= hi)
        seg_t = t[mask]
        seg_v = v[mask]
        # Remove duplicates (keep first)
        _, idx = np.unique(seg_t, return_index=True)
        seg_t = seg_t[idx]
        seg_v = seg_v[idx]
        segments.append(
            TimeSeries(
                timepoints=jnp.array(seg_t),
                values=jnp.array(seg_v),
            )
        )
    return segments


# ---------------------------------------------------------------------------
# 4) Choose fitting strategy
# ---------------------------------------------------------------------------

def choose_spline_kind(n_points: int) -> str:
    """Pick ``'smoothing_bspline'`` for large N, else ``'cubic_interp'``."""
    if n_points > SMOOTHING_THRESHOLD:
        return "smoothing_bspline"
    return "cubic_interp"


# ---------------------------------------------------------------------------
# 5) SciPy smoothing → interpax-compatible control points
# ---------------------------------------------------------------------------

def _fit_smoothing_segment(
    x: np.ndarray,
    y: np.ndarray,
    *,
    s: float,
    n_ctrl: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a SciPy smoothing B-spline, then resample to *n_ctrl* control points.

    Returns ``(x_ctrl, y_ctrl)`` suitable for ``interpax.CubicSpline``.
    """
    if len(x) < 4:
        # Not enough points for cubic; return raw points
        return x.copy(), y.copy()

    tck = interpolate.splrep(x, y, s=s, k=3)
    x_ctrl = np.linspace(x[0], x[-1], n_ctrl)
    y_ctrl = interpolate.splev(x_ctrl, tck)
    return x_ctrl, np.asarray(y_ctrl)


def _fit_interp_segment(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """For ≤ SMOOTHING_THRESHOLD points, store original sorted/unique data."""
    _, idx = np.unique(x, return_index=True)
    return x[idx].copy(), y[idx].copy()


# ---------------------------------------------------------------------------
# 6) Main fitting entry point
# ---------------------------------------------------------------------------

def fit_timeseries_spline(
    ts: TimeSeries,
    *,
    boundaries: Optional[np.ndarray] = None,
    smoothing_s: float = 0.0,
    n_ctrl: int = DEFAULT_MAX_CTRL_POINTS,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    max_ctrl_points: int = DEFAULT_MAX_CTRL_POINTS,
) -> SplineRepresentation:
    """Fit a (segmented) spline to a TimeSeries, returning a padded representation.

    Parameters
    ----------
    ts:
        Input time series.
    boundaries:
        Segment boundaries (from ``make_segment_boundaries``).  If *None*,
        a single segment spanning the full time range is used.
    smoothing_s:
        Smoothing factor passed to ``scipy.interpolate.splrep`` when the
        segment has more than ``SMOOTHING_THRESHOLD`` points.  0 means
        interpolation through all points.
    n_ctrl:
        Number of control points to resample smoothing splines onto.
    max_segments:
        Padding dimension for segments.
    max_ctrl_points:
        Padding dimension for control points per segment.

    Returns
    -------
    SplineRepresentation
        Padded, serializable representation.
    """
    t_np = np.asarray(ts.timepoints)
    v_np = np.asarray(ts.values)

    # Sort + deduplicate
    order = np.argsort(t_np)
    t_np = t_np[order]
    v_np = v_np[order]
    _, uidx = np.unique(t_np, return_index=True)
    t_np = t_np[uidx]
    v_np = v_np[uidx]

    if boundaries is None:
        boundaries = np.array([t_np[0], t_np[-1]])

    # Split into segments
    segments = split_timeseries(
        TimeSeries(timepoints=jnp.array(t_np), values=jnp.array(v_np)),
        boundaries,
    )
    actual_n_segments = len(segments)

    # Per-segment fitting
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ns: List[int] = []
    kind = "interpax_cubic"

    for seg in segments:
        seg_t = np.asarray(seg.timepoints)
        seg_v = np.asarray(seg.values)
        n_pts = len(seg_t)

        if n_pts < 2:
            # Constant / single point → duplicate to two points
            if n_pts == 1:
                seg_t = np.array([seg_t[0], seg_t[0] + 1e-6])
                seg_v = np.array([seg_v[0], seg_v[0]])
            else:
                # Empty segment — should not normally happen
                seg_t = np.array([0.0, 1e-6])
                seg_v = np.array([0.0, 0.0])
            xs.append(seg_t)
            ys.append(seg_v)
            ns.append(len(seg_t))
            continue

        strategy = choose_spline_kind(n_pts)
        if strategy == "smoothing_bspline":
            kind = "smoothing_bspline_approx"
            xc, yc = _fit_smoothing_segment(seg_t, seg_v, s=smoothing_s, n_ctrl=n_ctrl)
        else:
            xc, yc = _fit_interp_segment(seg_t, seg_v)

        xs.append(xc)
        ys.append(yc)
        ns.append(len(xc))

    # Pad to fixed shapes
    x_padded = np.zeros((max_segments, max_ctrl_points))
    y_padded = np.zeros((max_segments, max_ctrl_points))
    n_padded = np.zeros(max_segments, dtype=int)
    boundary_padded = np.full(max_segments + 1, boundaries[-1])
    boundary_padded[: len(boundaries)] = boundaries

    for i in range(actual_n_segments):
        n_pts = min(ns[i], max_ctrl_points)
        x_padded[i, :n_pts] = xs[i][:n_pts]
        y_padded[i, :n_pts] = ys[i][:n_pts]
        n_padded[i] = n_pts

    metadata = {
        "smoothing_s": float(smoothing_s),
        "n_ctrl": int(n_ctrl),
        "actual_segments": int(actual_n_segments),
    }

    return SplineRepresentation(
        kind=kind,
        x=jnp.array(x_padded),
        y=jnp.array(y_padded),
        n=jnp.array(n_padded),
        n_segments=actual_n_segments,
        segment_boundaries=jnp.array(boundary_padded),
        bc_type="natural",
        spline_metadata=metadata,
    )


# ---------------------------------------------------------------------------
# 7) Reconstruction: SplineRepresentation → callable interpax evaluator
# ---------------------------------------------------------------------------

def build_interpax_spline(rep: SplineRepresentation):
    """Reconstruct a list of ``interpax.CubicSpline`` objects from stored params.

    Returns ``(splines, boundaries)`` where *splines* is a list of
    ``interpax.CubicSpline`` (one per segment) and *boundaries* is a 1-D
    array of length ``n_segments + 1``.

    For use **outside** JIT.  Inside a jitted function, use
    :func:`evaluate_spline_at` instead.
    """
    import interpax

    splines = []
    for i in range(rep.n_segments):
        ni = int(rep.n[i])
        xi = rep.x[i, :ni]
        yi = rep.y[i, :ni]
        sp = interpax.CubicSpline(xi, yi, bc_type=rep.bc_type, check=False)
        splines.append(sp)

    boundaries = np.asarray(rep.segment_boundaries[: rep.n_segments + 1])
    return splines, boundaries


def evaluate_spline_at(rep: SplineRepresentation, t: float) -> float:
    """Evaluate a segmented spline at scalar time *t* (not jitted).

    This is a convenience for plotting / verification.  For JIT-compiled
    evaluation inside diffrax, pass the reconstructed ``interpax.CubicSpline``
    objects directly.
    """
    splines, boundaries = build_interpax_spline(rep)
    idx = int(np.searchsorted(boundaries[1:], float(t), side="right"))
    idx = max(0, min(idx, rep.n_segments - 1))
    return float(splines[idx](t))
