"""
Spline fitting, serialization and evaluation for bioprocess time series data.

This module is the canonical place for:
- Discrete event detection from VolumeChanges
- Segmented spline fitting (interpolating or smoothing)
- Conversion of SciPy smoothing B-spline → interpax-compatible parameters
- Reconstruction of interpax splines from stored SplineRepresentation
- Piecewise spline evaluation compatible with JAX JIT
- Pseudo-batch transformation for state variables with discrete feed events

All heavy arrays are stored in padded, fixed-shape format so that JAX
never recompiles due to shape changes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from scipy import interpolate

from .dataclasses import (
    BioProcess,
    DiscreteEvents,
    SplineRepresentation,
    StaticVariable,
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


# ---------------------------------------------------------------------------
# 8) Pseudo-batch transformation for state variables
# ---------------------------------------------------------------------------

def compute_volume_at_times(
    process: BioProcess, times: np.ndarray
) -> np.ndarray:
    """Compute reactor volume at given times using initial volume and discrete feeds.

    Parameters
    ----------
    process:
        A BioProcess with ``volume.initial_volume`` and ``volume.volume_changes``.
    times:
        1-D array of times at which to evaluate volume.

    Returns
    -------
    np.ndarray
        Volume at each time (same shape as *times*).
    """
    times = np.asarray(times, dtype=float)
    vol = np.full_like(times, process.volume.initial_volume)

    for _vc_name, vc in process.volume.volume_changes.items():
        if not vc.is_continuous:
            # Discrete bolus: add delta volumes at event times
            ev_times = np.asarray(vc.values.timepoints, dtype=float)
            ev_vols = np.asarray(vc.values.values, dtype=float)
            for et, ev in zip(ev_times, ev_vols):
                vol = vol + np.where(times >= et, ev, 0.0)
        # Future: handle continuous feeds, sampling, etc.

    return vol


def _step_eval(step_times: np.ndarray, step_values: np.ndarray, t: float) -> float:
    """Evaluate a piecewise-constant (step) function at scalar *t*.

    The function takes value ``step_values[i]`` for
    ``step_times[i] <= t < step_times[i+1]`` (the last value extends to +∞).
    For ``t < step_times[0]``, ``step_values[0]`` is returned (i.e. the
    initial value applies for all times before the first step transition).
    """
    step_times = np.asarray(step_times, dtype=float)
    step_values = np.asarray(step_values, dtype=float)
    idx = int(np.searchsorted(step_times, float(t), side="right")) - 1
    idx = max(0, min(idx, len(step_values) - 1))
    return float(step_values[idx])


def _step_eval_array(
    step_times: np.ndarray, step_values: np.ndarray, t_array: np.ndarray
) -> np.ndarray:
    """Vectorized version of :func:`_step_eval`."""
    step_times = np.asarray(step_times, dtype=float)
    step_values = np.asarray(step_values, dtype=float)
    t_array = np.asarray(t_array, dtype=float)
    idx = np.searchsorted(step_times, t_array, side="right").astype(int) - 1
    idx = np.clip(idx, 0, len(step_values) - 1)
    return step_values[idx]


def pseudo_batch_transform_timeseries(
    process: BioProcess,
    species_name: str,
    ts: TimeSeries,
) -> Dict:
    """Compute pseudo-batch transformed concentration for a state variable.

    Implements the discrete pseudo-batch transform from
    Hesselberg-Thomsen et al. (bioRxiv 2024.05.27.596043).

    Parameters
    ----------
    process:
        BioProcess for the run.
    species_name:
        Name of the reactor medium species (e.g. ``"glucose"``).
    ts:
        TimeSeries of the species concentration.

    Returns
    -------
    dict with keys:
        ``times`` – measurement times (np array),
        ``c_star`` – pseudo-batch transformed concentration,
        ``adf_step_times``, ``adf_step_values`` – ADF step function data,
        ``feed_term_step_times``, ``feed_term_step_values`` – feed term step data.
    """
    times = np.asarray(ts.timepoints, dtype=float)
    conc = np.asarray(ts.values, dtype=float)
    n = len(times)

    # --- Collect discrete bolus events ---
    # Each event: (time, delta_volume, feed_concentration_for_species)
    events: List[Tuple[float, float, float]] = []
    for _vc_name, vc in process.volume.volume_changes.items():
        if not vc.is_continuous:
            ev_times = np.asarray(vc.values.timepoints, dtype=float)
            ev_vols = np.asarray(vc.values.values, dtype=float)
            # Get feed concentration of this species from feed medium
            c_feed = 0.0
            if vc.feed_medium is not None and species_name in vc.feed_medium.components:
                comp = vc.feed_medium.components[species_name]
                if isinstance(comp.concentration, StaticVariable):
                    c_feed = float(comp.concentration.value)
                # Future: handle time-varying feed concentration
            for et, ev in zip(ev_times, ev_vols):
                events.append((float(et), float(ev), c_feed))

    events.sort(key=lambda x: x[0])

    V0 = float(process.volume.initial_volume)

    # --- Build step functions for ADF and feed_term at measurement times ---
    # Volume just before each event (cumulative)
    # ADF_i = product_{k : event_k_time <= times[i]} (V_after_k / V_before_k)
    # where V_after_k = V_before_k + delta_V_k  (S=0 for now)

    # Build event-aligned step data for ADF and feed_term
    # ADF starts at 1.0 before any event
    adf_step_times = [times[0] if n > 0 else 0.0]
    adf_step_values = [1.0]
    feed_term_step_times = [times[0] if n > 0 else 0.0]
    feed_term_step_values = [0.0]

    cumulative_adf = 1.0
    cumulative_feed_term = 0.0
    V_current = V0

    for ev_time, delta_v, c_feed in events:
        V_before = V_current
        V_after = V_before + delta_v
        # ADF factor for this event: V_after / V_before  (S=0)
        if V_before > 0:
            cumulative_adf *= V_after / V_before
        # Feed term increment: ADF_at_event * c_feed * (delta_v / V_after)
        feed_increment = cumulative_adf * c_feed * (delta_v / V_after) if V_after > 0 else 0.0
        cumulative_feed_term += feed_increment
        V_current = V_after

        adf_step_times.append(ev_time)
        adf_step_values.append(cumulative_adf)
        feed_term_step_times.append(ev_time)
        feed_term_step_values.append(cumulative_feed_term)

    adf_step_times = np.array(adf_step_times, dtype=float)
    adf_step_values = np.array(adf_step_values, dtype=float)
    feed_term_step_times = np.array(feed_term_step_times, dtype=float)
    feed_term_step_values = np.array(feed_term_step_values, dtype=float)

    # --- Compute c* at measurement times ---
    adf_at_meas = _step_eval_array(adf_step_times, adf_step_values, times)
    feed_at_meas = _step_eval_array(feed_term_step_times, feed_term_step_values, times)
    c_star = adf_at_meas * conc - feed_at_meas

    return {
        "times": times,
        "c_star": c_star,
        "adf_step_times": adf_step_times,
        "adf_step_values": adf_step_values,
        "feed_term_step_times": feed_term_step_times,
        "feed_term_step_values": feed_term_step_values,
    }


def fit_state_timeseries_spline_pseudobatch(
    ts: TimeSeries,
    process: BioProcess,
    species_name: str,
    *,
    boundaries: Optional[np.ndarray] = None,
    smoothing_s: float = 0.0,
    n_ctrl: int = DEFAULT_MAX_CTRL_POINTS,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    max_ctrl_points: int = DEFAULT_MAX_CTRL_POINTS,
) -> SplineRepresentation:
    """Fit a spline in pseudo-batch space for a state variable.

    The returned ``SplineRepresentation`` stores backtransform metadata so
    that :func:`evaluate_timeseries_spline_at` can map back to true
    concentration (with discrete jumps at feed events).

    Parameters
    ----------
    ts:
        Measured concentration time series for the species.
    process:
        BioProcess for the run (provides volume/feed info).
    species_name:
        Name of the species (must exist in reactor_medium and optionally
        in the feed medium of discrete volume changes).
    boundaries:
        Segment boundaries.  If *None*, a single segment is used.
    smoothing_s, n_ctrl, max_segments, max_ctrl_points:
        Forwarded to :func:`fit_timeseries_spline`.

    Returns
    -------
    SplineRepresentation
        Spline fitted in pseudo-batch space, with transform metadata stored
        in ``spline_metadata["transform"]``.
    """
    pb = pseudo_batch_transform_timeseries(process, species_name, ts)

    # Build a TimeSeries in pseudo-batch space
    ts_star = TimeSeries(
        timepoints=jnp.array(pb["times"]),
        values=jnp.array(pb["c_star"]),
    )

    # Fit spline on pseudo-batch concentration (no segmented boundaries needed
    # because pseudo-batch removes discontinuities)
    rep = fit_timeseries_spline(
        ts_star,
        boundaries=boundaries,
        smoothing_s=smoothing_s,
        n_ctrl=n_ctrl,
        max_segments=max_segments,
        max_ctrl_points=max_ctrl_points,
    )

    # Store backtransform metadata (all JSON-serializable)
    if rep.spline_metadata is None:
        rep.spline_metadata = {}
    rep.spline_metadata["transform"] = {
        "name": "pseudo_batch",
        "species": species_name,
        "adf_step_times": pb["adf_step_times"].tolist(),
        "adf_step_values": pb["adf_step_values"].tolist(),
        "feed_term_step_times": pb["feed_term_step_times"].tolist(),
        "feed_term_step_values": pb["feed_term_step_values"].tolist(),
    }

    return rep


def evaluate_timeseries_spline_at(rep: SplineRepresentation, t: float) -> float:
    """Evaluate a spline at scalar *t*, applying backtransform if present.

    If ``rep.spline_metadata`` contains a ``"transform"`` with
    ``name == "pseudo_batch"``, the spline value (in pseudo-batch space) is
    backtransformed to true concentration, re-introducing discrete jumps:

        ĉ(t) = (ĉ*(t) + feed_term(t)) / ADF(t)

    Otherwise falls back to :func:`evaluate_spline_at`.
    """
    meta = rep.spline_metadata
    if meta is not None and "transform" in meta:
        tr = meta["transform"]
        if tr.get("name") == "pseudo_batch":
            c_star_hat = evaluate_spline_at(rep, t)
            adf_t = _step_eval(
                np.array(tr["adf_step_times"]),
                np.array(tr["adf_step_values"]),
                t,
            )
            feed_t = _step_eval(
                np.array(tr["feed_term_step_times"]),
                np.array(tr["feed_term_step_values"]),
                t,
            )
            if adf_t == 0.0:
                import warnings
                warnings.warn(
                    f"ADF (Accumulated Dilution Factor) is zero at t={t}; "
                    "this indicates an invalid volume calculation "
                    "(e.g. zero initial volume). Returning raw spline value.",
                    stacklevel=2,
                )
                return c_star_hat  # Avoid division by zero
            return (c_star_hat + feed_t) / adf_t

    # No transform → plain evaluation
    return evaluate_spline_at(rep, t)


def evaluate_timeseries_spline(
    rep: SplineRepresentation, t_array: np.ndarray
) -> np.ndarray:
    """Vectorized version of :func:`evaluate_timeseries_spline_at`.

    Returns an array of backtransformed (or plain) spline values.
    """
    t_array = np.asarray(t_array, dtype=float)
    return np.array([evaluate_timeseries_spline_at(rep, float(t)) for t in t_array])
