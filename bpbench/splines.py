"""
Pseudobatch normalization utilities and spline storage for bioprocess data.

This module provides:
- Pseudobatch transform computation (c*, ADF, feed correction)
- Conversion of pseudobatch results to SplineRepresentation for minimal storage
- Reconstruction of backtransformed concentrations from stored SplineRepresentation
- Core spline infrastructure (fitting, evaluation, segmentation)

Design goals:
- Always compute the pseudobatch transform (c*) and the ADF / feed-correction
  at measurement times only.
- Avoid computing ADF by doing a dense-grid cumprod: that incorrectly treats
  continuous cumulative feed as many small discrete dilutions.
- Interpolate:
    - c* : Cubic spline (smooth pseudobatch space)
    - ADF and feed_correction: piecewise-linear interpolation between
      measurement times (no cubic overshoot, no fake steps)
- Keep a dense time grid (measurement times + feed event times with t-eps for
  discrete events) for plotting reactor volumes, but do NOT compute ADF on
  that dense grid.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import equinox as eqx
import interpax
import jax.numpy as jnp
from scipy import interpolate
import pseudobatch
import pseudobatch.data_correction

from .dataclasses import (
    BioProcess,
    DiscreteEvents,
    FeedVolumeChange,
    SampleVolumeChange,
    SplineRepresentation,
    StaticVariable,
    TimeSeries,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Small epsilon used when inserting pre-event points for discrete bolus events.
# Must be large enough to survive float32 quantization but tiny relative to true
# sampling intervals.
_EPS = 1e-4

DEFAULT_MAX_SEGMENTS = 16
DEFAULT_MAX_CTRL_POINTS = 128
SMOOTHING_THRESHOLD = 100  # > 100 points -> smoothing spline


# ===========================================================================
# Core spline infrastructure (fitting, evaluation, segmentation)
# ===========================================================================


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

    order = np.argsort(times)
    sorted_times = np.array(times)[order]
    sorted_labels = [labels[i] for i in order]
    unique_times, unique_idx = np.unique(sorted_times, return_index=True)
    unique_labels = [sorted_labels[i] for i in unique_idx]

    return DiscreteEvents(
        times=jnp.array(unique_times),
        labels=unique_labels,
    )


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


def split_timeseries(
    ts: TimeSeries, boundaries: np.ndarray
) -> List[TimeSeries]:
    """Split *ts* into segments defined by *boundaries*.

    Points that fall exactly on a boundary belong to both adjacent segments
    (duplicated at split point) to ensure each segment covers its endpoints.
    """
    t = np.asarray(ts.timepoints)
    v = np.asarray(ts.values)
    order = np.argsort(t)
    t = t[order]
    v = v[order]

    segments: List[TimeSeries] = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (t >= lo) & (t <= hi)
        seg_t = t[mask]
        seg_v = v[mask]
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


def choose_spline_kind(n_points: int) -> str:
    """Pick ``'smoothing_bspline'`` for large N, else ``'cubic_interp'``."""
    if n_points > SMOOTHING_THRESHOLD:
        return "smoothing_bspline"
    return "cubic_interp"


def _fit_smoothing_segment(
    x: np.ndarray,
    y: np.ndarray,
    *,
    s: float,
    n_ctrl: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a SciPy smoothing B-spline, then resample to *n_ctrl* control points."""
    if len(x) < 4:
        return x.copy(), y.copy()
    tck = interpolate.splrep(x, y, s=s, k=3)
    x_ctrl = np.linspace(x[0], x[-1], n_ctrl)
    y_ctrl = interpolate.splev(x_ctrl, tck)
    return x_ctrl, np.asarray(y_ctrl)


def _fit_interp_segment(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """For <= SMOOTHING_THRESHOLD points, store original sorted/unique data."""
    _, idx = np.unique(x, return_index=True)
    return x[idx].copy(), y[idx].copy()


def fit_timeseries_spline(
    ts: TimeSeries,
    *,
    boundaries: Optional[np.ndarray] = None,
    smoothing_s: float = 0.0,
    n_ctrl: int = DEFAULT_MAX_CTRL_POINTS,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    max_ctrl_points: int = DEFAULT_MAX_CTRL_POINTS,
) -> SplineRepresentation:
    """Fit a (segmented) spline to a TimeSeries, returning a padded representation."""
    t_np = np.asarray(ts.timepoints)
    v_np = np.asarray(ts.values)

    order = np.argsort(t_np)
    t_np = t_np[order]
    v_np = v_np[order]
    _, uidx = np.unique(t_np, return_index=True)
    t_np = t_np[uidx]
    v_np = v_np[uidx]

    if boundaries is None:
        boundaries = np.array([t_np[0], t_np[-1]])

    segments = split_timeseries(
        TimeSeries(timepoints=jnp.array(t_np), values=jnp.array(v_np)),
        boundaries,
    )
    actual_n_segments = len(segments)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ns: List[int] = []
    kind = "interpax_cubic"

    for seg in segments:
        seg_t = np.asarray(seg.timepoints)
        seg_v = np.asarray(seg.values)
        n_pts = len(seg_t)

        if n_pts < 2:
            if n_pts == 1:
                seg_t = np.array([seg_t[0], seg_t[0] + 1e-6])
                seg_v = np.array([seg_v[0], seg_v[0]])
            else:
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


def build_interpax_spline(rep: SplineRepresentation):
    """Reconstruct a list of ``interpax.CubicSpline`` objects from stored params.

    Returns ``(splines, boundaries)`` where *splines* is a list of
    ``interpax.CubicSpline`` (one per segment) and *boundaries* is a 1-D
    array of length ``n_segments + 1``.
    """
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
    """Evaluate a segmented spline at scalar time *t* (not jitted)."""
    splines, boundaries = build_interpax_spline(rep)
    idx = int(np.searchsorted(boundaries[1:], float(t), side="right"))
    idx = max(0, min(idx, rep.n_segments - 1))
    return float(splines[idx](t))


def make_constant_spline(
    value: float,
    t_min: float,
    t_max: float,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    max_ctrl_points: int = DEFAULT_MAX_CTRL_POINTS,
) -> SplineRepresentation:
    """Create a SplineRepresentation for a constant value over [t_min, t_max]."""
    x_padded = np.zeros((max_segments, max_ctrl_points))
    y_padded = np.zeros((max_segments, max_ctrl_points))
    n_padded = np.zeros(max_segments, dtype=int)
    boundary_padded = np.full(max_segments + 1, t_max)

    x_padded[0, 0] = t_min
    x_padded[0, 1] = t_max
    y_padded[0, 0] = value
    y_padded[0, 1] = value
    n_padded[0] = 2
    boundary_padded[0] = t_min

    return SplineRepresentation(
        kind="interpax_cubic",
        x=jnp.array(x_padded),
        y=jnp.array(y_padded),
        n=jnp.array(n_padded),
        n_segments=1,
        segment_boundaries=jnp.array(boundary_padded),
        bc_type="natural",
        spline_metadata={"constant_value": float(value)},
    )


# ===========================================================================
# Pseudobatch transform pipeline
# ===========================================================================


def _build_dense_time_grid(process: BioProcess, meas_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build dense time grid = sorted(unique(measurement times + event times)).
    For discrete (bolus) events we insert t - EPS so we have a point on both
    sides of the step (sharp step representation).
    Returns:
        dense_times, meas_indices_into_dense
    """
    extra_times = set()

    for vc_name, vc in process.volume.volume_changes.items():
        ev_t = np.asarray(vc.values.timepoints, dtype=float)
        for t in ev_t:
            extra_times.add(float(t))
            if not vc.is_continuous:
                t_pre = float(t) - _EPS
                if t_pre > 0:
                    extra_times.add(t_pre)
                if isinstance(vc, SampleVolumeChange):
                    extra_times.add(float(t) + _EPS)

    all_times = np.array(sorted(set(meas_times.tolist()) | extra_times), dtype=float)

    meas_indices = np.array([np.argmin(np.abs(all_times - mt)) for mt in meas_times], dtype=int)
    assert np.allclose(all_times[meas_indices], meas_times, atol=_EPS / 2), \
        "Some measurement times not found in dense grid"

    return all_times, meas_indices


def _compute_dense_volumes(process: BioProcess, dense_times: np.ndarray, species_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Compute reactor_volume (dense), accumulated_feed (dense), sample_volume (dense)
    and concentration_in_feed for a given species on the dense grid.
    """
    n = len(dense_times)
    reactor_volume = np.full(n, float(process.volume.initial_volume))
    feed_streams = []
    sample_volume = np.zeros(n)

    for vc_name, vc in process.volume.volume_changes.items():
        ev_times = np.asarray(vc.values.timepoints, dtype=float)
        ev_vals = np.asarray(vc.values.values, dtype=float)

        if isinstance(vc, FeedVolumeChange):
            c_feed = 0.0
            if (vc.feed_medium is not None and species_name in vc.feed_medium.components):
                fc = vc.feed_medium.components[species_name]
                if isinstance(fc.concentration, StaticVariable):
                    c_feed = float(fc.concentration.value)

            if vc.is_continuous:
                cum_feed = np.interp(dense_times, ev_times, ev_vals,
                                     left=ev_vals[0], right=ev_vals[-1])
            else:
                cum_feed = np.zeros(n)
                for et, ev in zip(ev_times, ev_vals):
                    cum_feed += np.where(dense_times >= et, float(ev), 0.0)

            reactor_volume += cum_feed
            feed_streams.append((cum_feed, c_feed))

        elif isinstance(vc, SampleVolumeChange):
            for et, ev in zip(ev_times, ev_vals):
                idx = np.argmin(np.abs(dense_times - et))
                if np.isclose(dense_times[idx], et, atol=_EPS / 2):
                    sample_volume[idx] += abs(float(ev))
                reactor_volume += np.where(dense_times > et, float(ev), 0.0)

    if len(feed_streams) == 1:
        accumulated_feed = feed_streams[0][0]
        concentration_in_feed = feed_streams[0][1]
    elif len(feed_streams) > 1:
        accumulated_feed = np.column_stack([fs[0] for fs in feed_streams])
        concentration_in_feed = np.array([fs[1] for fs in feed_streams])
    else:
        accumulated_feed = np.zeros(n)
        concentration_in_feed = 0.0

    return reactor_volume, accumulated_feed, sample_volume, concentration_in_feed


def make_interpax_spline(t: np.ndarray, y: np.ndarray, bc_type: str = "natural"):
    """
    Build a robust interpax.CubicSpline from numpy arrays. Ensures unique, sorted knots.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(t)
    t, y = t[order], y[order]
    _, idx = np.unique(t, return_index=True)
    t, y = t[idx], y[idx]
    if len(t) < 2:
        t = np.array([t[0], t[0] + 1e-6])
        y = np.array([y[0], y[0]])
    return interpax.CubicSpline(jnp.asarray(t), jnp.asarray(y), bc_type=bc_type, check=False)


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
    """
    # --- measurements
    comp = process.reactor_medium.components[species_name]
    ts = comp.concentration
    assert isinstance(ts, TimeSeries), f"{species_name} must be a TimeSeries"
    meas_times = np.asarray(ts.timepoints, dtype=float)
    meas_conc = np.asarray(ts.values, dtype=float)

    # --- dense grid and indices
    dense_times, meas_indices = _build_dense_time_grid(process, meas_times)

    # --- dense volumes / feeds / samples
    reactor_volume_dense, accumulated_feed_dense, sample_volume_dense, concentration_in_feed = \
        _compute_dense_volumes(process, dense_times, species_name)

    # --- slice at measurement indices
    mi = meas_indices
    rv_at_meas = reactor_volume_dense[mi]
    sv_at_meas = sample_volume_dense[mi]

    af = accumulated_feed_dense
    if af.ndim == 1:
        af_at_meas = af[mi]
    else:
        af_at_meas = af[mi, :]

    # --- ADF from BOLUS FEEDS ONLY (dense grid, then sliced)
    n_dense = len(dense_times)
    bolus_vol_dense = np.full(n_dense, float(process.volume.initial_volume))

    for _vc_name, vc in process.volume.volume_changes.items():
        if isinstance(vc, FeedVolumeChange) and not vc.is_continuous:
            ev_times = np.asarray(vc.values.timepoints, dtype=float)
            ev_vals = np.asarray(vc.values.values, dtype=float)
            for et, ev in zip(ev_times, ev_vals):
                bolus_vol_dense += np.where(dense_times >= et, float(ev), 0.0)

    no_sample_dense = np.zeros(n_dense)
    adf_dense = pseudobatch.data_correction.accumulated_dilution_factor(
        bolus_vol_dense, no_sample_dense
    )
    adf_at_meas = adf_dense[mi]

    # --- pseudobatch transform using bolus-only ADF
    def _fed_species_term(accum_feed, conc_in_feed):
        feed_in_interval = np.diff(accum_feed, prepend=accum_feed[0])
        return adf_at_meas * feed_in_interval * conc_in_feed / rv_at_meas

    if af_at_meas.ndim == 1:
        c_star = meas_conc * adf_at_meas - np.cumsum(
            _fed_species_term(af_at_meas, concentration_in_feed)
        )
    else:
        fed_sum = np.zeros(len(meas_times))
        for i in range(af_at_meas.shape[1]):
            fed_sum += _fed_species_term(af_at_meas[:, i], concentration_in_feed[i])
        c_star = meas_conc * adf_at_meas - np.cumsum(fed_sum)

    feed_corr_at_meas = meas_conc * adf_at_meas - c_star

    has_discrete_feed = any(not vc.is_continuous for vc in process.volume.volume_changes.values())

    return {
        "meas_times": meas_times,
        "meas_conc": meas_conc,
        "c_star": np.asarray(c_star),
        "dense_times": dense_times,
        "meas_indices": meas_indices,
        "reactor_volume_dense": reactor_volume_dense,
        "sample_volume_dense": sample_volume_dense,
        "accumulated_feed_dense": accumulated_feed_dense,
        "concentration_in_feed": concentration_in_feed,
        "adf_dense": np.asarray(adf_dense),
        "adf_at_meas": np.asarray(adf_at_meas),
        "feed_corr_at_meas": np.asarray(feed_corr_at_meas),
        "has_discrete_feed": has_discrete_feed,
    }


def build_splines(inputs: Dict[str, Any],
                   process: "BioProcess | None" = None,
                   species_name: "str | None" = None) -> Dict[str, Any]:
    """
    Build interpolators from the result of build_pseudobatch_inputs.

    When *process* and *species_name* are supplied the interpolation grid for
    ADF and feed_correction is augmented with points just before / after every
    discrete (bolus) feed event so that the backtransform reproduces the sharp
    concentration drops caused by dilution.

    Returns dict:
      - spline_cstar : interpax.CubicSpline built from (meas_times, c_star)
      - interp_times, adf_interp, feed_corr_interp  (arrays for np.interp)
    """
    spline_cstar = make_interpax_spline(inputs["meas_times"], inputs["c_star"])

    interp_times = inputs["meas_times"].copy()
    interp_adf = inputs["adf_at_meas"].copy()
    interp_fc = inputs["feed_corr_at_meas"].copy()

    if process is not None and species_name is not None:
        meas_t = inputs["meas_times"]

        bolus_events = []
        for _vc_name, vc in process.volume.volume_changes.items():
            if not isinstance(vc, FeedVolumeChange) or vc.is_continuous:
                continue
            c_feed = 0.0
            if (vc.feed_medium is not None
                    and species_name in vc.feed_medium.components):
                fc_comp = vc.feed_medium.components[species_name]
                if isinstance(fc_comp.concentration, StaticVariable):
                    c_feed = float(fc_comp.concentration.value)
            for t_b in np.asarray(vc.values.timepoints, dtype=float):
                if not np.any(np.abs(meas_t - t_b) < _EPS * 2):
                    bolus_events.append((float(t_b), c_feed))

        bolus_events.sort(key=lambda x: x[0])

        dense_t = inputs["dense_times"]
        dense_adf = inputs["adf_dense"]

        for t_b, c_feed in bolus_events:
            t_pre = t_b - _EPS

            order = np.argsort(interp_times)
            interp_times = interp_times[order]
            interp_adf = interp_adf[order]
            interp_fc = interp_fc[order]

            adf_pre = float(np.interp(t_pre, dense_t, dense_adf))
            adf_post = float(np.interp(t_b, dense_t, dense_adf))

            mask = interp_times <= t_pre
            fc_pre = float(interp_fc[mask][-1]) if mask.any() else 0.0

            cs_val = float(spline_cstar(jnp.array(t_b)))

            c_pre = (cs_val + fc_pre) / max(adf_pre, 1e-12)

            v_pre = float(np.interp(t_pre, dense_t,
                                    inputs["reactor_volume_dense"]))
            v_post = float(np.interp(t_b, dense_t,
                                     inputs["reactor_volume_dense"]))
            v_feed = v_post - v_pre

            c_post = (c_pre * v_pre + c_feed * v_feed) / v_post

            fc_post = c_post * adf_post - cs_val

            interp_times = np.append(interp_times, [t_pre, t_b])
            interp_adf = np.append(interp_adf, [adf_pre, adf_post])
            interp_fc = np.append(interp_fc, [fc_pre, fc_post])

        order = np.argsort(interp_times)
        interp_times = interp_times[order]
        interp_adf = interp_adf[order]
        interp_fc = interp_fc[order]

    spline_feed_corr = None
    if not inputs.get("has_discrete_feed", False):
        spline_feed_corr = make_interpax_spline(
            inputs["meas_times"], inputs["feed_corr_at_meas"]
        )

    return {
        "spline_cstar": spline_cstar,
        "spline_feed_corr": spline_feed_corr,
        "meas_times": interp_times,
        "adf_at_meas": interp_adf,
        "feed_corr_at_meas": interp_fc,
        "dense_times": inputs["dense_times"],
        "adf_dense": inputs["adf_dense"],
    }


def evaluate_real_concentration(t_eval: np.ndarray, splines: Dict[str, Any]) -> np.ndarray:
    """
    Backtransform c*(t) -> c(t) at arbitrary evaluation times t_eval.

    Uses:
      - c*(t) via interpax.CubicSpline (smooth)
      - ADF via dense grid (correct step-wise behaviour at bolus events)
      - feed_correction via np.interp (piecewise-linear from augmented grid)

    Returns:
      array of backtransformed concentrations evaluated at t_eval
    """
    t_eval = np.asarray(t_eval, dtype=float)
    cs = np.asarray(splines["spline_cstar"](jnp.asarray(t_eval)))

    adf = np.interp(t_eval, splines["dense_times"], splines["adf_dense"])
    if splines.get("spline_feed_corr") is not None:
        fc = np.asarray(splines["spline_feed_corr"](jnp.asarray(t_eval)))
    else:
        fc = np.interp(t_eval, splines["meas_times"], splines["feed_corr_at_meas"])

    adf = np.where(np.abs(adf) < 1e-12, 1e-12, adf)

    return (cs + fc) / adf


# ===========================================================================
# SplineRepresentation conversion and evaluation
# ===========================================================================


def to_spline_representation(
    inputs: Dict[str, Any],
    splines: Dict[str, Any],
    species_name: str,
    *,
    max_segments: int = 1,
    max_ctrl_points: int = DEFAULT_MAX_CTRL_POINTS,
) -> SplineRepresentation:
    """Convert pseudobatch pipeline outputs to a minimal SplineRepresentation.

    The c* cubic spline knots are stored in the main x/y arrays (single segment).
    ADF and feed_correction grids are stored in ``spline_metadata["transform"]``
    as compact lists (JSON-serializable).

    Parameters
    ----------
    inputs:
        Result of :func:`build_pseudobatch_inputs`.
    splines:
        Result of :func:`build_splines`.
    species_name:
        Name of the species.
    max_segments, max_ctrl_points:
        Padding dimensions for JAX compatibility.

    Returns
    -------
    SplineRepresentation
        Minimal representation that can reconstruct backtransformed
        concentrations via :func:`build_backtransform_spline`.
    """
    # Check if concentration is effectively zero everywhere.
    # Cubic splines through near-zero constant values oscillate wildly
    # when combined with the pseudobatch backtransform; bypass it entirely.
    meas_conc = np.asarray(inputs["meas_conc"], dtype=float)
    is_constant = float(np.max(np.abs(meas_conc))) < 1e-10

    # Fit c* spline as a single-segment SplineRepresentation
    ts_star = TimeSeries(
        timepoints=jnp.array(inputs["meas_times"]),
        values=jnp.array(inputs["c_star"]),
    )
    rep = fit_timeseries_spline(
        ts_star,
        max_segments=max_segments,
        max_ctrl_points=max_ctrl_points,
    )

    # Determine feed_corr interpolation mode
    has_discrete = inputs.get("has_discrete_feed", False)
    feed_corr_interp = "linear" if has_discrete else "cubic"

    # Compress dense ADF grid: keep only points where ADF value changes
    # (ADF is a step function that only changes at bolus feed events).
    # This reduces storage from thousands of points to ~2*N_bolus + 2.
    dense_t = inputs["dense_times"]
    dense_adf = inputs["adf_dense"]
    n_dense = len(dense_adf)
    if n_dense > 2:
        keep = np.zeros(n_dense, dtype=bool)
        keep[0] = True   # always keep first
        keep[-1] = True  # always keep last
        # Keep points where ADF changes AND the point before each change
        changes = np.abs(np.diff(dense_adf)) > 1e-12
        for i in range(len(changes)):
            if changes[i]:
                keep[i] = True      # point before change
                keep[i + 1] = True  # point after change
        adf_compact_t = dense_t[keep]
        adf_compact_v = dense_adf[keep]
    else:
        adf_compact_t = dense_t
        adf_compact_v = dense_adf

    # Store backtransform metadata (all JSON-serializable via lists)
    if rep.spline_metadata is None:
        rep.spline_metadata = {}
    rep.spline_metadata["transform"] = {
        "name": "pseudo_batch",
        "species": species_name,
        "feed_corr_interp": feed_corr_interp,
        "is_constant": is_constant,
        "constant_value": float(np.mean(meas_conc)) if is_constant else None,
        # ADF: compact grid with only step transition points
        "adf_times": adf_compact_t.tolist(),
        "adf_values": adf_compact_v.tolist(),
        # Feed correction: augmented measurement grid from build_splines
        "feed_corr_times": splines["meas_times"].tolist(),
        "feed_corr_values": splines["feed_corr_at_meas"].tolist(),
    }

    return rep


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

    Build with :func:`build_backtransform_spline` from a stored
    :class:`SplineRepresentation`.
    """
    c_star_spline: interpax.CubicSpline
    adf_times: jnp.ndarray
    adf_values: jnp.ndarray
    fc_spline: interpax.CubicSpline
    fc_times: jnp.ndarray
    fc_values: jnp.ndarray
    use_cubic_fc: bool = eqx.field(static=True)
    is_constant: bool = eqx.field(static=True)
    constant_value: jnp.ndarray

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate backtransformed concentration at time(s) *t*."""
        if self.is_constant:
            return self.constant_value + t * 0.0  # keep JAX tracing happy
        cs = self.c_star_spline(t)
        adf = jnp.interp(t, self.adf_times, self.adf_values)
        if self.use_cubic_fc:
            fc = self.fc_spline(t)
        else:
            fc = jnp.interp(t, self.fc_times, self.fc_values)
        adf = jnp.where(jnp.abs(adf) < 1e-12, 1e-12, adf)
        return (cs + fc) / adf


def build_backtransform_spline(rep: SplineRepresentation) -> BacktransformSpline:
    """Build a JIT-compatible :class:`BacktransformSpline` from a stored
    :class:`SplineRepresentation`.

    This is meant to be called **once** (outside JIT).  The returned module
    can then be passed into ``eqx.filter_jit``-compiled functions.

    Parameters
    ----------
    rep:
        SplineRepresentation with ``spline_metadata["transform"]`` containing
        ADF and feed_corr grids (as produced by :func:`to_spline_representation`).

    Returns
    -------
    BacktransformSpline
    """
    tr = rep.spline_metadata["transform"]

    is_constant = tr.get("is_constant", False)
    constant_value = jnp.array(tr.get("constant_value") or 0.0)

    # c* spline (single segment for pseudobatch representations)
    ni = int(rep.n[0])
    xi = rep.x[0, :ni]
    yi = rep.y[0, :ni]
    c_star_spline = interpax.CubicSpline(xi, yi, bc_type=rep.bc_type, check=False)

    # ADF grid
    adf_times = jnp.array(tr["adf_times"], dtype=float)
    adf_values = jnp.array(tr["adf_values"], dtype=float)

    # Feed correction
    fc_times = jnp.array(tr["feed_corr_times"], dtype=float)
    fc_values = jnp.array(tr["feed_corr_values"], dtype=float)
    use_cubic_fc = tr.get("feed_corr_interp") == "cubic"

    # Build feed_corr spline (used when use_cubic_fc=True; dummy otherwise,
    # but must be a valid CubicSpline to keep the pytree structure fixed)
    fc_spline = make_interpax_spline(
        np.asarray(fc_times), np.asarray(fc_values)
    )

    return BacktransformSpline(
        c_star_spline=c_star_spline,
        adf_times=adf_times,
        adf_values=adf_values,
        fc_spline=fc_spline,
        fc_times=fc_times,
        fc_values=fc_values,
        use_cubic_fc=use_cubic_fc,
        is_constant=is_constant,
        constant_value=constant_value,
    )
