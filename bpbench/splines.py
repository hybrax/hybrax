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
            tp = jnp.asarray(vc.values.timepoints).tolist()
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
    boundaries = jnp.unique(jnp.concatenate([jnp.array([t_min]), interior, jnp.array([t_max])]))
    return boundaries


def split_timeseries(
    ts: TimeSeries, boundaries: jnp.ndarray
) -> List[TimeSeries]:
    """Split *ts* into segments defined by *boundaries*.

    Points that fall exactly on a boundary belong to both adjacent segments
    (duplicated at split point) to ensure each segment covers its endpoints.
    """
    t = jnp.asarray(ts.timepoints)
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
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    s: float,
    n_ctrl: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Fit a SciPy smoothing B-spline, then resample to *n_ctrl* control points."""
    import numpy as _np  # scipy interop
    if len(x) < 4:
        return x, y
    tck = interpolate.splrep(_np.asarray(x), _np.asarray(y), s=s, k=3)
    x_ctrl = jnp.linspace(float(x[0]), float(x[-1]), n_ctrl)
    y_ctrl = jnp.asarray(interpolate.splev(_np.asarray(x_ctrl), tck))
    return x_ctrl, y_ctrl


def _fit_interp_segment(
    x: jnp.ndarray, y: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """For <= SMOOTHING_THRESHOLD points, store original sorted/unique data."""
    _, idx = jnp.unique(x, return_index=True)
    return x[idx], y[idx]


def fit_timeseries_spline(
    ts: TimeSeries,
    *,
    boundaries: Optional[jnp.ndarray] = None,
    smoothing_s: float = 0.0,
    n_ctrl: int = DEFAULT_MAX_CTRL_POINTS,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    max_ctrl_points: int = DEFAULT_MAX_CTRL_POINTS,
) -> SplineRepresentation:
    """Fit a (segmented) spline to a TimeSeries, returning a padded representation."""
    t_arr = jnp.asarray(ts.timepoints)
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
        TimeSeries(timepoints=jnp.array(t_arr), values=jnp.array(v_arr)),
        boundaries,
    )
    actual_n_segments = len(segments)

    xs: List[jnp.ndarray] = []
    ys: List[jnp.ndarray] = []
    ns: List[int] = []
    kind = "interpax_cubic"
    used_smoothing_fit = False

    for seg in segments:
        seg_t = jnp.asarray(seg.timepoints)
        seg_v = jnp.asarray(seg.values)
        n_pts = len(seg_t)

        if n_pts < 2:
            if n_pts == 1:
                seg_t = jnp.array([seg_t[0], seg_t[0] + 1e-6])
                seg_v = jnp.array([seg_v[0], seg_v[0]])
            else:
                seg_t = jnp.array([0.0, 1e-6])
                seg_v = jnp.array([0.0, 0.0])
            xs.append(seg_t)
            ys.append(seg_v)
            ns.append(len(seg_t))
            continue

        strategy = choose_spline_kind(n_pts)
        if strategy == "smoothing_bspline":
            used_smoothing_fit = True
            xc, yc = _fit_smoothing_segment(seg_t, seg_v, s=smoothing_s, n_ctrl=n_ctrl)
        else:
            xc, yc = _fit_interp_segment(seg_t, seg_v)

        xs.append(xc)
        ys.append(yc)
        ns.append(len(xc))

    x_padded = jnp.zeros((max_segments, max_ctrl_points))
    y_padded = jnp.zeros((max_segments, max_ctrl_points))
    n_padded = jnp.zeros(max_segments, dtype=int)
    boundary_padded = jnp.full(max_segments + 1, float(boundaries[-1]))
    boundary_padded = boundary_padded.at[: len(boundaries)].set(boundaries)

    for i in range(actual_n_segments):
        n_pts = min(ns[i], max_ctrl_points)
        x_padded = x_padded.at[i, :n_pts].set(xs[i][:n_pts])
        y_padded = y_padded.at[i, :n_pts].set(ys[i][:n_pts])
        n_padded = n_padded.at[i].set(n_pts)

    metadata = {
        "smoothing_s": float(smoothing_s),
        "n_ctrl": int(n_ctrl),
        "actual_segments": int(actual_n_segments),
        "fit_strategy": "smoothing_bspline" if used_smoothing_fit else "cubic_interp",
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

    boundaries = jnp.asarray(rep.segment_boundaries[: rep.n_segments + 1])
    return splines, boundaries


def evaluate_spline_at(rep: SplineRepresentation, t: float) -> float:
    """Evaluate a segmented spline at scalar time *t* (not jitted)."""
    splines, boundaries = build_interpax_spline(rep)
    idx = int(jnp.searchsorted(boundaries[1:], float(t), side="right"))
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
    x_padded = jnp.zeros((max_segments, max_ctrl_points))
    y_padded = jnp.zeros((max_segments, max_ctrl_points))
    n_padded = jnp.zeros(max_segments, dtype=int)
    boundary_padded = jnp.full(max_segments + 1, t_max)

    x_padded = x_padded.at[0, 0].set(t_min)
    x_padded = x_padded.at[0, 1].set(t_max)
    y_padded = y_padded.at[0, 0].set(value)
    y_padded = y_padded.at[0, 1].set(value)
    n_padded = n_padded.at[0].set(2)
    boundary_padded = boundary_padded.at[0].set(t_min)

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


def _build_dense_time_grid(process: BioProcess, meas_times: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build dense time grid = sorted(unique(measurement times + event times)).
    For discrete (bolus) events we insert t - EPS so we have a point on both
    sides of the step (sharp step representation).
    Returns:
        dense_times, meas_indices_into_dense
    """
    extra_times = set()

    for vc_name, vc in process.volume.volume_changes.items():
        ev_t = jnp.asarray(vc.values.timepoints, dtype=float)
        for t in ev_t:
            extra_times.add(float(t))
            if not vc.is_continuous:
                t_pre = float(t) - _EPS
                if t_pre > 0:
                    extra_times.add(t_pre)
                if isinstance(vc, SampleVolumeChange):
                    extra_times.add(float(t) + _EPS)

    all_times = jnp.array(sorted(set(meas_times.tolist()) | extra_times), dtype=float)

    meas_indices = jnp.array([int(jnp.argmin(jnp.abs(all_times - mt))) for mt in meas_times], dtype=int)
    assert jnp.allclose(all_times[meas_indices], meas_times, atol=_EPS / 2), \
        "Some measurement times not found in dense grid"

    return all_times, meas_indices


def _compute_dense_volumes(process: BioProcess, dense_times: jnp.ndarray, species_name: str) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]:
    """
    Compute reactor_volume (dense), accumulated_feed (dense), sample_volume (dense)
    and concentration_in_feed for a given species on the dense grid.
    """
    n = len(dense_times)
    reactor_volume = jnp.full(n, float(process.volume.initial_volume))
    feed_streams = []
    sample_volume = jnp.zeros(n)

    for vc_name, vc in process.volume.volume_changes.items():
        ev_times = jnp.asarray(vc.values.timepoints, dtype=float)
        ev_vals = jnp.asarray(vc.values.values, dtype=float)

        if isinstance(vc, FeedVolumeChange):
            c_feed = 0.0
            if (vc.feed_medium is not None and species_name in vc.feed_medium.components):
                fc = vc.feed_medium.components[species_name]
                if isinstance(fc.concentration, StaticVariable):
                    c_feed = float(fc.concentration.value)

            if vc.is_continuous:
                cum_feed = jnp.interp(dense_times, ev_times, ev_vals,
                                     left=ev_vals[0], right=ev_vals[-1])
            else:
                cum_feed = jnp.zeros(n)
                for et, ev in zip(ev_times, ev_vals):
                    cum_feed = cum_feed + jnp.where(dense_times >= et, float(ev), 0.0)

            reactor_volume = reactor_volume + cum_feed
            feed_streams.append((cum_feed, c_feed))

        elif isinstance(vc, SampleVolumeChange):
            for et, ev in zip(ev_times, ev_vals):
                idx = int(jnp.argmin(jnp.abs(dense_times - et)))
                if jnp.isclose(dense_times[idx], et, atol=_EPS / 2):
                    sample_volume = sample_volume.at[idx].add(abs(float(ev)))
                reactor_volume = reactor_volume + jnp.where(dense_times > et, float(ev), 0.0)

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


def make_interpax_spline(t: jnp.ndarray, y: jnp.ndarray, bc_type: str = "natural"):
    """
    Build a robust interpax.CubicSpline from arrays. Ensures unique, sorted knots.
    """
    t = jnp.asarray(t, dtype=float)
    y = jnp.asarray(y, dtype=float)
    order = jnp.argsort(t)
    t, y = t[order], y[order]
    _, idx = jnp.unique(t, return_index=True)
    t, y = t[idx], y[idx]
    if len(t) < 2:
        t = jnp.array([t[0], t[0] + 1e-6])
        y = jnp.array([y[0], y[0]])
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
    import numpy as _np  # pseudobatch interop

    # --- measurements
    comp = process.reactor_medium.components[species_name]
    ts = comp.concentration
    assert isinstance(ts, TimeSeries), f"{species_name} must be a TimeSeries"
    meas_times = jnp.asarray(ts.timepoints, dtype=float)
    meas_conc = jnp.asarray(ts.values, dtype=float)

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
    bolus_vol_dense = jnp.full(n_dense, float(process.volume.initial_volume))

    for _vc_name, vc in process.volume.volume_changes.items():
        if isinstance(vc, FeedVolumeChange) and not vc.is_continuous:
            ev_times = jnp.asarray(vc.values.timepoints, dtype=float)
            ev_vals = jnp.asarray(vc.values.values, dtype=float)
            for et, ev in zip(ev_times, ev_vals):
                bolus_vol_dense = bolus_vol_dense + jnp.where(dense_times >= et, float(ev), 0.0)

    no_sample_dense = jnp.zeros(n_dense)
    # pseudobatch requires numpy arrays
    adf_dense = jnp.asarray(pseudobatch.data_correction.accumulated_dilution_factor(
        _np.asarray(bolus_vol_dense), _np.asarray(no_sample_dense)
    ))
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
            fed_sum = fed_sum + _fed_species_term(af_at_meas[:, i], concentration_in_feed[i])
        c_star = meas_conc * adf_at_meas - jnp.cumsum(fed_sum)

    feed_corr_at_meas = meas_conc * adf_at_meas - c_star

    has_discrete_feed = any(not vc.is_continuous for vc in process.volume.volume_changes.values())

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
      - interp_times, adf_interp, feed_corr_interp  (arrays for jnp.interp)
    """
    spline_cstar = make_interpax_spline(inputs["meas_times"], inputs["c_star"])

    interp_times = jnp.array(inputs["meas_times"])
    interp_adf = jnp.array(inputs["adf_at_meas"])
    interp_fc = jnp.array(inputs["feed_corr_at_meas"])

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
            for t_b in jnp.asarray(vc.values.timepoints, dtype=float):
                if not jnp.any(jnp.abs(meas_t - t_b) < _EPS * 2):
                    bolus_events.append((float(t_b), c_feed))

        bolus_events.sort(key=lambda x: x[0])

        dense_t = inputs["dense_times"]
        dense_adf = inputs["adf_dense"]

        for t_b, c_feed in bolus_events:
            t_pre = t_b - _EPS

            order = jnp.argsort(interp_times)
            interp_times = interp_times[order]
            interp_adf = interp_adf[order]
            interp_fc = interp_fc[order]

            adf_pre = float(jnp.interp(t_pre, dense_t, dense_adf))
            adf_post = float(jnp.interp(t_b, dense_t, dense_adf))

            mask = interp_times <= t_pre
            fc_pre = float(interp_fc[mask][-1]) if jnp.any(mask) else 0.0

            cs_val = float(spline_cstar(jnp.array(t_b)))

            c_pre = (cs_val + fc_pre) / max(adf_pre, 1e-12)

            v_pre = float(jnp.interp(t_pre, dense_t,
                                    inputs["reactor_volume_dense"]))
            v_post = float(jnp.interp(t_b, dense_t,
                                     inputs["reactor_volume_dense"]))
            v_feed = v_post - v_pre

            c_post = (c_pre * v_pre + c_feed * v_feed) / v_post

            fc_post = c_post * adf_post - cs_val

            interp_times = jnp.append(interp_times, jnp.array([t_pre, t_b]))
            interp_adf = jnp.append(interp_adf, jnp.array([adf_pre, adf_post]))
            interp_fc = jnp.append(interp_fc, jnp.array([fc_pre, fc_post]))

        order = jnp.argsort(interp_times)
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


def evaluate_real_concentration(t_eval: jnp.ndarray, splines: Dict[str, Any]) -> jnp.ndarray:
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

    adf = jnp.interp(t_eval, splines["dense_times"], splines["adf_dense"])
    if splines.get("spline_feed_corr") is not None:
        fc = jnp.asarray(splines["spline_feed_corr"](jnp.asarray(t_eval)))
    else:
        fc = jnp.interp(t_eval, splines["meas_times"], splines["feed_corr_at_meas"])

    adf = jnp.where(jnp.abs(adf) < 1e-12, 1e-12, adf)

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
    meas_conc = jnp.asarray(inputs["meas_conc"], dtype=float)
    is_constant = float(jnp.max(jnp.abs(meas_conc))) < 1e-10

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
        # Vectorized: mark points before and after each change
        changes = jnp.abs(jnp.diff(dense_adf)) > 1e-12
        keep_before = jnp.concatenate([changes, jnp.array([False])])
        keep_after = jnp.concatenate([jnp.array([False]), changes])
        keep = keep_before | keep_after
        keep = keep.at[0].set(True)
        keep = keep.at[-1].set(True)
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
        "constant_value": float(jnp.mean(meas_conc)) if is_constant else None,
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

    def derivative(self):
        """Return a callable evaluating dc/dt at time *t*.

        Between discrete events ADF is piecewise constant, so:

        .. math::
            \\frac{dc}{dt} = \\frac{dc^*/dt + dfc/dt}{\\text{ADF}}

        ``dc^*/dt`` uses the analytical cubic spline derivative.
        ``dfc/dt`` uses:

        * The cubic spline derivative when ``use_cubic_fc`` is True
          (continuous feeds — smooth feed correction).
        * The exact piecewise-constant derivative of the linear
          interpolation when ``use_cubic_fc`` is False (bolus feeds —
          step-like feed correction).
        """
        dc_star = self.c_star_spline.derivative()
        if self.use_cubic_fc:
            dfc_cubic = self.fc_spline.derivative()
        else:
            # Precompute slopes of piecewise-linear fc interpolation.
            # Step-transition intervals (very narrow dt) produce extreme
            # slopes that are numerical artifacts — zero them out.
            dfc_cubic = None
            _fc_dt = jnp.diff(self.fc_times)
            _fc_slopes = jnp.diff(self.fc_values) / jnp.maximum(
                _fc_dt, jnp.array(1e-12)
            )
            median_dt = jnp.median(_fc_dt)
            _fc_slopes = jnp.where(_fc_dt < 0.1 * median_dt, 0.0, _fc_slopes)
            _fc_times = self.fc_times

        def _deriv(t):
            if self.is_constant:
                return t * 0.0
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
            return (dc_star_dt + dfc_dt) / adf

        return _deriv


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
    fc_spline = make_interpax_spline(fc_times, fc_values)

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

    c_star_ppoly: interpax.PPoly     # coeff shape (4, m, n_sp)
    fc_ppoly: interpax.PPoly         # coeff shape (4, m, n_sp)
    adf_times: jnp.ndarray           # (n_adf,)
    adf_values: jnp.ndarray          # (n_adf,)
    constant_mask: jnp.ndarray       # (n_sp,) bool
    constant_values: jnp.ndarray     # (n_sp,)
    n_species: int = eqx.field(static=True)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate all species concentrations at scalar time *t*.

        Returns
        -------
        jnp.ndarray
            Shape ``(n_sp,)``.
        """
        cs = self.c_star_ppoly(t)   # (n_sp,)
        fc = self.fc_ppoly(t)       # (n_sp,)
        adf = jnp.interp(t, self.adf_times, self.adf_values)
        adf = jnp.where(jnp.abs(adf) < 1e-12, 1e-12, adf)
        result = (cs + fc) / adf
        return jnp.where(self.constant_mask, self.constant_values, result)

    def eval_derivative(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate dc/dt for all species at scalar time *t*.

        Uses ``PPoly(t, nu=1)`` for analytical cubic-spline derivatives.

        Returns
        -------
        jnp.ndarray
            Shape ``(n_sp,)``.
        """
        dc_star = self.c_star_ppoly(t, nu=1)  # (n_sp,)
        dfc = self.fc_ppoly(t, nu=1)           # (n_sp,)
        adf = jnp.interp(t, self.adf_times, self.adf_values)
        adf = jnp.where(jnp.abs(adf) < 1e-12, 1e-12, adf)
        result = (dc_star + dfc) / adf
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
    constant_mask_list = []
    constant_values_list = []
    adf_times = None
    adf_values = None

    for sp_name in species_names:
        sp = conc_splines[sp_name]

        if isinstance(sp, BacktransformSpline):
            if adf_times is None:
                adf_times = sp.adf_times
                adf_values = sp.adf_values

            constant_mask_list.append(sp.is_constant)
            constant_values_list.append(float(sp.constant_value))

            if sp.is_constant:
                # Dummy splines for constant species (masked out in eval)
                c_star_resampled.append(jnp.zeros(n_knots))
                fc_resampled.append(jnp.zeros(n_knots))
            else:
                c_star_resampled.append(sp.c_star_spline(x_common))
                if sp.use_cubic_fc:
                    fc_resampled.append(sp.fc_spline(x_common))
                else:
                    # Piecewise-linear fc → resample via jnp.interp
                    fc_resampled.append(
                        jnp.interp(x_common, sp.fc_times, sp.fc_values)
                    )
        else:
            # Plain CubicSpline or other callable: treat as c*=spline, fc=0, ADF=1
            constant_mask_list.append(False)
            constant_values_list.append(0.0)
            c_star_resampled.append(sp(x_common))
            fc_resampled.append(jnp.zeros(n_knots))

    # If no BacktransformSpline was found, use trivial ADF
    if adf_times is None:
        adf_times = jnp.array([t_start, t_end])
        adf_values = jnp.array([1.0, 1.0])

    # Build batched PPoly for c* splines
    c_star_cubic = [
        interpax.CubicSpline(x_common, y, bc_type="natural", check=False)
        for y in c_star_resampled
    ]
    c_star_c = jnp.stack([s.c for s in c_star_cubic], axis=-1)  # (4, m, n_sp)
    c_star_ppoly = interpax.PPoly.construct_fast(
        c_star_c, x_common, extrapolate=True
    )

    # Build batched PPoly for fc splines
    fc_cubic = [
        interpax.CubicSpline(x_common, y, bc_type="natural", check=False)
        for y in fc_resampled
    ]
    fc_c = jnp.stack([s.c for s in fc_cubic], axis=-1)  # (4, m, n_sp)
    fc_ppoly = interpax.PPoly.construct_fast(fc_c, x_common, extrapolate=True)

    return BatchedBacktransformSpline(
        c_star_ppoly=c_star_ppoly,
        fc_ppoly=fc_ppoly,
        adf_times=adf_times,
        adf_values=adf_values,
        constant_mask=jnp.array(constant_mask_list, dtype=bool),
        constant_values=jnp.array(constant_values_list),
        n_species=n_sp,
    )
