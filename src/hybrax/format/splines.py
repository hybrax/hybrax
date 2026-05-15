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
- Fit:
    - c*: smoothing TimeSeries spline where possible
    - ADF: exact TimeSeries pieces with left-continuous jumps
    - feed_correction: exact TimeSeries pieces with left-continuous jumps
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy import interpolate

from .dataclasses import (
    BioProcess,
    DiscreteEvents,
    FeedVolumeChange,
    PseudobatchSpeciesTransform,
    PseudobatchTransform,
    SampleVolumeChange,
    StaticVariable,
    TimeSeries,
)
from .time_series import PPoly, spline_ops


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONSTANT_SPLINE_DT = 1e-6
_IS_CONSTANT_ABS_TOL = 1e-8
_IS_CONSTANT_REL_TOL = 1e-4
_JUMP_VALUE_ABS_TOL = 1e-12
_FLOAT_EQ_ATOL = 1e-12
_FLOAT32_TOL_MULTIPLIER = 64.0
_ADF_MIN_FOR_DIVISION = 1e-12
_MIN_REACTOR_VOLUME = 1e-10
_MIN_SMOOTHING_BSPLINE_SAMPLES = 4

DEFAULT_MAX_SEGMENTS = 16


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


def _adf_for_division(adf: jnp.ndarray) -> jnp.ndarray:
    """Fail fast when ADF is too close to zero before division."""
    adf_arr = jnp.asarray(adf)
    return eqx.error_if(
        adf_arr,
        jnp.any(jnp.abs(adf_arr) <= _ADF_MIN_FOR_DIVISION),
        "Pseudobatch ADF reached zero or near-zero; reactor volume/sample "
        "compensation is physically invalid.",
    )


def _require_reactor_volume_scalar(volume: float, *, context: str) -> float:
    """Python-side reactor-volume invariant check."""
    volume_float = float(volume)
    if volume_float <= _MIN_REACTOR_VOLUME:
        raise ValueError(f"{context} reached zero or near-zero reactor volume.")
    return volume_float


def _require_volume_piece_above_threshold(
    coeff_row: np.ndarray,
    width: float,
    *,
    context: str,
) -> None:
    """Fail if a local cubic reactor-volume piece crosses the volume floor."""
    a, b, c, d = np.asarray(coeff_row, dtype=np.float64)
    candidates = [0.0, float(width)]
    derivative_roots = np.roots(np.asarray([3.0 * d, 2.0 * c, b], dtype=np.float64))
    for root in derivative_roots:
        if abs(root.imag) <= _FLOAT_EQ_ATOL:
            t = float(root.real)
            if 0.0 <= t <= width:
                candidates.append(t)
    values = [a + b * t + c * (t**2) + d * (t**3) for t in candidates]
    if min(values) <= _MIN_REACTOR_VOLUME:
        raise ValueError(f"{context} reached zero or near-zero reactor volume.")


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


def _ppoly_from_spline_state(rep: TimeSeries) -> PPoly:
    """Return stored TimeSeries power-basis spline state as owned PPoly."""
    if not _has_spline_state(rep):
        raise ValueError("spline representation required")
    poly = rep.poly
    if poly is None:
        raise ValueError("spline representation required")
    return poly


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


_evaluate_piece_batched = jax.jit(jax.vmap(spline_ops.evaluate_piece))


def _break_values_from_coeffs(
    breaks: jnp.ndarray,
    coeffs: jnp.ndarray,
    *,
    continuity_side: str,
) -> jnp.ndarray:
    """Return representative values at breaks for metadata/sample payloads."""
    breaks_j = jnp.asarray(breaks, dtype=float)
    coeffs_j = jnp.asarray(coeffs, dtype=float)
    if continuity_side == "left":
        widths = breaks_j[1:] - breaks_j[:-1]
        evaluated = _evaluate_piece_batched(coeffs_j, widths)
        return jnp.concatenate([coeffs_j[:1, 0], evaluated])
    width = breaks_j[-1:] - breaks_j[-2:-1]
    last = _evaluate_piece_batched(coeffs_j[-1:], width)
    return jnp.concatenate([coeffs_j[:, 0], last])


def _piecewise_polynomial_timeseries(
    breaks: jnp.ndarray,
    coeffs: jnp.ndarray,
    *,
    continuity_side: str = "left",
    times: Optional[jnp.ndarray] = None,
    values: Optional[jnp.ndarray] = None,
    jump_times: Optional[jnp.ndarray] = None,
    metadata: Optional[dict] = None,
) -> TimeSeries:
    """Build a spline-backed TimeSeries with exact local polynomial coeffs."""
    breaks = jnp.asarray(breaks, dtype=float)
    coeffs = jnp.asarray(coeffs, dtype=float)
    if times is None:
        times = breaks
    if values is None:
        values = _break_values_from_coeffs(
            breaks, coeffs, continuity_side=continuity_side
        )
    if jump_times is None:
        jump_times = jnp.zeros(0, dtype=float)
    return TimeSeries(
        times=jnp.asarray(times, dtype=float),
        values=jnp.asarray(values, dtype=float),
        jump_times=jnp.asarray(jump_times, dtype=float),
        breaks=breaks,
        coeffs=coeffs,
        segment_start_piece_idx=jnp.asarray([0], dtype=jnp.int32),
        continuity_side=continuity_side,
        metadata=dict(metadata or {}),
    )


def _constant_timeseries(
    value: float,
    t_start: float,
    t_end: float,
    *,
    continuity_side: str = "left",
    metadata: Optional[dict] = None,
) -> TimeSeries:
    """Build an exact constant TimeSeries over ``[t_start, t_end]``."""
    end = float(t_end)
    if end <= float(t_start):
        end = float(t_start) + _CONSTANT_SPLINE_DT
    breaks = jnp.asarray([float(t_start), end], dtype=float)
    coeffs = jnp.asarray([[float(value), 0.0, 0.0, 0.0]], dtype=float)
    ts_metadata = dict(metadata or {})
    ts_metadata.setdefault("interp", "constant")
    return _piecewise_polynomial_timeseries(
        breaks,
        coeffs,
        continuity_side=continuity_side,
        metadata=ts_metadata,
    )


def _piecewise_constant_from_intervals(
    breaks: jnp.ndarray,
    interval_values: jnp.ndarray,
    *,
    continuity_side: str = "left",
    jump_times: Optional[jnp.ndarray] = None,
    metadata: Optional[dict] = None,
) -> TimeSeries:
    """Build an exact piecewise-constant TimeSeries from interval values."""
    values = jnp.asarray(interval_values, dtype=float)
    zeros = jnp.zeros_like(values)
    coeffs = jnp.stack([values, zeros, zeros, zeros], axis=1)
    ts_metadata = dict(metadata or {})
    ts_metadata.setdefault("interp", "piecewise_constant")
    return _piecewise_polynomial_timeseries(
        breaks,
        coeffs,
        continuity_side=continuity_side,
        jump_times=jump_times,
        metadata=ts_metadata,
    )


def _linear_coeffs_from_samples_on_breaks(
    series: TimeSeries,
    breaks: jnp.ndarray,
) -> jnp.ndarray:
    """Represent a sample-only TimeSeries as linear pieces on ``breaks``."""
    if series.times is None or series.values is None:
        raise ValueError("sample-based linear TimeSeries requires times and values")
    base_t, base_v = _prepare_knots(series.times, series.values)
    y = jnp.interp(breaks, base_t, base_v, left=base_v[0], right=base_v[-1])
    dx = jnp.diff(breaks)
    slope = jnp.diff(y) / jnp.maximum(dx, _CONSTANT_SPLINE_DT)
    zeros = jnp.zeros_like(slope)
    return jnp.stack([y[:-1], slope, zeros, zeros], axis=1)


def _series_coeffs_on_breaks(series: TimeSeries, breaks: jnp.ndarray) -> jnp.ndarray:
    """Express ``series`` exactly or linearly on a canonical break grid."""
    if _has_spline_state(series):
        return spline_ops.rebase_to_breaks(series.breaks, series.coeffs, breaks)
    if _has_discrete_samples(series):
        return _linear_coeffs_from_samples_on_breaks(series, breaks)
    raise ValueError("TimeSeries must provide spline state or discrete samples")


def _feed_concentration_for_species(
    vc: FeedVolumeChange,
    species_name: str,
) -> float:
    """Return static feed concentration for ``species_name`` or fail fast."""
    if vc.feed_medium is None or species_name not in vc.feed_medium.components:
        return 0.0
    concentration = vc.feed_medium.components[species_name].concentration
    if not isinstance(concentration, StaticVariable):
        raise NotImplementedError(
            "Pseudobatch feed correction currently requires static feed "
            f"concentration for species {species_name!r} in feed {vc.name!r}."
        )
    return float(concentration.value)


def _canonical_pseudobatch_breaks(
    process: BioProcess,
    meas_times: jnp.ndarray,
) -> jnp.ndarray:
    """Collect exact process, measurement, feed, and event breakpoints."""
    t_start = float(process.time_axis.start)
    t_end = float(process.time_axis.end)
    points: list[float] = [t_start, t_end]
    points.extend(float(t) for t in np.asarray(meas_times, dtype=float))
    for _, vc in process.volume.volume_changes.items():
        values = vc.values
        if _has_spline_state(values):
            points.extend(float(t) for t in np.asarray(values.breaks, dtype=float))
        if _has_discrete_samples(values):
            points.extend(float(t) for t in np.asarray(values.times, dtype=float))
    in_domain = [t for t in points if t_start <= t <= t_end]
    breaks = np.asarray(sorted(set(in_domain)), dtype=np.float64)
    if breaks.size < 2:
        breaks = np.asarray([t_start, t_start + _CONSTANT_SPLINE_DT], dtype=np.float64)
    return jnp.asarray(breaks, dtype=float)


def _discrete_volume_events(process: BioProcess, species_name: str) -> dict:
    """Aggregate exact sample and bolus deltas by event timestamp."""
    events: dict[float, dict[str, list]] = {}
    for vc_name, vc in process.volume.volume_changes.items():
        if vc.is_continuous:
            continue
        if vc.values.times is None or vc.values.values is None:
            raise ValueError(f"Discrete volume change {vc_name!r} needs samples")
        for t_ev, delta_v in zip(vc.values.times, vc.values.values):
            t_key = float(t_ev)
            record = events.setdefault(t_key, {"samples": [], "boluses": []})
            if isinstance(vc, SampleVolumeChange):
                record["samples"].append((vc_name, float(delta_v)))
            elif isinstance(vc, FeedVolumeChange):
                record["boluses"].append(
                    (
                        vc_name,
                        float(delta_v),
                        _feed_concentration_for_species(vc, species_name),
                    )
                )
    return events


def _jump_values_from_series(series: TimeSeries) -> jnp.ndarray:
    """Compute jump magnitudes from right-minus-left side evaluation."""
    jump_times = jnp.asarray(series.jump_times, dtype=float)
    if jump_times.size == 0:
        return jnp.zeros(0, dtype=float)
    right = series.evaluate_many(jump_times, side="right")
    left = series.evaluate_many(jump_times, side="left")
    return jnp.asarray(right - left, dtype=float)


def _timeseries_base_values_without_jumps(series: TimeSeries) -> jnp.ndarray:
    """Return sample values with left-continuous jump offsets removed."""
    if series.times is None or series.values is None:
        return jnp.zeros(0, dtype=float)
    values = jnp.asarray(series.values, dtype=float)
    jump_times = jnp.asarray(series.jump_times, dtype=float)
    if jump_times.size == 0:
        return values
    jump_values = _timeseries_jump_values(series)
    jump_cumsum = jnp.concatenate(
        [jnp.asarray([0.0], dtype=float), jnp.cumsum(jump_values)]
    )
    idx = jnp.searchsorted(
        jump_times, jnp.asarray(series.times, dtype=float), side="left"
    )
    return values - jump_cumsum[idx]


def _series_jump_times_and_values(
    series: TimeSeries,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return jump metadata, adding exact-start boundary jump when needed."""
    jump_times = jnp.asarray(series.jump_times, dtype=float)
    jump_values = _timeseries_jump_values(series)
    metadata = series.metadata if isinstance(series.metadata, dict) else {}
    if (
        "boundary_start_value" in metadata
        and series.breaks is not None
        and series.coeffs is not None
    ):
        start = jnp.asarray(series.breaks[0], dtype=float)
        pre = jnp.asarray(float(metadata["boundary_start_value"]), dtype=float)
        post = series.evaluate(start, side="right")
        delta = post - pre
        has_start = bool(jnp.any(jump_times == start)) if jump_times.size > 0 else False
        if (not has_start) and abs(float(delta)) > _JUMP_VALUE_ABS_TOL:
            jump_times = jnp.concatenate([jnp.asarray([start]), jump_times])
            jump_values = jnp.concatenate([jnp.asarray([delta]), jump_values])
    if jump_times.size == 0:
        return jump_times, jump_values
    order = jnp.argsort(jump_times)
    jump_times = jump_times[order]
    jump_values = jump_values[order]
    unique_times, inverse = jnp.unique(jump_times, return_inverse=True)
    aggregated = jnp.zeros(unique_times.shape[0], dtype=jump_values.dtype)
    aggregated = aggregated.at[inverse].add(jump_values)
    return unique_times, aggregated


def _baseline_values_on_grid(series: TimeSeries, grid: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the jump-free baseline for a series on a grid."""
    values = _evaluate_many_with_boundary_start(series, grid)
    jump_times, jump_values = _series_jump_times_and_values(series)
    if jump_times.size == 0:
        return values
    jump_cumsum = jnp.concatenate(
        [jnp.asarray([0.0], dtype=float), jnp.cumsum(jump_values)]
    )
    idx = jnp.searchsorted(jump_times, grid, side="left")
    return values - jump_cumsum[idx]


def _baseline_coeffs_on_breaks(
    series: TimeSeries,
    target_breaks: jnp.ndarray,
) -> jnp.ndarray:
    """Express a TimeSeries jump-free baseline on ``target_breaks``."""
    if series.breaks is None or series.coeffs is None:
        values = _baseline_values_on_grid(series, target_breaks)
        dx = jnp.diff(target_breaks)
        slope = jnp.diff(values) / jnp.maximum(dx, _CONSTANT_SPLINE_DT)
        zeros = jnp.zeros_like(slope)
        return jnp.stack([values[:-1], slope, zeros, zeros], axis=1)

    coeffs = spline_ops.rebase_to_breaks(series.breaks, series.coeffs, target_breaks)
    jump_times, jump_values = _series_jump_times_and_values(series)
    if jump_times.size == 0:
        metadata = series.metadata if isinstance(series.metadata, dict) else {}
        if "boundary_start_value" in metadata:
            coeffs = coeffs.at[0, 0].set(float(metadata["boundary_start_value"]))
        return coeffs

    jump_cumsum = jnp.concatenate(
        [jnp.asarray([0.0], dtype=float), jnp.cumsum(jump_values)]
    )
    idx = jnp.searchsorted(jump_times, target_breaks[:-1], side="right")
    coeffs = coeffs.at[:, 0].add(-jump_cumsum[idx])
    metadata = series.metadata if isinstance(series.metadata, dict) else {}
    if "boundary_start_value" in metadata:
        coeffs = coeffs.at[0, 0].set(float(metadata["boundary_start_value"]))
    return coeffs


def _evaluate_many_with_boundary_start(
    series: TimeSeries,
    times: jnp.ndarray,
    *,
    side: str = "left",
) -> jnp.ndarray:
    """Evaluate a TimeSeries, honoring optional exact-start pre-event metadata."""
    times_arr = jnp.asarray(times, dtype=float)
    values = series.evaluate_many(times_arr, side=side)
    metadata = series.metadata if isinstance(series.metadata, dict) else {}
    if "boundary_start_value" not in metadata or series.breaks is None:
        return values
    start = jnp.asarray(series.breaks[0], dtype=times_arr.dtype)
    mask = times_arr == start
    boundary = jnp.asarray(float(metadata["boundary_start_value"]), dtype=values.dtype)
    return jnp.where(mask, boundary, values)


def _evaluate_with_boundary_start(
    series: TimeSeries,
    t: jnp.ndarray,
    *,
    side: str = "left",
) -> jnp.ndarray:
    """Evaluate one point, honoring optional exact-start pre-event metadata."""
    value = series.evaluate(t, side=side)
    metadata = series.metadata if isinstance(series.metadata, dict) else {}
    if "boundary_start_value" not in metadata or series.breaks is None:
        return value
    boundary = jnp.asarray(float(metadata["boundary_start_value"]), dtype=value.dtype)
    return jnp.where(t == series.breaks[0], boundary, value)


def _build_direct_pseudobatch_series(
    process: BioProcess,
    species_name: str,
    meas_times: jnp.ndarray,
) -> Dict[str, Any]:
    """Build physical pseudobatch transform quantities as exact TimeSeries."""
    breaks = _canonical_pseudobatch_breaks(process, meas_times)
    breaks_np = np.asarray(breaks, dtype=np.float64)
    n_pieces = breaks_np.size - 1
    V_init = float(process.volume.initial_volume)
    _require_reactor_volume_scalar(V_init, context="initial reactor volume")

    continuous_feed_coeffs: dict[str, jnp.ndarray] = {}
    feed_concentrations: dict[str, float] = {}
    feed_stream_names: list[str] = []
    continuous_volume_coeffs = jnp.zeros((n_pieces, 4), dtype=float)

    for vc_name, vc in process.volume.volume_changes.items():
        if not isinstance(vc, FeedVolumeChange):
            continue
        feed_stream_names.append(vc_name)
        feed_concentrations[vc_name] = _feed_concentration_for_species(vc, species_name)
        if vc.is_continuous:
            coeffs = _series_coeffs_on_breaks(vc.values, breaks)
            continuous_feed_coeffs[vc_name] = coeffs
            continuous_volume_coeffs = continuous_volume_coeffs + coeffs

    events = _discrete_volume_events(process, species_name)
    discrete_feed_names = [
        name
        for name, vc in process.volume.volume_changes.items()
        if isinstance(vc, FeedVolumeChange) and not vc.is_continuous
    ]
    discrete_feed_current = {name: 0.0 for name in discrete_feed_names}
    discrete_feed_interval_values = {
        name: np.zeros(n_pieces, dtype=np.float64) for name in discrete_feed_names
    }

    reactor_coeffs = np.array(continuous_volume_coeffs, dtype=np.float64, copy=True)
    sample_comp_values = np.zeros(n_pieces, dtype=np.float64)
    adf_coeffs = np.zeros((n_pieces, 4), dtype=np.float64)
    feed_corr_coeffs = np.zeros((n_pieces, 4), dtype=np.float64)

    discrete_volume_delta = 0.0
    sample_compensation = 1.0
    feed_corr_current = 0.0
    feed_corr_jump_times: list[float] = []
    feed_corr_jump_values: list[float] = []

    boundary_volume = V_init
    if n_pieces > 0:
        boundary_volume += float(continuous_volume_coeffs[0, 0])
    _require_reactor_volume_scalar(
        boundary_volume,
        context="pseudobatch boundary reactor volume",
    )
    boundary_sample_comp = sample_compensation
    boundary_adf = boundary_volume * boundary_sample_comp / V_init
    boundary_feed_corr = feed_corr_current

    for i, t_i in enumerate(breaks_np[:-1]):
        event = events.get(float(t_i))
        continuous_at_break = float(continuous_volume_coeffs[i, 0])
        if event is not None:
            sample_delta = sum(delta for _, delta in event["samples"])
            if sample_delta != 0.0:
                v_before_sample = V_init + continuous_at_break + discrete_volume_delta
                v_after_sample = v_before_sample + sample_delta
                _require_reactor_volume_scalar(
                    v_before_sample,
                    context="pre-sampling reactor volume",
                )
                _require_reactor_volume_scalar(
                    v_after_sample,
                    context="post-sampling reactor volume",
                )
                sample_compensation *= v_before_sample / v_after_sample
                discrete_volume_delta += sample_delta

            for vc_name, delta_v, c_feed in event["boluses"]:
                v_after_bolus = (
                    V_init + continuous_at_break + discrete_volume_delta + delta_v
                )
                _require_reactor_volume_scalar(
                    v_after_bolus,
                    context="post-bolus reactor volume",
                )
                delta_fc = sample_compensation * delta_v * c_feed / V_init
                if delta_fc != 0.0:
                    feed_corr_jump_times.append(float(t_i))
                    feed_corr_jump_values.append(delta_fc)
                feed_corr_current += delta_fc
                discrete_volume_delta += delta_v
                if vc_name in discrete_feed_current:
                    discrete_feed_current[vc_name] += delta_v

        vol_coeff = np.asarray(continuous_volume_coeffs[i], dtype=np.float64).copy()
        vol_coeff[0] += V_init + discrete_volume_delta
        _require_volume_piece_above_threshold(
            vol_coeff,
            breaks_np[i + 1] - breaks_np[i],
            context="pseudobatch reactor volume spline",
        )
        reactor_coeffs[i] = vol_coeff
        sample_comp_values[i] = sample_compensation
        adf_coeffs[i] = vol_coeff * (sample_compensation / V_init)

        fc_coeff = np.asarray([feed_corr_current, 0.0, 0.0, 0.0], dtype=np.float64)
        for vc_name, coeffs in continuous_feed_coeffs.items():
            scale = sample_compensation * feed_concentrations[vc_name] / V_init
            fc_coeff[1:] += scale * np.asarray(coeffs[i], dtype=np.float64)[1:]
        feed_corr_coeffs[i] = fc_coeff

        width = breaks_np[i + 1] - breaks_np[i]
        a_fc, b_fc, c_fc, d_fc = fc_coeff
        feed_corr_current = float(a_fc + width * (b_fc + width * (c_fc + width * d_fc)))
        for vc_name in discrete_feed_names:
            discrete_feed_interval_values[vc_name][i] = discrete_feed_current[vc_name]

    event_times = np.asarray(sorted(events), dtype=np.float64)
    interior_event_times = event_times[
        (event_times > breaks_np[0]) & (event_times < breaks_np[-1])
    ]
    feed_corr_jump_times_arr = jnp.asarray(feed_corr_jump_times, dtype=float)

    reactor_volume_ts = _piecewise_polynomial_timeseries(
        breaks,
        jnp.asarray(reactor_coeffs, dtype=float),
        continuity_side="left",
        jump_times=jnp.asarray(interior_event_times, dtype=float),
        metadata={
            "interp": "piecewise_polynomial",
            "boundary_start_value": boundary_volume,
        },
    )
    sample_compensation_ts = _piecewise_constant_from_intervals(
        breaks,
        jnp.asarray(sample_comp_values, dtype=float),
        continuity_side="left",
        jump_times=jnp.asarray(interior_event_times, dtype=float),
        metadata={
            "interp": "piecewise_constant",
            "boundary_start_value": boundary_sample_comp,
        },
    )

    adf_ts = _piecewise_polynomial_timeseries(
        breaks,
        jnp.asarray(adf_coeffs, dtype=float),
        continuity_side="left",
        jump_times=jnp.asarray(interior_event_times, dtype=float),
        metadata={
            "interp": "piecewise_polynomial",
            "boundary_start_value": boundary_adf,
        },
    )
    adf_jump_values = _jump_values_from_series(adf_ts)
    adf_metadata = dict(adf_ts.metadata or {})
    adf_metadata["jump_values"] = np.asarray(adf_jump_values, dtype=float).tolist()
    adf_ts = _piecewise_polynomial_timeseries(
        adf_ts.breaks,
        adf_ts.coeffs,
        continuity_side="left",
        jump_times=adf_ts.jump_times,
        metadata=adf_metadata,
    )

    fc_metadata = {
        "interp": "piecewise_polynomial",
        "boundary_start_value": boundary_feed_corr,
        "jump_values": np.asarray(feed_corr_jump_values, dtype=float).tolist(),
    }
    feed_corr_ts = _piecewise_polynomial_timeseries(
        breaks,
        jnp.asarray(feed_corr_coeffs, dtype=float),
        continuity_side="left",
        jump_times=feed_corr_jump_times_arr,
        metadata=fc_metadata,
    )

    accumulated_feed_ts: dict[str, TimeSeries] = {}
    for vc_name in feed_stream_names:
        vc = process.volume.volume_changes[vc_name]
        if vc.is_continuous:
            coeffs = continuous_feed_coeffs[vc_name]
            accumulated_feed_ts[vc_name] = _piecewise_polynomial_timeseries(
                breaks,
                coeffs,
                continuity_side="left",
                metadata={"interp": "piecewise_polynomial"},
            )
        else:
            accumulated_feed_ts[vc_name] = _piecewise_constant_from_intervals(
                breaks,
                jnp.asarray(discrete_feed_interval_values[vc_name], dtype=float),
                continuity_side="left",
                jump_times=jnp.asarray(
                    [
                        t
                        for t, record in events.items()
                        if any(name == vc_name for name, _, _ in record["boluses"])
                    ],
                    dtype=float,
                ),
                metadata={"interp": "piecewise_constant"},
            )

    dense_times = breaks
    reactor_volume_dense = _evaluate_many_with_boundary_start(
        reactor_volume_ts, dense_times
    )
    sample_compensation_dense = _evaluate_many_with_boundary_start(
        sample_compensation_ts, dense_times
    )
    adf_dense = _evaluate_many_with_boundary_start(adf_ts, dense_times)
    feed_corr_dense = _evaluate_many_with_boundary_start(feed_corr_ts, dense_times)

    sample_volume_dense_np = np.zeros(breaks_np.shape[0], dtype=np.float64)
    for t_ev, record in events.items():
        sample_delta = sum(delta for _, delta in record["samples"])
        if sample_delta == 0.0:
            continue
        idx = np.where(breaks_np == float(t_ev))[0]
        if idx.size > 0:
            sample_volume_dense_np[int(idx[0])] += abs(sample_delta)

    accumulated_feed_dense_list = [
        _evaluate_many_with_boundary_start(accumulated_feed_ts[name], dense_times)
        for name in feed_stream_names
    ]
    if len(accumulated_feed_dense_list) == 0:
        accumulated_feed_dense = jnp.zeros_like(dense_times)
        concentration_in_feed = 0.0
    elif len(accumulated_feed_dense_list) == 1:
        accumulated_feed_dense = accumulated_feed_dense_list[0]
        concentration_in_feed = feed_concentrations[feed_stream_names[0]]
    else:
        accumulated_feed_dense = jnp.column_stack(accumulated_feed_dense_list)
        concentration_in_feed = jnp.asarray(
            [feed_concentrations[name] for name in feed_stream_names],
            dtype=float,
        )

    meas_np = np.asarray(meas_times, dtype=np.float64)
    meas_indices_np = np.asarray(
        [int(np.argmin(np.abs(breaks_np - float(mt)))) for mt in meas_np],
        dtype=np.int64,
    )
    meas_indices = jnp.asarray(meas_indices_np, dtype=int)
    has_discrete_feed = any(
        isinstance(vc, FeedVolumeChange) and not vc.is_continuous
        for vc in process.volume.volume_changes.values()
    )

    return {
        "dense_times": dense_times,
        "meas_indices": meas_indices,
        "reactor_volume_ts": reactor_volume_ts,
        "sample_compensation_ts": sample_compensation_ts,
        "accumulated_feed_ts": accumulated_feed_ts,
        "adf_ts": adf_ts,
        "feed_corr_ts": feed_corr_ts,
        "reactor_volume_dense": reactor_volume_dense,
        "sample_volume_dense": jnp.asarray(sample_volume_dense_np, dtype=float),
        "accumulated_feed_dense": accumulated_feed_dense,
        "concentration_in_feed": concentration_in_feed,
        "adf_dense": adf_dense,
        "feed_corr_dense": feed_corr_dense,
        "sample_compensation_dense": sample_compensation_dense,
        "has_discrete_feed": has_discrete_feed,
    }


# ===========================================================================
# Pseudobatch transform pipeline
# ===========================================================================


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

    series_inputs = _build_direct_pseudobatch_series(
        process,
        species_name,
        meas_times,
    )
    adf_at_meas = _evaluate_many_with_boundary_start(
        series_inputs["adf_ts"], meas_times
    )
    feed_corr_at_meas = _evaluate_many_with_boundary_start(
        series_inputs["feed_corr_ts"], meas_times
    )
    c_star = meas_conc * adf_at_meas - feed_corr_at_meas

    return {
        "meas_times": meas_times,
        "meas_conc": meas_conc,
        "c_star": jnp.asarray(c_star),
        "dense_times": series_inputs["dense_times"],
        "meas_indices": series_inputs["meas_indices"],
        "reactor_volume_dense": series_inputs["reactor_volume_dense"],
        "sample_volume_dense": series_inputs["sample_volume_dense"],
        "accumulated_feed_dense": series_inputs["accumulated_feed_dense"],
        "concentration_in_feed": series_inputs["concentration_in_feed"],
        "reactor_volume_ts": series_inputs["reactor_volume_ts"],
        "sample_compensation_ts": series_inputs["sample_compensation_ts"],
        "accumulated_feed_ts": series_inputs["accumulated_feed_ts"],
        "adf_ts": series_inputs["adf_ts"],
        "feed_corr_ts": series_inputs["feed_corr_ts"],
        "adf_dense": series_inputs["adf_dense"],
        "adf_at_meas": jnp.asarray(adf_at_meas),
        "feed_corr_at_meas": jnp.asarray(feed_corr_at_meas),
        "feed_corr_dense": series_inputs["feed_corr_dense"],
        "sample_compensation_dense": series_inputs["sample_compensation_dense"],
        "has_discrete_feed": series_inputs["has_discrete_feed"],
    }


def build_splines(
    inputs: Dict[str, Any],
    process: "BioProcess | None" = None,
    species_name: "str | None" = None,
    *,
    cstar_smoothing_s: float = 0.0,
) -> Dict[str, Any]:
    """
    Build the runtime pseudobatch spline payload from
    ``build_pseudobatch_inputs``.

    The feed-correction and ADF trajectories are represented by canonical
    ``TimeSeries`` payloads built directly from physical volume/feed events.
    No epsilon-offset dense event grid is used.

    Returns a runtime payload dict consumed by
    ``evaluate_real_concentration``, ``to_timeseries``,
    ``BacktransformSpline``, and the mechanistic pseudobatch path.
    """
    c_star_ts = fit_timeseries_spline(
        TimeSeries(times=inputs["meas_times"], values=inputs["c_star"]),
        smoothing_s=cstar_smoothing_s,
    )
    spline_cstar = _ppoly_from_spline_state(c_star_ts)

    meas_times = jnp.array(inputs["meas_times"])
    adf_at_meas = jnp.array(inputs["adf_at_meas"])
    fc_at_meas = jnp.array(inputs["feed_corr_at_meas"])

    has_discrete_feed = bool(inputs.get("has_discrete_feed", False))
    spline_feed_corr = None
    if not has_discrete_feed:
        spline_feed_corr = make_cubic_ppoly(
            inputs["meas_times"], inputs["feed_corr_at_meas"]
        )

    if "adf_ts" in inputs and "feed_corr_ts" in inputs:
        adf_ts = inputs["adf_ts"]
        feed_corr_ts = inputs["feed_corr_ts"]
        adf_jump_values = _jump_values_from_series(adf_ts)
        feed_corr_jump_values = _jump_values_from_series(feed_corr_ts)
        return {
            "spline_cstar": spline_cstar,
            "spline_feed_corr": spline_feed_corr,
            "meas_times": meas_times,
            "adf_at_meas": adf_at_meas,
            "feed_corr_at_meas": fc_at_meas,
            "adf_ts": adf_ts,
            "feed_corr_ts": feed_corr_ts,
            "feed_corr_base_times": jnp.asarray(feed_corr_ts.times, dtype=float),
            "feed_corr_base_values": jnp.asarray(feed_corr_ts.values, dtype=float),
            "feed_corr_jump_times": jnp.asarray(feed_corr_ts.jump_times, dtype=float),
            "feed_corr_jump_values": feed_corr_jump_values,
            "adf_base_times": jnp.asarray(adf_ts.times, dtype=float),
            "adf_base_values": jnp.asarray(adf_ts.values, dtype=float),
            "adf_jump_times": jnp.asarray(adf_ts.jump_times, dtype=float),
            "adf_jump_values": adf_jump_values,
            # Compatibility arrays are now evaluations of the canonical
            # TimeSeries on the canonical no-epsilon break grid.
            "adf_times": jnp.asarray(inputs["dense_times"], dtype=float),
            "adf_values": jnp.asarray(inputs["adf_dense"], dtype=float),
            "dense_times": inputs["dense_times"],
            "adf_dense": inputs["adf_dense"],
        }

    raise ValueError(
        "build_splines requires TimeSeries-first pseudobatch inputs. "
        "Regenerate inputs with build_pseudobatch_inputs."
    )


def _cstar_metadata(
    species_name: str,
    *,
    fit_strategy: str,
    is_constant: bool,
    constant_value: float | None,
) -> dict:
    """Build lightweight pseudobatch provenance metadata for c* carriers."""
    return {
        "transform": {
            "name": "pseudo_batch",
            "species": species_name,
            "cstar_fit_strategy": fit_strategy,
            "is_constant": bool(is_constant),
            "constant_value": constant_value,
        }
    }


def _build_cstar_timeseries(
    meas_times: jnp.ndarray,
    c_star: jnp.ndarray,
    *,
    species_name: str,
    is_constant: bool,
    constant_value: float | None,
    smoothing_s: float = 0.0,
) -> TimeSeries:
    """Fit the transformed concentration carrier as a TimeSeries spline."""
    fitted = fit_timeseries_spline(
        TimeSeries(times=meas_times, values=c_star),
        smoothing_s=smoothing_s,
    )
    return dataclasses.replace(
        fitted,
        metadata={
            **(fitted.metadata or {}),
            **_cstar_metadata(
                species_name,
                fit_strategy=fitted.metadata["fit_strategy"],
                is_constant=is_constant,
                constant_value=constant_value,
            ),
        },
    )


def _assert_same_timeseries(
    left: TimeSeries,
    right: TimeSeries,
    *,
    name: str,
    species_name: str,
) -> None:
    """Fail if two shared process-level TimeSeries differ materially."""

    def _allclose_for_dtype(left_value, right_value) -> bool:
        left_arr = np.asarray(left_value)
        right_arr = np.asarray(right_value)
        dtype = np.result_type(left_arr.dtype, right_arr.dtype)
        if np.issubdtype(dtype, np.floating):
            tol = max(
                _FLOAT_EQ_ATOL,
                _FLOAT32_TOL_MULTIPLIER * float(np.finfo(dtype).eps),
            )
            left_arr = left_arr.astype(dtype, copy=False)
            right_arr = right_arr.astype(dtype, copy=False)
        else:
            tol = _FLOAT_EQ_ATOL
        return bool(np.allclose(left_arr, right_arr, rtol=tol, atol=tol))

    for attr in ("breaks", "coeffs", "jump_times"):
        left_arr = getattr(left, attr, None)
        right_arr = getattr(right, attr, None)
        if left_arr is None and right_arr is None:
            continue
        if left_arr is None or right_arr is None:
            raise ValueError(
                f"Shared {name} TimeSeries mismatch for species {species_name!r}."
            )
        if not _allclose_for_dtype(left_arr, right_arr):
            raise ValueError(
                f"Shared {name} TimeSeries mismatch for species {species_name!r}."
            )

    left_meta = left.metadata if isinstance(left.metadata, dict) else {}
    right_meta = right.metadata if isinstance(right.metadata, dict) else {}
    left_jumps = np.asarray(left_meta.get("jump_values", []), dtype=float)
    right_jumps = np.asarray(right_meta.get("jump_values", []), dtype=float)
    if left_jumps.shape != right_jumps.shape or not _allclose_for_dtype(
        left_jumps, right_jumps
    ):
        raise ValueError(
            f"Shared {name} jump metadata mismatch for species {species_name!r}."
        )


def _assert_same_shared_series(
    reference: Dict[str, Any],
    current: Dict[str, Any],
    *,
    species_name: str,
) -> None:
    """Verify species-independent pseudobatch series stayed shared."""
    for key in ("adf_ts", "reactor_volume_ts", "sample_compensation_ts"):
        _assert_same_timeseries(
            reference[key],
            current[key],
            name=key,
            species_name=species_name,
        )

    ref_feed = reference["accumulated_feed_ts"]
    cur_feed = current["accumulated_feed_ts"]
    if set(ref_feed) != set(cur_feed):
        raise ValueError(
            f"Shared accumulated feed streams mismatch for species {species_name!r}."
        )
    for feed_name in ref_feed:
        _assert_same_timeseries(
            ref_feed[feed_name],
            cur_feed[feed_name],
            name=f"accumulated_feed_ts[{feed_name!r}]",
            species_name=species_name,
        )


def _selected_pseudobatch_species(
    process: BioProcess,
    species_names: Optional[List[str]],
) -> list[str]:
    """Return reactor-medium species with TimeSeries concentrations."""

    def _assert_raw_concentration(name: str, ts: TimeSeries) -> None:
        metadata = ts.metadata if isinstance(ts.metadata, dict) else {}
        transform = metadata.get("transform")
        if isinstance(transform, dict) and transform.get("name") == "pseudo_batch":
            raise ValueError(
                f"Species {name!r} concentration already carries pseudobatch "
                "c* metadata. build_pseudobatch_transform expects real measured "
                "concentrations; use the existing process.pseudobatch_transform "
                "or restore raw concentrations before rebuilding."
            )

    if species_names is None:
        selected = []
        for name, component in process.reactor_medium.components.items():
            if isinstance(component.concentration, TimeSeries):
                _assert_raw_concentration(name, component.concentration)
                selected.append(name)
        return selected

    selected = list(species_names)
    for name in selected:
        if name not in process.reactor_medium.components:
            raise KeyError(name)
        concentration = process.reactor_medium.components[name].concentration
        if not isinstance(concentration, TimeSeries):
            raise TypeError(f"{name} must have a TimeSeries concentration")
        _assert_raw_concentration(name, concentration)
    return selected


def build_pseudobatch_transform(
    process: BioProcess,
    species_names: Optional[List[str]] = None,
    *,
    cstar_smoothing_s: float = 0.0,
) -> PseudobatchTransform:
    """Build a shared process-level pseudobatch transform bundle."""
    selected_species = _selected_pseudobatch_species(process, species_names)
    if not selected_species:
        raise ValueError("No TimeSeries reactor-medium species selected.")

    species_times: dict[str, jnp.ndarray] = {}
    shared_times: list[float] = []
    for name in selected_species:
        ts = process.reactor_medium.components[name].concentration
        if not isinstance(ts, TimeSeries):
            raise TypeError(f"{name} must have a TimeSeries concentration")
        meas_times = _series_reference_times(ts)
        species_times[name] = meas_times
        shared_times.extend(float(t) for t in np.asarray(meas_times, dtype=float))
    shared_meas_times = jnp.asarray(sorted(set(shared_times)), dtype=float)

    reference_inputs = None
    species_transforms: dict[str, PseudobatchSpeciesTransform] = {}
    for name in selected_species:
        component = process.reactor_medium.components[name]
        ts = component.concentration
        if not isinstance(ts, TimeSeries):
            raise TypeError(f"{name} must have a TimeSeries concentration")

        meas_times = species_times[name]
        meas_conc = _evaluate_timeseries_on_grid(ts, meas_times)
        series_inputs = _build_direct_pseudobatch_series(
            process,
            name,
            shared_meas_times,
        )
        if reference_inputs is None:
            reference_inputs = series_inputs
        else:
            _assert_same_shared_series(
                reference_inputs,
                series_inputs,
                species_name=name,
            )

        adf_at_meas = _evaluate_many_with_boundary_start(
            series_inputs["adf_ts"], meas_times
        )
        feed_corr_at_meas = _evaluate_many_with_boundary_start(
            series_inputs["feed_corr_ts"], meas_times
        )
        c_star = jnp.asarray(meas_conc * adf_at_meas - feed_corr_at_meas)
        is_constant = _is_near_constant(meas_conc) and not bool(
            series_inputs["has_discrete_feed"]
        )
        constant_value = float(jnp.mean(meas_conc)) if is_constant else None
        c_star_ts = _build_cstar_timeseries(
            meas_times,
            c_star,
            species_name=name,
            is_constant=is_constant,
            constant_value=constant_value,
            smoothing_s=cstar_smoothing_s,
        )
        species_transforms[name] = PseudobatchSpeciesTransform(
            species=name,
            c_star_ts=c_star_ts,
            feed_corr_ts=series_inputs["feed_corr_ts"],
            is_constant=is_constant,
            constant_value=constant_value,
            cstar_fit_strategy=c_star_ts.metadata["fit_strategy"],
        )

    assert reference_inputs is not None
    return PseudobatchTransform(
        adf_ts=reference_inputs["adf_ts"],
        reactor_volume_ts=reference_inputs["reactor_volume_ts"],
        sample_compensation_ts=reference_inputs["sample_compensation_ts"],
        accumulated_feed_ts=reference_inputs["accumulated_feed_ts"],
        species=species_transforms,
    )


def evaluate_real_concentration(
    t_eval: jnp.ndarray, splines: Dict[str, Any]
) -> jnp.ndarray:
    """
    Backtransform c*(t) -> c(t) at arbitrary evaluation times t_eval.

    Uses:
      - c*(t) via the fitted owned PPoly spline
      - ADF via canonical ``TimeSeries.evaluate``
      - feed_correction via canonical ``TimeSeries.evaluate``

    Returns:
      array of backtransformed concentrations evaluated at t_eval
    """
    t_eval = jnp.asarray(t_eval, dtype=float)
    cs = jnp.asarray(splines["spline_cstar"](jnp.asarray(t_eval)))

    if "adf_ts" not in splines or "feed_corr_ts" not in splines:
        raise ValueError(
            "Pseudobatch backtransform requires canonical TimeSeries transform "
            "entries 'adf_ts' and 'feed_corr_ts'."
        )

    adf_ts = splines["adf_ts"]
    if t_eval.ndim == 0:
        adf = _evaluate_with_boundary_start(adf_ts, t_eval, side="left")
    else:
        adf = _evaluate_many_with_boundary_start(adf_ts, t_eval)

    feed_corr_ts = splines["feed_corr_ts"]
    if t_eval.ndim == 0:
        fc = _evaluate_with_boundary_start(feed_corr_ts, t_eval, side="left")
    else:
        fc = _evaluate_many_with_boundary_start(feed_corr_ts, t_eval)

    adf = _adf_for_division(adf)

    return (cs + fc) / adf


# ===========================================================================
# TimeSeries conversion and evaluation
# ===========================================================================


def to_timeseries(
    inputs: Dict[str, Any],
    splines: Dict[str, Any],
    species_name: str,
    *,
    cstar_smoothing_s: float = 0.0,
) -> TimeSeries:
    """Convert pseudobatch pipeline outputs to a TimeSeries-first carrier."""
    meas_conc = jnp.asarray(inputs["meas_conc"], dtype=float)
    has_discrete = bool(inputs.get("has_discrete_feed", False))
    is_constant = _is_near_constant(meas_conc) and not has_discrete
    return _build_cstar_timeseries(
        inputs["meas_times"],
        inputs["c_star"],
        species_name=species_name,
        is_constant=is_constant,
        constant_value=float(jnp.mean(meas_conc)) if is_constant else None,
        smoothing_s=cstar_smoothing_s,
    )


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

    c_star_spline: PPoly
    adf_ts: TimeSeries
    feed_corr_ts: TimeSeries
    dadf_ts: TimeSeries
    dfc_ts: TimeSeries
    adf_times: jnp.ndarray
    adf_values: jnp.ndarray
    adf_jump_times: jnp.ndarray
    adf_jump_values: jnp.ndarray
    fc_times: jnp.ndarray
    fc_values: jnp.ndarray
    fc_jump_times: jnp.ndarray
    fc_jump_values: jnp.ndarray
    is_constant: bool = eqx.field(static=True)
    constant_value: jnp.ndarray

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate backtransformed concentration at time(s) *t*."""
        if self.is_constant:
            return self.constant_value + t * 0.0  # keep JAX tracing happy
        cs = self.c_star_spline(t)
        adf = _evaluate_with_boundary_start(self.adf_ts, t, side="left")
        fc = _evaluate_with_boundary_start(self.feed_corr_ts, t, side="left")
        adf = _adf_for_division(adf)
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

        ``dc^*/dt`` uses the analytical c* spline derivative.
        ``dfc/dt`` and ``d(ADF)/dt`` use the derivative TimeSeries, ignoring
        instantaneous jump impulses.
        """
        dc_star = self.c_star_spline.derivative()

        def _deriv(t):
            if self.is_constant:
                return t * 0.0
            adf = _evaluate_with_boundary_start(self.adf_ts, t, side="left")
            dadf_dt = self.dadf_ts.evaluate(t, side="left")
            adf = _adf_for_division(adf)
            dc_star_dt = dc_star(t)
            dfc_dt = self.dfc_ts.evaluate(t, side="left")
            # c(t) = (cs + fc) / adf; derive dc/dt from the evaluated c(t)
            fc = _evaluate_with_boundary_start(self.feed_corr_ts, t, side="left")
            cs = self.c_star_spline(t)
            c_val = (cs + fc) / adf
            return (dc_star_dt + dfc_dt - c_val * dadf_dt) / adf

        return _deriv


def build_backtransform_spline(
    transform: PseudobatchTransform,
    species_name: str,
) -> BacktransformSpline:
    """Build a JIT-compatible backtransform from a pseudobatch bundle.

    This is meant to be called **once** (outside JIT).  The returned module
    can then be passed into ``eqx.filter_jit``-compiled functions.

    Parameters
    ----------
    transform:
        Process-level pseudobatch transform bundle.
    species_name:
        Reactor-medium species name stored in ``transform.species``.

    Returns
    -------
    BacktransformSpline
    """
    if species_name not in transform.species:
        raise KeyError(species_name)
    species_transform = transform.species[species_name]
    if species_transform.species != species_name:
        raise ValueError(
            f"Pseudobatch species key {species_name!r} does not match stored "
            f"species {species_transform.species!r}."
        )

    rep = species_transform.c_star_ts
    if not _has_spline_state(rep):
        raise ValueError(
            "Pseudobatch c_star_ts must carry stored spline state; "
            "regenerate the pseudobatch transform."
        )
    xi_source = rep.times if rep.times is not None else rep.breaks
    xi = jnp.asarray(xi_source, dtype=float)

    is_constant = bool(species_transform.is_constant)
    constant_value = jnp.array(species_transform.constant_value or 0.0)

    c_star_spline = _ppoly_from_spline_state(rep)

    adf_ts = transform.adf_ts
    feed_corr_ts = species_transform.feed_corr_ts

    adf_times = jnp.asarray(adf_ts.times, dtype=float)
    adf_values = _timeseries_base_values_without_jumps(adf_ts)
    adf_jump_times, adf_jump_values = _series_jump_times_and_values(adf_ts)

    fc_interp = _timeseries_interp_mode(
        feed_corr_ts,
        "piecewise_polynomial",
    )
    fc_times = jnp.asarray(feed_corr_ts.times, dtype=float)
    fc_values = _timeseries_base_values_without_jumps(feed_corr_ts)
    if fc_interp in {"piecewise_polynomial", "piecewise_constant"}:
        fc_jump_times, fc_jump_values = _series_jump_times_and_values(feed_corr_ts)
    elif fc_interp == "cubic":
        fc_jump_times = jnp.zeros(0, dtype=float)
        fc_jump_values = jnp.zeros(0, dtype=float)
    elif fc_interp in {"linear", "linear_plus_step"}:
        raise ValueError(
            f"Legacy feed_corr_interp={fc_interp!r} unsupported for "
            "pseudobatch backtransform; regenerate transformed TimeSeries "
            "payloads."
        )
    else:
        raise ValueError(
            f"Unknown feed_corr_interp={fc_interp!r}; expected 'cubic', "
            "'piecewise_polynomial', or 'piecewise_constant'."
        )
    dadf_ts = (
        adf_ts.deriv()
        if _has_spline_state(adf_ts)
        else _constant_timeseries(0.0, float(xi[0]), float(xi[-1]))
    )
    dfc_ts = (
        feed_corr_ts.deriv()
        if _has_spline_state(feed_corr_ts)
        else _constant_timeseries(0.0, float(xi[0]), float(xi[-1]))
    )

    return BacktransformSpline(
        c_star_spline=c_star_spline,
        adf_ts=adf_ts,
        feed_corr_ts=feed_corr_ts,
        dadf_ts=dadf_ts,
        dfc_ts=dfc_ts,
        adf_times=adf_times,
        adf_values=adf_values,
        adf_jump_times=adf_jump_times,
        adf_jump_values=adf_jump_values,
        fc_times=fc_times,
        fc_values=fc_values,
        fc_jump_times=fc_jump_times,
        fc_jump_values=fc_jump_values,
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
    batched owned ``PPoly`` objects.  A single call evaluates all N
    species simultaneously, replacing the N separate Python-loop calls
    that dominate ODE RHS cost.

    Build with :func:`build_batched_conc_splines`.
    """

    c_star_ppoly: PPoly  # coeff shape (m, 4, n_sp)
    fc_ppoly: PPoly  # coeff shape (m, 4, n_sp)
    adf_deriv_ppoly: PPoly
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
        """Evaluate all species concentrations at scalar or vector time *t*.

        Returns
        -------
        jnp.ndarray
            Shape ``(n_sp,)`` for scalar ``t`` or ``(n_t, n_sp)`` for
            one-dimensional vector ``t``.
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
        adf = _adf_for_division(adf)[..., jnp.newaxis]
        result = (cs + fc) / adf
        return jnp.where(self.constant_mask, self.constant_values, result)

    def eval_derivative(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate dc/dt for all species at scalar or vector time *t*.

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
            Shape ``(n_sp,)`` for scalar ``t`` or ``(n_t, n_sp)`` for
            one-dimensional vector ``t``.
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
        adf = _adf_for_division(adf)[..., jnp.newaxis]
        # Derivative ignores instantaneous jumps and follows exact smooth pieces.
        dadf_dt = self.adf_deriv_ppoly(t)[..., jnp.newaxis]
        c_val = (cs + fc) / adf
        result = (dc_star + dfc - c_val * dadf_dt) / adf
        return jnp.where(self.constant_mask, 0.0, result)


def build_batched_conc_splines(
    transform: PseudobatchTransform,
    species_names: Optional[List[str]] = None,
    t_start: float | None = None,
    t_end: float | None = None,
    n_knots: int = _DEFAULT_BATCH_KNOTS,
):
    """Build a :class:`BatchedBacktransformSpline` from a transform bundle.

    The bundle supplies one shared ADF TimeSeries and one c*/feed-correction
    pair per species.

    Parameters
    ----------
    transform : PseudobatchTransform
        Process-level pseudobatch transform bundle.
    species_names : list[str] | None
        Ordered species names (determines column order in batched arrays).
    t_start, t_end : float | None
        Time range for resampling.
    n_knots : int
        Number of uniformly-spaced knots for resampling (default 128).

    Returns
    -------
    BatchedBacktransformSpline
    """
    if species_names is None:
        species_names = list(transform.species)
    else:
        species_names = list(species_names)
    conc_splines = {
        name: build_backtransform_spline(transform, name) for name in species_names
    }
    if t_start is None:
        t_start = float(transform.adf_ts.breaks[0])
    if t_end is None:
        t_end = float(transform.adf_ts.breaks[-1])

    common_points = [float(t) for t in np.linspace(t_start, t_end, n_knots)]
    for sp_name in species_names:
        sp = conc_splines[sp_name]
        if isinstance(sp, BacktransformSpline):
            if sp.adf_ts.breaks is not None:
                common_points.extend(float(t) for t in np.asarray(sp.adf_ts.breaks))
            if sp.feed_corr_ts.breaks is not None:
                common_points.extend(
                    float(t) for t in np.asarray(sp.feed_corr_ts.breaks)
                )
            common_points.extend(float(t) for t in np.asarray(sp.adf_jump_times))
            common_points.extend(float(t) for t in np.asarray(sp.fc_jump_times))
    common_np = np.asarray(sorted(set(common_points)), dtype=np.float64)
    common_np = common_np[(common_np >= float(t_start)) & (common_np <= float(t_end))]
    if common_np.size < 2:
        common_np = np.asarray([float(t_start), float(t_end)], dtype=np.float64)
    x_common = jnp.asarray(common_np, dtype=float)
    n_knots = int(x_common.shape[0])
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
    adf_deriv_coeffs = None
    fc_ppoly_coeffs = []

    for sp_name in species_names:
        sp = conc_splines[sp_name]

        if isinstance(sp, BacktransformSpline):
            if adf_times is None:
                adf_times = x_common
                adf_values = _baseline_values_on_grid(sp.adf_ts, x_common)
                adf_base_coeffs = _baseline_coeffs_on_breaks(sp.adf_ts, x_common)
                adf_deriv_coeffs = spline_ops.derivative_coeffs(adf_base_coeffs)
                adf_jump_times, adf_jump_values = _series_jump_times_and_values(
                    sp.adf_ts
                )
                adf_jump_cumsum = jnp.cumsum(
                    jnp.concatenate(
                        [
                            jnp.asarray([0.0], dtype=float),
                            jnp.asarray(adf_jump_values, dtype=float),
                        ]
                    )
                )
            else:
                other_jump_times, other_jump_values = _series_jump_times_and_values(
                    sp.adf_ts
                )
                other_base_coeffs = _baseline_coeffs_on_breaks(sp.adf_ts, x_common)
                other_jump_cumsum = jnp.cumsum(
                    jnp.concatenate(
                        [
                            jnp.asarray([0.0], dtype=float),
                            jnp.asarray(other_jump_values, dtype=float),
                        ]
                    )
                )
                if not (
                    jnp.array_equal(adf_times, x_common)
                    and jnp.allclose(
                        adf_values,
                        _baseline_values_on_grid(sp.adf_ts, x_common),
                        atol=_FLOAT_EQ_ATOL,
                    )
                    and jnp.array_equal(adf_jump_times, other_jump_times)
                    and jnp.allclose(
                        adf_jump_cumsum,
                        other_jump_cumsum,
                        atol=_FLOAT_EQ_ATOL,
                    )
                    and jnp.allclose(
                        adf_deriv_coeffs,
                        spline_ops.derivative_coeffs(other_base_coeffs),
                        atol=_FLOAT_EQ_ATOL,
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
                fc_ppoly_coeffs.append(jnp.zeros((n_knots - 1, 4), dtype=float))
                fc_step_mask_list.append(False)
                fc_jump_times_list.append(jnp.zeros(0, dtype=float))
                fc_jump_values_list.append(jnp.zeros(0, dtype=float))
            else:
                c_star_resampled.append(sp.c_star_spline(x_common))
                fc_coeffs = _baseline_coeffs_on_breaks(sp.feed_corr_ts, x_common)
                fc_resampled.append(fc_coeffs[:, 0])
                fc_ppoly_coeffs.append(fc_coeffs)
                fc_jump_times, fc_jump_values = _series_jump_times_and_values(
                    sp.feed_corr_ts
                )
                has_jump = int(fc_jump_times.shape[0]) > 0
                fc_step_mask_list.append(has_jump)
                fc_jump_times_list.append(fc_jump_times)
                fc_jump_values_list.append(fc_jump_values)
        else:
            # Plain callable: treat as c*=spline, fc=0, ADF=1
            constant_mask_list.append(False)
            constant_values_list.append(0.0)
            c_star_resampled.append(sp(x_common))
            fc_resampled.append(jnp.zeros(n_knots))
            fc_ppoly_coeffs.append(jnp.zeros((n_knots - 1, 4), dtype=float))
            fc_step_mask_list.append(False)
            fc_jump_times_list.append(jnp.zeros(0, dtype=float))
            fc_jump_values_list.append(jnp.zeros(0, dtype=float))

    # If no BacktransformSpline was found, use trivial ADF
    if adf_times is None:
        adf_times = jnp.array([t_start, t_end])
        adf_values = jnp.array([1.0, 1.0])
        adf_jump_times = jnp.zeros(0, dtype=float)
        adf_jump_cumsum = jnp.asarray([0.0], dtype=float)
        adf_deriv_coeffs = jnp.zeros((n_knots - 1, 4), dtype=float)
    adf_deriv_ppoly = PPoly(x_common, adf_deriv_coeffs)

    # Build batched PPoly for c* splines.
    c_star_cubic = [make_cubic_ppoly(x_common, y) for y in c_star_resampled]
    c_star_c = jnp.stack([s.coeffs for s in c_star_cubic], axis=-1)
    c_star_ppoly = PPoly(x_common, c_star_c)

    # Build batched PPoly for the exact jump-free feed-correction baseline.
    fc_c = jnp.stack(fc_ppoly_coeffs, axis=-1)
    fc_ppoly = PPoly(x_common, fc_c)

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
            valid &= np.isclose(shared_jump_times[idx], jt_np, atol=_FLOAT_EQ_ATOL)
            np.add.at(jump_matrix[i], idx[valid], jv_np[valid])
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
        adf_deriv_ppoly=adf_deriv_ppoly,
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
