"""Serialization and input adapters for TimeSeries."""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BSpline, PPoly as SciPyPPoly

from .ppoly import PPoly


def _array_to_list(value: Any) -> Any:
    if value is None:
        return None
    return np.asarray(value).tolist()


def timeseries_to_dict(series: Any) -> dict[str, Any]:
    """Serialize a TimeSeries instance to canonical dict form."""
    return {
        "times": _array_to_list(series.times),
        "values": _array_to_list(series.values),
        "derived": bool(series.derived),
        "jump_times": _array_to_list(series.jump_times),
        "breaks": _array_to_list(series.breaks),
        "coeffs": _array_to_list(series.coeffs),
        "segment_start_piece_idx": _array_to_list(series.segment_start_piece_idx),
        "continuity_side": series.continuity_side,
        "metadata": series.metadata,
    }


def timeseries_from_dict(cls: Any, data: Mapping[str, Any]) -> Any:
    """Construct a TimeSeries from canonical dict form.

    All floating-point fields are float64 (x64 enabled package-wide). A legacy
    ``"dtype"`` key in older serialized payloads is ignored.
    """
    kwargs = dict(
        times=data.get("times"),
        values=data.get("values"),
        derived=bool(data.get("derived", False)),
        jump_times=data.get("jump_times"),
        breaks=data.get("breaks"),
        coeffs=data.get("coeffs"),
        segment_start_piece_idx=data.get("segment_start_piece_idx"),
        continuity_side=data.get("continuity_side", "right"),
        metadata=data.get("metadata"),
    )
    return cls(**kwargs)


def _convert_bspline_segments(
    segments: list[Mapping[str, Any]],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert input-json segment B-spline representation to canonical arrays."""
    all_breaks: list[np.ndarray] = []
    all_coeffs: list[np.ndarray] = []
    segment_starts: list[int] = []
    piece_cursor = 0

    for segment in segments:
        knots = np.asarray(segment["knots"], dtype=np.float64)
        coeffs = np.asarray(segment["coeffs"], dtype=np.float64)
        degree = int(segment.get("degree", 3))
        if degree != 3:
            raise ValueError("Only cubic segments are supported")
        spline = BSpline(knots, coeffs, degree, extrapolate=True)
        seg_poly = PPoly.from_scipy_ppoly(SciPyPPoly.from_spline(spline))
        seg_breaks = np.asarray(seg_poly.breaks, dtype=np.float64)
        seg_coeffs = np.asarray(seg_poly.coeffs, dtype=np.float64)
        if seg_coeffs.shape[0] == 0:
            continue
        segment_starts.append(piece_cursor)
        piece_cursor += int(seg_coeffs.shape[0])
        if not all_breaks:
            all_breaks.append(seg_breaks)
        else:
            if np.isclose(all_breaks[-1][-1], seg_breaks[0]):
                all_breaks.append(seg_breaks[1:])
            else:
                all_breaks.append(seg_breaks)
        all_coeffs.append(seg_coeffs)

    if not all_coeffs:
        raise ValueError("No valid spline pieces found in segments")

    flat_breaks = np.concatenate(all_breaks, axis=0).astype(np.float64)
    flat_coeffs = np.concatenate(all_coeffs, axis=0).astype(np.float64)
    start_idx = np.asarray(segment_starts, dtype=np.int32)
    return (
        jnp.asarray(flat_breaks, dtype=jnp.float64),
        jnp.asarray(flat_coeffs, dtype=jnp.float64),
        jnp.asarray(start_idx, dtype=jnp.int32),
    )


def _extract_jump_times(
    detected_jumps: Mapping[str, Any],
    variable: str,
) -> list[float]:
    """Extract one variable's jump times from a ``detected_jumps`` mapping."""
    raw = detected_jumps.get(variable, [])
    out: list[float] = []
    for item in raw:
        if isinstance(item, Mapping):
            out.append(float(item["time"]))
        else:
            out.append(float(item))
    return out


def timeseries_from_process_state(
    cls: Any,
    process_state: Mapping[str, Any],
    variable: str,
) -> Any:
    """Build one TimeSeries from metadata.hybrax-format.process_state for a variable."""
    spline_results = process_state["spline_results"][variable]
    detected_jumps = process_state.get("detected_jumps", {})
    jumps = _extract_jump_times(detected_jumps, variable)

    segments = spline_results.get("segments", [])
    times = spline_results.get("smoothed_times")
    values = spline_results.get("smoothed_values")
    if times is None and segments:
        times = [time for segment in segments for time in segment["smoothed_times"]]
    if values is None and segments:
        values = [value for segment in segments for value in segment["smoothed_values"]]
    breaks = None
    coeffs = None
    segment_start_piece_idx = None
    if segments:
        breaks, coeffs, segment_start_piece_idx = _convert_bspline_segments(segments)

    metadata = {
        "source": "metadata.hybrax-format.process_state",
        "variable": variable,
        "k": spline_results.get("k"),
        "s": spline_results.get("s"),
    }

    return cls(
        times=times,
        values=values,
        derived=False,
        jump_times=jumps,
        breaks=breaks,
        coeffs=coeffs,
        segment_start_piece_idx=segment_start_piece_idx,
        continuity_side="right",
        metadata=metadata,
    )


def timeseries_from_input_dict(
    cls: Any,
    input_data: Mapping[str, Any],
    process_key: str,
    variable: str,
) -> Any:
    """Build one TimeSeries from full input.json-like data."""
    process_state = input_data["metadata"]["hybrax-format"]["process_state"][process_key]
    return timeseries_from_process_state(cls, process_state, variable)
