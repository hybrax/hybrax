"""Dict serialization for TimeSeries."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


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
