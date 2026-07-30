"""Stable API tests for bp_format TimeSeries behavior."""

from __future__ import annotations

import numpy as np
import pytest

import bp_format
from bp_format.time_series import TimeSeries


def test_model_requires_representation_and_matching_discrete_fields() -> None:
    with pytest.raises(ValueError, match="Provide discrete samples and/or spline"):
        TimeSeries()

    with pytest.raises(ValueError, match="times and values must either both"):
        TimeSeries(times=[0.0, 1.0])

    with pytest.raises(ValueError, match="times and values must either both"):
        TimeSeries(values=[1.0, 2.0])


def test_model_validates_discrete_and_spline_invariants() -> None:
    with pytest.raises(ValueError, match="times must be strictly increasing"):
        TimeSeries(times=[0.0, 0.0, 1.0], values=[1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match=r"len\(breaks\) must be len\(coeffs\) \+ 1"):
        TimeSeries(
            breaks=[0.0, 1.0, 2.0],
            coeffs=[[1.0, 2.0, 3.0, 4.0]],
            segment_start_piece_idx=[0],
        )


def _const_piece_series() -> TimeSeries:
    return TimeSeries(
        breaks=[0.0, 1.0, 2.0],
        coeffs=[[1.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]],
        segment_start_piece_idx=[0, 1],
        jump_times=[1.0],
    )


def test_eval_continuity_side_behavior_at_internal_break() -> None:
    ts = _const_piece_series()
    assert float(ts.evaluate(1.0, side="right")) == pytest.approx(10.0)
    assert float(ts.evaluate(1.0, side="left")) == pytest.approx(1.0)


def test_io_round_trip_preserves_canonical_fields() -> None:
    original = TimeSeries(
        times=[0.0, 1.0, 2.0],
        values=[3.0, 4.0, 5.0],
        derived=True,
        jump_times=[1.0],
        breaks=[0.0, 1.0, 2.0],
        coeffs=[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
        segment_start_piece_idx=[0],
        continuity_side="left",
        metadata={"source": "api-test"},
    )

    payload = original.to_dict()
    rebuilt = TimeSeries.from_dict(payload)

    np.testing.assert_allclose(np.asarray(rebuilt.times), np.asarray(original.times))
    np.testing.assert_allclose(np.asarray(rebuilt.values), np.asarray(original.values))
    np.testing.assert_allclose(np.asarray(rebuilt.breaks), np.asarray(original.breaks))
    np.testing.assert_allclose(np.asarray(rebuilt.coeffs), np.asarray(original.coeffs))
    np.testing.assert_array_equal(
        np.asarray(rebuilt.segment_start_piece_idx),
        np.asarray(original.segment_start_piece_idx),
    )
    assert rebuilt.derived is True
    assert rebuilt.continuity_side == "left"
    assert rebuilt.metadata == {"source": "api-test"}


def test_process_state_adapter_uses_segment_samples_without_aggregates() -> None:
    process_state = {
        "spline_results": {
            "X": {
                "segments": [
                    {
                        "knots": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                        "coeffs": [0.0, 1.0, 2.0, 3.0],
                        "degree": 3,
                        "smoothed_times": [0.0, 0.5, 1.0],
                        "smoothed_values": [0.0, 1.5, 3.0],
                    }
                ]
            }
        }
    }

    series = TimeSeries.from_process_state(process_state, "X")

    np.testing.assert_allclose(np.asarray(series.times), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(np.asarray(series.values), [0.0, 1.5, 3.0])


def test_exact_add_matches_pointwise_eval() -> None:
    a = TimeSeries(
        times=[0.0, 1.0, 2.0],
        values=[1.0, 2.0, 3.0],
        breaks=[0.0, 1.0, 2.0],
        coeffs=[[1.0, 2.0, 0.0, 0.0], [5.0, 1.0, 0.0, 0.0]],
        segment_start_piece_idx=[0],
        jump_times=[1.0],
    )
    b = TimeSeries(
        times=[0.0, 0.5, 2.0],
        values=[4.0, 5.0, 6.0],
        breaks=[0.0, 0.5, 2.0],
        coeffs=[[3.0, 0.0, 0.0, 0.0], [7.0, -2.0, 0.0, 0.0]],
        segment_start_piece_idx=[0, 1],
        jump_times=[0.5],
    )

    added = a + b
    probe = np.asarray([0.0, 0.25, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    np.testing.assert_allclose(
        np.asarray(added.evaluate_many(probe)),
        np.asarray(a.evaluate_many(probe)) + np.asarray(b.evaluate_many(probe)),
        atol=1e-6,
        rtol=0.0,
    )
    assert added.times is not None
    assert added.values is not None


def test_public_bp_format_timeseries_rejects_legacy_timepoints_constructor() -> None:
    assert bp_format.TimeSeries is TimeSeries
    with pytest.raises(TypeError):
        bp_format.TimeSeries(
            timepoints=np.array([0.0, 1.0]), values=np.array([1.0, 2.0])
        )


def test_canonical_times_mode_keeps_strict_invariants() -> None:
    with pytest.raises(ValueError, match="times and values must have the same length"):
        bp_format.TimeSeries(
            times=np.array([0.0, 1.0, 2.0]),
            values=np.array([1.0, 2.0]),
        )
    with pytest.raises(ValueError, match="times must be strictly increasing"):
        bp_format.TimeSeries(
            times=np.array([0.0, 0.0, 1.0]),
            values=np.array([1.0, 2.0, 3.0]),
        )


def test_timepoints_property_is_removed() -> None:
    ts = bp_format.TimeSeries(times=np.array([0.0, 1.0]), values=np.array([1.0, 2.0]))
    with pytest.raises(AttributeError):
        _ = ts.timepoints
