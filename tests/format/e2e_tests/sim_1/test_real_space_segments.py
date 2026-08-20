"""Tests for sim_1 real-space integration segment construction."""

import jax.numpy as jnp
import numpy as np
import pytest

import hybrax.format as bp

from .real_space_segments import (
    EXPECTED_PROCESS_IDS,
    FERMENTATION_END_ROW,
    OFFLINE_ROW,
    ONLINE_ROW,
    POST_EVENT_ROW,
    PRE_EVENT_ROW,
    SIMULATION_DENSE_OUTPUT,
    RealSpaceSegment,
    build_real_space_segments,
    dense_rows_by_process,
    segment_spline_times,
    segment_state_matrix,
    segment_times,
)


def test_real_space_segments_split_at_pre_and_post_event_rows():
    rows_by_process = dense_rows_by_process(
        SIMULATION_DENSE_OUTPUT,
        EXPECTED_PROCESS_IDS,
    )
    # `dense_rows_by_process` seeds its keys from EXPECTED_PROCESS_IDS, so a
    # `set(...) == EXPECTED_PROCESS_IDS` check would be tautological. Assert
    # instead that every expected process actually has dense rows, which catches
    # a process missing from the CSV.
    assert all(rows_by_process[process_id] for process_id in EXPECTED_PROCESS_IDS)

    for rows in rows_by_process.values():
        segments = build_real_space_segments(rows)
        assert segments

        segment_end_times = {
            segment.rows[-1]["time"]
            for segment in segments
            if segment.rows[-1]["row_type"] == PRE_EVENT_ROW
        }
        pre_event_times = {
            row["time"] for row in rows if row["row_type"] == PRE_EVENT_ROW
        }
        assert segment_end_times == pre_event_times

        post_event_times = {
            row["time"] for row in rows if row["row_type"] == POST_EVENT_ROW
        }
        segment_start_times = {
            segment.rows[0]["time"]
            for segment in segments
            if segment.starts_after_event
        }
        assert segment_start_times == post_event_times

        assert segments[-1].rows[-1]["row_type"] == FERMENTATION_END_ROW

        for segment in segments:
            row_types = [row["row_type"] for row in segment.rows]
            assert OFFLINE_ROW not in row_types
            assert POST_EVENT_ROW not in row_types[1:]
            assert PRE_EVENT_ROW not in row_types[:-1]
            assert np.all(np.diff(segment_times(segment)) > 0.0)


def test_real_space_segments_synthetic_contract():
    rows = [
        {"time": "0.0", "row_type": ONLINE_ROW},
        {"time": "1.0", "row_type": ONLINE_ROW},
        {"time": "2.0", "row_type": ONLINE_ROW},
        {"time": "2.0", "row_type": PRE_EVENT_ROW},
        {"time": "2.0", "row_type": OFFLINE_ROW},
        {"time": "2.0", "row_type": POST_EVENT_ROW},
        {"time": "3.0", "row_type": ONLINE_ROW},
        {"time": "4.0", "row_type": ONLINE_ROW},
        {"time": "4.0", "row_type": FERMENTATION_END_ROW},
    ]

    segments = build_real_space_segments(rows)

    assert [[row["row_type"] for row in segment.rows] for segment in segments] == [
        [ONLINE_ROW, ONLINE_ROW, PRE_EVENT_ROW],
        [POST_EVENT_ROW, ONLINE_ROW, FERMENTATION_END_ROW],
    ]
    np.testing.assert_array_equal(segment_times(segments[0]), [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(segment_times(segments[1]), [2.0, 3.0, 4.0])


def test_post_event_segment_spline_times_shift_selects_right_event_branch():
    carrier = bp.TimeSeries(
        breaks=jnp.asarray([0.0, 2.0, 4.0]),
        coeffs=jnp.asarray([[10.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0]]),
        segment_start_piece_idx=jnp.asarray([0]),
        continuity_side="left",
    )

    rows_by_process = dense_rows_by_process(
        SIMULATION_DENSE_OUTPUT,
        EXPECTED_PROCESS_IDS,
    )
    for rows in rows_by_process.values():
        post_event_segments = [
            segment
            for segment in build_real_space_segments(rows)
            if segment.starts_after_event
        ]
        assert post_event_segments
        for segment in post_event_segments:
            times = segment_times(segment)
            spline_times = segment_spline_times(segment)
            assert spline_times[0] > times[0]
            assert spline_times[0] < times[1]
            np.testing.assert_array_equal(spline_times[1:], times[1:])

    synthetic = RealSpaceSegment(
        (
            {"time": "2.0", "row_type": POST_EVENT_ROW},
            {"time": "3.0", "row_type": ONLINE_ROW},
        )
    )
    exact_event_time = segment_times(synthetic)[0]
    shifted_event_time = segment_spline_times(synthetic)[0]

    assert float(carrier.evaluate(exact_event_time, side="left")) == pytest.approx(10.0)
    assert float(carrier.evaluate(shifted_event_time, side="left")) == pytest.approx(
        20.0
    )


def test_post_event_segment_spline_times_rejects_collapsed_shift():
    start = 1.0
    next_float = np.nextafter(start, 2.0)
    segment = RealSpaceSegment(
        (
            {"time": str(start), "row_type": POST_EVENT_ROW},
            {"time": str(next_float), "row_type": ONLINE_ROW},
        )
    )

    with pytest.raises(ValueError, match="shift must stay inside segment"):
        segment_spline_times(segment)


def test_segment_state_matrix_uses_requested_order():
    segment = RealSpaceSegment(
        (
            {"time": "0.0", "row_type": "online", "glucose": "1.0", "volume": "2.0"},
            {"time": "1.0", "row_type": "online", "glucose": "3.0", "volume": "4.0"},
        )
    )

    np.testing.assert_array_equal(
        segment_state_matrix(segment, ("volume", "glucose")),
        np.asarray([[2.0, 1.0], [4.0, 3.0]]),
    )


def test_real_space_segments_reject_duplicate_post_event_start_time():
    rows = [
        {"time": "0.0", "row_type": ONLINE_ROW},
        {"time": "1.0", "row_type": PRE_EVENT_ROW},
        {"time": "1.0", "row_type": POST_EVENT_ROW},
        {"time": "1.0", "row_type": ONLINE_ROW},
    ]

    with pytest.raises(ValueError, match="unexpected duplicate timestamp"):
        build_real_space_segments(rows)


def test_real_space_segments_reject_misaligned_event_pair():
    rows = [
        {"time": "0.0", "row_type": ONLINE_ROW},
        {"time": "1.0", "row_type": PRE_EVENT_ROW},
        {"time": "1.1", "row_type": POST_EVENT_ROW},
        {"time": "2.0", "row_type": ONLINE_ROW},
    ]

    with pytest.raises(ValueError, match="does not match pre-event"):
        build_real_space_segments(rows)


def test_real_space_segments_reject_interrupted_event_pair():
    rows = [
        {"time": "0.0", "row_type": ONLINE_ROW},
        {"time": "1.0", "row_type": PRE_EVENT_ROW},
        {"time": "1.5", "row_type": ONLINE_ROW},
        {"time": "1.0", "row_type": POST_EVENT_ROW},
    ]

    with pytest.raises(ValueError, match="expected post-event row after pre-event"):
        build_real_space_segments(rows)


def test_real_space_segments_reject_unexpected_duplicate_timestamp():
    rows = [
        {"time": "0.0", "row_type": ONLINE_ROW},
        {"time": "1.0", "row_type": ONLINE_ROW},
        {"time": "1.0", "row_type": ONLINE_ROW},
    ]

    with pytest.raises(ValueError, match="unexpected duplicate timestamp"):
        build_real_space_segments(rows)


def test_real_space_segments_reject_nonmonotone_segment_times():
    rows = [
        {"time": "0.0", "row_type": ONLINE_ROW},
        {"time": "2.0", "row_type": ONLINE_ROW},
        {"time": "1.0", "row_type": PRE_EVENT_ROW},
        {"time": "1.0", "row_type": POST_EVENT_ROW},
        {"time": "3.0", "row_type": ONLINE_ROW},
    ]

    with pytest.raises(ValueError, match="strictly increasing"):
        build_real_space_segments(rows)


def test_real_space_segments_reject_nonterminal_fermentation_end():
    rows = [
        {"time": "0.0", "row_type": ONLINE_ROW},
        {"time": "1.0", "row_type": FERMENTATION_END_ROW},
        {"time": "2.0", "row_type": ONLINE_ROW},
    ]

    with pytest.raises(ValueError, match="fermentation_end must be the final"):
        build_real_space_segments(rows)


def test_real_space_segments_reject_trailing_one_row_post_event_segment():
    rows = [
        {"time": "0.0", "row_type": ONLINE_ROW},
        {"time": "1.0", "row_type": PRE_EVENT_ROW},
        {"time": "1.0", "row_type": POST_EVENT_ROW},
    ]

    with pytest.raises(ValueError, match="trailing one-row"):
        build_real_space_segments(rows)
