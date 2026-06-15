"""Dense-row segmentation helpers for sim_1 real-space integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

ONLINE_ROW = "online"
PRE_EVENT_ROW = "pre-event"
POST_EVENT_ROW = "post-event"
OFFLINE_ROW = "offline"
FERMENTATION_END_ROW = "fermentation_end"

CONTINUOUS_ROW_TYPES = {ONLINE_ROW, PRE_EVENT_ROW, POST_EVENT_ROW, FERMENTATION_END_ROW}


@dataclass(frozen=True)
class RealSpaceSegment:
    """One continuous real-space integration segment from dense simulator rows."""

    rows: tuple[dict[str, str], ...]

    @property
    def starts_after_event(self) -> bool:
        return self.rows[0]["row_type"] == POST_EVENT_ROW


def _row_time(row: dict[str, str]) -> float:
    return float(row["time"])


def segment_times(segment: RealSpaceSegment) -> np.ndarray:
    """Dense output times represented by a segment."""
    return np.asarray([_row_time(row) for row in segment.rows], dtype=float)


def segment_spline_times(segment: RealSpaceSegment) -> np.ndarray:
    """Times to use when evaluating left-continuous state/transform splines.

    A post-event row starts the right side of an event. Pseudobatch carriers are
    stored as left-continuous series, so an exact event-time lookup would select the
    pre-event branch. Shift only the first timestamp of post-event-start segments to
    the next representable float toward the following row.
    """
    times = segment_times(segment)
    if segment.starts_after_event:
        if len(times) < 2:
            raise ValueError("post-event segment must contain at least two rows")
        shifted = np.nextafter(times[0], times[1])
        if not times[0] < shifted < times[1]:
            raise ValueError("post-event spline-time shift must stay inside segment")
        times[0] = shifted
    return times


def segment_state_matrix(
    segment: RealSpaceSegment,
    state_names: Sequence[str],
) -> np.ndarray:
    """Return dense row values in a caller-specified state-vector order."""
    return np.asarray(
        [[float(row[name]) for name in state_names] for row in segment.rows],
        dtype=float,
    )


def build_real_space_segments(rows: Sequence[dict[str, str]]) -> list[RealSpaceSegment]:
    """Split dense simulator rows into continuous real-space integration segments.

    Real-space states can jump at discrete events, so integrations must stop at
    pre-event rows and restart from post-event rows. Offline rows are observations,
    not continuous simulator states, and are ignored. At duplicate timestamps, only
    `online -> pre-event` and `online -> fermentation_end` replacements are valid;
    these rows mark the boundary state used to close the segment.
    """
    segments: list[RealSpaceSegment] = []
    current: list[dict[str, str]] = []
    pending_pre_event_time: float | None = None
    seen_terminal = False

    def append_segment(segment_rows: list[dict[str, str]]) -> None:
        times = np.asarray([_row_time(row) for row in segment_rows], dtype=float)
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("real-space segment times must be strictly increasing")
        segments.append(RealSpaceSegment(tuple(segment_rows)))

    for row in rows:
        row_type = row["row_type"]
        if row_type == OFFLINE_ROW:
            continue
        if row_type not in CONTINUOUS_ROW_TYPES:
            raise ValueError(f"unknown dense row_type: {row_type!r}")
        if seen_terminal:
            raise ValueError("fermentation_end must be the final continuous row")

        if row_type == POST_EVENT_ROW:
            event_time = _row_time(row)
            if pending_pre_event_time is None:
                raise ValueError("post-event row has no matching pre-event row")
            if event_time != pending_pre_event_time:
                raise ValueError("post-event row time does not match pre-event time")
            current = [row]
            pending_pre_event_time = None
            continue

        if pending_pre_event_time is not None:
            raise ValueError(
                "expected post-event row after pre-event at "
                f"t={pending_pre_event_time}, got {row_type!r}"
            )

        if current and _row_time(current[-1]) == _row_time(row):
            previous_type = current[-1]["row_type"]
            if previous_type == ONLINE_ROW and row_type in {
                PRE_EVENT_ROW,
                FERMENTATION_END_ROW,
            }:
                current[-1] = row
            else:
                raise ValueError(
                    f"unexpected duplicate timestamp row transition: "
                    f"{previous_type!r} -> {row_type!r}"
                )
        else:
            current.append(row)

        if row_type == PRE_EVENT_ROW:
            if len(current) >= 2:
                append_segment(current)
            current = []
            pending_pre_event_time = _row_time(row)
        elif row_type == FERMENTATION_END_ROW:
            if len(current) >= 2:
                append_segment(current)
            current = []
            seen_terminal = True

    if pending_pre_event_time is not None:
        raise ValueError("pre-event row has no matching post-event row")
    if len(current) == 1:
        raise ValueError("trailing one-row real-space segment has no integration span")
    if len(current) >= 2:
        append_segment(current)

    if not segments:
        raise ValueError("no continuous real-space segments found")
    return segments
