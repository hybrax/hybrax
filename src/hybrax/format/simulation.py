"""Small simulation base API with generic sample/bolus event plumbing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

ROW_TYPE_ONLINE = "online"
ROW_TYPE_OFFLINE = "offline"
ROW_TYPE_PRE_EVENT = "pre-event"
ROW_TYPE_POST_EVENT = "post-event"
ROW_TYPE_FERMENTATION_END = "fermentation_end"
ROW_TYPE_ORDER = (
    ROW_TYPE_ONLINE,
    ROW_TYPE_OFFLINE,
    ROW_TYPE_PRE_EVENT,
    ROW_TYPE_POST_EVENT,
    ROW_TYPE_FERMENTATION_END,
)

EVENT_TYPE_SAMPLE = "sample"
EVENT_TYPE_BOLUS = "bolus"
EVENT_TYPE_FERMENTATION_END = "fermentation_end"
_EVENT_ORDER = {
    EVENT_TYPE_SAMPLE: 0,
    EVENT_TYPE_BOLUS: 1,
    EVENT_TYPE_FERMENTATION_END: 2,
}


@dataclass(frozen=True)
class SimulationEvent:
    """One realized discrete event operation."""

    process: str
    time: float
    event_type: str
    delta_volume: float
    feed_id: str | None = None
    feed_concentrations: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationResult:
    """In-memory simulation output plus rows ready for CSV writing."""

    process: str
    times: np.ndarray
    states: np.ndarray
    state_names: tuple[str, ...]
    reactor_state_names: tuple[str, ...]
    dense_rows: list[dict[str, Any]]
    event_rows: list[dict[str, Any]]
    row_columns: tuple[str, ...]
    event_columns: tuple[str, ...]


class Simulation:
    """Common simulation event/output helpers."""

    def group_events(
        self,
        events: Sequence[SimulationEvent],
    ) -> dict[tuple[str, float], list[SimulationEvent]]:
        """Group events and enforce sample-before-bolus multiplicity rules."""
        grouped: dict[tuple[str, float], list[SimulationEvent]] = {}
        for event in events:
            if event.event_type not in _EVENT_ORDER:
                raise ValueError(f"Unsupported event_type: {event.event_type!r}.")
            if event.event_type == EVENT_TYPE_SAMPLE and event.delta_volume >= 0.0:
                raise ValueError("sample events must have negative delta_volume.")
            if event.event_type == EVENT_TYPE_BOLUS and event.delta_volume <= 0.0:
                raise ValueError("bolus events must have positive delta_volume.")
            if (
                event.event_type == EVENT_TYPE_FERMENTATION_END
                and event.delta_volume != 0.0
            ):
                raise ValueError("fermentation_end events must have zero delta_volume.")
            grouped.setdefault((event.process, float(event.time)), []).append(event)

        for (process, time), group in grouped.items():
            seen: set[str] = set()
            for event in group:
                if event.event_type in seen:
                    raise ValueError(
                        "At most one sample and one bolus are allowed per "
                        f"timestamp per process: process={process!r}, time={time}."
                    )
                seen.add(event.event_type)
            group.sort(key=lambda event: _EVENT_ORDER[event.event_type])
        return grouped

    def apply_events(
        self,
        state: Sequence[float],
        events: Sequence[SimulationEvent],
        *,
        state_names: Sequence[str],
        reactor_state_names: Sequence[str],
        volume_state_name: str = "volume",
    ) -> np.ndarray:
        """Apply sample-before-bolus events to one pre-event state."""
        grouped = self.group_events(events)
        if len(grouped) > 1:
            raise ValueError("apply_events expects one process/timestamp group.")
        ordered_events = next(iter(grouped.values()), [])

        out = np.asarray(state, dtype=float).copy()
        state_idx = {name: index for index, name in enumerate(state_names)}
        volume_idx = state_idx[volume_state_name]
        reactor_indices = [state_idx[name] for name in reactor_state_names]

        for event in ordered_events:
            volume = out[volume_idx]
            new_volume = volume + event.delta_volume
            if new_volume <= 0.0:
                raise ValueError("event would make reactor volume non-positive.")

            if event.event_type == EVENT_TYPE_FERMENTATION_END:
                continue
            if event.event_type == EVENT_TYPE_SAMPLE:
                out[volume_idx] = new_volume
                continue

            for name, index in zip(reactor_state_names, reactor_indices):
                feed_conc = float(event.feed_concentrations.get(name, 0.0))
                out[index] = (out[index] * volume + feed_conc * event.delta_volume) / (
                    new_volume
                )
            out[volume_idx] = new_volume
        return out

    def build_result(
        self,
        *,
        process: str,
        state_times: Sequence[float],
        states: Sequence[Sequence[float]],
        online_times: Sequence[float],
        state_names: Sequence[str],
        reactor_state_names: Sequence[str],
        events: Sequence[SimulationEvent] = (),
        extra_columns: Mapping[str, Sequence[float]] | None = None,
        volume_state_name: str = "volume",
    ) -> SimulationResult:
        """Build dense rows, event rows, and optionally write CSV outputs."""
        times = np.asarray(state_times, dtype=float)
        state_array = np.asarray(states, dtype=float)
        state_names = tuple(state_names)
        reactor_state_names = tuple(reactor_state_names)
        if state_array.shape != (len(times), len(state_names)):
            raise ValueError("states shape must match state_times and state_names.")

        grouped_events = self.group_events(events)
        events_by_time = {
            time: group
            for (event_process, time), group in grouped_events.items()
            if event_process == process
        }
        online_set = {float(time) for time in online_times}
        row_times = sorted(online_set | set(events_by_time))
        state_by_time = {
            float(time): state_array[index] for index, time in enumerate(times)
        }
        missing_times = sorted(set(row_times) - set(state_by_time))
        if missing_times:
            raise ValueError(f"Missing state values for times: {missing_times}.")

        extra_columns = extra_columns or {}
        for name, values in extra_columns.items():
            if len(values) != len(times):
                raise ValueError(f"extra column {name!r} must align to state_times.")
        extra_by_time = {
            float(time): {
                name: float(values[index]) for name, values in extra_columns.items()
            }
            for index, time in enumerate(times)
        }
        process_events = [event for event in events if event.process == process]
        row_columns = (
            "process_id",
            "time",
            "row_type",
            *state_names,
            "cum_bolus_feed",
            *extra_columns,
        )
        event_columns = self.event_columns(reactor_state_names)
        event_rows = self.build_event_rows(process_events, reactor_state_names)
        dense_rows = self.build_dense_rows(
            process=process,
            row_times=row_times,
            online_times=online_set,
            states_by_time=state_by_time,
            extra_by_time=extra_by_time,
            events_by_time=events_by_time,
            state_names=state_names,
            reactor_state_names=reactor_state_names,
            volume_state_name=volume_state_name,
        )

        result = SimulationResult(
            process=process,
            times=times.copy(),
            states=state_array.copy(),
            state_names=state_names,
            reactor_state_names=reactor_state_names,
            dense_rows=dense_rows,
            event_rows=event_rows,
            row_columns=row_columns,
            event_columns=event_columns,
        )
        return result

    def build_dense_rows(
        self,
        *,
        process: str,
        row_times: Sequence[float],
        online_times: set[float],
        states_by_time: Mapping[float, np.ndarray],
        extra_by_time: Mapping[float, Mapping[str, float]],
        events_by_time: Mapping[float, Sequence[SimulationEvent]],
        state_names: Sequence[str],
        reactor_state_names: Sequence[str],
        volume_state_name: str,
    ) -> list[dict[str, Any]]:
        """Build rows ordered by time then online/offline/pre-event/post-event."""
        rows: list[dict[str, Any]] = []
        cum_bolus_feed = 0.0
        for time in row_times:
            state = states_by_time[float(time)]
            extras = extra_by_time.get(float(time), {})
            if time in online_times:
                rows.append(
                    self._dense_row(
                        process,
                        time,
                        ROW_TYPE_ONLINE,
                        state,
                        state_names,
                        cum_bolus_feed,
                        extras,
                    )
                )

            event_group = list(events_by_time.get(float(time), ()))
            if not event_group:
                continue
            if any(
                event.event_type == EVENT_TYPE_FERMENTATION_END for event in event_group
            ):
                rows.append(
                    self._dense_row(
                        process,
                        time,
                        ROW_TYPE_OFFLINE,
                        state,
                        state_names,
                        cum_bolus_feed,
                        extras,
                    )
                )
                rows.append(
                    self._dense_row(
                        process,
                        time,
                        ROW_TYPE_FERMENTATION_END,
                        state,
                        state_names,
                        cum_bolus_feed,
                        extras,
                    )
                )
                continue
            if any(event.event_type == EVENT_TYPE_SAMPLE for event in event_group):
                rows.append(
                    self._dense_row(
                        process,
                        time,
                        ROW_TYPE_OFFLINE,
                        state,
                        state_names,
                        cum_bolus_feed,
                        extras,
                    )
                )
            rows.append(
                self._dense_row(
                    process,
                    time,
                    ROW_TYPE_PRE_EVENT,
                    state,
                    state_names,
                    cum_bolus_feed,
                    extras,
                )
            )

            post_state = self.apply_events(
                state,
                event_group,
                state_names=state_names,
                reactor_state_names=reactor_state_names,
                volume_state_name=volume_state_name,
            )
            cum_bolus_feed += sum(
                event.delta_volume
                for event in event_group
                if event.event_type == EVENT_TYPE_BOLUS
            )
            rows.append(
                self._dense_row(
                    process,
                    time,
                    ROW_TYPE_POST_EVENT,
                    post_state,
                    state_names,
                    cum_bolus_feed,
                    extras,
                )
            )
        return rows

    def build_event_rows(
        self,
        events: Sequence[SimulationEvent],
        reactor_state_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Build one events.csv row per event operation."""
        grouped_events = self.group_events(events)
        rows: list[dict[str, Any]] = []
        for key in sorted(grouped_events):
            for event_order, event in enumerate(grouped_events[key]):
                row: dict[str, Any] = {
                    "process_id": event.process,
                    "time": float(event.time),
                    "event_order": event_order,
                    "event_type": event.event_type,
                    "delta_volume": float(event.delta_volume),
                    "feed_id": event.feed_id or "",
                }
                for name in reactor_state_names:
                    value: float | str = ""
                    if event.event_type == EVENT_TYPE_BOLUS:
                        value = float(event.feed_concentrations.get(name, 0.0))
                    row[f"feed_{name}"] = value
                rows.append(row)
        return rows

    def event_columns(self, reactor_state_names: Sequence[str]) -> tuple[str, ...]:
        """Return events.csv columns for given reactor state order."""
        return (
            "process_id",
            "time",
            "event_order",
            "event_type",
            "delta_volume",
            "feed_id",
            *(f"feed_{name}" for name in reactor_state_names),
        )

    def _dense_row(
        self,
        process: str,
        time: float,
        row_type: str,
        state: Sequence[float],
        state_names: Sequence[str],
        cum_bolus_feed: float,
        extras: Mapping[str, float],
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "process_id": process,
            "time": float(time),
            "row_type": row_type,
        }
        row.update({name: float(value) for name, value in zip(state_names, state)})
        row["cum_bolus_feed"] = float(cum_bolus_feed)
        row.update(extras)
        return row
