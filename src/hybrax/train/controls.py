from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from bp_format.dataclasses import (
    BioProcess,
    FeedVolumeChange,
    FeedMedium,
    FeedMediumComponent,
    ProcessVariable,
    SampleVolumeChange,
    StaticVariable,
    TimeSeries,
)


@dataclass
class SignalSource:
    name: str
    kind: str
    times: np.ndarray
    values: np.ndarray
    evaluator: Callable[[np.ndarray], np.ndarray]
    derivative: Callable[[np.ndarray], np.ndarray]
    metadata: dict[str, Any]


def _as_numpy(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _safe_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    if xp.size == 1:
        return np.full_like(x, fp[0], dtype=float)
    return np.interp(x, xp, fp, left=fp[0], right=fp[-1])


def _piecewise_linear_derivative(
    x: np.ndarray, xp: np.ndarray, fp: np.ndarray
) -> np.ndarray:
    if xp.size <= 1:
        return np.zeros_like(x, dtype=float)

    dx = np.diff(xp)
    slopes = np.divide(np.diff(fp), dx, out=np.zeros_like(dx), where=dx != 0)
    indices = np.searchsorted(xp[1:], x, side="right")
    indices = np.clip(indices, 0, slopes.size - 1)
    return slopes[indices]


def _make_source_from_xy(
    name: str,
    kind: str,
    times: np.ndarray,
    values: np.ndarray,
    metadata: dict[str, Any] | None = None,
    fallback_end: float | None = None,
) -> SignalSource:
    times = _as_numpy(times)
    values = _as_numpy(values)

    order = np.argsort(times, kind="stable")
    times = times[order]
    values = values[order]

    if times.size == 0:
        raise ValueError(f"{name}: empty time series")
    if times.size == 1:
        end = (
            times[0] + 1.0
            if fallback_end is None
            else max(float(fallback_end), float(times[0]))
        )
        if end == times[0]:
            end = float(times[0]) + 1.0
        times = np.asarray([times[0], end], dtype=float)
        values = np.asarray([values[0], values[0]], dtype=float)

    return SignalSource(
        name=name,
        kind=kind,
        times=times,
        values=values,
        evaluator=lambda ts: _safe_interp(_as_numpy(ts), times, values),
        derivative=lambda ts: _piecewise_linear_derivative(
            _as_numpy(ts), times, values
        ),
        metadata=dict(metadata or {}),
    )


def _make_source_from_process_variable(
    process: BioProcess,
    name: str,
    process_variable: ProcessVariable,
) -> SignalSource:
    if isinstance(process_variable.values, TimeSeries):
        if process_variable.values.breaks is not None:
            raise ValueError(
                f"{name}: spline-backed TimeSeries controls are not supported; "
                "sample the control to times/values during prepare"
            )
        return _make_source_from_xy(
            name=name,
            kind="process_variable",
            times=process_variable.values.times,
            values=process_variable.values.values,
            metadata={"source": "timeseries"},
            fallback_end=float(process.time_axis.end),
        )

    if isinstance(process_variable.values, StaticVariable):
        t_start = float(process.time_axis.start)
        t_end = float(process.time_axis.end)
        return _make_source_from_xy(
            name=name,
            kind="process_variable",
            times=np.asarray([t_start, t_end], dtype=float),
            values=np.asarray(
                [
                    float(process_variable.values.value),
                    float(process_variable.values.value),
                ],
                dtype=float,
            ),
            metadata={"source": "static"},
        )

    raise TypeError(f"Unsupported process-variable value type for {name}")


def _serialize_concentration(value: TimeSeries | StaticVariable) -> dict[str, Any]:
    if isinstance(value, StaticVariable):
        return {
            "kind": "static",
            "value": float(value.value),
        }
    return {
        "kind": "timeseries",
        "times": _as_numpy(value.times).tolist(),
        "values": _as_numpy(value.values).tolist(),
    }


def _serialize_feed_medium(feed_medium: FeedMedium) -> dict[str, Any]:
    components: dict[str, Any] = {}
    for component_name, component in feed_medium.components.items():
        assert isinstance(component, FeedMediumComponent)
        components[component_name] = {
            "name": component.name,
            "unit": component.unit,
            "concentration": _serialize_concentration(component.concentration),
            "is_controlled": bool(component.is_controlled),
        }

    return {
        "name": feed_medium.name,
        "density": float(feed_medium.density),
        "density_unit": feed_medium.density_unit,
        "components": components,
    }


def _make_source_from_volume_change(
    name: str, volume_change: FeedVolumeChange
) -> SignalSource:
    if volume_change.values.breaks is not None:
        raise ValueError(
            f"{name}: spline-backed TimeSeries controls are not supported; "
            "sample the control to times/values during prepare"
        )
    return _make_source_from_xy(
        name=name,
        kind="volume_change",
        times=volume_change.values.times,
        values=volume_change.values.values,
        metadata={
            "source": "timeseries",
            "source_kind": "control",
            "signal_family": "feed",
            "feed_name": name,
            "inlet_feed_medium": (
                _serialize_feed_medium(volume_change.feed_medium)
                if volume_change.feed_medium is not None
                else None
            ),
        },
    )


def _feed_medium_cin_row(
    feed_medium: FeedMedium | None,
    species_names: tuple[str, ...],
    *,
    feed_name: str,
) -> list[float]:
    if feed_medium is None:
        raise ValueError(
            f"FeedVolumeChange {feed_name!r} must define feed_medium for transport."
        )
    row: list[float] = []
    for species_name in species_names:
        component = feed_medium.components.get(species_name)
        if component is None:
            row.append(0.0)
            continue
        concentration = component.concentration
        if not isinstance(concentration, StaticVariable):
            raise NotImplementedError(
                "TimeSeries feed concentrations are not supported for pseudobatch "
                f"event transport. Found species {species_name!r} "
                f"in feed {feed_name!r}."
            )
        row.append(float(concentration.value))
    return row


def collect_discrete_event_metadata(
    process: BioProcess,
    species_names: tuple[str, ...],
) -> dict[str, Any]:
    """Collect true sample/bolus events for pseudobatch algebraic forcing."""
    t_start = float(process.time_axis.start)
    t_end = float(process.time_axis.end)
    sample_events: list[tuple[float, float]] = []
    bolus_events: list[tuple[float, float, list[float]]] = []

    for name, volume_change in process.volume.volume_changes.items():
        times = _as_numpy(volume_change.values.times)
        values = _as_numpy(volume_change.values.values)
        if times.size != values.size:
            raise ValueError(f"{name}: volume-change times and values differ in size")

        if isinstance(volume_change, SampleVolumeChange):
            for event_time, delta_v in zip(
                times.tolist(), values.tolist(), strict=False
            ):
                event_time = float(event_time)
                delta_v = float(delta_v)
                if delta_v == 0.0:
                    continue
                if delta_v > 0.0:
                    raise ValueError(
                        f"Sample {name!r} has positive delta_v at "
                        f"t={event_time}: {delta_v}. Samples must remove volume."
                    )
                sample_v = abs(delta_v)
                if event_time < t_start:
                    raise ValueError(
                        f"Sample {name!r} timestamp {event_time} before "
                        f"process start {t_start}."
                    )
                if event_time > t_end:
                    continue
                sample_events.append((event_time, sample_v))
            continue

        if not isinstance(volume_change, FeedVolumeChange):
            continue
        if volume_change.is_continuous or not volume_change.is_controlled:
            continue

        cin_row = _feed_medium_cin_row(
            volume_change.feed_medium,
            species_names,
            feed_name=name,
        )
        for event_time, delta_v in zip(times.tolist(), values.tolist(), strict=False):
            event_time = float(event_time)
            delta_v = float(delta_v)
            if delta_v == 0.0:
                continue
            if delta_v < 0.0:
                raise ValueError(
                    f"Bolus feed {name!r} has negative delta_v at "
                    f"t={event_time}: {delta_v}. Boluses must add volume."
                )
            if event_time < t_start or event_time >= t_end:
                raise ValueError(
                    f"Bolus feed {name!r} timestamp {event_time} outside "
                    f"[{t_start}, {t_end})."
                )
            bolus_events.append((event_time, delta_v, cin_row))

    sample_by_time: dict[float, float] = {}
    for event_time, sample_v in sample_events:
        sample_by_time[event_time] = sample_by_time.get(event_time, 0.0) + sample_v
    sample_events = sorted(sample_by_time.items(), key=lambda item: item[0])
    bolus_events.sort(key=lambda item: item[0])
    return {
        "sample_times": [t for t, _ in sample_events],
        "sample_volumes": [v for _, v in sample_events],
        "bolus_times": [t for t, _, _ in bolus_events],
        "bolus_volumes": [v for _, v, _ in bolus_events],
        "bolus_Cin": [row for _, _, row in bolus_events],
    }


@dataclass(frozen=True)
class ControlSourceBundle:
    """Categorised control sources from a process.

    Layout mirrors bp-format ``ControlSplines`` — the continuous controls the
    RHS integrates via RhsOde's ``u`` argument. Discrete bolus/sample events are
    NOT controls here; they are applied as state jumps by the callbacks solve:

        [name_controlled_FVCs | name_controlled_SVCs | name_controlled_PVs]
    """

    name_controlled_FVCs: tuple[str, ...]
    name_controlled_SVCs: tuple[str, ...]
    name_controlled_PVs: tuple[str, ...]
    sources_by_name: dict[str, SignalSource]

    @property
    def all_names(self) -> tuple[str, ...]:
        return (
            self.name_controlled_FVCs
            + self.name_controlled_SVCs
            + self.name_controlled_PVs
        )

    @property
    def all_sources(self) -> list[SignalSource]:
        return [self.sources_by_name[n] for n in self.all_names]


def select_control_sources(process: BioProcess) -> ControlSourceBundle:
    fvc_continuous: dict[str, SignalSource] = {}
    pv_controlled: dict[str, SignalSource] = {}

    for name, volume_change in process.volume.volume_changes.items():
        if not isinstance(volume_change, FeedVolumeChange):
            continue
        if not volume_change.is_controlled:
            continue
        if not volume_change.is_continuous:
            continue
        fvc_continuous[name] = _make_source_from_volume_change(name, volume_change)

    for name, process_variable in process.process_variables.items():
        if not process_variable.is_controlled:
            continue
        pv_controlled[name] = _make_source_from_process_variable(
            process=process,
            name=name,
            process_variable=process_variable,
        )

    name_controlled_FVCs = tuple(sorted(fvc_continuous))
    name_controlled_SVCs: tuple[str, ...] = ()
    name_controlled_PVs = tuple(sorted(pv_controlled))

    sources_by_name: dict[str, SignalSource] = {
        **fvc_continuous,
        **pv_controlled,
    }

    return ControlSourceBundle(
        name_controlled_FVCs=name_controlled_FVCs,
        name_controlled_SVCs=name_controlled_SVCs,
        name_controlled_PVs=name_controlled_PVs,
        sources_by_name=sources_by_name,
    )


def compute_signal_spreads(
    process_sources: dict[str, list[SignalSource]],
) -> dict[str, float]:
    values_by_name: dict[str, list[float]] = {}

    for sources in process_sources.values():
        for source in sources:
            values_by_name.setdefault(source.name, []).extend(source.values.tolist())

    spreads: dict[str, float] = {}
    for name, values in values_by_name.items():
        arr = np.asarray(values, dtype=float)
        spread = float(np.max(arr) - np.min(arr)) if arr.size else 0.0
        spreads[name] = spread if spread > 0 else 1.0
    return spreads


def _linear_interp_from_grid(
    ts: np.ndarray, grid: np.ndarray, values: np.ndarray
) -> np.ndarray:
    if values.ndim == 1:
        return _safe_interp(ts, grid, values)

    out = np.empty((ts.size, values.shape[1]), dtype=float)
    for idx in range(values.shape[1]):
        out[:, idx] = _safe_interp(ts, grid, values[:, idx])
    return out


def build_dense_payload(
    process: BioProcess,
    sources: list[SignalSource],
    spreads: dict[str, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    start = float(process.time_axis.start)
    end = float(process.time_axis.end)
    initial_grid_points = int(config.get("initial_grid_points", 16))
    max_rel_error = float(config.get("max_rel_error", 1e-4))
    max_refinement_rounds = int(config.get("max_refinement_rounds", 8))

    source_knots: list[float] = []
    for source in sources:
        source_knots.extend(source.times.tolist())

    grid = np.unique(
        np.concatenate(
            [
                np.asarray(source_knots, dtype=float),
                np.linspace(start, end, num=max(initial_grid_points, 2), dtype=float),
            ]
        )
    )

    if not sources:
        # A process with no continuous control sources (e.g. driven only by
        # discrete bolus/sample events handled in the callbacks solve). Emit a
        # zero-width control payload on the base grid — there is nothing to
        # refine and ``np.column_stack`` would reject the empty source list.
        return {
            "grid": grid.tolist(),
            "values": [[] for _ in range(grid.size)],
            "derivatives": [[] for _ in range(grid.size)],
        }

    for _ in range(max_refinement_rounds):
        mids = 0.5 * (grid[:-1] + grid[1:])
        if mids.size == 0:
            break

        source_values = np.column_stack([source.evaluator(mids) for source in sources])
        grid_values = np.column_stack([source.evaluator(grid) for source in sources])
        interp_values = _linear_interp_from_grid(mids, grid, grid_values)

        rel_errors = np.zeros_like(source_values)
        for idx, source in enumerate(sources):
            denom = spreads.get(source.name, 1.0)
            rel_errors[:, idx] = (
                np.abs(source_values[:, idx] - interp_values[:, idx]) / denom
            )

        failing = np.any(rel_errors > max_rel_error, axis=1)
        if not np.any(failing):
            break
        grid = np.unique(np.concatenate([grid, mids[failing]]))

    values = np.column_stack([source.evaluator(grid) for source in sources])
    derivatives = np.column_stack([source.derivative(grid) for source in sources])

    return {
        "grid": grid.tolist(),
        "values": values.tolist(),
        "derivatives": derivatives.tolist(),
    }
