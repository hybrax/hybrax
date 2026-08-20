from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from bp_format.dataclasses import (
    BioProcess,
    FeedMedium,
    FeedMediumComponent,
    Inflow,
    Outflow,
    ProcessVariable,
    StaticVariable,
    TimeSeries,
)
from bp_format.time_series.spline_ops import rebase_to_breaks


@dataclass
class SignalSource:
    name: str
    kind: str
    times: np.ndarray
    values: np.ndarray
    evaluator: Callable[[np.ndarray], np.ndarray]
    derivative: Callable[[np.ndarray], np.ndarray]
    metadata: dict[str, Any]
    spline_breaks: np.ndarray | None = None
    spline_coeffs: np.ndarray | None = None
    continuity_side: str | None = None
    is_static: bool = False

    @property
    def support(self) -> tuple[float, float]:
        """Closed source support; static controls are unbounded."""
        if self.is_static:
            return (-float("inf"), float("inf"))
        return (float(self.times[0]), float(self.times[-1]))


def _as_numpy(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _safe_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    if xp.size == 1:
        return np.full_like(x, fp[0], dtype=float)
    return np.interp(x, xp, fp, left=fp[0], right=fp[-1])


def _piecewise_linear_derivative(
    x: np.ndarray, xp: np.ndarray, fp: np.ndarray, side: str
) -> np.ndarray:
    dx = np.diff(xp)
    slopes = np.divide(np.diff(fp), dx, out=np.zeros_like(dx), where=dx != 0)
    indices = np.searchsorted(xp, x, side=side) - 1
    indices = np.clip(indices, 0, slopes.size - 1)
    # Validated solves cannot query outside support; clamping preserves the only
    # in-domain interval rate at each closed support endpoint.
    return slopes[indices]


def _make_source_from_xy(
    name: str,
    kind: str,
    times: np.ndarray,
    values: np.ndarray,
    metadata: dict[str, Any] | None = None,
    continuity_side: str = "right",
) -> SignalSource:
    times = _as_numpy(times)
    values = _as_numpy(values)

    order = np.argsort(times, kind="stable")
    times = times[order]
    values = values[order]

    if times.size == 0:
        raise ValueError(f"{name}: empty time series")
    if times.size == 1:
        raise ValueError(
            f"{name}: continuous-control TimeSeries must contain at least two points"
        )

    return SignalSource(
        name=name,
        kind=kind,
        times=times,
        values=values,
        evaluator=lambda ts: _safe_interp(_as_numpy(ts), times, values),
        derivative=lambda ts: _piecewise_linear_derivative(
            _as_numpy(ts), times, values, continuity_side
        ),
        metadata=dict(metadata or {}),
        continuity_side=continuity_side,
    )


def _eval_ppoly_numpy(
    breaks: np.ndarray, coeffs: np.ndarray, side: str, ts: Any
) -> np.ndarray:
    """Pure-numpy cubic-PPoly eval (matches bp-format ``PPoly.__call__``).

    Power-basis pieces ``p(dt) = a + dt·(b + dt·(c + dt·d))`` with
    ``idx = searchsorted(breaks, t, side) - 1`` clamped to a valid piece. This
    NumPy evaluator is used during preparation and scale estimation.
    """
    ts_arr = np.atleast_1d(_as_numpy(ts))
    idx = np.clip(np.searchsorted(breaks, ts_arr, side=side) - 1, 0, len(breaks) - 2)
    dt = ts_arr - breaks[idx]
    p = coeffs[idx]
    return p[:, 0] + dt * (p[:, 1] + dt * (p[:, 2] + dt * p[:, 3]))


def _make_source_from_spline(
    name: str,
    kind: str,
    series: TimeSeries,
    metadata: dict[str, Any],
) -> SignalSource:
    """Build a source retaining a spline's direct runtime representation."""
    breaks = _as_numpy(series.breaks)
    coeffs = _as_numpy(series.coeffs)
    side = str(getattr(series, "continuity_side", "right"))

    deriv = series.deriv()
    d_breaks = _as_numpy(deriv.breaks)
    d_coeffs = _as_numpy(deriv.coeffs)
    d_side = str(getattr(deriv, "continuity_side", side))

    def _eval(ts: Any) -> np.ndarray:
        return _eval_ppoly_numpy(breaks, coeffs, side, ts)

    def _deriv(ts: Any) -> np.ndarray:
        return _eval_ppoly_numpy(d_breaks, d_coeffs, d_side, ts)

    return SignalSource(
        name=name,
        kind=kind,
        times=breaks,
        values=_eval(breaks),
        evaluator=_eval,
        derivative=_deriv,
        metadata=dict(metadata),
        spline_breaks=breaks,
        spline_coeffs=coeffs,
        continuity_side=side,
    )


def _make_source_from_process_variable(
    process: BioProcess,
    name: str,
    process_variable: ProcessVariable,
) -> SignalSource:
    if isinstance(process_variable.values, TimeSeries):
        if process_variable.values.breaks is not None:
            return _make_source_from_spline(
                name=name,
                kind="process_variable",
                series=process_variable.values,
                metadata={"source": "spline"},
            )
        return _make_source_from_xy(
            name=name,
            kind="process_variable",
            times=process_variable.values.times,
            values=process_variable.values.values,
            metadata={"source": "timeseries"},
            continuity_side=str(process_variable.values.continuity_side),
        )

    if isinstance(process_variable.values, StaticVariable):
        value = float(process_variable.values.value)
        return SignalSource(
            name=name,
            kind="process_variable",
            times=np.asarray([-np.inf, np.inf]),
            values=np.asarray([value, value]),
            evaluator=lambda ts: np.full_like(_as_numpy(ts), value, dtype=float),
            derivative=lambda ts: np.zeros_like(_as_numpy(ts), dtype=float),
            metadata={"source": "static"},
            is_static=True,
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


def _make_source_from_volume_change(name: str, volume_change: Inflow) -> SignalSource:
    metadata = {
        "source": "timeseries",
        "source_kind": "control",
        "signal_family": "feed",
        "feed_name": name,
        "inlet_feed_medium": (
            _serialize_feed_medium(volume_change.feed_medium)
            if volume_change.feed_medium is not None
            else None
        ),
    }
    if volume_change.values.breaks is not None:
        return _make_source_from_spline(
            name=name,
            kind="volume_change",
            series=volume_change.values,
            metadata={**metadata, "source": "spline"},
        )
    return _make_source_from_xy(
        name=name,
        kind="volume_change",
        times=volume_change.values.times,
        values=volume_change.values.values,
        metadata=metadata,
        continuity_side=str(volume_change.values.continuity_side),
    )


def _feed_medium_cin_row(
    feed_medium: FeedMedium | None,
    species_names: tuple[str, ...],
    *,
    feed_name: str,
) -> list[float]:
    if feed_medium is None:
        raise ValueError(f"Inflow {feed_name!r} must define feed_medium for transport.")
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

        if isinstance(volume_change, Outflow):
            if volume_change.is_continuous:
                continue
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

        if not isinstance(volume_change, Inflow):
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
        if not isinstance(volume_change, Inflow):
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


def build_spline_payload(sources: list[SignalSource]) -> dict[str, Any]:
    """Rebase spline sources onto one union break grid."""
    if not sources:
        return {"breaks": [], "coeffs": []}

    breaks = np.unique(
        np.concatenate(
            [
                source.spline_breaks
                for source in sources
                if source.spline_breaks is not None
            ]
        )
    )
    coeffs = np.stack(
        [
            np.asarray(
                rebase_to_breaks(
                    source.spline_breaks,
                    source.spline_coeffs,
                    breaks,
                )
            )
            for source in sources
        ],
        axis=1,
    )
    return {"breaks": breaks.tolist(), "coeffs": coeffs.tolist()}


def build_linear_payload(
    process: BioProcess,
    sources: list[SignalSource],
) -> dict[str, Any]:
    """Build an exact process-local payload for raw and static controls."""
    start = float(process.time_axis.start)
    end = float(process.time_axis.end)
    raw_knots = [
        time
        for source in sources
        if not source.is_static
        for time in source.times.tolist()
        if start <= time <= end
    ]
    grid = np.unique(np.asarray([start, end, *raw_knots], dtype=float))

    if not sources:
        return {
            "grid": grid.tolist(),
            "values": [[] for _ in range(grid.size)],
            "derivatives": [[] for _ in range(grid.size)],
        }

    values = np.column_stack([source.evaluator(grid) for source in sources])
    if grid.size > 1:
        interval_midpoints = 0.5 * (grid[:-1] + grid[1:])
        interval_rates = np.column_stack(
            [source.derivative(interval_midpoints) for source in sources]
        )
        # BatchControls may index padded grid tails, so repeat the final interval rate.
        derivatives = np.concatenate([interval_rates, interval_rates[-1:]], axis=0)
    else:
        derivatives = np.zeros_like(values)

    return {
        "grid": grid.tolist(),
        "values": values.tolist(),
        "derivatives": derivatives.tolist(),
    }
