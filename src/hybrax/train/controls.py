from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from hybrax.format.dataclasses import (
    BioProcess,
    FeedMedium,
    FeedMediumComponent,
    Inflow,
    Outflow,
    ProcessVariable,
    StaticVariable,
    TimeSeries,
)
from hybrax.format.mechanistic import extract_discrete_events, get_process_ordering
from hybrax.format.time_series.spline_ops import rebase_to_breaks


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


def _validate_cumulative_direction(
    name: str,
    volume_change: Inflow | Outflow,
    process: BioProcess,
) -> None:
    """Validate the cumulative-flow derivative over the closed solve interval."""
    values = volume_change.values
    if values.breaks is None:
        slopes = np.diff(_as_numpy(values.values)) / np.diff(_as_numpy(values.times))
    else:
        breaks = _as_numpy(values.breaks)
        coeffs = _as_numpy(values.coeffs)
        rates: list[float] = []
        for index, (left, right) in enumerate(
            zip(breaks[:-1], breaks[1:], strict=True)
        ):
            clipped_left = max(float(left), float(process.time_axis.start))
            clipped_right = min(float(right), float(process.time_axis.end))
            if clipped_left > clipped_right:
                continue
            rate_coeffs = np.polynomial.polynomial.polyder(coeffs[index])
            candidates = [clipped_left - left, clipped_right - left]
            acceleration_coeffs = np.polynomial.polynomial.polyder(rate_coeffs)
            for root in np.polynomial.polynomial.polyroots(acceleration_coeffs):
                if np.isreal(root):
                    root = float(np.real(root))
                    if candidates[0] <= root <= candidates[1]:
                        candidates.append(root)
            rates.extend(np.polynomial.polynomial.polyval(candidates, rate_coeffs))
        slopes = np.asarray(rates)

    direction = 1.0 if isinstance(volume_change, Inflow) else -1.0
    if np.any(direction * slopes < 0.0):
        expected = "nonnegative" if direction > 0.0 else "nonpositive"
        raise ValueError(
            f"Continuous {type(volume_change).__name__} {name!r} must have "
            f"{expected} cumulative derivatives."
        )


def _make_source_from_volume_change(
    process: BioProcess,
    name: str,
    volume_change: Inflow | Outflow,
) -> SignalSource:
    metadata: dict[str, Any] = {
        "source": "timeseries",
        "source_kind": "control",
        "signal_family": ("inflow" if isinstance(volume_change, Inflow) else "outflow"),
    }
    if isinstance(volume_change, Inflow):
        metadata["inlet_feed_medium"] = _serialize_feed_medium(
            volume_change.feed_medium
        )
    else:
        metadata["retention"] = dict(volume_change.retention)

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


def collect_discrete_event_metadata(
    process: BioProcess,
    species_names: tuple[str, ...],
) -> dict[str, Any]:
    """Collect true sample/bolus events for pseudobatch algebraic forcing.

    bp-format owns event validation and species alignment. Signed Outflow deltas
    become positive removal magnitudes exactly once at this solver boundary.
    """
    t_end = float(process.time_axis.end)
    sample_events: list[tuple[float, float]] = []
    bolus_events: list[tuple[float, float, list[float]]] = []
    ordering = get_process_ordering(process)
    if tuple(ordering.name_modeled_RMCs) != species_names:
        raise ValueError("species_names must match bp-format's modeled RMC order")

    for event in extract_discrete_events(process, ordering):
        if event["t"] > t_end:
            continue
        if event["kind"] == "sample":
            sample_events.append((event["t"], -event["dV"]))
        else:
            bolus_events.append(
                (event["t"], event["dV"], np.asarray(event["Cin"]).tolist())
            )

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

        [name_controlled_Inflows | name_controlled_Outflows | name_controlled_PVs]
    """

    name_controlled_Inflows: tuple[str, ...]
    name_controlled_Outflows: tuple[str, ...]
    name_controlled_PVs: tuple[str, ...]
    sources_by_name: dict[str, SignalSource]

    @property
    def all_names(self) -> tuple[str, ...]:
        return (
            self.name_controlled_Inflows
            + self.name_controlled_Outflows
            + self.name_controlled_PVs
        )

    @property
    def all_sources(self) -> list[SignalSource]:
        return [self.sources_by_name[n] for n in self.all_names]


def select_control_sources(process: BioProcess) -> ControlSourceBundle:
    ordering = get_process_ordering(process)
    volume_changes = process.volume.volume_changes
    for name, volume_change in volume_changes.items():
        if volume_change.is_continuous:
            _validate_cumulative_direction(name, volume_change, process)

    controlled_inflows = {
        name: _make_source_from_volume_change(process, name, volume_changes[name])
        for name in ordering.name_controlled_Inflows
    }
    controlled_outflows = {
        name: _make_source_from_volume_change(process, name, volume_changes[name])
        for name in ordering.name_controlled_Outflows
    }
    controlled_pvs = {
        name: _make_source_from_process_variable(
            process=process,
            name=name,
            process_variable=process.process_variables[name],
        )
        for name in ordering.name_controlled_PVs
    }

    name_controlled_Inflows = tuple(controlled_inflows)
    name_controlled_Outflows = tuple(controlled_outflows)
    name_controlled_PVs = tuple(controlled_pvs)

    sources_by_name: dict[str, SignalSource] = {
        **controlled_inflows,
        **controlled_outflows,
        **controlled_pvs,
    }

    return ControlSourceBundle(
        name_controlled_Inflows=name_controlled_Inflows,
        name_controlled_Outflows=name_controlled_Outflows,
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
