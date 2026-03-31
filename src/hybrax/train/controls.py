from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from bpbench.dataclasses import (
    BioProcess,
    FeedVolumeChange,
    FeedMedium,
    FeedMediumComponent,
    ProcessVariable,
    SampleVolumeChange,
    StaticVariable,
    TimeSeries,
)


BP_TRAIN_SAMPLE_ACC_NAME = "V_sample_acc"


@dataclass
class SignalSource:
    name: str
    kind: str
    times: np.ndarray
    values: np.ndarray
    evaluator: Callable[[np.ndarray], np.ndarray]
    derivative: Callable[[np.ndarray], np.ndarray]
    step_ts: list[float]
    metadata: dict[str, Any]


BOLUS_DUPLICATE_THRESHOLD_REL = 1e-4


def _as_numpy(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _dedupe_sorted(values: list[float] | np.ndarray) -> list[float]:
    if len(values) == 0:
        return []
    arr = np.asarray(values, dtype=float)
    arr = np.unique(np.round(arr, decimals=12))
    return arr.tolist()


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


def _compact_ppoly_breaks(
    x: np.ndarray, coefficients: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    breaks: list[float] = []
    coeff_cols: list[np.ndarray] = []
    for idx in range(x.size - 1):
        left = float(x[idx])
        right = float(x[idx + 1])
        if right <= left:
            continue
        breaks.append(left)
        coeff_cols.append(coefficients[:, idx])
    if not breaks:
        raise ValueError("interpax_ppoly did not contain any positive-width interval")
    breaks.append(float(x[-1]))
    return np.asarray(breaks, dtype=float), np.stack(coeff_cols, axis=1)


def _eval_ppoly(
    x: np.ndarray,
    coefficients: np.ndarray,
    ts: np.ndarray,
    order: int = 0,
) -> np.ndarray:
    breaks, coeff_cols = _compact_ppoly_breaks(x, coefficients)
    n_intervals = coeff_cols.shape[1]
    idx = np.searchsorted(breaks[1:], ts, side="right")
    idx = np.clip(idx, 0, n_intervals - 1)
    dx = ts - breaks[idx]
    selected = coeff_cols[:, idx]

    degree = coeff_cols.shape[0] - 1
    if order > degree:
        return np.zeros_like(ts, dtype=float)

    if order > 0:
        deriv = selected.copy()
        current_degree = degree
        for _ in range(order):
            powers = np.arange(current_degree, 0, -1, dtype=float)[:, None]
            deriv = deriv[:-1] * powers
            current_degree -= 1
        selected = deriv

    out = selected[0]
    for row in selected[1:]:
        out = out * dx + row
    return np.asarray(out, dtype=float)


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
        step_ts=_dedupe_sorted(times.tolist()),
        metadata=dict(metadata or {}),
    )


def _make_source_from_process_variable(
    process: BioProcess,
    name: str,
    process_variable: ProcessVariable,
) -> SignalSource:
    interpolator = process_variable.interpolator
    if interpolator is not None and interpolator.kind == "interpax_ppoly":
        x = _as_numpy(interpolator.x)
        coefficients = _as_numpy(interpolator.coefficients)
        breaks, _ = _compact_ppoly_breaks(x, coefficients)
        return SignalSource(
            name=name,
            kind="process_variable",
            times=breaks,
            values=_eval_ppoly(x, coefficients, breaks),
            evaluator=lambda ts: _eval_ppoly(x, coefficients, _as_numpy(ts), order=0),
            derivative=lambda ts: _eval_ppoly(x, coefficients, _as_numpy(ts), order=1),
            step_ts=_dedupe_sorted(breaks.tolist()),
            metadata={"source": "ppoly"},
        )

    if isinstance(process_variable.values, TimeSeries):
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


def _collect_online_time_points(process: BioProcess) -> np.ndarray:
    times: list[float] = []

    for process_variable in process.process_variables.values():
        if isinstance(process_variable.values, TimeSeries):
            times.extend(float(t) for t in _as_numpy(process_variable.values.times))

    for volume_change in process.volume.volume_changes.values():
        times.extend(float(t) for t in _as_numpy(volume_change.values.times))

    return np.asarray(sorted(set(times)), dtype=float)


def get_shortest_time_diff(process: BioProcess) -> float:
    points = _collect_online_time_points(process)
    total_duration = max(
        float(process.time_axis.end) - float(process.time_axis.start), 1.0
    )
    if points.size <= 1:
        return total_duration / 100.0

    points = np.unique(np.round(points, decimals=9))
    diffs = np.diff(points)
    meaningful_floor = max(total_duration * 1e-6, 1e-9)
    diffs = diffs[diffs > meaningful_floor]
    if diffs.size == 0:
        return total_duration / 100.0
    return float(np.min(diffs))


def build_sample_acc_source_default(process: BioProcess) -> SignalSource:
    sample_changes: list[tuple[float, float]] = []
    for volume_change in process.volume.volume_changes.values():
        if not isinstance(volume_change, SampleVolumeChange):
            continue
        times = _as_numpy(volume_change.values.times)
        values = _as_numpy(volume_change.values.values)
        for t, delta in zip(times.tolist(), values.tolist(), strict=False):
            sample_changes.append((float(t), abs(float(delta))))

    sample_changes.sort(key=lambda item: item[0])
    ramp_duration = get_shortest_time_diff(process)
    t_end = float(process.time_axis.end)

    times = [float(process.time_axis.start)]
    values = [0.0]
    cumulative = 0.0
    step_ts: list[float] = []

    for event_time, sample_amount in sample_changes:
        ramp_end = min(event_time + ramp_duration, t_end)
        times.extend([event_time, ramp_end])
        values.extend([cumulative, cumulative + sample_amount])
        cumulative += sample_amount
        step_ts.extend([event_time, ramp_end])

    if times[-1] < t_end:
        times.append(t_end)
        values.append(cumulative)

    metadata = {
        "source": "sample_volume_changes",
        "event_count": len(sample_changes),
        "ramp_duration": ramp_duration,
    }
    source = _make_source_from_xy(
        name=BP_TRAIN_SAMPLE_ACC_NAME,
        kind="derived_control",
        times=np.asarray(times, dtype=float),
        values=np.asarray(values, dtype=float),
        metadata=metadata,
    )
    source.step_ts = _dedupe_sorted(step_ts + source.step_ts)
    return source


def build_bolus_sources(process: BioProcess) -> list[SignalSource]:
    ramp_duration = get_shortest_time_diff(process)
    t_end = float(process.time_axis.end)
    sources: list[SignalSource] = []

    for name, volume_change in process.volume.volume_changes.items():
        if not isinstance(volume_change, FeedVolumeChange):
            continue
        if volume_change.is_continuous:
            continue
        if not volume_change.is_controlled:
            continue

        times = [float(process.time_axis.start)]
        values = [0.0]
        step_ts: list[float] = []

        event_pairs = list(
            zip(
                _as_numpy(volume_change.values.times).tolist(),
                _as_numpy(volume_change.values.values).tolist(),
                strict=False,
            )
        )

        threshold = BOLUS_DUPLICATE_THRESHOLD_REL * (
            t_end - float(process.time_axis.start)
        )
        event_times = sorted(float(et) for et, _ in event_pairs)
        for i in range(len(event_times) - 1):
            if event_times[i + 1] - event_times[i] < threshold:
                raise ValueError(
                    f"Feed '{name}' has duplicate bolus timestamps:"
                    f" {event_times[i]} and {event_times[i + 1]} are within"
                    f" the deduplication threshold ({threshold:.3g})."
                )

        for event_time, delta_v in event_pairs:
            event_time = float(event_time)
            delta_v = float(delta_v)
            ramp_end = min(event_time + ramp_duration, t_end)
            rate = 0.0 if ramp_end <= event_time else delta_v / (ramp_end - event_time)
            times.extend([event_time, event_time, ramp_end])
            values.extend([0.0, rate, 0.0])
            step_ts.extend([event_time, ramp_end])

        if times[-1] < t_end:
            times.append(t_end)
            values.append(0.0)

        source = _make_source_from_xy(
            name=name,
            kind="volume_change",
            times=np.asarray(times, dtype=float),
            values=np.asarray(values, dtype=float),
            metadata={
                "source": "bolus_ramp",
                "source_kind": "control",
                "signal_family": "feed",
                "feed_name": name,
                "ramp_duration": ramp_duration,
                "inlet_feed_medium": (
                    _serialize_feed_medium(volume_change.feed_medium)
                    if volume_change.feed_medium is not None
                    else None
                ),
            },
        )
        source.step_ts = _dedupe_sorted(step_ts + source.step_ts)
        sources.append(source)

    return sources


def select_control_sources(
    process_name: str,
    process: BioProcess,
    config: dict[str, Any],
) -> list[SignalSource]:
    volume_sources: dict[str, SignalSource] = {}
    process_var_sources: dict[str, SignalSource] = {}

    for name, volume_change in process.volume.volume_changes.items():
        if not isinstance(volume_change, FeedVolumeChange):
            continue
        if not volume_change.is_controlled:
            continue
        if not volume_change.is_continuous:
            continue
        volume_sources[name] = _make_source_from_volume_change(name, volume_change)

    for source in build_bolus_sources(process):
        if source.name in volume_sources:
            raise ValueError(
                f"{process_name}: duplicate control source name {source.name}"
            )
        volume_sources[source.name] = source

    for name, process_variable in process.process_variables.items():
        if not process_variable.is_controlled:
            continue
        process_var_sources[name] = _make_source_from_process_variable(
            process=process,
            name=name,
            process_variable=process_variable,
        )

    explicit = config.get("control_order")
    if isinstance(explicit, dict):
        explicit = explicit.get(process_name)

    ordered_names: list[str] = []
    if explicit is not None:
        all_names = set(volume_sources) | set(process_var_sources)
        missing = [name for name in explicit if name not in all_names]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                f"{process_name}: control_order references missing controls:"
                f" {missing_str}"
            )
        ordered_names.extend(explicit)

    volume_names = [name for name in volume_sources if name not in ordered_names]
    process_var_names = [
        name for name in process_var_sources if name not in ordered_names
    ]

    ordered_names.extend(volume_names)
    ordered_names.extend(process_var_names)

    sources: list[SignalSource] = []
    for name in ordered_names:
        if name in volume_sources:
            sources.append(volume_sources[name])
        else:
            sources.append(process_var_sources[name])
    return sources


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

    step_ts: list[float] = [start, end]
    source_knots: list[float] = []
    for source in sources:
        step_ts.extend(source.step_ts)
        source_knots.extend(source.times.tolist())

    grid = np.unique(
        np.concatenate(
            [
                np.asarray(source_knots, dtype=float),
                np.linspace(start, end, num=max(initial_grid_points, 2), dtype=float),
            ]
        )
    )

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
        "step_ts": _dedupe_sorted(step_ts),
    }
