from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from bp_format.dataclasses import (
    BioProcess,
    BioProcessCollection,
    FeedVolumeChange,
    FeedMedium,
    FeedMediumComponent,
    ProcessVariable,
    SampleVolumeChange,
    StaticVariable,
    TimeSeries,
)


BP_TRAIN_SAMPLE_ACC_NAME = "V_sample_acc"
EVENT_RUN_MIN_DT_CONFIG_KEY = "bolus_run_min_dt"


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


BOLUS_MIN_DT_DURATION_DENOMINATOR = 1000.0


def _as_numpy(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _dedupe_sorted(values: list[float] | np.ndarray) -> list[float]:
    if len(values) == 0:
        return []
    arr = np.asarray(values, dtype=float)
    arr = np.unique(arr)
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
        # `step_ts` is a hint for the ODE solver about points where the signal
        # has a true discontinuity (e.g. start/end of a sampling or bolus
        # ramp).  Generic xy time series are smooth signals (linearly
        # interpolated) and therefore contribute NO jump times.  Builders that
        # need ramp boundaries (build_sample_acc_source_default,
        # build_bolus_sources) post-merge their explicit ramp times into
        # `source.step_ts` after construction.  Populating step_ts with the
        # full input grid here used to force the diffrax solver to take a step
        # at every input timestamp (~1500 forced steps for the Kittler dataset)
        # — see plan/cheerful-inventing-lightning.md for the speedup analysis.
        step_ts=[],
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


def _collect_online_time_points(process: BioProcess) -> np.ndarray:
    times: list[float] = []

    for process_variable in process.process_variables.values():
        if isinstance(process_variable.values, TimeSeries):
            times.extend(float(t) for t in _as_numpy(process_variable.values.times))

    for volume_change in process.volume.volume_changes.values():
        times.extend(float(t) for t in _as_numpy(volume_change.values.times))

    return np.asarray(sorted(set(times)), dtype=float)


def _minimum_positive_delta(points: np.ndarray) -> float | None:
    finite = points[np.isfinite(points)]
    unique = np.unique(finite)
    if unique.size <= 1:
        return None
    diffs = np.diff(unique)
    positive_diffs = diffs[diffs > 0.0]
    if positive_diffs.size == 0:
        return None
    return float(np.min(positive_diffs))


def get_collection_bolus_min_dt(collection: BioProcessCollection) -> float:
    per_process_min_dts: list[float] = []
    fallback_duration_caps: list[float] = []
    for process in collection.processes.values():
        duration = float(process.time_axis.end) - float(process.time_axis.start)
        if np.isfinite(duration) and duration > 0.0:
            fallback_duration_caps.append(
                float(duration / BOLUS_MIN_DT_DURATION_DENOMINATOR)
            )
        points = _collect_online_time_points(process)
        min_dt = _minimum_positive_delta(points)
        if min_dt is not None:
            per_process_min_dts.append(min_dt)

    if per_process_min_dts:
        return float(min(per_process_min_dts))

    if fallback_duration_caps:
        return float(min(fallback_duration_caps))

    raise ValueError(
        "Bolus controls require either a strictly positive online timestamp "
        "delta within a process or a positive process duration to compute "
        "run-level min_dt."
    )


def get_collection_event_min_dt_if_needed(
    collection: BioProcessCollection,
    *,
    include_samples: bool = True,
) -> float | None:
    for process in collection.processes.values():
        for volume_change in process.volume.volume_changes.values():
            times = np.asarray(volume_change.values.times, dtype=float)
            values = np.asarray(volume_change.values.values, dtype=float)
            if times.size == 0 or values.size == 0:
                continue
            if isinstance(volume_change, FeedVolumeChange):
                if not bool(volume_change.is_controlled) or bool(
                    volume_change.is_continuous
                ):
                    continue
                return get_collection_bolus_min_dt(collection)
            if include_samples and isinstance(volume_change, SampleVolumeChange):
                return get_collection_bolus_min_dt(collection)
    return None


def get_bolus_min_dt(process: BioProcess, run_min_dt: float | None = None) -> float:
    duration = float(process.time_axis.end) - float(process.time_axis.start)
    if duration <= 0.0:
        raise ValueError(
            "Bolus controls require process duration > 0 to compute min_dt cap."
        )
    duration_cap = duration / BOLUS_MIN_DT_DURATION_DENOMINATOR

    if run_min_dt is not None:
        if not np.isfinite(run_min_dt) or run_min_dt <= 0.0:
            raise ValueError(
                f"Invalid run-level bolus min_dt: {run_min_dt}. Must be finite > 0."
            )
        return float(min(run_min_dt, duration_cap))

    points = _collect_online_time_points(process)
    process_min_dt = _minimum_positive_delta(points)
    if process_min_dt is None:
        raise ValueError(
            "Bolus controls require at least one strictly positive online "
            "timestamp delta to compute min_dt."
        )
    return float(min(process_min_dt, duration_cap))


def build_sample_acc_source_default(
    process: BioProcess, run_min_dt: float | None = None
) -> SignalSource:
    sample_changes: list[tuple[float, float]] = []
    for volume_change in process.volume.volume_changes.values():
        if not isinstance(volume_change, SampleVolumeChange):
            continue
        times = _as_numpy(volume_change.values.times)
        values = _as_numpy(volume_change.values.values)
        for t, delta in zip(times.tolist(), values.tolist(), strict=False):
            sample_changes.append((float(t), abs(float(delta))))

    sample_changes.sort(key=lambda item: item[0])
    t_end = float(process.time_axis.end)

    times = [float(process.time_axis.start)]
    values = [0.0]
    cumulative = 0.0
    step_ts: list[float] = []

    ramp_duration = 0.0
    if sample_changes:
        ramp_duration = get_bolus_min_dt(process, run_min_dt=run_min_dt)

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


def build_bolus_sources(
    process: BioProcess, run_min_dt: float | None = None
) -> list[SignalSource]:
    t_start = float(process.time_axis.start)
    t_end = float(process.time_axis.end)
    sources: list[SignalSource] = []

    for name, volume_change in process.volume.volume_changes.items():
        if not isinstance(volume_change, FeedVolumeChange):
            continue
        if volume_change.is_continuous:
            continue
        if not volume_change.is_controlled:
            continue

        step_ts: list[float] = []
        triangles: list[tuple[float, float, float, float]] = []

        event_pairs = list(
            zip(
                _as_numpy(volume_change.values.times).tolist(),
                _as_numpy(volume_change.values.values).tolist(),
                strict=False,
            )
        )
        validated_events: list[tuple[float, float]] = []

        for event_time, delta_v in event_pairs:
            event_time = float(event_time)
            delta_v = float(delta_v)
            if not np.isfinite(event_time) or not np.isfinite(delta_v):
                raise ValueError(
                    f"Feed '{name}' has non-finite bolus event values: "
                    f"time={event_time}, delta_v={delta_v}."
                )
            if event_time < t_start:
                raise ValueError(
                    f"Feed '{name}' has bolus timestamp before process start "
                    f"({event_time} < {t_start}); this is not supported."
                )
            if event_time >= t_end:
                raise ValueError(
                    f"Feed '{name}' has bolus timestamp at/after process end "
                    f"({event_time} >= {t_end}); this is not supported."
                )
            if delta_v <= 0.0:
                raise ValueError(
                    f"Feed '{name}' has non-positive bolus delta_v at t={event_time}: "
                    f"{delta_v}. Bolus deltas must be strictly positive."
                )
            validated_events.append((event_time, delta_v))

        min_dt: float | None = None
        triangle_width = 0.0
        if validated_events:
            min_dt = get_bolus_min_dt(process, run_min_dt=run_min_dt)
            triangle_width = min_dt

            for event_time, delta_v in validated_events:
                peak_time = event_time + 0.5 * min_dt
                end_time = event_time + triangle_width
                if end_time > t_end:
                    raise ValueError(
                        f"Feed '{name}' has bolus event at t={event_time} that cannot "
                        f"fit triangle width min_dt={triangle_width} before process "
                        f"end t_end={t_end}."
                    )

                peak_rate = 2.0 * delta_v / min_dt
                triangles.append((event_time, peak_time, end_time, peak_rate))
                step_ts.extend([event_time, peak_time, end_time])

        breakpoints = [t_start, t_end]
        for event_start, event_peak, event_end, _ in triangles:
            breakpoints.extend([event_start, event_peak, event_end])

        times = np.asarray(_dedupe_sorted(breakpoints), dtype=float)
        values = np.zeros(times.size, dtype=float)
        for idx, t in enumerate(times.tolist()):
            total_rate = 0.0
            for event_start, event_peak, event_end, peak_rate in triangles:
                if t <= event_start or t >= event_end:
                    continue
                if t <= event_peak:
                    total_rate += (
                        peak_rate * (t - event_start) / (event_peak - event_start)
                    )
                else:
                    total_rate += peak_rate * (event_end - t) / (event_end - event_peak)
            values[idx] = total_rate

        source = _make_source_from_xy(
            name=name,
            kind="volume_change",
            times=times,
            values=values,
            metadata={
                "source": "bolus_triangle",
                "triangle_min_dt": min_dt,
                "triangle_width": triangle_width,
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

    run_min_dt = run_min_dt_from_config(config)
    for source in build_bolus_sources(process, run_min_dt=run_min_dt):
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


def run_min_dt_from_config(config: dict[str, Any]) -> float | None:
    run_min_dt_cfg = config.get(EVENT_RUN_MIN_DT_CONFIG_KEY)
    return None if run_min_dt_cfg is None else float(run_min_dt_cfg)


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
