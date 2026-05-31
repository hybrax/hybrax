from collections import defaultdict
from collections.abc import Iterable
import csv
from pathlib import Path

import bp_format as bp
import jax.numpy as jnp
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
SIMULATION_DENSE_OUTPUT = DATA_DIR / "simulation_dense_output.csv"
EVENTS_OUTPUT = DATA_DIR / "simulation_events.csv"
CANONICAL_ARTIFACTS = (
    Path("simulation_dense_output.csv"),
    Path("simulation_events.csv"),
    Path("simulation_plots/process_variables.png"),
    Path("simulation_plots/rates.png"),
    Path("simulation_plots/reactor_states_and_volumes.png"),
)
EXPECTED_PROCESS_IDS = {"sim_1_run_1", "sim_1_run_2"}
EXPECTED_REACTOR_COMPONENT_ORDER = (
    "biomass",
    "product_extracellular",
    "product_intracellular",
    "dead_cells",
    "glucose",
    "glutamine",
    "lactate",
    "ammonia",
)
CSTAR_SPLINE_SMOOTHING_S = 0.0
EVENT_TIME_ATOL = 1e-12


def _time_grid_from_volume_changes(
    volume_changes: Iterable[bp.VolumeChange],
) -> np.ndarray:
    return np.unique(
        np.concatenate(
            [
                np.asarray(volume_change.values.times, dtype=float)
                for volume_change in volume_changes
            ]
        )
    )


def _value_at_time(times: np.ndarray, values: np.ndarray, time: float) -> float:
    matches = np.flatnonzero(np.isclose(times, time, rtol=0.0, atol=EVENT_TIME_ATOL))
    if len(matches):
        return float(values[int(matches[0])])
    return float(np.interp(time, times, values))


def _as_timeseries(times: np.ndarray, values: np.ndarray, *, jump_times=None):
    if jump_times is None:
        jump_times = jnp.asarray([], dtype=float)
    return bp.TimeSeries(
        times=jnp.asarray(times),
        values=jnp.asarray(values),
        jump_times=jump_times,
    )


def populate_exact_pseudobatch_transform(process: bp.BioProcess) -> None:
    volume_changes = process.volume.volume_changes
    times = _time_grid_from_volume_changes(volume_changes.values())
    total_volume = np.full(len(times), process.volume.initial_volume, dtype=float)
    sample_deltas: defaultdict[float, float] = defaultdict(float)
    discrete_deltas: defaultdict[float, float] = defaultdict(float)

    for volume_change in volume_changes.values():
        change_times = np.asarray(volume_change.values.times, dtype=float)
        change_values = np.asarray(volume_change.values.values, dtype=float)
        assert len(change_times) == len(change_values)

        if volume_change.is_continuous:
            total_volume += np.interp(times, change_times, change_values)
            continue

        for time, value in zip(change_times, change_values, strict=True):
            total_volume[times > time] += value
            discrete_deltas[float(time)] += float(value)
            if isinstance(volume_change, bp.SampleVolumeChange):
                sample_deltas[float(time)] += float(value)

    jump_times = np.asarray(sorted(discrete_deltas), dtype=float)
    jump_times = jump_times[(times[0] < jump_times) & (jump_times < times[-1])]
    process.volume.total_volume = _as_timeseries(
        times,
        total_volume,
        jump_times=jnp.asarray(jump_times),
    )

    sample_compensation = np.ones(len(times), dtype=float)
    for time, delta in sorted(sample_deltas.items()):
        pre_sample_volume = _value_at_time(times, total_volume, time)
        post_sample_volume = pre_sample_volume + delta
        sample_compensation[times > time] *= pre_sample_volume / post_sample_volume

    accumulated_feeds = {}
    for name, volume_change in volume_changes.items():
        if not isinstance(volume_change, bp.FeedVolumeChange):
            continue
        change_times = np.asarray(volume_change.values.times, dtype=float)
        change_values = np.asarray(volume_change.values.values, dtype=float)
        if volume_change.is_continuous:
            accumulated_feed = np.interp(times, change_times, change_values)
        else:
            accumulated_feed = np.zeros(len(times), dtype=float)
            for time, value in zip(change_times, change_values, strict=True):
                accumulated_feed[times > time] += value
        accumulated_feeds[name] = _as_timeseries(times, accumulated_feed)

    adf = total_volume / process.volume.initial_volume * sample_compensation
    feed_corrections = {}
    for species_name in process.reactor_medium.components:
        feed_correction = np.zeros(len(times), dtype=float)

        for name, volume_change in volume_changes.items():
            if not isinstance(volume_change, bp.FeedVolumeChange):
                continue

            feed_concentration = volume_change.feed_medium.components[
                species_name
            ].concentration.value
            if feed_concentration == 0.0:
                continue

            change_times = np.asarray(volume_change.values.times, dtype=float)
            change_values = np.asarray(volume_change.values.values, dtype=float)

            if volume_change.is_continuous:
                cumulative_feed = np.asarray(
                    accumulated_feeds[name].values, dtype=float
                )
                feed_volume_added = np.diff(cumulative_feed, prepend=cumulative_feed[0])
                feed_correction += np.cumsum(
                    sample_compensation
                    / process.volume.initial_volume
                    * feed_volume_added
                    * feed_concentration
                )
                continue

            for time, feed_volume_added in zip(
                change_times,
                change_values,
                strict=True,
            ):
                sample_compensation_at_bolus = _value_at_time(
                    times,
                    sample_compensation,
                    float(time),
                )
                sample_delta = sample_deltas[float(time)]
                if sample_delta != 0.0:
                    pre_sample_volume = _value_at_time(times, total_volume, float(time))
                    post_sample_volume = pre_sample_volume + sample_delta
                    sample_compensation_at_bolus *= (
                        pre_sample_volume / post_sample_volume
                    )
                feed_correction[times > time] += (
                    sample_compensation_at_bolus
                    * feed_volume_added
                    * feed_concentration
                    / process.volume.initial_volume
                )

        feed_corrections[species_name] = _as_timeseries(times, feed_correction)

    process.pseudobatch_transform = bp.PseudobatchTransform(
        adf=_as_timeseries(times, adf),
        feed_corrections=feed_corrections,
        sample_compensation=_as_timeseries(times, sample_compensation),
        accumulated_feeds=accumulated_feeds,
    )


def _series_values_at_times(series: bp.TimeSeries, times: np.ndarray) -> np.ndarray:
    if series.times is None or series.values is None:
        raise ValueError("Stored pseudobatch transform carriers need raw samples.")
    series_times = np.asarray(series.times, dtype=float)
    series_values = np.asarray(series.values, dtype=float)
    return np.asarray(
        [
            _value_at_time(series_times, series_values, float(time))
            for time in np.asarray(times, dtype=float)
        ],
        dtype=float,
    )


def cstar_values_for_component(
    process: bp.BioProcess,
    component: bp.ReactorMediumComponent,
) -> tuple[np.ndarray, np.ndarray]:
    transform = process.pseudobatch_transform
    if transform is None:
        raise ValueError("process.pseudobatch_transform is required.")
    if component.name not in transform.feed_corrections:
        raise KeyError(component.name)
    if not isinstance(component.concentration, bp.TimeSeries):
        raise TypeError(f"{component.name} concentration must be a TimeSeries.")

    times = np.asarray(component.concentration.times, dtype=float)
    raw_values = np.asarray(component.concentration.values, dtype=float)
    adf = _series_values_at_times(transform.adf, times)
    feed_correction = _series_values_at_times(
        transform.feed_corrections[component.name],
        times,
    )
    return times, raw_values * adf - feed_correction


def fit_cstar_timeseries(
    process: bp.BioProcess,
    component: bp.ReactorMediumComponent,
) -> bp.TimeSeries:
    times, cstar_values = cstar_values_for_component(process, component)
    fitted = bp.splines.fit_timeseries_spline(
        bp.TimeSeries(times=jnp.asarray(times), values=jnp.asarray(cstar_values)),
        smoothing_s=CSTAR_SPLINE_SMOOTHING_S,
    )
    metadata = dict(fitted.metadata or {})
    metadata["transform"] = {
        "name": "pseudo_batch",
        "component": component.name,
        "source": "stored_pseudobatch_transform",
    }
    return bp.TimeSeries(
        times=fitted.times,
        values=fitted.values,
        jump_times=fitted.jump_times,
        breaks=fitted.breaks,
        coeffs=fitted.coeffs,
        segment_start_piece_idx=fitted.segment_start_piece_idx,
        continuity_side=fitted.continuity_side,
        metadata=metadata,
        dtype=fitted.dtype,
    )


def event_jump_times(process: bp.BioProcess) -> np.ndarray:
    assert process.volume.total_volume is not None
    return np.asarray(process.volume.total_volume.jump_times, dtype=float)


def _is_event_jump_time(time: float, jump_times: np.ndarray) -> bool:
    return bool(np.any(np.isclose(jump_times, time, rtol=0.0, atol=EVENT_TIME_ATOL)))


def _discrete_volume_delta_at(process: bp.BioProcess, time: float) -> float:
    delta = 0.0
    for volume_change in process.volume.volume_changes.values():
        if volume_change.is_continuous:
            continue
        change_times = np.asarray(volume_change.values.times, dtype=float)
        change_values = np.asarray(volume_change.values.values, dtype=float)
        matches = np.flatnonzero(
            np.isclose(change_times, time, rtol=0.0, atol=EVENT_TIME_ATOL)
        )
        if len(matches):
            delta += float(change_values[int(matches[0])])
    return delta


def _sample_volume_delta_at(process: bp.BioProcess, time: float) -> float:
    delta = 0.0
    for volume_change in process.volume.volume_changes.values():
        if not isinstance(volume_change, bp.SampleVolumeChange):
            continue
        change_times = np.asarray(volume_change.values.times, dtype=float)
        change_values = np.asarray(volume_change.values.values, dtype=float)
        matches = np.flatnonzero(
            np.isclose(change_times, time, rtol=0.0, atol=EVENT_TIME_ATOL)
        )
        if len(matches):
            delta += float(change_values[int(matches[0])])
    return delta


def _sample_compensation_after_sample_event(
    process: bp.BioProcess,
    time: float,
) -> float:
    transform = process.pseudobatch_transform
    assert transform is not None
    assert transform.sample_compensation is not None
    assert process.volume.total_volume is not None
    sample_compensation = _value_at_time(
        np.asarray(transform.sample_compensation.times, dtype=float),
        np.asarray(transform.sample_compensation.values, dtype=float),
        time,
    )
    sample_delta = _sample_volume_delta_at(process, time)
    if sample_delta == 0.0:
        return sample_compensation
    pre_sample_volume = _value_at_time(
        np.asarray(process.volume.total_volume.times, dtype=float),
        np.asarray(process.volume.total_volume.values, dtype=float),
        time,
    )
    post_sample_volume = pre_sample_volume + sample_delta
    return sample_compensation * pre_sample_volume / post_sample_volume


def _adf_jump_delta_at(process: bp.BioProcess, time: float) -> float:
    transform = process.pseudobatch_transform
    assert transform is not None
    assert process.volume.total_volume is not None
    total_volume_times = np.asarray(process.volume.total_volume.times, dtype=float)
    total_volume_values = np.asarray(process.volume.total_volume.values, dtype=float)
    adf_times = np.asarray(transform.adf.times, dtype=float)
    adf_values = np.asarray(transform.adf.values, dtype=float)
    pre_adf = _value_at_time(adf_times, adf_values, time)
    volume_delta = _discrete_volume_delta_at(process, time)
    sample_delta = _sample_volume_delta_at(process, time)
    if volume_delta == 0.0 and sample_delta == 0.0:
        return 0.0
    pre_total_volume = _value_at_time(total_volume_times, total_volume_values, time)
    post_total_volume = pre_total_volume + volume_delta
    sample_compensation = _sample_compensation_after_sample_event(process, time)
    post_adf = post_total_volume / process.volume.initial_volume * sample_compensation
    return post_adf - pre_adf


def _feed_correction_jump_delta_at(
    process: bp.BioProcess,
    component_name: str,
    time: float,
) -> float:
    sample_compensation = _sample_compensation_after_sample_event(process, time)
    delta = 0.0
    for volume_change in process.volume.volume_changes.values():
        if volume_change.is_continuous:
            continue
        if not isinstance(volume_change, bp.FeedVolumeChange):
            continue
        change_times = np.asarray(volume_change.values.times, dtype=float)
        change_values = np.asarray(volume_change.values.values, dtype=float)
        matches = np.flatnonzero(
            np.isclose(change_times, time, rtol=0.0, atol=EVENT_TIME_ATOL)
        )
        if not len(matches):
            continue
        feed_concentration = volume_change.feed_medium.components[
            component_name
        ].concentration.value
        delta += (
            sample_compensation
            * float(change_values[int(matches[0])])
            * feed_concentration
            / process.volume.initial_volume
        )
    return delta


def _event_aware_series_value(
    series: bp.TimeSeries,
    time: float,
    jump_times: np.ndarray,
    jump_delta_at,
    *,
    right_of_jump: bool = False,
) -> float:
    if series.times is None or series.values is None:
        raise ValueError("Stored pseudobatch transform carriers need raw samples.")
    times = np.asarray(series.times, dtype=float)
    values = np.asarray(series.values, dtype=float)
    if time < times[0] or time > times[-1]:
        raise ValueError(f"Time {time} is outside the transform carrier domain.")

    exact_matches = np.flatnonzero(
        np.isclose(times, time, rtol=0.0, atol=EVENT_TIME_ATOL)
    )
    if len(exact_matches):
        index = int(exact_matches[0])
        value = float(values[index])
        if right_of_jump and _is_event_jump_time(time, jump_times):
            value += jump_delta_at(time)
        return value

    left_index = int(np.searchsorted(times, time, side="right") - 1)
    right_index = left_index + 1
    left_time = float(times[left_index])
    left_value = float(values[left_index])
    if _is_event_jump_time(left_time, jump_times):
        left_value += jump_delta_at(left_time)
    return float(
        np.interp(
            time,
            [left_time, times[right_index]],
            [left_value, values[right_index]],
        )
    )


def event_aware_adf_value(
    process: bp.BioProcess,
    time: float,
    jump_times: np.ndarray,
    *,
    right_of_jump: bool = False,
) -> float:
    transform = process.pseudobatch_transform
    assert transform is not None
    return _event_aware_series_value(
        transform.adf,
        time,
        jump_times,
        lambda jump_time: _adf_jump_delta_at(process, jump_time),
        right_of_jump=right_of_jump,
    )


def event_aware_feed_correction_value(
    process: bp.BioProcess,
    component_name: str,
    time: float,
    jump_times: np.ndarray,
    *,
    right_of_jump: bool = False,
) -> float:
    transform = process.pseudobatch_transform
    assert transform is not None
    return _event_aware_series_value(
        transform.feed_corrections[component_name],
        time,
        jump_times,
        lambda jump_time: _feed_correction_jump_delta_at(
            process,
            component_name,
            jump_time,
        ),
        right_of_jump=right_of_jump,
    )


def dense_reactor_reference(process_id: str, max_time: float) -> dict[str, np.ndarray]:
    with SIMULATION_DENSE_OUTPUT.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["process_id"] == process_id
            and row["row_type"] != "offline"
            and float(row["time"]) <= max_time
        ]
    return {
        "time": np.asarray([float(row["time"]) for row in rows], dtype=float),
        **{
            name: np.asarray([float(row[name]) for row in rows], dtype=float)
            for name in EXPECTED_REACTOR_COMPONENT_ORDER
        },
    }
