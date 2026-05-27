from collections import defaultdict
from collections.abc import Iterable
import os
import sys
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")

import bp_format as bp  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from bp_format.serialization import (  # noqa: E402
    load_process_collection_json,
    save_process_collection_json,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "ex14_fixture"
SIMULATION_DIR = FIXTURE_ROOT / "00_simulation"
SIMULATION_DENSE_OUTPUT = SIMULATION_DIR / "simulation_dense_output.csv"
EVENTS_OUTPUT = SIMULATION_DIR / "events.csv"
CANONICAL_ARTIFACTS = ("simulation_dense_output.csv", "events.csv")
EXPECTED_PROCESS_IDS = {"ex14_run_1", "ex14_run_2"}
EXPECTED_REACTOR_COMPONENTS = {
    "biomass",
    "product_extracellular",
    "product_intracellular",
    "dead_cells",
    "glucose",
    "glutamine",
    "lactate",
    "ammonia",
}
EXPECTED_CONCENTRATION_TIMES = [0.0, 24.0, 48.0, 72.0, 96.0]
EXPECTED_SAMPLE_EVENT_TIMES = [24.0, 48.0, 72.0, 96.0]
EXPECTED_EVENT_JUMP_TIMES = [24.0, 36.0, 48.0, 72.0, 96.0, 108.0]
EXPECTED_NONZERO_FEED_CORRECTION_COMPONENTS = {"glucose", "glutamine"}
EXPECTED_VOLUME_CHANGES = {"conti_feed", "base_feed", "sampling", "bolus_feed"}


def _clear_ex14_fixture_modules() -> None:
    sys.modules.pop("load_utils", None)
    sys.modules.pop("ex14_simulation", None)


@contextmanager
def _isolated_ex14_fixture_imports():
    sys_path = list(sys.path)
    dont_write_bytecode = sys.dont_write_bytecode
    _clear_ex14_fixture_modules()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(FIXTURE_ROOT))
    sys.path.insert(0, str(SIMULATION_DIR))
    try:
        yield
    finally:
        sys.path[:] = sys_path
        sys.dont_write_bytecode = dont_write_bytecode
        _clear_ex14_fixture_modules()


def _regenerate_simulation_artifacts(output_dir: Path) -> Path:
    with _isolated_ex14_fixture_imports():
        from ex14_simulation import run_all_default

        run_all_default(output_dir=output_dir)
    return output_dir


def _parse_lab_like_collection():
    with _isolated_ex14_fixture_imports():
        from load_utils import parse_all_processes

        return parse_all_processes(
            dense_csv=SIMULATION_DENSE_OUTPUT,
            events_csv=EVENTS_OUTPUT,
            collection_name="ex14_lab_like_e2e",
        )


def _array_values(values):
    return [float(value) for value in values]


def _series_values(series):
    return _array_values(series.values)


def _series_times(series):
    return _array_values(series.times)


def _assert_series_equal(actual, expected) -> None:
    assert _series_times(actual) == _series_times(expected)
    assert _series_values(actual) == _series_values(expected)
    assert _array_values(actual.jump_times) == _array_values(expected.jump_times)


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
    matches = np.flatnonzero(times == time)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one grid entry at time {time}.")
    return float(values[int(matches[0])])


def _as_timeseries(times: np.ndarray, values: np.ndarray, **kwargs) -> bp.TimeSeries:
    return bp.TimeSeries(
        times=jnp.asarray(times),
        values=jnp.asarray(values),
        **kwargs,
    )


def _populate_exact_pseudobatch_transform(process: bp.BioProcess) -> None:
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


def _assert_finite_positive(series: bp.TimeSeries) -> None:
    values = np.asarray(series.values, dtype=float)
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)


def _assert_finite_nonnegative(series: bp.TimeSeries) -> None:
    values = np.asarray(series.values, dtype=float)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)


def _assert_exact_pseudobatch_transform_sane(process: bp.BioProcess) -> None:
    times = _time_grid_from_volume_changes(process.volume.volume_changes.values())
    total_volume = process.volume.total_volume
    transform = process.pseudobatch_transform
    assert total_volume is not None
    assert transform is not None
    assert _series_times(total_volume) == list(times)
    assert _series_times(transform.adf) == list(times)
    assert set(transform.feed_corrections) == set(process.reactor_medium.components)
    assert _series_times(total_volume) == _series_times(transform.adf)

    assert _array_values(total_volume.jump_times) == EXPECTED_EVENT_JUMP_TIMES
    _assert_finite_positive(total_volume)
    _assert_finite_positive(transform.adf)
    assert transform.sample_compensation is not None
    assert _series_times(transform.sample_compensation) == list(times)
    _assert_finite_positive(transform.sample_compensation)
    assert set(transform.accumulated_feeds) == {
        name
        for name, volume_change in process.volume.volume_changes.items()
        if isinstance(volume_change, bp.FeedVolumeChange)
    }
    for accumulated_feed in transform.accumulated_feeds.values():
        assert _series_times(accumulated_feed) == list(times)
        _assert_finite_nonnegative(accumulated_feed)

    final_volume = process.volume.initial_volume
    final_time = float(times[-1])
    for volume_change in process.volume.volume_changes.values():
        change_times = np.asarray(volume_change.values.times, dtype=float)
        change_values = np.asarray(volume_change.values.values, dtype=float)
        if volume_change.is_continuous:
            final_volume += float(np.interp(final_time, change_times, change_values))
        else:
            for time, value in zip(change_times, change_values, strict=True):
                if final_time > time:
                    final_volume += float(value)
    assert float(total_volume.values[-1]) == final_volume

    for species_name in EXPECTED_NONZERO_FEED_CORRECTION_COMPONENTS:
        feed_correction = transform.feed_corrections[species_name]
        _assert_finite_nonnegative(feed_correction)
        assert float(feed_correction.values[-1]) > 0.0
    for species_name, feed_correction in transform.feed_corrections.items():
        assert _series_times(feed_correction) == list(times)
        _assert_finite_nonnegative(feed_correction)
        if species_name not in EXPECTED_NONZERO_FEED_CORRECTION_COMPONENTS:
            assert _series_values(feed_correction) == [0.0] * len(times)


def _assert_pseudobatch_transform_equal(actual, expected) -> None:
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    _assert_series_equal(actual.adf, expected.adf)
    assert set(actual.feed_corrections) == set(expected.feed_corrections)
    for name, series in expected.feed_corrections.items():
        _assert_series_equal(actual.feed_corrections[name], series)
    if expected.sample_compensation is None:
        assert actual.sample_compensation is None
    else:
        _assert_series_equal(actual.sample_compensation, expected.sample_compensation)
    assert set(actual.accumulated_feeds) == set(expected.accumulated_feeds)
    for name, series in expected.accumulated_feeds.items():
        _assert_series_equal(actual.accumulated_feeds[name], series)


def _assert_feed_media_equal(actual, expected) -> None:
    assert actual.name == expected.name
    assert actual.density == expected.density
    assert actual.density_unit == expected.density_unit
    assert set(actual.components) == set(expected.components)
    for name, component in expected.components.items():
        actual_component = actual.components[name]
        assert actual_component.name == component.name
        assert actual_component.unit == component.unit
        assert actual_component.is_controlled == component.is_controlled
        assert actual_component.concentration.value == component.concentration.value


def _assert_reloaded_collection_matches_parsed(parsed, reloaded) -> None:
    assert reloaded.metadata == parsed.metadata
    assert set(reloaded.processes) == set(parsed.processes)
    for process_id, process in parsed.processes.items():
        reloaded_process = reloaded.processes[process_id]
        assert reloaded_process.metadata == process.metadata
        assert reloaded_process.time_axis == process.time_axis
        assert reloaded_process.volume.initial_volume == process.volume.initial_volume
        assert reloaded_process.volume.unit == process.volume.unit
        if process.volume.total_volume is None:
            assert reloaded_process.volume.total_volume is None
        else:
            _assert_series_equal(
                reloaded_process.volume.total_volume,
                process.volume.total_volume,
            )
        assert reloaded_process.reactor_medium.name == process.reactor_medium.name
        assert reloaded_process.reactor_medium.density == process.reactor_medium.density
        assert (
            reloaded_process.reactor_medium.density_unit
            == process.reactor_medium.density_unit
        )
        assert set(reloaded_process.reactor_medium.components) == set(
            process.reactor_medium.components
        )
        assert set(reloaded_process.process_variables) == set(process.process_variables)
        assert set(reloaded_process.volume.volume_changes) == set(
            process.volume.volume_changes
        )
        assert reloaded_process.biological_ode == process.biological_ode

        for name, component in process.reactor_medium.components.items():
            reloaded_component = reloaded_process.reactor_medium.components[name]
            assert reloaded_component.name == component.name
            assert reloaded_component.unit == component.unit
            assert reloaded_component.bounds == component.bounds
            assert (
                reloaded_component.c_star_concentration
                == component.c_star_concentration
            )
            _assert_series_equal(
                reloaded_component.concentration,
                component.concentration,
            )

        for name, variable in process.process_variables.items():
            reloaded_variable = reloaded_process.process_variables[name]
            assert reloaded_variable.name == variable.name
            assert reloaded_variable.unit == variable.unit
            assert reloaded_variable.is_controlled == variable.is_controlled
            assert reloaded_variable.bounds == variable.bounds
            _assert_series_equal(reloaded_variable.values, variable.values)

        for name, volume_change in process.volume.volume_changes.items():
            reloaded_volume_change = reloaded_process.volume.volume_changes[name]
            assert type(reloaded_volume_change) is type(volume_change)
            assert reloaded_volume_change.name == volume_change.name
            assert reloaded_volume_change.unit == volume_change.unit
            assert reloaded_volume_change.is_controlled == volume_change.is_controlled
            assert reloaded_volume_change.is_continuous == volume_change.is_continuous
            _assert_series_equal(reloaded_volume_change.values, volume_change.values)
            if hasattr(volume_change, "feed_medium"):
                _assert_feed_media_equal(
                    reloaded_volume_change.feed_medium,
                    volume_change.feed_medium,
                )

        _assert_pseudobatch_transform_equal(
            reloaded_process.pseudobatch_transform,
            process.pseudobatch_transform,
        )


def test_ex14_simulation_artifacts_match_tracked_canonical_files(tmp_path):
    regenerated_dir = _regenerate_simulation_artifacts(tmp_path / "simulation")

    for filename in CANONICAL_ARTIFACTS:
        tracked_path = SIMULATION_DIR / filename
        regenerated_path = regenerated_dir / filename
        assert regenerated_path.read_bytes() == tracked_path.read_bytes(), (
            f"ex14 simulation artifact drifted: {filename}\n"
            f"tracked: {tracked_path}\n"
            f"regenerated: {regenerated_path}\n"
            "Inspect both files. If the change is intentional, regenerate the "
            "tracked fixture artifact and commit it with the simulation change."
        )


def test_ex14_lab_like_csv_parse_roundtrips_basic_json(tmp_path):
    collection = _parse_lab_like_collection()

    assert collection.metadata["name"] == "ex14_lab_like_e2e"
    assert set(collection.processes) == EXPECTED_PROCESS_IDS
    for process in collection.processes.values():
        assert process.time_axis.start == 0.0
        assert process.time_axis.end == 120.0
        assert set(process.reactor_medium.components) == EXPECTED_REACTOR_COMPONENTS
        assert set(process.volume.volume_changes) == EXPECTED_VOLUME_CHANGES

        for component in process.reactor_medium.components.values():
            assert (
                _series_times(component.concentration) == EXPECTED_CONCENTRATION_TIMES
            )
            assert len(component.concentration.times) == 5

        ph = process.process_variables["pH"].values
        assert _series_times(ph)[0] == 0.0
        assert _series_times(ph)[-1] == 120.0
        assert len(ph.times) > len(EXPECTED_CONCENTRATION_TIMES)

        sampling = process.volume.volume_changes["sampling"].values
        assert _series_times(sampling) == EXPECTED_SAMPLE_EVENT_TIMES
        assert _series_values(sampling) == [-0.05, -0.05, -0.05, -0.05]

        bolus = process.volume.volume_changes["bolus_feed"].values
        assert _series_times(bolus) == [36.0, 72.0, 108.0]
        assert all(value > 0.0 for value in _series_values(bolus))

    output_path = tmp_path / "parsed_collection.json"
    save_process_collection_json(collection, output_path)
    reloaded = load_process_collection_json(output_path)
    _assert_reloaded_collection_matches_parsed(collection, reloaded)


def test_ex14_exact_pseudobatch_transform_is_sane_and_roundtrips(tmp_path):
    collection = _parse_lab_like_collection()

    for process in collection.processes.values():
        _populate_exact_pseudobatch_transform(process)
        _assert_exact_pseudobatch_transform_sane(process)
        for component in process.reactor_medium.components.values():
            assert component.c_star_concentration is None

    output_path = tmp_path / "transformed_collection.json"
    save_process_collection_json(collection, output_path)
    reloaded = load_process_collection_json(output_path)
    _assert_reloaded_collection_matches_parsed(collection, reloaded)
