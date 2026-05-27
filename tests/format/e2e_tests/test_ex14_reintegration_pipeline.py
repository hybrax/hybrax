import os
import sys
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")

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


def _series_values(series):
    return [float(value) for value in series.values]


def _series_times(series):
    return [float(time) for time in series.times]


def _assert_series_equal(actual, expected) -> None:
    assert _series_times(actual) == _series_times(expected)
    assert _series_values(actual) == _series_values(expected)


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
