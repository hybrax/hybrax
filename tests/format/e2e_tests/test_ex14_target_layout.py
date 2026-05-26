import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from bp_format.serialization import load_process_collection_json
from examples import validate_example
from tests.unit_tests.loader_helpers import load_module

EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2] / "examples/14_simulation_intracellular"
)
SIMULATION_DIR = EXAMPLE_ROOT / "00_simulation"
ALL_PROCESS_OUTPUT = EXAMPLE_ROOT / "02_all_processes" / "output"
SIMULATION_DENSE_OUTPUT = SIMULATION_DIR / "simulation_dense_output.csv"
EXPECTED_REQUIRED_COLUMNS = [
    "process_id",
    "time",
    "row_type",
    "biomass",
    "product_extracellular",
    "product_intracellular",
    "dead_cells",
    "glucose",
    "glutamine",
    "lactate",
    "ammonia",
    "intracellular_product_ratio",
    "volume",
]
EXPECTED_EXTRA_COLUMNS = {"pH", "temperature", "q_biomass", "r_biomass"}
ALLOWED_ROW_TYPES = {
    "online",
    "offline",
    "pre-event",
    "post-event",
    "fermentation_end",
}
EXPECTED_PROCESS_IDS = {"ex14_run_1", "ex14_run_2"}


def _copy_ex14_files(target: Path, *, include_outputs: bool) -> None:
    (target / "00_simulation").mkdir(parents=True)
    (target / "01_single_process").mkdir(parents=True)
    (target / "02_all_processes").mkdir(parents=True)
    shutil.copy2(EXAMPLE_ROOT / "load_utils.py", target / "load_utils.py")
    for name in ["ex14_simulation.py", "simulation_dense_output.csv", "events.csv"]:
        shutil.copy2(SIMULATION_DIR / name, target / "00_simulation" / name)
    for directory, loader_name in [
        ("01_single_process", "load_single_process.py"),
        ("02_all_processes", "load_all_processes.py"),
    ]:
        shutil.copy2(
            EXAMPLE_ROOT / directory / loader_name,
            target / directory / loader_name,
        )
        if include_outputs:
            (target / directory / "output").mkdir()
            shutil.copy2(
                EXAMPLE_ROOT / directory / "output" / "data.json",
                target / directory / "output" / "data.json",
            )


def _copy_ex14(tmp_path: Path) -> Path:
    target = tmp_path / "14_simulation_intracellular"
    _copy_ex14_files(target, include_outputs=True)
    (target / "03_validate").mkdir(parents=True)
    return target


def _copy_ex14_loader_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "14_simulation_intracellular"
    _copy_ex14_files(target, include_outputs=False)
    return target


def _summary(root: Path) -> dict:
    path = root / "03_validate" / "output" / "validation_summary.json"
    return json.loads(path.read_text())


def _perturb_first_event_row_state(
    root: Path,
    *,
    row_type: str,
    state_name: str,
    delta: float,
    keep_offline_matched: bool = False,
) -> None:
    path = root / "00_simulation" / "simulation_dense_output.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        rows = list(reader)
        fieldnames = reader.fieldnames

    perturbed_time = None
    for row in rows:
        if row["row_type"] == row_type:
            perturbed_time = row["time"]
            row[state_name] = str(float(row[state_name]) + delta)
            break
    else:  # pragma: no cover - fixture contract guard.
        raise AssertionError(f"expected at least one {row_type} row")

    if keep_offline_matched:
        for row in rows:
            if row["row_type"] == "offline" and row["time"] == perturbed_time:
                row[state_name] = str(float(row[state_name]) + delta)
                break

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_ex14_visible_loaders_regenerate_reloadable_outputs(tmp_path):
    single_loader = load_module(
        "ex14_single_loader",
        EXAMPLE_ROOT / "01_single_process" / "load_single_process.py",
    )
    all_loader = load_module(
        "ex14_all_loader",
        EXAMPLE_ROOT / "02_all_processes" / "load_all_processes.py",
    )

    all_output_dir = tmp_path / "all"
    all_output_dir.mkdir()
    (all_output_dir / "dense_truth.csv").write_text("stale\n")

    single = single_loader.generate_single_process_output(tmp_path / "single")
    all_processes = all_loader.generate_all_processes_output(all_output_dir)

    reloaded_single = load_process_collection_json(tmp_path / "single" / "data.json")
    reloaded_all = load_process_collection_json(tmp_path / "all" / "data.json")
    assert set(single.processes) == {"ex14_run_1"}
    assert set(reloaded_single.processes) == {"ex14_run_1"}
    assert set(all_processes.processes) == EXPECTED_PROCESS_IDS
    assert set(reloaded_all.processes) == EXPECTED_PROCESS_IDS
    assert not (tmp_path / "single" / "dense_truth.csv").exists()
    assert not (tmp_path / "all" / "dense_truth.csv").exists()


def test_ex14_loaders_consume_existing_simulation_artifacts(tmp_path):
    root = _copy_ex14_loader_fixture(tmp_path)
    single_loader_path = root / "01_single_process" / "load_single_process.py"
    all_loader_path = root / "02_all_processes" / "load_all_processes.py"
    assert "run_all_default" not in single_loader_path.read_text()
    assert "run_all_default" not in all_loader_path.read_text()
    assert "TemporaryDirectory" not in single_loader_path.read_text()
    assert "TemporaryDirectory" not in all_loader_path.read_text()

    sys.modules.pop("load_utils", None)
    sys.modules.pop("ex14_simulation", None)
    single_loader = load_module("copied_ex14_single_loader", single_loader_path)
    all_loader = load_module("copied_ex14_all_loader", all_loader_path)

    single_output_dir = tmp_path / "single_from_copy"
    all_output_dir = tmp_path / "all_from_copy"
    (all_output_dir).mkdir()
    (all_output_dir / "dense_truth.csv").write_text("stale\n")
    single = single_loader.generate_single_process_output(single_output_dir)
    all_processes = all_loader.generate_all_processes_output(all_output_dir)

    assert set(single.processes) == {"ex14_run_1"}
    assert set(all_processes.processes) == EXPECTED_PROCESS_IDS
    assert (single_output_dir / "data.json").exists()
    assert (all_output_dir / "data.json").exists()
    assert not (single_output_dir / "dense_truth.csv").exists()
    assert not (all_output_dir / "dense_truth.csv").exists()


def test_ex14_simulation_dense_output_contract_matches_all_process_json():
    collection = load_process_collection_json(ALL_PROCESS_OUTPUT / "data.json")
    json_process_ids = set(collection.processes)

    with SIMULATION_DENSE_OUTPUT.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert all(column in reader.fieldnames for column in EXPECTED_REQUIRED_COLUMNS)
        assert EXPECTED_EXTRA_COLUMNS <= set(reader.fieldnames)
        assert "process" not in reader.fieldnames

        process_ids = set()
        row_types = set()
        row_types_by_process = defaultdict(set)
        for row in reader:
            process_id = row["process_id"]
            row_type = row["row_type"]
            process_ids.add(process_id)
            row_types.add(row_type)
            row_types_by_process[process_id].add(row_type)

    assert json_process_ids == EXPECTED_PROCESS_IDS
    assert process_ids == json_process_ids
    assert row_types <= ALLOWED_ROW_TYPES
    assert row_types == ALLOWED_ROW_TYPES
    assert set(row_types_by_process) == json_process_ids
    for process_id in json_process_ids:
        assert row_types_by_process[process_id] == ALLOWED_ROW_TYPES


RAW_MAX_ABS_ERROR_BOUND = 100.0
PSEUDOBATCH_MAX_ABS_ERROR_BOUND = 80.0


def test_ex14_reintegration_script_writes_metrics_and_plots(tmp_path, capsys):
    root = _copy_ex14(tmp_path)
    script_path = root / "03_validate" / "verify_reintegration.py"
    shutil.copy2(EXAMPLE_ROOT / "03_validate" / "verify_reintegration.py", script_path)
    verify_reintegration = load_module("copied_ex14_verify_reintegration", script_path)

    exit_code = verify_reintegration.main([str(root)])

    assert exit_code == 0
    metrics_path = root / "03_validate" / "output" / "reintegration_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    assert metrics["ok"] is True
    assert set(metrics["processes"]) == EXPECTED_PROCESS_IDS
    # Lax accuracy bounds for raw/pseudobatch (current observed maxima are
    # ~54 and ~34); we only want to catch a regression, not noise wobble.
    # max_rel_error is unbounded for those modes because product_intracellular
    # hits zero at t0 and dwarfs the relative scale.
    mode_max_abs_bound = {
        "raw": RAW_MAX_ABS_ERROR_BOUND,
        "pseudobatch": PSEUDOBATCH_MAX_ABS_ERROR_BOUND,
        "dense_pseudobatch": verify_reintegration.DENSE_PSEUDOBATCH_MAX_ABS_ERROR,
    }
    for process_id in EXPECTED_PROCESS_IDS:
        process_metrics = metrics["processes"][process_id]
        assert set(process_metrics) == set(mode_max_abs_bound)
        for mode, max_abs_bound in mode_max_abs_bound.items():
            mode_metrics = process_metrics[mode]
            assert mode_metrics["point_count"] > 0
            assert mode_metrics["segment_count"] > 0
            assert mode_metrics["max_abs_error"] <= max_abs_bound
            if mode == "dense_pseudobatch":
                assert mode_metrics["max_rel_error"] <= (
                    verify_reintegration.DENSE_PSEUDOBATCH_MAX_REL_ERROR
                )
                assert mode_metrics["max_rhs_derivative_abs_residual"] <= (
                    verify_reintegration.DENSE_PSEUDOBATCH_MAX_RHS_RESIDUAL
                )
            plot_path = root / mode_metrics["plot_path"]
            assert plot_path.exists()
            assert plot_path.stat().st_size > 0
    capsys.readouterr()


def test_ex14_validator_dense_event_summary_passes(tmp_path, capsys):
    root = _copy_ex14(tmp_path)

    exit_code = validate_example.main([str(root)])

    assert exit_code == 0
    summary = _summary(root)
    simulation_dense_output = summary["simulation_dense_output"]
    dense_events = summary["dense_event_validation"]
    dense_trajectory = summary["dense_trajectory_validation"]
    assert simulation_dense_output["ok"] is True
    assert simulation_dense_output["row_count"] > 0
    assert set(simulation_dense_output["process_ids"]) == EXPECTED_PROCESS_IDS
    assert dense_events["ok"] is True
    assert dense_events["scope"] == "event-side physical validation only"
    assert dense_events["event_checks"] > 0
    assert dense_events["pre_event_online_checks"] > 0
    assert dense_events["pre_event_online_checks_skipped"] == 0
    assert dense_events["max_abs_state_error"] <= dense_events["volume_atol"]
    assert dense_events["max_abs_volume_error"] <= dense_events["volume_atol"]
    assert dense_trajectory["ok"] is True
    assert dense_trajectory["scope"] == validate_example.DENSE_TRAJECTORY_SCOPE
    assert "no trajectory recovery" in dense_trajectory["scope"]
    assert "no integrated-state comparison" in dense_trajectory["scope"]
    assert dense_trajectory["segment_count"] > 0
    assert dense_trajectory["diagnostic_point_count"] > 0
    assert dense_trajectory["max_diagnostic_points_per_segment"] == 2
    assert dense_trajectory["expected_rank"] == 8
    assert dense_trajectory["min_rank"] == dense_trajectory["expected_rank"]
    assert dense_trajectory["max_condition_number"] < 10.0
    assert dense_trajectory["max_inversion_abs_residual"] <= 1e-5
    assert dense_trajectory["max_inversion_relative_residual"] <= 1e-5
    assert dense_trajectory["max_volume_derivative_residual"] <= 1e-5
    for process_id in EXPECTED_PROCESS_IDS:
        process_summary = dense_events["processes"][process_id]
        assert process_summary["event_checks"] > 0
        assert process_summary["same_time_event_timestamps"]
        assert set(process_summary["event_kinds_checked"]) == {
            "bolus_feed",
            "sample",
        }
    text = (root / "03_validate" / "output" / "validation.txt").read_text()
    assert "Simulation dense output:" in text
    assert "Dense event validation:" in text
    assert "Dense trajectory diagnostics:" in text
    assert "event-side physical validation only" in text
    assert validate_example.DENSE_TRAJECTORY_SCOPE in text
    capsys.readouterr()


def test_ex14_validator_rejects_pre_event_online_mismatch(tmp_path, capsys):
    root = _copy_ex14(tmp_path)
    _perturb_first_event_row_state(
        root,
        row_type="pre-event",
        state_name="biomass",
        delta=1e-4,
        keep_offline_matched=True,
    )

    exit_code = validate_example.main([str(root)])

    assert exit_code == 1
    errors = _summary(root)["dense_event_validation"]["errors"]
    assert any("Pre-event state mismatch" in error for error in errors)
    capsys.readouterr()


def test_ex14_validator_rejects_small_state_relative_event_mismatch(
    tmp_path,
    capsys,
):
    root = _copy_ex14(tmp_path)
    _perturb_first_event_row_state(
        root,
        row_type="post-event",
        state_name="intracellular_product_ratio",
        delta=2e-6,
    )

    exit_code = validate_example.main([str(root)])

    assert exit_code == 1
    errors = _summary(root)["dense_event_validation"]["errors"]
    assert any("Post-event state mismatch" in error for error in errors)
    capsys.readouterr()


def test_ex14_dense_segments_are_side_aware_at_shared_sample_bolus_time():
    collection = load_process_collection_json(ALL_PROCESS_OUTPUT / "data.json")
    dense, dense_result = validate_example.validate_simulation_dense_output(
        SIMULATION_DENSE_OUTPUT,
        collection,
    )
    assert dense_result["ok"] is True
    assert dense is not None

    process = collection.processes["ex14_run_1"]
    ordering = validate_example.get_process_ordering(process)
    events = validate_example.extract_discrete_events(process, ordering)
    events_by_time = defaultdict(list)
    for event in events:
        events_by_time[float(event["t"])].append(event)
    shared_event_time = next(
        time for time, time_events in events_by_time.items() if len(time_events) > 1
    )

    segments, errors = validate_example.build_dense_reference_segments(
        process,
        dense.processes["ex14_run_1"],
        ordering.name_modeled_RMCs + ordering.name_modeled_PVs,
        events,
    )

    assert errors == []
    previous_segment = next(
        segment for segment in segments if segment.end_time == shared_event_time
    )
    next_segment = next(
        segment for segment in segments if segment.start_time == shared_event_time
    )
    assert previous_segment.end_row_type == "pre-event"
    assert next_segment.start_row_type == "post-event"
    assert "offline" not in previous_segment.row_types
    assert "offline" not in next_segment.row_types
    assert len(previous_segment.times) == len(set(previous_segment.times))
    assert len(next_segment.times) == len(set(next_segment.times))


def test_ex14_validator_fails_on_online_state_perturbation(tmp_path, capsys):
    root = _copy_ex14(tmp_path)
    simulation_dense_output = root / "00_simulation" / "simulation_dense_output.csv"
    collection = load_process_collection_json(
        root / "02_all_processes" / "output" / "data.json"
    )
    dense, dense_result = validate_example.validate_simulation_dense_output(
        simulation_dense_output,
        collection,
    )
    assert dense_result["ok"] is True
    assert dense is not None
    process = collection.processes["ex14_run_1"]
    ordering = validate_example.get_process_ordering(process)
    events = validate_example.extract_discrete_events(process, ordering)
    segments, errors = validate_example.build_dense_reference_segments(
        process,
        dense.processes["ex14_run_1"],
        ordering.name_modeled_RMCs + ordering.name_modeled_PVs,
        events,
    )
    assert errors == []
    diagnostic_time = validate_example.segment_diagnostic_times(segments[0])[0]

    with simulation_dense_output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = rows[0].keys()

    for row in rows:
        if (
            row["process_id"] == "ex14_run_1"
            and row["row_type"] == "online"
            and abs(float(row["time"]) - diagnostic_time) < 1e-12
        ):
            row["intracellular_product_ratio"] = str(
                float(row["intracellular_product_ratio"]) + 0.1
            )
            break

    with simulation_dense_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    exit_code = validate_example.main([str(root)])

    assert exit_code == 1
    summary = _summary(root)
    assert summary["simulation_dense_output"]["ok"] is True
    assert summary["dense_event_validation"]["ok"] is True
    assert summary["dense_trajectory_validation"]["ok"] is False
    assert summary["dense_trajectory_validation"]["errors"]
    capsys.readouterr()


@pytest.mark.parametrize("column", ["volume", "biomass"])
def test_ex14_validator_fails_on_post_event_perturbation(tmp_path, capsys, column):
    root = _copy_ex14(tmp_path)
    simulation_dense_output = root / "00_simulation" / "simulation_dense_output.csv"
    with simulation_dense_output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = rows[0].keys()

    for row in rows:
        if row["row_type"] == "post-event":
            row[column] = str(float(row[column]) + 1e-4)
            break

    with simulation_dense_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    exit_code = validate_example.main([str(root)])

    assert exit_code == 1
    summary = _summary(root)
    assert summary["simulation_dense_output"]["ok"] is True
    assert summary["dense_event_validation"]["ok"] is False
    assert summary["dense_event_validation"]["errors"]
    capsys.readouterr()
