import csv
import importlib.util
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pytest

from bp_format.serialization import load_process_collection_json
from examples import validate_example

EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[1] / "examples/14_simulation_intracellular"
)
ALL_PROCESS_OUTPUT = EXAMPLE_ROOT / "02_all_processes" / "output"
DENSE_TRUTH = ALL_PROCESS_OUTPUT / "dense_truth.csv"
EXPECTED_HEADER = [
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
ALLOWED_ROW_TYPES = {"online", "offline", "pre-event", "post-event"}
EXPECTED_PROCESS_IDS = {"ex14_run_1", "ex14_run_2"}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_ex14(tmp_path: Path) -> Path:
    target = tmp_path / "14_simulation_intracellular"
    (target / "01_single_process" / "output").mkdir(parents=True)
    (target / "02_all_processes" / "output").mkdir(parents=True)
    (target / "03_validate").mkdir(parents=True)
    shutil.copy2(
        EXAMPLE_ROOT / "01_single_process" / "load_single_process.py",
        target / "01_single_process" / "load_single_process.py",
    )
    shutil.copy2(
        EXAMPLE_ROOT / "01_single_process" / "output" / "data.json",
        target / "01_single_process" / "output" / "data.json",
    )
    shutil.copy2(
        EXAMPLE_ROOT / "02_all_processes" / "load_all_processes.py",
        target / "02_all_processes" / "load_all_processes.py",
    )
    shutil.copy2(
        EXAMPLE_ROOT / "02_all_processes" / "output" / "data.json",
        target / "02_all_processes" / "output" / "data.json",
    )
    shutil.copy2(
        DENSE_TRUTH, target / "02_all_processes" / "output" / "dense_truth.csv"
    )
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
    path = root / "02_all_processes" / "output" / "dense_truth.csv"
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
    assert not (EXAMPLE_ROOT / "_target_generation.py").exists()

    single_loader = _load_module(
        "ex14_single_loader",
        EXAMPLE_ROOT / "01_single_process" / "load_single_process.py",
    )
    all_loader = _load_module(
        "ex14_all_loader",
        EXAMPLE_ROOT / "02_all_processes" / "load_all_processes.py",
    )

    single = single_loader.generate_single_process_output(tmp_path / "single")
    all_processes = all_loader.generate_all_processes_output(tmp_path / "all")

    reloaded_single = load_process_collection_json(tmp_path / "single" / "data.json")
    reloaded_all = load_process_collection_json(tmp_path / "all" / "data.json")
    assert set(single.processes) == {"ex14_run_1"}
    assert set(reloaded_single.processes) == {"ex14_run_1"}
    assert set(all_processes.processes) == EXPECTED_PROCESS_IDS
    assert set(reloaded_all.processes) == EXPECTED_PROCESS_IDS
    assert not (tmp_path / "single" / "dense_truth.csv").exists()
    assert (tmp_path / "all" / "dense_truth.csv").exists()


def test_ex14_dense_truth_contract_matches_all_process_json():
    collection = load_process_collection_json(ALL_PROCESS_OUTPUT / "data.json")
    json_process_ids = set(collection.processes)

    with DENSE_TRUTH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == EXPECTED_HEADER
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


def test_ex14_validator_dense_event_summary_passes(tmp_path, capsys):
    root = _copy_ex14(tmp_path)

    exit_code = validate_example.main([str(root)])

    assert exit_code == 0
    summary = _summary(root)
    dense_truth = summary["dense_truth"]
    dense_events = summary["dense_event_validation"]
    dense_trajectory = summary["dense_trajectory_validation"]
    assert dense_truth["ok"] is True
    assert dense_truth["row_count"] > 0
    assert set(dense_truth["process_ids"]) == EXPECTED_PROCESS_IDS
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
    assert "Dense truth:" in text
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
    dense, dense_result = validate_example.validate_dense_truth(DENSE_TRUTH, collection)
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
    dense_truth = root / "02_all_processes" / "output" / "dense_truth.csv"
    collection = load_process_collection_json(
        root / "02_all_processes" / "output" / "data.json"
    )
    dense, dense_result = validate_example.validate_dense_truth(dense_truth, collection)
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

    with dense_truth.open(newline="") as handle:
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

    with dense_truth.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    exit_code = validate_example.main([str(root)])

    assert exit_code == 1
    summary = _summary(root)
    assert summary["dense_truth"]["ok"] is True
    assert summary["dense_event_validation"]["ok"] is True
    assert summary["dense_trajectory_validation"]["ok"] is False
    assert summary["dense_trajectory_validation"]["errors"]
    capsys.readouterr()


@pytest.mark.parametrize("column", ["volume", "biomass"])
def test_ex14_validator_fails_on_post_event_perturbation(tmp_path, capsys, column):
    root = _copy_ex14(tmp_path)
    dense_truth = root / "02_all_processes" / "output" / "dense_truth.csv"
    with dense_truth.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = rows[0].keys()

    for row in rows:
        if row["row_type"] == "post-event":
            row[column] = str(float(row[column]) + 1e-4)
            break

    with dense_truth.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    exit_code = validate_example.main([str(root)])

    assert exit_code == 1
    summary = _summary(root)
    assert summary["dense_truth"]["ok"] is True
    assert summary["dense_event_validation"]["ok"] is False
    assert summary["dense_event_validation"]["errors"]
    capsys.readouterr()
