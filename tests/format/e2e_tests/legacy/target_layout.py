"""Archived target-layout reference checks.

This module is intentionally not collected by pytest. It depends on legacy
fixtures that are no longer part of the active e2e suite.
"""

import csv
import os
import shutil
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")

from bp_format.serialization import load_process_collection_json
from tests.e2e_tests.legacy.loader_helpers import load_module

FIXTURE_ROOT = Path(__file__).resolve().parent / "ex14_fixture"
SIMULATION_DIR = FIXTURE_ROOT / "00_simulation"
ALL_PROCESS_OUTPUT = FIXTURE_ROOT / "02_all_processes" / "output"
SIMULATION_DENSE_OUTPUT = SIMULATION_DIR / "simulation_dense_output.csv"
EVENTS_OUTPUT = SIMULATION_DIR / "events.csv"
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
ALLOWED_EVENT_TYPES = {"sample", "bolus", "fermentation_end"}
EXPECTED_PROCESS_IDS = {"ex14_run_1", "ex14_run_2"}


def _clear_ex14_loader_modules() -> None:
    sys.modules.pop("load_utils", None)
    sys.modules.pop("ex14_simulation", None)


@contextmanager
def _isolated_ex14_loader_imports():
    sys_path = list(sys.path)
    dont_write_bytecode = sys.dont_write_bytecode
    _clear_ex14_loader_modules()
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.path[:] = sys_path
        sys.dont_write_bytecode = dont_write_bytecode
        _clear_ex14_loader_modules()


def _copy_ex14_loader_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "14_simulation_intracellular"
    shutil.copytree(
        FIXTURE_ROOT,
        target,
        ignore=shutil.ignore_patterns("output", "__pycache__"),
    )
    return target


def test_ex14_visible_loaders_regenerate_reloadable_outputs(tmp_path):
    with _isolated_ex14_loader_imports():
        single_loader = load_module(
            "ex14_single_loader",
            FIXTURE_ROOT / "01_single_process" / "load_single_process.py",
        )
        all_loader = load_module(
            "ex14_all_loader",
            FIXTURE_ROOT / "02_all_processes" / "load_all_processes.py",
        )

        single_output_dir = tmp_path / "single"
        all_output_dir = tmp_path / "all"
        all_output_dir.mkdir()
        (all_output_dir / "dense_truth.csv").write_text("stale\n")

        single = single_loader.generate_single_process_output(single_output_dir)
        all_processes = all_loader.generate_all_processes_output(all_output_dir)

        reloaded_single = load_process_collection_json(single_output_dir / "data.json")
        reloaded_all = load_process_collection_json(all_output_dir / "data.json")
        assert set(single.processes) == {"ex14_run_1"}
        assert set(reloaded_single.processes) == {"ex14_run_1"}
        assert set(all_processes.processes) == EXPECTED_PROCESS_IDS
        assert set(reloaded_all.processes) == EXPECTED_PROCESS_IDS
        assert (single_output_dir / "data.json").read_text() == (
            FIXTURE_ROOT / "01_single_process" / "output" / "data.json"
        ).read_text()
        assert (all_output_dir / "data.json").read_text() == (
            FIXTURE_ROOT / "02_all_processes" / "output" / "data.json"
        ).read_text()
        assert not (all_output_dir / "dense_truth.csv").exists()


def test_ex14_loaders_consume_existing_simulation_artifacts(tmp_path):
    root = _copy_ex14_loader_fixture(tmp_path)
    single_loader_path = root / "01_single_process" / "load_single_process.py"
    all_loader_path = root / "02_all_processes" / "load_all_processes.py"
    assert "run_all_default" not in single_loader_path.read_text()
    assert "run_all_default" not in all_loader_path.read_text()
    assert "TemporaryDirectory" not in single_loader_path.read_text()
    assert "TemporaryDirectory" not in all_loader_path.read_text()

    with _isolated_ex14_loader_imports():
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


def test_ex14_events_contract_matches_dense_output_processes():
    with SIMULATION_DENSE_OUTPUT.open(newline="") as handle:
        dense_process_ids = {row["process_id"] for row in csv.DictReader(handle)}

    with EVENTS_OUTPUT.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert [*reader.fieldnames[:5]] == [
            "process_id",
            "time",
            "event_order",
            "event_type",
            "delta_volume",
        ]
        rows = list(reader)

    event_process_ids = {row["process_id"] for row in rows}
    event_types = {row["event_type"] for row in rows}
    events_by_process_time = defaultdict(list)
    for row in rows:
        event_type = row["event_type"]
        delta_volume = float(row["delta_volume"])
        assert event_type in ALLOWED_EVENT_TYPES
        if event_type == "sample":
            assert delta_volume < 0.0
        elif event_type == "bolus":
            assert delta_volume > 0.0
        else:
            assert delta_volume == 0.0
        events_by_process_time[(row["process_id"], row["time"])].append(row)

    assert event_process_ids == dense_process_ids == EXPECTED_PROCESS_IDS
    assert event_types == ALLOWED_EVENT_TYPES
    for key, group in events_by_process_time.items():
        ordered = sorted(group, key=lambda row: int(row["event_order"]))
        assert [int(row["event_order"]) for row in ordered] == list(range(len(group)))
        event_order = [row["event_type"] for row in ordered]
        assert event_order in (
            ["sample"],
            ["bolus"],
            ["sample", "bolus"],
            ["fermentation_end"],
        ), key
