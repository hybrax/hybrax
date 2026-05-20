import json
import shutil
from pathlib import Path

from bp_format.serialization import load_process_collection_json
from examples import validate_example
from tests.loader_helpers import load_module

EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "examples/01_kittler_2022"
SINGLE_OUTPUT = EXAMPLE_ROOT / "01_single_process" / "output"
ALL_PROCESS_OUTPUT = EXAMPLE_ROOT / "02_all_processes" / "output"
SINGLE_PROCESS_ID = "DoE1_R1"
EXPECTED_PROCESS_COUNT = 12


def _copy_ex01(tmp_path: Path) -> Path:
    target = tmp_path / "01_kittler_2022"
    (target / "01_single_process" / "output").mkdir(parents=True)
    (target / "02_all_processes" / "output").mkdir(parents=True)
    (target / "03_validate").mkdir(parents=True)
    shutil.copytree(
        EXAMPLE_ROOT / "00_data_preprocessing",
        target / "00_data_preprocessing",
    )
    shutil.copytree(
        EXAMPLE_ROOT / "01_bp_format_data_single",
        target / "01_bp_format_data_single",
    )
    shutil.copytree(
        EXAMPLE_ROOT / "02_bp_format_data_all",
        target / "02_bp_format_data_all",
    )
    shutil.copy2(
        EXAMPLE_ROOT / "01_single_process" / "load_single_process.py",
        target / "01_single_process" / "load_single_process.py",
    )
    shutil.copy2(
        SINGLE_OUTPUT / "data.json",
        target / "01_single_process" / "output" / "data.json",
    )
    shutil.copy2(
        EXAMPLE_ROOT / "02_all_processes" / "load_all_processes.py",
        target / "02_all_processes" / "load_all_processes.py",
    )
    shutil.copy2(
        ALL_PROCESS_OUTPUT / "data.json",
        target / "02_all_processes" / "output" / "data.json",
    )
    shutil.copy2(
        EXAMPLE_ROOT / "03_validate" / "config.json",
        target / "03_validate" / "config.json",
    )
    return target


def _summary(root: Path) -> dict:
    path = root / "03_validate" / "output" / "validation_summary.json"
    return json.loads(path.read_text())


def test_ex01_target_layout_json_contract():
    single = load_process_collection_json(SINGLE_OUTPUT / "data.json")
    all_processes = load_process_collection_json(ALL_PROCESS_OUTPUT / "data.json")

    assert set(single.processes) == {SINGLE_PROCESS_ID}
    assert len(all_processes.processes) == EXPECTED_PROCESS_COUNT
    assert SINGLE_PROCESS_ID in all_processes.processes
    assert single.metadata["case_id"] == "protein_L"
    assert all_processes.metadata["case_id"] == "protein_L"
    assert not (ALL_PROCESS_OUTPUT / "dense_truth.csv").exists()


def test_ex01_visible_loaders_write_reloadable_json(tmp_path):
    assert not (EXAMPLE_ROOT / "_target_generation.py").exists()
    assert (EXAMPLE_ROOT / "00_data_preprocessing" / "target_conversion.py").exists()

    single_loader = load_module(
        "ex01_single_loader",
        EXAMPLE_ROOT / "01_single_process" / "load_single_process.py",
    )
    all_loader = load_module(
        "ex01_all_loader",
        EXAMPLE_ROOT / "02_all_processes" / "load_all_processes.py",
    )

    single = single_loader.generate_single_process_output(tmp_path / "single")
    all_processes = all_loader.generate_all_processes_output(tmp_path / "all")

    reloaded_single = load_process_collection_json(tmp_path / "single" / "data.json")
    reloaded_all = load_process_collection_json(tmp_path / "all" / "data.json")
    assert set(single.processes) == {SINGLE_PROCESS_ID}
    assert set(reloaded_single.processes) == {SINGLE_PROCESS_ID}
    assert set(all_processes.processes) == set(reloaded_all.processes)
    assert len(reloaded_all.processes) == EXPECTED_PROCESS_COUNT
    assert SINGLE_PROCESS_ID in reloaded_all.processes


def test_ex01_validator_sparse_real_summary_passes(tmp_path, capsys):
    root = _copy_ex01(tmp_path)

    exit_code = validate_example.main([str(root)])

    assert exit_code == 0
    summary = _summary(root)
    assert summary["ok"] is True
    assert summary["config"]["values"]["kind"] == "real"
    assert summary["files"]["simulation_dense_output"] is None
    assert "simulation_dense_output" not in summary
    assert "dense_event_validation" not in summary
    assert "dense_trajectory_validation" not in summary

    sparse = summary["sparse_real_diagnostics"]
    assert sparse["ok"] is True
    assert sparse["status"] == "ok_with_warnings"
    assert sparse["scope"] == validate_example.SPARSE_REAL_DIAGNOSTICS_SCOPE
    assert sparse["error_count"] == 0
    assert sparse["process_count"] == EXPECTED_PROCESS_COUNT
    assert sparse["plot_count"] == EXPECTED_PROCESS_COUNT
    assert len(sparse["processes"]) == EXPECTED_PROCESS_COUNT
    plot_paths = [
        plot_path
        for process_summary in sparse["processes"].values()
        for plot_path in process_summary["plot_paths"]
    ]
    assert len(plot_paths) == EXPECTED_PROCESS_COUNT
    assert all(Path(path).is_file() for path in plot_paths)

    text = (root / "03_validate" / "output" / "validation.txt").read_text()
    assert "Sparse/real diagnostics:" in text
    assert "reintegration" not in text.lower()
    assert "truth recovery" not in text.lower()
    capsys.readouterr()
