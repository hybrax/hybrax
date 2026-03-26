from __future__ import annotations

from pathlib import Path

from bpbench.serialization import load_process_collection_json

from bp_train.prepare import load_raw_collection, prepare_artifact


def test_load_raw_collection_reads_input():
    collection = load_raw_collection("input.json")
    assert len(collection.processes) == 12
    assert "hybrax" in (collection.metadata or {})


def test_prepare_artifact_writes_bp_train_metadata(tmp_path):
    output = tmp_path / "prepared.json"
    prepare_artifact("input.json", output)

    prepared = load_process_collection_json(output)
    metadata = prepared.metadata["bp_train"]

    assert metadata["shape_metadata"]["n_processes"] == len(prepared.processes)
    assert metadata["shape_metadata"]["max_grid_length"] >= 2
    assert metadata["shape_metadata"]["max_controls"] >= 1

    first_name = metadata["process_order"][0]
    process_md = metadata["processes"][first_name]
    assert process_md["sample_acc_name"] == "V_sample_acc"
    assert process_md["control_names"][process_md["sample_acc_index"]] == "V_sample_acc"
    assert len(process_md["dense_grid"]) == metadata["shape_metadata"]["max_grid_length"]
    assert len(process_md["control_values"]) == metadata["shape_metadata"]["max_grid_length"]
    assert len(process_md["control_values"][0]) == metadata["shape_metadata"]["max_controls"]
    assert any(v > 0 for row in process_md["control_values"] for v in row)
    assert len(process_md["step_ts"]) >= 2


def test_prepare_artifact_respects_custom_control_order(tmp_path):
    custom_py = tmp_path / "custom.py"
    custom_py.write_text(
        "\n".join(
            [
                "CONFIG = {'control_order': ['CF', 'T']}",
                "",
                "def transform_controls(process, config):",
                "    process.process_variables['CF'].is_controlled = True",
                "    process.process_variables['T'].is_controlled = True",
                "    return process",
            ]
        ),
        encoding="utf-8",
    )

    output = tmp_path / "prepared-custom.json"
    prepare_artifact("input.json", output, custom_py=custom_py)

    prepared = load_process_collection_json(output)
    metadata = prepared.metadata["bp_train"]
    first_name = metadata["process_order"][0]
    control_names = metadata["processes"][first_name]["control_names"]

    assert control_names[:2] == ["CF", "T"]
    assert control_names[-1] == "V_sample_acc"
