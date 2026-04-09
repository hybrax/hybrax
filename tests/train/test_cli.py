from __future__ import annotations

import json
from pathlib import Path

from bp_train import cli
from bp_train.harness import TrainHarnessResult


def test_prepare_cli_dispatches_to_prepare_artifact(monkeypatch):
    captured: dict[str, object] = {}

    def fake_prepare_artifact(
        *, input_json, output_json, custom_py, config, case_study=None
    ):
        captured["input_json"] = input_json
        captured["output_json"] = output_json
        captured["custom_py"] = custom_py
        captured["config"] = config

    monkeypatch.setattr(cli, "prepare_artifact", fake_prepare_artifact)

    exit_code = cli.main(
        [
            "prepare",
            "--input",
            "input.json",
            "--output",
            "output.json",
            "--custom",
            "custom.py",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "input_json": "input.json",
        "output_json": "output.json",
        "custom_py": "custom.py",
        "config": None,
    }


def test_prepare_cli_loads_config_json(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}
    config_path = tmp_path / "prepare-config.json"
    config_path.write_text(
        json.dumps({"required_control_names": ["carbon_feed", "temperature"]}),
        encoding="utf-8",
    )

    def fake_prepare_artifact(
        *, input_json, output_json, custom_py, config, case_study=None
    ):
        captured["input_json"] = input_json
        captured["output_json"] = output_json
        captured["custom_py"] = custom_py
        captured["config"] = config

    monkeypatch.setattr(cli, "prepare_artifact", fake_prepare_artifact)

    exit_code = cli.main(
        [
            "prepare",
            "--input",
            "input.json",
            "--output",
            "output.json",
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    assert captured == {
        "input_json": "input.json",
        "output_json": "output.json",
        "custom_py": None,
        "config": {"required_control_names": ["carbon_feed", "temperature"]},
    }


def test_train_cli_dispatches_to_train_harness(monkeypatch):
    captured: dict[str, object] = {}

    sentinel_collection = object()

    def fake_load(json_path):
        captured["loaded_path"] = json_path
        return sentinel_collection

    def fake_train_from_collection(collection, *, config, custom_py, runtime_config):
        captured["collection"] = collection
        captured["config"] = config
        captured["custom_py"] = custom_py
        captured["runtime_config"] = runtime_config
        return TrainHarnessResult(
            trained_wrapper=None,
            mean_loss_by_step=(1.0, 0.5),
            sampled_loss_by_process_at_log_steps={1: (("p1", 1.0),)},
            batch_process_names_by_step=(("p1",), ("p1",)),
            per_process_loss_by_step=((1.0,), (0.5,)),
            compile_warmup_seconds=0.1,
            step_time_seconds=(0.01, 0.01),
            train_step_input_signature=(("array", (1,), "int32"),),
            train_step_rebuild_count=1,
        )

    monkeypatch.setattr(cli, "train_from_collection", fake_train_from_collection)
    import bpbench.serialization as bpbench_serialization

    monkeypatch.setattr(
        bpbench_serialization, "load_process_collection_json", fake_load
    )

    # Stub out model saving (trained_wrapper is None in this test)
    from bp_train import postprocessing

    monkeypatch.setattr(postprocessing, "save_model", lambda wrapper, path: None)

    exit_code = cli.main(
        [
            "train",
            "--input",
            "prepared.json",
            "--custom",
            "custom.py",
            "--process",
            "p1,p2",
            "--process",
            "p3",
            "--target",
            "X",
            "--target",
            "P,G",
            "--target-source",
            "reactor_components",
            "--steps",
            "7",
            "--batch-size",
            "4",
            "--batch-seed",
            "99",
            "--optimizer",
            "sgd",
            "--no-shuffle-batches",
            "--learning-rate",
            "0.02",
            "--seed",
            "12",
            "--log-every",
            "3",
            "--solver-max-steps",
            "250000",
            "--solver-rtol",
            "1e-4",
            "--solver-atol",
            "1e-6",
            "--no-jump-ts",
            "--no-plot",
        ]
    )

    assert exit_code == 0
    cfg = captured["config"]
    assert cfg.process_names == ("p1", "p2", "p3")
    assert cfg.target_variable_order == ("X", "P", "G")
    assert cfg.target_source == "reactor_components"
    assert cfg.steps == 7
    assert cfg.batch_size == 4
    assert cfg.batch_seed == 99
    assert cfg.shuffle_batches is False
    assert cfg.optimizer_name == "sgd"
    assert cfg.learning_rate == 0.02
    assert cfg.seed == 12
    assert cfg.log_every == 3
    assert cfg.solver_max_steps == 250000
    assert cfg.solver_rtol == 1e-4
    assert cfg.solver_atol == 1e-6
    assert cfg.solver_use_jump_ts is False
    assert captured["collection"] is sentinel_collection
    assert str(captured["loaded_path"]) == "prepared.json"
    assert captured["custom_py"] == "custom.py"
