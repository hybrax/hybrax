from __future__ import annotations

import json
from pathlib import Path

import pytest

from bp_train import cli
from bp_train.harness import ForwardResult, TrainHarnessConfig, TrainHarnessResult


class _DummyCollection:
    processes = {"p1": object(), "p2": object(), "p3": object()}


class _DummyStore:
    process_order = ("p1", "p2", "p3")


def _stub_forward_result() -> ForwardResult:
    return ForwardResult(
        trained_wrapper=None,
        store=_DummyStore(),
        process_names=("p1",),
        target_names=("X",),
        name_modeled_FVCs=(),
        name_modeled_SVCs=(),
        training_process_names=("p1",),
        per_process_total_loss={"p1": 0.1},
        per_process_per_target_loss={"p1": (0.1,)},
    )


@pytest.fixture(autouse=True)
def _run_each_test_in_tmp_cwd(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)


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

    sentinel_collection = _DummyCollection()

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
    monkeypatch.setattr(cli, "load_process_collection_json", fake_load)
    monkeypatch.setattr(cli, "load_custom_module", lambda _path: object())
    monkeypatch.setattr(cli, "resolve_config", lambda _module, _config: {})
    monkeypatch.setattr(
        cli, "forward_from_collection", lambda *a, **k: _stub_forward_result()
    )
    monkeypatch.setattr(cli, "plot_process_simulations", lambda *a, **k: None)

    # Stub out model saving (trained_wrapper is None in this test)
    monkeypatch.setattr(cli, "save_model", lambda wrapper, path: None)

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


def test_train_cli_plots_only_selected_processes(monkeypatch):
    captured: dict[str, object] = {}

    sentinel_collection = _DummyCollection()

    def fake_load(json_path):
        captured["loaded_path"] = json_path
        return sentinel_collection

    def fake_train_from_collection(collection, *, config, custom_py, runtime_config):
        captured["config"] = config
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
    monkeypatch.setattr(cli, "load_process_collection_json", fake_load)

    monkeypatch.setattr(cli, "save_model", lambda wrapper, path: None)
    monkeypatch.setattr(
        cli, "forward_from_collection", lambda *a, **k: _stub_forward_result()
    )

    def fake_plot_training_results(
        result,
        collection,
        store,
        output_dir,
        dense_exports,
        process_names=None,
        *,
        per_process_named_losses=None,
        per_process_total_loss=None,
        timeseries_csv_path=None,
    ):
        del result, collection, store, output_dir, dense_exports
        del per_process_named_losses, per_process_total_loss
        captured["plot_process_names"] = process_names
        captured["predictions_csv"] = timeseries_csv_path

    monkeypatch.setattr(cli, "plot_training_results", fake_plot_training_results)

    exit_code = cli.main(
        [
            "train",
            "--input",
            "prepared.json",
            "--process",
            "p1",
            "--process",
            "p3",
            "--steps",
            "2",
            "--no-jump-ts",
            "--output-dir",
            "out",
            "--plot",
        ]
    )

    assert exit_code == 0
    assert captured["plot_process_names"] == ("p1", "p3")
    assert str(captured["predictions_csv"]).endswith("out/predictions.csv")


def test_train_cli_writes_losses_and_predictions_even_with_no_plot(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "load_process_collection_json",
        lambda _p: _DummyCollection(),
    )
    monkeypatch.setattr(cli, "load_custom_module", lambda _path: object())
    monkeypatch.setattr(cli, "resolve_config", lambda _module, _config: {})

    def fake_train_from_collection(collection, *, config, custom_py, runtime_config):
        del collection, config, custom_py, runtime_config
        return TrainHarnessResult(
            trained_wrapper=None,
            mean_loss_by_step=(1.0, 0.5),
            sampled_loss_by_process_at_log_steps={},
            batch_process_names_by_step=(("p1",),),
            per_process_loss_by_step=((0.5,),),
            compile_warmup_seconds=0.1,
            step_time_seconds=(0.01,),
            train_step_input_signature=(),
            train_step_rebuild_count=0,
        )

    monkeypatch.setattr(cli, "train_from_collection", fake_train_from_collection)
    monkeypatch.setattr(
        cli, "forward_from_collection", lambda *a, **k: _stub_forward_result()
    )

    def fake_write_loss_csv(rows, path):
        captured["loss_path"] = path
        captured["loss_rows"] = rows

    def fake_plot_process_simulations(*args, **kwargs):
        captured["render_plots"] = kwargs.get("render_plots")
        captured["timeseries_csv_path"] = kwargs.get("timeseries_csv_path")

    monkeypatch.setattr(cli, "_write_loss_csv", fake_write_loss_csv)
    monkeypatch.setattr(cli, "plot_process_simulations", fake_plot_process_simulations)

    exit_code = cli.main(
        [
            "train",
            "--input",
            "prepared.json",
            "--output-dir",
            "out",
            "--no-jump-ts",
            "--no-plot",
        ]
    )

    assert exit_code == 0
    assert str(captured["loss_path"]).endswith("out/losses.csv")
    assert captured["loss_rows"][0][0] == "process"
    assert captured["render_plots"] is False
    assert str(captured["timeseries_csv_path"]).endswith("out/predictions.csv")


def test_train_cli_defaults_match_TrainHarnessConfig(monkeypatch):
    """CLI defaults with no flags set must match `TrainHarnessConfig()` defaults."""
    captured: dict[str, object] = {}

    def fake_load(json_path):
        return _DummyCollection()

    def fake_train_from_collection(collection, *, config, custom_py, runtime_config):
        captured["config"] = config
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
    monkeypatch.setattr(cli, "load_process_collection_json", fake_load)
    monkeypatch.setattr(
        cli, "forward_from_collection", lambda *a, **k: _stub_forward_result()
    )
    monkeypatch.setattr(cli, "plot_process_simulations", lambda *a, **k: None)
    monkeypatch.setattr(cli, "save_model", lambda wrapper, path: None)

    cli.main(["train", "--input", "prepared.json", "--no-plot"])

    cfg = captured["config"]
    d = TrainHarnessConfig()
    assert cfg.steps == d.steps
    assert cfg.batch_size == d.batch_size
    assert cfg.batch_seed == d.batch_seed
    assert cfg.optimizer_name == d.optimizer_name
    assert cfg.shuffle_batches == d.shuffle_batches
    assert cfg.learning_rate == d.learning_rate
    assert cfg.grad_clip_norm == d.grad_clip_norm
    assert cfg.seed == d.seed
    assert cfg.log_every == d.log_every
    assert cfg.solver_max_steps == d.solver_max_steps
    assert cfg.solver_rtol == d.solver_rtol
    assert cfg.solver_atol == d.solver_atol
    assert cfg.solver_use_jump_ts == d.solver_use_jump_ts
    assert cfg.log_process_losses == d.log_process_losses
    assert cfg.metrics_csv == d.metrics_csv
    assert cfg.metrics_jsonl == d.metrics_jsonl
    assert cfg.log_decimals == d.log_decimals
    assert cfg.log_header_every == d.log_header_every


def test_train_cli_uses_config_targets_when_target_flag_missing(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli, "load_process_collection_json", lambda _p: _DummyCollection()
    )
    monkeypatch.setattr(
        cli,
        "load_custom_module",
        lambda path: {"custom_path": path},
    )
    monkeypatch.setattr(
        cli,
        "resolve_config",
        lambda _module, _config: {"target_variable_order": ["X", "P", "G"]},
    )

    def fake_train_from_collection(collection, *, config, custom_py, runtime_config):
        del collection, custom_py, runtime_config
        captured["config"] = config
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
    monkeypatch.setattr(
        cli, "forward_from_collection", lambda *a, **k: _stub_forward_result()
    )
    monkeypatch.setattr(cli, "plot_process_simulations", lambda *a, **k: None)
    monkeypatch.setattr(cli, "save_model", lambda wrapper, path: None)

    exit_code = cli.main(
        [
            "train",
            "--input",
            "prepared.json",
            "--custom",
            "custom.py",
            "--no-plot",
        ]
    )

    assert exit_code == 0
    cfg = captured["config"]
    assert cfg.target_variable_order == ("X", "P", "G")


def test_train_cli_uses_none_targets_when_cli_and_config_missing(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli, "load_process_collection_json", lambda _p: _DummyCollection()
    )
    monkeypatch.setattr(
        cli,
        "load_custom_module",
        lambda _path: object(),
    )
    monkeypatch.setattr(cli, "resolve_config", lambda _module, _config: {})

    def fake_train_from_collection(collection, *, config, custom_py, runtime_config):
        del collection, custom_py, runtime_config
        captured["config"] = config
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
    monkeypatch.setattr(
        cli, "forward_from_collection", lambda *a, **k: _stub_forward_result()
    )
    monkeypatch.setattr(cli, "plot_process_simulations", lambda *a, **k: None)
    monkeypatch.setattr(cli, "save_model", lambda wrapper, path: None)

    exit_code = cli.main(
        [
            "train",
            "--input",
            "prepared.json",
            "--custom",
            "custom.py",
            "--no-plot",
        ]
    )

    assert exit_code == 0
    cfg = captured["config"]
    assert cfg.target_variable_order is None


def test_train_cli_target_flag_overrides_config_targets(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli, "load_process_collection_json", lambda _p: _DummyCollection()
    )
    monkeypatch.setattr(cli, "load_custom_module", lambda _path: object())
    monkeypatch.setattr(
        cli,
        "resolve_config",
        lambda _module, _config: {"target_variable_order": ["X", "P", "G"]},
    )

    def fake_train_from_collection(collection, *, config, custom_py, runtime_config):
        del collection, custom_py, runtime_config
        captured["config"] = config
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
    monkeypatch.setattr(
        cli, "forward_from_collection", lambda *a, **k: _stub_forward_result()
    )
    monkeypatch.setattr(cli, "plot_process_simulations", lambda *a, **k: None)
    monkeypatch.setattr(cli, "save_model", lambda wrapper, path: None)

    exit_code = cli.main(
        [
            "train",
            "--input",
            "prepared.json",
            "--custom",
            "custom.py",
            "--target",
            "A,B",
            "--target",
            "C",
            "--no-plot",
        ]
    )

    assert exit_code == 0
    cfg = captured["config"]
    assert cfg.target_variable_order == ("A", "B", "C")
