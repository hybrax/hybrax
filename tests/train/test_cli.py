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


def _stub_train_result() -> TrainHarnessResult:
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


@pytest.fixture(autouse=True)
def _run_each_test_in_tmp_cwd(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)


def test_prepare_cli_dispatches_loaded_config(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"prepare": {"raw_input": "raw.json"}}))

    def fake_prepare_artifact(loaded_config, *, output_json):
        captured["loaded_config"] = loaded_config
        captured["output_json"] = output_json

    monkeypatch.setattr(cli, "prepare_artifact", fake_prepare_artifact)

    exit_code = cli.main(
        ["prepare", "--config", str(config_path), "--output", "prepared.json"]
    )

    assert exit_code == 0
    loaded = captured["loaded_config"]
    assert loaded.config.prepare.raw_input == tmp_path / "raw.json"
    assert captured["output_json"] == "prepared.json"


@pytest.mark.parametrize("flag", ["--input", "--custom", "--case-study"])
def test_prepare_cli_rejects_legacy_experiment_flags(flag: str) -> None:
    args = ["prepare", "--config", "config.json", "--output", "prepared.json"]
    args.extend([flag, "value"])

    with pytest.raises(SystemExit):
        cli.main(args)


def test_train_cli_maps_run_config_to_harness(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}
    prepared = tmp_path / "prepared.json"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "prepared": "prepared.json",
                    "processes": ["p1", "p2"],
                    "targets": ["X", "P"],
                    "target_source": "reactor_components",
                },
                "train": {
                    "steps": 7,
                    "seed": 12,
                    "optimizer": "sgd",
                    "learning_rate": 0.02,
                    "grad_clip_norm": 3.0,
                    "batch_size": 4,
                    "shuffle": False,
                    "batch_seed": 99,
                },
                "solver": {
                    "max_steps": 250000,
                    "rtol": 1e-4,
                    "atol": 1e-6,
                    "jump_ts": False,
                },
            }
        )
    )

    def fake_load(json_path):
        captured["loaded_path"] = json_path
        return _DummyCollection()

    def fake_train_from_collection(collection, *, config, custom_module, run_config):
        captured["collection"] = collection
        captured["config"] = config
        captured["custom_module"] = custom_module
        captured["run_config"] = run_config
        return _stub_train_result()

    monkeypatch.setattr(cli, "load_process_collection_json", fake_load)
    monkeypatch.setattr(cli, "train_from_collection", fake_train_from_collection)
    monkeypatch.setattr(cli, "save_model", lambda wrapper, path: None)
    monkeypatch.setattr(cli, "save_model_metadata", lambda path, meta: None)
    monkeypatch.setattr(
        cli,
        "_write_train_results",
        lambda **kwargs: captured.setdefault("write_kwargs", kwargs),
    )

    exit_code = cli.main(
        [
            "train",
            "--config",
            str(config_path),
            "--log-every",
            "3",
            "--metrics-csv",
            "metrics.csv",
            "--no-plot",
        ]
    )

    assert exit_code == 0
    cfg = captured["config"]
    assert cfg.process_names == ("p1", "p2")
    assert cfg.target_variable_order == ("X", "P")
    assert cfg.target_source == "reactor_components"
    assert cfg.steps == 7
    assert cfg.batch_size == 4
    assert cfg.batch_seed == 99
    assert cfg.shuffle_batches is False
    assert cfg.optimizer_name == "sgd"
    assert cfg.learning_rate == 0.02
    assert cfg.grad_clip_norm == 3.0
    assert cfg.seed == 12
    assert cfg.log_every == 3
    assert cfg.solver_max_steps == 250000
    assert cfg.solver_rtol == 1e-4
    assert cfg.solver_atol == 1e-6
    assert cfg.solver_use_jump_ts is False
    assert cfg.metrics_csv == "metrics.csv"
    assert captured["loaded_path"] == prepared
    assert captured["custom_module"] is None
    assert captured["run_config"].data.prepared == prepared
    assert captured["write_kwargs"]["run_config"] is captured["run_config"]


def test_train_cli_writes_custom_py_sha256_to_sidecar(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}
    custom_py = tmp_path / "custom.py"
    custom_py.write_text("VALUE = 1\n")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"prepared": "prepared.json"},
                "custom_py": "custom.py",
            }
        )
    )

    monkeypatch.setattr(cli, "load_process_collection_json", lambda _p: _DummyCollection())
    monkeypatch.setattr(
        cli,
        "train_from_collection",
        lambda collection, *, config, custom_module, run_config: _stub_train_result(),
    )
    monkeypatch.setattr(cli, "save_model", lambda wrapper, path: None)
    monkeypatch.setattr(
        cli,
        "save_model_metadata",
        lambda path, meta: captured.update({"meta_path": path, "meta": meta}),
    )
    monkeypatch.setattr(cli, "_write_train_results", lambda **kwargs: _stub_forward_result())

    cli.main(["train", "--config", str(config_path), "--no-plot"])

    assert captured["meta"]["custom_py_sha256"] is not None
    assert captured["meta"]["custom_py"] is not None


@pytest.mark.parametrize(
    "flag",
    [
        "--input",
        "--custom",
        "--process",
        "--target",
        "--target-source",
        "--steps",
        "--seed",
        "--batch-seed",
        "--learning-rate",
        "--optimizer",
        "--batch-size",
        "--grad-clip-norm",
        "--solver-max-steps",
        "--solver-rtol",
        "--solver-atol",
    ],
)
def test_train_cli_rejects_removed_experiment_flags(flag: str) -> None:
    with pytest.raises(SystemExit):
        cli.main(["train", "--config", "config.json", flag, "value"])


@pytest.mark.parametrize("flag", ["--shuffle-batches", "--no-shuffle-batches", "--no-jump-ts"])
def test_train_cli_rejects_removed_boolean_experiment_flags(flag: str) -> None:
    with pytest.raises(SystemExit):
        cli.main(["train", "--config", "config.json", flag])


def test_format_loss_table_includes_split_summaries() -> None:
    result = ForwardResult(
        trained_wrapper=None,
        store=_DummyStore(),
        process_names=("p1", "p2"),
        target_names=("X",),
        name_modeled_FVCs=(),
        name_modeled_SVCs=(),
        training_process_names=("p1",),
        per_process_total_loss={"p1": 1.0, "p2": 3.0},
        per_process_per_target_loss={"p1": (1.0,), "p2": (3.0,)},
    )

    table, rows = cli._format_loss_table(result)

    assert "train (mean)" in table
    assert "holdout (mean)" in table
    assert rows[-2][0] == "train (mean)"
    assert rows[-1][0] == "holdout (mean)"
