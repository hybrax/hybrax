from __future__ import annotations

import json
from pathlib import Path

import pytest

from bp_train import cli
from bp_train.harness import ForwardResult, TrainHarnessResult


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
        name_modeled_Inflows=(),
        name_modeled_Outflows=(),
        training_process_names=("p1",),
        per_process_total_loss={"p1": 0.1},
        per_process_per_target_loss={"p1": (0.1,)},
    )


def _stub_train_result() -> TrainHarnessResult:
    return TrainHarnessResult(
        trained_wrapper=None,
        mean_loss_by_step=(1.0, 0.5),
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

    def fake_prepare_artifact(loaded_config, *, output_dir, overwrite=False):
        captured["loaded_config"] = loaded_config
        captured["output_dir"] = output_dir
        captured["overwrite"] = overwrite

    monkeypatch.setattr(cli, "prepare_artifact", fake_prepare_artifact)

    exit_code = cli.main(
        ["prepare", "--config", str(config_path), "--output-dir", "prepared"]
    )

    assert exit_code == 0
    loaded = captured["loaded_config"]
    assert loaded.config.prepare.raw_input == tmp_path / "raw.json"
    assert captured["output_dir"] == "prepared"


@pytest.mark.parametrize("flag", ["--input", "--custom", "--case-study"])
def test_prepare_cli_rejects_legacy_experiment_flags(flag: str) -> None:
    args = ["prepare", "--config", "config.json", "--output-dir", "prepared"]
    args.extend([flag, "value"])

    with pytest.raises(SystemExit):
        cli.main(args)


def test_train_harness_config_from_run_config_maps_sections():
    """RunConfig sections + run-dir paths map onto the harness config.

    (Config drives everything now — the old per-flag CLI mapping is gone; the
    full FAIR run-dir behavior is covered in tests/test_cli_run_dir.py.)
    """
    from bp_train.harness import train_harness_config_from_run_config
    from bp_train.run_config import RunConfig

    cfg = RunConfig.model_validate(
        {
            "data": {
                "prepared": "prepared.json",
                "processes": ["p1", "p2"],
                "targets": ["X", "P"],
                "target_source": "reactor_components",
            },
            "train": {
                "epochs": 7,
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
            "checkpoint": {"every": 0.5},
            "output": {"predictions": "none"},
            "logging": {"decimals": 3},
        }
    )
    run_dir = Path("/tmp/run")
    h = train_harness_config_from_run_config(cfg, run_dir=run_dir)

    assert h.process_names == ("p1", "p2")
    assert h.target_variable_order == ("X", "P")
    assert h.target_source == "reactor_components"
    assert h.epochs == 7
    assert h.batch_size == 4
    assert h.batch_seed == 99
    assert h.shuffle_batches is False
    assert h.optimizer_name == "sgd"
    assert h.learning_rate == 0.02
    assert h.grad_clip_norm == 3.0
    assert h.seed == 12
    assert h.log_decimals == 3
    assert h.solver_max_steps == 250000
    assert h.solver_rtol == 1e-4
    assert h.solver_atol == 1e-6
    assert h.solver_use_jump_ts is False
    assert h.checkpoint_every == 0.5
    assert h.checkpoint_dir == run_dir / "checkpoints"
    assert str(h.metrics_csv).endswith("metrics.csv")
    assert h.prepared_path is not None


# The old custom.py-sha256 sidecar is replaced by config.json
# inputs.custom_py.file_hash; see tests/test_cli_run_dir.py.


@pytest.mark.parametrize(
    "flag",
    [
        "--input",
        "--custom",
        "--process",
        "--target",
        "--target-source",
        "--seed",
        "--batch-seed",
        "--learning-rate",
        "--optimizer",
        "--batch-size",
        "--grad-clip-norm",
        "--solver-max-steps",
        "--solver-rtol",
        "--solver-atol",
        # Cadence / logging knobs are config-only now (the `logging` section).
        "--log-every",
        "--metrics-csv",
        "--metrics-jsonl",
        "--log-decimals",
        "--log-header-every",
    ],
)
def test_train_cli_rejects_removed_experiment_flags(flag: str) -> None:
    with pytest.raises(SystemExit):
        cli.main(["train", "--config", "config.json", flag, "value"])


@pytest.mark.parametrize(
    "flag",
    [
        "--shuffle-batches",
        "--no-shuffle-batches",
        "--no-jump-ts",
        "--log-process-losses",
    ],
)
def test_train_cli_rejects_removed_boolean_experiment_flags(flag: str) -> None:
    with pytest.raises(SystemExit):
        cli.main(["train", "--config", "config.json", flag])


def test_format_loss_table_includes_split_summaries() -> None:
    result = ForwardResult(
        trained_wrapper=None,
        store=_DummyStore(),
        process_names=("p1", "p2"),
        target_names=("X",),
        name_modeled_Inflows=(),
        name_modeled_Outflows=(),
        training_process_names=("p1",),
        per_process_total_loss={"p1": 1.0, "p2": 3.0},
        per_process_per_target_loss={"p1": (1.0,), "p2": (3.0,)},
    )

    table, rows = cli._format_loss_table(result)

    assert "train (mean)" in table
    assert "holdout (mean)" in table
    assert rows[-2][0] == "train (mean)"
    assert rows[-1][0] == "holdout (mean)"
