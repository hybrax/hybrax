from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from bp_format.serialization import save_process_collection

import bp_train
from bp_train.cli import main

# Tiny single-process collection fixture (parametrizable biomass values).
from test_serialization import _collection


def _write_prepared(path: Path, biomass_values=(1.0, 0.8, 0.64)) -> Path:
    save_process_collection(_collection(biomass_values), path)
    return path


def _write_config(
    config_path: Path,
    *,
    prepared: Path,
    run_dir: Path,
    steps: int = 4,
    every: int = 2,
) -> Path:
    config = {
        "data": {
            "prepared": str(prepared),
            "targets": ["biomass"],
            "target_source": "reactor_components",
        },
        "train": {"steps": steps, "learning_rate": 0.05, "seed": 0},
        "solver": {"max_steps": 2048},
        "checkpoint": {"every": every, "keep": "best+latest"},
        "output": {"dir": str(run_dir), "plots": False},
        "logging": {"every": 1},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_train_cli_produces_fair_run_dir_and_load_run(tmp_path: Path):
    prepared = _write_prepared(tmp_path / "prepared.json")
    run_dir = tmp_path / "run"
    config = _write_config(tmp_path / "config.json", prepared=prepared, run_dir=run_dir)

    assert main(["train", "--config", str(config)]) == 0

    # FAIR layout.
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.csv").is_file()
    assert (run_dir / "predictions.csv").is_file()
    assert (run_dir / "model" / "params.eqx").is_file()
    assert (run_dir / "checkpoints" / "latest").is_symlink()
    # each checkpoint is self-contained (config + prepared bundled; custom.py too
    # when the run has one)
    latest = (run_dir / "checkpoints" / "latest").resolve()
    for fname in ("config.json", "prepared.json.gz", "params.eqx"):
        assert (latest / fname).is_file(), fname

    doc = json.loads((run_dir / "config.json").read_text())
    assert doc["status"] == "complete"
    assert doc["steps_completed"] == 4
    assert doc["inputs"]["prepared_input"]["content_hash"].startswith("sha256:")
    assert "jax" in doc["environment"]

    # load_run reconstructs from the run dir alone (prepared.json via recorded path).
    run = bp_train.load_run(run_dir)
    assert run.wrapper is not None
    assert run.config.train.steps == 4
    # A re-prepared (byte-different but identical) prepared.json still loads.
    _write_prepared(prepared)
    bp_train.load_run(run_dir)


def test_train_cli_rerun_guard_and_overwrite(tmp_path: Path):
    prepared = _write_prepared(tmp_path / "prepared.json")
    run_dir = tmp_path / "run"
    config = _write_config(tmp_path / "config.json", prepared=prepared, run_dir=run_dir)

    assert main(["train", "--config", str(config)]) == 0
    # A completed run blocks a plain re-run ...
    assert main(["train", "--config", str(config)]) == 1
    # ... but --overwrite proceeds.
    assert main(["train", "--config", str(config), "--overwrite"]) == 0


def test_train_cli_resume_extends(tmp_path: Path):
    prepared = _write_prepared(tmp_path / "prepared.json")
    run_dir = tmp_path / "run"
    config = _write_config(
        tmp_path / "config.json", prepared=prepared, run_dir=run_dir, steps=2, every=2
    )

    assert main(["train", "--config", str(config)]) == 0
    assert len(pd.read_csv(run_dir / "metrics.csv")) == 2

    # Resume, extending the target to 4 steps.
    assert main(["train", "--resume", str(run_dir), "--steps", "4"]) == 0
    doc = json.loads((run_dir / "config.json").read_text())
    assert doc["status"] == "complete"
    assert doc["steps_completed"] == 4
    assert len(pd.read_csv(run_dir / "metrics.csv")) == 4  # appended, not truncated


def test_train_cli_resume_accepts_checkpoint_subdir(tmp_path: Path):
    """--resume tolerates being pointed at checkpoints/latest (resolves up to
    the run dir) instead of erroring."""
    prepared = _write_prepared(tmp_path / "prepared.json")
    run_dir = tmp_path / "run"
    config = _write_config(
        tmp_path / "config.json", prepared=prepared, run_dir=run_dir, steps=2, every=2
    )
    assert main(["train", "--config", str(config)]) == 0

    # Point at the checkpoint sub-dir, with a trailing slash, like a user would.
    assert (
        main(["train", "--resume", str(run_dir / "checkpoints" / "latest") + "/", "--steps", "4"])
        == 0
    )
    assert json.loads((run_dir / "config.json").read_text())["steps_completed"] == 4


def test_resume_rejects_output_dir(tmp_path: Path):
    """Resume replays the saved config; --output-dir is rejected, not ignored."""
    prepared = _write_prepared(tmp_path / "prepared.json")
    run_dir = tmp_path / "run"
    config = _write_config(tmp_path / "config.json", prepared=prepared, run_dir=run_dir, steps=2)
    assert main(["train", "--config", str(config)]) == 0

    assert main(["train", "--resume", str(run_dir), "--output-dir", str(tmp_path / "x")]) == 1
    # The rejected run did not touch the completed run's state.
    assert json.loads((run_dir / "config.json").read_text())["steps_completed"] == 2


def test_resume_rejects_no_plot(tmp_path: Path):
    """--no-plot is a fresh-run-only override; rejected on resume."""
    prepared = _write_prepared(tmp_path / "prepared.json")
    run_dir = tmp_path / "run"
    config = _write_config(tmp_path / "config.json", prepared=prepared, run_dir=run_dir, steps=2)
    assert main(["train", "--config", str(config)]) == 0

    assert main(["train", "--resume", str(run_dir), "--no-plot"]) == 1


def test_resume_allows_steps(tmp_path: Path):
    """--steps remains the one legal override; the new guard must not block it."""
    prepared = _write_prepared(tmp_path / "prepared.json")
    run_dir = tmp_path / "run"
    config = _write_config(tmp_path / "config.json", prepared=prepared, run_dir=run_dir, steps=2)
    assert main(["train", "--config", str(config)]) == 0

    assert main(["train", "--resume", str(run_dir), "--steps", "4"]) == 0
    assert json.loads((run_dir / "config.json").read_text())["steps_completed"] == 4


def test_load_params_refreshes_without_reading_prepared(tmp_path: Path, monkeypatch):
    prepared = _write_prepared(tmp_path / "prepared.json")
    run_dir = tmp_path / "run"
    config = _write_config(tmp_path / "config.json", prepared=prepared, run_dir=run_dir)
    assert main(["train", "--config", str(config)]) == 0

    run = bp_train.load_run(run_dir, checkpoint="latest")

    # load_params is the lightweight path: it must NOT re-read the dataset.
    import bp_train.serialization as S

    def _boom(*_a, **_k):
        raise AssertionError("load_params must not load the collection")

    monkeypatch.setattr(S, "load_process_collection", _boom)
    refreshed = bp_train.load_params(run_dir, into=run.wrapper, checkpoint="latest")
    assert refreshed is not None
    run.reload("latest")  # LoadedRun.reload uses the same lightweight path


def test_load_run_integrity_guard_on_content_change(tmp_path: Path):
    prepared = _write_prepared(tmp_path / "prepared.json")
    run_dir = tmp_path / "run"
    config = _write_config(tmp_path / "config.json", prepared=prepared, run_dir=run_dir)
    assert main(["train", "--config", str(config)]) == 0

    # Tamper the prepared.json *content* → content_hash mismatch → hard error.
    _write_prepared(prepared, biomass_values=(2.0, 1.5, 1.0))
    with pytest.raises(ValueError, match="differs"):
        bp_train.load_run(run_dir)
