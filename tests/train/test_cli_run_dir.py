from __future__ import annotations

import json
from pathlib import Path

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
    epochs: int = 4,
    every: float = 2.0,
) -> Path:
    config = {
        "data": {
            "prepared": str(prepared),
            "targets": ["biomass"],
            "target_source": "reactor_components",
        },
        "train": {"epochs": epochs, "learning_rate": 0.05, "seed": 0},
        "solver": {"max_steps": 2048},
        "checkpoint": {"every": every},
        "output": {"dir": str(run_dir), "plots": False},
        "logging": {"decimals": 4},
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
    assert doc["updates_completed"] == 4
    assert doc["inputs"]["prepared_input"]["content_hash"].startswith("sha256:")
    assert "jax" in doc["environment"]

    # load_run reconstructs from the run dir alone (prepared.json via recorded path).
    run = bp_train.load_run(run_dir)
    assert run.wrapper is not None
    assert run.config.train.epochs == 4
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

    checkpoints = run_dir / "checkpoints"
    assert (checkpoints / "step_00004").is_dir()
    (checkpoints / "best").symlink_to("step_00004")
    (run_dir / "obsolete.txt").write_text("old run", encoding="utf-8")

    # ... but --overwrite starts a clean, shorter run.
    _write_config(
        config,
        prepared=prepared,
        run_dir=run_dir,
        epochs=2,
        every=2.0,
    )
    assert main(["train", "--config", str(config), "--overwrite"]) == 0
    assert {path.name for path in checkpoints.iterdir()} == {
        "latest",
        "step_00002",
    }
    assert not (run_dir / "obsolete.txt").exists()


def test_train_overwrite_rejects_inputs_inside_output_dir(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    prepared = _write_prepared(tmp_path / "prepared.json")
    config = _write_config(
        run_dir / "source-config.json",
        prepared=prepared,
        run_dir=run_dir,
    )

    with pytest.raises(ValueError, match="contains input file"):
        main(["train", "--config", str(config), "--overwrite"])
    assert config.is_file()


def test_train_cli_has_epochs_override_but_no_individual_resume():
    with pytest.raises(SystemExit):
        main(["train", "--resume", "run"])
    with pytest.raises(SystemExit):
        main(["train", "--config", "config.json", "--steps", "2"])


def test_train_cli_epochs_override_takes_effect(tmp_path: Path):
    prepared = _write_prepared(tmp_path / "prepared.json")
    run_dir = tmp_path / "run"
    # Config says epochs=4, but the CLI override below asks for 2. batch_size is
    # unset, so batches_per_epoch == 1 and updates_completed == epochs directly;
    # if the override were silently ignored, this run would produce 4 updates
    # and the assertions below would fail.
    config = _write_config(
        tmp_path / "config.json", prepared=prepared, run_dir=run_dir, epochs=4
    )

    assert main(["train", "--config", str(config), "--epochs", "2"]) == 0

    doc = json.loads((run_dir / "config.json").read_text())
    assert doc["updates_completed"] == 2

    run = bp_train.load_run(run_dir)
    assert run.config.train.epochs == 2


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
