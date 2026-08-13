from __future__ import annotations

import gc
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

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
    predictions: str = "parents",
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
        "output": {"dir": str(run_dir), "predictions": predictions},
        "logging": {"decimals": 4},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_train_cli_releases_collection_before_executor(tmp_path: Path, monkeypatch):
    prepared_path = _write_prepared(tmp_path / "prepared.json")
    config = _write_config(
        tmp_path / "config.json",
        prepared=prepared_path,
        run_dir=tmp_path / "run",
        epochs=1,
        predictions="none",
    )
    collection_ref = None

    def load_collection(_path):
        nonlocal collection_ref
        collection = _collection()
        collection_ref = weakref.ref(collection)
        return collection

    def execute(*_args, **_kwargs):
        gc.collect()
        assert collection_ref is not None
        assert collection_ref() is None
        return SimpleNamespace(
            mean_loss_by_step=(1.0,), updates_completed=1, trained_wrapper=object()
        )

    monkeypatch.setattr("bp_train.cli.load_process_collection", load_collection)
    monkeypatch.setattr("bp_train.cli.train_collection", execute)

    def evaluate(*_args, **kwargs):
        assert kwargs["prediction_process_names"] == ()
        return SimpleNamespace()

    monkeypatch.setattr("bp_train.cli.evaluate_trained_wrapper", evaluate)
    monkeypatch.setattr("bp_train.cli._write_train_results", lambda **_kwargs: None)
    monkeypatch.setattr("bp_train.cli._finalize_run_dir", lambda *_args: None)

    assert main(["train", "--config", str(config)]) == 0


def test_train_cli_produces_fair_run_dir_and_model_load(tmp_path: Path):
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

    # model_load reconstructs from the run dir alone (prepared.json via recorded path).
    wrapper, config = bp_train.model_load(run_dir)
    assert wrapper is not None
    assert config.train.epochs == 4
    # A re-prepared (byte-different but identical) prepared.json still loads.
    _write_prepared(prepared)
    bp_train.model_load(run_dir)


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

    _wrapper, config = bp_train.model_load(run_dir)
    assert config.train.epochs == 2


def test_model_reload_refreshes_without_reading_prepared(tmp_path: Path, monkeypatch):
    prepared = _write_prepared(tmp_path / "prepared.json")
    run_dir = tmp_path / "run"
    config = _write_config(tmp_path / "config.json", prepared=prepared, run_dir=run_dir)
    assert main(["train", "--config", str(config)]) == 0

    wrapper, _config = bp_train.model_load(run_dir)

    # model_reload is the lightweight path: it must NOT re-read the dataset.
    import bp_train.serialization as S

    def _boom(*_a, **_k):
        raise AssertionError("model_reload must not load the collection")

    monkeypatch.setattr(S, "load_process_collection", _boom)
    refreshed, config = bp_train.model_reload(run_dir, wrapper)
    assert refreshed is not None
    # Same 2-tuple shape as model_load, so the two are interchangeable.
    assert config.train.epochs == 4
    # A named checkpoint is addressed by its path, not a selector string.
    bp_train.model_reload(run_dir / "checkpoints" / "latest", wrapper)


def test_model_load_integrity_guard_on_content_change(tmp_path: Path):
    prepared = _write_prepared(tmp_path / "prepared.json")
    run_dir = tmp_path / "run"
    config = _write_config(tmp_path / "config.json", prepared=prepared, run_dir=run_dir)
    assert main(["train", "--config", str(config)]) == 0

    # Tamper the prepared.json *content* → content_hash mismatch → hard error.
    _write_prepared(prepared, biomass_values=(2.0, 1.5, 1.0))
    with pytest.raises(ValueError, match="differs"):
        bp_train.model_load(run_dir)
