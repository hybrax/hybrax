"""Solver customization survives training, bundled reloads, and prediction."""

import json
import shutil
from types import SimpleNamespace

import diffrax
import pytest
from test_model_load_predict import _write_config, _write_prepared

from hybrax.format.serialization import load_process_collection
from hybrax.train import (
    harness,
    model_load,
    model_predict,
    model_reload,
    physical_solve,
)
from hybrax.train.cli import main


def test_default_solver_and_invalid_hook_result():
    assert isinstance(harness.build_solver_for_run(None, None), diffrax.Tsit5)
    with pytest.raises(TypeError, match="AbstractSolver instance"):
        harness.build_solver_for_run(
            SimpleNamespace(build_solver=lambda config: diffrax.Kvaerno3), None
        )


def test_prepare_preserves_explicit_solver_without_hook(tmp_path):
    collection = load_process_collection(_write_prepared(tmp_path / "prepared.json"))
    solver = diffrax.Kvaerno3(root_find_max_steps=7)
    prepared = harness.prepare_training(
        collection, config=harness.TrainHarnessConfig(ode_solver=solver)
    )
    assert prepared.config.ode_solver is solver
    override = diffrax.Dopri8()
    assert (
        harness.build_solver_for_run(
            SimpleNamespace(build_solver=lambda config: override), None, default=solver
        )
        is override
    )


def test_forward_preserves_explicit_solver_without_hook(tmp_path, monkeypatch):
    prepared = _write_prepared(tmp_path / "prepared.json")
    run = tmp_path / "run"
    config = _write_config(tmp_path / "train.json", prepared=prepared, run_dir=run)
    assert main(["train", "--config", str(config)]) == 0
    collection = load_process_collection(prepared)
    solver = diffrax.Kvaerno3(root_find_max_steps=7)
    captured = []

    def evaluate(*args, **kwargs):
        captured.append(kwargs["config"].ode_solver)

    monkeypatch.setattr(harness, "evaluate_trained_wrapper", evaluate)
    harness.forward_from_collection(
        collection, model_path=run, config=harness.ForwardConfig(ode_solver=solver)
    )
    assert captured == [solver]


def test_custom_solver_reaches_train_and_relocated_prediction(tmp_path, monkeypatch):
    prepared = _write_prepared(tmp_path / "prepared.json")
    run = tmp_path / "run"
    custom = tmp_path / "custom.py"
    custom.write_text(
        "import diffrax\n"
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Config:\n"
        "    root_steps: int = 7\n"
        "def get_custom_config(raw, config):\n"
        "    return Config(**raw)\n"
        "def build_solver(config):\n"
        "    return diffrax.Kvaerno3(root_find_max_steps=config.custom.root_steps)\n"
    )
    config_path = _write_config(tmp_path / "train.json", prepared=prepared, run_dir=run)
    config = json.loads(config_path.read_text())
    config["custom_py"] = str(custom)
    config["custom"] = {"root_steps": 7}
    config_path.write_text(json.dumps(config))
    seen = []
    solve = physical_solve.diffeqsolve_with_callbacks

    def record_solver(term, solver, **kwargs):
        seen.append(solver)
        return solve(term, solver, **kwargs)

    monkeypatch.setattr(physical_solve, "diffeqsolve_with_callbacks", record_solver)
    assert main(["train", "--config", str(config_path)]) == 0
    assert seen
    assert all(isinstance(solver, diffrax.Kvaerno3) for solver in seen)
    assert all(solver.root_find_max_steps == 7 for solver in seen)

    moved = tmp_path / "moved" / "run"
    moved.parent.mkdir()
    shutil.move(run, moved)
    custom.unlink()
    collection = load_process_collection(prepared)
    model, loaded_config = model_load(moved)
    assert loaded_config.custom_py == moved / "custom.py"
    assert loaded_config.custom.root_steps == 7
    model, loaded_config = model_reload(moved, model)
    assert loaded_config.custom.root_steps == 7
    captured = []
    compute = harness.compute_dense_exports

    def record_prediction_solver(*args, **kwargs):
        captured.append(kwargs["ode_solver"])
        return compute(*args, **kwargs)

    monkeypatch.setattr(harness, "compute_dense_exports", record_prediction_solver)
    predictions = model_predict(model, loaded_config, collection, grid_n=5)
    assert predictions
    assert isinstance(captured[-1], diffrax.Kvaerno3)
    assert captured[-1].root_find_max_steps == 7

    evaluate = harness.evaluate_trained_wrapper

    def record_forward_solver(*args, **kwargs):
        captured.append(kwargs["config"].ode_solver)
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(harness, "evaluate_trained_wrapper", record_forward_solver)
    forward_config = tmp_path / "forward.json"
    forward_config.write_text(
        json.dumps(
            {
                "models": [str(moved)],
                "data": {"prepared": str(prepared)},
                "output": {"plots": False},
            }
        )
    )
    assert (
        main(
            [
                "forward",
                "--config",
                str(forward_config),
                "--output-dir",
                str(tmp_path / "forward"),
            ]
        )
        == 0
    )
    assert len(captured) >= 2
    assert all(isinstance(solver, diffrax.Kvaerno3) for solver in captured)
