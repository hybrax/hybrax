"""The user-facing model API: ``model_load`` / ``model_predict`` / ``model_reload``.

Covers the path-resolution rule, the ``(wrapper, config)`` contract both loaders
share, and the property that makes ``model_predict`` safe on a collection other
than the training one: the trained ``SCALE_*`` are reused, never re-estimated.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bp_format.serialization import load_process_collection, save_process_collection

import bp_train
from bp_train.cli import main

# Tiny single-process collection fixture, shared with the serialization tests.
from test_serialization import _collection


def _write_prepared(path: Path, biomass_values=(1.0, 0.8, 0.64), *, n_processes=1):
    save_process_collection(_collection(biomass_values, n_processes=n_processes), path)
    return path


def _write_config(
    config_path: Path,
    *,
    prepared: Path,
    run_dir: Path,
    solver: dict | None = None,
    predictions: str = "parents",
) -> Path:
    config = {
        "data": {
            "prepared": str(prepared),
            "targets": ["biomass"],
            "target_source": "reactor_components",
        },
        "train": {"epochs": 2, "learning_rate": 0.05, "seed": 0},
        # Deliberately NOT the SolverConfig defaults (2048 / 1e-5 / 1e-7), so a
        # test that asserts on these values fails if anything silently
        # substitutes a default instead of reading the run's own config.
        "solver": solver or {"max_steps": 3072, "rtol": 1e-4, "atol": 1e-6},
        "checkpoint": {"every": 1.0},
        "output": {"dir": str(run_dir), "predictions": predictions},
        "logging": {"decimals": 4},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


@pytest.fixture
def trained_run(tmp_path: Path):
    """A completed run directory plus the prepared collection it trained on."""
    prepared = _write_prepared(tmp_path / "prepared.json", n_processes=2)
    run_dir = tmp_path / "run"
    config = _write_config(tmp_path / "config.json", prepared=prepared, run_dir=run_dir)
    assert main(["train", "--config", str(config)]) == 0
    return run_dir, prepared


def _trainable_leaves(module):
    trainable, _ = bp_train.partition_trainable(module)
    return [
        leaf
        for leaf in jax.tree_util.tree_leaves(trainable)
        if eqx.is_inexact_array(leaf)
    ]


def _scale_leaves(wrapper):
    """Every SCALE_* array on the reaction module, flattened."""
    return [
        np.asarray(leaf)
        for name, leaf in vars(wrapper.reaction_module).items()
        if name.startswith("SCALE_")
        for leaf in jax.tree_util.tree_leaves(leaf)
        if eqx.is_inexact_array(leaf)
    ]


# The fixture collection measures biomass at these times; model_predict splices a
# node onto each of them, so the returned grid is the union of the even grid and
# the measurement times rather than a bare linspace.
_MEASUREMENT_TIMES = (0.0, 1.0, 2.0)


def _assert_prediction_grid(t: np.ndarray, *, grid_n: int) -> None:
    """The output grid is ``linspace(t0, t1, grid_n)`` unioned with the process's
    own measurement times, sorted."""
    t = np.asarray(t)
    assert np.all(np.diff(t) >= 0), "prediction grid must be sorted"
    linspace = np.linspace(t[0], t[-1], grid_n)
    for point in linspace:
        assert np.isclose(t, point).any(), f"even-grid node {point} missing"
    for measured in _MEASUREMENT_TIMES:
        assert np.isclose(t, measured).any(), f"measurement node {measured} missing"
    # grid_n even nodes, plus the measurement times that do not coincide with one.
    extra = sum(0 if np.isclose(linspace, m).any() else 1 for m in _MEASUREMENT_TIMES)
    assert t.shape == (grid_n + extra,)


# ---------------------------------------------------------------------------
# model_load
# ---------------------------------------------------------------------------


def test_model_load_returns_wrapper_and_the_runs_own_solver(trained_run):
    run_dir, _prepared = trained_run

    wrapper, config = bp_train.model_load(run_dir)

    assert wrapper is not None
    # The whole point: the solver settings arrive with the model, and they are
    # the run's recorded ones rather than any dataclass default.
    assert (config.solver.max_steps, config.solver.rtol, config.solver.atol) == (
        3072,
        1e-4,
        1e-6,
    )
    assert config.solver.jump_ts is True
    recorded = json.loads((run_dir / "config.json").read_text())["config"]["solver"]
    assert config.solver.model_dump() == recorded


def test_model_load_accepts_run_dir_checkpoint_dir_and_params_file(trained_run):
    run_dir, _prepared = trained_run
    latest = run_dir / "checkpoints" / "latest"

    from_run, _ = bp_train.model_load(run_dir)
    from_ckpt, _ = bp_train.model_load(latest)
    from_file, _ = bp_train.model_load(run_dir / "model" / "params.eqx")

    # model/params.eqx is a copy of checkpoints/latest/params.eqx, so a completed
    # run yields identical weights through all three addressing forms.
    for other in (from_ckpt, from_file):
        for a, b in zip(
            _trainable_leaves(from_run), _trainable_leaves(other), strict=True
        ):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_model_load_falls_through_to_latest_checkpoint_when_model_dir_absent(
    trained_run,
):
    """An in-progress run has no model/ yet; it must still load."""
    run_dir, _prepared = trained_run
    expected, _ = bp_train.model_load(run_dir / "checkpoints" / "latest")
    shutil.rmtree(run_dir / "model")

    wrapper, config = bp_train.model_load(run_dir)

    assert config.solver.max_steps == 3072
    for a, b in zip(
        _trainable_leaves(expected), _trainable_leaves(wrapper), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_model_load_rejects_a_file_that_is_not_params_eqx(trained_run):
    """A legacy trained_wrapper.eqx must raise, not silently fall through to the
    run's final weights — that would hand back different parameters than asked."""
    run_dir, _prepared = trained_run
    legacy = run_dir / "checkpoints" / "latest" / "trained_wrapper.eqx"
    shutil.copyfile(run_dir / "model" / "params.eqx", legacy)

    with pytest.raises(FileNotFoundError, match="must be a params.eqx"):
        bp_train.model_load(legacy)


def test_model_load_errors_are_specific(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        bp_train.model_load(tmp_path / "nope")

    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(FileNotFoundError, match="no params.eqx"):
        bp_train.model_load(bare)

    # params.eqx present but no config.json at or above it.
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    (orphan / "params.eqx").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="config.json"):
        bp_train.model_load(orphan)


# ---------------------------------------------------------------------------
# model_predict
# ---------------------------------------------------------------------------


def test_model_predict_matches_the_runs_own_predictions(trained_run):
    """The forward path reproduces what training itself exported."""
    run_dir, prepared = trained_run
    wrapper, config = bp_train.model_load(run_dir)
    collection = load_process_collection(prepared)

    dense = bp_train.model_predict(wrapper, config, collection)

    assert set(dense) == set(collection.processes)
    import pandas as pd

    exported = pd.read_csv(run_dir / "predictions.csv")
    assert not exported.empty
    for name, export in dense.items():
        rows = exported[exported["process"] == name].sort_values("t")
        # Same dense grid and same trajectory as the training-time export, which
        # ran through the same code path under the same recorded solver settings.
        np.testing.assert_allclose(export.t, rows["t"].to_numpy(), rtol=0, atol=1e-9)
        np.testing.assert_allclose(
            export.c_species[:, 0], rows["c_biomass"].to_numpy(), rtol=1e-6, atol=1e-8
        )


def test_model_predict_on_an_unseen_process_preserves_trained_scales(trained_run):
    """A process the model never trained on works, and the wrapper's SCALE_* are
    untouched — this is the regression guard for silent scale re-estimation."""
    run_dir, _prepared = trained_run
    wrapper, config = bp_train.model_load(run_dir)
    scales_before = _scale_leaves(wrapper)

    # A collection the model has never seen: one differently-named process whose
    # measurements differ, so a re-estimated SCALE_* would visibly change.
    unseen = _collection(biomass_values=(5.0, 4.0, 3.2))
    unseen = replace(
        unseen,
        processes={
            "unseen_1": replace(
                unseen.processes["p1"],
                metadata=replace(unseen.processes["p1"].metadata, name="unseen_1"),
            )
        },
    )

    dense = bp_train.model_predict(wrapper, config, unseen, grid_n=64)

    assert set(dense) == {"unseen_1"}
    _assert_prediction_grid(dense["unseen_1"].t, grid_n=64)
    assert np.all(np.isfinite(dense["unseen_1"].c_species))

    scales_after = _scale_leaves(wrapper)
    assert len(scales_before) == len(scales_after) > 0
    for before, after in zip(scales_before, scales_after, strict=True):
        np.testing.assert_array_equal(before, after)


def test_model_predict_selects_processes_and_grid(trained_run):
    run_dir, prepared = trained_run
    wrapper, config = bp_train.model_load(run_dir)
    collection = load_process_collection(prepared)

    dense = bp_train.model_predict(
        wrapper, config, collection, process_names=("p2",), grid_n=32
    )

    assert set(dense) == {"p2"}
    _assert_prediction_grid(dense["p2"].t, grid_n=32)

    with pytest.raises(ValueError, match="unknown process names"):
        bp_train.model_predict(wrapper, config, collection, process_names=("nope",))


def test_model_predict_fails_fast_on_incompatible_layout(trained_run):
    """A collection whose RhsOde axes differ must raise, not integrate the wrong
    axes silently."""
    run_dir, prepared = trained_run
    wrapper, config = bp_train.model_load(run_dir)
    collection = load_process_collection(prepared)

    # Rename the modeled component so the RhsOde name ordering no longer matches.
    process = collection.processes["p1"]
    medium = process.reactor_medium
    component = medium.components["biomass"]
    renamed = replace(
        process,
        reactor_medium=replace(
            medium, components={"glucose": replace(component, name="glucose")}
        ),
    )
    foreign = replace(collection, processes={"p1": renamed})

    with pytest.raises(ValueError):
        bp_train.model_predict(wrapper, config, foreign)


# ---------------------------------------------------------------------------
# model_reload
# ---------------------------------------------------------------------------


def test_model_reload_returns_the_same_pair_as_model_load_and_always_warns(
    trained_run, caplog
):
    run_dir, _prepared = trained_run
    wrapper, config = bp_train.model_load(run_dir)

    with caplog.at_level(logging.WARNING, logger="bp_train.serialization"):
        reloaded, reloaded_config = bp_train.model_reload(run_dir, wrapper)
        first = len([r for r in caplog.records if "model_reload" in r.message])
        bp_train.model_reload(run_dir, wrapper)
        second = len([r for r in caplog.records if "model_reload" in r.message])

    # Warned on EVERY call, not deduplicated.
    assert (first, second) == (1, 2)

    # Same 2-tuple contract as model_load, so the two are interchangeable.
    assert reloaded_config.solver.model_dump() == config.solver.model_dump()
    for a, b in zip(
        _trainable_leaves(wrapper), _trainable_leaves(reloaded), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_model_reload_swaps_weights_but_keeps_the_static_half(trained_run):
    run_dir, _prepared = trained_run
    wrapper, config = bp_train.model_load(run_dir)

    # Perturb the trainable leaves, then reload them back from disk.
    perturbed = eqx.apply_updates(
        wrapper,
        jax.tree_util.tree_map(
            lambda leaf: jnp.ones_like(leaf) if eqx.is_inexact_array(leaf) else None,
            bp_train.partition_trainable(wrapper)[0],
        ),
    )
    scales_before = _scale_leaves(perturbed)

    restored, _ = bp_train.model_reload(run_dir, perturbed)

    for a, b in zip(
        _trainable_leaves(wrapper), _trainable_leaves(restored), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # The static half came from the wrapper passed in, never from the checkpoint.
    for before, after in zip(scales_before, _scale_leaves(restored), strict=True):
        np.testing.assert_array_equal(before, after)


def test_model_reload_predicts_identically_to_model_load(trained_run):
    """The documented use: reload a checkpoint of the SAME run, then predict."""
    run_dir, prepared = trained_run
    wrapper, config = bp_train.model_load(run_dir)
    collection = load_process_collection(prepared)

    reloaded, reloaded_config = bp_train.model_reload(
        run_dir / "checkpoints" / "latest", wrapper
    )

    a = bp_train.model_predict(wrapper, config, collection, grid_n=32)
    b = bp_train.model_predict(reloaded, reloaded_config, collection, grid_n=32)
    for name in a:
        np.testing.assert_array_equal(a[name].c_species, b[name].c_species)
