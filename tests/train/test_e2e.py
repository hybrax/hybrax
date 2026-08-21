"""End-to-end tests spanning bp-format input, training, and prediction."""

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest
from hybrax.format.serialization import load_process_collection, save_process_collection

from hybrax.train.harness import (
    TrainHarnessConfig,
    compute_dense_exports,
    prepare_training,
    train_collection,
)
from hybrax.train.model_api import partition_trainable
from hybrax.train.prepare import prepare_artifact
from hybrax.train.run_config import load_prepare_config

from test_serialization import _collection


@pytest.mark.integration
def test_training_end_to_end_from_raw_json(tmp_path: Path):
    raw_path = tmp_path / "raw.json"
    save_process_collection(_collection(), raw_path)
    prepare_config = tmp_path / "prepare-config.json"
    prepare_config.write_text(
        json.dumps({"prepare": {"raw_input": str(raw_path)}}), encoding="utf-8"
    )

    prepared_dir = tmp_path / "prepared"
    prepare_artifact(load_prepare_config(prepare_config), prepared_dir)
    collection = load_process_collection(prepared_dir / "prepared.json")
    assert collection.processes["p1"].biological_ode is not None

    config = TrainHarnessConfig(
        process_names=("p1",),
        target_variable_order=("biomass",),
        target_source="reactor_components",
        epochs=1,
        batch_size=1,
        learning_rate=1e-2,
        solver_rtol=1e-4,
        solver_atol=1e-6,
    )
    training = prepare_training(collection, config=config)

    def trainable_arrays(module):
        trainable, _ = partition_trainable(module)
        return [
            np.asarray(leaf).copy()
            for leaf in jax.tree_util.tree_leaves(trainable)
            if eqx.is_inexact_array(leaf)
        ]

    before = trainable_arrays(training.reaction_module)
    result = train_collection(
        training.store,
        reaction_module=training.reaction_module,
        loss_module=training.loss_module,
        config=training.config,
        optimizer=training.optimizer,
    )
    after = trainable_arrays(result.trained_wrapper.reaction_module)

    assert result.updates_completed == 1
    assert np.isfinite(result.mean_loss_by_step[0])
    assert np.isfinite(result.grad_norm_by_step[0])
    assert result.grad_norm_by_step[0] > 0.0
    assert any(not np.array_equal(a, b) for a, b in zip(before, after, strict=True))

    total_loss, target_loss, exports = compute_dense_exports(
        result.trained_wrapper,
        training.store,
        ("p1",),
        solver_max_steps=config.solver_max_steps,
        solver_rtol=config.solver_rtol,
        solver_atol=config.solver_atol,
        solver_use_jump_ts=config.solver_use_jump_ts,
        prediction_grid_n=8,
    )
    export = exports["p1"]
    assert np.all(np.isfinite(total_loss))
    assert np.all(np.isfinite(target_loss))
    assert export.t.size >= 8
    assert export.c_species.shape[0] == export.t.size
    for values in (
        export.t,
        export.c_species,
        export.v_real,
        export.b_modeled_cum,
        export.q_rates,
        export.modeled_Inflow_rates,
        export.modeled_Outflow_rates,
    ):
        assert np.all(np.isfinite(values))
