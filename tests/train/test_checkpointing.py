from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from bp_format.dataclasses import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    ReactorMedium,
    ReactorMediumComponent,
    SampleVolumeChange,
    TimeAxis,
    TimeSeries,
    Volume,
)

from bp_train.checkpointing import CheckpointWriter
from bp_train.harness import TrainHarnessConfig, train_collection
from bp_train.model_api import (
    ReactionOutputs,
    UserReactionModule,
    frozen_field,
    partition_trainable,
    trainable_field,
)
from bp_train.postprocessing import plot_loss_curve
from bp_train.serialization import load_trained_wrapper
from bp_train.training_data import TrainingDataStore


# --------------------------------------------------------------------------
# plot_loss_curve (still a pure array→PNG helper)
# --------------------------------------------------------------------------


def test_plot_loss_curve_writes_png(tmp_path: Path):
    out = tmp_path / "curve.png"
    plot_loss_curve([1.0, 0.5, 0.25, 0.1], out)
    assert out.exists()
    assert out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_loss_curve_accepts_empty(tmp_path: Path):
    out = tmp_path / "curve_empty.png"
    plot_loss_curve([], out)
    assert out.exists()


def test_plot_loss_curve_with_per_target_holdout(tmp_path: Path):
    # total + 2 target panels, each with a per-target holdout (monitor) overlay.
    out = tmp_path / "curve_holdout.png"
    plot_loss_curve(
        [1.0, 0.6, 0.3, 0.15],
        out,
        per_target_loss_by_step=[(0.7, 0.3), (0.4, 0.2), (0.2, 0.1), (0.1, 0.05)],
        target_names=["biomass", "glucose"],
        monitor_loss_by_step={2: 0.5, 4: 0.2},
        monitor_per_target_by_step={2: (0.35, 0.15), 4: (0.12, 0.08)},
        monitor_label="holdout",
    )
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------
# CheckpointWriter in isolation
# --------------------------------------------------------------------------


class _TrainableModule(eqx.Module):
    w: jax.Array = trainable_field()
    frozen: jax.Array = frozen_field()

    def __init__(self) -> None:
        self.w = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
        self.frozen = jnp.asarray([9.0], dtype=jnp.float32)


def _opt_state_for(module: eqx.Module):
    trainable, _ = partition_trainable(module)
    return optax.adam(1e-2).init(trainable)


def _dummy_predictions(path: Path) -> None:
    Path(path).write_text("process,t,c_x\np1,0.0,1.0\n", encoding="utf-8")


def test_fractional_checkpoint_boundaries_are_exact_and_distinct():
    from bp_train.harness import _checkpoint_update_boundaries

    assert _checkpoint_update_boundaries(
        0.25, batches_per_epoch=10, total_updates=10
    ) == frozenset({3, 5, 8, 10})
    assert _checkpoint_update_boundaries(
        0.1, batches_per_epoch=10, total_updates=10
    ) == frozenset(range(1, 11))
    assert _checkpoint_update_boundaries(
        0.01, batches_per_epoch=2, total_updates=2
    ) == frozenset({1, 2})
    assert _checkpoint_update_boundaries(
        0, batches_per_epoch=10, total_updates=10
    ) == frozenset({10})


def test_checkpoint_writer_keeps_all_and_updates_latest(tmp_path: Path):
    module = _TrainableModule()
    writer = CheckpointWriter(
        tmp_path / "checkpoints", plotter=None, plots_enabled=False
    )
    for step in (2, 4):
        writer.write(
            step=step,
            samples_seen=step * 3,
            wrapper=module,
            opt_state=_opt_state_for(module),
            mean_loss=1.0 / step,
            holdout_loss=0.25 if step == 4 else None,
            render_predictions_fn=_dummy_predictions,
            loss_by_step=[1.0],
        )
    checkpoints = tmp_path / "checkpoints"
    assert {p.name for p in checkpoints.glob("step_*")} == {
        "step_00002",
        "step_00004",
    }
    assert (checkpoints / "latest").resolve().name == "step_00004"
    assert not (checkpoints / "best").exists()
    state = json.loads((checkpoints / "latest" / "train_state.json").read_text())
    assert state["samples_seen"] == 12
    assert state["holdout_loss"] == pytest.approx(0.25)
    reloaded = load_trained_wrapper(
        checkpoints / "latest" / "params.eqx", template=_TrainableModule()
    )
    assert jnp.allclose(reloaded.w, module.w)


def test_checkpoint_writer_export_failure_does_not_publish(tmp_path: Path):
    module = _TrainableModule()
    writer = CheckpointWriter(
        tmp_path / "checkpoints", plotter=None, plots_enabled=False
    )

    def boom(_path):
        raise RuntimeError("export failed")

    with pytest.raises(RuntimeError, match="export failed"):
        writer.write(
            step=1,
            samples_seen=1,
            wrapper=module,
            opt_state=_opt_state_for(module),
            mean_loss=1.0,
            holdout_loss=None,
            render_predictions_fn=boom,
            loss_by_step=[1.0],
        )
    assert not (tmp_path / "checkpoints" / "latest").exists()


# --------------------------------------------------------------------------
# End-to-end via train_collection
# --------------------------------------------------------------------------


_DEFAULT_CHECKPOINTING_SCALES: dict[str, jnp.ndarray] = {
    "SCALE_modeled_RMCs": jnp.ones(1, dtype=jnp.float32),
    "SCALE_V_in_cumulative": jnp.asarray(1.0, dtype=jnp.float32),
    "SCALE_modeled_FVCs_cumulative": jnp.ones(0, dtype=jnp.float32),
    "SCALE_controlled_FVCs_cumulative": jnp.ones(0, dtype=jnp.float32),
    "SCALE_controlled_FVCs_rates": jnp.ones(0, dtype=jnp.float32),
    "SCALE_controlled_FVCs_Cin": jnp.ones((0, 1), dtype=jnp.float32),
    "SCALE_controlled_PVs": jnp.ones(0, dtype=jnp.float32),
    "SCALE_modeled_FVCs_Cin": jnp.ones((0, 1), dtype=jnp.float32),
    "SCALE_modeled_BiologicalOde_rates": jnp.ones(1, dtype=jnp.float32),
    "SCALE_modeled_FVCs_rates": jnp.ones(0, dtype=jnp.float32),
}


class _LinearReactionModule(UserReactionModule):
    model: eqx.nn.Linear = trainable_field()
    non_model_bias: jax.Array = frozen_field()

    def __init__(self, **scale_kwargs):
        super().__init__(**{**_DEFAULT_CHECKPOINTING_SCALES, **scale_kwargs})
        self.model = eqx.nn.Linear(1, 1, key=jax.random.key(42))
        self.non_model_bias = jnp.asarray([0.05], dtype=jnp.float32)

    def __call__(self, t, inputs):
        del t
        SCL_modeled_RMCs = inputs.SCL_modeled_RMCs
        rate = self.model(SCL_modeled_RMCs)[0] + self.non_model_bias[0]
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray(
                [rate], dtype=SCL_modeled_RMCs.dtype
            ),
            SCL_modeled_FVCs_rates=jnp.zeros((0,), dtype=SCL_modeled_RMCs.dtype),
        )


def _make_collection() -> BioProcessCollection:
    p1 = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "sample_1": SampleVolumeChange(
                    name="sample_1",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([1.0]),
                        values=jnp.asarray([-0.1]),
                    ),
                )
            },
        ),
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray([0.0, 1.0, 2.0]),
                        values=jnp.asarray([1.0, 0.8, 0.64]),
                    ),
                ),
            },
        ),
        process_variables={},
    )
    return BioProcessCollection(processes={"p1": p1}, metadata={})


def _run_train(
    *,
    checkpoint_dir: Path | None,
    checkpoint_every: float,
    epochs: int,
    plots: bool = False,
    metrics_csv: str | None = None,
):
    collection = _make_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    return train_collection(
        store,
        reaction_module=_LinearReactionModule(),
        collection=collection,
        config=TrainHarnessConfig(
            process_names=("p1",),
            epochs=epochs,
            batch_size=1,
            optimizer_name="adam",
            learning_rate=5e-2,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
            plots=plots,
            metrics_csv=metrics_csv,
        ),
    )


def test_train_collection_with_no_checkpoint_dir_is_artifact_free(tmp_path: Path):
    result = _run_train(checkpoint_dir=None, checkpoint_every=1, epochs=2)
    assert result.updates_completed == 2
    assert not (tmp_path / "checkpoints").exists()


def test_train_collection_keeps_periodic_and_final_checkpoints(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    result = _run_train(
        checkpoint_dir=checkpoints, checkpoint_every=2, epochs=5, plots=False
    )
    assert result.updates_completed == 5
    assert sorted(p.name for p in checkpoints.glob("step_*")) == [
        "step_00002",
        "step_00004",
        "step_00005",
    ]
    assert (checkpoints / "latest").resolve().name == "step_00005"
    assert not (checkpoints / "best").exists()


def test_checkpoint_every_zero_still_writes_final(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    _run_train(checkpoint_dir=checkpoints, checkpoint_every=0, epochs=3)
    assert [p.name for p in checkpoints.glob("step_*")] == ["step_00003"]
    state = json.loads((checkpoints / "latest" / "train_state.json").read_text())
    assert state["samples_seen"] == 3


def test_train_collection_checkpoint_params_reload(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    result = _run_train(
        checkpoint_dir=checkpoints, checkpoint_every=2, epochs=2, plots=False
    )
    reloaded = load_trained_wrapper(
        checkpoints / "latest" / "params.eqx", template=result.trained_wrapper
    )
    trained = jax.tree_util.tree_leaves(
        eqx.filter(result.trained_wrapper.reaction_module, eqx.is_inexact_array)
    )
    got = jax.tree_util.tree_leaves(
        eqx.filter(reloaded.reaction_module, eqx.is_inexact_array)
    )
    assert len(trained) == len(got) and len(trained) > 0
    for expected, actual in zip(trained, got):
        assert expected.shape == actual.shape
        assert jnp.allclose(expected, actual)
