from __future__ import annotations

import csv
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
import bp_train.harness as harness_module
from bp_train.harness import TrainHarnessConfig, train_collection
from bp_train.model_api import (
    ReactionOutputs,
    UserReactionModule,
    frozen_field,
    partition_trainable,
    trainable_field,
)
from bp_train.postprocessing import plot_loss_curve
from bp_train.run_config import CheckpointConfig
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


def _writer(tmp_path: Path, *, every: int, keep: str = "best+latest") -> CheckpointWriter:
    return CheckpointWriter(
        tmp_path / "checkpoints",
        CheckpointConfig(every=every, keep=keep),
        plotter=None,
        plots_enabled=False,
    )


def test_checkpoint_writer_disabled_when_every_zero(tmp_path: Path):
    writer = _writer(tmp_path, every=0)
    assert not writer.enabled
    out = writer.maybe_write(
        step=5,
        wrapper=_TrainableModule(),
        opt_state=_opt_state_for(_TrainableModule()),
        mean_loss=1.0,
        best_loss=1.0,
        render_predictions_fn=_dummy_predictions,
        loss_by_step=[1.0],
    )
    assert out is None
    assert not (tmp_path / "checkpoints").exists()


def test_checkpoint_writer_cadence_and_latest(tmp_path: Path):
    module = _TrainableModule()
    opt_state = _opt_state_for(module)
    writer = _writer(tmp_path, every=3, keep="all")
    for step in range(1, 11):
        writer.maybe_write(
            step=step,
            wrapper=module,
            opt_state=opt_state,
            mean_loss=float(step),
            best_loss=float(step),
            render_predictions_fn=_dummy_predictions,
            loss_by_step=[float(step)],
        )
    ckpt = tmp_path / "checkpoints"
    step_dirs = {p.name for p in ckpt.iterdir() if p.is_dir() and not p.is_symlink()}
    assert step_dirs == {"step_00003", "step_00006", "step_00009"}
    assert (ckpt / "latest").resolve().name == "step_00009"


def test_checkpoint_writer_writes_resumable_state_and_roundtrips(tmp_path: Path):
    module = _TrainableModule()
    opt_state = _opt_state_for(module)
    writer = _writer(tmp_path, every=1)
    d = writer.maybe_write(
        step=7,
        wrapper=module,
        opt_state=opt_state,
        mean_loss=0.12345,
        best_loss=0.1,
        render_predictions_fn=_dummy_predictions,
        loss_by_step=[0.9, 0.5, 0.12345],
    )
    assert d == tmp_path / "checkpoints" / "step_00007"
    assert (d / "params.eqx").is_file()
    assert (d / "opt_state.eqx").is_file()
    assert (d / "train_state.json").is_file()
    assert (d / "predictions.csv").is_file()

    state = json.loads((d / "train_state.json").read_text())
    assert state["step"] == 7
    assert state["mean_loss"] == pytest.approx(0.12345)
    assert state["best_loss"] == pytest.approx(0.1)
    assert "timestamp" in state

    reloaded = load_trained_wrapper(d / "params.eqx", template=_TrainableModule())
    assert jnp.allclose(reloaded.w, module.w)


def test_checkpoint_writer_best_latest_pruning(tmp_path: Path):
    module = _TrainableModule()
    opt_state = _opt_state_for(module)
    writer = _writer(tmp_path, every=1, keep="best+latest")
    # loss: step1=1.0, step2=0.2 (best), step3=0.5, step4=0.8 (latest)
    losses = {1: 1.0, 2: 0.2, 3: 0.5, 4: 0.8}
    for step, loss in losses.items():
        writer.maybe_write(
            step=step,
            wrapper=module,
            opt_state=opt_state,
            mean_loss=loss,
            best_loss=min(list(losses.values())[:step]),
            render_predictions_fn=_dummy_predictions,
            loss_by_step=[loss],
        )
    ckpt = tmp_path / "checkpoints"
    surviving = {p.name for p in ckpt.iterdir() if p.is_dir() and not p.is_symlink()}
    assert surviving == {"step_00002", "step_00004"}  # best + latest
    assert (ckpt / "best").resolve().name == "step_00002"
    assert (ckpt / "latest").resolve().name == "step_00004"


def test_checkpoint_writer_export_failure_does_not_publish(tmp_path: Path):
    module = _TrainableModule()
    opt_state = _opt_state_for(module)
    writer = _writer(tmp_path, every=1)

    def _boom(_path):
        raise RuntimeError("export failed")

    with pytest.raises(RuntimeError, match="export failed"):
        writer.maybe_write(
            step=1,
            wrapper=module,
            opt_state=opt_state,
            mean_loss=1.0,
            best_loss=1.0,
            render_predictions_fn=_boom,
            loss_by_step=[1.0],
        )
    ckpt = tmp_path / "checkpoints"
    d = ckpt / "step_00001"
    assert (d / "params.eqx").is_file()
    assert (d / "opt_state.eqx").is_file()
    assert (d / "train_state.json").is_file()
    assert not (d / "predictions.csv").exists()
    assert not (ckpt / "latest").exists()
    assert not (ckpt / "best").exists()


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
    checkpoint_every: int,
    steps: int,
    checkpoint_keep: str = "best+latest",
    plots: bool = False,
    metrics_csv: str | None = None,
    start_step: int = 0,
    initial_trainable_params=None,
    initial_optimizer_state=None,
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
            steps=steps,
            batch_size=1,
            optimizer_name="adam",
            learning_rate=5e-2,
            log_every=1,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
            checkpoint_keep=checkpoint_keep,
            plots=plots,
            metrics_csv=metrics_csv,
        ),
        start_step=start_step,
        initial_trainable_params=initial_trainable_params,
        initial_optimizer_state=initial_optimizer_state,
    )


def test_train_collection_disabled_when_checkpoint_dir_is_none(tmp_path: Path):
    result = _run_train(checkpoint_dir=None, checkpoint_every=1, steps=3)
    assert result.trained_wrapper is not None
    assert not (tmp_path / "checkpoints").exists()


def test_train_collection_writes_resumable_checkpoints(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoints"
    result = _run_train(
        checkpoint_dir=ckpt_dir, checkpoint_every=2, steps=6, plots=False
    )
    assert result.optimizer_state is not None
    assert result.steps_completed == 6

    # best+latest retention: only the best/latest step dirs survive (they may
    # coincide when loss decreases monotonically — then a single dir remains).
    step_dirs = {
        p.name for p in ckpt_dir.iterdir() if p.is_dir() and not p.is_symlink()
    }
    assert (ckpt_dir / "latest").resolve().name == "step_00006"
    assert (ckpt_dir / "best").is_symlink()
    expected = {
        (ckpt_dir / "latest").resolve().name,
        (ckpt_dir / "best").resolve().name,
    }
    assert step_dirs == expected
    assert 1 <= len(step_dirs) <= 2

    latest = ckpt_dir / "latest"
    assert sorted(p.name for p in latest.resolve().iterdir()) == [
        "opt_state.eqx",
        "params.eqx",
        "predictions.csv",
        "train_state.json",
    ]
    with (latest / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert header == ["process", "t", "c_biomass", "V_real", "q_biomass"]


def test_train_collection_keep_all_retains_every_checkpoint(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoints"
    _run_train(
        checkpoint_dir=ckpt_dir,
        checkpoint_every=2,
        steps=6,
        checkpoint_keep="all",
        plots=False,
    )
    step_dirs = sorted(
        p.name for p in ckpt_dir.iterdir() if p.is_dir() and not p.is_symlink()
    )
    assert step_dirs == ["step_00002", "step_00004", "step_00006"]


def test_resume_continues_bit_identically(tmp_path: Path):
    # Reference: a single 6-step run.
    full = _run_train(
        checkpoint_dir=None,
        checkpoint_every=0,
        steps=6,
        metrics_csv=str(tmp_path / "full.csv"),
    )
    # Split: train 3 steps, then resume to 6 reusing trainable params + opt state.
    metrics_split = tmp_path / "split.csv"
    run1 = _run_train(
        checkpoint_dir=None,
        checkpoint_every=0,
        steps=3,
        metrics_csv=str(metrics_split),
    )
    trainable1, _ = partition_trainable(run1.trained_wrapper)
    resumed = _run_train(
        checkpoint_dir=None,
        checkpoint_every=0,
        steps=6,
        metrics_csv=str(metrics_split),  # same file → appended on resume
        start_step=3,
        initial_trainable_params=trainable1,
        initial_optimizer_state=run1.optimizer_state,
    )
    # The resumed session covers exactly steps 4..6 ...
    assert len(resumed.mean_loss_by_step) == 3
    # ... and those losses match the single-run trajectory bit-for-bit.
    assert resumed.mean_loss_by_step == pytest.approx(
        full.mean_loss_by_step[3:], rel=1e-6, abs=1e-8
    )
    # metrics.csv was appended (3 + 3 = 6 rows).
    import pandas as pd

    assert len(pd.read_csv(metrics_split)) == 6


def test_train_collection_checkpoint_params_reload(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoints"
    result = _run_train(
        checkpoint_dir=ckpt_dir, checkpoint_every=2, steps=4, plots=False
    )
    reloaded = load_trained_wrapper(
        ckpt_dir / "latest" / "params.eqx", template=result.trained_wrapper
    )
    trained = jax.tree_util.tree_leaves(
        eqx.filter(result.trained_wrapper.reaction_module, eqx.is_inexact_array)
    )
    got = jax.tree_util.tree_leaves(
        eqx.filter(reloaded.reaction_module, eqx.is_inexact_array)
    )
    assert len(trained) == len(got) and len(trained) > 0
    for a, b in zip(trained, got):
        assert a.shape == b.shape and jnp.allclose(a, b)
