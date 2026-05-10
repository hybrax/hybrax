from __future__ import annotations

import csv
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
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

from bp_train.checkpointing import CheckpointConfig, CheckpointWriter
import bp_train.harness as harness_module
from bp_train.harness import TrainHarnessConfig, train_collection
from bp_train.model_api import (
    ReactionOutputs,
    UserReactionModule,
    frozen_field,
    trainable_field,
)
from bp_train.postprocessing import plot_loss_curve
from bp_train.training_data import TrainingDataStore


# --------------------------------------------------------------------------
# plot_loss_curve
# --------------------------------------------------------------------------


def test_plot_loss_curve_writes_png(tmp_path: Path):
    out = tmp_path / "curve.png"
    plot_loss_curve([1.0, 0.5, 0.25, 0.1], out)
    assert out.exists()
    assert out.stat().st_size > 0
    # PNG magic number check.
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_loss_curve_accepts_empty(tmp_path: Path):
    out = tmp_path / "curve_empty.png"
    plot_loss_curve([], out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_loss_curve_creates_parent_dir(tmp_path: Path):
    out = tmp_path / "nested" / "dir" / "curve.png"
    plot_loss_curve([1.0, 0.5], out)
    assert out.exists()


# --------------------------------------------------------------------------
# CheckpointWriter in isolation
# --------------------------------------------------------------------------


def _tiny_module() -> eqx.nn.Linear:
    return eqx.nn.Linear(2, 1, key=jax.random.key(0))


def test_checkpoint_writer_disabled_when_every_zero(tmp_path: Path):
    writer = CheckpointWriter(CheckpointConfig(output_dir=tmp_path / "ckpt", every=0))
    assert not writer.enabled
    writer.maybe_write(step=5, wrapper=_tiny_module(), mean_loss_by_step=[1.0])
    assert not (tmp_path / "ckpt").exists()


def test_checkpoint_writer_writes_only_at_cadence(tmp_path: Path):
    out_dir = tmp_path / "ckpt"
    writer = CheckpointWriter(CheckpointConfig(output_dir=out_dir, every=3))
    module = _tiny_module()
    for step in range(1, 11):
        step_dir = writer.maybe_write(
            step=step,
            wrapper=module,
            mean_loss_by_step=[float(step)],
        )
        if step_dir is not None:
            writer.publish_latest(step_dir)
    expected_step_dirs = {"step_00003", "step_00006", "step_00009"}
    actual_step_dirs = {
        p.name for p in out_dir.iterdir() if p.is_dir() and not p.is_symlink()
    }
    assert actual_step_dirs == expected_step_dirs
    # Latest symlink resolves to most-recently-written dir.
    latest = out_dir / "latest"
    assert latest.is_symlink()
    assert latest.resolve().name == "step_00009"


def test_checkpoint_writer_contents_roundtrip(tmp_path: Path):
    out_dir = tmp_path / "ckpt"
    writer = CheckpointWriter(CheckpointConfig(output_dir=out_dir, every=1))
    module = _tiny_module()
    step_dir = writer.maybe_write(
        step=7,
        wrapper=module,
        mean_loss_by_step=[0.9, 0.5, 0.12345],
    )

    assert step_dir == out_dir / "step_00007"
    assert (step_dir / "trained_wrapper.eqx").is_file()
    assert (step_dir / "trained_wrapper.meta.json").is_file()
    assert (step_dir / "loss_curve.png").is_file()

    meta = json.loads((step_dir / "trained_wrapper.meta.json").read_text())
    assert meta["step"] == 7
    assert meta["mean_loss"] == pytest.approx(0.12345)
    assert "timestamp" in meta

    reloaded = eqx.tree_deserialise_leaves(
        step_dir / "trained_wrapper.eqx", like=module
    )
    # Same weights as the source module (since no training happened in-between).
    assert jnp.allclose(reloaded.weight, module.weight)
    assert jnp.allclose(reloaded.bias, module.bias)


def test_checkpoint_writer_does_not_publish_latest_before_finalize(tmp_path: Path):
    out_dir = tmp_path / "ckpt"
    writer = CheckpointWriter(CheckpointConfig(output_dir=out_dir, every=1))
    step_dir = writer.maybe_write(
        step=1,
        wrapper=_tiny_module(),
        mean_loss_by_step=[1.0],
    )
    assert step_dir == out_dir / "step_00001"
    assert not (out_dir / "latest").exists()


def test_checkpoint_writer_latest_updates_across_writes(tmp_path: Path):
    out_dir = tmp_path / "ckpt"
    writer = CheckpointWriter(CheckpointConfig(output_dir=out_dir, every=2))
    module = _tiny_module()
    step_dir = writer.maybe_write(step=2, wrapper=module, mean_loss_by_step=[1.0])
    assert step_dir is not None
    writer.publish_latest(step_dir)
    assert (out_dir / "latest").resolve().name == "step_00002"
    step_dir = writer.maybe_write(step=4, wrapper=module, mean_loss_by_step=[1.0, 0.5])
    assert step_dir is not None
    writer.publish_latest(step_dir)
    assert (out_dir / "latest").resolve().name == "step_00004"


# --------------------------------------------------------------------------
# End-to-end via train_collection
# --------------------------------------------------------------------------


class _LinearReactionModule(UserReactionModule):
    model: eqx.nn.Linear = trainable_field()
    non_model_bias: jax.Array = frozen_field()

    def __init__(self):
        self.model = eqx.nn.Linear(1, 1, key=jax.random.key(42))
        self.non_model_bias = jnp.asarray([0.05], dtype=jnp.float32)

    def __call__(self, t, c_species, controls_vector):
        del t, controls_vector
        rate = self.model(c_species)[0] + self.non_model_bias[0]
        return ReactionOutputs(
            specific_rates=jnp.asarray([rate], dtype=c_species.dtype),
            modeled_feed_rates=jnp.zeros((0,), dtype=c_species.dtype),
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
    log_every: int,
    steps: int,
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
            log_every=log_every,
            checkpoint_dir=checkpoint_dir,
        ),
    )


def test_train_collection_disabled_when_checkpoint_dir_is_none(tmp_path: Path):
    result = _run_train(checkpoint_dir=None, log_every=1, steps=3)
    assert result.trained_wrapper is not None
    # No checkpoints/ directory anywhere under tmp_path (test didn't ask for one).
    assert not (tmp_path / "checkpoints").exists()


def test_train_collection_writes_checkpoints_at_log_every(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoints"
    result = _run_train(checkpoint_dir=ckpt_dir, log_every=2, steps=6)
    assert result.trained_wrapper is not None

    step_dirs = sorted(
        p.name for p in ckpt_dir.iterdir() if p.is_dir() and not p.is_symlink()
    )
    assert step_dirs == ["step_00002", "step_00004", "step_00006"]

    for step in (2, 4, 6):
        d = ckpt_dir / f"step_{step:05d}"
        assert sorted(path.name for path in d.iterdir()) == [
            "grad_norm_curve.png",
            "loss_curve.png",
            "predictions.csv",
            "trained_wrapper.eqx",
            "trained_wrapper.meta.json",
        ]
        meta = json.loads((d / "trained_wrapper.meta.json").read_text())
        assert meta["step"] == step
        with (d / "predictions.csv").open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            first_row = next(reader)
        assert header == ["process", "t", "c_biomass", "V_cont", "V_real", "q_biomass"]
        assert first_row[0] == "p1"

    assert (ckpt_dir / "latest").is_symlink()
    assert (ckpt_dir / "latest").resolve().name == "step_00006"


def test_train_collection_checkpoint_eqx_reloads_into_final_wrapper(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoints"
    result = _run_train(checkpoint_dir=ckpt_dir, log_every=2, steps=4)

    reloaded = eqx.tree_deserialise_leaves(
        ckpt_dir / "step_00004" / "trained_wrapper.eqx",
        like=result.trained_wrapper,
    )
    # Shape / structure match: same reaction module type and trainable leaves.
    trained_leaves = jax.tree_util.tree_leaves(
        eqx.filter(result.trained_wrapper.reaction_module, eqx.is_inexact_array)
    )
    reloaded_leaves = jax.tree_util.tree_leaves(
        eqx.filter(reloaded.reaction_module, eqx.is_inexact_array)
    )
    assert len(trained_leaves) == len(reloaded_leaves)
    for a, b in zip(trained_leaves, reloaded_leaves):
        assert a.shape == b.shape
        assert a.dtype == b.dtype


def test_train_collection_does_not_publish_latest_when_export_fails(
    monkeypatch, tmp_path: Path
):
    ckpt_dir = tmp_path / "checkpoints"

    def _boom(*args, **kwargs):
        raise RuntimeError("export failed")

    monkeypatch.setattr(harness_module, "export_predictions_csv", _boom)

    with pytest.raises(RuntimeError, match="export failed"):
        _run_train(checkpoint_dir=ckpt_dir, log_every=2, steps=2)

    step_dir = ckpt_dir / "step_00002"
    assert step_dir.is_dir()
    assert (step_dir / "trained_wrapper.eqx").is_file()
    assert (step_dir / "trained_wrapper.meta.json").is_file()
    assert (step_dir / "loss_curve.png").is_file()
    assert not (step_dir / "predictions.csv").exists()
    assert not (ckpt_dir / "latest").exists()
