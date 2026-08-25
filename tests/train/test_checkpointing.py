from __future__ import annotations

import json
import logging
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from hybrax.format.dataclasses import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    ReactorMedium,
    ReactorMediumComponent,
    Outflow,
    TimeAxis,
    TimeSeries,
    Volume,
)

from hybrax.train.checkpointing import CheckpointWriter
from hybrax.train.harness import TrainHarnessConfig, train_collection
from hybrax.train.model_api import (
    ReactionOutputs,
    UserReactionModule,
    frozen_field,
    partition_trainable,
    trainable_field,
)
from hybrax.train.postprocessing import plot_grad_norm_curve, plot_loss_curve
from hybrax.train.serialization import load_trained_wrapper
from hybrax.train.training_data import TrainingDataStore


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


def test_plot_grad_norm_curve_writes_png(tmp_path: Path):
    out = tmp_path / "grad_norm_curve.png"
    plot_grad_norm_curve([2.0, 1.0, 0.5], out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


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
        self.w = jnp.asarray([1.0, 2.0])
        self.frozen = jnp.asarray([9.0])


def _opt_state_for(module: eqx.Module):
    trainable, _ = partition_trainable(module)
    return optax.adam(1e-2).init(trainable)


def test_fractional_checkpoint_boundaries_are_exact_and_distinct():
    from hybrax.train.harness import _checkpoint_update_boundaries

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


def test_automatic_checkpoint_cadence_uses_at_least_five_epochs_and_at_most_20():
    from hybrax.train.harness import _checkpoint_update_boundaries

    assert _checkpoint_update_boundaries(
        None, batches_per_epoch=3, total_updates=30
    ) == frozenset({15, 30})
    assert _checkpoint_update_boundaries(
        None, batches_per_epoch=1, total_updates=100
    ) == frozenset(range(5, 101, 5))
    assert _checkpoint_update_boundaries(
        None, batches_per_epoch=1, total_updates=1500
    ) == frozenset(range(75, 1501, 75))
    assert (
        len(_checkpoint_update_boundaries(None, batches_per_epoch=3, total_updates=303))
        <= 20
    )
    huge_epochs = 18_014_398_509_481_001
    huge_boundaries = _checkpoint_update_boundaries(
        None, batches_per_epoch=1, total_updates=huge_epochs
    )
    assert min(huge_boundaries) == (huge_epochs + 19) // 20
    assert len(huge_boundaries) == 20


def test_checkpoint_writer_keeps_all_and_updates_latest(tmp_path: Path):
    module = _TrainableModule()
    writer = CheckpointWriter(tmp_path / "checkpoints")
    for step in (2, 4):
        writer.write(
            step=step,
            samples_seen=step * 3,
            wrapper=module,
            opt_state=_opt_state_for(module),
            mean_loss=1.0 / step,
            holdout_loss=0.25 if step == 4 else None,
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
    assert not list(checkpoints.glob("step_*/predictions.csv"))
    assert not list(checkpoints.glob("step_*/holdout_predictions.csv"))


def test_checkpoint_writer_normalizes_nonfinite_losses(tmp_path: Path):
    module = _TrainableModule()
    writer = CheckpointWriter(tmp_path / "checkpoints")
    writer.write(
        step=1,
        samples_seen=1,
        wrapper=module,
        opt_state=_opt_state_for(module),
        mean_loss=float("inf"),
        holdout_loss=float("nan"),
    )

    path = tmp_path / "checkpoints" / "latest" / "train_state.json"
    text = path.read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text
    assert json.loads(text)["mean_loss"] is None
    assert json.loads(text)["holdout_loss"] is None


# --------------------------------------------------------------------------
# End-to-end via train_collection
# --------------------------------------------------------------------------


_DEFAULT_CHECKPOINTING_SCALES: dict[str, jnp.ndarray] = {
    "SCALE_modeled_RMCs": jnp.ones(1),
    "SCALE_V_in_cumulative": jnp.asarray(1.0),
    "SCALE_modeled_Inflows_cumulative": jnp.ones(0),
    "SCALE_modeled_Outflows_cumulative": jnp.ones(0),
    "SCALE_controlled_Inflows_cumulative": jnp.ones(0),
    "SCALE_controlled_Inflows_rates": jnp.ones(0),
    "SCALE_controlled_Inflows_Cin": jnp.ones((0, 1)),
    "SCALE_controlled_Outflows_cumulative": jnp.ones(0),
    "SCALE_controlled_Outflows_rates": jnp.ones(0),
    "SCALE_controlled_PVs": jnp.ones(0),
    "SCALE_modeled_Inflows_Cin": jnp.ones((0, 1)),
    "SCALE_modeled_BiologicalOde_rates": jnp.ones(1),
    "SCALE_modeled_Inflows_rates": jnp.ones(0),
    "SCALE_modeled_Outflows_rates": jnp.ones(0),
}


class _LinearReactionModule(UserReactionModule):
    model: eqx.nn.Linear = trainable_field()
    non_model_bias: jax.Array = frozen_field()

    def __init__(self, **scale_kwargs):
        super().__init__(**{**_DEFAULT_CHECKPOINTING_SCALES, **scale_kwargs})
        self.model = eqx.nn.Linear(1, 1, key=jax.random.key(42))
        self.non_model_bias = jnp.asarray([0.05])

    def __call__(self, t, inputs):
        del t
        SCL_modeled_RMCs = inputs.SCL_modeled_RMCs
        rate = self.model(SCL_modeled_RMCs)[0] + self.non_model_bias[0]
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray(
                [rate], dtype=SCL_modeled_RMCs.dtype
            ),
            SCL_modeled_Inflows_rates=jnp.zeros((0,), dtype=SCL_modeled_RMCs.dtype),
            SCL_modeled_Outflows_rates=jnp.zeros((0,), dtype=SCL_modeled_RMCs.dtype),
        )


def _make_collection() -> BioProcessCollection:
    p1 = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "sample_1": Outflow(
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
    checkpoint_every: float | None,
    epochs: int,
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
        config=TrainHarnessConfig(
            process_names=("p1",),
            epochs=epochs,
            batch_size=1,
            optimizer_name="adam",
            learning_rate=5e-2,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
            metrics_csv=metrics_csv,
        ),
    )


def test_train_collection_with_no_checkpoint_dir_is_artifact_free(tmp_path: Path):
    result = _run_train(checkpoint_dir=None, checkpoint_every=1, epochs=2)
    assert result.updates_completed == 2
    assert not (tmp_path / "checkpoints").exists()


def test_train_collection_keeps_periodic_and_final_checkpoints(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    result = _run_train(checkpoint_dir=checkpoints, checkpoint_every=2, epochs=5)
    assert result.updates_completed == 5
    assert sorted(p.name for p in checkpoints.glob("step_*")) == [
        "step_00002",
        "step_00004",
        "step_00005",
    ]
    assert (checkpoints / "latest").resolve().name == "step_00005"
    assert not (checkpoints / "best").exists()


def test_training_writes_run_level_training_plots(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    _run_train(checkpoint_dir=checkpoints, checkpoint_every=1, epochs=1)

    assert not tuple(checkpoints.rglob("*.png"))
    assert (tmp_path / "loss_curve.png").is_file()
    assert (tmp_path / "grad_norm_curve.png").is_file()
    assert not (tmp_path / "predictions.csv").exists()


def test_training_refreshes_plots_at_every_checkpoint(monkeypatch, tmp_path: Path):
    plotted_loss_steps = []
    plotted_grad_norm_steps = []

    def record_loss_plot(losses, *_args, **_kwargs):
        plotted_loss_steps.append(len(losses))

    def record_grad_norm_plot(grad_norms, *_args, **_kwargs):
        plotted_grad_norm_steps.append(len(grad_norms))

    monkeypatch.setattr("hybrax.train.harness.plot_loss_curve", record_loss_plot)
    monkeypatch.setattr(
        "hybrax.train.harness.plot_grad_norm_curve", record_grad_norm_plot
    )

    _run_train(checkpoint_dir=tmp_path / "checkpoints", checkpoint_every=1, epochs=3)

    assert plotted_loss_steps == [1, 2, 3]
    assert plotted_grad_norm_steps == [1, 2, 3]


@pytest.mark.parametrize(
    ("plotter", "message"),
    [
        ("plot_loss_curve", "failed to write loss curve at checkpoint"),
        (
            "plot_grad_norm_curve",
            "failed to write gradient norm curve at checkpoint",
        ),
    ],
)
def test_training_survives_checkpoint_plot_failure(
    monkeypatch, tmp_path, caplog, plotter, message
):
    def fail_plot(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(f"hybrax.train.harness.{plotter}", fail_plot)

    result = _run_train(
        checkpoint_dir=tmp_path / "checkpoints", checkpoint_every=1, epochs=1
    )

    assert result.updates_completed == 1
    assert message in caplog.text


def test_automatic_checkpoint_cadence_is_logged(tmp_path: Path, caplog):
    caplog.set_level(logging.INFO, logger="hybrax.train.harness")

    _run_train(
        checkpoint_dir=tmp_path / "checkpoints", checkpoint_every=None, epochs=10
    )

    assert (
        "checkpoint_every is null; using sensible automatic default "
        "every=5 epochs (2 checkpoints including final)" in caplog.messages
    )


def test_checkpoint_every_zero_still_writes_final(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    _run_train(checkpoint_dir=checkpoints, checkpoint_every=0, epochs=3)
    assert [p.name for p in checkpoints.glob("step_*")] == ["step_00003"]
    state = json.loads((checkpoints / "latest" / "train_state.json").read_text())
    assert state["samples_seen"] == 3


def test_train_collection_checkpoint_params_reload(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    result = _run_train(checkpoint_dir=checkpoints, checkpoint_every=2, epochs=2)
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


# --- `latest` on filesystems that reject symlinks (SMB/NAS shares, WSL drvfs)
# ---


def _write_step(root: Path, name: str) -> Path:
    d = root / name
    (d / "sub").mkdir(parents=True)
    (d / "params.eqx").write_bytes(b"params-" + name.encode())
    (d / "sub" / "nested.bin").write_bytes(b"nested-" + name.encode())
    return d


def _latest_of(root: Path) -> CheckpointWriter:
    w = CheckpointWriter.__new__(CheckpointWriter)
    w._dir = root
    return w


def test_update_latest_uses_a_symlink_when_the_filesystem_allows_it(tmp_path):
    w = _latest_of(tmp_path)
    w._update_latest(_write_step(tmp_path, "step_00100"))
    assert (tmp_path / "latest").is_symlink()
    assert (tmp_path / "latest").resolve().name == "step_00100"


def test_update_latest_falls_back_to_a_copy_when_symlinks_are_refused(
    tmp_path, monkeypatch
):
    """SMB/NAS shares and WSL drvfs reject os.symlink. Training onto such a share must
    still work; readers only ever touch `checkpoints/latest/params.eqx`, which resolves
    in both forms."""
    monkeypatch.setattr(
        "pathlib.Path.symlink_to",
        lambda *a, **k: (_ for _ in ()).throw(OSError(1, "Operation not permitted")),
    )
    w = _latest_of(tmp_path)
    w._update_latest(_write_step(tmp_path, "step_00100"))

    latest = tmp_path / "latest"
    assert latest.is_dir() and not latest.is_symlink()
    assert (latest / "params.eqx").read_bytes() == b"params-step_00100"
    assert (
        latest / "sub" / "nested.bin"
    ).read_bytes() == b"nested-step_00100"  # recurses


def test_copy_fallback_is_replaced_not_merged_on_the_next_checkpoint(
    tmp_path, monkeypatch
):
    """A stale `latest` must be cleared, not written over — otherwise files from an
    older step linger beside the new ones."""
    monkeypatch.setattr(
        "pathlib.Path.symlink_to",
        lambda *a, **k: (_ for _ in ()).throw(OSError(1, "Operation not permitted")),
    )
    w = _latest_of(tmp_path)
    w._update_latest(_write_step(tmp_path, "step_00100"))
    (tmp_path / "latest" / "only_in_the_old_step.txt").write_text("stale")

    w._update_latest(_write_step(tmp_path, "step_00200"))
    latest = tmp_path / "latest"
    assert (latest / "params.eqx").read_bytes() == b"params-step_00200"
    assert not (latest / "only_in_the_old_step.txt").exists()


def test_symlink_form_is_replaced_by_the_copy_form_and_vice_versa(
    tmp_path, monkeypatch
):
    """A directory that moved between filesystems (or a mount whose options changed)
    must not wedge on `latest` already existing in the other form."""
    w = _latest_of(tmp_path)
    w._update_latest(_write_step(tmp_path, "step_00100"))  # symlink
    assert (tmp_path / "latest").is_symlink()

    monkeypatch.setattr(
        "pathlib.Path.symlink_to",
        lambda *a, **k: (_ for _ in ()).throw(OSError(1, "Operation not permitted")),
    )
    w._update_latest(_write_step(tmp_path, "step_00200"))  # -> copy
    assert (tmp_path / "latest").is_dir() and not (tmp_path / "latest").is_symlink()

    monkeypatch.undo()
    w._update_latest(_write_step(tmp_path, "step_00300"))  # -> symlink again
    assert (tmp_path / "latest").is_symlink()
    assert (tmp_path / "latest").resolve().name == "step_00300"
