"""Training plots: drawn once at the end, and redrawn by `hybrax plot`."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from test_harness import (
    _biomass_loss,
    _LinearReactionModule,
    _make_multi_process_collection,
)

from hybrax.train import cli, postprocessing
from hybrax.train.harness import TrainHarnessConfig, train_collection
from hybrax.train.postprocessing import TRAINING_PLOTS, read_training_history
from hybrax.train.training_data import TrainingDataStore

_PNG = b"\x89PNG\r\n\x1a\n"
_PLOTTED_KEYS = (
    "mean_loss_by_step",
    "per_target_loss_by_step",
    "target_names",
    "grad_norm_by_step",
    "holdout_loss_by_step",
    "holdout_per_target_by_step",
    "holdout_label",
)


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """A run dir with metrics.csv, plus the history training plotted from."""
    run_dir = tmp_path_factory.mktemp("run")
    store = TrainingDataStore.from_collection(
        _make_multi_process_collection(4),
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    plotted = []
    with pytest.MonkeyPatch.context() as patch:
        for kind, write in TRAINING_PLOTS.items():

            def record(history, output_dir, write=write):
                plotted.append(history)
                return write(history, output_dir)

            patch.setitem(TRAINING_PLOTS, kind, record)
        train_collection(
            store,
            reaction_module=_LinearReactionModule(),
            loss_module=_biomass_loss(),
            config=TrainHarnessConfig(
                process_names=("p1", "p2", "p3"),
                holdout_processes=("p4",),
                epochs=4,
                batch_size=3,
                learning_rate=0.05,
                checkpoint_dir=run_dir / "checkpoints",
                checkpoint_every=2,
                metrics_csv=str(run_dir / "metrics.csv"),
            ),
        )
    assert len(plotted) == len(TRAINING_PLOTS)
    return run_dir, plotted[0]


def _assert_same_history(actual, expected):
    for key in _PLOTTED_KEYS:
        a, e = actual[key], expected[key]
        if isinstance(e, dict):
            assert sorted(a) == sorted(e), key
            a, e = [a[k] for k in sorted(a)], [e[k] for k in sorted(e)]
        if key in ("target_names", "holdout_label"):
            assert a == e, key
        else:
            np.testing.assert_allclose(
                np.asarray(a, float), np.asarray(e, float), rtol=1e-9, err_msg=key
            )


def _copy_metrics(trained_run, tmp_path, transform=lambda text: text) -> Path:
    run_dir, _history = trained_run
    path = tmp_path / "metrics.csv"
    path.write_text(transform((run_dir / "metrics.csv").read_text()))
    return path


def test_metrics_csv_reads_back_the_plotted_history(trained_run):
    run_dir, history = trained_run
    assert len(history["mean_loss_by_step"]) == 4
    assert sorted(history["holdout_loss_by_step"]) == [2, 4]
    assert sorted(history["holdout_per_target_by_step"]) == [2, 4]
    _assert_same_history(read_training_history(run_dir / "metrics.csv"), history)


def test_plot_writers_pass_the_whole_history(trained_run, tmp_path, monkeypatch):
    _run_dir, history = trained_run
    calls = {}

    def record(name):
        def plot(values, path, **kwargs):
            calls[name] = (values, path, kwargs)

        return plot

    monkeypatch.setattr(postprocessing, "plot_loss_curve", record("loss"))
    monkeypatch.setattr(postprocessing, "plot_grad_norm_curve", record("grad"))
    for write in TRAINING_PLOTS.values():
        write(history, tmp_path)

    losses, path, kwargs = calls["loss"]
    assert losses == history["mean_loss_by_step"]
    assert path == tmp_path / "loss_curve.png"
    assert kwargs == {
        "title": "Training loss (through step 4)",
        "per_target_loss_by_step": history["per_target_loss_by_step"],
        "target_names": ("biomass",),
        "monitor_loss_by_step": history["holdout_loss_by_step"],
        "monitor_per_target_by_step": history["holdout_per_target_by_step"],
        "monitor_label": "holdout",
    }
    grad_norms, path, kwargs = calls["grad"]
    assert len(grad_norms) == 4
    assert grad_norms == history["grad_norm_by_step"]
    assert path == tmp_path / "grad_norm_curve.png"
    assert kwargs == {"title": "Gradient norm (through step 4)"}


@pytest.mark.parametrize("target", ["dir", "file"])
def test_plot_command_writes_the_training_plots(trained_run, tmp_path, target):
    path = _copy_metrics(trained_run, tmp_path)

    assert cli.main(["plot", str(tmp_path if target == "dir" else path)]) == 0

    for name in ("loss_curve.png", "grad_norm_curve.png"):
        assert (tmp_path / name).read_bytes()[:8] == _PNG


def test_plot_command_draws_only_the_selected_kind(trained_run, tmp_path):
    _copy_metrics(trained_run, tmp_path)

    assert cli.main(["plot", str(tmp_path), "--kind", "grad-norm"]) == 0

    assert (tmp_path / "grad_norm_curve.png").is_file()
    assert not (tmp_path / "loss_curve.png").exists()


@pytest.mark.parametrize(
    "tail",
    [
        b"5,4,3,1,15,0.",
        # Cut inside a multi-byte UTF-8 character.
        "5,4,3,1,15,0.5,0.5,1,\u03b2".encode()[:-1],
        # Cut after a newline inside a quoted field of the last row.
        b'5,4,3,1,15,0.5,0.5,1,"two\nlines',
    ],
    ids=["mid-number", "mid-character", "open-quoted-newline"],
)
def test_metrics_csv_skips_a_partly_written_last_row(trained_run, tmp_path, tail):
    run_dir, history = trained_run
    path = tmp_path / "metrics.csv"
    path.write_bytes((run_dir / "metrics.csv").read_bytes() + tail)

    read = read_training_history(path)

    _assert_same_history(read, history)
    assert cli.main(["plot", str(path)]) == 0


def test_metrics_csv_without_holdout_per_target_column(trained_run, tmp_path):
    _run_dir, history = trained_run

    def drop_column(text):
        rows = list(csv.DictReader(text.splitlines()))
        out = tmp_path / "scratch.csv"
        with out.open("w", newline="") as handle:
            fields = [f for f in rows[0] if f != "holdout_per_target_loss"]
            writer = csv.DictWriter(handle, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return out.read_text()

    path = _copy_metrics(trained_run, tmp_path, drop_column)

    read = read_training_history(path)

    assert read["holdout_per_target_by_step"] == {}
    np.testing.assert_allclose(read["mean_loss_by_step"], history["mean_loss_by_step"])
    assert cli.main(["plot", str(path)]) == 0
    assert (tmp_path / "loss_curve.png").is_file()


def test_plot_command_fails_without_complete_rows(trained_run, tmp_path, capsys):
    path = _copy_metrics(trained_run, tmp_path, lambda text: text.split("\n")[0])

    assert cli.main(["plot", str(tmp_path)]) == 1
    assert "holds no training steps yet" in capsys.readouterr().err
    assert cli.main(["plot", str(tmp_path / "missing")]) == 1
    other = tmp_path / "losses.csv"
    other.write_text("process,total\np1,0.5\n")
    assert cli.main(["plot", str(other)]) == 1
    assert "missing metrics.csv columns" in capsys.readouterr().err
    assert not (path.parent / "loss_curve.png").exists()
