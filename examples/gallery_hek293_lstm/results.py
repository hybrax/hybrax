"""Select by Br7 all-state loss and compare all 25 synthetic benchmark states."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

import hybrax.format as hxf
import hybrax.train as hxt


def load_benchmark(work):
    """Load measurements and the authors' saved predictions in source order."""
    work = Path(work)
    source = loadmat(work / "source-data.mat", squeeze_me=True, struct_as_record=False)[
        "data"
    ]
    reference = np.load(work / "matlab-reference.npz")["predictions"]
    maxima = np.concatenate([batch.conc.T for batch in source]).max(axis=0)
    return source, reference, maxima


def evaluate_checkpoints(work):
    """Read training losses and Br7 losses at the configured evaluation cadence."""
    work = Path(work)
    metrics = pd.read_csv(work / "run/metrics.csv").set_index("epoch")
    assert np.isfinite(metrics.mean_loss).all()
    assert np.isfinite(metrics.holdout_loss).any(), "No successful Br7 evaluations"
    assert not metrics.holdout_loss.eq(-np.inf).any()
    curves = pd.DataFrame(
        {
            "Train NMSE": metrics.mean_loss,
            "Br7 NMSE": metrics.holdout_loss,
        }
    )
    curves.to_csv(work / "loss-curves.csv")
    return curves


def select_checkpoint(work, curves):
    """Select the minimum finite Br7 max-normalized reconstruction loss."""
    epoch = int(curves["Br7 NMSE"].idxmin())
    assert (Path(work) / f"run/checkpoints/step_{epoch:05d}").is_dir()
    (Path(work) / "selected-checkpoint.txt").write_text(f"{epoch}\n")
    return epoch


def plot_loss_curves(work, curves):
    """Plot training and Br7 NMSE with the saved MATLAB model as a reference."""
    work = Path(work)
    source, reference, maxima = load_benchmark(work)
    errors = ((reference - np.stack([batch.conc.T for batch in source])) / maxima)[
        :, 1:
    ] ** 2
    matlab = {
        "Train": errors[[0, 1, 2, 3, 8]].mean(),
        "Br7": errors[6].mean(),
    }
    selected = select_checkpoint(work, curves)
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for axis, group, color in zip(
        axes, ("Train", "Br7"), ("#17613f", "#b54e28"), strict=True
    ):
        values = curves[f"{group} NMSE"].dropna()
        values = values.where(np.isfinite(values))
        axis.plot(values.index, values, color=color, linewidth=0.6)
        axis.axhline(matlab[group], color="#276d9d", linestyle="--", label="MATLAB")
        axis.set(
            title=group,
            xlabel="Epoch",
            ylabel="NMSE · all 25 states",
        )
        axis.set_yscale("log")
        axis.grid(alpha=0.16)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].plot(
        selected,
        curves.loc[selected, "Br7 NMSE"],
        "o",
        color="#b54e28",
        label=f"Selected epoch {selected:,}",
    )
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(work / "loss-curves.png", dpi=160, facecolor="white")
    plt.close(figure)


def score_selected_checkpoint(work, epoch):
    """Score all splits, reusing Br7 predictions and simulating the other cultures.

    The checkpoint has already been selected using Br7. Model parameters and
    normalization remain fixed during these forward simulations. Pool squared
    normalized errors before taking the root, excluding the initial condition.
    """
    work = Path(work)
    source, reference, maxima = load_benchmark(work)
    wrapper, config = hxt.model_load(work / f"run/checkpoints/step_{epoch:05d}")
    names = config.custom.state_names
    columns = [wrapper.modeled_RMC_names.index(name) for name in names]
    groups = {
        "Training": ("Br1", "Br2", "Br3", "Br4", "Br9"),
        "Validation (Br7)": ("Br7",),
        "Test": ("Br5", "Br6", "Br8"),
    }
    collection = hxf.serialization.load_process_collection(
        work / "prepared/prepared.json"
    )
    predictions = hxt.model_predict(
        wrapper,
        config,
        collection,
        process_names=groups["Training"] + groups["Test"],
        grid_n=21,
    )
    saved_br7 = pd.read_csv(
        work / f"run/checkpoints/step_{epoch:05d}/holdout_predictions.csv"
    )
    assert set(saved_br7.process) == {"Br7"}
    errors = {"Hybrax": {}, "MATLAB": {}}
    for index, batch in enumerate(source):
        name = f"Br{index + 1}"
        if name == "Br7":
            np.testing.assert_array_equal(saved_br7.t, batch.time)
            assert saved_br7.prediction_valid.all()
            values = saved_br7[[f"c_{state}" for state in names]].to_numpy()
        else:
            prediction = predictions[name]
            rows = np.searchsorted(prediction.t, batch.time)
            np.testing.assert_array_equal(prediction.t[rows], batch.time)
            assert prediction.prediction_valid[rows].all(), name
            values = prediction.c_species[rows][:, columns]
        assert np.isfinite(values).all(), name
        errors["Hybrax"][name] = ((values - batch.conc.T) / maxima)[1:] ** 2
        errors["MATLAB"][name] = ((reference[index] - batch.conc.T) / maxima)[1:] ** 2
    scores = pd.DataFrame(
        {
            model: {
                group: np.sqrt(np.stack([squared[name] for name in cultures]).mean())
                for group, cultures in groups.items()
            }
            for model, squared in errors.items()
        }
    )
    scores.to_csv(work / "comparison-scores.csv")
    return scores


def plot_trajectories(work, epoch):
    """Plot all 25 Br7 states; grey panels mark lower Hybrax per-state NRMSE."""
    work = Path(work)
    source, reference, maxima = load_benchmark(work)
    names = json.loads((work / "train-config.json").read_text())["custom"][
        "state_names"
    ]
    frame = pd.read_csv(
        work / f"run/checkpoints/step_{epoch:05d}/holdout_predictions.csv"
    )
    assert set(frame.process) == {"Br7"}
    assert frame.prediction_valid.all()
    batch = source[6]
    np.testing.assert_array_equal(frame.t, batch.time)
    values = frame[[f"c_{name}" for name in names]].to_numpy()
    assert np.isfinite(values).all()
    squared = {
        "Hybrax": ((values - batch.conc.T) / maxima)[1:] ** 2,
        "MATLAB": ((reference[6] - batch.conc.T) / maxima)[1:] ** 2,
    }
    scores = pd.DataFrame(
        {
            label: {
                "All 25": np.sqrt(error.mean()),
                **dict(zip(names, np.sqrt(error.mean(axis=0)), strict=True)),
            }
            for label, error in squared.items()
        }
    ).T
    metrics = pd.read_csv(work / "run/metrics.csv").set_index("epoch")
    np.testing.assert_allclose(
        squared["Hybrax"].mean(), metrics.loc[epoch, "holdout_loss"], rtol=1e-8
    )
    figure, axes = plt.subplots(5, 5, figsize=(16, 15))
    for column, (axis, name) in enumerate(zip(axes.flat, names, strict=True)):
        ours, matlab = scores.loc["Hybrax", name], scores.loc["MATLAB", name]
        axis.set_facecolor("#eeeeee" if ours < matlab else "white")
        axis.plot(frame.t, values[:, column], color="#b54e28", label="Hybrax")
        axis.plot(
            batch.time, reference[6, :, column], "--", color="#276d9d", label="MATLAB"
        )
        axis.plot(
            batch.time,
            batch.conc.T[:, column],
            "o",
            color="#242a29",
            markerfacecolor="white",
            markersize=3,
            label="Synthetic data",
        )
        axis.axvline(96, color="#999999", linestyle=":", linewidth=0.8)
        axis.set_title(f"{name}\nNRMSE {ours:.4f} · MATLAB {matlab:.4f}", fontsize=9)
        axis.set_xlabel("Time [h]", fontsize=8)
        axis.set_ylabel(
            "g/L" if column == 0 else "mg/L" if column == 1 else "mM", fontsize=8
        )
        axis.tick_params(labelsize=8)
        axis.grid(alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(work / "Br7-trajectories.png", dpi=160, facecolor="white")
    plt.close(figure)
    scores.to_csv(work / "Br7-scores.csv")
    return scores


if __name__ == "__main__":
    work = Path(__file__).resolve().parent
    curves = evaluate_checkpoints(work)
    plot_loss_curves(work, curves)
    selected_epoch = select_checkpoint(work, curves)
    print(f"Checkpoint: epoch {selected_epoch:,}")
    print(score_selected_checkpoint(work, selected_epoch))
    print(plot_trajectories(work, selected_epoch))
