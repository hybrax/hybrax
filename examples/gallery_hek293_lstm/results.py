"""Select checkpoints by Br7 loss and compare them with the authors' MATLAB model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

import hybrax.format as hxf
import hybrax.train as hxt

SPLITS = {
    "Train": ("Br1", "Br2", "Br3", "Br4", "Br9"),
    "Validation (Br7)": ("Br7",),
    "Test": ("Br5", "Br6", "Br8"),
}
CULTURES = tuple(f"Br{i}" for i in range(1, 10))
COLORS = {
    "Data": "#242a29",
    "MATLAB": "#276d9d",
    "Hybrax": "#b54e28",
    "Hybrax, fine-tuned": "#17613f",
}


def load_benchmark(work):
    """Load measurements and the authors' saved predictions in source order."""
    work = Path(work)
    source = loadmat(work / "source-data.mat", squeeze_me=True, struct_as_record=False)[
        "data"
    ]
    reference = np.load(work / "matlab-reference.npz")["predictions"]
    maxima = np.concatenate([batch.conc.T for batch in source]).max(axis=0)
    return source, reference, maxima


def output_path(work, run, name):
    """Prefix outputs of runs other than `run` with their directory name."""
    return Path(work) / (name if run == "run" else f"{run}-{name}")


def read_loss_curves(work, run="run"):
    """Read per-epoch training and Br7 losses from `metrics.csv`.

    Hybrax logs one value per loss term: the 25 concentration fits, followed by
    the 25 `lwr_bnd/<state>` penalty terms if the loss has them. The fit terms
    give the max-normalized NMSE of the concentrations. Hybrax's Br7 loss, used
    for checkpoint selection, is the mean of all terms.
    """
    work = Path(work)
    metrics = pd.read_csv(
        work / run / "metrics.csv",
        dtype={"per_target_loss": str, "holdout_per_target_loss": str},
    ).set_index("epoch")
    # Training uses all five cultures in every update, so one row per epoch.
    assert metrics.index.is_unique
    assert np.isfinite(metrics.mean_loss).all()
    assert np.isfinite(metrics.holdout_loss).any(), "No successful Br7 evaluations"
    names = np.array(metrics.target_names.iloc[0].split(";"))
    penalty = np.char.startswith(names, "lwr_bnd/")

    def terms(column):
        return np.stack([np.array(row.split(";"), dtype=float) for row in column])

    train = terms(metrics.per_target_loss)
    checked = metrics.dropna(subset=["holdout_loss"])
    br7 = terms(checked.holdout_per_target_loss)
    np.testing.assert_allclose(br7.mean(axis=1), checked.holdout_loss, rtol=1e-6)
    curves = pd.DataFrame(
        {"Train NMSE": train[:, ~penalty].mean(axis=1)}, index=metrics.index
    )
    curves["Br7 NMSE"] = pd.Series(br7[:, ~penalty].mean(axis=1), index=checked.index)
    if penalty.any():
        curves["Train penalty"] = train[:, penalty].mean(axis=1)
        curves["Br7 penalty"] = pd.Series(
            br7[:, penalty].mean(axis=1), index=checked.index
        )
    curves["Br7 loss"] = checked.holdout_loss
    curves.to_csv(output_path(work, run, "loss-curves.csv"))
    return curves


def select_checkpoint(work, run="run"):
    """Select the checkpoint with the lowest Br7 loss, as used in training."""
    epoch = int(read_loss_curves(work, run)["Br7 loss"].idxmin())
    assert (Path(work) / f"{run}/checkpoints/step_{epoch:05d}").is_dir()
    output_path(work, run, "selected-checkpoint.txt").write_text(f"{epoch}\n")
    return epoch


def plot_loss_curves(work, run="run", log_scale=True):
    """Plot training and Br7 NRMSE with the saved MATLAB model as a reference.

    With a penalty, dashed lines add it to the fit error, in the same units. On
    Br7, the minimum of that line is the selected checkpoint.
    """
    work = Path(work)
    curves = read_loss_curves(work, run)
    selected = select_checkpoint(work, run)
    source, reference, maxima = load_benchmark(work)
    matlab = {
        "Train": nrmse(source, reference, maxima, SPLITS["Train"]),
        "Br7": nrmse(source, reference, maxima, ("Br7",)),
    }
    penalized = "Br7 penalty" in curves
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for axis, group in zip(axes, ("Train", "Br7"), strict=True):
        values = np.sqrt(curves[f"{group} NMSE"].dropna())
        axis.plot(
            values.index,
            values,
            color=COLORS["Hybrax"],
            linewidth=0.8,
            label="Hybrax, fit only" if penalized else "Hybrax",
        )
        axis.axhline(
            matlab[group], color=COLORS["MATLAB"], linestyle="--", label="MATLAB"
        )
        axis.set(
            title=group,
            xlabel="Epoch",
            ylabel="NRMSE, all 25 states",
            yscale="log" if log_scale else "linear",
        )
        if penalized:
            total = np.sqrt(curves[f"{group} NMSE"] + curves[f"{group} penalty"])
            total = total.dropna()
            axis.plot(
                total.index,
                total,
                "--",
                color=COLORS["Hybrax"],
                linewidth=0.8,
                label="Hybrax, fit + penalty",
            )
        axis.grid(alpha=0.16)
        axis.spines[["top", "right"]].set_visible(False)
    # Selection uses the full Br7 loss, penalty included.
    selected_line = np.sqrt(curves["Br7 NMSE"] + curves.get("Br7 penalty", 0))
    axes[1].plot(
        selected,
        selected_line[selected],
        "o",
        color=COLORS["Hybrax"],
        label=f"Selected epoch {selected:,}",
    )
    if penalized:
        # Sit between the fit-only and fit + penalty lines.
        axes[1].legend(frameon=False, loc="center right", bbox_to_anchor=(1, 0.36))
    else:
        axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(
        output_path(work, run, "loss-curves.png"), dpi=160, facecolor="white"
    )
    plt.close(figure)


def predict_cultures(work, run, epoch, grid_n=241):
    """Simulate all nine cultures with a checkpoint's parameters and scaling.

    Returns dense predictions `{culture: (t, values)}` in source-species order;
    the time grid includes every measurement time.
    """
    work = Path(work)
    wrapper, config = hxt.model_load(work / f"{run}/checkpoints/step_{epoch:05d}")
    columns = [
        wrapper.modeled_RMC_names.index(name) for name in config.custom.state_names
    ]
    collection = hxf.serialization.load_process_collection(
        work / "prepared/prepared.json"
    )
    predictions = hxt.model_predict(
        wrapper, config, collection, process_names=CULTURES, grid_n=grid_n
    )
    dense = {}
    for name, prediction in predictions.items():
        assert prediction.prediction_valid.all(), name
        dense[name] = (prediction.t, prediction.c_species[:, columns])
    return dense


def at_measurements(source, dense):
    """Pick the measurement times from dense predictions, as a culture array."""
    values = []
    for name, batch in zip(CULTURES, source, strict=True):
        t, prediction = dense[name]
        rows = np.searchsorted(t, batch.time)
        np.testing.assert_array_equal(t[rows], batch.time)
        values.append(prediction[rows])
    return np.stack(values)


def nrmse(source, values, maxima, cultures):
    """Pool max-normalized squared errors over cultures, excluding t=0."""
    index = [CULTURES.index(name) for name in cultures]
    truth = np.stack([batch.conc.T for batch in source])
    return np.sqrt(np.mean(((values[index] - truth[index]) / maxima)[:, 1:] ** 2))


def undershoot(values, maxima, cultures):
    """Mean amount below zero as % of each state's maximum, excluding t=0.

    Predictions at or above zero count as zero.
    """
    index = [CULTURES.index(name) for name in cultures]
    return np.mean(np.maximum(-values[index, 1:], 0) / maxima * 100)


def score(work, models, name, with_undershoot=False):
    """Score MATLAB and Hybrax predictions at the measurement times of each split.

    `models` maps labels to dense predictions from `predict_cultures`.
    """
    work = Path(work)
    source, reference, maxima = load_benchmark(work)
    values = {"MATLAB": reference} | {
        label: at_measurements(source, dense) for label, dense in models.items()
    }
    rows = {}
    for label, prediction in values.items():
        assert np.isfinite(prediction).all(), label
        rows[label] = {
            ("NRMSE", split): nrmse(source, prediction, maxima, cultures)
            for split, cultures in SPLITS.items()
        }
        if with_undershoot:
            rows[label] |= {
                ("Undershoot [%]", split): undershoot(prediction, maxima, cultures)
                for split, cultures in SPLITS.items()
            }
    scores = pd.DataFrame(rows).T
    if not with_undershoot:
        scores.columns = scores.columns.droplevel(0)
    scores.to_csv(Path(work) / name)
    return scores


def bold(text):
    """Typeset `text` in bold with matplotlib's mathtext."""
    return r"$\bf{" + text.replace(" ", r"\ ").replace("-", r"\text{-}") + "}$"


def plot_test_cultures(work, models, name, states=("biomass", "mAb", "Glc")):
    """Plot the test cultures' main states for MATLAB and each Hybrax model.

    Panel titles give each model's NRMSE for that culture and state, with the
    lowest in bold.
    """
    work = Path(work)
    source, reference, maxima = load_benchmark(work)
    names = list(np.atleast_1d(source[0].names))
    names[0] = "biomass"
    tests = SPLITS["Test"]
    figure, axes = plt.subplots(len(tests), len(states), figsize=(13, 9.5))
    for row, culture in zip(axes, tests, strict=True):
        index = CULTURES.index(culture)
        batch = source[index]
        for axis, state in zip(row, states, strict=True):
            column = names.index(state)
            truth = batch.conc[column]
            at_samples = {"MATLAB": reference[index, :, column]}
            axis.plot(
                batch.time,
                reference[index, :, column],
                "--",
                color=COLORS["MATLAB"],
                label="MATLAB",
            )
            for label, dense in models.items():
                t, values = dense[culture]
                axis.plot(t, values[:, column], color=COLORS[label], label=label)
                at_samples[label] = values[np.searchsorted(t, batch.time), column]
            axis.plot(
                batch.time,
                truth,
                "o",
                color=COLORS["Data"],
                markerfacecolor="white",
                markersize=3.5,
                label="Synthetic data",
            )
            axis.axhline(0, color="#999999", linewidth=0.6)
            errors = {
                label.removeprefix("Hybrax, "): round(
                    np.sqrt(np.mean(((value - truth) / maxima[column])[1:] ** 2)), 3
                )
                for label, value in at_samples.items()
            }
            scores = " · ".join(
                bold(f"{label} {error:.3f}")
                if error == min(errors.values())
                else f"{label} {error:.3f}"
                for label, error in errors.items()
            )
            axis.set_title(f"{culture}, {state}\n{scores}", fontsize=8.5)
            axis.set(
                xlabel="Time [h]",
                ylabel="g/L" if column == 0 else "mg/L" if column == 1 else "mM",
            )
            axis.grid(alpha=0.15)
            axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # A handle-less entry explains the numbers in the panel titles.
    handles.append(plt.Line2D([], [], linestyle="none"))
    labels.append("Titles: NRMSE")
    figure.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(Path(work) / name, dpi=160, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    work = Path(__file__).resolve().parent
    epochs = {run: select_checkpoint(work, run) for run in ("run", "run-nonneg")}
    for run, epoch in epochs.items():
        plot_loss_curves(work, run, log_scale=run == "run")
        print(f"{run}: checkpoint at epoch {epoch:,}")
    main = {"Hybrax": predict_cultures(work, "run", epochs["run"])}
    print(score(work, main, "comparison-scores.csv"))
    plot_test_cultures(work, main, "test-cultures.png")
    fine_tuned = predict_cultures(work, "run-nonneg", epochs["run-nonneg"])
    both = main | {"Hybrax, fine-tuned": fine_tuned}
    print(score(work, both, "nonneg-scores.csv", with_undershoot=True))
    plot_test_cultures(work, both, "nonneg-test-cultures.png")
