"""Select checkpoints by Br7 loss and compare them with the authors' MATLAB model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_cultures
from scipy.io import loadmat

import hybrax.format as hxf
import hybrax.train as hxt

SPLITS = {
    "Train": ("Br1", "Br2", "Br3", "Br4", "Br9"),
    "Validation (Br7)": ("Br7",),
    "Test": ("Br5", "Br6", "Br8"),
}
CULTURES = tuple(f"Br{i}" for i in range(1, 10))
COLORS = {label: style["color"] for label, style in plot_cultures.STYLES.items()}


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


def plot_loss_curves(work, run="run"):
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
            yscale="log",
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
    axes[1].legend(frameon=False, loc="upper right")
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


def measurement_values(work, models):
    """MATLAB and Hybrax predictions at the measurement times, as culture arrays.

    `models` maps labels to dense predictions from `predict_cultures`.
    """
    source, reference, maxima = load_benchmark(Path(work))
    values = {"MATLAB": reference} | {
        label: at_measurements(source, dense) for label, dense in models.items()
    }
    for label, prediction in values.items():
        assert np.isfinite(prediction).all(), label
    return source, values, maxima


def score(work, models, name):
    """Score MATLAB and Hybrax NRMSE at the measurement times of each split."""
    source, values, maxima = measurement_values(work, models)
    scores = pd.DataFrame(
        {
            label: {
                split: nrmse(source, prediction, maxima, cultures)
                for split, cultures in SPLITS.items()
            }
            for label, prediction in values.items()
        }
    ).T
    scores.to_csv(Path(work) / name)
    return scores


def test_scores(work, models, name):
    """Score NRMSE and undershoot on the test cultures only."""
    source, values, maxima = measurement_values(work, models)
    tests = SPLITS["Test"]
    scores = pd.DataFrame(
        {
            label: {
                "Test NRMSE": nrmse(source, prediction, maxima, tests),
                "Test undershoot [%]": undershoot(prediction, maxima, tests),
            }
            for label, prediction in values.items()
        }
    ).T
    scores.to_csv(Path(work) / name)
    return scores


def test_predictions(work, models):
    """Collect test-culture data and predictions in `plot_cultures` format.

    MATLAB's saved predictions exist only at the measurement times.
    """
    source, reference, maxima = load_benchmark(Path(work))
    states = list(np.atleast_1d(source[0].names))
    states[0] = "biomass"
    data, matlab = {}, {}
    for culture in SPLITS["Test"]:
        index = CULTURES.index(culture)
        time = source[index].time
        data[culture] = (time, source[index].conc.T)
        matlab[culture] = (time, reference[index])
    models = {"MATLAB": matlab} | {
        label: {c: dense[c] for c in SPLITS["Test"]} for label, dense in models.items()
    }
    return data, models, states, maxima


def plot_test_cultures(work, models, name):
    """Plot the test cultures' main states for MATLAB and each Hybrax model.

    Saves the plotted data to `test-predictions.npz` first and plots from that
    file, so the file always matches the plot. Each call replaces the file, so
    it holds the models of the last plot. `plot_cultures.py` can redraw it
    without Hybrax, e.g. for a paper figure.
    """
    path = Path(work) / "test-predictions.npz"
    plot_cultures.save_predictions(path, *test_predictions(work, models))
    plot_cultures.plot_test_cultures(
        *plot_cultures.load_predictions(path), Path(work) / name, figsize=(10, 7.4)
    )


if __name__ == "__main__":
    work = Path(__file__).resolve().parent
    epochs = {run: select_checkpoint(work, run) for run in ("run", "run-nonneg")}
    for run, epoch in epochs.items():
        plot_loss_curves(work, run)
        print(f"{run}: checkpoint at epoch {epoch:,}")
    main = {"Hybrax": predict_cultures(work, "run", epochs["run"])}
    print(score(work, main, "comparison-scores.csv"))
    plot_test_cultures(work, main, "test-cultures.png")
    penalty = predict_cultures(work, "run-nonneg", epochs["run-nonneg"])
    both = main | {"Hybrax with penalty": penalty}
    print(test_scores(work, both, "nonneg-scores.csv"))
    plot_test_cultures(work, both, "nonneg-test-cultures.png")
