"""Plot test-culture predictions of MATLAB and the Hybrax models.

The Hybrax paper draws its HEK293 figure with this module and the bundled
`test-predictions.npz`, downloaded from a pinned commit of this repository, so
the paper and this gallery example show the same plot from the same data. This
is why the module needs only NumPy and Matplotlib and does not import Hybrax:
the figure can be redrawn without installing Hybrax or retraining the models.
`results.py` writes `test-predictions.npz` and draws the gallery plots from it.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

# Okabe-Ito hues, darkened where needed for at least 4.5:1 text contrast on
# white. Line styles tell the models apart without color.
STYLES = {
    "MATLAB": {"color": "#737373", "linestyle": "--", "zorder": 1},
    "Hybrax": {"color": "#BE5400", "linestyle": "-", "zorder": 2},
    "Hybrax, fine-tuned": {"color": "#0072B2", "linestyle": "-.", "zorder": 3},
}
LEGEND = {"Hybrax, fine-tuned": "Hybrax, fine-tuned with penalty"}
DATA_COLOR = "#242a29"
STATES = {"biomass": "Biomass [g/L]", "mAb": "mAb [mg/L]", "Glc": "Glucose [mM]"}


def save_predictions(path, data, models, states, maxima):
    """Save measurements and model predictions for the cultures in `data`.

    `data` maps each culture to `(time, conc)` with `conc` of shape
    `(len(time), len(states))`. `models` maps each label to
    `{culture: (t, values)}` in the same state order.
    """
    cultures = list(data)
    arrays = {"cultures": np.array(cultures), "states": np.array(states)}
    arrays |= {"maxima": np.asarray(maxima), "models": np.array(list(models))}
    for culture in cultures:
        arrays[f"data|{culture}|t"], arrays[f"data|{culture}|values"] = data[culture]
        for label, dense in models.items():
            t, values = dense[culture]
            arrays[f"{label}|{culture}|t"] = t
            arrays[f"{label}|{culture}|values"] = values
    np.savez_compressed(path, **arrays)


def load_predictions(path):
    """Load what `save_predictions` wrote, as `(data, models, states, maxima)`."""
    with np.load(path) as archive:
        cultures = list(archive["cultures"])

        def series(label):
            return {
                c: (archive[f"{label}|{c}|t"], archive[f"{label}|{c}|values"])
                for c in cultures
            }

        models = {label: series(label) for label in archive["models"]}
        return series("data"), models, list(archive["states"]), archive["maxima"]


def at_times(dense, time):
    """Pick the rows of a dense prediction at the measurement times."""
    t, values = dense
    rows = np.searchsorted(t, time)
    np.testing.assert_array_equal(t[rows], time)
    return values[rows]


def nrmse(errors, maxima):
    """Root mean square of max-normalized errors."""
    return np.sqrt(np.mean((errors / maxima) ** 2))


def colored_scores(axis, errors, size):
    """Write each model's NRMSE above the panel in its line color.

    An asterisk marks the lowest, including ties.
    """
    best = min(errors.values())
    bold = {"fontsize": size, "fontweight": "bold"}
    parts = [TextArea("NRMSE:", textprops=bold)]
    for i, (label, error) in enumerate(errors.items()):
        text = f"{error:.3f}" + ("*" if error == best else "")
        text += "," if i < len(errors) - 1 else ""
        parts.append(TextArea(text, textprops=bold | {"color": STYLES[label]["color"]}))
    axis.add_artist(
        AnchoredOffsetbox(
            loc="lower center",
            child=HPacker(children=parts, sep=0.4 * size, pad=0),
            bbox_to_anchor=(0.5, 1.04),
            bbox_transform=axis.transAxes,
            frameon=False,
            pad=0,
            borderpad=0,
        )
    )


def plot_test_cultures(
    data, models, states, maxima, path, row_labels=None, dpi=160, **kwargs
):
    """Plot biomass, mAb and glucose of each culture in `data`, one row each.

    Panel titles give each model's NRMSE for that culture and state; the legend
    gives the NRMSE over all states and cultures, excluding t=0. Font sizes and
    line widths follow Matplotlib's `rcParams`; `kwargs` go to `plt.subplots`.
    """
    cultures = list(data)
    row_labels = row_labels or cultures
    size = plt.rcParams["font.size"]
    figure, axes = plt.subplots(len(cultures), len(STATES), sharex=True, **kwargs)
    for row, culture, row_label in zip(axes, cultures, row_labels, strict=True):
        time, truth = data[culture]
        at_samples = {
            label: at_times(dense[culture], time) for label, dense in models.items()
        }
        for i, (axis, state) in enumerate(zip(row, STATES, strict=True)):
            column = states.index(state)
            for label, dense in models.items():
                t, values = dense[culture]
                axis.plot(t, values[:, column], label=label, **STYLES[label])
            axis.plot(
                time,
                truth[:, column],
                "o",
                color=DATA_COLOR,
                markerfacecolor="white",
                markersize=0.55 * plt.rcParams["lines.markersize"],
                markeredgewidth=0.6,
                zorder=4,
                label="Synthetic data",
            )
            axis.axhline(0, color="#c8c8c8", linewidth=0.5, zorder=0)
            errors = {
                # As in `results.nrmse`, the initial condition is excluded.
                label: round(
                    nrmse(values[1:, column] - truth[1:, column], maxima[column]), 3
                )
                for label, values in at_samples.items()
            }
            colored_scores(axis, errors, size)
            if culture == cultures[0]:
                axis.annotate(
                    STATES[state],
                    xy=(0.5, 1.2),
                    xycoords="axes fraction",
                    ha="center",
                    va="bottom",
                    fontsize=size + 1,
                    fontweight="bold",
                )
            if i == 0:
                axis.set_ylabel(row_label, fontsize=size + 1, fontweight="bold")
            if culture == cultures[-1]:
                axis.set_xlabel("Time [h]")
            axis.grid(color="#e6e6e6", linewidth=0.4)
            axis.spines[["top", "right"]].set_visible(False)
    totals = {
        label: nrmse(
            np.concatenate(
                [at_times(dense[c], data[c][0])[1:] - data[c][1][1:] for c in cultures]
            ),
            maxima,
        )
        for label, dense in models.items()
    }
    handles, labels = axes[0, 0].get_legend_handles_labels()
    labels = [
        f"{LEGEND.get(label, label)} ({totals[label]:.3f})"
        if label in totals
        else label
        for label in labels
    ]
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        handlelength=2.6,
        columnspacing=1.2,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.6, w_pad=1.0)
    figure.savefig(path, dpi=dpi, facecolor="white")
    plt.close(figure)
