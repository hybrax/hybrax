"""Plot the nine synthetic cultures in physical units, before model fitting."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def plot_data(work):
    """Write a process overview and all 25 concentration trajectories."""
    work = Path(work)
    source = loadmat(work / "source-data.mat", squeeze_me=True, struct_as_record=False)[
        "data"
    ]
    names = source[0].names.tolist()
    # Split determines the color family; shade and pattern identify each culture.
    styles = {
        1: ("train", "#111111", "-"),
        2: ("train", "#383838", "--"),
        3: ("train", "#606060", "-."),
        4: ("train", "#858585", ":"),
        9: ("train", "#aaaaaa", (0, (5, 1, 1, 1, 1, 1))),
        7: ("validation", "#d55e00", "-"),
        5: ("test", "#b2182b", "-"),
        6: ("test", "#df536b", "--"),
        8: ("test", "#ef94ad", "-."),
    }
    labels = {"Xv": "Biomass", "mAb": "mAb", "Glc": "Glucose", "Gln": "Glutamine"}

    def draw(axis, values, title, unit, *, intervals=False):
        for i, (batch, y) in enumerate(zip(source, values, strict=True), 1):
            role, color, style = styles[i]
            line_style = {
                "linestyle": style,
                "linewidth": 1.2,
                "color": color,
                "label": f"Br{i} ({role})",
            }
            if intervals:
                axis.stairs(y, batch.time, baseline=None, **line_style)
            else:
                axis.plot(batch.time, y, marker="o", markersize=2, **line_style)
        axis.set(title=title, xlabel="Time [h]", ylabel=unit, xlim=(0, 240))
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)

    def save(figure, axes, filename):
        handles, legend = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, legend, loc="upper center", ncol=3, frameon=False)
        figure.tight_layout(rect=(0, 0, 1, 0.91))
        figure.savefig(work / filename, dpi=160, facecolor="white")
        plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    for axis, name in zip(axes.flat, ("Xv", "mAb", "Glc", "Gln")):
        column = names.index(name)
        unit = "g/L" if column == 0 else "mg/L" if column == 1 else "mM"
        draw(axis, [b.conc[column] for b in source], labels[name], unit)
    draw(axes[1, 1], [b.vol / 1000 for b in source], "Reactor volume", "L")
    draw(
        axes[1, 2],
        [np.diff(b.vol) / 1000 / np.diff(b.time) for b in source],
        "Feed rate",
        "L/h",
        intervals=True,
    )
    save(figure, axes, "synthetic-overview.png")

    figure, axes = plt.subplots(5, 5, figsize=(15, 15))
    for column, (axis, name) in enumerate(zip(axes.flat, names, strict=True)):
        unit = "g/L" if column == 0 else "mg/L" if column == 1 else "mM"
        draw(axis, [b.conc[column] for b in source], labels.get(name, name), unit)
        axis.tick_params(labelsize=8)
    save(figure, axes, "synthetic-all-states.png")


if __name__ == "__main__":
    plot_data(Path(__file__).resolve().parent)
