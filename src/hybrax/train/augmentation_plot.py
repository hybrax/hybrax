from __future__ import annotations

from pathlib import Path

import numpy as np
from bp_format.dataclasses import AugmentedBioProcess, BioProcessCollection

from .augmentation import _residual_statistics, _state_series


AUGMENTATION_PLOT_FILENAME = "augmented-data.png"
_SPLINE_GRID_POINTS = 200
_PANEL_WIDTH = 4
_PANEL_HEIGHT = 3
_OUTPUT_DPI = 200


def _state_unit(process, state_name: str) -> str:
    if state_name in process.reactor_medium.components:
        return str(process.reactor_medium.components[state_name].unit)
    return str(process.process_variables[state_name].unit)


def render_augmentation_plot(
    collection: BioProcessCollection,
    variable_names: tuple[str, ...],
    output_path: Path,
) -> None:
    """Plot parent spline fit quality, data, and augmented observations."""
    # Keep non-plotting CLI commands from paying matplotlib's import cost.
    from matplotlib.figure import Figure

    parents = {
        name: process
        for name, process in collection.processes.items()
        if not isinstance(process, AugmentedBioProcess)
    }
    children_by_parent = {name: [] for name in parents}
    for process in collection.processes.values():
        if isinstance(process, AugmentedBioProcess):
            children_by_parent[process.parent_process].append(process)

    figure = Figure(
        figsize=(
            _PANEL_WIDTH * len(variable_names),
            _PANEL_HEIGHT * len(parents),
        )
    )
    axes = figure.subplots(
        len(parents),
        len(variable_names),
        squeeze=False,
    )
    for row, (parent_name, parent) in enumerate(parents.items()):
        for column, state_name in enumerate(variable_names):
            axis = axes[row, column]
            parent_series = _state_series(parent, state_name)
            for child_index, child in enumerate(children_by_parent[parent_name]):
                child_series = _state_series(child, state_name)
                axis.scatter(
                    child_series.times,
                    child_series.values,
                    s=3,
                    color="#888888",
                    alpha=0.35,
                    linewidths=0,
                    label=("Augmented observations" if child_index == 0 else None),
                    zorder=1,
                )

            smooth_times = np.linspace(
                float(parent.time_axis.start),
                float(parent.time_axis.end),
                _SPLINE_GRID_POINTS,
            )
            spline_values = np.asarray(
                parent_series.evaluate_many(smooth_times),
                dtype=float,
            )
            residual_rms, _ = _residual_statistics(
                parent_name,
                state_name,
                parent_series,
            )

            axis.fill_between(
                smooth_times,
                spline_values - residual_rms,
                spline_values + residual_rms,
                color="#A23B72",
                alpha=0.2,
                label="Spline fit +/- residual RMS",
                zorder=2,
            )
            axis.plot(
                smooth_times,
                spline_values,
                color="#2E86AB",
                linewidth=2,
                label="Spline fit",
                zorder=3,
            )
            axis.scatter(
                parent_series.times,
                parent_series.values,
                s=20,
                color="#F24236",
                alpha=0.8,
                label="Original data",
                zorder=4,
            )
            if row == 0:
                axis.set_title(state_name)
            unit_label = f"[{_state_unit(parent, state_name)}]"
            axis.set_ylabel(
                f"{parent_name}\n{unit_label}" if column == 0 else unit_label
            )
            if row == len(parents) - 1:
                axis.set_xlabel(f"Time [{parent.time_axis.unit}]")
            axis.set_xlim(parent.time_axis.start, parent.time_axis.end)
            axis.grid(alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(labels),
    )
    figure.suptitle("Spline fits and augmented observations", y=0.999)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=_OUTPUT_DPI, bbox_inches="tight")
