"""Best-effort prediction plots for ``bp-train forward``."""

import logging
from pathlib import Path

import numpy as np
from bp_format import BioProcess, BioProcessCollection, FeedVolumeChange

from .postprocessing import DenseProcessExport
from .wrapper import HybridOdeWrapper

log = logging.getLogger(__name__)


def plot_forward_predictions(
    collection: BioProcessCollection,
    wrapper: HybridOdeWrapper,
    mean_exports: dict[str, DenseProcessExport],
    std_exports: dict[str, DenseProcessExport] | None,
    losses: dict[str, tuple[float, dict[str, float]]],
    output_dir: Path,
) -> None:
    """Write one prediction figure per dense export, continuing after failures."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir()
    for process_name, mean in mean_exports.items():
        try:
            if (
                not process_name
                or process_name in {".", ".."}
                or "/" in process_name
                or "\\" in process_name
            ):
                raise ValueError(f"process name is not filename-safe: {process_name!r}")
            _plot_process(
                process_name,
                collection.processes[process_name],
                wrapper,
                mean,
                std_exports.get(process_name) if std_exports else None,
                losses.get(process_name, (float("nan"), {})),
                plot_dir / f"{process_name}.png",
            )
        except Exception:
            log.exception("failed to plot predictions for process %s", process_name)


def _plot_process(
    process_name: str,
    process: BioProcess,
    wrapper: HybridOdeWrapper,
    mean: DenseProcessExport,
    std: DenseProcessExport | None,
    losses: tuple[float, dict[str, float]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total_loss, named_losses = losses
    state_names = wrapper.modeled_RMC_names + wrapper.modeled_PV_names
    rate_names = tuple(wrapper.rhs_ode.name_modeled_rates)
    feed_names = wrapper.modeled_FVC_names
    state_rate_rows = max(len(state_names), len(rate_names), 1)
    n_rows = state_rate_rows + 1 + len(feed_names)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(12, 3 * n_rows),
        squeeze=False,
    )
    try:
        t = np.asarray(mean.t)
        for row in range(state_rate_rows):
            if row < len(state_names):
                name = state_names[row]
                measured = _measured_series(process, name)
                r_squared = _plot_prediction(
                    axes[row, 0],
                    t,
                    mean.c_species[:, row],
                    std.c_species[:, row] if std else None,
                    measured,
                )
                axes[row, 0].set_title(
                    _fit_title(name, named_losses.get(name), r_squared)
                )
                axes[row, 0].set_ylabel(_state_unit(process, name))
            else:
                axes[row, 0].set_visible(False)

            if row < len(rate_names):
                _plot_prediction(
                    axes[row, 1],
                    t,
                    mean.q_rates[:, row],
                    std.q_rates[:, row] if std else None,
                    None,
                )
                axes[row, 1].axhline(0, color="gray", linewidth=0.5, linestyle="--")
                axes[row, 1].set_title(rate_names[row])
            else:
                axes[row, 1].set_visible(False)

        volume_row = state_rate_rows
        _plot_volume(axes[volume_row, 0], process, t, mean, std)
        _plot_volume_changes(axes[volume_row, 1], process, t)

        for index, feed_name in enumerate(feed_names):
            row = volume_row + 1 + index
            measured = _modeled_feed_series(process, feed_name, t)
            r_squared = _plot_prediction(
                axes[row, 0],
                t,
                mean.b_modeled_cum[:, index],
                std.b_modeled_cum[:, index] if std else None,
                measured,
                measured_as_line=True,
            )
            loss_name = f"B_{feed_name}_cum"
            axes[row, 0].set_title(
                _fit_title(loss_name, named_losses.get(loss_name), r_squared)
            )
            axes[row, 0].set_ylabel(process.volume.volume_changes[feed_name].unit)
            axes[row, 1].set_visible(False)

        for axis in axes.flat:
            if axis.get_visible():
                axis.grid(alpha=0.25)
                axis.set_xlabel(f"time [{process.time_axis.unit}]")

        title = f"{process_name} — total loss {total_loss:.4g}"
        extra_losses = [
            f"{name}={value:.3g}"
            for name, value in named_losses.items()
            if name not in state_names
            and name not in {f"B_{feed_name}_cum" for feed_name in feed_names}
        ]
        if extra_losses:
            title += "\n" + "  ".join(extra_losses)
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        fig.savefig(temporary_path, format="png", dpi=150, bbox_inches="tight")
        temporary_path.replace(output_path)
    finally:
        if "temporary_path" in locals():
            temporary_path.unlink(missing_ok=True)
        plt.close(fig)


def _plot_prediction(
    axis,
    t: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray | None,
    measured: tuple[np.ndarray, np.ndarray] | None,
    *,
    measured_as_line: bool = False,
) -> float | None:
    mean = np.asarray(mean)
    axis.plot(t, mean, label="prediction")
    if std is not None:
        std = np.asarray(std)
        axis.fill_between(t, mean - std, mean + std, alpha=0.2, label="±1 std")
    r_squared = None
    if measured is not None:
        r_squared = float("nan")
        measured_t, measured_y = measured
        if measured_as_line:
            axis.plot(measured_t, measured_y, color="black", label="recorded")
        else:
            axis.scatter(
                measured_t,
                measured_y,
                color="black",
                s=18,
                zorder=3,
                label="measured",
            )
        if len(measured_y) > 1:
            predicted = np.interp(measured_t, t, mean)
            denominator = np.sum((measured_y - np.mean(measured_y)) ** 2)
            r_squared = (
                1 - np.sum((measured_y - predicted) ** 2) / denominator
                if denominator > 0
                else float("nan")
            )
    axis.legend(fontsize="small")
    return r_squared


def _measured_series(
    process: BioProcess, name: str
) -> tuple[np.ndarray, np.ndarray] | None:
    if name in process.reactor_medium.components:
        values = process.reactor_medium.components[name].concentration
    elif name in process.process_variables:
        values = process.process_variables[name].values
    else:
        return None
    return _time_series_values(values)


def _state_unit(process: BioProcess, name: str) -> str:
    if name in process.reactor_medium.components:
        return process.reactor_medium.components[name].unit
    return process.process_variables[name].unit


def _modeled_feed_series(
    process: BioProcess, name: str, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    change = process.volume.volume_changes.get(name)
    if not isinstance(change, FeedVolumeChange):
        return None
    return t, _volume_change_cumulative(change, t, process.time_axis.start)


def _plot_volume(
    axis,
    process: BioProcess,
    t: np.ndarray,
    mean: DenseProcessExport,
    std: DenseProcessExport | None,
) -> None:
    predicted = np.asarray(mean.v_real)
    expected = _volume_from_changes(process, t)
    axis.plot(t, expected, color="black", label="from volume changes")
    total_volume = _time_series_values(process.volume.total_volume)
    if total_volume is not None:
        measured_t, measured_y = total_volume
        axis.scatter(
            measured_t,
            measured_y,
            color="black",
            s=18,
            zorder=3,
            label="measured",
        )
    else:
        measured_t, measured_y = t, expected
    axis.plot(t, predicted, linestyle="--", label="V_real")
    if std is not None:
        deviation = np.asarray(std.v_real)
        axis.fill_between(
            t,
            predicted - deviation,
            predicted + deviation,
            alpha=0.2,
            label="±1 std",
        )
    predicted_at_measurements = np.interp(measured_t, t, predicted)
    denominator = np.sum((measured_y - np.mean(measured_y)) ** 2)
    r_squared = (
        1 - np.sum((measured_y - predicted_at_measurements) ** 2) / denominator
        if denominator > 0
        else float("nan")
    )
    axis.set_title(_fit_title("V_real", None, r_squared))
    axis.set_ylabel(process.volume.unit)
    axis.legend(fontsize="small")


def _time_series_values(values) -> tuple[np.ndarray, np.ndarray] | None:
    if values is None:
        return None
    times = getattr(values, "times", None)
    data = getattr(values, "values", None)
    if times is None or data is None:
        return None
    return np.asarray(times), np.asarray(data)


def _volume_from_changes(process: BioProcess, t: np.ndarray) -> np.ndarray:
    volume = np.full(t.shape, process.volume.initial_volume, dtype=float)
    for change in process.volume.volume_changes.values():
        volume += _volume_change_cumulative(change, t, process.time_axis.start)
    return volume


def _volume_change_cumulative(
    change, t: np.ndarray, process_start: float
) -> np.ndarray:
    if change.is_continuous:
        values = _continuous_volume_values(change, t)
        baseline = _continuous_volume_values(change, np.asarray([process_start]))[0]
        return values - baseline
    times, values = _time_series_samples(change)
    cumulative = np.cumsum(values)
    indices = np.searchsorted(times, t, side="right") - 1
    contribution = np.zeros_like(t, dtype=float)
    present = indices >= 0
    contribution[present] = cumulative[indices[present]]
    return contribution


def _continuous_volume_values(change, t: np.ndarray) -> np.ndarray:
    if change.values.breaks is not None:
        return np.asarray(change.values.evaluate_many(t))
    times, values = _time_series_samples(change)
    return np.interp(t, times, values, left=values[0], right=values[-1])


def _time_series_samples(change) -> tuple[np.ndarray, np.ndarray]:
    if change.values.times is not None:
        return np.asarray(change.values.times), np.asarray(change.values.values)
    times = np.asarray(change.values.breaks)
    return times, np.asarray(change.values.evaluate_many(times))


def _plot_volume_changes(axis, process: BioProcess, t: np.ndarray) -> None:
    width = max((process.time_axis.end - process.time_axis.start) * 0.008, 0.001)
    for name, change in process.volume.volume_changes.items():
        times, values = _time_series_samples(change)
        if isinstance(change, FeedVolumeChange):
            kind = "feed" if change.is_continuous else "bolus"
        else:
            kind = "sampling"
        if change.is_continuous:
            if change.values.breaks is not None:
                times, values = t, _continuous_volume_values(change, t)
            axis.plot(times, values, label=f"{kind}: {name}")
        else:
            axis.bar(times, values, width=width, alpha=0.6, label=f"{kind}: {name}")
    axis.axhline(0, color="black", linewidth=0.5)
    axis.set_title("volume_changes")
    axis.set_ylabel(process.volume.unit)
    if process.volume.volume_changes:
        axis.legend(fontsize="small")


def _fit_title(name: str, loss: float | None, r_squared: float | None) -> str:
    details = []
    if r_squared is not None:
        details.append(
            f"R²={r_squared:.3g}" if np.isfinite(r_squared) else "R²=undefined"
        )
    if loss is not None:
        details.append(f"loss[{name}]={loss:.3g}")
    return f"{name} ({', '.join(details)})" if details else name
