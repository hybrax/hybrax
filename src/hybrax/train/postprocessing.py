"""Post-training outputs: model serialization and result plots."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from bp_format.dataclasses import BioProcessCollection, FeedVolumeChange
from bp_format.json_io import load_json

from .training_data import TrainingDataStore
from .wrapper import HybridOdeWrapper, SaveOutputs

logger = logging.getLogger(__name__)

# Bolus bar width in `plot_process_simulations`, expressed as a fraction of the
# process time span.
BAR_WIDTH_FRACTION = 0.02


@dataclass(frozen=True)
class DenseProcessExport:
    """Dense, human-facing per-process export arrays in physical units.

    ``q_rates`` is aligned with ``rhs_ode.name_modeled_rates`` (the modelled
    rate vector), not with ``modeled_RMC_names``. Under a user-defined
    ``BiologicalOde`` these orderings differ.
    """

    t: np.ndarray
    c_species: np.ndarray
    v_real: np.ndarray
    b_modeled_cum: np.ndarray
    q_rates: np.ndarray
    auxiliary: dict[str, np.ndarray] | None = None


def aggregate_dense_exports(
    per_model: list[dict[str, DenseProcessExport]],
) -> tuple[dict[str, DenseProcessExport], dict[str, DenseProcessExport]]:
    """Stack per-model dense exports into per-process (mean, std) exports.

    All models must share the same processes and dense ``t`` grid (guaranteed
    when every model forwards the same prepared data); mismatched shapes raise.
    """
    if not per_model:
        raise ValueError("aggregate_dense_exports: empty model list")
    processes = list(per_model[0])
    mean_out: dict[str, DenseProcessExport] = {}
    std_out: dict[str, DenseProcessExport] = {}
    for proc in processes:
        exports = [m[proc] for m in per_model]

        def _mean_std(attr: str) -> tuple[np.ndarray, np.ndarray]:
            stacked = np.stack([getattr(e, attr) for e in exports], axis=0)
            return stacked.mean(axis=0), stacked.std(axis=0)

        c_m, c_s = _mean_std("c_species")
        v_m, v_s = _mean_std("v_real")
        b_m, b_s = _mean_std("b_modeled_cum")
        q_m, q_s = _mean_std("q_rates")
        aux_keys = list(exports[0].auxiliary or {})
        aux_m = {
            k: np.stack([e.auxiliary[k] for e in exports]).mean(0) for k in aux_keys
        } or None
        aux_s = {
            k: np.stack([e.auxiliary[k] for e in exports]).std(0) for k in aux_keys
        } or None
        t = exports[0].t
        mean_out[proc] = DenseProcessExport(t, c_m, v_m, b_m, q_m, aux_m)
        std_out[proc] = DenseProcessExport(t, c_s, v_s, b_s, q_s, aux_s)
    return mean_out, std_out


def dense_exports_from_save_outputs(
    prediction_t: jnp.ndarray,
    prediction_save_outputs: SaveOutputs,
    trained_wrapper: HybridOdeWrapper,
    process_names: Sequence[str],
) -> dict[str, DenseProcessExport]:
    """Slice one batched forward solve's prediction block into per-process exports.

    Pure reshaping (no ODE solve): the leading axis of every leaf is the process
    batch (aligned with ``process_names``); un-scale ``SCL_states`` to physical
    and pick the canonical export columns (``c_*``, ``V_real``,
    cumulative modeled feeds, ``q_*`` rates, and per-key ``auxiliary``).
    """
    module = trained_wrapper.reaction_module
    # "species" export columns = the [RMCs | PVs] leading state block.
    n_species = len(trained_wrapper.modeled_RMC_names) + len(
        trained_wrapper.modeled_PV_names
    )
    n_modeled = len(trained_wrapper.modeled_FVC_names)
    # Un-scale [N, n_pred, state] → physical in one vmapped pass, then to numpy.
    RAW_states = np.asarray(
        jax.vmap(jax.vmap(module.unscale_state))(prediction_save_outputs.SCL_states)
    )
    t_np = np.asarray(prediction_t)
    v_real_np = np.asarray(prediction_save_outputs.RAW_V_export)
    q_np = np.asarray(prediction_save_outputs.RAW_modeled_BiologicalOde_rates)
    auxiliary = prediction_save_outputs.auxiliary

    exports: dict[str, DenseProcessExport] = {}
    for i, name in enumerate(process_names):
        # The export grid splices the measurement grid into the prediction
        # linspace so every measurement time is an exact node (for jump-correct
        # scoring). Sort by time and drop duplicate times to collapse the padded
        # measurement repeats at t1 and the shared t0/t1 endpoints into one node.
        t_i = t_np[i]
        order = np.argsort(t_i, kind="stable")
        t_sorted = t_i[order]
        keep = np.concatenate(([True], np.diff(t_sorted) > 0))
        sel = order[keep]
        aux_i = (
            None
            if auxiliary is None
            else {key: np.asarray(values[i])[sel] for key, values in auxiliary.items()}
        )
        exports[name] = DenseProcessExport(
            t=t_i[sel],
            c_species=RAW_states[i, sel, :n_species],
            v_real=v_real_np[i][sel],
            b_modeled_cum=RAW_states[i, sel, n_species + 1 : n_species + 1 + n_modeled],
            q_rates=q_np[i][sel],
            auxiliary=aux_i,
        )
    return exports


def _predictions_csv_header(
    modeled_RMC_names: tuple[str, ...],
    modeled_PV_names: tuple[str, ...],
    modeled_FVC_names: tuple[str, ...],
    rate_names: tuple[str, ...],
    auxiliary_columns: Sequence[str] = (),
) -> list[str]:
    """Build stable predictions.csv column order.

    The leading state block is ``[modeled_RMCs | modeled_PVs]``. Rate columns
    are derived from ``rate_names`` (i.e. ``rhs_ode.name_modeled_rates``); these
    already carry the ``q_``/``r_`` prefix used in bp-format, so they are written
    verbatim.
    """
    return (
        ["process", "t"]
        + [f"c_{name}" for name in modeled_RMC_names]
        + [f"c_{name}" for name in modeled_PV_names]
        + ["V_real"]
        + [f"B_{name}_cum" for name in modeled_FVC_names]
        + list(rate_names)
        + list(auxiliary_columns)
    )


def _auxiliary_csv_columns(auxiliary: dict[str, np.ndarray] | None) -> list[str]:
    """Build stable CSV columns for dense stacked auxiliary outputs."""
    if auxiliary is None:
        return []

    columns: list[str] = []
    for key in sorted(auxiliary):
        values = np.asarray(auxiliary[key])
        if values.ndim == 1:
            columns.append(f"aux_{key}")
        elif values.ndim == 2:
            columns.extend(f"aux_{key}_{i}" for i in range(values.shape[1]))
        else:
            raise ValueError(
                "dense auxiliary exports must stack to rank 1 or 2 arrays, "
                f"got key {key!r} with shape {values.shape}"
            )
    return columns


def _auxiliary_row_values(
    auxiliary: dict[str, np.ndarray] | None,
    row_index: int,
) -> list[Any]:
    """Flatten one dense auxiliary row in stable key order."""
    if auxiliary is None:
        return []

    row_values: list[Any] = []
    for key in sorted(auxiliary):
        values = np.asarray(auxiliary[key])
        if values.ndim == 1:
            row_values.append(values[row_index].item())
        elif values.ndim == 2:
            row_values.extend(np.asarray(values[row_index]).tolist())
        else:
            raise ValueError(
                "dense auxiliary exports must stack to rank 1 or 2 arrays, "
                f"got key {key!r} with shape {values.shape}"
            )
    return row_values


def save_model_metadata(path: str | Path, meta: dict[str, Any]) -> None:
    """Write a small JSON sidecar next to a saved model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("model metadata saved to %s", path)


def load_model_metadata(path: str | Path) -> dict[str, Any]:
    """Read a model metadata sidecar; returns {} if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return {}
    return load_json(path)


def _mse_and_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Compute MSE and R² for two 1D arrays of equal length."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_pred - y_true) ** 2))
    ss_res = float(np.sum((y_pred - y_true) ** 2))
    var = float(np.var(y_true))
    if var <= 0.0:
        # Constant target: R² is undefined; report 1.0 if predictions also
        # constant-equal, NaN otherwise.
        r2 = float("nan") if ss_res > 0 else 1.0
    else:
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - ss_res / ss_tot
    return mse, r2


def _annotate_fit(
    ax,
    r2: float,
    *,
    loss_label: str | None = None,
    loss_value: float | None = None,
) -> None:
    """Annotate an axis with R² and, when available, the named loss term.

    The named loss term is the value actually optimized (SCL space, user's
    reduction). When no term maps to this panel (``loss_value is None``), only
    R² is shown — never raise on a missing name.
    """
    if loss_value is not None:
        text = f"{loss_label}={loss_value:.4g}\nR²={r2:.4f}"
    else:
        text = f"R²={r2:.4f}"
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
    )


def plot_loss_curve(
    losses: Sequence[float],
    output_path: str | Path,
    *,
    title: str = "Training loss",
    per_target_loss_by_step: Sequence[tuple[float, ...]] | None = None,
    target_names: Sequence[str] | None = None,
    monitor_loss_by_step: dict[int, float] | None = None,
    monitor_per_target_by_step: dict[int, tuple[float, ...]] | None = None,
    monitor_label: str | None = None,
) -> None:
    """Draw loss-vs-step curves on a log-y axis and save as PNG.

    Layout: one subplot per loss. The first panel ("total") shows the
    ``losses`` series and, if provided, the total ``monitor_loss_by_step``
    series overlaid as a dashed line. If ``per_target_loss_by_step`` is
    provided, one additional panel is added per target — each also overlays the
    matching ``monitor_per_target_by_step`` term (dashed) so the holdout loss is
    broken down per target, not just in total.
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_steps = list(range(1, len(losses) + 1))
    monitor_steps = sorted(monitor_loss_by_step.keys()) if monitor_loss_by_step else []
    monitor_values = (
        [monitor_loss_by_step[s] for s in monitor_steps] if monitor_loss_by_step else []
    )
    mpt_steps = (
        sorted(monitor_per_target_by_step.keys()) if monitor_per_target_by_step else []
    )

    panels: list[tuple[str, list[float]]] = [("total", list(losses))]
    if per_target_loss_by_step:
        n_targets = max(len(row) for row in per_target_loss_by_step)
        names = (
            list(target_names) if target_names else [f"t{i}" for i in range(n_targets)]
        )
        names = (names + [f"t{i}" for i in range(n_targets)])[:n_targets]
        for i, name in enumerate(names):
            series = [
                row[i] if i < len(row) else float("nan")
                for row in per_target_loss_by_step
            ]
            panels.append((name, series))

    n_panels = len(panels)
    ncols = min(n_panels, 2)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(5.0, 4.0 * ncols), 3.0 * nrows),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    axes_flat = list(axes.flat)

    total_series = panels[0][1] if panels and panels[0][0] == "total" else None

    holdout_label = monitor_label or "validation"
    for panel_idx, (ax, (panel_label, series)) in enumerate(zip(axes_flat, panels)):
        # On per-target panels, plot the total loss as a faint reference line
        # so the per-target curve can be read against the overall trajectory
        # despite the per-panel y-scale.
        if (
            panel_label != "total"
            and total_series is not None
            and len(total_series) > 0
        ):
            ax.plot(
                train_steps,
                total_series,
                color="grey",
                linewidth=0.8,
                alpha=0.5,
                label="total",
            )
        if series:
            ax.plot(train_steps, series, color="C0", linewidth=1.2, label="train")
        # Holdout (monitor) overlay: the total on the "total" panel, the matching
        # per-target term on each target panel (panels[1:] map to targets 0..n-1).
        if panel_label == "total" and monitor_values:
            ax.plot(
                monitor_steps,
                monitor_values,
                linestyle="--",
                color="C3",
                label=holdout_label,
            )
        elif panel_label != "total" and mpt_steps:
            target_idx = panel_idx - 1
            steps_i = [
                s for s in mpt_steps if target_idx < len(monitor_per_target_by_step[s])
            ]
            values_i = [monitor_per_target_by_step[s][target_idx] for s in steps_i]
            if values_i:
                ax.plot(
                    steps_i,
                    values_i,
                    linestyle="--",
                    color="C3",
                    label=holdout_label,
                )
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="best", fontsize="small")
        ax.set_title(panel_label, fontsize="small")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    for ax in axes_flat[:n_panels]:
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("Step")
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("Mean loss")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("loss curve saved to %s", output_path)


def plot_grad_norm_curve(
    grad_norms: Sequence[float],
    output_path: str | Path,
    *,
    title: str = "Gradient norm",
) -> None:
    """Draw global L2 gradient-norm vs step on a log-y axis and save as PNG."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    steps = list(range(1, len(grad_norms) + 1))
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    if grad_norms:
        ax.plot(steps, list(grad_norms), color="C2", linewidth=1.2, label="grad norm")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("||grad||₂")
    ax.grid(True, alpha=0.3)
    if grad_norms:
        ax.legend(loc="best", fontsize="small")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("grad-norm curve saved to %s", output_path)


def plot_cross_fold_loss_curves(
    fold_curves: Sequence[
        tuple[str, Sequence[float], Sequence[float], Sequence[float], Sequence[float]]
    ],
    output_path: str | Path,
    *,
    title: str = "Cross-fold loss",
    monitor_label: str = "holdout",
) -> None:
    """Overlay every fold's train and monitor loss on shared log-y axes.

    Each entry in `fold_curves` is
    `(label, train_steps, train_loss, monitor_steps, monitor_loss)`. Train
    curves are drawn solid and monitor (holdout) curves dashed in the same
    per-fold colour; folds with no usable history are skipped.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # tab20's discrete colours stay distinct up to 20 folds; beyond that, sample
    # a continuous map so colours never silently wrap onto each other.
    n = len(fold_curves)
    if n <= 20:
        tab20 = plt.get_cmap("tab20")
        colors = [tab20(i) for i in range(n)]
    else:
        viridis = plt.get_cmap("viridis")
        colors = [viridis(i / (n - 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    drawn = False
    for i, (label, tr_steps, tr_loss, mon_steps, mon_loss) in enumerate(fold_curves):
        color = colors[i]
        if len(tr_loss):
            ax.plot(tr_steps, tr_loss, color=color, linewidth=1.0, label=label)
            drawn = True
        if len(mon_loss):
            ax.plot(mon_steps, mon_loss, color=color, linewidth=1.0, linestyle="--")
            drawn = True

    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    if drawn:
        fold_legend = ax.legend(loc="upper right", fontsize="small", title="fold")
        ax.add_artist(fold_legend)
        style_handles = [
            Line2D([0], [0], color="black", linestyle="-", label="train"),
            Line2D([0], [0], color="black", linestyle="--", label=monitor_label),
        ]
        ax.legend(handles=style_handles, loc="lower left", fontsize="small")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("cross-fold loss curves saved to %s", output_path)


def plot_training_results(
    result: Any,
    plot_sources: dict[str, ProcessPlotSource],
    store: TrainingDataStore,
    output_dir: str | Path,
    dense_exports: dict[str, DenseProcessExport],
    process_names: tuple[str, ...] | None = None,
    *,
    per_process_named_losses: dict[str, dict[str, float]] | None = None,
    per_process_total_loss: dict[str, float] | None = None,
    timeseries_csv_path: str | Path | None = None,
) -> None:
    """Generate loss curve and per-process concentration / rate / volume plots.

    Per-process trajectories come from precomputed ``dense_exports`` (no solve).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_curve(
        result.mean_loss_by_step,
        output_dir / "loss_curve.png",
        per_target_loss_by_step=getattr(result, "per_target_loss_by_step", None)
        or None,
        target_names=getattr(result, "target_names", None) or None,
        monitor_loss_by_step=(getattr(result, "holdout_loss_by_step", None) or None),
        monitor_label=getattr(result, "holdout_label", None),
    )

    grad_norm_by_step = getattr(result, "grad_norm_by_step", None) or None
    if grad_norm_by_step:
        plot_grad_norm_curve(
            grad_norm_by_step,
            output_dir / "grad_norm_curve.png",
        )

    plot_process_simulations(
        result.trained_wrapper,
        plot_sources,
        store,
        output_dir,
        dense_exports,
        process_names=process_names,
        per_process_named_losses=per_process_named_losses,
        per_process_total_loss=per_process_total_loss,
        timeseries_csv_path=timeseries_csv_path,
    )


def export_predictions_csv(
    trained_wrapper: HybridOdeWrapper,
    dense_exports: dict[str, DenseProcessExport],
    output_path: str | Path,
    process_names: tuple[str, ...] | None = None,
) -> None:
    """Write dense predictions.csv from precomputed per-process exports (no solve).

    ``dense_exports`` comes from :func:`bp_train.harness.compute_dense_exports`
    (the single batched solve). This function only formats and writes rows.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    modeled_RMC_names = trained_wrapper.modeled_RMC_names
    modeled_PV_names = trained_wrapper.modeled_PV_names
    modeled_FVC_names = trained_wrapper.modeled_FVC_names
    rate_names = tuple(trained_wrapper.rhs_ode.name_modeled_rates)
    # Leading state block written as ``c_*`` columns = modeled RMCs then PVs.
    n_species = len(modeled_RMC_names) + len(modeled_PV_names)
    n_modeled = len(modeled_FVC_names)
    n_rates = len(rate_names)

    if process_names is None:
        selected_processes = tuple(dense_exports.keys())
    else:
        missing = [name for name in process_names if name not in dense_exports]
        if missing:
            raise ValueError(
                f"export_predictions_csv missing dense exports for {missing}"
            )
        selected_processes = tuple(process_names)

    if not selected_processes:
        header = _predictions_csv_header(
            modeled_RMC_names=modeled_RMC_names,
            modeled_PV_names=modeled_PV_names,
            modeled_FVC_names=modeled_FVC_names,
            rate_names=rate_names,
        )
        pd.DataFrame(columns=header).to_csv(output_path, index=False)
        logger.info("timeseries csv saved to %s", output_path)
        return

    auxiliary_columns = _auxiliary_csv_columns(
        dense_exports[selected_processes[0]].auxiliary
    )
    header = _predictions_csv_header(
        modeled_RMC_names=modeled_RMC_names,
        modeled_PV_names=modeled_PV_names,
        modeled_FVC_names=modeled_FVC_names,
        rate_names=rate_names,
        auxiliary_columns=auxiliary_columns,
    )
    pd.DataFrame(columns=header).to_csv(output_path, index=False)

    for process_name in selected_processes:
        dense_export = dense_exports[process_name]
        aux_failed = dense_export.auxiliary is None
        if (
            not aux_failed
            and _auxiliary_csv_columns(dense_export.auxiliary) != auxiliary_columns
        ):
            raise ValueError(
                "predictions.csv auxiliary columns differ across processes; "
                f"expected {auxiliary_columns}, got "
                f"{_auxiliary_csv_columns(dense_export.auxiliary)} "
                f"for process {process_name!r}"
            )
        ts_rows: list[list[float | str]] = []
        for i_t in range(len(dense_export.t)):
            aux_cells = (
                [float("nan")] * len(auxiliary_columns)
                if aux_failed
                else _auxiliary_row_values(dense_export.auxiliary, i_t)
            )
            row = (
                [process_name, float(dense_export.t[i_t])]
                + [float(dense_export.c_species[i_t, j]) for j in range(n_species)]
                + [float(dense_export.v_real[i_t])]
                + [float(dense_export.b_modeled_cum[i_t, k]) for k in range(n_modeled)]
                + [float(dense_export.q_rates[i_t, j]) for j in range(n_rates)]
                + aux_cells
            )
            ts_rows.append(row)
        pd.DataFrame(ts_rows, columns=header).to_csv(
            output_path, mode="a", header=False, index=False
        )

    logger.info("timeseries csv saved to %s", output_path)


@dataclass(frozen=True)
class ProcessPlotSource:
    """Raw, collection-independent inputs needed to materialize one plot."""

    time_unit: str
    t_start: float
    t_end: float
    v_unit: str
    initial_volume: float
    measured_series: tuple[tuple[str, str, np.ndarray, np.ndarray], ...]
    volume_changes: tuple[tuple[str, str, bool, str, np.ndarray, np.ndarray], ...]


@dataclass(frozen=True)
class ProcessPlotData:
    """Picklable per-process plotting inputs — plain numpy + str, no JAX/bp_format.

    Built in the main process by :func:`build_process_plot_data` and consumed by
    :func:`render_process_figures` (the single per-process renderer used by the
    run-root/forward path AND the ``spawn`` background checkpoint worker).
    """

    process_name: str
    is_train: bool | None
    time_unit: str
    t_start: float
    t_end: float
    v_unit: str
    modeled_RMC_names: tuple[str, ...]
    modeled_PV_names: tuple[str, ...]
    modeled_FVC_names: tuple[str, ...]
    rate_names: tuple[str, ...]
    fvc_units: tuple[str, ...]
    t_dense: np.ndarray
    c_dense: np.ndarray
    q_dense: np.ndarray
    v_real_pred: np.ndarray
    b_modeled_pred: np.ndarray
    c_std: np.ndarray | None
    q_std: np.ndarray | None
    v_std: np.ndarray | None
    v_real_true_dense: np.ndarray
    b_modeled_true_dense: np.ndarray
    measured_series: tuple[tuple[str, str, np.ndarray, np.ndarray], ...]
    volume_changes: tuple[tuple[str, str, bool, np.ndarray, np.ndarray], ...]
    named_losses: dict[str, float] | None
    total_loss: float | None


def _resolve_selected_processes(
    store: TrainingDataStore, process_names: tuple[str, ...] | None
) -> tuple[str, ...]:
    if process_names is None:
        return tuple(store.process_order)
    missing = [name for name in process_names if name not in store.process_order]
    if missing:
        raise ValueError(
            f"unknown process names: {missing}; available={store.process_order}"
        )
    return tuple(process_names)


def extract_process_plot_sources(
    collection: BioProcessCollection,
    rhs_ode: Any,
    process_names: tuple[str, ...],
) -> dict[str, ProcessPlotSource]:
    """Copy the raw plot inputs needed after the collection is released."""
    modeled_RMC_names = tuple(rhs_ode.name_modeled_RMCs)
    modeled_PV_names = tuple(rhs_ode.name_modeled_PVs)
    sources: dict[str, ProcessPlotSource] = {}
    for process_name in process_names:
        process = collection.processes[process_name]
        measured_series = tuple(
            (
                name,
                process.reactor_medium.components[name].unit,
                np.array(
                    process.reactor_medium.components[name].concentration.times,
                    dtype=float,
                    copy=True,
                ),
                np.array(
                    process.reactor_medium.components[name].concentration.values,
                    dtype=float,
                    copy=True,
                ),
            )
            for name in modeled_RMC_names
        ) + tuple(
            (
                name,
                process.process_variables[name].unit,
                np.array(
                    process.process_variables[name].values.times,
                    dtype=float,
                    copy=True,
                ),
                np.array(
                    process.process_variables[name].values.values,
                    dtype=float,
                    copy=True,
                ),
            )
            for name in modeled_PV_names
        )
        volume_changes = tuple(
            (
                name,
                "feed" if isinstance(change, FeedVolumeChange) else "sample",
                bool(change.is_continuous),
                change.unit,
                np.array(change.values.times, dtype=float, copy=True),
                np.array(change.values.values, dtype=float, copy=True),
            )
            for name, change in process.volume.volume_changes.items()
        )
        sources[process_name] = ProcessPlotSource(
            time_unit=process.time_axis.unit,
            t_start=float(process.time_axis.start),
            t_end=float(process.time_axis.end),
            v_unit=process.volume.unit,
            initial_volume=float(process.volume.initial_volume),
            measured_series=measured_series,
            volume_changes=volume_changes,
        )
    return sources


def build_process_plot_data(
    trained_wrapper: HybridOdeWrapper,
    plot_sources: dict[str, ProcessPlotSource],
    store: TrainingDataStore,
    dense_exports: dict[str, DenseProcessExport],
    process_names: tuple[str, ...] | None = None,
    *,
    std_exports: dict[str, DenseProcessExport] | None = None,
    training_process_names: tuple[str, ...] | None = None,
    per_process_named_losses: dict[str, dict[str, float]] | None = None,
    per_process_total_loss: dict[str, float] | None = None,
) -> list[ProcessPlotData]:
    """Materialize picklable plots from runtime exports and raw plot sources."""
    modeled_RMC_names = tuple(trained_wrapper.modeled_RMC_names)
    modeled_PV_names = tuple(trained_wrapper.modeled_PV_names)
    modeled_FVC_names = tuple(trained_wrapper.modeled_FVC_names)
    rate_names = tuple(trained_wrapper.rhs_ode.name_modeled_rates)
    n_modeled = len(modeled_FVC_names)
    selected = _resolve_selected_processes(store, process_names)
    training_set = (
        set(training_process_names) if training_process_names is not None else None
    )

    out: list[ProcessPlotData] = []
    for process_name in selected:
        source = plot_sources[process_name]
        dense_export = dense_exports[process_name]
        std_export = std_exports.get(process_name) if std_exports else None
        t_dense = np.asarray(dense_export.t, dtype=float)

        # Ground-truth V_real(t): V0 + signed cumulative volume changes.
        # Continuous feeds add their cumulative inflow; discrete events (bolus
        # adds, sample removals) step at each event time by the signed delta
        # (bolus > 0, sample < 0) — the same arithmetic the callbacks solve uses.
        v_real_true_dense = np.full(t_dense.shape, source.initial_volume, dtype=float)
        for _name, kind, is_continuous, _unit, vc_t, vc_v in source.volume_changes:
            if kind == "feed" and is_continuous:
                v_real_true_dense += np.interp(
                    t_dense, vc_t, vc_v, left=float(vc_v[0]), right=float(vc_v[-1])
                )
            else:
                cumulative = np.cumsum(vc_v, dtype=float)
                idx = np.searchsorted(vc_t, t_dense, side="right") - 1
                contribution = np.zeros_like(t_dense, dtype=float)
                valid = idx >= 0
                contribution[valid] = cumulative[idx[valid]]
                v_real_true_dense += contribution

        # Cumulative measured B_modeled per modeled flow on the dense grid.
        b_modeled_true_dense = np.zeros((len(t_dense), n_modeled), dtype=float)
        changes_by_name = {
            name: (unit, times, values)
            for name, _kind, _continuous, unit, times, values in source.volume_changes
        }
        for k, fn in enumerate(modeled_FVC_names):
            _unit, vc_t, vc_v = changes_by_name[fn]
            b_modeled_true_dense[:, k] = np.interp(
                t_dense, vc_t, vc_v, left=float(vc_v[0]), right=float(vc_v[-1])
            )

        volume_changes = tuple(
            (name, kind, is_continuous, times, values)
            for name, kind, is_continuous, _unit, times, values in source.volume_changes
        )
        fvc_units = tuple(changes_by_name[name][0] for name in modeled_FVC_names)

        out.append(
            ProcessPlotData(
                process_name=process_name,
                is_train=(
                    (process_name in training_set) if training_set is not None else None
                ),
                time_unit=source.time_unit,
                t_start=source.t_start,
                t_end=source.t_end,
                v_unit=source.v_unit,
                modeled_RMC_names=modeled_RMC_names,
                modeled_PV_names=modeled_PV_names,
                modeled_FVC_names=modeled_FVC_names,
                rate_names=rate_names,
                fvc_units=fvc_units,
                t_dense=t_dense,
                c_dense=np.asarray(dense_export.c_species, dtype=float),
                q_dense=np.asarray(dense_export.q_rates, dtype=float),
                v_real_pred=np.asarray(dense_export.v_real, dtype=float),
                b_modeled_pred=np.asarray(dense_export.b_modeled_cum, dtype=float),
                c_std=(
                    np.asarray(std_export.c_species, dtype=float)
                    if std_export is not None
                    else None
                ),
                q_std=(
                    np.asarray(std_export.q_rates, dtype=float)
                    if std_export is not None
                    else None
                ),
                v_std=(
                    np.asarray(std_export.v_real, dtype=float)
                    if std_export is not None
                    else None
                ),
                v_real_true_dense=v_real_true_dense,
                b_modeled_true_dense=b_modeled_true_dense,
                measured_series=source.measured_series,
                volume_changes=volume_changes,
                named_losses=(per_process_named_losses or {}).get(process_name),
                total_loss=(per_process_total_loss or {}).get(process_name),
            )
        )
    return out


@dataclass(frozen=True)
class ControlDiagnostic:
    """One control's prepare-time diagnostic (picklable, plain numpy)."""

    name: str
    unit: str
    raw_times: np.ndarray
    raw_values: np.ndarray
    curve_t: np.ndarray
    curve_values: np.ndarray
    grid_t: np.ndarray
    is_spline: bool
    max_rel_dev: float


@dataclass(frozen=True)
class ProcessControlDiagnostics:
    """Per-process control diagnostics — fed to :func:`render_control_diagnostics`."""

    process_name: str
    time_unit: str
    controls: tuple[ControlDiagnostic, ...]


def render_control_diagnostics(
    diagnostics: ProcessControlDiagnostics,
    output_dir: str | Path,
) -> None:
    """Render one figure overlaying, per control: the raw measured samples, the
    stored control curve the solver uses (fitted spline or linear interpolation),
    and the dense-grid knot density. Pure numpy/matplotlib, picklable — safe in the
    ``spawn`` background plot worker. Writes ``<process>_controls.png``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    controls = diagnostics.controls
    if not controls:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(controls)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.1 * n), squeeze=False)
    for ax, c in zip(axes[:, 0], controls):
        if c.raw_times.size:
            ax.plot(
                c.raw_times,
                c.raw_values,
                ".",
                ms=3,
                color="0.6",
                label=f"raw ({c.raw_times.size})",
            )
        ax.plot(
            c.curve_t,
            c.curve_values,
            "-",
            lw=1.4,
            color="C0",
            label="spline" if c.is_spline else "linear",
        )
        y0 = ax.get_ylim()[0]
        ax.plot(
            c.grid_t,
            np.full(c.grid_t.size, y0),
            "|",
            color="C3",
            ms=6,
            alpha=0.4,
            label=f"dense-grid knots ({c.grid_t.size})",
        )
        ax.set_ylabel(f"{c.name}\n[{c.unit}]", fontsize=8)
        ax.legend(
            loc="best",
            fontsize=7,
            title=f"grid={c.grid_t.size}  maxΔ={c.max_rel_dev:.1e}",
            title_fontsize=7,
        )
    axes[-1, 0].set_xlabel(f"time [{diagnostics.time_unit}]")
    fig.suptitle(f"control splines vs data — {diagnostics.process_name}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{diagnostics.process_name}_controls.png", dpi=110)
    plt.close(fig)


def render_process_figures(
    plot_data: Sequence[ProcessPlotData],
    output_dir: str | Path,
    *,
    filename_suffix: str = "",
) -> None:
    """The single per-process figure renderer — pure numpy/matplotlib, picklable.

    Draws, per :class:`ProcessPlotData`: species rows (measured scatter + dense
    integration + optional ±1σ, R²/loss annotation) beside rate panels, a volume
    row (true vs integrated V_real + raw volume_changes), and cumulative
    modeled-feed rows; writes ``<process>{filename_suffix}.png``. No JAX/bp_format
    — safe to run in the ``spawn`` background plot worker.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdat in plot_data:
        n_species = len(pdat.measured_series)
        n_modeled = len(pdat.modeled_FVC_names)
        n_rates = len(pdat.rate_names)
        t_dense = pdat.t_dense
        c_dense = pdat.c_dense
        q_dense = pdat.q_dense
        c_std, q_std, v_std = pdat.c_std, pdat.q_std, pdat.v_std
        t_start, t_end, time_unit = pdat.t_start, pdat.t_end, pdat.time_unit
        named_losses = pdat.named_losses

        n_rows = n_species + 1 + n_modeled
        fig, axes = plt.subplots(n_rows, 2, squeeze=False, figsize=(10, 3 * n_rows))

        for i, (sp_name, sp_unit, t_meas, v_meas) in enumerate(pdat.measured_series):
            ax_c = axes[i, 0]
            ax_c.scatter(
                t_meas, v_meas, s=16, zorder=5, color="black", label="measured"
            )
            ax_c.plot(
                t_dense, c_dense[:, i], "-", lw=1.5, color="C0", label="integrated"
            )
            if c_std is not None:
                ax_c.fill_between(
                    t_dense,
                    c_dense[:, i] - c_std[:, i],
                    c_dense[:, i] + c_std[:, i],
                    color="C0",
                    alpha=0.2,
                    lw=0,
                    label="±1σ",
                )
            v_pred_at_meas = np.interp(t_meas, t_dense, c_dense[:, i])
            _mse, r2 = _mse_and_r2(v_meas, v_pred_at_meas)
            _sp_loss = named_losses.get(sp_name) if named_losses else None
            _annotate_fit(ax_c, r2, loss_label=sp_name, loss_value=_sp_loss)
            ax_c.set_title(f"{sp_name} [{sp_unit}]")
            ax_c.set_xlabel(f"time [{time_unit}]")
            ax_c.set_xlim(t_start, t_end)
            ax_c.legend(fontsize="small")
            ax_c.grid(True, alpha=0.3)

            ax_q = axes[i, 1]
            if i < n_rates:
                ax_q.plot(t_dense, q_dense[:, i], "-", lw=1.5, color="black")
                if q_std is not None:
                    ax_q.fill_between(
                        t_dense,
                        q_dense[:, i] - q_std[:, i],
                        q_dense[:, i] + q_std[:, i],
                        color="black",
                        alpha=0.15,
                        lw=0,
                    )
                ax_q.axhline(0, color="gray", lw=0.5, ls="--")
                ax_q.set_title(pdat.rate_names[i])
                ax_q.set_xlabel(f"time [{time_unit}]")
                ax_q.set_xlim(t_start, t_end)
                ax_q.grid(True, alpha=0.3)
            else:
                ax_q.set_visible(False)

        # ---- Volume row: true vs integrated V_real + raw volume_changes ----
        ax_v = axes[n_species, 0]
        ax_v.plot(
            t_dense,
            pdat.v_real_true_dense,
            "-",
            lw=1.5,
            color="black",
            label="measured",
        )
        ax_v.plot(
            t_dense, pdat.v_real_pred, "--", lw=1.5, color="C0", label="integrated"
        )
        if v_std is not None:
            ax_v.fill_between(
                t_dense,
                pdat.v_real_pred - v_std,
                pdat.v_real_pred + v_std,
                color="C0",
                alpha=0.2,
                lw=0,
            )
        _v_mse, v_r2 = _mse_and_r2(pdat.v_real_true_dense, pdat.v_real_pred)
        _annotate_fit(ax_v, v_r2)
        ax_v.set_title(f"V_real [{pdat.v_unit}]")
        ax_v.set_xlabel(f"time [{time_unit}]")
        ax_v.set_xlim(t_start, t_end)
        ax_v.legend(fontsize="small")
        ax_v.grid(True, alpha=0.3)

        ax_vc = axes[n_species, 1]
        bar_width = (t_end - t_start) * BAR_WIDTH_FRACTION
        cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        extra_handles: list[Any] = []
        for idx, (vc_name, kind, is_continuous, vc_t, vc_v) in enumerate(
            pdat.volume_changes
        ):
            label = f"{vc_name} ({kind})"
            color = cycle_colors[idx % len(cycle_colors)]
            if is_continuous:
                ax_vc.plot(vc_t, vc_v, "-", lw=1.2, label=label, color=color)
            elif vc_t.size == 0:
                extra_handles.append(Patch(facecolor=color, edgecolor="k", label=label))
            else:
                ax_vc.bar(
                    vc_t, vc_v, width=bar_width, label=label, edgecolor="k", color=color
                )
        ax_vc.set_title(f"volume_changes [{pdat.v_unit}]")
        ax_vc.set_xlabel(f"time [{time_unit}]")
        ax_vc.set_xlim(t_start, t_end)
        ax_vc.grid(True, alpha=0.3)
        if pdat.volume_changes:
            handles, _ = ax_vc.get_legend_handles_labels()
            ax_vc.legend(handles=handles + extra_handles, fontsize="small")

        # ---- Cumulative modeled-feed rows ----
        for k, fn in enumerate(pdat.modeled_FVC_names):
            row = n_species + 1 + k
            ax_b = axes[row, 0]
            ax_b.plot(
                t_dense,
                pdat.b_modeled_true_dense[:, k],
                "-",
                lw=1.5,
                color="black",
                label="measured",
            )
            ax_b.plot(
                t_dense,
                pdat.b_modeled_pred[:, k],
                "-",
                lw=1.5,
                color="C0",
                label="integrated",
            )
            _b_mse, b_r2 = _mse_and_r2(
                pdat.b_modeled_true_dense[:, k], pdat.b_modeled_pred[:, k]
            )
            _b_label = f"B_{fn}_cum"
            _b_loss = named_losses.get(_b_label) if named_losses else None
            _annotate_fit(ax_b, b_r2, loss_label=_b_label, loss_value=_b_loss)
            ax_b.set_title(f"cumulative {fn} [{pdat.fvc_units[k]}]")
            ax_b.set_xlabel(f"time [{time_unit}]")
            ax_b.set_xlim(t_start, t_end)
            ax_b.legend(fontsize="small")
            ax_b.grid(True, alpha=0.3)
            axes[row, 1].set_visible(False)

        split_tag = (
            ""
            if pdat.is_train is None
            else (" [train]" if pdat.is_train else " [holdout]")
        )
        suptitle = f"{pdat.process_name}{split_tag}"
        if pdat.total_loss is not None:
            suptitle += f" — total loss {pdat.total_loss:.4g}"
            shown = (
                set(pdat.modeled_RMC_names)
                | set(pdat.modeled_PV_names)
                | {f"B_{fn}_cum" for fn in pdat.modeled_FVC_names}
            )
            extras = [
                f"{name}={value:.3g}"
                for name, value in (named_losses or {}).items()
                if name not in shown
            ]
            if extras:
                suptitle += "\n" + "  ".join(extras)
        fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{pdat.process_name}{filename_suffix}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    logger.info("plots saved to %s", output_dir)


def plot_process_simulations(
    trained_wrapper: HybridOdeWrapper,
    plot_sources: dict[str, ProcessPlotSource],
    store: TrainingDataStore,
    output_dir: str | Path,
    dense_exports: dict[str, DenseProcessExport],
    process_names: tuple[str, ...] | None = None,
    *,
    std_exports: dict[str, DenseProcessExport] | None = None,
    training_process_names: tuple[str, ...] | None = None,
    per_process_named_losses: dict[str, dict[str, float]] | None = None,
    per_process_total_loss: dict[str, float] | None = None,
    timeseries_csv_path: str | Path | None = None,
    filename_suffix: str = "",
) -> None:
    """Write predictions.csv and/or render per-process plots from precomputed
    dense exports — no ODE solve here.

    Thin orchestration over the single per-process renderer: the merged CSV goes
    to :func:`export_predictions_csv` (single writer), and the figures are built
    by :func:`build_process_plot_data` (the JAX/bp_format extraction) and drawn by
    :func:`render_process_figures` (the one renderer, shared with the background
    checkpoint worker).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_processes = _resolve_selected_processes(store, process_names)

    # CSV: delegate to the single writer (no solve, no per-process row logic here).
    if timeseries_csv_path is not None:
        export_predictions_csv(
            trained_wrapper, dense_exports, timeseries_csv_path, selected_processes
        )
    if not selected_processes:
        return

    plot_data = build_process_plot_data(
        trained_wrapper,
        plot_sources,
        store,
        dense_exports,
        selected_processes,
        std_exports=std_exports,
        training_process_names=training_process_names,
        per_process_named_losses=per_process_named_losses,
        per_process_total_loss=per_process_total_loss,
    )
    render_process_figures(plot_data, output_dir, filename_suffix=filename_suffix)
