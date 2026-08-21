"""Post-training model serialization, prediction exports, and loss curves."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from bp_format.json_io import load_json

from .serialization import write_json
from .wrapper import HybridOdeWrapper, SaveOutputs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DenseProcessExport:
    """Dense, human-facing per-process export arrays in physical units.

    ``q_rates`` is aligned with ``rhs_ode.name_modeled_rates`` (the modeled
    biological rate vector), not with ``modeled_RMC_names``. Under a
    user-defined ``BiologicalOde`` these orderings differ. Modeled Inflow and
    Outflow rates remain separate physical arrays with their storage signs.
    """

    t: np.ndarray
    c_species: np.ndarray
    v_real: np.ndarray
    b_modeled_cum: np.ndarray
    q_rates: np.ndarray
    modeled_Inflow_rates: np.ndarray
    modeled_Outflow_rates: np.ndarray
    auxiliary: dict[str, np.ndarray] | None = None


def aggregate_dense_exports(
    per_model: list[dict[str, DenseProcessExport]],
) -> tuple[dict[str, DenseProcessExport], dict[str, DenseProcessExport]]:
    """Stack per-model dense exports into per-process (mean, std) exports.

    All models must share the same processes and dense ``t`` grid. Mismatched
    time grids are rejected before any values are combined.
    """
    if not per_model:
        raise ValueError("aggregate_dense_exports: empty model list")
    processes = list(per_model[0])
    mean_out: dict[str, DenseProcessExport] = {}
    std_out: dict[str, DenseProcessExport] = {}
    for proc in processes:
        exports = [m[proc] for m in per_model]
        t = exports[0].t
        if any(not np.array_equal(export.t, t) for export in exports[1:]):
            raise ValueError(
                f"ensemble members have different time grids for process {proc!r}"
            )

        def _mean_std(attr: str) -> tuple[np.ndarray, np.ndarray]:
            stacked = np.stack([getattr(e, attr) for e in exports], axis=0)
            return stacked.mean(axis=0), stacked.std(axis=0)

        c_m, c_s = _mean_std("c_species")
        v_m, v_s = _mean_std("v_real")
        b_m, b_s = _mean_std("b_modeled_cum")
        q_m, q_s = _mean_std("q_rates")
        inflow_m, inflow_s = _mean_std("modeled_Inflow_rates")
        outflow_m, outflow_s = _mean_std("modeled_Outflow_rates")
        aux_keys = list(exports[0].auxiliary or {})
        aux_m = {
            k: np.stack([e.auxiliary[k] for e in exports]).mean(0) for k in aux_keys
        } or None
        aux_s = {
            k: np.stack([e.auxiliary[k] for e in exports]).std(0) for k in aux_keys
        } or None
        mean_out[proc] = DenseProcessExport(
            t=t,
            c_species=c_m,
            v_real=v_m,
            b_modeled_cum=b_m,
            q_rates=q_m,
            modeled_Inflow_rates=inflow_m,
            modeled_Outflow_rates=outflow_m,
            auxiliary=aux_m,
        )
        std_out[proc] = DenseProcessExport(
            t=t,
            c_species=c_s,
            v_real=v_s,
            b_modeled_cum=b_s,
            q_rates=q_s,
            modeled_Inflow_rates=inflow_s,
            modeled_Outflow_rates=outflow_s,
            auxiliary=aux_s,
        )
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
    and pick the canonical export columns (``c_*``, ``V_real``, cumulative
    modeled Inflows and Outflows, biological ``q_*`` rates, separate modeled
    Inflow/Outflow rates, and per-key ``auxiliary``).
    """
    module = trained_wrapper.reaction_module
    # "species" export columns = the [RMCs | PVs] leading state block.
    n_species = len(trained_wrapper.modeled_RMC_names) + len(
        trained_wrapper.modeled_PV_names
    )
    n_modeled = len(trained_wrapper.modeled_Inflow_names) + len(
        trained_wrapper.modeled_Outflow_names
    )
    # Un-scale [N, n_pred, state] → physical in one vmapped pass, then to numpy.
    RAW_states = np.asarray(
        jax.vmap(jax.vmap(module.unscale_state))(prediction_save_outputs.SCL_states)
    )
    t_np = np.asarray(prediction_t)
    v_real_np = np.asarray(prediction_save_outputs.RAW_V_export)
    q_np = np.asarray(prediction_save_outputs.RAW_modeled_BiologicalOde_rates)
    inflow_rates_np = np.asarray(prediction_save_outputs.RAW_modeled_Inflows_rates)
    outflow_rates_np = np.asarray(prediction_save_outputs.RAW_modeled_Outflows_rates)
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
            modeled_Inflow_rates=inflow_rates_np[i][sel],
            modeled_Outflow_rates=outflow_rates_np[i][sel],
            auxiliary=aux_i,
        )
    return exports


def _predictions_csv_header(
    modeled_RMC_names: tuple[str, ...],
    modeled_PV_names: tuple[str, ...],
    modeled_Inflow_names: tuple[str, ...],
    modeled_Outflow_names: tuple[str, ...],
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
        + [f"B_{name}_cum" for name in modeled_Inflow_names]
        + [f"B_{name}_cum" for name in modeled_Outflow_names]
        + list(rate_names)
        + [f"B_{name}_rate" for name in modeled_Inflow_names]
        + [f"B_{name}_rate" for name in modeled_Outflow_names]
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
    write_json(path, meta, sort_keys=True)
    logger.info("model metadata saved to %s", path)


def load_model_metadata(path: str | Path) -> dict[str, Any]:
    """Read a model metadata sidecar; returns {} if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return {}
    return load_json(path)


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
    import matplotlib

    matplotlib.use("Agg")
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
    """Draw global L2 gradient norm against training step and save as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    steps = range(1, len(grad_norms) + 1)
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    if grad_norms:
        ax.plot(steps, grad_norms, color="C2", linewidth=1.2, label="grad norm")
        ax.legend(loc="best", fontsize="small")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("||grad||₂")
    ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("gradient norm curve saved to %s", output_path)


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
    import matplotlib

    matplotlib.use("Agg")
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
    modeled_Inflow_names = trained_wrapper.modeled_Inflow_names
    modeled_Outflow_names = trained_wrapper.modeled_Outflow_names
    rate_names = tuple(trained_wrapper.rhs_ode.name_modeled_rates)
    # Leading state block written as ``c_*`` columns = modeled RMCs then PVs.
    n_species = len(modeled_RMC_names) + len(modeled_PV_names)
    n_modeled = len(modeled_Inflow_names) + len(modeled_Outflow_names)
    n_rates = len(rate_names)
    n_modeled_Inflows = len(modeled_Inflow_names)
    n_modeled_Outflows = len(modeled_Outflow_names)

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
            modeled_Inflow_names=modeled_Inflow_names,
            modeled_Outflow_names=modeled_Outflow_names,
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
        modeled_Inflow_names=modeled_Inflow_names,
        modeled_Outflow_names=modeled_Outflow_names,
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
                + [
                    float(dense_export.modeled_Inflow_rates[i_t, j])
                    for j in range(n_modeled_Inflows)
                ]
                + [
                    float(dense_export.modeled_Outflow_rates[i_t, j])
                    for j in range(n_modeled_Outflows)
                ]
                + aux_cells
            )
            ts_rows.append(row)
        pd.DataFrame(ts_rows, columns=header).to_csv(
            output_path, mode="a", header=False, index=False
        )

    logger.info("timeseries csv saved to %s", output_path)


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
    and the dense-grid knot density. Writes ``<process>_controls.png``.
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
