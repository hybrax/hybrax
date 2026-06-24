"""Post-training outputs: model serialization and result plots."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from bp_format.dataclasses import BioProcessCollection, FeedVolumeChange

from .serialization import (  # re-exported: canonical home is serialization.py
    load_trained_wrapper,
    save_model,
)
from .training_data import TrainingDataStore
from .wrapper import HybridOdeWrapper, SaveOutputs

__all_serialization__ = ["save_model", "load_trained_wrapper"]

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
        aux_m = (
            {k: np.stack([e.auxiliary[k] for e in exports]).mean(0) for k in aux_keys}
            or None
        )
        aux_s = (
            {k: np.stack([e.auxiliary[k] for e in exports]).std(0) for k in aux_keys}
            or None
        )
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
    n_species = len(trained_wrapper.modeled_RMC_names)
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
        aux_i = (
            None
            if auxiliary is None
            else {key: np.asarray(values[i]) for key, values in auxiliary.items()}
        )
        exports[name] = DenseProcessExport(
            t=t_np[i],
            c_species=RAW_states[i, :, :n_species],
            v_real=v_real_np[i],
            b_modeled_cum=RAW_states[i, :, n_species + 1 : n_species + 1 + n_modeled],
            q_rates=q_np[i],
            auxiliary=aux_i,
        )
    return exports


def _predictions_csv_header(
    modeled_RMC_names: tuple[str, ...],
    modeled_FVC_names: tuple[str, ...],
    rate_names: tuple[str, ...],
    auxiliary_columns: Sequence[str] = (),
) -> list[str]:
    """Build stable predictions.csv column order.

    Rate columns are derived from ``rate_names`` (i.e.
    ``rhs_ode.name_modeled_rates``); these already carry the ``q_``/``r_``
    prefix used in bp-format, so they are written verbatim.
    """
    return (
        ["process", "t"]
        + [f"c_{name}" for name in modeled_RMC_names]
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
    return json.loads(path.read_text(encoding="utf-8"))


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
    monitor_label: str | None = None,
) -> None:
    """Draw loss-vs-step curves on a log-y axis and save as PNG.

    Layout: one subplot per loss. The first panel ("total") shows the
    ``losses`` series and, if provided, the ``monitor_loss_by_step``
    series overlaid as a dashed line. If ``per_target_loss_by_step`` is
    provided, one additional panel is added per target. All panels share
    the same y-axis (log scale) so curves are directly comparable.
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_steps = list(range(1, len(losses) + 1))
    monitor_steps = sorted(monitor_loss_by_step.keys()) if monitor_loss_by_step else []
    monitor_values = (
        [monitor_loss_by_step[s] for s in monitor_steps] if monitor_loss_by_step else []
    )

    panels: list[tuple[str, list[float]]] = [("total", list(losses))]
    if per_target_loss_by_step:
        n_targets = max(len(row) for row in per_target_loss_by_step)
        names = list(target_names) if target_names else [f"t{i}" for i in range(n_targets)]
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

    for ax, (panel_label, series) in zip(axes_flat, panels):
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
        if panel_label == "total" and monitor_values:
            ax.plot(
                monitor_steps,
                monitor_values,
                marker="o",
                linestyle="--",
                color="C3",
                label=monitor_label or "validation",
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


def plot_training_results(
    result: Any,
    collection: BioProcessCollection,
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
        per_target_loss_by_step=getattr(result, "per_target_loss_by_step", None) or None,
        target_names=getattr(result, "target_names", None) or None,
        monitor_loss_by_step=getattr(result, "monitor_loss_by_log_step", None) or None,
        monitor_label=getattr(result, "monitor_label", None),
    )

    grad_norm_by_step = getattr(result, "grad_norm_by_step", None) or None
    if grad_norm_by_step:
        plot_grad_norm_curve(
            grad_norm_by_step,
            output_dir / "grad_norm_curve.png",
        )

    plot_process_simulations(
        result.trained_wrapper,
        collection,
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
    modeled_FVC_names = trained_wrapper.modeled_FVC_names
    rate_names = tuple(trained_wrapper.rhs_ode.name_modeled_rates)
    n_species = len(modeled_RMC_names)
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


def _measured_timeseries(
    process: Any, variable: str, *, use_rmc: bool
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(times, values)`` for a measured target, or None if unavailable.

    Reads RMC targets from ``reactor_medium.components`` and PV targets from
    ``process_variables``; static (timeless) measurements are skipped.
    """
    from bp_format.dataclasses import TimeSeries

    if use_rmc:
        comp = process.reactor_medium.components.get(variable)
        src = comp.concentration if comp is not None else None
    else:
        pv = process.process_variables.get(variable)
        src = pv.values if pv is not None else None
    if src is None or not isinstance(src, TimeSeries):
        return None
    return np.asarray(src.times, dtype=float), np.asarray(src.values, dtype=float)


def measured_points_records(
    collection: BioProcessCollection,
    store: TrainingDataStore,
    process_names: tuple[str, ...] | None = None,
) -> list[tuple[str, str, float, float]]:
    """Measured target points as picklable ``(process, variable, t, value)`` rows.

    Extracted in the MAIN process (the collection load pulls bp_format/jax) and
    handed to the lightweight :func:`render_process_plots_from_csv` background
    worker as plain records — so there is no intermediate observations.csv file.
    """
    selected = tuple(process_names) if process_names else tuple(store.process_order)
    measured_names = tuple(store.name_measured)
    use_rmc = bool(store.name_measured_RMCs)
    rows: list[tuple[str, str, float, float]] = []
    for process_name in selected:
        process = collection.processes[process_name]
        for variable in measured_names:
            ts = _measured_timeseries(process, variable, use_rmc=use_rmc)
            if ts is None:
                continue
            times, values = ts
            for t, v in zip(times, values, strict=False):
                rows.append((process_name, variable, float(t), float(v)))
    return rows


def render_process_plots_from_csv(
    predictions_csv: str | Path,
    measured_records: list[tuple[str, str, float, float]] | None,
    output_dir: str | Path,
    *,
    process_names: tuple[str, ...] | None = None,
    target_names: tuple[str, ...] | None = None,
    training_process_names: tuple[str, ...] | None = None,
) -> None:
    """Render per-process prediction-vs-observation plots from a predictions CSV
    plus precomputed measured records **only**.

    Pure numpy/pandas/matplotlib — **NO jax / bp_train heavy imports**. Picklable,
    safe to call inside a ``spawn`` background worker. ``measured_records`` are
    ``(process, variable, t, value)`` rows from
    :func:`measured_points_records` (extracted in the main process). Writes
    ``<process>.png`` per process: predicted species trajectories (``c_<name>``)
    with measured overlays, plus a ``V_real`` panel when present.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    del target_names  # accepted for API symmetry; species inferred from columns

    predictions_csv = Path(predictions_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(predictions_csv)
    obs = pd.DataFrame(
        list(measured_records) if measured_records else [],
        columns=["process", "variable", "t", "value"],
    )

    species_cols = [c for c in pred.columns if c.startswith("c_")]
    procs = (
        tuple(process_names)
        if process_names is not None
        else tuple(dict.fromkeys(pred["process"].tolist()))
    )
    training_set = (
        set(training_process_names) if training_process_names is not None else None
    )

    for process_name in procs:
        pp = pred[pred["process"] == process_name]
        if pp.empty:
            continue
        po = obs[obs["process"] == process_name]
        panels = list(species_cols)
        if "V_real" in pred.columns:
            panels.append("V_real")
        if not panels:
            continue
        n_panels = len(panels)
        ncols = min(n_panels, 2)
        nrows = (n_panels + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, squeeze=False, figsize=(5.0 * ncols, 3.0 * nrows)
        )
        axes_flat = list(axes.flat)
        for ax, col in zip(axes_flat, panels):
            ax.plot(pp["t"], pp[col], "-", color="C0", lw=1.5, label="integrated")
            if col.startswith("c_"):
                variable = col[2:]
                measured = po[po["variable"] == variable]
                if not measured.empty:
                    ax.scatter(
                        measured["t"],
                        measured["value"],
                        s=16,
                        color="black",
                        zorder=5,
                        label="measured",
                    )
            ax.set_title(col, fontsize="small")
            ax.set_xlabel("time")
            ax.grid(True, alpha=0.3)
            if ax.get_legend_handles_labels()[1]:
                ax.legend(fontsize="small")
        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)
        split = ""
        if training_set is not None:
            split = " (train)" if process_name in training_set else " (holdout)"
        fig.suptitle(f"{process_name}{split}")
        fig.tight_layout()
        fig.savefig(output_dir / f"{process_name}.png", dpi=150)
        plt.close(fig)


def plot_process_simulations(
    trained_wrapper: HybridOdeWrapper,
    collection: BioProcessCollection,
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
    render_plots: bool = True,
) -> None:
    """Render per-process plots and/or write predictions.csv from precomputed
    dense exports — no ODE solve here.

    ``dense_exports`` and the per-process losses come from
    :func:`bp_train.harness.compute_dense_exports` and the forward loss table.
    The merged CSV is delegated to :func:`export_predictions_csv` (single writer).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modeled_RMC_names = trained_wrapper.modeled_RMC_names
    modeled_FVC_names = trained_wrapper.modeled_FVC_names
    n_species = len(modeled_RMC_names)
    n_modeled = len(modeled_FVC_names)
    if process_names is None:
        selected_processes = tuple(store.process_order)
    else:
        missing = [name for name in process_names if name not in store.process_order]
        if missing:
            raise ValueError(
                "plot_process_simulations received unknown process names: "
                f"{missing}; available={store.process_order}"
            )
        selected_processes = tuple(process_names)

    training_set = (
        set(training_process_names) if training_process_names is not None else None
    )

    rate_names = tuple(trained_wrapper.rhs_ode.name_modeled_rates)
    n_rates = len(rate_names)

    # CSV: delegate to the single writer (no solve, no per-process row logic here).
    if timeseries_csv_path is not None:
        export_predictions_csv(
            trained_wrapper, dense_exports, timeseries_csv_path, selected_processes
        )
    if not render_plots:
        return

    # --- Per-process rendering (from precomputed dense exports) ---
    for process_name in selected_processes:
        process = collection.processes[process_name]
        process_data = store.get_process(process_name)
        time_unit = process.time_axis.unit

        process_named_losses = (per_process_named_losses or {}).get(process_name)
        process_total_loss = (per_process_total_loss or {}).get(process_name)

        dense_export = dense_exports[process_name]
        t_start = float(process.time_axis.start)
        t_end = float(process.time_axis.end)
        t_dense_np = dense_export.t
        c_dense = dense_export.c_species
        v_real_pred = dense_export.v_real
        b_modeled_pred = dense_export.b_modeled_cum
        q_dense = dense_export.q_rates

        # Optional ensemble ±1σ bands (mean is dense_export; std is std_export).
        std_export = std_exports.get(process_name) if std_exports else None
        c_std = std_export.c_species if std_export is not None else None
        v_std = std_export.v_real if std_export is not None else None
        q_std = std_export.q_rates if std_export is not None else None

        if render_plots:
            import matplotlib.pyplot as plt

            # ---- Dense ground-truth time series for plotting ----
            # V_real_true(t) on the dense grid: V0 + cumulative inflows
            # - V_sample_acc.
            v0 = float(process.volume.initial_volume)
            v_cont_true_dense = np.full(t_dense_np.shape, v0, dtype=float)
            for vc in process.volume.volume_changes.values():
                if not isinstance(vc, FeedVolumeChange):
                    continue
                vc_t = np.asarray(vc.values.times, dtype=float)
                vc_v = np.asarray(vc.values.values, dtype=float)
                if bool(vc.is_continuous):
                    v_cont_true_dense += np.interp(
                        t_dense_np,
                        vc_t,
                        vc_v,
                        left=float(vc_v[0]),
                        right=float(vc_v[-1]),
                    )
                else:
                    cumulative = np.cumsum(vc_v, dtype=float)
                    idx = np.searchsorted(vc_t, t_dense_np, side="right") - 1
                    contribution = np.zeros_like(t_dense_np, dtype=float)
                    valid = idx >= 0
                    contribution[valid] = cumulative[idx[valid]]
                    v_cont_true_dense += contribution

            sample_acc_index = trained_wrapper.sample_acc_control_index
            v_sample_acc_dense = np.asarray(
                process_data.controls.eval(jnp.asarray(t_dense_np))[:, sample_acc_index]
            )
            v_real_true_dense = v_cont_true_dense - v_sample_acc_dense

            # Cumulative measured B_modeled per modeled flow on the dense grid.
            b_modeled_true_dense = np.zeros((len(t_dense_np), n_modeled), dtype=float)
            for k, fn in enumerate(modeled_FVC_names):
                vc = process.volume.volume_changes[fn]
                vc_t = np.asarray(vc.values.times, dtype=float)
                vc_v = np.asarray(vc.values.values, dtype=float)
                b_modeled_true_dense[:, k] = np.interp(
                    t_dense_np,
                    vc_t,
                    vc_v,
                    left=float(vc_v[0]),
                    right=float(vc_v[-1]),
                )

            # --- Layout: species rows + volume row + modeled-feed rows ---
            n_rows = n_species + 1 + n_modeled
            fig, axes = plt.subplots(n_rows, 2, squeeze=False, figsize=(10, 3 * n_rows))

            for i, sp_name in enumerate(modeled_RMC_names):
                ax_c = axes[i, 0]
                comp = process.reactor_medium.components[sp_name]
                t_measured = np.asarray(comp.concentration.times, dtype=float)
                v_meas = np.asarray(comp.concentration.values, dtype=float)
                ax_c.scatter(
                    t_measured, v_meas, s=16, zorder=5, color="black", label="measured"
                )
                ax_c.plot(
                    t_dense_np,
                    c_dense[:, i],
                    "-",
                    lw=1.5,
                    color="C0",
                    label="integrated",
                )
                if c_std is not None:
                    ax_c.fill_between(
                        t_dense_np,
                        c_dense[:, i] - c_std[:, i],
                        c_dense[:, i] + c_std[:, i],
                        color="C0",
                        alpha=0.2,
                        lw=0,
                        label="±1σ",
                    )
                # Interpolate dense prediction at measurement times for R².
                v_pred_at_meas = np.interp(t_measured, t_dense_np, c_dense[:, i])
                _mse, r2 = _mse_and_r2(v_meas, v_pred_at_meas)
                _sp_loss = (
                    process_named_losses.get(sp_name)
                    if process_named_losses
                    else None
                )
                _annotate_fit(ax_c, r2, loss_label=sp_name, loss_value=_sp_loss)
                ax_c.set_title(f"{sp_name} [{comp.unit}]")
                ax_c.set_xlabel(f"time [{time_unit}]")
                ax_c.set_xlim(t_start, t_end)
                ax_c.legend(fontsize="small")
                ax_c.grid(True, alpha=0.3)

                ax_q = axes[i, 1]
                # Rate panels are aligned with rhs_ode.name_modeled_rates, NOT
                # with species; under a user-defined BiologicalOde the orderings
                # differ. Title with the rate name so labels match values.
                if i < n_rates:
                    ax_q.plot(t_dense_np, q_dense[:, i], "-", lw=1.5, color="black")
                    if q_std is not None:
                        ax_q.fill_between(
                            t_dense_np,
                            q_dense[:, i] - q_std[:, i],
                            q_dense[:, i] + q_std[:, i],
                            color="black",
                            alpha=0.15,
                            lw=0,
                        )
                    ax_q.axhline(0, color="gray", lw=0.5, ls="--")
                    ax_q.set_title(rate_names[i])
                    ax_q.set_xlabel(f"time [{time_unit}]")
                    ax_q.set_xlim(t_start, t_end)
                    ax_q.grid(True, alpha=0.3)
                else:
                    ax_q.set_visible(False)

            # ---- Volume panel: dense true V_real + integrated curve ----
            ax_v = axes[n_species, 0]
            ax_v.plot(
                t_dense_np,
                v_real_true_dense,
                "-",
                lw=1.5,
                color="black",
                label="measured",
            )
            ax_v.plot(
                t_dense_np,
                v_real_pred,
                "--",
                lw=1.5,
                color="C0",
                label="integrated",
            )
            if v_std is not None:
                ax_v.fill_between(
                    t_dense_np,
                    v_real_pred - v_std,
                    v_real_pred + v_std,
                    color="C0",
                    alpha=0.2,
                    lw=0,
                )
            _v_mse, v_r2 = _mse_and_r2(v_real_true_dense, v_real_pred)
            # V_real is not a loss target → R²-only annotation.
            _annotate_fit(ax_v, v_r2)
            ax_v.set_title(f"V_real [{process.volume.unit}]")
            ax_v.set_xlabel(f"time [{time_unit}]")
            ax_v.set_xlim(t_start, t_end)
            ax_v.legend(fontsize="small")
            ax_v.grid(True, alpha=0.3)

            # ---- Right panel: raw volume_changes overlaid ----
            from matplotlib.patches import Patch

            ax_vc = axes[n_species, 1]
            bar_width = (t_end - t_start) * BAR_WIDTH_FRACTION
            cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            extra_handles: list[Any] = []
            for idx, (vc_name, vc) in enumerate(process.volume.volume_changes.items()):
                vc_t = np.asarray(vc.values.times, dtype=float)
                vc_v = np.asarray(vc.values.values, dtype=float)
                kind = "feed" if isinstance(vc, FeedVolumeChange) else "sample"
                label = f"{vc_name} ({kind})"
                color = cycle_colors[idx % len(cycle_colors)]
                if vc.is_continuous:
                    ax_vc.plot(vc_t, vc_v, "-", lw=1.2, label=label, color=color)
                elif vc_t.size == 0:
                    # Empty discrete series: proxy patch keeps legend swatch consistent
                    extra_handles.append(
                        Patch(facecolor=color, edgecolor="k", label=label)
                    )
                else:
                    ax_vc.bar(
                        vc_t,
                        vc_v,
                        width=bar_width,
                        label=label,
                        edgecolor="k",
                        color=color,
                    )
            ax_vc.set_title(f"volume_changes [{process.volume.unit}]")
            ax_vc.set_xlabel(f"time [{time_unit}]")
            ax_vc.set_xlim(t_start, t_end)
            ax_vc.grid(True, alpha=0.3)
            if process.volume.volume_changes:
                handles, _ = ax_vc.get_legend_handles_labels()
                ax_vc.legend(handles=handles + extra_handles, fontsize="small")

            # ---- Cumulative modeled feed panels ----
            for k, fn in enumerate(modeled_FVC_names):
                row = n_species + 1 + k
                ax_b = axes[row, 0]
                ax_b.plot(
                    t_dense_np,
                    b_modeled_true_dense[:, k],
                    "-",
                    lw=1.5,
                    color="black",
                    label="measured",
                )
                ax_b.plot(
                    t_dense_np,
                    b_modeled_pred[:, k],
                    "-",
                    lw=1.5,
                    color="C0",
                    label="integrated",
                )
                _b_mse, b_r2 = _mse_and_r2(
                    b_modeled_true_dense[:, k], b_modeled_pred[:, k]
                )
                _b_label = f"B_{fn}_cum"
                _b_loss = (
                    process_named_losses.get(_b_label)
                    if process_named_losses
                    else None
                )
                _annotate_fit(ax_b, b_r2, loss_label=_b_label, loss_value=_b_loss)
                unit = process.volume.volume_changes[fn].unit
                ax_b.set_title(f"cumulative {fn} [{unit}]")
                ax_b.set_xlabel(f"time [{time_unit}]")
                ax_b.set_xlim(t_start, t_end)
                ax_b.legend(fontsize="small")
                ax_b.grid(True, alpha=0.3)
                axes[row, 1].set_visible(False)

            if training_set is None:
                split_tag = ""
            else:
                is_train = process_name in training_set
                split_tag = " [train]" if is_train else " [holdout]"
            suptitle = f"{process_name}{split_tag}"
            if process_total_loss is not None:
                suptitle += f" — total loss {process_total_loss:.4g}"
                # Named terms with no per-subplot home (penalties, aux): list them.
                shown = set(modeled_RMC_names) | {
                    f"B_{fn}_cum" for fn in modeled_FVC_names
                }
                extras = [
                    f"{name}={value:.3g}"
                    for name, value in process_named_losses.items()
                    if name not in shown
                ]
                if extras:
                    suptitle += "\n" + "  ".join(extras)
            fig.suptitle(suptitle, fontsize=12)
            fig.tight_layout()
            fig.savefig(
                output_dir / f"{process_name}{filename_suffix}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

    logger.info("plots saved to %s", output_dir)
