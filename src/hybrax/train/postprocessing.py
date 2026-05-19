"""Post-training outputs: model serialization and result plots."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import diffrax
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd

from bp_format.dataclasses import BioProcessCollection, FeedVolumeChange
from bp_format.mechanistic import build_rhs_ode

from .training_data import TrainingDataStore
from .wrapper import HybridOdeWrapper, SaveOutputs

logger = logging.getLogger(__name__)

# Bolus bar width in `plot_process_simulations`, expressed as a fraction of the
# process time span.
BAR_WIDTH_FRACTION = 0.02


def _wrapper_vector_field(
    t: Any, y: jnp.ndarray, wrapper: HybridOdeWrapper
) -> jnp.ndarray:
    """Stable Diffrax vector field; wrapper params stay dynamic via ``args``."""
    return wrapper(t, y)


def _wrapper_save_outputs(
    t: Any, y: jnp.ndarray, wrapper: HybridOdeWrapper
) -> SaveOutputs:
    """Stable Diffrax save function; wrapper params stay dynamic via ``args``."""
    return wrapper.save_outputs(t, y)


@dataclass(frozen=True)
class DenseProcessExport:
    """Dense, human-facing per-process export arrays in physical units.

    ``q_rates`` is aligned with ``rhs_ode.name_modeled_rates`` (the modelled
    rate vector), not with ``species_names``. Under a user-defined
    ``BiologicalOde`` these orderings differ.
    """

    t: np.ndarray
    c_species: np.ndarray
    v_cont: np.ndarray
    v_real: np.ndarray
    b_modeled_cum: np.ndarray
    q_rates: np.ndarray
    auxiliary: dict[str, np.ndarray] | None = None


def _predictions_csv_header(
    species_names: tuple[str, ...],
    modeled_flow_names: tuple[str, ...],
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
        + [f"c_{name}" for name in species_names]
        + ["V_cont", "V_real"]
        + [f"B_{name}_cum" for name in modeled_flow_names]
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


def _compute_dense_process_export(
    trained_wrapper: HybridOdeWrapper,
    collection: BioProcessCollection,
    store: TrainingDataStore,
    process_name: str,
    *,
    solver_max_steps: int,
    solver_rtol: float,
    solver_atol: float,
    solver_use_jump_ts: bool,
    n_dense: int = 200,
) -> DenseProcessExport:
    """Solve one process on dense grid and return export-ready arrays."""
    process = collection.processes[process_name]
    process_data = store.get_process(process_name)
    rhs_ode = build_rhs_ode(process)

    process_wrapper = eqx.tree_at(
        lambda w: (w.controls, w.rhs_ode.Cin_controlled_FVCs, w.rhs_ode.Cin_modeled_FVCs),
        trained_wrapper,
        (process_data.controls, rhs_ode.Cin_controlled_FVCs, rhs_ode.Cin_modeled_FVCs),
    )

    t_dense = jnp.linspace(
        float(process.time_axis.start),
        float(process.time_axis.end),
        n_dense,
    )
    y0_scaled = process_wrapper.scale_state(process_data.y0_measured)
    term = diffrax.ODETerm(_wrapper_vector_field)
    jump_ts = process_data.controls.active_step_ts if solver_use_jump_ts else None
    sol = diffrax.diffeqsolve(
        term,
        diffrax.Tsit5(),
        t0=t_dense[0],
        t1=t_dense[-1],
        dt0=None,
        y0=y0_scaled,
        args=process_wrapper,
        saveat=diffrax.SaveAt(ts=t_dense, fn=_wrapper_save_outputs),
        stepsize_controller=diffrax.PIDController(
            rtol=solver_rtol,
            atol=solver_atol,
            jump_ts=jump_ts,
        ),
        max_steps=solver_max_steps,
        throw=False,
    )
    n_species = len(process_wrapper.species_names)
    n_modeled = len(process_wrapper.modeled_flow_names)
    n_rates = len(process_wrapper.rhs_ode.name_modeled_rates)

    if sol.result != diffrax.RESULTS.successful:
        # Training can survive a transient stiff forward pass, so don't kill the
        # whole run at checkpoint time. Emit NaN rows; auxiliary keys are unknown
        # on a failed solve so propagate None.
        logger.warning(
            "dense export solve failed for process %r (result=%s); "
            "writing NaN rows to predictions.csv",
            process_name,
            sol.result,
        )
        return DenseProcessExport(
            t=np.asarray(t_dense),
            c_species=np.full((n_dense, n_species), np.nan),
            v_cont=np.full(n_dense, np.nan),
            v_real=np.full(n_dense, np.nan),
            b_modeled_cum=np.full((n_dense, n_modeled), np.nan),
            q_rates=np.full((n_dense, n_rates), np.nan),
            auxiliary=None,
        )

    states_physical = np.asarray(sol.ys.states_physical)
    return DenseProcessExport(
        t=np.asarray(t_dense),
        c_species=states_physical[:, :n_species],
        v_cont=states_physical[:, n_species],
        v_real=np.asarray(sol.ys.v_real_export),
        b_modeled_cum=states_physical[:, n_species + 1 : n_species + 1 + n_modeled],
        q_rates=np.asarray(sol.ys.specific_rates_physical),
        auxiliary=(
            None
            if sol.ys.auxiliary is None
            else {key: np.asarray(values) for key, values in sol.ys.auxiliary.items()}
        ),
    )


def _write_predictions_csv(
    trained_wrapper: HybridOdeWrapper,
    collection: BioProcessCollection,
    store: TrainingDataStore,
    output_path: str | Path,
    process_names: tuple[str, ...] | None = None,
    *,
    solver_max_steps: int,
    solver_rtol: float,
    solver_atol: float,
    solver_use_jump_ts: bool,
) -> None:
    """Write dense predictions.csv without any plotting-only work."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    species_names = trained_wrapper.species_names
    modeled_flow_names = trained_wrapper.modeled_flow_names
    rate_names = tuple(trained_wrapper.rhs_ode.name_modeled_rates)
    n_species = len(species_names)
    n_modeled = len(modeled_flow_names)
    n_rates = len(rate_names)
    if process_names is None:
        selected_processes = tuple(store.process_order)
    else:
        missing = [name for name in process_names if name not in store.process_order]
        if missing:
            raise ValueError(
                "export_predictions_csv received unknown process names: "
                f"{missing}; available={store.process_order}"
            )
        selected_processes = tuple(process_names)

    if not selected_processes:
        header = _predictions_csv_header(
            species_names=species_names,
            modeled_flow_names=modeled_flow_names,
            rate_names=rate_names,
        )
        pd.DataFrame(columns=header).to_csv(output_path, index=False)
        logger.info("timeseries csv saved to %s", output_path)
        return

    first_process = selected_processes[0]
    first_export = _compute_dense_process_export(
        trained_wrapper,
        collection,
        store,
        first_process,
        solver_max_steps=solver_max_steps,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
        solver_use_jump_ts=solver_use_jump_ts,
    )
    auxiliary_columns = _auxiliary_csv_columns(first_export.auxiliary)
    header = _predictions_csv_header(
        species_names=species_names,
        modeled_flow_names=modeled_flow_names,
        rate_names=rate_names,
        auxiliary_columns=auxiliary_columns,
    )
    pd.DataFrame(columns=header).to_csv(output_path, index=False)

    def _append_process_rows(
        process_name: str,
        dense_export: DenseProcessExport,
    ) -> None:
        # On a failed solve `auxiliary` is None — that's not a column-set drift,
        # just absence of data, so emit NaN values for the previously-seen aux
        # columns instead of raising.
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
                + [float(dense_export.v_cont[i_t]), float(dense_export.v_real[i_t])]
                + [float(dense_export.b_modeled_cum[i_t, k]) for k in range(n_modeled)]
                + [float(dense_export.q_rates[i_t, j]) for j in range(n_rates)]
                + aux_cells
            )
            ts_rows.append(row)
        pd.DataFrame(ts_rows, columns=header).to_csv(
            output_path,
            mode="a",
            header=False,
            index=False,
        )

    _append_process_rows(first_process, first_export)
    for process_name in selected_processes[1:]:
        dense_export = _compute_dense_process_export(
            trained_wrapper,
            collection,
            store,
            process_name,
            solver_max_steps=solver_max_steps,
            solver_rtol=solver_rtol,
            solver_atol=solver_atol,
            solver_use_jump_ts=solver_use_jump_ts,
        )
        _append_process_rows(process_name, dense_export)

    logger.info("timeseries csv saved to %s", output_path)


def save_model(wrapper: HybridOdeWrapper, path: str | Path) -> None:
    """Serialize a trained wrapper to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, wrapper)
    logger.info("trained model saved to %s", path)


def load_trained_wrapper(
    path: str | Path, *, template: HybridOdeWrapper
) -> HybridOdeWrapper:
    """Deserialize a trained wrapper from disk using ``template`` as pytree shape."""
    return eqx.tree_deserialise_leaves(Path(path), like=template)


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


def _annotate_fit(ax, mse: float, r2: float) -> None:
    """Add a text box with MSE and R² in the upper-left corner of an axis."""
    text = f"MSE={mse:.4g}\nR²={r2:.4f}"
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
    process_names: tuple[str, ...] | None = None,
    *,
    solver_max_steps: int = 4096,
    solver_rtol: float = 1e-3,
    solver_atol: float = 1e-5,
    solver_use_jump_ts: bool = True,
    timeseries_csv_path: str | Path | None = None,
) -> None:
    """Generate loss curve and per-process concentration / rate / volume plots."""
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
        process_names=process_names,
        solver_max_steps=solver_max_steps,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
        solver_use_jump_ts=solver_use_jump_ts,
        timeseries_csv_path=timeseries_csv_path,
    )


def export_predictions_csv(
    trained_wrapper: HybridOdeWrapper,
    collection: BioProcessCollection,
    store: TrainingDataStore,
    output_path: str | Path,
    process_names: tuple[str, ...] | None = None,
    *,
    solver_max_steps: int = 4096,
    solver_rtol: float = 1e-3,
    solver_atol: float = 1e-5,
    solver_use_jump_ts: bool = True,
) -> None:
    """Write dense predictions.csv without rendering per-process plots."""
    _write_predictions_csv(
        trained_wrapper,
        collection,
        store,
        output_path=output_path,
        process_names=process_names,
        solver_max_steps=solver_max_steps,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
        solver_use_jump_ts=solver_use_jump_ts,
    )


def plot_process_simulations(
    trained_wrapper: HybridOdeWrapper,
    collection: BioProcessCollection,
    store: TrainingDataStore,
    output_dir: str | Path,
    process_names: tuple[str, ...] | None = None,
    *,
    solver_max_steps: int = 4096,
    solver_rtol: float = 1e-3,
    solver_atol: float = 1e-5,
    solver_use_jump_ts: bool = True,
    training_process_names: tuple[str, ...] | None = None,
    timeseries_csv_path: str | Path | None = None,
    filename_suffix: str = "",
    render_plots: bool = True,
) -> None:
    """Simulate each selected process on a dense grid and render result plots.

    Optionally appends all dense trajectories into a single merged CSV at
    ``timeseries_csv_path`` with a leading ``process`` column.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    species_names = trained_wrapper.species_names
    modeled_flow_names = trained_wrapper.modeled_flow_names
    n_species = len(species_names)
    n_modeled = len(modeled_flow_names)
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

    # Prepare merged timeseries rows (one file, all processes).
    ts_header: list[str] | None = None
    ts_auxiliary_columns: list[str] | None = None
    ts_path: Path | None = None
    if timeseries_csv_path is not None:
        ts_path = Path(timeseries_csv_path)
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        if not selected_processes:
            ts_header = _predictions_csv_header(
                species_names=species_names,
                modeled_flow_names=modeled_flow_names,
                rate_names=rate_names,
            )
            pd.DataFrame(columns=ts_header).to_csv(ts_path, index=False)

    # --- Per-process simulation/exports ---
    for process_name in selected_processes:
        process = collection.processes[process_name]
        process_data = store.get_process(process_name)
        time_unit = process.time_axis.unit

        dense_export = _compute_dense_process_export(
            trained_wrapper,
            collection,
            store,
            process_name,
            solver_max_steps=solver_max_steps,
            solver_rtol=solver_rtol,
            solver_atol=solver_atol,
            solver_use_jump_ts=solver_use_jump_ts,
        )
        t_start = float(process.time_axis.start)
        t_end = float(process.time_axis.end)
        t_dense_np = dense_export.t
        c_dense = dense_export.c_species
        v_cont_pred = dense_export.v_cont
        v_real_pred = dense_export.v_real
        b_modeled_pred = dense_export.b_modeled_cum
        q_dense = dense_export.q_rates
        auxiliary_dense = dense_export.auxiliary

        if ts_path is not None and ts_header is None:
            ts_auxiliary_columns = _auxiliary_csv_columns(auxiliary_dense)
            ts_header = _predictions_csv_header(
                species_names=species_names,
                modeled_flow_names=modeled_flow_names,
                rate_names=rate_names,
                auxiliary_columns=ts_auxiliary_columns,
            )
            pd.DataFrame(columns=ts_header).to_csv(ts_path, index=False)
        elif (
            ts_path is not None
            and ts_auxiliary_columns is not None
            and auxiliary_dense is not None
            and _auxiliary_csv_columns(auxiliary_dense) != ts_auxiliary_columns
        ):
            raise ValueError(
                "timeseries auxiliary columns differ across processes; "
                f"expected {ts_auxiliary_columns}, got "
                f"{_auxiliary_csv_columns(auxiliary_dense)} "
                f"for process {process_name!r}"
            )

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
            for k, fn in enumerate(modeled_flow_names):
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

            for i, sp_name in enumerate(species_names):
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
                # Interpolate dense prediction at measurement times for fit metrics.
                v_pred_at_meas = np.interp(t_measured, t_dense_np, c_dense[:, i])
                mse, r2 = _mse_and_r2(v_meas, v_pred_at_meas)
                _annotate_fit(ax_c, mse, r2)
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
            v_mse, v_r2 = _mse_and_r2(v_real_true_dense, v_real_pred)
            _annotate_fit(ax_v, v_mse, v_r2)
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
            for k, fn in enumerate(modeled_flow_names):
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
                b_mse, b_r2 = _mse_and_r2(
                    b_modeled_true_dense[:, k], b_modeled_pred[:, k]
                )
                _annotate_fit(ax_b, b_mse, b_r2)
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
            fig.suptitle(f"{process_name}{split_tag}", fontsize=12)
            fig.tight_layout()
            fig.savefig(
                output_dir / f"{process_name}{filename_suffix}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

        if ts_header is not None:
            assert ts_path is not None
            ts_rows: list[list[float | str]] = []
            aux_failed = auxiliary_dense is None
            for i_t in range(len(t_dense_np)):
                aux_cells = (
                    [float("nan")] * len(ts_auxiliary_columns or ())
                    if aux_failed
                    else _auxiliary_row_values(auxiliary_dense, i_t)
                )
                row = (
                    [process_name, float(t_dense_np[i_t])]
                    + [float(c_dense[i_t, j]) for j in range(n_species)]
                    + [float(v_cont_pred[i_t]), float(v_real_pred[i_t])]
                    + [float(b_modeled_pred[i_t, k]) for k in range(n_modeled)]
                    + [float(q_dense[i_t, j]) for j in range(n_rates)]
                    + aux_cells
                )
                ts_rows.append(row)
            pd.DataFrame(ts_rows, columns=ts_header).to_csv(
                ts_path,
                mode="a",
                header=False,
                index=False,
            )

    if ts_header is not None and timeseries_csv_path is not None:
        logger.info("timeseries csv saved to %s", timeseries_csv_path)

    if render_plots:
        logger.info("plots saved to %s", output_dir)
