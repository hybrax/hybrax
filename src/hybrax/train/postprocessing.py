"""Post-training outputs: model serialization and result plots."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bp_format.dataclasses import BioProcessCollection, FeedVolumeChange
from bp_format.mechanistic import get_rhs_ode

from .training_data import TrainingDataStore
from .wrapper import HybridOdeWrapper

logger = logging.getLogger(__name__)


def save_model(wrapper: HybridOdeWrapper, path: str | Path) -> None:
    """Serialize a trained wrapper to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, wrapper)
    logger.info("trained model saved to %s", path)


def load_trained_wrapper(
    path: str | Path, *, template: HybridOdeWrapper
) -> HybridOdeWrapper:
    """Deserialize a trained wrapper from disk using ``template`` as the pytree shape."""
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
) -> None:
    """Generate loss curve and per-process concentration / rate / volume plots."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Loss curve ---
    fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
    ax_loss.plot(range(1, len(result.mean_loss_by_step) + 1), result.mean_loss_by_step)
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Mean loss (MSE)")
    ax_loss.set_title("Training loss")
    ax_loss.set_yscale("log")
    ax_loss.grid(True, alpha=0.3)
    fig_loss.tight_layout()
    fig_loss.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close(fig_loss)
    logger.info("loss curve saved to %s", output_dir / "loss_curve.png")

    plot_process_simulations(
        result.trained_wrapper,
        collection,
        store,
        output_dir,
        process_names=process_names,
        solver_max_steps=solver_max_steps,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
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
    training_process_names: tuple[str, ...] | None = None,
    timeseries_csv_path: str | Path | None = None,
    filename_suffix: str = "",
) -> None:
    """Simulate each selected process on a dense grid and render result plots.

    Optionally appends all dense trajectories into a single merged CSV at
    ``timeseries_csv_path`` with a leading ``process`` column.
    """
    import diffrax
    import matplotlib.pyplot as plt

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

    per_process_rhs = {
        name: get_rhs_ode(collection.processes[name]) for name in selected_processes
    }

    training_set = (
        set(training_process_names) if training_process_names is not None else None
    )

    # Prepare merged timeseries CSV writer (one file, all processes).
    ts_file = None
    ts_writer = None
    ts_header: list[str] | None = None
    if timeseries_csv_path is not None:
        ts_path = Path(timeseries_csv_path)
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        ts_file = ts_path.open("w", newline="", encoding="utf-8")
        ts_writer = csv.writer(ts_file)
        ts_header = (
            ["process", "t"]
            + [f"c_{name}" for name in species_names]
            + ["V_cont", "V_real"]
            + [f"B_{name}_cum" for name in modeled_flow_names]
            + [f"q_{name}" for name in species_names]
        )
        ts_writer.writerow(ts_header)

    # --- Per-process plots ---
    for process_name in selected_processes:
        process = collection.processes[process_name]
        process_data = store.get_process(process_name)
        rhs = per_process_rhs[process_name]
        time_unit = process.time_axis.unit

        # Inject per-process controls and Cin
        process_wrapper = eqx.tree_at(
            lambda w: (w.controls, w.rhs_ode.Cin, w.rhs_ode.Cin_modeled),
            trained_wrapper,
            (process_data.controls, rhs.Cin, rhs.Cin_modeled),
        )

        # Simulate on a dense time grid
        t_start = float(process.time_axis.start)
        t_end = float(process.time_axis.end)
        t_dense = jnp.linspace(t_start, t_end, 200)

        y0_scaled = process_wrapper.scale_state(process_data.y0)
        term = diffrax.ODETerm(lambda t, y, args: process_wrapper(t, y))
        sol = diffrax.diffeqsolve(
            term,
            diffrax.Tsit5(),
            t0=t_dense[0],
            t1=t_dense[-1],
            dt0=None,
            y0=y0_scaled,
            saveat=diffrax.SaveAt(ts=t_dense),
            stepsize_controller=diffrax.PIDController(
                rtol=solver_rtol,
                atol=solver_atol,
                jump_ts=process_data.controls.active_step_ts,
            ),
            max_steps=solver_max_steps,
            throw=False,
        )
        states_physical = jax.vmap(process_wrapper.unscale_state)(sol.ys)
        # State layout: [c_species..., V_cont, B_modeled_cum_0, ...]
        c_dense = np.asarray(states_physical[:, :n_species])
        v_cont_pred = np.asarray(states_physical[:, n_species])
        b_modeled_pred = np.asarray(
            states_physical[:, n_species + 1 : n_species + 1 + n_modeled]
        )
        t_dense_np = np.asarray(t_dense)

        # ---- Dense ground-truth time series for plotting ----
        # V_real_true(t) on the dense grid: V0 + sum(cumulative inflows) - V_sample_acc
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

        v_sample_acc_dense = np.array(
            [
                float(
                    process_wrapper.controls.eval(jnp.asarray(float(t_)))[
                        process_wrapper.sample_acc_control_index
                    ]
                )
                for t_ in t_dense_np
            ]
        )
        v_real_pred = v_cont_pred - v_sample_acc_dense
        v_real_true_dense = v_cont_true_dense - v_sample_acc_dense

        # Cumulative measured B_modeled per modeled flow on the dense grid
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

        # ---- Specific rates q(t) along the trajectory ----
        q_dense = []
        for i_t in range(len(t_dense_np)):
            t_val = jnp.asarray(t_dense_np[i_t])
            y_scaled = sol.ys[i_t]
            c_scaled = jnp.clip(y_scaled[:n_species], 0.0)
            controls_vec = process_wrapper.controls.eval(t_val)
            cin_flat = jnp.concatenate(
                [
                    process_wrapper.rhs_ode.Cin.reshape(-1),
                    process_wrapper.rhs_ode.Cin_modeled.reshape(-1),
                ]
            )
            U_aug = jnp.concatenate([controls_vec, cin_flat])
            if process_wrapper.include_v_real_feature:
                v_real_unclipped = states_physical[i_t, n_species] - jnp.asarray(
                    v_sample_acc_dense[i_t], dtype=U_aug.dtype
                )
                v_real_val = jnp.maximum(
                    v_real_unclipped,
                    jnp.asarray(process_wrapper.min_real_volume, dtype=U_aug.dtype),
                )
                U_aug = jnp.concatenate(
                    [U_aug, jnp.asarray([v_real_val], dtype=U_aug.dtype)]
                )
            u_scaled = U_aug / process_wrapper.controls_scale
            outputs = process_wrapper.reaction_module(t_val, c_scaled, u_scaled)
            Q = np.asarray(outputs.specific_rates) * np.asarray(process_wrapper.q_scale)
            q_dense.append(Q)
        q_dense = np.stack(q_dense, axis=0)

        # --- Layout: species rows + volume row + 1 row per modeled feed ---
        n_rows = n_species + 1 + n_modeled
        fig, axes = plt.subplots(n_rows, 2, squeeze=False, figsize=(10, 3 * n_rows))

        for i, sp_name in enumerate(species_names):
            ax_c = axes[i, 0]
            comp = process.reactor_medium.components[sp_name]
            t_meas = np.asarray(comp.concentration.times, dtype=float)
            v_meas = np.asarray(comp.concentration.values, dtype=float)
            ax_c.scatter(
                t_meas, v_meas, s=16, zorder=5, color="black", label="measured"
            )
            ax_c.plot(
                t_dense_np,
                c_dense[:, i],
                "-",
                lw=1.5,
                color="C0",
                label="integrated",
            )
            # Fit metrics: interpolate dense prediction at measurement times.
            v_pred_at_meas = np.interp(t_meas, t_dense_np, c_dense[:, i])
            mse, r2 = _mse_and_r2(v_meas, v_pred_at_meas)
            _annotate_fit(ax_c, mse, r2)
            ax_c.set_title(f"{sp_name} [{comp.unit}]")
            ax_c.set_xlabel(f"time [{time_unit}]")
            ax_c.set_xlim(t_start, t_end)
            ax_c.legend(fontsize="small")
            ax_c.grid(True, alpha=0.3)

            ax_q = axes[i, 1]
            ax_q.plot(t_dense_np, q_dense[:, i], "-", lw=1.5, color="black")
            ax_q.axhline(0, color="gray", lw=0.5, ls="--")
            ax_q.set_title(f"q_{sp_name}")
            ax_q.set_xlabel(f"time [{time_unit}]")
            ax_q.set_xlim(t_start, t_end)
            ax_q.grid(True, alpha=0.3)

        # ---- Volume panel: dense true V_real curve + integrated curve ----
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
        ax_vc = axes[n_species, 1]
        bar_width = (t_end - t_start) * 0.02  # 2% of time range
        for vc_name, vc in process.volume.volume_changes.items():
            vc_t = np.asarray(vc.values.times, dtype=float)
            vc_v = np.asarray(vc.values.values, dtype=float)
            kind = "feed" if isinstance(vc, FeedVolumeChange) else "sample"
            if vc.is_continuous:
                ax_vc.plot(vc_t, vc_v, "-", lw=1.2, label=f"{vc_name} ({kind})")
            else:
                ax_vc.bar(vc_t, vc_v, width=bar_width, label=f"{vc_name} ({kind})", edgecolor="k")
        ax_vc.set_title(f"volume_changes [{process.volume.unit}]")
        ax_vc.set_xlabel(f"time [{time_unit}]")
        ax_vc.set_xlim(t_start, t_end)
        ax_vc.grid(True, alpha=0.3)
        if process.volume.volume_changes:
            ax_vc.legend(fontsize="small")

        # ---- Cumulative modeled feed panels (one row per modeled flow) ----
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
            b_mse, b_r2 = _mse_and_r2(b_modeled_true_dense[:, k], b_modeled_pred[:, k])
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
            split_tag = " [train]" if process_name in training_set else " [holdout]"
        fig.suptitle(f"{process_name}{split_tag}", fontsize=12)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{process_name}{filename_suffix}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        if ts_writer is not None:
            for i_t in range(len(t_dense_np)):
                row = (
                    [process_name, float(t_dense_np[i_t])]
                    + [float(c_dense[i_t, j]) for j in range(n_species)]
                    + [float(v_cont_pred[i_t]), float(v_real_pred[i_t])]
                    + [float(b_modeled_pred[i_t, k]) for k in range(n_modeled)]
                    + [float(q_dense[i_t, j]) for j in range(n_species)]
                )
                ts_writer.writerow(row)

    if ts_file is not None:
        ts_file.close()
        logger.info("timeseries csv saved to %s", timeseries_csv_path)

    logger.info("plots saved to %s", output_dir)
