"""Post-training outputs: model serialization and result plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bpbench.dataclasses import BioProcessCollection
from bpbench.mechanistic import get_rhs_ode

from .training_data import TrainingDataStore
from .wrapper import HybridOdeWrapper

logger = logging.getLogger(__name__)


def save_model(wrapper: HybridOdeWrapper, path: str | Path) -> None:
    """Serialize a trained wrapper to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, wrapper)
    logger.info("trained model saved to %s", path)


def plot_training_results(
    result: Any,
    collection: BioProcessCollection,
    store: TrainingDataStore,
    output_dir: str | Path,
    *,
    solver_max_steps: int = 4096,
    solver_rtol: float = 1e-3,
    solver_atol: float = 1e-5,
) -> None:
    """Generate loss curve and per-process concentration / rate / volume plots."""
    import diffrax
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trained_wrapper = result.trained_wrapper
    species_names = trained_wrapper.species_names
    n_species = len(species_names)
    per_process_rhs = {
        name: get_rhs_ode(collection.processes[name])
        for name in store.process_order
    }

    # --- Loss curve ---
    fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
    ax_loss.plot(
        range(1, len(result.mean_loss_by_step) + 1), result.mean_loss_by_step
    )
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Mean loss (MSE)")
    ax_loss.set_title("Training loss")
    ax_loss.set_yscale("log")
    ax_loss.grid(True, alpha=0.3)
    fig_loss.tight_layout()
    fig_loss.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close(fig_loss)
    logger.info("loss curve saved to %s", output_dir / "loss_curve.png")

    # --- Per-process plots ---
    for process_name in store.process_order:
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
                rtol=solver_rtol, atol=solver_atol
            ),
            max_steps=solver_max_steps,
            throw=False,
        )
        states_physical = jax.vmap(process_wrapper.unscale_state)(sol.ys)
        c_dense = np.asarray(states_physical[:, :-1])
        v_cont = np.asarray(states_physical[:, -1])
        t_dense_np = np.asarray(t_dense)

        # Real volume = container volume - accumulated sample volume
        v_sample_acc = np.array(
            [
                float(
                    process_wrapper.controls.eval(jnp.asarray(float(t_)))[
                        process_wrapper.sample_acc_control_index
                    ]
                )
                for t_ in t_dense_np
            ]
        )
        v_real = v_cont - v_sample_acc

        # Specific rates q(t) along the trajectory
        q_dense = []
        for i_t in range(len(t_dense_np)):
            t_val = jnp.asarray(t_dense_np[i_t])
            y_scaled = sol.ys[i_t]
            c_scaled = jnp.clip(y_scaled[:-1], 0.0)
            controls_vec = process_wrapper.controls.eval(t_val)
            cin_flat = jnp.concatenate(
                [
                    process_wrapper.rhs_ode.Cin.reshape(-1),
                    process_wrapper.rhs_ode.Cin_modeled.reshape(-1),
                ]
            )
            U_aug = jnp.concatenate([controls_vec, cin_flat])
            u_scaled = U_aug / process_wrapper.controls_scale
            outputs = process_wrapper.reaction_module(t_val, c_scaled, u_scaled)
            Q = np.asarray(outputs.specific_rates) * np.asarray(
                process_wrapper.q_scale
            )
            q_dense.append(Q)
        q_dense = np.stack(q_dense, axis=0)

        # --- 2-column layout: left=state, right=rate, + volume row ---
        n_rows = n_species + 1
        fig, axes = plt.subplots(
            n_rows, 2, squeeze=False, figsize=(10, 3 * n_rows)
        )

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

        ax_v = axes[n_species, 0]
        ax_v.plot(t_dense_np, v_real, "-", lw=1.5, color="C0", label="integrated")
        ax_v.set_title(f"Volume [{process.volume.unit}]")
        ax_v.set_xlabel(f"time [{time_unit}]")
        ax_v.set_xlim(t_start, t_end)
        ax_v.legend(fontsize="small")
        ax_v.grid(True, alpha=0.3)

        axes[n_species, 1].set_visible(False)

        fig.suptitle(f"{process_name}", fontsize=12)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{process_name}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    logger.info("plots saved to %s", output_dir)
