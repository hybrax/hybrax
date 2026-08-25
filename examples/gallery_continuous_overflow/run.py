"""Gallery: train hybrid models across batch, fed-batch, and continuous phases.

A prescribed feed fills the reactor from 0.5 L to 1.0 L. An equal prescribed
outflow then represents the realized passive overflow and keeps the volume
constant. The same noiseless trajectory trains a two-parameter Monod model and
a 33-parameter neural reaction module.

See docs/source/gallery/continuous_overflow.md for the narrated version.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import hybrax.format as hxf
import hybrax.train as hxt
from hybrax.format.mechanistic import build_rhs_ode
from hybrax.train import ReactionInputs, ReactionOutputs, UserReactionModule
from hybrax.train.controls_store import ControlsStore
from hybrax.train.physical_solve import solve_physical_states
from hybrax.train.wrapper import HybridOdeWrapper

HERE = Path(__file__).parent
ENV = {
    **os.environ,
    "JAX_PLATFORMS": "cpu",
    "HYBRAX_TRAIN_DEVICES": "1",
    "MPLBACKEND": "Agg",
}
STAGES = (1, 50, 200)


def hxt_cli(*args):
    """Run one Hybrax CLI command from the example directory."""
    proc = subprocess.run(
        [sys.executable, "-m", "hybrax.train.cli", *args],
        cwd=HERE,
        env=ENV,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout + proc.stderr


class GroundTruthMonod(UserReactionModule):
    """Fixed Monod law used only to reconstruct the dense reference curve."""

    mu_max: float = eqx.field(static=True)
    ks: float = eqx.field(static=True)
    i_glucose: int = eqx.field(static=True)

    def __init__(self, *, mu_max, ks, i_glucose, **scale_kwargs):
        super().__init__(**scale_kwargs)
        self.mu_max = mu_max
        self.ks = ks
        self.i_glucose = i_glucose

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        states = self.unscale_modeled_RMCs(inputs.SCL_modeled_RMCs)
        glucose = jnp.maximum(states[self.i_glucose], 0.0)
        mu = self.mu_max * glucose / (self.ks + glucose)
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=(
                self.scale_modeled_BiologicalOde_rates(jnp.asarray([mu]))
            ),
            SCL_modeled_Inflows_rates=jnp.zeros(0),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
        )


def unit_scales():
    """Unit scales for a fixed physical-space reference model."""
    return {
        "SCALE_modeled_RMCs": jnp.ones(2),
        "SCALE_V_in_cumulative": jnp.asarray(1.0),
        "SCALE_modeled_Inflows_cumulative": jnp.zeros(0),
        "SCALE_modeled_Outflows_cumulative": jnp.zeros(0),
        "SCALE_controlled_Inflows_cumulative": jnp.ones(1),
        "SCALE_controlled_Inflows_rates": jnp.ones(1),
        "SCALE_controlled_Inflows_Cin": jnp.ones((1, 2)),
        "SCALE_controlled_Outflows_cumulative": jnp.ones(1),
        "SCALE_controlled_Outflows_rates": jnp.ones(1),
        "SCALE_controlled_PVs": jnp.zeros(0),
        "SCALE_modeled_Inflows_Cin": jnp.zeros((0, 2)),
        "SCALE_modeled_BiologicalOde_rates": jnp.ones(1),
        "SCALE_modeled_Inflows_rates": jnp.zeros(0),
        "SCALE_modeled_Outflows_rates": jnp.zeros(0),
    }


def solve_states(wrapper, times, truth):
    """Solve physical biomass, glucose, and volume at ``times``."""
    return np.asarray(
        solve_physical_states(
            wrapper,
            t_eval=times,
            n_measured=times.size,
            RAW_y0=jnp.asarray(
                [
                    truth["initial_biomass"],
                    truth["initial_glucose"],
                    truth["initial_volume"],
                ]
            ),
            max_steps=100_000,
            rtol=1e-8,
            atol=1e-10,
            jump_ts=jnp.asarray([truth["feed_start"], truth["overflow_start"]]),
            n_linspace=times.size,
        )
    )


def train_stages(model):
    """Train once for epoch 1 and once through epoch 200, retaining 1/50/200."""
    print(f"training {model} model...")
    hxt_cli("train", "--config", f"train-{model}-early.json", "--overwrite")
    hxt_cli("train", "--config", f"train-{model}.json", "--overwrite")

    checkpoints = HERE / f"run_{model}/checkpoints"
    early_run = HERE / f"run_{model}_early"
    shutil.copytree(
        early_run / "checkpoints/step_00001",
        checkpoints / "step_00001",
    )
    for path in checkpoints.glob("step_*"):
        if int(path.name.removeprefix("step_")) not in STAGES:
            shutil.rmtree(path)
    shutil.rmtree(early_run)


def load_stages(model):
    """Load the three retained training snapshots."""
    paths = sorted((HERE / f"run_{model}/checkpoints").glob("step_*"))
    assert [int(path.name.removeprefix("step_")) for path in paths] == list(STAGES)
    return [hxt.model_load(path)[0] for path in paths]


def ann_growth_curve(module, substrate_grid):
    """Evaluate the ANN's physical growth rate over a glucose grid."""
    states = jnp.zeros(module.n_modeled_RMCs)
    return np.asarray(
        [
            module.unscale_modeled_BiologicalOde_rates(
                module.mlp(
                    module.scale_modeled_RMCs(
                        states.at[module.i_glucose].set(substrate)
                    )[module.i_glucose : module.i_glucose + 1]
                )
            )[0]
            for substrate in substrate_grid
        ]
    )


def plot_process(times, states, controls, truth):
    """Plot the four operational phases and their asymptotic steady state."""
    biomass, glucose, volume = states.T
    rates = np.asarray(
        [
            jnp.concatenate(
                [
                    controls.eval_controlled_Inflows_rates(t, None),
                    controls.eval_controlled_Outflows_rates(t, None),
                ]
            )
            for t in times
        ]
    )
    dilution_rate = truth["flow_rate"] / truth["overflow_volume"]
    steady_glucose = truth["Ks"] * dilution_rate / (truth["mu_max"] - dilution_rate)
    steady_biomass = truth["yield_xs"] * (truth["feed_glucose"] - steady_glucose)

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 9))
    axes[0].plot(times, biomass, label="biomass")
    axes[0].axhline(steady_biomass, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Biomass [g/L]")
    axes[1].plot(times, glucose, color="tab:orange")
    axes[1].axhline(steady_glucose, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Glucose [g/L]")
    axes[2].plot(times, volume)
    axes[2].axhline(
        truth["overflow_volume"], color="black", linestyle="--", linewidth=1
    )
    axes[2].set_ylabel("Volume [L]")
    axes[3].plot(times, rates[:, 0], label="feed")
    axes[3].plot(times, -rates[:, 1], label="overflow")
    axes[3].set(xlabel="Time [h]", ylabel="Flow [L/h]")
    axes[3].legend()
    for axis in axes:
        axis.axvspan(truth["batch_end"], truth["feed_start"], color="black", alpha=0.08)
        axis.axvline(truth["feed_start"], color="tab:green", linestyle=":")
        axis.axvline(truth["overflow_start"], color="tab:red", linestyle=":")
    fig.tight_layout()
    fig.savefig(HERE / "process.png", dpi=150)
    plt.close(fig)
    return steady_biomass, steady_glucose


def plot_training(times, truth_states, substrate_grid, true_mu, model_data):
    """Compare rate laws and biomass trajectories as training progresses."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    styles = ("-", ":", "--")
    alphas = (0.4, 1.0, 1.0)
    stage_names = ("early", "middle", "late")

    for column, (title, rate_curves, state_curves, color) in enumerate(model_data):
        axes[0, column].set_title(title)
        axes[0, column].axhline(0.0, color="0.6", linewidth=1, zorder=0)
        axes[0, column].plot(
            substrate_grid,
            true_mu,
            color="black",
            label="ground truth",
            linewidth=2.5,
        )
        axes[1, column].axhline(0.0, color="0.6", linewidth=1, zorder=0)
        axes[1, column].plot(
            times,
            truth_states[:, 0],
            color="black",
            label="ground truth",
            linewidth=2.5,
        )
        for stage, epoch, style, alpha, rate_curve, state_curve in zip(
            stage_names,
            STAGES,
            styles,
            alphas,
            rate_curves,
            state_curves,
            strict=True,
        ):
            unit = "epoch" if epoch == 1 else "epochs"
            label = f"{stage} ({epoch} {unit})"
            axes[0, column].plot(
                substrate_grid,
                rate_curve,
                color=color,
                linestyle=style,
                alpha=alpha,
                linewidth=2.5,
                label=label,
            )
            axes[1, column].plot(
                times,
                state_curve[:, 0],
                color=color,
                linestyle=style,
                alpha=alpha,
                linewidth=2.5,
                label=label,
            )
        axes[0, column].set_xlabel("Glucose [g/L]")
        axes[1, column].set_xlabel("Time [h]")
        axes[0, column].legend()
    axes[0, 0].set_ylabel("Growth rate [1/h]")
    axes[1, 0].set_ylabel("Biomass [g/L]")
    fig.tight_layout()
    fig.savefig(HERE / "training.png", dpi=150)
    plt.close(fig)


def main():
    truth = json.loads((HERE / "ground_truth.json").read_text())
    collection = hxf.serialization.load_process_collection(HERE / "data.json")
    process = collection.processes["continuous_1"]
    controls = ControlsStore.from_collection(collection).get_controls("continuous_1")
    names = list(build_rhs_ode(process).name_modeled_RMCs)
    truth_module = GroundTruthMonod(
        mu_max=truth["mu_max"],
        ks=truth["Ks"],
        i_glucose=names.index("glucose"),
        **unit_scales(),
    )
    truth_wrapper = HybridOdeWrapper.from_process(
        reaction_module=truth_module,
        process=process,
        controls=controls,
    )
    times = jnp.linspace(0.0, truth["end_time"], 801)
    truth_states = solve_states(truth_wrapper, times, truth)

    hxt_cli(
        "prepare",
        "--config",
        "prepare-config.json",
        "--output-dir",
        "prepared",
        "--overwrite",
    )
    for model in ("monod", "ann"):
        train_stages(model)

    monod_wrappers = load_stages("monod")
    ann_wrappers = load_stages("ann")
    monod_modules = [wrapper.reaction_module for wrapper in monod_wrappers]
    ann_modules = [wrapper.reaction_module for wrapper in ann_wrappers]
    monod_states = [solve_states(wrapper, times, truth) for wrapper in monod_wrappers]
    ann_states = [solve_states(wrapper, times, truth) for wrapper in ann_wrappers]

    substrate_grid = np.linspace(0.0, float(np.max(truth_states[:, 1])), 300)
    true_mu = truth["mu_max"] * substrate_grid / (truth["Ks"] + substrate_grid)
    monod_mu = [
        float(jnp.exp(module.log_mu_max))
        * substrate_grid
        / (float(jnp.exp(module.log_ks)) + substrate_grid)
        for module in monod_modules
    ]
    ann_mu = [ann_growth_curve(module, substrate_grid) for module in ann_modules]

    fitted_mu_max = float(jnp.exp(monod_modules[-1].log_mu_max))
    fitted_ks = float(jnp.exp(monod_modules[-1].log_ks))
    ann_rmse = float(np.sqrt(np.mean((ann_mu[-1] - true_mu) ** 2)))

    steady_biomass, steady_glucose = plot_process(times, truth_states, controls, truth)
    np.testing.assert_array_less(truth_states[:, 2], truth["overflow_volume"] + 1e-7)
    np.testing.assert_allclose(truth_states[-1, 2], truth["overflow_volume"], atol=1e-7)
    assert truth_states[-1, 0] < steady_biomass
    assert truth_states[-1, 1] > steady_glucose

    plot_training(
        times,
        truth_states,
        substrate_grid,
        true_mu,
        (
            ("Mechanistic Monod", monod_mu, monod_states, "purple"),
            ("ANN hybrid model", ann_mu, ann_states, "green"),
        ),
    )

    results = {
        "fitted_mu_max": fitted_mu_max,
        "fitted_Ks": fitted_ks,
        "ann_rate_rmse": ann_rmse,
        "final_biomass": float(truth_states[-1, 0]),
        "final_glucose": float(truth_states[-1, 1]),
        "maximum_volume": float(np.max(truth_states[:, 2])),
        "steady_biomass": steady_biomass,
        "steady_glucose": steady_glucose,
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"process plot: {HERE / 'process.png'}")
    print(f"training plot: {HERE / 'training.png'}")


if __name__ == "__main__":
    main()
