"""Gallery: a Gaussian-process model.

A sparse Gaussian process (mean and variance, via a Cholesky solve over
trainable inducing points) occupying the reaction module's slot, trained
end to end by hybrax.train's own optimizer. Predictive uncertainty is read
out through ReactionOutputs.auxiliary.

See docs/source/gallery/gaussian_process.md for the narrated version.
"""

import os
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hybrax.format as hxf
import hybrax.train as hxt

HERE = Path(__file__).parent
ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "HYBRAX_TRAIN_DEVICES": "1",
       "MPLBACKEND": "Agg"}


def hxt_cli(*args):
    proc = subprocess.run([sys.executable, "-m", "hybrax.train.cli", *args],
                          cwd=HERE, env=ENV, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout + proc.stderr


_collection = hxf.serialization.load_process_collection(HERE / "data.json")


def r2_by_target(run_dir):
    df = pd.read_csv(HERE / run_dir / "predictions.csv")
    per_target = {}
    for name, process in _collection.processes.items():
        proc_df = df[df["process"] == name]
        t_pred = proc_df["t"].to_numpy()
        for species in ("biomass", "glucose", "product"):
            comp = process.reactor_medium.components[species].concentration
            t_meas, y_meas = np.asarray(comp.times), np.asarray(comp.values)
            y_pred = np.interp(t_meas, t_pred, proc_df[f"c_{species}"].to_numpy())
            ss_res = np.sum((y_meas - y_pred) ** 2)
            ss_tot = np.sum((y_meas - y_meas.mean()) ** 2)
            per_target.setdefault(species, []).append(1 - ss_res / ss_tot)
    return {k: float(np.mean(v)) for k, v in per_target.items()}


hxt_cli("prepare", "--config", "prepare-config.json",
        "--output-dir", "prepared", "--overwrite")
hxt_cli("train", "--config", "train-config.json", "--overwrite")

r2 = r2_by_target("run")
for name, value in r2.items():
    print(f"{name:10s} R2 = {value:.4f}")

hxt_cli("forward", "--config", "forward-config.json",
        "--output-dir", "run/forward", "--overwrite")
print(f"forward plot: {HERE / 'run/forward/forward-results/plots/run_1.png'}")

wrapper, cfg = hxt.model_load(str(HERE / "run"))
gp = wrapper.reaction_module
print(f"centers: spread (std) per feature = {jnp.std(gp.centers, axis=0)}")
print(f"lengthscale (exp)   = {jnp.exp(gp.log_lengthscale)}")
print(f"output_scale (exp)  = {float(jnp.exp(gp.log_output_scale)):.3f}")
print(f"noise (exp)         = {float(jnp.exp(gp.log_noise)):.3f}")

df = pd.read_csv(HERE / "run" / "predictions.csv")
run_1 = df[df["process"] == "run_1"]
t = run_1["t"].to_numpy()

fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharex=True)
for ax, (qcol, stdcol) in zip(axes, [("q_biomass", "aux_rate_std_0"),
                                      ("q_glucose", "aux_rate_std_1"),
                                      ("q_product", "aux_rate_std_2")]):
    q = run_1[qcol].to_numpy()
    std = run_1[stdcol].to_numpy()
    ax.plot(t, q, color="tab:blue")
    ax.fill_between(t, q - 2 * std, q + 2 * std, alpha=0.25, color="tab:blue")
    ax.set_title(qcol)
    ax.set_xlabel("t (h)")
fig.suptitle("run_1: predicted rate +/- 2 rate_std")
fig.tight_layout()
out = HERE / "out"
out.mkdir(exist_ok=True)
fig.savefig(out / "uncertainty.png")
print(f"wrote {out / 'uncertainty.png'}")
