"""Gallery: a KAN model.

A Kolmogorov-Arnold Network occupying the reaction module's slot: every edge
between an input and a hidden or output node carries its own learnable
univariate function, summed at each node, instead of an MLP's fixed
activation with learned linear weights.

See docs/source/gallery/kan.md for the narrated version.
"""

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hybrax.format as hxf
import hybrax.train as hxt

HERE = Path(__file__).parent
ENV = {
    **os.environ,
    "JAX_PLATFORMS": "cpu",
    "HYBRAX_TRAIN_DEVICES": "1",
    "MPLBACKEND": "Agg",
}


def hxt_cli(*args):
    proc = subprocess.run(
        [sys.executable, "-m", "hybrax.train.cli", *args],
        cwd=HERE,
        env=ENV,
        capture_output=True,
        text=True,
    )
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


hxt_cli(
    "prepare",
    "--config",
    "prepare-config.json",
    "--output-dir",
    "prepared",
    "--overwrite",
)
hxt_cli("train", "--config", "train-config.json", "--overwrite")

r2 = r2_by_target("run")
for name, value in r2.items():
    print(f"{name:10s} R2 = {value:.4f}")

hxt_cli(
    "forward",
    "--config",
    "forward-config.json",
    "--output-dir",
    "run/forward",
    "--overwrite",
)
print(f"forward plot: {HERE / 'run/forward/forward-results/plots/run_1.png'}")

wrapper, cfg = hxt.model_load(str(HERE / "run"))
kan = wrapper.reaction_module
l1 = kan.l1
names = list(wrapper.modeled_RMC_names)
scale = np.asarray(kan.SCALE_modeled_RMCs.scale)

df = pd.read_csv(HERE / "run" / "predictions.csv")


def edge_curve(o, i, xs_scl):
    xb = jnp.tanh(xs_scl)
    rbf = jnp.exp(-l1.inv_h2 * (xb[:, None] - l1.centers[None, :]) ** 2)
    spline = jnp.einsum("g,ng->n", l1.spline_c[o, i], rbf)
    base = l1.base_w[o, i] * jax.nn.silu(xb)
    return spline + base


fig, axes = plt.subplots(1, 3, figsize=(11, 3))
for ax, species in zip(axes, names):
    i = names.index(species)
    vals = df[f"c_{species}"].to_numpy()
    lo, hi = max(float(vals.min()), 0.0), float(vals.max())
    xs_raw = np.linspace(lo, hi, 60)
    xs_scl = jnp.asarray(xs_raw / scale[i])
    spreads = [
        float(jnp.ptp(edge_curve(o, i, xs_scl))) for o in range(l1.base_w.shape[0])
    ]
    o = int(np.argmax(spreads))
    ys = np.asarray(edge_curve(o, i, xs_scl))
    ax.plot(xs_raw, ys)
    ax.set_xlabel(f"{species} (g/L)")
    ax.set_title(f"edge: {species} -> hidden {o}")
fig.tight_layout()
out = HERE / "out"
out.mkdir(exist_ok=True)
fig.savefig(out / "edges.png")
print(f"wrote {out / 'edges.png'}")
