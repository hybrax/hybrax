"""Gallery: custom losses on the dense grid.

Adds three terms on top of Tutorial 4's reaction module and scales, all
evaluated on a dense time grid rather than only at measurement times: a state
bounds hinge, a rate bounds hinge, and a rate-smoothness penalty. Compares
against Tutorial 4's unconstrained loss on the same data/seed/epochs.

See docs/source/gallery/dense_loss.md for the narrated version.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import hybrax.format as hxf

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


def dense_diagnostics(run_dir):
    """Worst glucose excursion and RMS curvature per rate, in physical space."""
    df = pd.read_csv(HERE / run_dir / "predictions.csv")
    min_glucose = float(df["c_glucose"].min())
    curvature = {"q_biomass": [], "q_glucose": []}
    for _, proc_df in df.groupby("process"):
        t = proc_df["t"].to_numpy()
        dt = t[1] - t[0]
        for rate in curvature:
            y = proc_df[rate].to_numpy()
            d2 = (y[2:] - 2 * y[1:-1] + y[:-2]) / dt**2
            curvature[rate].append(float(np.sqrt(np.mean(d2**2))))
    return min_glucose, {k: float(np.mean(v)) for k, v in curvature.items()}


hxt_cli(
    "prepare",
    "--config",
    "prepare-config.json",
    "--output-dir",
    "prepared",
    "--overwrite",
)
out = hxt_cli("train", "--config", "train-full.json", "--overwrite")
print([l for l in out.splitlines() if "training complete" in l][-1])

r2 = r2_by_target("run_full")
min_glucose, curvature = dense_diagnostics("run_full")
print(f"{'target':10s} {'R2':>8s}")
for name, value in r2.items():
    print(f"{name:10s} {value:8.4f}")
print(f"min predicted glucose : {min_glucose:+.4f} g/L  (bound: >= 0)")

hxt_cli("train", "--config", "train-base.json", "--overwrite")
r2_base = r2_by_target("run_base")
min_glucose_base, curvature_base = dense_diagnostics("run_base")

print(f"{'target':10s} {'no penalty':>12s} {'with penalty':>14s}")
for name in ("biomass", "glucose", "product"):
    print(f"{name:10s} {r2_base[name]:12.4f} {r2[name]:14.4f}")
print(f"{'min glucose':10s} {min_glucose_base:12.4f} {min_glucose:14.4f}")
print(
    f"{'curv(q_X)':10s} {curvature_base['q_biomass']:12.4f} {curvature['q_biomass']:14.4f}"
)
print(
    f"{'curv(q_S)':10s} {curvature_base['q_glucose']:12.4f} {curvature['q_glucose']:14.4f}"
)

hxt_cli(
    "forward",
    "--config",
    "forward-config.json",
    "--output-dir",
    "run_full/forward",
    "--overwrite",
)
print(f"forward plot: {HERE / 'run_full/forward/forward-results/plots/run_1.png'}")
