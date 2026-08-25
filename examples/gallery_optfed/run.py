"""Gallery: OptFed.

A real, published mechanistic rate law (non-competitive-inhibition
Michaelis-Menten kinetics with an Eyring-equation temperature dependence)
occupying the reaction module's slot, with a controlled process variable
(temperature) feeding directly into the kinetics.

See docs/source/gallery/optfed.md for the narrated version.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
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
    """Pooled R2: concatenate every process's residuals/variance first, so
    one narrow-range process can't make an otherwise-good fit look bad."""
    df = pd.read_csv(HERE / run_dir / "predictions.csv")
    per_target = {s: ([], []) for s in ("biomass", "glucose", "product")}
    for name, process in _collection.processes.items():
        proc_df = df[df["process"] == name]
        t_pred = proc_df["t"].to_numpy()
        for species in per_target:
            comp = process.reactor_medium.components[species].concentration
            t_meas, y_meas = np.asarray(comp.times), np.asarray(comp.values)
            y_pred = np.interp(t_meas, t_pred, proc_df[f"c_{species}"].to_numpy())
            per_target[species][0].append(y_meas)
            per_target[species][1].append(y_pred)
    out = {}
    for species, (meas_list, pred_list) in per_target.items():
        y_meas, y_pred = np.concatenate(meas_list), np.concatenate(pred_list)
        ss_res = np.sum((y_meas - y_pred) ** 2)
        ss_tot = np.sum((y_meas - y_meas.mean()) ** 2)
        out[species] = 1 - ss_res / ss_tot
    return out


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
print(f"forward plot (T_high): {HERE / 'run/forward/plots/T_high.png'}")

wrapper, cfg = hxt.model_load(str(HERE / "run"))
m = wrapper.reaction_module
fitted_Teq_K = 290.0 + 40.0 * jax.nn.sigmoid(m.eyring_raw_Teq)
fitted_Teq_C = np.asarray(fitted_Teq_K) - 273.15

truth = json.loads((HERE / "ground_truth.json").read_text())
true_Teq_C = [
    truth["eyring_deg"]["Teq"] - 273.15,
    truth["eyring_pi"]["Teq"] - 273.15,
    truth["eyring_alpha"]["Teq"] - 273.15,
]

print(f"{'rate':10s} {'fitted Teq':>12s} {'true Teq':>10s}")
for name, fit, true in zip(
    ("uptake", "production", "maintenance"), fitted_Teq_C, true_Teq_C
):
    print(f"{name:10s} {fit:11.1f}C {true:9.1f}C")
