"""Gallery: FBA-Hyb.

A hybrid dynamic-FBA reaction module: two small MLPs predict a glucose-uptake
rate and an FBA objective from the current state, a frozen, pole-free
surrogate (fit offline against 10,000 real pFBA solves on e_coli_core.xml,
see 01_generate_fba_data.py / 02_fit_surrogate.py, not run here) converts
that into real metabolic rates.

See docs/source/gallery/fba_hyb.md for the narrated version.
"""

import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    per_target = {s: ([], []) for s in ("biomass", "glucose", "acetate")}
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
print(f"forward plot: {HERE / 'run/forward/forward-results/plots/run_1.png'}")

df = pd.read_csv(HERE / "run" / "predictions.csv")
run_1 = df[df["process"] == "run_1"]
t = run_1["t"].to_numpy()

fig, ax = plt.subplots(figsize=(7, 3.5))
for i, label in enumerate(("n_X (biomass)", "n_M (maintenance)", "n_A (acetate)")):
    ax.plot(t, run_1[f"aux_n_weights_{i}"].to_numpy(), label=label)
ax.set_xlabel("t (h)")
ax.set_ylabel("predicted FBA objective weight")
ax.legend(fontsize=8)
fig.tight_layout()
out = HERE / "out"
out.mkdir(exist_ok=True)
fig.savefig(out / "objective_weights.png")
print(f"wrote {out / 'objective_weights.png'}")
