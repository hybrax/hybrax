"""Gallery: a modeled process variable.

glyco_frac is declared as a modeled (uncontrolled) process variable, with its
own trained rate r_glyco_frac, alongside the ordinary modeled reactor
component biomass. One large dilution bolus lands midway through each run:
biomass concentration steps down, glyco_frac does not, because hybrax.format
never applies a feed/dilution term to process-variable states.

See docs/source/gallery/modeled_pv.md for the narrated version.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
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
process = _collection.processes["run_1"]

modeled_pv_ode = hxf.ReactionOde(
    rates={"q_biomass": (0.0, None), "r_glyco_frac": (0.0, None)},
    derivatives={
        "biomass": "q_biomass * biomass",
        "glyco_frac": "-r_glyco_frac * glyco_frac",
    },
)
print(
    "matches the real declared reaction_ode:",
    modeled_pv_ode == process.reaction_ode,
)
hxf.print_rhs_ode(process)


def r2_by_target(run_dir):
    """Pooled R2: concatenate every process's residuals/variance first."""
    df = pd.read_csv(HERE / run_dir / "predictions.csv")
    per_target = {"biomass": ([], []), "glyco_frac": ([], [])}
    for name, proc in _collection.processes.items():
        proc_df = df[df["process"] == name]
        t_pred = proc_df["t"].to_numpy()
        biomass = proc.reactor_medium.components["biomass"].concentration
        glyco = proc.process_variables["glyco_frac"].values
        for species, comp in (("biomass", biomass), ("glyco_frac", glyco)):
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
    print(f"{name:12s} R2 = {value:.4f}")

hxt_cli(
    "forward",
    "--config",
    "forward-config.json",
    "--output-dir",
    "run/forward",
    "--overwrite",
)
print(f"forward plot: {HERE / 'run/forward/plots/run_1.png'}")

wrapper, cfg = hxt.model_load(str(HERE / "run"))
rm = wrapper.reaction_module
truth = json.loads((HERE / "ground_truth.json").read_text())

fitted = {
    "q_biomass": float(jnp.exp(rm.log_q_biomass)),
    "r_glyco_frac": float(jnp.exp(rm.log_r_glyco_frac)),
}
print(f"{'parameter':14s} {'fitted':>12s} {'true':>12s}")
for name in truth:
    print(f"{name:14s} {fitted[name]:12.6g} {truth[name]:12.6g}")
