"""Gallery: glutamine decay.

One physical rate, r_Gln, declared once in biological_ode.rates, feeding two
different derivatives at once (a sink in Gln, a source in NH4). Checks that
hybrax.train recovers that single shared number from data alone.

See docs/source/gallery/glutamine_decay.md for the narrated version.
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

glutamine_ode = hxf.BiologicalOde(
    rates={"q_biomass": (None, None), "q_Gln": (None, None), "r_Gln": (None, None)},
    derivatives={
        "biomass": "q_biomass * biomass",
        "Gln": "-q_Gln * biomass - r_Gln * Gln",
        "NH4": "r_Gln * Gln",
    },
)
print(
    "matches the real declared biological_ode:", glutamine_ode == process.biological_ode
)
hxf.print_rhs_ode(process)


def r2_by_target(run_dir):
    """Pooled R2: concatenate every process's residuals/variance first."""
    df = pd.read_csv(HERE / run_dir / "predictions.csv")
    per_target = {s: ([], []) for s in ("biomass", "Gln", "NH4")}
    for name, proc in _collection.processes.items():
        proc_df = df[df["process"] == name]
        t_pred = proc_df["t"].to_numpy()
        for species in per_target:
            comp = proc.reactor_medium.components[species].concentration
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
print(f"forward plot: {HERE / 'run/forward/plots/run_1.png'}")

wrapper, cfg = hxt.model_load(str(HERE / "run"))
rm = wrapper.reaction_module
truth = json.loads((HERE / "ground_truth.json").read_text())

fitted = {
    "q_biomass": float(jnp.exp(rm.log_q_biomass)),
    "q_Gln": float(jnp.exp(rm.log_q_Gln)),
    "r_Gln": float(jnp.exp(rm.log_r_Gln)),
}
print(f"{'parameter':10s} {'fitted':>12s} {'true':>12s}")
for name in truth:
    print(f"{name:10s} {fitted[name]:12.6g} {truth[name]:12.6g}")
