"""Tutorial 4: replace the reaction module and scale estimation, and measure
what it bought you.

Trains demo_batch twice (built-in defaults vs. custom.py's scaled MLP), then
compares them by R^2 in physical space (the fair comparison: their losses
live in different units).

See docs/source/tutorials/04_your_first_custom_py.md for the narrated version.
"""

import os
import subprocess
import sys
from pathlib import Path

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
    """Physical-space R^2 per target, averaged over processes."""
    df = pd.read_csv(HERE / run_dir / "predictions.csv")
    per_target = {}
    for name, process in _collection.processes.items():
        proc_df = df[df["process"] == name]
        t_pred = proc_df["t"].to_numpy()
        for species in ("biomass", "glucose", "product"):
            comp = process.reactor_medium.components[species].concentration
            t_meas = np.asarray(comp.times)
            y_meas = np.asarray(comp.values)
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
hxt_cli("train", "--config", "train-default.json", "--overwrite")
hxt_cli("train", "--config", "train-custom.json", "--overwrite")

r2_default = r2_by_target("run_default")
r2_custom = r2_by_target("run_custom")

print(f"{'target':10s} {'defaults':>10s} {'custom.py':>10s}")
for name in ("biomass", "glucose", "product"):
    print(f"{name:10s} {r2_default[name]:10.4f} {r2_custom[name]:10.4f}")

hxt_cli(
    "forward",
    "--config",
    "forward-config.json",
    "--output-dir",
    "run_custom/forward",
    "--overwrite",
)
print(f"forward plot: {HERE / 'run_custom/forward/forward-results/plots/run_1.png'}")

wrapper, cfg = hxt.model_load(str(HERE / "run_custom"))
hxt.print_trainable_structure(wrapper)
hxt.print_reaction_schema(wrapper)
