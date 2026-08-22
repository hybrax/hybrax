"""Tutorial 5: use a trained model, from the command line and from Python.

Trains the same model as Tutorial 4, then demonstrates `forward` (CLI) and
`model_load` / `model_predict` (Python), plotting the fit against measurements.

See docs/source/tutorials/05_predict.md for the narrated version.
"""

import json
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


hxt_cli("prepare", "--config", "prepare-config.json",
        "--output-dir", "prepared", "--overwrite")
hxt_cli("train", "--config", "train-config.json", "--overwrite")

# --- The command-line way: forward -------------------------------------------
hxt_cli("forward", "--config", "forward-config.json",
        "--output-dir", "run/forward", "--overwrite")
print((HERE / "run/forward/forward-results/losses.csv").read_text())

df = pd.read_csv(HERE / "run/forward/forward-results/predictions.csv")
print("columns:", ", ".join(df.columns))
print("rows   :", len(df))

# --- The Python way: model_load and model_predict ----------------------------
wrapper, config = hxt.model_load(str(HERE / "run"))
print(type(wrapper).__name__)

collection = hxf.serialization.load_process_collection(HERE / "data.json")
predictions = hxt.model_predict(wrapper, config, collection, grid_n=200)
print(list(predictions))

export = predictions["run_1"]
print("t          ", export.t.shape)
print("c_species  ", export.c_species.shape)
print("q_rates    ", export.q_rates.shape)
print("v_real     ", export.v_real.shape)

truth = json.loads((HERE / "ground_truth.json").read_text())
learned_mu = float(export.q_rates[0, 0])
print(f"learned q_biomass(t=0) : {learned_mu:.3f} 1/h")
print(f"true mu_max            : {truth['mu_max']:.3f} 1/h")

measured = collection.processes["run_1"].reactor_medium.components
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)
for ax, (i, name) in zip(axes, enumerate(["biomass", "glucose", "product"])):
    ax.plot(export.t, export.c_species[:, i], label="predicted")
    ax.plot(np.asarray(measured[name].concentration.times),
            np.asarray(measured[name].concentration.values),
            "k.", label="measured")
    ax.set_title(name)
    ax.set_xlabel("time [h]")
axes[0].set_ylabel("g/L")
fig.suptitle("run_1: fit on training data")
axes[0].legend()
out = HERE / "out"
out.mkdir(exist_ok=True)
fig.savefig(out / "run_1_fit.png")
print(f"wrote {out / 'run_1_fit.png'}")
