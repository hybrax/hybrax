"""Gallery: freezing parameters.

Splits a reaction module into a frozen encoder and a trainable head, checked
with print_trainable_structure, then compares against the same architecture
with the encoder also trainable to show what freezing costs.

See docs/source/gallery/freezing.md for the narrated version.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
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


hxt_cli(
    "prepare",
    "--config",
    "prepare-config.json",
    "--output-dir",
    "prepared",
    "--overwrite",
)
hxt_cli("train", "--config", "train-config.json", "--overwrite")

hxt_cli(
    "forward",
    "--config",
    "forward-config.json",
    "--output-dir",
    "run/forward",
    "--overwrite",
)
print(f"forward plot: {HERE / 'run/forward/forward-results/plots/run_1.png'}")

wrapper, _cfg = hxt.model_load(str(HERE / "run"))
hxt.print_trainable_structure(wrapper)

hxt_cli("train", "--config", "train-config-unfrozen.json", "--overwrite")


def final_loss(run_dir):
    return float(pd.read_csv(HERE / run_dir / "metrics.csv")["mean_loss"].iloc[-1])


print(f"frozen encoder    final mean_loss = {final_loss('run'):.5f}")
print(f"trainable encoder final mean_loss = {final_loss('run_unfrozen'):.5f}")
