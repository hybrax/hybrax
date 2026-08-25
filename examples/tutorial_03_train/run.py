"""Tutorial 3: fit a hybrid ODE to demo_batch using every default.

Runs prepare -> train -> forward via the hybrax CLI, in this directory, then
prints the per-epoch metrics and the final rows of forward's loss report.

See docs/source/tutorials/03_train.md for the narrated version.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

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

metrics = pd.read_csv(HERE / "run/metrics.csv")
print("columns:", ", ".join(metrics.columns))
for i in (0, len(metrics) // 2, len(metrics) - 1):
    print(metrics.iloc[i, :4].to_dict())

print(f"loss curve: {HERE / 'run/loss_curve.png'}")
print(f"forward plot: {HERE / 'run/forward/plots/run_1.png'}")
