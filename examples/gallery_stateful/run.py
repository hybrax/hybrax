"""Gallery: stateful models.

A reaction module with its own memory (a continuous-time LSTM whose hidden
and cell state are integrated as extra ODE dimensions), and the opt-in
(`allow_stateful_models: true`) that guards it.

See docs/source/gallery/stateful.md for the narrated version.
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


def hxt_cli(*args, check=True):
    proc = subprocess.run(
        [sys.executable, "-m", "hybrax.train.cli", *args],
        cwd=HERE,
        env=ENV,
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
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

# Declaring a nonzero latent without opting in is a hard error, by design.
out = hxt_cli("train", "--config", "train-no-optin.json", "--overwrite", check=False)
print([l for l in out.splitlines() if "ValueError" in l][-1])

hxt_cli("train", "--config", "train.json", "--overwrite")
final_loss = pd.read_csv(HERE / "run/metrics.csv")["mean_loss"].iloc[-1]
print(f"final mean_loss = {final_loss:.5g}")

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
print("n_latent =", wrapper.reaction_module.n_latent)
hxt.print_trainable_structure(wrapper)
