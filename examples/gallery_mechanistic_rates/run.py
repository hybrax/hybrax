"""Gallery: mechanistic models.

Monod growth + Luedeking-Piret product formation instead of a bare MLP: named,
trainable, physically interpretable constants. Checks which recovered
constants match the true simulation parameters, and why the rest don't.

See docs/source/gallery/mechanistic_rates.md for the narrated version.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
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
rm = wrapper.reaction_module
truth = json.loads((HERE / "ground_truth.json").read_text())

fitted = {
    "mu_max": float(jnp.exp(rm.log_mu_max)),
    "Ks": float(jnp.exp(rm.log_Ks)),
    "Y_XS": float(jnp.exp(rm.log_Yxs)),
    "m_s": float(jnp.exp(rm.log_ms)),
    "alpha": float(jnp.exp(rm.log_alpha)),
    "beta": float(jnp.exp(rm.log_beta)),
}

print(f"{'parameter':10s} {'fitted':>10s} {'true':>10s}")
for name in truth:
    print(f"{name:10s} {fitted[name]:10.4f} {truth[name]:10.4f}")

fitted_combo = fitted["alpha"] * fitted["mu_max"] + fitted["beta"]
true_combo = truth["alpha"] * truth["mu_max"] + truth["beta"]
print(f"alpha*mu_max + beta   fitted={fitted_combo:.4f}   true={true_combo:.4f}")
