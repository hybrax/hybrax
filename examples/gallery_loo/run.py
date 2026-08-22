"""Gallery: cross-validation.

A cheap holdout check with no fold loop (Python API), then a full
leave-one-out run via the CLI: real folds, one per process.

Neither needs a custom reaction module; both train the plain default MLP.

See docs/source/gallery/loo.md for the narrated version.
"""

import contextlib
import io
import os
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import hybrax.format as hxf
from hybrax.train import TrainHarnessConfig, train_from_collection

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

# --- A cheap first check: holdout_processes, no fold loop -------------------
collection = hxf.serialization.load_process_collection(str(HERE / "data.json"))

cfg = TrainHarnessConfig(
    epochs=300, seed=0, learning_rate=0.02,
    holdout_processes=("run_3",),
    checkpoint_dir=str(HERE / "holdout_check/checkpoints"),
    checkpoint_every=50,
)
with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
    warnings.simplefilter("ignore")
    result = train_from_collection(collection, config=cfg)

print(f"final train mean_loss   = {result.mean_loss_by_step[-1]:.4f}")
last_step = max(result.holdout_loss_by_step)
print(f"final holdout (run_3)   = {result.holdout_loss_by_step[last_step]:.4f}"
      f"  (label: {result.holdout_label})")

steps = sorted(result.holdout_loss_by_step)
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.semilogy(range(1, len(result.mean_loss_by_step) + 1), result.mean_loss_by_step,
            label="train (all 3 processes)")
ax.semilogy(steps, [result.holdout_loss_by_step[s] for s in steps], "o--",
            label="holdout (run_3)")
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.legend()
fig.tight_layout()
out = HERE / "out"
out.mkdir(exist_ok=True)
fig.savefig(out / "holdout_check.png")
print(f"wrote {out / 'holdout_check.png'}")

# --- Full leave-one-out, via the CLI -----------------------------------------
hxt_cli("loo", "--config", "loo-config.json", "--overwrite")

summary = pd.read_csv(HERE / "loo_run/loo_summary.csv")
folds = summary[summary["fold_idx"] != "mean"]
print(f"{'fold':10s} {'held out':10s} {'holdout loss':>13s} {'train loss':>11s}")
for _, r in folds.iterrows():
    print(f"{r['fold_slug']:10s} {r['test']:10s} "
          f"{r['holdout_total']:13.4f} {r['train_mean_total']:11.4f}")

print(f"run directory: {HERE / 'loo_run'}")
for path in sorted((HERE / "loo_run").iterdir()):
    print("  " + path.name + ("/" if path.is_dir() else ""))
