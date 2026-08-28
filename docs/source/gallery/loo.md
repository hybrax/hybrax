---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Cross-Validation

> This page runs a quick holdout check with no fold loop, then a full leave-one-out
> cross-validation over real folds. It also shows the files each training run
> produces.

Neither of these needs a custom reaction module: the point here is the cross-validation
machinery itself, so this page trains the plain default MLP throughout.

```{code-cell} ipython3
:tags: [remove-cell]

import json, os, shutil, subprocess, sys
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_loo").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
EXAMPLE = Path("../../../examples/gallery_loo").resolve()
shutil.copy(EXAMPLE / "data.json", WORK / "data.json")

ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "HYBRAX_TRAIN_DEVICES": "1",
       "MPLBACKEND": "Agg"}

def hxt_cli(*args):
    proc = subprocess.run([sys.executable, "-m", "hybrax.train.cli", *args],
                          cwd=WORK, env=ENV, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout + proc.stderr

shutil.copy(EXAMPLE / "prepare-config.json", WORK / "prepare-config.json")
hxt_cli("prepare", "--config", "prepare-config.json",
         "--output-dir", "prepared", "--overwrite")
```

Everything below runs in `WORK`, printed at the end: inspect, copy or modify the real
files it produced.

## A Cheap First Check: `holdout_processes`

Before committing to N full trainings, hold out one process inside a single run. This
is Python-API-only (see [Cross-Validation](../train/loo.md#holdout-without-cross-validation)):

```{code-cell} ipython3
:tags: [remove-input]

import contextlib, io, warnings

import hybrax.format as hxf
from hybrax.train import TrainHarnessConfig, train_from_collection

collection = hxf.serialization.load_process_collection(str(WORK / "data.json"))

cfg = TrainHarnessConfig(
    epochs=300, seed=0, learning_rate=0.02,
    holdout_processes=("run_3",),
    checkpoint_dir=str(WORK / "holdout_check/checkpoints"),
    checkpoint_every=50,
)
with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
    warnings.simplefilter("ignore")
    result = train_from_collection(collection, config=cfg)

print(f"final train mean_loss   = {result.mean_loss_by_step[-1]:.4f}")
last_step = max(result.holdout_loss_by_step)
print(f"final holdout (run_3)   = {result.holdout_loss_by_step[last_step]:.4f}"
      f"  (label: {result.holdout_label})")
```

```{code-cell} ipython3
:tags: [remove-input]

import matplotlib.pyplot as plt

steps = sorted(result.holdout_loss_by_step)
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.semilogy(range(1, len(result.mean_loss_by_step) + 1), result.mean_loss_by_step,
           label="train (all 3 processes)")
ax.semilogy(steps, [result.holdout_loss_by_step[s] for s in steps], "o--",
           label="holdout (run_3)")
ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend()
fig.tight_layout()
```

Train and holdout track each other closely here: on this small, easy dataset, one
held-out run generalises about as well as the training set fits. That is a genuine
result worth confirming, and it is the first thing to check before trusting a model
further.

## Full Leave-One-Out

One fold per process, via the CLI. The config is a train config plus a `loo` section:

```{code-cell} ipython3
:tags: [remove-cell]

shutil.copy(EXAMPLE / "loo-config.json", WORK / "loo-config.json")
```

```json
{
  "loo": {
    "per_fold_holdout_sets": [
      {"test": ["run_1"]},
      {"test": ["run_2"]},
      {"test": ["run_3"]}
    ]
  }
}
```

Each entry is a `HoldoutSet`: `test` (required) is the held-out set for that fold;
`name` (optional) labels the fold's directory and summary row; `train` (optional) pins
the exact training set, otherwise every other process is used. With no
`per_fold_holdout_sets` at all, `hybrax.train` does the same thing automatically: one fold
per process.

```{code-cell} ipython3
:tags: [remove-input]

out = hxt_cli("loo", "--config", "loo-config.json", "--overwrite")
lines = [l for l in out.splitlines() if "LOO complete" in l]
print(lines[0] if lines else "LOO complete")
```

## What It Produced

```{code-cell} ipython3
:tags: [remove-input]

import pandas as pd

summary = pd.read_csv(WORK / "loo_run/loo_summary.csv")
folds = summary[summary["fold_idx"] != "mean"]
print(f"{'fold':10s} {'held out':10s} {'holdout loss':>13s} {'train loss':>11s}")
for _, r in folds.iterrows():
    print(f"{r['fold_slug']:10s} {r['test']:10s} "
          f"{r['holdout_total']:13.4f} {r['train_mean_total']:11.4f}")
```

```{code-cell} ipython3
:tags: [remove-input]

from IPython.display import Image
Image(filename=str(WORK / "loo_run/loo_loss_curves.png"))
```

Read the **spread across folds** as well as the mean: one fold noticeably worse than
the others usually means that process is different in a way the model does not capture,
which is more informative than an averaged score.

```{code-cell} ipython3
:tags: [remove-input]

print(f"run directory: ./{(WORK / 'loo_run').relative_to(WORK.parents[4])}")
for path in sorted((WORK / "loo_run").iterdir()):
    print(("  " if path.is_file() else "  ") + path.name + ("/" if path.is_dir() else ""))
```

Each `folds/<name>/` is a complete, ordinary run directory: `forward`, `model_load`,
plotting all work on it exactly like any other trained run.

## Gotchas

- **`per_fold_holdout_sets` entries must be objects.** `["run_1", "run_2"]`
  fails validation; `{"test": ["run_1", "run_2"]}` is the shape.
- **N folds means N trainings.** Get one fold converging well first (or use
  `holdout_processes` above) before multiplying everything, mistakes included.
- **`holdout_processes` needs a `checkpoint_dir`.** Holdout evaluation happens when a
  checkpoint is written; with no checkpoint directory configured, nothing is ever
  evaluated and `holdout_loss_by_step` stays empty.
- **Augmented data is fold-aware automatically.** A `parent_process` and all its
  synthetic children land in the same fold, never split across train/holdout. See
  [Augmentation](augmentation.md).
- **`--resume` re-runs only incomplete folds** and ignores config changes; a fresh run
  is what you want if you edited the config.

## See Also

The full example (configs, data) lives in `examples/gallery_loo/` at the repo root, no
docs build required. This page's own executed run is at
`./source/_data/out/runs/gallery_loo/`.

- [Cross-Validation](../train/loo.md): the full reference, including parallelism and
  resuming.
- [Forward](../train/forward.md): what to do with a trained fold.
- [Augmentation](augmentation.md): why LOO needs to be group-aware.
