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

<!-- LOCK -->
# 5. Predict Processes
<!-- UNLOCK -->

> Use a trained model: from the command line for exports, and from Python when you want
> the arrays.
>
> Last tutorial. After this, the [gallery](../gallery/index.md).

```{code-cell} ipython3
:tags: [remove-cell]

import os, shutil, subprocess, sys
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/tutorial_05").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
EXAMPLE = Path("../../../examples/tutorial_05_predict").resolve()
shutil.copy(EXAMPLE / "data.json", WORK / "data.json")
shutil.copy(EXAMPLE / "custom.py", WORK / "custom.py")

ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "HYBRAX_TRAIN_DEVICES": "1",
       "MPLBACKEND": "Agg"}

def hxt_cli(*args):
    proc = subprocess.run([sys.executable, "-m", "hybrax.train.cli", *args],
                          cwd=WORK, env=ENV, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout + proc.stderr

shutil.copy(EXAMPLE / "prepare-config.json", WORK / "prepare-config.json")
shutil.copy(EXAMPLE / "train-config.json", WORK / "train-config.json")
shutil.copy(EXAMPLE / "forward-config.json", WORK / "forward-config.json")
hxt_cli("prepare", "--config", "prepare-config.json",
             "--output-dir", "prepared", "--overwrite")
hxt_cli("train", "--config", "train-config.json", "--overwrite")
```

We start from a run directory trained exactly as in [Tutorial 4](04_your_first_custom_py.md).

```{code-cell} ipython3
:tags: [remove-input]

print(f"./{(WORK / 'run').relative_to(WORK.parents[4])}")
```

Self-contained: `cp -r` it anywhere and keep working from the copy.

## The command-line way: `forward`

```bash
hybrax forward --config forward-config.json
```

where the config names the run directories to use, and requests predictions and plots
explicitly (both default to off):

```json
{ "models": ["run"], "output": { "predictions": "parents", "plots": true } }
```

```{code-cell} ipython3
:tags: [remove-input]

hxt_cli("forward", "--config", "forward-config.json",
             "--output-dir", "run/forward", "--overwrite")
print((WORK / "run/forward/forward-results/losses.csv").read_text())
```

`forward` re-solves each process with the trained model on a dense time grid and writes
`forward-results/predictions.csv`, `forward-results/losses.csv`, and (with `plots: true`)
one figure per process under `forward-results/plots/`.

:::{admonition} Why `forward` exists separately from `train`
:class: note
Training reports a loss. `forward` reports a *trajectory*: dense states, the inferred
rates, and the real volume: the things you plot, hand to a colleague, or compare
against another model. It also re-solves from scratch, so it is an honest check that the
saved model reproduces what training claimed.
:::

```{code-cell} ipython3
:tags: [remove-input]

import pandas as pd
df = pd.read_csv(WORK / "run/forward/forward-results/predictions.csv")
print("columns:", ", ".join(df.columns))
print("rows   :", len(df))
```

`c_*` are concentrations, `q_*` are the inferred specific rates, `V_real` is the physical
volume. One row per process per grid point.

## The Python way: `model_load` and `model_predict`

When you want arrays rather than CSVs:

```{code-cell} ipython3
import hybrax.format as hxf
import hybrax.train as hxt

wrapper, config = hxt.model_load(str(WORK / "run"))
type(wrapper).__name__
```

`model_load` rebuilds the whole model: only the trainable parameters were saved, and
everything else (the controls, the assembled ODE, the scales) is reconstructed from
the `prepared.json` and `custom.py` bundled inside the run directory. That is why a run
directory is self-contained and why you can move it between machines.

```{code-cell} ipython3
collection = hxf.serialization.load_process_collection(WORK / "data.json")

predictions = hxt.model_predict(wrapper, config, collection, grid_n=200)
list(predictions)
```

Each entry is a `DenseProcessExport`:

```{code-cell} ipython3
export = predictions["run_1"]
print("t          ", export.t.shape)
print("c_species  ", export.c_species.shape, "  states, in the ProcessOrdering layout")
print("q_rates    ", export.q_rates.shape,   "  the inferred rates")
print("v_real     ", export.v_real.shape)
```

Which lets you do your own analysis: for example, checking a rate against the value the
data was generated with:

```{code-cell} ipython3
import json
import numpy as np

truth = json.loads(Path("../../../examples/tutorial_05_predict/ground_truth.json").read_text())
learned_mu = float(export.q_rates[0, 0])     # q_biomass at t = 0
print(f"learned q_biomass(t=0) : {learned_mu:.3f} 1/h")
print(f"true mu_max            : {truth['mu_max']:.3f} 1/h")
```

And to plot it yourself. `run_1` is one of the three processes the model was **trained**
on: this is checking the fit, not testing generalisation to new data (that is what
[cross-validation](../train/loo.md) is for):

```{code-cell} ipython3
import matplotlib.pyplot as plt

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
axes[0].legend();
```

## One thing to be careful about

There are two ways to load a model, and they are not interchangeable:

| | Rebuilds the static half from | Safe to use on |
|---|---|---|
| `model_load(run_dir)` | the run directory's own bundled data | anything |
| `model_reload(...)` | **whatever you pass it** | only the data it was trained on |

`model_reload` keeps the static half (including the scales) from the object you hand
it. Point it at a different dataset and it will load the weights into a *different scaled
space*, then predict confidently and wrongly. No exception, no NaN. See
[Silent failures](../troubleshooting/silent_failures.md).

## What you learned

- `forward` gives you dense trajectories, per-target losses and plots.
- `model_load` + `model_predict` give you arrays straight from a loaded
  `BioProcessCollection`, no extra wrapping needed.
- Only trainable parameters are saved; the rest is rebuilt, which is what makes run
  directories portable.

## What's next

Run the tutorial yourself at `./source/_data/out/runs/tutorial_05/`.

You have now seen the whole loop. Everything the tutorials deliberately left out is in
the gallery, each as a self-contained example:

- [Fed-batch: feeds, boluses and samples](../gallery/fed_batch.md)
- [Mechanistic models](../gallery/mechanistic_rates.md): real kinetics, partially trained
- [Custom losses on the dense grid](../gallery/dense_loss.md)
- [Cross-Validation](../train/loo.md): how well does this generalise to a run it never saw?
- [Ensembles](../train/forward.md#ensembles): averaging several trained models for a
  cheap uncertainty estimate
