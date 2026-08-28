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
# 3. Train a Model
<!-- UNLOCK -->

> Fit a hybrid ODE to the dataset using every default, and learn to
> read what came out.

The quickstart ran these commands. This tutorial explains them.

```{code-cell} ipython3
:tags: [remove-cell]

import os, shutil, subprocess, sys
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/tutorial_03").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
EXAMPLE = Path("../../../examples/tutorial_03_train").resolve()
shutil.copy(EXAMPLE / "data.json", WORK / "data.json")

ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "HYBRAX_TRAIN_DEVICES": "1",
       "MPLBACKEND": "Agg"}

def hxt_cli(*args):
    proc = subprocess.run([sys.executable, "-m", "hybrax.train.cli", *args],
                          cwd=WORK, env=ENV, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return (proc.stdout + proc.stderr).replace(str(WORK), ".")

def show(text, n=4, match=None):
    lines = [l for l in text.splitlines() if l.strip()]
    if match:
        lines = [l for l in lines if match in l]
    print("\n".join(lines[-n:]))
```

## What a hybrid model is here

The ODE `hybrax.train` solves has two halves:

```
d(state)/dt  =  biology(rates)          ← your model predicts this
              + transport(feeds, dilution, samples, volume)   ← hybrax.format wrote this
```

Training adjusts the first half so the integrated trajectory matches your measurements.
For `demo_batch` the second half is nearly empty (a batch run has no feeds) so
everything the model does is visible in three specific rates.

## 3.1 The `prepare` command

```{code-cell} ipython3
:tags: [remove-cell]

shutil.copy(EXAMPLE / "prepare-config.json", WORK / "prepare-config.json")
```

```json
{ "prepare": { "raw_input": "data.json" } }
```

```{code-cell} ipython3
:tags: [remove-input]

show(hxt_cli("prepare", "--config", "prepare-config.json",
              "--output-dir", "prepared", "--overwrite"), n=2)
```

`prepare` is a separate step for a reason: it is where the dataset stops being *data*
and becomes *a training problem*. It resolves which measurements are targets, fits the
control splines, fixes the state and control layout, and writes all of it to
`prepared/prepared.json`. Training reads only that.

Because it is separated, you can prepare once and train twenty models against an
identical, reproducible starting point.

```{code-cell} ipython3
:tags: [remove-input]

import json
prep = json.loads((WORK / "prepared/prepared.json").read_text())
print("top-level keys:", sorted(prep)[:8])
```

## 3.2 The `train` command

```json
{
  "data": { "prepared": "prepared" },
  "train": { "epochs": 300, "seed": 0 },
  "output": { "dir": "run" }
}
```

```{code-cell} ipython3
:tags: [remove-cell]

shutil.copy(EXAMPLE / "train-config.json", WORK / "train-config.json")
shutil.copy(EXAMPLE / "forward-config.json", WORK / "forward-config.json")
```

```{code-cell} ipython3
:tags: [remove-input]

show(hxt_cli("train", "--config", "train-config.json", "--overwrite"),
     n=1, match="training complete")
```

Everything not named in that config is a default, and each one is replaceable:

| Default | What it is | Replace it with |
|---|---|---|
| reaction module | A 2-layer MLP over the modeled state | [`build_reaction_module`](../train/reaction_module.md) |
| loss module | Per-target mean squared error | [`build_loss_module`](../train/loss_module.md) |
| scales | **All ones: i.e. no scaling** | [`estimate_all_scales`](../train/scaling.md) |
| optimizer | Adam, lr 1e-3, gradient clipping at norm 1000 | [`build_optimizer`](../train/train.md) |
| batching | Full batch, shuffled | `train.batch_size` |

## 3.3 Read the output

```{code-cell} ipython3
:tags: [remove-input]

print(f"./{(WORK / 'run').relative_to(WORK.parents[4])}")
```

Everything below lives in that directory, self-contained: `cp -r` it anywhere and
keep working from the copy.

The per-epoch record:

```{code-cell} ipython3
:tags: [remove-input]

import pandas as pd
metrics = pd.read_csv(WORK / "run/metrics.csv")
print("columns:", ", ".join(metrics.columns))
print()
for i in (0, len(metrics) // 2, len(metrics) - 1):
    print(metrics.iloc[i, :4].to_dict())
```

And the loss curve:

```{code-cell} ipython3
:tags: [remove-input]

from IPython.display import Image
Image(filename=str(WORK / "run/loss_curve.png"))
```

Then the fit itself: measurements against the integrated trajectory on the left,
the inferred rates on the right:

```{code-cell} ipython3
:tags: [remove-input]

hxt_cli("forward", "--config", "forward-config.json",
         "--output-dir", "run/forward", "--overwrite")
Image(filename=str(WORK / "run/forward/plots/run_1.png"))
```

### What to look at, in order

1. **Does the loss go down and stay down?** A curve that drops then explodes usually
   means the solve is struggling, not that the model is wrong: try tightening
   `solver.rtol`/`atol` or lowering the learning rate.
2. **Do the trajectories track the dots?** R² per target is printed in each panel.
3. **Are the rates physically plausible?** This is the check people skip. A model can fit
   concentrations beautifully with rates that are nonsense: growth and death both far
   too high, or uptake compensating for a transport error. The right-hand column is where
   that shows up.

:::{admonition} Gradient norms are worth a glance
:class: tip
`grad_norm_curve.png` shows the **raw** gradient norm, before clipping. If it sits
permanently at the clip threshold (`grad_clip_norm`, default 1000), your effective step
size is not what you think it is, and that is usually a scaling problem, which is
[Tutorial 4](04_your_first_custom_py.md).
:::

## 3.4 Second runs

`--overwrite` is required to reuse an output directory. You will hit this immediately on
your second run; it is deliberate, so a long training run cannot be silently destroyed.

```bash
hybrax train --config train-config.json --overwrite
hybrax train --config train-config.json --epochs 50    # flags beat the config file
```

## What you learned

- `prepare` and `train` are separate so training starts from a reproducible artifact.
- The run directory is self-contained and records exactly what was run.
- Every part of the model is a default you can replace, one hook at a time.
- Judge a fit by its **rates**, not only by its trajectories.

## What's next

You can run the full tutorial at `examples/tutorial_03_train/run.py`.

- **[Tutorial 4](04_your_first_custom_py.md)**: replace the two most important defaults
  and measure the difference.
- Config in full: [Configuration](../train/config.md).
- What `prepare` does in detail: [Prepare](../train/prepare.md).
