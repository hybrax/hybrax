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

# Quickstart

> **In one sentence.** Three commands, two small config files, no Python — train a
> hybrid model on a dataset that ships with these docs and look at what came out.
>
> **You need this if** you have never run bp-train. **You can skip it if** you already
> have a run directory you understand.

This takes about ten minutes, and you write no code. The point is to see the whole loop
once, end to end, before learning any of the parts. Loading *your own* data comes next,
in [Tutorial 1](../tutorials/01_your_first_dataset.md).

```{code-cell} ipython3
:tags: [remove-cell]

import os, shutil, subprocess, sys, textwrap
from pathlib import Path

DATA = Path("../_data/out/demo_batch").resolve()
WORK = Path("../_data/out/runs/quickstart").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
shutil.copy(DATA / "data.json", WORK / "data.json")

ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "BP_TRAIN_DEVICES": "1",
       "MPLBACKEND": "Agg"}

def bp_train(*args):
    """Run the bp-train CLI in WORK and return its combined output."""
    proc = subprocess.run([sys.executable, "-m", "bp_train.cli", *args],
                          cwd=WORK, env=ENV, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout + proc.stderr

def tail(text, n=6, match=None):
    lines = [l for l in text.splitlines() if l.strip()]
    if match:
        lines = [l for l in lines if match in l]
    print("\n".join(lines[-n:]))
```

## 0. The dataset

These docs ship a simulated case study, `demo_batch`: **three batch runs** of *E. coli*
on glucose, each with offline measurements of biomass, glucose and product roughly every
hour for 14 hours. No feeds, no boluses, no sampling volume — the simplest thing that is
still a real bioprocess.

It is one file:

```{code-cell} ipython3
:tags: [remove-input]

import json
raw = json.loads((WORK / "data.json").read_text())
print("case_id :", raw["case_id"])
print("organism:", raw["organism"])
print("runs    :", list(raw["processes"]))
p = raw["processes"]["run_1"]
print("run_1 measures:", list(p["reactor_medium"]["components"]))
print("run_1 volume   :", p["volume"]["initial_volume"], p["volume"]["unit"],
      "— no volume changes")
```

Everything below assumes that file is in your working directory.

## 1. Write two config files

bp-train is driven by JSON config files. They are small. This is the whole prepare
config:

```json
{ "prepare": { "raw_input": "data.json" } }
```

and this is the whole train config:

```json
{
  "data": { "prepared": "prepared" },
  "train": { "epochs": 300, "seed": 0 },
  "output": { "dir": "run" }
}
```

Two things to know now, and no more:

- **Paths in a config are relative to the config file**, not to your shell's working
  directory.
- **Unknown keys are rejected.** A typo is a hard error, never a silently ignored
  setting.

```{code-cell} ipython3
:tags: [remove-cell]

(WORK / "prepare-config.json").write_text(
    '{ "prepare": { "raw_input": "data.json" } }\n')
(WORK / "train-config.json").write_text(textwrap.dedent("""\
    {
      "data": { "prepared": "prepared" },
      "train": { "epochs": 300, "seed": 0 },
      "output": { "dir": "run" }
    }
    """))
(WORK / "forward-config.json").write_text('{ "models": ["run"] }\n')
```

## 2. Prepare

```bash
bp-train prepare --config prepare-config.json --output-dir prepared
```

```{code-cell} ipython3
:tags: [remove-input]

tail(bp_train("prepare", "--config", "prepare-config.json",
              "--output-dir", "prepared", "--overwrite"), n=3)
```

`prepare` reads your bp-format file and writes a **prepared artifact** — the dataset
plus everything derived from it that training needs: the control splines, the state and
control layout, which measured quantities are the fit targets. Training never touches
the raw file again, so a prepared artifact is a reproducible starting point.

It also drops diagnostic plots in `prepared/prepare_diagnostics/` showing how it read
each run's controls. Worth a look the first time you prepare your own data.

## 3. Train

```bash
bp-train train --config train-config.json
```

```{code-cell} ipython3
:tags: [remove-input]

out = bp_train("train", "--config", "train-config.json", "--overwrite")
tail(out, n=1, match="training complete")
```

That is a hybrid ODE model: bp-format supplies the mass balance, and a small neural
network — the default reaction module — supplies the three specific rates
`q_biomass`, `q_glucose`, `q_product`. You did not choose the network, the loss, or the
optimizer; every one of those is a default you can replace later.

## 4. Forward

```bash
bp-train forward --config forward-config.json
```

with `forward-config.json` being just `{ "models": ["run"] }`.

```{code-cell} ipython3
:tags: [remove-input]

bp_train("forward", "--config", "forward-config.json",
         "--output-dir", "run/forward", "--overwrite")
print((WORK / "run/forward/losses.csv").read_text())
```

`forward` re-simulates each run with the trained model and writes a dense trajectory to
`predictions.csv` plus these per-process, per-target losses. Training gives you a number;
forward gives you something you can plot and hand to a colleague.

## 5. Look at it

```{code-cell} ipython3
:tags: [remove-input]

from IPython.display import Image
Image(filename=str(WORK / "run/forward/run_1.png"))
```

Left column: measurements (dots) against the integrated trajectory (line), with R² per
target. Right column: the specific rates the network learned — these were never measured,
they are what the model inferred.

That right column is the payoff. The data was simulated with a maximum specific growth
rate of **0.45 1/h**, and nobody told the model that:

```{code-cell} ipython3
:tags: [remove-input]

import csv
with (WORK / "run/forward/predictions.csv").open() as fh:
    first = next(r for r in csv.DictReader(fh) if r["process"] == "run_1")
print(f"learned q_biomass at t=0 : {float(first['q_biomass']):.3f} 1/h")
print( "true    mu_max           : 0.450 1/h")
```

## 6. What you got

```{code-cell} ipython3
:tags: [remove-input]

for path in sorted((WORK / "run").rglob("*")):
    if "checkpoints" in path.parts:
        continue
    rel = path.relative_to(WORK / "run")
    print(("  " * (len(rel.parts) - 1)) + rel.name + ("/" if path.is_dir() else ""))
```

| File | What it is |
|---|---|
| `config.json`, `custom.py` | Exactly what was run. Every run directory is self-contained. |
| `metrics.csv` | Per-epoch loss and gradient norm. |
| `loss_curve.png`, `grad_norm_curve.png` | The same, plotted. |
| `model/params.eqx` | The trained parameters — only the trainable ones. |
| `predictions.csv` | Dense trajectories and rates, at the end of training. |
| `<run>.png` | One panel figure per process. |
| `forward/` | The output of step 4. |

## What to do next

:::{admonition} You have not seen any of the interesting parts yet
:class: tip

This run used every default: a generic MLP for the rates, mean-squared error for the
loss, **and no scaling at all**. It fits this small, well-behaved dataset anyway. On real
data the scaling in particular matters a great deal — see
[Tutorial 4](../tutorials/04_your_first_custom_py.md).
:::

- **Get your own data in** → [Tutorial 1](../tutorials/01_your_first_dataset.md).
- **Understand the words used above** → [Concepts and vocabulary](concepts.md).
- **Replace the model** → [Tutorial 4](../tutorials/04_your_first_custom_py.md), then
  [the reaction module](../train/reaction_module.md).
- **Fed-batch, boluses, sampling** → [Gallery: fed-batch](../gallery/fed_batch.md).
