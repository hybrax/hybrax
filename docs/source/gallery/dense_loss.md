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

# Custom losses on the dense grid

> **Demonstrates.** A loss module that constrains the trajectory *between*
> measurements (bounds on states and rates, and a smoothness penalty on rate
> curvature) using bp-format's own `Bounds` metadata and bp-train's jump-aware
> dense-grid helpers.

By default, a loss only ever looks at measurement times. Between them, the model is free
to do anything that reproduces the endpoints: including going negative, or oscillating
wildly. The [dense grid](../train/loss_module.md#the-dense-grid) exists to close that
gap: opt in, and the trainer also saves on a uniform time linspace, which the loss can
then penalise.

This example adds three terms on top of [Tutorial 4](../tutorials/04_your_first_custom_py.md)'s
reaction module and scales, none of which change the fit: they change what the model is
*allowed* to do while fitting.

The walkthrough below shows the file in pieces, next to the reasoning for each one. For
the whole thing at once: to copy, diff against your own, or just read top to bottom:

:::{dropdown} Full `custom.py`
```{literalinclude} _files/dense_loss_custom.py
:language: python
:linenos:
```
:::

```{code-cell} ipython3
:tags: [remove-cell]

import os, shutil, subprocess, sys, textwrap
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_dense_loss").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
shutil.copy(Path("../_data/out/demo_batch/data.json").resolve(), WORK / "data.json")
shutil.copy(Path("_files/dense_loss_custom.py").resolve(), WORK / "custom.py")

ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "BP_TRAIN_DEVICES": "1",
       "MPLBACKEND": "Agg"}

def bp_train(*args):
    proc = subprocess.run([sys.executable, "-m", "bp_train.cli", *args],
                          cwd=WORK, env=ENV, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout + proc.stderr

(WORK / "prepare-config.json").write_text(
    '{ "prepare": { "raw_input": "data.json" } }\n')
(WORK / "train-full.json").write_text(textwrap.dedent("""\
    {
      "data": { "prepared": "prepared" },
      "custom_py": "custom.py",
      "train": { "epochs": 800, "seed": 0, "learning_rate": 0.01 },
      "output": { "dir": "run_full" }
    }
    """))

import numpy as np
import bp_format as bp
import csv

_case_study = bp.serialization.load_case_study(WORK / "data.json")

def r2_by_target(run_dir):
    rows_by_process = {}
    with (WORK / run_dir / "predictions.csv").open() as fh:
        for row in csv.DictReader(fh):
            rows_by_process.setdefault(row["process"], []).append(row)
    per_target = {}
    for name, process in _case_study.processes.items():
        rows = rows_by_process[name]
        t_pred = np.array([float(r["t"]) for r in rows])
        for species in ("biomass", "glucose", "product"):
            comp = process.reactor_medium.components[species].concentration
            t_meas, y_meas = np.asarray(comp.times), np.asarray(comp.values)
            y_pred = np.interp(t_meas, t_pred,
                               np.array([float(r[f"c_{species}"]) for r in rows]))
            ss_res = np.sum((y_meas - y_pred) ** 2)
            ss_tot = np.sum((y_meas - y_meas.mean()) ** 2)
            per_target.setdefault(species, []).append(1 - ss_res / ss_tot)
    return {k: float(np.mean(v)) for k, v in per_target.items()}

def dense_diagnostics(run_dir):
    """Worst glucose excursion and RMS curvature per rate, in physical space."""
    rows_by_process = {}
    with (WORK / run_dir / "predictions.csv").open() as fh:
        for row in csv.DictReader(fh):
            rows_by_process.setdefault(row["process"], []).append(row)
    min_glucose = float("inf")
    curvature = {"q_biomass": [], "q_glucose": []}
    for rows in rows_by_process.values():
        t = np.array([float(r["t"]) for r in rows])
        min_glucose = min(min_glucose, min(float(r["c_glucose"]) for r in rows))
        dt = t[1] - t[0]
        for rate in curvature:
            y = np.array([float(r[rate]) for r in rows])
            d2 = (y[2:] - 2 * y[1:-1] + y[:-2]) / dt**2
            curvature[rate].append(float(np.sqrt(np.mean(d2**2))))
    return min_glucose, {k: float(np.mean(v)) for k, v in curvature.items()}
```

## 1. Bounds, from the data itself

`ReactorMediumComponent.bounds` is metadata bp-format already carries: `demo_batch`
declares `(0.0, None)` on every species, because a concentration cannot be negative. But
`BiologicalOde.rates` bounds default to unbounded, since auto-generation has no way to
know what a plausible rate range is. Attaching them is one `transform_process_collection`
hook:

```{literalinclude} _files/dense_loss_custom.py
:language: python
:linenos:
:lines: 115-122
```

This is exactly the use bp-format's docs describe for `Bounds`: *"pure metadata (not
enforced inside RhsOde or the integrator; downstream consumers read them off the process
to build soft-constraint penalties."* Nothing threads these bounds into bp-train
automatically) the loss module below reads them itself, once, at construction:

```{literalinclude} _files/dense_loss_custom.py
:language: python
:linenos:
:lines: 195-216
```

## 2. The hinge penalty, on the dense grid

```{literalinclude} _files/dense_loss_custom.py
:language: python
:linenos:
:lines: 160-174
```

`-inf`/`+inf` for an unbounded side falls straight out of the `clip`, so there is no
branching on which bounds are set. The penalty is evaluated on `dense_RAW_states` and
`dense_RAW_modeled_BiologicalOde_rates` (**RAW**, because "negative" and "above 1.0 1/h"
only mean something in physical units) and masked by `dense_valid_time`, the dense
grid's own post-solver-failure mask.

## 3. Smoothness, without penalising real jumps

```{literalinclude} _files/dense_loss_custom.py
:language: python
:linenos:
:lines: 176-186
```

The curvature is a plain central second difference; `dense_t` is a uniform linspace, so a
single `dt` is valid across the whole grid. The only non-obvious part is the mask:

```python
triple_mask = all_triple(valid) & dense_triple_mask_away_from_jumps(
    inputs.dense_t, inputs.jump_ts, jump_epsilon_h=2.0 * dt)
```

A bolus creates a real, physical kink in concentration (and therefore in the inferred
rate. Penalising curvature there would fight the data. `dense_triple_mask_away_from_jumps`
is shipped by bp-train for exactly this: it excludes any three-point window whose span
straddles a jump time, so smoothness is only asked for where the process is actually
smooth. `demo_batch` has no events, so this mask is a no-op here) see
[Fed-batch](fed_batch.md) for where it matters.

## Training

```{code-cell} ipython3
:tags: [remove-input]

bp_train("prepare", "--config", "prepare-config.json",
         "--output-dir", "prepared", "--overwrite")
out = bp_train("train", "--config", "train-full.json", "--overwrite")
print([l for l in out.splitlines() if "training complete" in l][0])
print(f"run directory: ./{(WORK / 'run_full').relative_to(WORK.parents[4])}")
```

## Did it cost anything?

```{code-cell} ipython3
:tags: [remove-input]

r2 = r2_by_target("run_full")
min_glucose, curvature = dense_diagnostics("run_full")

print(f"{'target':10s} {'R2':>8s}")
for name, value in r2.items():
    print(f"{name:10s} {value:8.4f}")
print()
print(f"min predicted glucose : {min_glucose:+.4f} g/L  (bound: >= 0)")
print(f"curvature(q_biomass)  : {curvature['q_biomass']:.4f}")
print(f"curvature(q_glucose)  : {curvature['q_glucose']:.4f}")
```

Compare against [Tutorial 4](../tutorials/04_your_first_custom_py.md)'s plain fit on the
same data, same epochs, same learning rate: everything below used the *unconstrained*
loss:

```{code-cell} ipython3
:tags: [remove-input]

shutil.copy(Path("../_data/out/demo_batch/data.json").resolve(), WORK / "data_base.json")
(WORK / "base.py").write_text(
    (Path("../tutorials/_files/tutorial_04_custom.py")).read_text())
(WORK / "train-base.json").write_text(textwrap.dedent("""\
    {
      "data": { "prepared": "prepared" },
      "custom_py": "base.py",
      "train": { "epochs": 800, "seed": 0, "learning_rate": 0.01 },
      "output": { "dir": "run_base" }
    }
    """))
bp_train("train", "--config", "train-base.json", "--overwrite", "--no-plot")
print(f"comparison run directory: ./{(WORK / 'run_base').relative_to(WORK.parents[4])}")

r2_base = r2_by_target("run_base")
min_glucose_base, curvature_base = dense_diagnostics("run_base")

print(f"{'target':10s} {'no penalty':>12s} {'with penalty':>14s}")
for name in ("biomass", "glucose", "product"):
    print(f"{name:10s} {r2_base[name]:12.4f} {r2[name]:14.4f}")
print()
print(f"{'min glucose':10s} {min_glucose_base:12.4f} {min_glucose:14.4f}")
print(f"{'curv(q_X)':10s} {curvature_base['q_biomass']:12.4f} {curvature['q_biomass']:14.4f}")
print(f"{'curv(q_S)':10s} {curvature_base['q_glucose']:12.4f} {curvature['q_glucose']:14.4f}")
```

Fit quality is essentially unchanged: every R² moves by less than a percentage point.
What changed is what happens *between* measurements: the worst glucose excursion goes
from clearly negative to essentially zero, and both rate trajectories are visibly less
wiggly. This is the actual case for dense-grid penalties: they are close to free when the
constraint is compatible with the data, and they catch exactly the failure mode that
measurement-only losses cannot see.

```{code-cell} ipython3
:tags: [remove-input]

from IPython.display import Image
Image(filename=str(WORK / "run_full/run_1.png"))
```

## See also

Run the example yourself at `./source/_data/out/runs/gallery_dense_loss/`.

- [The loss module](../train/loss_module.md#the-dense-grid): `dense_grid_n` and every
  `dense_*` field.
- [Tutorial 4](../tutorials/04_your_first_custom_py.md): the reaction module and scales
  this builds on.
- [Fed-batch](fed_batch.md): a process with real jumps, where the away-from-jumps
  masking actually does something.
