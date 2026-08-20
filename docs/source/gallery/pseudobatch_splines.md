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

# Pseudobatch splines through a jump

> **Demonstrates.** Recovering a smooth concentration curve from just 5 noisy
> measurements straddling a discrete feed jump, checked against a known ground truth.
> bp-format only: no reaction module, no training.

The other gallery entries fit a model. This one fits a curve: it exercises
[the pseudobatch transform](../format/pseudobatch_transform.md) and
[spline fitting](../format/time_series_and_splines.md) on their own, with nothing else
in the way.

```{code-cell} ipython3
:tags: [remove-cell]

import shutil
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_pseudobatch_splines").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
shutil.copy(Path("../_data/out/demo_spline_jump/data.json").resolve(), WORK / "data.json")
```

## The setup

One species (`solute`), first-order decay, and one feed bolus part-way through that
jumps mass and volume together. Both phases are closed-form exponential decay at
constant volume, so the dense curve below is the exact ground truth, not a numerical
approximation of it:

```{literalinclude} ../_data/generate.py
:language: python
:start-at: SJ_K = 0.15
:end-before: def build_demo_spline_jump
```

```{code-cell} ipython3
import numpy as np
import bp_format as bp
import sys
sys.path.insert(0, "../_data")
from generate import spline_jump_truth, SJ_T_JUMP, SJ_T_END

dense_t = np.linspace(0.0, SJ_T_END, 400)
truth = spline_jump_truth(dense_t)

collection = bp.serialization.load_process_collection(WORK / "data.json")
process = collection.processes["run_1"]

solute = process.reactor_medium.components["solute"]
print("measurement times :", np.asarray(solute.concentration.times))
print("measured values   :", np.round(np.asarray(solute.concentration.values), 3))
```

Two measurements land inside 1 h of the jump on either side (9 h and 11 h), and the raw
value nearly quintuples between them: that jump is the feed bolus adding mass, not the
solute suddenly reappearing.

## Fit and backtransform

```{code-cell} ipython3
from bp_format.splines import build_pseudobatch_transform, build_backtransform_spline

bundle = build_pseudobatch_transform(process)
process.pseudobatch_transform = bundle

back = build_backtransform_spline(process, "solute")
recovered = np.asarray(back(dense_t))
```

`build_pseudobatch_transform` removes the physical jump before fitting a spline to it;
`build_backtransform_spline` then wraps that fit so evaluating it returns real
concentration, not the intermediate `c*`. The result is a single spline, valid on both
sides of the jump, that `back` maps to real concentration wherever you evaluate it.

## Recovered curve versus ground truth

```{code-cell} ipython3
:tags: [remove-input]

import matplotlib.pyplot as plt

meas_t = np.asarray(solute.concentration.times)
meas_v = np.asarray(solute.concentration.values)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(dense_t, truth, color="black", lw=1.5, label="ground truth")
ax.plot(dense_t, recovered, color="tab:blue", lw=1.5, ls="--",
        label="recovered (fit + backtransform)")
ax.scatter(meas_t, meas_v, color="tab:red", zorder=5, label="5 measurements")
ax.axvline(SJ_T_JUMP, color="gray", lw=0.8, ls=":")
ax.set_xlabel("t (h)"); ax.set_ylabel("solute (g/L)"); ax.legend()
fig.tight_layout()
fig.savefig(WORK / "recovery.png", dpi=110)
plt.close(fig)
```

```{code-cell} ipython3
:tags: [remove-input]

from IPython.display import Image
Image(filename=str(WORK / "recovery.png"))
```

```{code-cell} ipython3
# Error only where a segment actually has data on both sides. Evaluating a
# 2-point segment beyond its own last measurement is extrapolation, not recovery.
pre = np.linspace(0.0, SJ_T_JUMP - 1.0, 200)
post = np.linspace(SJ_T_JUMP + 1.0, SJ_T_END, 200)
for label, grid in [("pre-jump [0, 9]", pre), ("post-jump [11, 17]", post)]:
    rec = np.asarray(back(grid))
    rel = np.abs(rec - spline_jump_truth(grid)) / spline_jump_truth(grid)
    print(f"{label}: max relative error {rel.max() * 100:4.1f}%   "
          f"mean {rel.mean() * 100:4.1f}%")
```

A real, honest fit: within roughly 14% of the true curve at its worst, both before and
after the jump, from 5 points total. The dashed curve tracks the true decay's shape
closely, including its curvature, not just its two endpoints.

:::{admonition} Why the post-jump segment is the weaker fit
:class: note
The pre-jump segment has 3 measurements (0, 4, 9 h); the post-jump segment has only 2
(11, 17 h). `fit_timeseries_spline` needs at least 4 points per segment for its smoothing
B-spline path, so both segments here fall back to natural cubic interpolation, and 2
points pin a natural cubic down to very nearly a straight line. That line still tracks
an honestly-curved exponential decay to within ~14% here, but it is not a coincidence
that the weaker half is the 2-point one.
:::

## What made this example different

- **No reaction module, no `custom.py`, no training.** The pseudobatch transform and
  spline fitting are bp-format's own, and stop being useful to demonstrate the moment
  bp-train enters the picture.
- **A closed-form ground truth.** Every other demo dataset in this site is simulated
  with RK4 and compared to noisy measurements of itself. This one has an exact answer to
  check the fit against, because that is the whole point of the page.
- **5 points is not a comfortable number.** One of the two segments this jump creates
  is forced into the 2-point, near-linear fallback. That the fit still tracks a real,
  curved decay to ~14% is the actual demonstration.

## See also

Run the example yourself at `./source/_data/out/runs/gallery_pseudobatch_splines/`.

- [The pseudobatch transform](../format/pseudobatch_transform.md): what
  `build_pseudobatch_transform` and `build_backtransform_spline` actually compute.
- [Time series and splines](../format/time_series_and_splines.md): the segmentation and
  spline-fitting machinery underneath.
- [Gallery: fed-batch](fed_batch.md): the same transform on a full, trained fed-batch
  model.
