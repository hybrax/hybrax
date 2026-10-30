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

# Pseudobatch Splines Through a Jump

> Recovering a smooth concentration curve from just 5 noisy measurements straddling a
> discrete feed jump, checked against a known ground truth. Uses hybrax.format only,
> with no reaction module or training involved.

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
shutil.copy(Path("../../../examples/gallery_pseudobatch_splines/data.json").resolve(), WORK / "data.json")
```

## The Setup

One species (`solute`), first-order decay, and one feed bolus part-way through that
jumps mass and volume together. Both phases are closed-form exponential decay at
constant volume, so the dense curve below is the exact ground truth:

```{literalinclude} ../_data/generate.py
:language: python
:start-at: SJ_K = 0.15
:end-before: def build_demo_spline_jump
```

```{code-cell} ipython3
import numpy as np
import hybrax.format as hxf
import sys
sys.path.insert(0, "../_data")
from generate import spline_jump_truth, SJ_T_JUMP, SJ_T_END

dense_t = np.linspace(0.0, SJ_T_END, 400)
truth = spline_jump_truth(dense_t)

collection = hxf.serialization.load_process_collection(WORK / "data.json")
process = collection.processes["run_1"]

solute = process.reactor_medium.components["solute"]
print("measurement times :", np.asarray(solute.concentration.times))
print("measured values   :", np.round(np.asarray(solute.concentration.values), 3))
```

Two measurements land inside 1 h of the jump on either side (9 h and 11 h), and the raw
value nearly quintuples between them: that jump is the feed bolus adding mass.

## Fit and Backtransform

```{code-cell} ipython3
from hybrax.format.splines import build_pseudobatch_transform, build_backtransform_spline

bundle = build_pseudobatch_transform(process)
process.pseudobatch_transform = bundle

back = build_backtransform_spline(process, "solute")
recovered = np.asarray(back(dense_t))
```

`build_pseudobatch_transform` removes the physical jump before fitting a spline to it;
`build_backtransform_spline` then wraps that fit so evaluating it returns real
concentration directly, skipping the intermediate `pseudobatch_concentration`. The
result is a single spline, valid on both sides of the jump, that `back` maps to real
concentration wherever you evaluate it.

## Recovered Curve Versus Ground Truth

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
# Compare the intervals away from the jump and the full plotted grid.
pre = np.linspace(0.0, SJ_T_JUMP - 1.0, 200)
post = np.linspace(SJ_T_JUMP + 1.0, SJ_T_END, 200)
for label, grid in [
    ("pre-jump [0, 9]", pre),
    ("post-jump [11, 17]", post),
    ("full plotted grid [0, 17]", dense_t),
]:
    rec = np.asarray(back(grid))
    rel = np.abs(rec - spline_jump_truth(grid)) / spline_jump_truth(grid)
    print(f"{label}: max relative error {rel.max() * 100:4.1f}%   "
          f"mean {rel.mean() * 100:4.1f}%")
```

Maximum relative error is about 14% on `[0, 9]` and `[11, 17]`. On the full plotted
grid it reaches about 22% near the jump.

:::{admonition} One fit through all five transformed measurements
:class: note
The transform fits one cubic B-spline through all five transformed measurements,
with zero smoothing. The backtransform restores the physical bolus jump using the
ADF and feed correction. Evaluating between 9 h and 11 h is interpolation within
this fit, but measurements are sparse near the jump, so accuracy there must be
checked separately.
:::

## What Made This Example Different

- **No reaction module, no `custom.py`, no training.** The pseudobatch transform and
  spline fitting are hybrax.format's own, and stop being useful to demonstrate the moment
  `hybrax.train` enters the picture.
- **A closed-form ground truth.** It lets us check interpolation error between the
  sparse measurements, including near the bolus.

## See Also

The full example (configs, data) lives in `examples/gallery_pseudobatch_splines/` at the
repo root, no docs build required. This page's own executed run is at
`./docs/source/_data/out/runs/gallery_pseudobatch_splines/`.

- [The pseudobatch transform](../format/pseudobatch_transform.md): what
  `build_pseudobatch_transform` and `build_backtransform_spline` actually compute.
- [Time series and splines](../format/time_series_and_splines.md): the segmentation and
  spline-fitting machinery underneath.
