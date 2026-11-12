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

# FBA-Hyb

> A hybrid dynamic-FBA reaction module: two small neural networks predict a
> glucose-uptake rate and combined-demand coefficients from the current state.
> A frozen, offline-fit surrogate converts those into real metabolic rates.
> No linear program ever gets solved during training.

Dynamic flux-balance analysis (dFBA) normally means solving a real linear program,
`max c^T v` subject to `S.v = 0` and flux bounds, at every integration step. JAX cannot
differentiate through or `vmap`/`jit` a linear-program solve the way it does a neural
network, and `hybrax.train` has no LP solver in its training loop. The fix, from Gotsmy &
Guillén-Gosálbez's FBA-Hyb [<a href="#ref-fbahyb">1</a>]: fit a closed-form,
pole-free surrogate of the FBA solution **once, offline** (10,000 real linear
programs solved outside the training loop), then embed that surrogate directly
inside an ordinary `RateModule`. No LP solve ever happens during training.

The surrogate here is fit fresh against `e_coli_core.xml`
[<a href="#ref-ecolicore">2</a>], a small, real, published teaching model (95
reactions, 72 metabolites): see [How the Surrogate Was Fit](#how-the-surrogate-was-fit)
below for the offline fitting workflow. See [PLS-dFBA](pls_dfba.md) for this page's
sibling, which builds on the same surrogate with a product-forming, media-blend-aware
extension.

The walkthrough below shows the file in pieces, next to the reasoning for each one.

```{code-cell} ipython3
:tags: [remove-cell]

import os, shutil, subprocess, sys
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_fba_hyb").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
EXAMPLE = Path("../../../examples/gallery_fba_hyb").resolve()
shutil.copy(EXAMPLE / "data.json", WORK / "data.json")
shutil.copy(EXAMPLE / "custom.py", WORK / "custom.py")

ENV = {**os.environ, "JAX_PLATFORMS": "cpu",
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hybrax.format as hxf
import hybrax.train as hxt

_collection = hxf.serialization.load_process_collection(WORK / "data.json")

def r2_by_target(run_dir):
    """Pooled R2: concatenate every process's residuals/variance before
    dividing, so one narrow-range process can't make an otherwise-good fit
    look catastrophic."""
    df = pd.read_csv(WORK / run_dir / "predictions.csv")
    per_target = {s: ([], []) for s in ("biomass", "glucose", "acetate")}
    for name, process in _collection.processes.items():
        proc_df = df[df["process"] == name]
        t_pred = proc_df["t"].to_numpy()
        for species in per_target:
            comp = process.reactor_medium.components[species].concentration
            t_meas, y_meas = np.asarray(comp.times), np.asarray(comp.values)
            y_pred = np.interp(t_meas, t_pred, proc_df[f"c_{species}"].to_numpy())
            per_target[species][0].append(y_meas)
            per_target[species][1].append(y_pred)
    out = {}
    for species, (meas_list, pred_list) in per_target.items():
        y_meas, y_pred = np.concatenate(meas_list), np.concatenate(pred_list)
        ss_res = np.sum((y_meas - y_pred) ** 2)
        ss_tot = np.sum((y_meas - y_meas.mean()) ** 2)
        out[species] = 1 - ss_res / ss_tot
    return out
```

## The Surrogate

```{literalinclude} ../../../examples/gallery_fba_hyb/custom.py
:language: python
:linenos:
:pyobject: surrogate_fba
```

`AVG_QG`/`AVG_N` are the scaler constants from the offline fit; `q_glc` is analytic
(the glucose-uptake bound fixes it exactly), the other three fluxes are the fitted
pole-free rational form: `q = qG · (A1·n)(A2·n) / ((pos(B1·n)+d)(pos(B2·n)+d))`,
`pos(B) = 0.5(B + sqrt(B² + 1.5))`. Every denominator factor is at least `d > 0` for
any input, by construction: the surrogate cannot blow up inside an ODE solver's
adjoint however far the reaction module's own predictions wander, which is exactly
what a real LP solve *would* do if you tried to differentiate through it.

## The Reaction Module

```{literalinclude} ../../../examples/gallery_fba_hyb/custom.py
:language: python
:linenos:
:pyobject: FBAHybReactionModule
```

Two small MLPs predict `qG` (glucose uptake) and combined-demand coefficients
`(n_X, n_M, n_A)` (biomass, ATP maintenance, acetate) from the current state.
The offline LP maximizes a combined-demand flux, with artificial-metabolite
balances enforcing `v_i = n_i * v_combined` for each target flux. These are
prescribed flux proportions. `_bounded_softplus` caps `qG` at 18 and the
coefficients at 1.8, but `qG` can fall below the offline sampling minimum of 0.5.
Uptake near zero therefore extrapolates. `n_S` (succinate) is fixed at `0.0`:
this page has no deliberate product, only real *E. coli* overflow metabolism
(growth versus maintenance versus acetate secretion), the same trade-off the
surrogate was trained on. The predicted coefficients go out through `auxiliary`,
which `hybrax.train` threads straight into `predictions.csv`.

## Training

```{code-cell} ipython3
:tags: [remove-input]

hxt_cli("prepare", "--config", "prepare-config.json",
         "--output-dir", "prepared", "--overwrite")
out = hxt_cli("train", "--config", "train-config.json", "--overwrite")
lines = [l for l in out.splitlines() if "training complete" in l]
print(lines[0] if lines else "training complete")
print(f"run directory: ./{(WORK / 'run').relative_to(WORK.parents[5])}")
```

```{code-cell} ipython3
:tags: [remove-input]

r2 = r2_by_target("run")
for name, value in r2.items():
    print(f"{name:10s} R2 = {value:.4f}")
```

```{code-cell} ipython3
:tags: [remove-input]

hxt_cli("forward", "--config", "forward-config.json",
         "--output-dir", "run/forward", "--overwrite")
from IPython.display import Image
Image(filename=str(WORK / "run/forward/plots/run_1.png"))
```

## What the Model Believes Is Happening

```{code-cell} ipython3
:tags: [remove-input]

df = pd.read_csv(WORK / "run" / "predictions.csv")
run_1 = df[df["process"] == "run_1"]
t = run_1["t"].to_numpy()
coefficient_names = ["aux_n_weights_0", "aux_n_weights_1", "aux_n_weights_2"]
coefficients = run_1[coefficient_names].to_numpy()
shares = coefficients / coefficients.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.stackplot(
    t, shares.T, labels=("n_X (biomass)", "n_M (maintenance)", "n_A (acetate)")
)
ax.set_xlabel("t (h)")
ax.set_ylabel("share of combined-demand coefficients")
ax.set_ylim(0, 1)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
```

`aux_n_weights_0/1/2` are `ReactionOutputs.auxiliary["n_weights"]`, written out with
no extra plumbing beyond what every other page already gets, normalized here to show
each coefficient's share of their sum. The networks infer these coefficients from
state, without time as an input. Their normalized shares describe the prescribed
flux proportions, rather than contributions to a weighted-sum objective.

## How the Surrogate Was Fit

The surrogate is fit offline the same way as the FBA-Hyb paper
[<a href="#ref-fbahyb">1</a>]: 10,000 real FBA solves on `e_coli_core.xml`
[<a href="#ref-ecolicore">2</a>], then a pole-free rational fit against that data and a
sampled overshoot screen at 200,000 points in the sampling box (validation R² ≥ 0.999 on
all four fitted fluxes). The screen is not a proof that the surrogate stays bounded
everywhere in the box. The fitting scripts live in `examples/gallery_fba_hyb/` for anyone who
wants to reproduce it; they don't run as part of building these docs, since solving
10,000 LPs takes a couple of minutes and needs `cobra`, which every doc build should
not pay for. The coefficients are frozen into `surrogate_fba` above. The generator
downloads `e_coli_core.xml` from [BiGG Models](https://bigg.ucsd.edu/models/e_coli_core)
when it runs; BiGG's [license](https://bigg.ucsd.edu/license) permits educational,
research and non-profit use only, while Hybrax uses the MIT license.

## Gotchas

- **The surrogate is frozen, never trained.** Only the two MLPs predicting its
  inputs are `trainable_field()`s; `surrogate_fba` itself has no learnable
  parameters at all.
- **A surrogate that looks accurate on validation data can still be dangerous.**
  An unconstrained fit can free-run to far outside the training data's range in the
  sparse corners of a 5-dimensional sampling box. The sampled overshoot screen can
  expose such excursions, but it cannot guarantee stable training.
- **`_bounded_softplus` caps the MLP outputs.** `qG` can still fall below the
  offline sampling minimum of 0.5. Uptake near zero extrapolates beyond the box
  checked by the boundedness certificate.
- **`n_S` fixed at `0.0` is a deliberate modeling choice.** It means
  this reaction module can never represent a deliberate product, by construction.

## See Also

The full, runnable example (`custom.py`, configs, data) lives in
`examples/gallery_fba_hyb/` at the repo root, no docs build required. This page's own
executed run is at `./docs/source/_data/out/runs/gallery_fba_hyb/`.

- [PLS-dFBA](pls_dfba.md): builds on this exact surrogate, adds a deliberate
  product and a media-blend-aware PLS component.
- [The Reaction Module](../train/reaction_module.md): `auxiliary`, and everything
  else a `RateModule` can return.

## References

1. <a id="ref-fbahyb"></a>Gotsmy, M., & Guillén-Gosálbez, G. (2026). Integrating
   metabolic networks into hybrid bioprocess models. *bioRxiv*.
   [doi:10.64898/2026.04.22.720062](https://doi.org/10.64898/2026.04.22.720062).
2. <a id="ref-ecolicore"></a>Orth, J. D., Fleming, R. M. T., & Palsson, B. Ø.
   (2010). Reconstruction and use of microbial metabolic networks: the core
   *Escherichia coli* metabolic model as an educational guide. *EcoSal Plus*, 4(1).
   [doi:10.1128/ecosalplus.10.2.1](https://doi.org/10.1128/ecosalplus.10.2.1).
