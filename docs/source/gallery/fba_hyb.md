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

> **Demonstrates.** A hybrid dynamic-FBA reaction module: two small MLPs predict a
> glucose-uptake rate and a flux-balance-analysis objective from the current state,
> a frozen, pole-free surrogate converts that into real metabolic rates, and
> `ReactionOutputs.auxiliary` exposes what the model believes the cell's metabolic
> allocation is doing over the batch.

Dynamic flux-balance analysis (dFBA) normally means solving a real linear program,
`max c^T v` subject to `S.v = 0` and flux bounds, at every integration step. That is
not something JAX can differentiate through or `vmap`/`jit` the way it does a neural
network, and bp-train has no LP solver in its training loop. The fix, from Gotsmy &
Guillén-Gosálbez's FBA-Hyb <a href="#ref-fbahyb">[1]</a>: fit a closed-form,
pole-free surrogate of the FBA solution **once, offline** (10,000 real linear
programs solved outside the training loop), then embed that surrogate directly
inside an ordinary `UserReactionModule`. No LP solve ever happens during training.

The surrogate here is fit fresh against `e_coli_core.xml`
<a href="#ref-ecolicore">[2]</a>, a small, real, published teaching model (95
reactions, 72 metabolites), not a proprietary genome-scale network: see
[Knowledge](#how-the-surrogate-was-fit) below for the full, reproducible chain. See
[PLS-dFBA](pls_dfba.md) for this page's sibling, which builds on the same surrogate
with a product-forming, media-blend-aware extension.

The walkthrough below shows the file in pieces, next to the reasoning for each one. For
the whole thing at once: to copy, diff against your own, or just read top to bottom:

:::{dropdown} Full `custom.py`
```{literalinclude} _files/fba_hyb_custom.py
:language: python
:linenos:
```
:::

```{code-cell} ipython3
:tags: [remove-cell]

import os, shutil, subprocess, sys, textwrap
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_fba_hyb").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
shutil.copy(Path("../_data/out/demo_ecoli_fba/data.json").resolve(), WORK / "data.json")
shutil.copy(Path("_files/fba_hyb_custom.py").resolve(), WORK / "custom.py")

ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "BP_TRAIN_DEVICES": "1",
       "MPLBACKEND": "Agg"}

def bp_train_cli(*args):
    proc = subprocess.run([sys.executable, "-m", "bp_train.cli", *args],
                          cwd=WORK, env=ENV, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout + proc.stderr

(WORK / "prepare-config.json").write_text(
    '{ "prepare": { "raw_input": "data.json" } }\n')
(WORK / "train-config.json").write_text(textwrap.dedent("""\
    {
      "data": { "prepared": "prepared" },
      "custom_py": "custom.py",
      "train": { "epochs": 800, "seed": 0, "learning_rate": 0.01 },
      "output": { "dir": "run" }
    }
    """))
(WORK / "forward-config.json").write_text('{ "models": ["run"] }\n')

import csv
import numpy as np
import matplotlib.pyplot as plt
import bp_format as bp
import bp_train

_collection = bp.serialization.load_process_collection(WORK / "data.json")

def r2_by_target(run_dir):
    rows_by_process = {}
    with (WORK / run_dir / "predictions.csv").open() as fh:
        for row in csv.DictReader(fh):
            rows_by_process.setdefault(row["process"], []).append(row)
    per_target = {s: ([], []) for s in ("biomass", "glucose", "acetate")}
    for name, process in _collection.processes.items():
        rows = rows_by_process[name]
        t_pred = np.array([float(r["t"]) for r in rows])
        for species in per_target:
            comp = process.reactor_medium.components[species].concentration
            t_meas, y_meas = np.asarray(comp.times), np.asarray(comp.values)
            y_pred = np.interp(t_meas, t_pred,
                               np.array([float(r[f"c_{species}"]) for r in rows]))
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

## The surrogate

```{literalinclude} _files/fba_hyb_custom.py
:language: python
:linenos:
:lines: 36-50
```

`AVG_QG`/`AVG_N` are the scaler constants from the offline fit; `q_glc` is analytic
(the glucose-uptake bound fixes it exactly), the other three fluxes are the fitted
pole-free rational form: `q = qG · (A1·n)(A2·n) / ((pos(B1·n)+d)(pos(B2·n)+d))`,
`pos(B) = 0.5(B + sqrt(B² + 1.5))`. Every denominator factor is at least `d > 0` for
any input, by construction: the surrogate cannot blow up inside an ODE solver's
adjoint however far the reaction module's own predictions wander, which is exactly
what a real LP solve *would* do if you tried to differentiate through it.

## The reaction module

```{literalinclude} _files/fba_hyb_custom.py
:language: python
:linenos:
:lines: 69-105
```

Two small MLPs predict `qG` (glucose uptake) and the FBA objective weights
`(n_X, n_M, n_A)` (biomass, ATP maintenance, acetate) from the current state, each
squashed through `_bounded_softplus` to stay inside the range the surrogate was
actually fit over. `n_S` (succinate) is fixed at `0.0`: this page has no deliberate
product, only real *E. coli* overflow metabolism (growth versus maintenance versus
acetate secretion), the same trade-off the surrogate was trained on. The predicted
weights go out through `auxiliary`, which bp-train threads straight into
`predictions.csv`.

## Training

```{code-cell} ipython3
:tags: [remove-input]

bp_train_cli("prepare", "--config", "prepare-config.json",
         "--output-dir", "prepared", "--overwrite")
out = bp_train_cli("train", "--config", "train-config.json", "--overwrite", "--no-plot")
print([l for l in out.splitlines() if "training complete" in l][0])
print(f"run directory: ./{(WORK / 'run').relative_to(WORK.parents[4])}")
```

```{code-cell} ipython3
:tags: [remove-input]

r2 = r2_by_target("run")
for name, value in r2.items():
    print(f"{name:10s} R2 = {value:.4f}")
```

```{code-cell} ipython3
:tags: [remove-input]

bp_train_cli("forward", "--config", "forward-config.json",
         "--output-dir", "run/forward", "--overwrite")
from IPython.display import Image
Image(filename=str(WORK / "run/forward/run_1.png"))
```

## What the model believes is happening

```{code-cell} ipython3
:tags: [remove-input]

rows = [r for r in csv.DictReader((WORK / "run" / "predictions.csv").open())
        if r["process"] == "run_1"]
t = np.array([float(r["t"]) for r in rows])

fig, ax = plt.subplots(figsize=(7, 3.5))
for i, label in enumerate(("n_X (biomass)", "n_M (maintenance)", "n_A (acetate)")):
    ax.plot(t, [float(r[f"aux_n_weights_{i}"]) for r in rows], label=label)
ax.set_xlabel("t (h)")
ax.set_ylabel("predicted FBA objective weight")
ax.legend(fontsize=8)
fig.tight_layout()
```

`aux_n_weights_0/1/2` are `ReactionOutputs.auxiliary["n_weights"]`, written out with
no extra plumbing beyond what every other page already gets. The trend is genuinely
learned, not told: the model was never given time as an input, only state, yet `n_A`
rises steadily over the batch (real overflow metabolism intensifying as glucose
depletes and biomass accumulates), inferred purely from the state it was shown.

## How the surrogate was fit

10,000 real parsimonious-FBA solves on `e_coli_core.xml`
<a href="#ref-ecolicore">[2]</a>, Latin-hypercube-sampled over
`(qG, n_X, n_M, n_A, n_S)`:

:::{dropdown} `01_generate_fba_data.py`
```{literalinclude} _files/01_generate_fba_data.py
:language: python
:linenos:
```
:::

Then a fresh pole-free rational fit against that data (multi-restart L-BFGS, method
from FBA-Hyb <a href="#ref-fbahyb">[1]</a>), with a boundedness certificate checked
before acceptance: the metric that actually predicts training stability, since a
surrogate that looks perfect on held-out data can still spike in the sparse gaps of
the sampling box and blow up an ODE adjoint:

:::{dropdown} `02_fit_surrogate.py`
```{literalinclude} _files/02_fit_surrogate.py
:language: python
:linenos:
```
:::

Validation R² ≥ 0.999 on all four fitted fluxes; boundedness certificate passed (max
overshoot 1.7× over the sampling box, min denominator 0.199 > 0). Neither script runs
as part of building these docs: solving 10,000 LPs takes a couple of minutes and
needs `cobra`, which every doc build should not pay for. The coefficients are frozen
into `surrogate_fba` above.

## Gotchas

- **The surrogate is frozen, never trained.** Only the two MLPs predicting its
  inputs are `trainable_field()`s; `surrogate_fba` itself has no learnable
  parameters at all.
- **A surrogate that looks accurate on validation data can still be dangerous.**
  The boundedness certificate, not validation R² alone, is what predicts whether
  training will stay stable: an unconstrained fit can free-run to far outside the
  training data's range in the sparse corners of a 5-dimensional sampling box.
- **`_bounded_softplus` keeps the MLPs inside the surrogate's fitted range.**
  Removing it does not error immediately: the surrogate just starts extrapolating
  into territory the boundedness certificate never checked.
- **`n_S` fixed at `0.0` is a real modeling choice, not a placeholder.** It means
  this reaction module can never represent a deliberate product, by construction.

## See also

Run the example yourself at `./source/_data/out/runs/gallery_fba_hyb/`.

- [PLS-dFBA](pls_dfba.md): builds on this exact surrogate, adds a deliberate
  product and a media-blend-aware PLS component.
- [Gaussian process reaction module](gaussian_process.md): another closed-form,
  non-neural-network reaction module, same `auxiliary` mechanism.
- [The reaction module](../train/reaction_module.md): `auxiliary`, and everything
  else a `UserReactionModule` can return.

## References

1. <a id="ref-fbahyb"></a>Gotsmy, M., & Guillén-Gosálbez, G. (2026). Integrating
   metabolic networks into hybrid bioprocess models. *bioRxiv*.
   [https://doi.org/10.64898/2026.04.22.720062v1](https://www.biorxiv.org/content/10.64898/2026.04.22.720062v1)
2. <a id="ref-ecolicore"></a>Orth, J. D., Fleming, R. M. T., & Palsson, B. Ø.
   (2010). Reconstruction and use of microbial metabolic networks: the core
   *Escherichia coli* metabolic model as an educational guide. *EcoSal Plus*, 4(1).
   [https://doi.org/10.1128/ecosalplus.10.2.1](https://doi.org/10.1128/ecosalplus.10.2.1)
