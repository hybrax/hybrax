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

# OptFed

> A real, published rate law for substrate uptake, growth, and production, driven by a
> controlled process variable, `temperature`, instead of a neural network. Its
> temperature term is the Eyring equation: a reaction speeds up with heat, then
> collapses again as the enzyme denatures.

Inspired by Schlögl, Lück, Kittler, Spadiut, Kopp, Zanghellini & Gotsmy 2024
<a href="#ref-optfed">[1]</a>, *"Optimizing bioprocessing efficiency with
OptFed: Dynamic nonlinear modeling improves product-to-biomass yield,"* whose model
predicts substrate uptake, growth, and product formation as independent, multiplicative
inhibition and activation terms, each carrying its own Eyring-equation temperature
term. This page reproduces that structure for all three rates, with a smaller set of
inhibiting and activating variables than the paper uses, fit by ordinary gradient
descent through the whole ODE rather than the paper's own multi-stage fitting
pipeline.

The walkthrough below shows the file in pieces, next to the reasoning for each one.

```{code-cell} ipython3
:tags: [remove-cell]

import os, shutil, subprocess, sys
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_optfed").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
EXAMPLE = Path("../../../examples/gallery_optfed").resolve()
shutil.copy(EXAMPLE / "data.json", WORK / "data.json")
shutil.copy(EXAMPLE / "ground_truth.json", WORK / "ground_truth.json")
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

import json
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hybrax.format as hxf
import hybrax.train as hxt

_collection = hxf.serialization.load_process_collection(WORK / "data.json")

def r2_by_target(run_dir):
    """Pooled R2: concatenate every process's residuals/variance before
    dividing, so one narrow-range process (e.g. a low-feed run's glucose,
    which barely moves) can't make an otherwise-good fit look catastrophic."""
    df = pd.read_csv(WORK / run_dir / "predictions.csv")
    per_target = {s: ([], []) for s in ("biomass", "glucose", "product")}
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

## The Rate Law

```{literalinclude} ../../../examples/gallery_optfed/custom.py
:language: python
:linenos:
:lines: 24-45
```

`_eyring_rate_ceiling` computes the temperature term above, vectorized: given a scalar
temperature and arrays of its four constants, it returns one rate ceiling per instance
at once. `_inhibition_product`/`_activation_product` are the paper's own
independent-multiplicative-terms idea: each named influence variable contributes its
own factor, `1/(1+v/K)` to suppress a rate, `1+v/K` to boost it, and the factors
multiply.

## The Reaction Module

```{literalinclude} ../../../examples/gallery_optfed/custom.py
:language: python
:linenos:
:lines: 48-73
```

Twenty trainable scalars: 4 per temperature term (`log_A`, `log_Ea_R`,
`raw_Teq`, `log_dHeq_R`) times 3 instances, plus the two Michaelis constants
and the four inhibition/activation constants. `Y_XrG`/`Y_PG` are
`frozen_field()`s: the paper itself states these yields come from a genome-scale
model, so they stay fixed rather than fitted.

```{literalinclude} ../../../examples/gallery_optfed/custom.py
:language: python
:linenos:
:lines: 75-109
```

`temperature` comes in through `unscale_controlled_PVs(inputs.SCL_controlled_PVs)`,
the same mechanism [PLS-dFBA](pls_dfba.md) uses for its blend fraction, here
feeding a real physical rate equation instead of a neural network's input
layer. `gamma_deg` (uptake) is computed first, `gamma_alpha` (maintenance)
next since it uses `gamma_deg` as one of its own activation variables (the
paper's real coupling: more total uptake activates more maintenance), then
`gamma_pi` (production) saturates in `gamma_deg - gamma_alpha`. `X_active`
(declared in the dataset's `biological_ode` block as `biomass - product`)
does the rest for every rate except `q_glucose`. `q_biomass` and `q_product` are
*specific* (per unit active biomass), so no manual `Xr/X` correction is needed inside
`__call__` for them. `X_active` in the declared derivative strings does that
division-then-multiplication implicitly and exactly. `q_glucose` is the one rate that
stays scaled by total `biomass` instead: a deliberate simplification versus the paper's
own scaling, the same one [the data generator](../_data/generate.py) applies when it
forward-simulates this page's training data, so the model and its own ground truth stay
self-consistent.

## Training

```{code-cell} ipython3
:tags: [remove-input]

hxt_cli("prepare", "--config", "prepare-config.json",
         "--output-dir", "prepared", "--overwrite")
out = hxt_cli("train", "--config", "train-config.json", "--overwrite")
lines = [l for l in out.splitlines() if "training complete" in l]
print(lines[0] if lines else "training complete")
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

hxt_cli("forward", "--config", "forward-config.json",
         "--output-dir", "run/forward", "--overwrite")
from IPython.display import Image
Image(filename=str(WORK / "run/forward/plots/T_high.png"))
```

`T_high` (40°C, the hottest of this page's six runs) is worth looking at
specifically: glucose visibly accumulates instead of being consumed, the
temperature term genuinely suppressing uptake capacity at high temperature.

## Did It Recover the True Parameters?

```{code-cell} ipython3
:tags: [remove-input]

wrapper, cfg = hxt.model_load(str(WORK / "run"))
m = wrapper.reaction_module
fitted_Teq_K = 290.0 + 40.0 * jax.nn.sigmoid(m.eyring_raw_Teq)
fitted_Teq_C = np.asarray(fitted_Teq_K) - 273.15

truth = json.loads((WORK / "ground_truth.json").read_text())
true_Teq_C = [truth["eyring_deg"]["Teq"] - 273.15,
              truth["eyring_pi"]["Teq"] - 273.15,
              truth["eyring_alpha"]["Teq"] - 273.15]

print(f"{'rate':12s} {'fitted Teq':>12s} {'true Teq':>12s}")
for name, fit, true in zip(("uptake", "production", "maintenance"), fitted_Teq_C, true_Teq_C):
    print(f"{name:12s} {fit:9.1f} °C {true:9.1f} °C")
```

Uptake's and production's fitted thermal optima land within a couple of
degrees of the true values: both denaturation cliffs sit inside this page's
28-40°C training range, so there is real data on both sides of the curve.
Maintenance's true optimum (46.85°C) sits *above* every temperature this page
trains on, and the fit shows it: nothing in the data distinguishes a
denaturation cliff at 47°C from one further away still, so gradient descent
settles for a curve that merely fits the mild, still-rising behavior actually
observed. That is a structural identifiability limit: parameters whose defining
feature sits outside the sampled operating envelope come back looking however
gradient descent found cheapest.

## Gotchas

- **A rate's own optimum has to fall inside the training temperature range to
  be identifiable.** See "maintenance" above: fitting a denaturation cliff
  from data that never reaches it asks the model to extrapolate.
- **`gamma_mu = gamma_deg - gamma_pi - gamma_alpha` can go transiently
  negative early in training.** It is left unclipped, matching the paper's own
  rate law literally; a bad early step just means slower growth.
- **The temperature term's `exp` terms need `T` in Kelvin.** The dataset
  stores `temperature` in °C (the unit a real process log would use);
  `__call__` converts on the way in, once.
- **`n_pv`/`n_fvc` being zero anywhere breaks a naive `jnp.asarray([])`
  reshape.** `estimate_all_scales` guards every controlled axis with `if
  n_fvc else empty`/`if n_pv else empty` for exactly this reason.

## See Also

The full, runnable example (`custom.py`, configs, data) lives in
`examples/gallery_optfed/` at the repo root, no docs build required. This page's own
executed run is at `./source/_data/out/runs/gallery_optfed/`.

- [Mechanistic Models](mechanistic_rates.md): a smaller structured rate law,
  the same "did it recover the true parameters" question asked there too.
- [PLS-dFBA](pls_dfba.md): the same controlled-process-variable-as-extra
  -input mechanism, there feeding a learned component instead of closed-form
  kinetics.
- [The Reaction Module](../train/reaction_module.md): `frozen_field`,
  `trainable_field`, and everything else a `UserReactionModule` can return.
- [The Bioprocess ODE](../format/bioprocess_ode.md): `biological_ode`,
  `algebraic`, and the `X_active` pattern this page reuses.
- [Glutamine Decay](glutamine_decay.md): a smaller worked example of the same
  "true value must fall inside the sampled range to be identifiable" lesson.

## References

1. <a id="ref-optfed"></a>Schlögl, G., Lück, R., Kittler, S., Spadiut, O.,
   Kopp, J., Zanghellini, J., & Gotsmy, M. (2024). Optimizing bioprocessing
   efficiency with OptFed: Dynamic nonlinear modeling improves
   product-to-biomass yield. *Computational and Structural Biotechnology
   Journal*, 23, 3651-3661.
   [https://doi.org/10.1016/j.csbj.2024.09.024](https://doi.org/10.1016/j.csbj.2024.09.024)
