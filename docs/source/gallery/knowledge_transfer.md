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

# Knowledge Transfer

> Pooling data from several products to help a data-poor new one, using a
> constant-valued controlled process variable to mark which product each run belongs
> to. An ensemble of Gaussian processes, anchored to real training data, does the
> pooling.

Inspired by Helleckes et al. 2024 <a href="#ref-helleckes2024">[1]</a>, whose headline result is
that pooling data across products, "horizontal knowledge transfer," measurably helps
a new product with few runs of its own, provided the historical products actually
resemble it. This page reproduces that qualitative result natively in `hybrax.train`,
on synthetic data, using an ensemble version of [Gaussian process](gaussian_process.md)'s
`GPReactionModule`, trained by gradient descent through the ODE solve and pooled through
a controlled process variable rather than the paper's own maximum-likelihood fit and
one-hot or learned-embedding pooling.

The walkthrough below shows the file in pieces, next to the reasoning for each one.

```{code-cell} ipython3
:tags: [remove-cell]

import os, shutil, subprocess, sys
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_knowledge_transfer").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
EXAMPLE = Path("../../../examples/gallery_knowledge_transfer").resolve()
shutil.copy(EXAMPLE / "custom.py", WORK / "custom.py")

ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "HYBRAX_TRAIN_DEVICES": "1",
       "MPLBACKEND": "Agg"}

def hxt_cli(*args):
    proc = subprocess.run([sys.executable, "-m", "hybrax.train.cli", *args],
                          cwd=WORK, env=ENV, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout + proc.stderr

import numpy as np
import matplotlib.pyplot as plt
import hybrax.format as hxf

_all = hxf.serialization.load_process_collection(EXAMPLE / "data.json")
_historical = {n: p for n, p in _all.processes.items() if not n.startswith("T_")}
_t_train = {n: p for n, p in _all.processes.items() if n in ("T_run_1", "T_run_2")}
_t_heldout = {n: p for n, p in _all.processes.items() if n in ("T_run_3", "T_run_4")}

hxf.serialization.save_process_collection(
    hxf.BioProcessCollection(processes={**_historical, **_t_train}), WORK / "pooled.json")
hxf.serialization.save_process_collection(
    hxf.BioProcessCollection(processes=dict(_t_train)), WORK / "local.json")
hxf.serialization.save_process_collection(
    hxf.BioProcessCollection(processes=dict(_t_heldout)), WORK / "heldout.json")

for variant in ("local", "pooled"):
    shutil.copy(EXAMPLE / f"prepare-{variant}.json", WORK / f"prepare-{variant}.json")
    shutil.copy(EXAMPLE / f"train-{variant}.json", WORK / f"train-{variant}.json")
```

## Two Products, One Shared Identity Feature

`demo_products` has five products: four "historical" (`H1`-`H4`) with 6 runs each,
and one "target" (`T`) with only 4, held data-poor on purpose. All five share
similar kinetics (slow growth, low glucose affinity, product-forming) as distinguishable
cell lines: see [the data generator](../_data/generate.py) for the exact numbers.

A constant controlled process variable gives the reaction module a
"which product produced this state" signal, using only existing, unmodified
hybrax machinery. [The data generator](../_data/generate.py)'s
`build_demo_products()` attaches it directly, once, when it builds each process:

```{literalinclude} ../_data/generate.py
:language: python
:linenos:
:start-at: processes[run_name] = _products_process
:end-before: collection = hxf.BioProcessCollection
:dedent: 12
```

`is_new_product` is `0.0` for every historical run's process, `1.0` for the target's:
a one-hot product-identity feature, concatenated onto the physiological state before
the model sees it (below). It ships as a `StaticVariable`-valued controlled process
variable in `data.json` itself.

## The Ensemble

`EnsembleGPReactionModule` extends [Gaussian process](gaussian_process.md)'s
`GPReactionModule` to K heads, each anchored to a bootstrap subsample of the same
real `(centers, targets)` pairs. `centers` pairs real `(state, is_new_product)`
locations with `targets`, the real rate estimate at that same state,
bootstrap-resampled *together* per head so a center never gets separated from its own
target. Only the kernel hyperparameters (`log_lengthscale`, `log_output_scale`,
`log_noise`) are trained.

```{literalinclude} ../../../examples/gallery_knowledge_transfer/custom.py
:language: python
:linenos:
:lines: 86-119
```

Each head runs the same closed-form GP posterior `GPReactionModule` does, vmapped
across all K heads at once. The final prediction is the mean across heads; the
**spread across heads** stands in for `rate_std`. This mirrors Helleckes et al.
2024's <a href="#ref-helleckes2024">[1]</a> own "mean averaging ensemble... 30 GP
models, each subsampling 50% of the training data experiments," scaled down to 5
heads here for tractability.

## Fitting the Ensemble

```{literalinclude} ../../../examples/gallery_knowledge_transfer/custom.py
:language: python
:linenos:
:lines: 144-192
```

For every process, {py:func}`hybrax.format.splines.build_pseudobatch_transform` and
{py:func}`hybrax.format.splines.build_backtransform_spline` fit a smooth spline
through the real measured concentrations, the same real-rate-estimation machinery
`GPReactionModule` uses. Calling `.derivative()` and dividing by the real biomass
value turns it into a real specific rate, matching the declared ODE. The first 3
(of 17) samples of every process get dropped first: biomass is still near its
small inoculum value there, and a derivative divided by a small denominator is
unreliable. `is_new_product` becomes one more `centers` column.

```{literalinclude} ../../../examples/gallery_knowledge_transfer/custom.py
:language: python
:linenos:
:lines: 121-141
```

`marginal_nll` is the standard GP negative log marginal likelihood, computed
per-head from that head's own real `centers` and `targets`, then averaged across
the K heads. It's the quantity a textbook GP fit maximizes, reusing the same
`_chol` every head's own posterior already computes.

```{literalinclude} ../../../examples/gallery_knowledge_transfer/custom.py
:language: python
:linenos:
:lines: 195-221
```

`EnsembleGPLossModule` adds one more named loss, `gp_nll`, to the usual per-target
trajectory MSE terms, exactly like `GPReactionModule`'s own loss module. Both
terms drive the same gradient step: the trajectory loss keeps the whole ensemble
anchored to the real measured concentrations, and `gp_nll` fits every head's
kernel hyperparameters the way a real GP does. `nll_weight` scales `gp_nll` down
to keep it in the same rough range as the trajectory terms, since `hybrax.train`
averages named losses.

## Local vs. Pooled

```{code-cell} ipython3
:tags: [remove-input]

for variant in ("local", "pooled"):
    hxt_cli("prepare", "--config", f"prepare-{variant}.json",
             "--output-dir", f"prepared_{variant}", "--overwrite")
    out = hxt_cli("train", "--config", f"train-{variant}.json", "--overwrite")
    lines = [l for l in out.splitlines() if "training complete" in l]
    print(variant, lines[0] if lines else "training complete")
print(f"run directory: ./{(WORK).relative_to(WORK.parents[4])}")
```

Both runs use the identical architecture, epoch budget and learning rate: only the
training data differs. `local` sees `T`'s 2 training runs alone; `pooled` sees those
plus all 24 historical runs (4 products × 6 runs each).

## Evaluating on the Held-Out Runs

The real test is `T`'s 2 held-out runs, never seen by either model. `T`'s training
runs sit in a narrow initial-condition slice (`S0` 12-14 g/L); its held-out runs sit
at the extremes of a much wider range (`S0` 6 and 26 g/L) that only the historical
products actually cover.

```{code-cell} ipython3
:tags: [remove-input]

import hybrax.train as hxt

_heldout = hxf.serialization.load_process_collection(WORK / "heldout.json")

def r2_by_target(run_dir):
    wrapper, cfg = hxt.model_load(str(WORK / run_dir))
    preds = hxt.model_predict(wrapper, cfg, _heldout, grid_n=200)
    species_order = list(wrapper.modeled_RMC_names)
    per_target = {s: [] for s in ("biomass", "glucose", "product")}
    for proc_name, export in preds.items():
        process = _heldout.processes[proc_name]
        for species in per_target:
            comp = process.reactor_medium.components[species].concentration
            t_meas, y_meas = np.asarray(comp.times), np.asarray(comp.values)
            idx = species_order.index(species)
            y_pred = np.interp(t_meas, np.asarray(export.t),
                               np.asarray(export.c_species)[:, idx])
            ss_res = np.sum((y_meas - y_pred) ** 2)
            ss_tot = np.sum((y_meas - y_meas.mean()) ** 2)
            per_target[species].append(1 - ss_res / ss_tot)
    return {k: float(np.mean(v)) for k, v in per_target.items()}

r2_local = r2_by_target("run_local")
r2_pooled = r2_by_target("run_pooled")
print(f"{'target':10s} {'local':>10s} {'pooled':>10s}")
for name in ("biomass", "glucose", "product"):
    print(f"{name:10s} {r2_local[name]:10.4f} {r2_pooled[name]:10.4f}")
```

`local`'s R² is strongly negative, a real extrapolation failure. Trained
only on `S0` 12-14 g/L, it never saw glucose run out at the held-out run's lower
`S0`, so it keeps extrapolating the exponential growth phase it learned instead of
saturating. `pooled` recovers because the historical products, sampled across the
full design space, give the model real information about what happens at both
extremes, even though their exact kinetics differ from `T`'s.

```{code-cell} ipython3
:tags: [remove-input]

_process_name = "T_run_4"
_process = _heldout.processes[_process_name]

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
_species_list = ("biomass", "glucose", "product")

for run_dir, label, style in [("run_local", "local", "--"), ("run_pooled", "pooled", "-")]:
    wrapper, cfg = hxt.model_load(str(WORK / run_dir))
    preds = hxt.model_predict(wrapper, cfg, _heldout, grid_n=200)
    export = preds[_process_name]
    species_order = list(wrapper.modeled_RMC_names)
    for ax, species in zip(axes, _species_list):
        idx = species_order.index(species)
        ax.plot(export.t, np.asarray(export.c_species)[:, idx], style, label=label)

for ax, species in zip(axes, _species_list):
    comp = _process.reactor_medium.components[species].concentration
    ax.plot(comp.times, comp.values, "ko", ms=4, label="measured")
    ax.set_title(species)
    ax.set_xlabel("t (h)")
axes[0].legend(fontsize=8)
fig.suptitle(f"Held-out {_process_name}: local vs. pooled")
fig.tight_layout()
```

`local` (dashed) tracks the measured points until roughly `t=13h`, then diverges hard:
it keeps extrapolating the growth phase it learned from `S0` 12-14 g/L data, well past
where this run's lower `S0` actually runs out of glucose. `pooled` (solid) stays on the
measured trajectory throughout.

## Gotchas

- **Pooling only helps if the historical products actually resemble the target.**
  With dissimilar kinetics, one model has to reconcile mostly-unrelated products:
  pooled can end up *worse* than local. This matches Helleckes et al.
  2024's <a href="#ref-helleckes2024">[1]</a> own finding: "in case the historical
  data are more heterogeneous... OHE models performed more similarly to local
  models," i.e. pooling's benefit shrinks toward zero as similarity drops.
- **A fair "local" baseline needs held-out data outside its training coverage.** If a
  data-poor model's held-out runs are drawn from the same narrow initial-condition
  range as its training runs, it can look deceptively good. Test on conditions the
  training runs did not cover, the way real "few experiments for a new product"
  data actually looks.
- **A constant-valued controlled PV does not scale to many products without a real
  embedding.** One-hot works for a handful of products; Hutter et al. <a href="#ref-hutter2021">[2]</a>
  use a learned embedding instead once the product count grows.
- **`build_reaction_module` needing `training_parent_collection`** (to pull real
  anchor data) is the one place this page's hook signature differs from every
  simpler gallery page's `(*, seed, **kwargs)`.

## See Also

The full, runnable example (`custom.py`, configs, data) lives in
`examples/gallery_knowledge_transfer/` at the repo root, no docs build required. This
page's own executed run is at `./source/_data/out/runs/gallery_knowledge_transfer/`.

- [Gaussian process](gaussian_process.md): the single-GP version this builds on.
- [Fed-batch](fed_batch.md): another reaction module reading a controlled PV as a real
  input, the same mechanism `is_new_product` uses here.
- [Scaling](../train/scaling.md): `Scaler.scale_value()`/`scale_derivative()`, used
  inside `build_reaction_module` here to convert real states and real rate estimates
  to SCL space.

## References

1. <a id="ref-helleckes2024"></a>Helleckes, L. M., Wirnsperger, C., Polak, J.,
   Guillén-Gosálbez, G., Butté, A., & von Stosch, M. (2024). Novel calibration
   design improves knowledge transfer across products for the characterization of
   pharmaceutical bioprocesses. *Biotechnology Journal*, 19(7), e202400080.
   [https://doi.org/10.1002/biot.202400080](https://doi.org/10.1002/biot.202400080)
2. <a id="ref-hutter2021"></a>Hutter, C., von Stosch, M., Cruz Bournazou, M. N., &
   Butté, A. (2021). Knowledge transfer across cell lines using hybrid Gaussian
   process models with entity embedding vectors. *Biotechnology and
   Bioengineering*, 118(11), 4389-4401.
   [https://doi.org/10.1002/bit.27907](https://doi.org/10.1002/bit.27907)
