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

# Knowledge transfer

> **Demonstrates.** Pooling data from several products to help a data-poor new one,
> using a constant-valued controlled process variable as a one-hot product-identity
> feature, and an ensemble of GPs anchored to real training data instead of one GP
> with free-floating inducing points.

Inspired by Helleckes et al. 2024 <a href="#ref-helleckes2024">[1]</a>, whose headline result is
that pooling data across products, "horizontal knowledge transfer," measurably helps
a new product with few runs of its own, provided the historical products actually
resemble it. This page reproduces that qualitative result natively in hybrax.train, on
synthetic data, built on [the GP reaction module](gaussian_process.md). It is not a
replication of their method (their model is fit by maximum-likelihood estimation on
a precomputed rate target and pools via one-hot encoding or a PACOH meta-learned
prior; this page trains by gradient descent through the ODE solve and pools via a
controlled process variable), only of the finding.

The walkthrough below shows the file in pieces, next to the reasoning for each one. For
the whole thing at once: to copy, diff against your own, or just read top to bottom:

:::{dropdown} Full `custom.py`
```{literalinclude} _files/knowledge_transfer_custom.py
:language: python
:linenos:
```
:::

```{code-cell} ipython3
:tags: [remove-cell]

import os, shutil, subprocess, sys, textwrap
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_knowledge_transfer").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
shutil.copy(Path("_files/knowledge_transfer_custom.py").resolve(), WORK / "custom.py")

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

_all = hxf.serialization.load_process_collection(
    Path("../_data/out/demo_products/data.json").resolve())
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
    (WORK / f"prepare-{variant}.json").write_text(
        f'{{ "prepare": {{ "raw_input": "{variant}.json" }}, "custom_py": "custom.py" }}\n')
    (WORK / f"train-{variant}.json").write_text(textwrap.dedent(f"""\
        {{
          "data": {{ "prepared": "prepared_{variant}" }},
          "custom_py": "custom.py",
          "train": {{ "epochs": 400, "seed": 0, "learning_rate": 0.01 }},
          "checkpoint": {{ "every": 0 }},
          "output": {{ "dir": "run_{variant}" }}
        }}
        """))
```

## Two products, one shared identity feature

`demo_products` has five products: four "historical" (`H1`-`H4`) with 6 runs each,
and one "target" (`T`) with only 4, held data-poor on purpose. All five share
similar kinetics (slow growth, low glucose affinity, product-forming) but are
distinguishable cell lines, not re-seeded noise: see
[the data generator](../_data/generate.py) for the exact numbers.

`ReactionInputs` has no "which process produced this state" field, by design: the
same reaction module applies uniformly regardless of source process. A constant
controlled process variable does exactly this job instead, using only existing,
unmodified hybrax machinery:

```{literalinclude} _files/knowledge_transfer_custom.py
:language: python
:linenos:
:lines: 39-49
```

`is_new_product` is `0.0` for every timepoint of every historical run, `1.0` for the
target's: a one-hot product-identity feature, concatenated onto the physiological
state before the kernel sees it (below). Attaching it via `transform_process_collection`
means `demo_products` itself carries no such column: the page's own setup mutates a
working copy.

## The ensemble

Pooling several products' data behind one model raises a specific risk for the
data-poor target: with free-floating trainable inducing points shared across the
whole pooled set, capacity can get spent wherever gradient descent happens to push
it, not necessarily where the target needs it. `EnsembleGPReactionModule` addresses
this two ways: several independent GP heads instead of one, and every head's
inducing points anchored to a real bootstrap subsample of training data instead of
free vectors.

```{literalinclude} _files/knowledge_transfer_custom.py
:language: python
:linenos:
:lines: 52-85
```

`centers` is a `frozen_field()`: real `(state, is_new_product)` pairs pulled from
`training_parent_collection` at construction time via `build_reaction_module`, never trained.
`pseudo_targets` stays trainable: rates are never directly observed, only inferred
through the ODE fit, unlike the real state locations.

```{literalinclude} _files/knowledge_transfer_custom.py
:language: python
:linenos:
:lines: 87-110
```

The final prediction is the mean across heads; the **spread across heads** stands in
for `rate_std`, replacing the closed-form single-GP variance from the previous page.
This mirrors Helleckes et al. 2024's <a href="#ref-helleckes2024">[1]</a> own "mean averaging
ensemble... 30 GP models, each subsampling 50% of the training data experiments,"
scaled down (5 heads here, not 30) and subsampled at the point level rather than the
experiment level, both for tractability inside hybrax.train's per-solver-step
reaction-module call.

```{literalinclude} _files/knowledge_transfer_custom.py
:language: python
:linenos:
:lines: 113-132
```

`build_reaction_module` is the piece that changed shape from every other gallery
page: it reads real training data through `training_parent_collection` *before* the
module exists, using the reaction module's own `Scaler.scale_value()` to convert RAW
states to the same SCL space the module will see at call time.

## Local vs. pooled

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
plus all 24 historical runs.

## Evaluating on the held-out runs

The real test is `T`'s 2 held-out runs, never seen by either model. `T`'s training
runs sit in a narrow initial-condition slice (`S0` 12-14 g/L); its held-out runs sit
at the extremes of a much wider range (`S0` 6 and 26 g/L) that only the historical
products actually cover.

```{code-cell} ipython3
:tags: [remove-input]

sys.path.insert(0, str(WORK))
from custom import transform_process_collection
import hybrax.train as hxt

_heldout_raw = hxf.serialization.load_process_collection(WORK / "heldout.json")
_heldout = transform_process_collection(_heldout_raw, config=None)

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

`local`'s R² is strongly negative on `biomass`: not a bug, a real extrapolation
failure. Trained only on `S0` 12-14 g/L, it never saw glucose run out at the held-out
run's lower `S0`, so it keeps extrapolating the exponential growth phase it learned
instead of saturating. `pooled` recovers because the historical products, sampled
across the full design space, give the shared kernel real information about what
happens at both extremes, even though their exact kinetics differ from `T`'s: the
general shape (growth decelerates as glucose depletes) transfers across products even
when the precise rate does not.

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
  With dissimilar kinetics, a shared kernel has to reconcile mostly-unrelated
  products, and a fixed, small set of real-data anchors per ensemble head is not
  enough to resolve multiple regimes at once: pooled can end up *worse* than local,
  not better. This matches Helleckes et al. 2024's <a href="#ref-helleckes2024">[1]</a> own
  finding: "in case the historical data are more heterogeneous... OHE models
  performed more similarly to local models," i.e. pooling's benefit shrinks toward
  zero as similarity drops.
- **A fair "local" baseline needs held-out data outside its training coverage.** If a
  data-poor model's held-out runs are drawn from the same narrow initial-condition
  range as its training runs, it can look deceptively good: a low-dimensional, smooth
  interpolation problem any flexible model solves easily with a couple of examples.
  Test on conditions the training runs did not cover, the way real "few experiments
  for a new product" data actually looks.
- **`prepare-config.json` needs `custom_py` at the top level**, not just the train
  config. Omit it and `transform_process_collection` silently never runs: no error,
  no warning, `is_new_product` just never gets attached. See
  [Prepare](../train/prepare.md#configuration).
- **A constant-valued controlled PV does not scale to many products without a real
  embedding.** One-hot works for a handful of products; Hutter et al. <a href="#ref-hutter2021">[2]</a>
  use a learned embedding instead once the product count grows.
- **More training epochs is not a reliable fix for a pooled model stuck above
  local's loss floor.** Watch for held-out R² getting *worse* while training loss
  keeps (barely) improving with more epochs: a sign of overfitting the
  majority-product data, not under-convergence.
- **`build_reaction_module` needing `training_parent_collection`** (to pull real
  anchor data) is the one place this page's hook signature differs from every
  simpler gallery page's `(*, seed, **kwargs)`.

## See also

Run the example yourself at `./source/_data/out/runs/gallery_knowledge_transfer/`.

- [Gaussian process reaction module](gaussian_process.md): the single-GP version this
  builds on.
- [Fed-batch](fed_batch.md): another reaction module reading a controlled PV as a real
  input, the same mechanism `is_new_product` uses here.
- [Scaling](../train/scaling.md): `Scaler.scale_value()`, used inside
  `build_reaction_module` here to convert real training data to SCL space.

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
