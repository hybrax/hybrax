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
# A KAN model
<!-- UNLOCK -->

> **Demonstrates.** A Kolmogorov-Arnold Network (KAN) occupying the reaction module's
> slot: every edge between an input and a hidden or output node carries its own
> learnable univariate function (a SiLU base term plus a small Gaussian
> radial-basis expansion), summed at each node, instead of an MLP's fixed
> activation with learned linear weights. Trained end to end by bp-train's own
> optimizer; each edge's learned curve can be read out directly after training.

Inspired by Bühler & Guillén-Gosálbez 2026 <a href="#ref-srkan">[1]</a>, whose
SR-KAN framework builds on Kolmogorov-Arnold Networks
<a href="#ref-kan">[2]</a> and, in their own bioprocess case study, recovers
interpretable kinetic rate laws for a batch fermentation of biomass, substrate
and product, a system whose shape lines up closely with `demo_batch`'s own
biomass/glucose/product state. This page reproduces the core architectural idea
(learnable univariate functions on edges, summed at nodes, in place of an MLP)
as a live bp-train reaction module, using a Gaussian radial-basis edge function
in place of B-splines, an equivalent formulation per Li 2024
<a href="#ref-rbf">[3]</a>, the same reasoning SR-KAN itself uses to justify
swapping B-splines for a different fast, localized basis.

**Two pieces worth being explicit about.** This page trains the KAN as the live
reaction module inside bp-train's own Diffrax-integrated, end-to-end
differentiable training loop. SR-KAN's own bioprocess case study is a two-stage
pipeline instead: a Neural Controlled Differential Equation first fits smooth
derivatives from noisy measurements, then those derivatives are symbolically
regressed offline. Nothing here reproduces the Neural CDE stage; bp-format's own
ODE integration already provides a differentiable trajectory directly.

This page also does not reproduce SR-KAN's post-hoc symbolic-extraction pipeline
(matching each trained edge's curve against a closed-form function library via
BFGS, with entropy/sparsity regularization and separability/symmetry detection,
to arrive at a literal equation). The KAN below stays a trained numerical model:
a reader can look directly at each edge's learned curve (below, in
[What each edge learned](#what-each-edge-learned)), which is the practical core
of SR-KAN's own interpretability pitch, but nothing here distills that curve
into a symbolic formula.

The walkthrough below shows the file in pieces, next to the reasoning for each one. For
the whole thing at once: to copy, diff against your own, or just read top to bottom:

:::{dropdown} Full `custom.py`
```{literalinclude} _files/kan_custom.py
:language: python
:linenos:
```
:::

```{code-cell} ipython3
:tags: [remove-cell]

import os, shutil, subprocess, sys, textwrap
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_kan").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
shutil.copy(Path("../_data/out/demo_batch/data.json").resolve(), WORK / "data.json")
shutil.copy(Path("_files/kan_custom.py").resolve(), WORK / "custom.py")

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
      "output": { "dir": "run", "predictions": "parents" }
    }
    """))
(WORK / "forward-config.json").write_text(
    '{ "models": ["run"], "output": { "predictions": "parents", "plots": true } }\n')

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import bp_format as bp
import bp_train

_collection = bp.serialization.load_process_collection(WORK / "data.json")

def r2_by_target(run_dir):
    df = pd.read_csv(WORK / run_dir / "predictions.csv")
    per_target = {}
    for name, process in _collection.processes.items():
        proc_df = df[df["process"] == name]
        t_pred = proc_df["t"].to_numpy()
        for species in ("biomass", "glucose", "product"):
            comp = process.reactor_medium.components[species].concentration
            t_meas, y_meas = np.asarray(comp.times), np.asarray(comp.values)
            y_pred = np.interp(t_meas, t_pred, proc_df[f"c_{species}"].to_numpy())
            ss_res = np.sum((y_meas - y_pred) ** 2)
            ss_tot = np.sum((y_meas - y_meas.mean()) ** 2)
            per_target.setdefault(species, []).append(1 - ss_res / ss_tot)
    return {k: float(np.mean(v)) for k, v in per_target.items()}
```

## The KAN layer

```{literalinclude} _files/kan_custom.py
:language: python
:linenos:
:lines: 27-46
```

Per edge `(i, o)`: `base_w[o,i]·SiLU(x_i)` plus `Σ_g spline_c[o,i,g]·rbf_g(x_i)`, a
small Gaussian radial-basis expansion centered on a fixed grid
<a href="#ref-rbf">[3]</a>. A node's output is the sum over its incoming edges'
curves, `Σ_i φ_oi(x_i)`, the defining property that makes this a KAN
<a href="#ref-kan">[2]</a> rather than an MLP: the learnable function lives on
the edge, not folded into a fixed node activation. `x` is `tanh`-bounded before
hitting the grid, so every edge's spline term always sees an input inside the
range it was actually fit over.

## The reaction module

```{literalinclude} _files/kan_custom.py
:language: python
:linenos:
:lines: 49-75
```

Two stacked `KANLayer`s, both genuinely KAN-shaped, no plain linear or MLP layer
anywhere in between. `_features` is just the raw `SCL_modeled_RMCs`
(biomass, glucose, product): no hand-engineered saturation term is fed in
alongside it, so any Monod-like or threshold shape the model ends up using has
to be discovered from state alone. `l2`'s near-zero initialization keeps the ODE
flat at the very first step while `l1` still receives real gradient, the same
cold-start pattern other reaction modules in this gallery use.

`estimate_all_scales` is unchanged from [Tutorial 4](../tutorials/04_your_first_custom_py.md).

## Training

```{code-cell} ipython3
:tags: [remove-input]

bp_train_cli("prepare", "--config", "prepare-config.json",
         "--output-dir", "prepared", "--overwrite")
out = bp_train_cli("train", "--config", "train-config.json", "--overwrite")
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
Image(filename=str(WORK / "run/forward/forward-results/plots/run_1.png"))
```

R² above 0.99 on all three species: the KAN found a good fit to this page's own
data. Left column: measured points against the integrated trajectory. Right
column: the rates the KAN actually predicted along the way,
`q_biomass`/`q_glucose`/`q_product`, which never enter the loss directly, only
their integral does.

## What each edge learned

```{code-cell} ipython3
:tags: [remove-input]

wrapper, cfg = bp_train.model_load(str(WORK / "run"))
kan = wrapper.reaction_module
l1 = kan.l1
names = list(wrapper.modeled_RMC_names)
scale = np.asarray(kan.SCALE_modeled_RMCs.scale)

df = pd.read_csv(WORK / "run" / "predictions.csv")

def edge_curve(o, i, xs_scl):
    xb = jnp.tanh(xs_scl)
    rbf = jnp.exp(-l1.inv_h2 * (xb[:, None] - l1.centers[None, :]) ** 2)
    spline = jnp.einsum("g,ng->n", l1.spline_c[o, i], rbf)
    base = l1.base_w[o, i] * jax.nn.silu(xb)
    return spline + base

fig, axes = plt.subplots(1, 3, figsize=(11, 3))
for ax, species in zip(axes, names):
    i = names.index(species)
    vals = df[f"c_{species}"].to_numpy()
    lo, hi = max(float(vals.min()), 0.0), float(vals.max())
    xs_raw = np.linspace(lo, hi, 60)
    xs_scl = jnp.asarray(xs_raw / scale[i])
    spreads = [float(jnp.ptp(edge_curve(o, i, xs_scl))) for o in range(l1.base_w.shape[0])]
    o = int(np.argmax(spreads))
    ys = np.asarray(edge_curve(o, i, xs_scl))
    ax.plot(xs_raw, ys)
    ax.set_xlabel(f"{species} (g/L)")
    ax.set_title(f"edge: {species} -> hidden {o}")
fig.tight_layout()
```

Each panel is one edge, `φ_oi(x_i)`, evaluated directly over that input's own
observed range and picked as the widest-swinging edge from that input (the
most visually informative one, out of `l1`'s `hidden` edges per input). None of
these are flat or noisy: each shows genuine curvature, rising or falling then
leveling off, the shape a saturating uptake or a fading effect would actually
have. Reading a curve like this straight off the trained module is the honest,
scoped-down version of SR-KAN's own interpretability pitch: this page stops at
the curve itself, it does not fit that curve to a symbolic function library the
way SR-KAN's own extraction pipeline does.

## Gotchas

- **Inputs are `tanh`-bounded before hitting the RBF grid.** An untransformed
  input that wandered past the fitted centers' range would extrapolate through
  only the base `SiLU` term, silently losing the spline's local detail.
- **`centers` is a `frozen_field()`, never trained.** The grid is fixed; only
  `base_w` and `spline_c`, the edge functions' actual shape, move under
  gradient descent.
- **`l2`'s near-zero initialization is what keeps the ODE flat at step 0**,
  not a small learning rate or a warmup schedule. Removing it does not error,
  it just starts training from a wild, effectively random rate prediction.
- **`hidden` and `grid` are real capacity knobs**, exactly like a network's
  width. Too few edges or basis functions and the model cannot represent the
  trajectory; too many and every edge pays for a wider, mostly redundant
  radial-basis expansion.

## See also

Run the example yourself at `./source/_data/out/runs/gallery_kan/`.

- [The reaction module](../train/reaction_module.md): `auxiliary`, and
  everything else a `UserReactionModule` can return.
- [Gaussian process reaction module](gaussian_process.md): another
  reaction-module architecture, with its own way of reading out what it
  learned.
- [Mechanistic models](mechanistic_rates.md): a reaction module built from
  explicit kinetics instead.

## References

1. <a id="ref-srkan"></a>Bühler, M. A., & Guillén-Gosálbez, G. (2026). SR-KAN:
   A Kolmogorov-Arnold Network guided symbolic regression framework.
   *Computers and Chemical Engineering*, 213, 109721.
   [https://doi.org/10.1016/j.compchemeng.2026.109721](https://doi.org/10.1016/j.compchemeng.2026.109721)
2. <a id="ref-kan"></a>Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J.,
   Soljačić, M., Hou, T. Y., & Tegmark, M. (2024). KAN: Kolmogorov-Arnold
   Networks. *arXiv*.
   [https://arxiv.org/abs/2404.19756](https://arxiv.org/abs/2404.19756)
3. <a id="ref-rbf"></a>Li, Z. (2024). Kolmogorov-Arnold Networks are Radial
   Basis Function Networks. *arXiv*.
   [https://arxiv.org/abs/2405.06721](https://arxiv.org/abs/2405.06721)
