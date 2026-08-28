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

> A Kolmogorov-Arnold Network (KAN) occupying the reaction module's slot: each edge
> between two nodes carries its own learnable curve, instead of a neural network's
> fixed activation with learned weights. After training, each edge's curve is matched
> against a small shape library and checked against the real process that generated the
> training data.

Inspired by Bühler & Guillén-Gosálbez 2026 <a href="#ref-srkan">[1]</a>, whose SR-KAN
framework builds on Kolmogorov-Arnold Networks <a href="#ref-kan">[2]</a> to recover
interpretable kinetic rate laws from a batch fermentation case study similar to this
page's own biomass, glucose, and product data. This page reproduces the core idea,
learnable curves on edges in place of a neural network's fixed activations, as a single
reaction module trained directly inside `hybrax.train`'s own loop rather than the
paper's own two-stage pipeline. It also reuses the paper's post-training step of
matching each learned edge against a small library of simple curves, stopping short of
the paper's further step of composing those matches into one closed-form rate law.

The walkthrough below shows the file in pieces, next to the reasoning for each one.

```{code-cell} ipython3
:tags: [remove-cell]

import os, shutil, subprocess, sys
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_kan").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
EXAMPLE = Path("../../../examples/gallery_kan").resolve()
shutil.copy(EXAMPLE / "data.json", WORK / "data.json")
shutil.copy(EXAMPLE / "custom.py", WORK / "custom.py")
shutil.copy(EXAMPLE / "shape_match.py", WORK / "shape_match.py")
sys.path.insert(0, str(WORK))

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

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import hybrax.format as hxf
import hybrax.train as hxt
from shape_match import best_match

_collection = hxf.serialization.load_process_collection(WORK / "data.json")

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

## The KAN Layer

```{literalinclude} ../../../examples/gallery_kan/custom.py
:language: python
:linenos:
:lines: 29-47
```

Per edge `(i, o)`: `base_w[o,i]·SiLU(x_i)` plus `Σ_g spline_c[o,i,g]·rbf_g(x_i)`, a
small Gaussian radial-basis expansion centered on a fixed grid
<a href="#ref-rbf">[3]</a>. A node's output is the sum over its incoming edges'
curves, `Σ_i φ_oi(x_i)`, the defining property that makes this a KAN
<a href="#ref-kan">[2]</a> rather than an MLP: the learnable function lives on
the edge, separate from a fixed node activation. `x` is `tanh`-bounded before
hitting the grid, so every edge's spline term always sees an input inside the
range it was actually fit over.

## The Reaction Module

```{literalinclude} ../../../examples/gallery_kan/custom.py
:language: python
:linenos:
:lines: 59-90
```

Two stacked `KANLayer`s, both genuinely KAN-shaped, no plain linear or MLP layer
anywhere in between. The input is just the raw `SCL_modeled_RMCs`
(biomass, glucose, product): no hand-engineered saturation term is fed in
alongside it, so any Monod-like or threshold shape the model ends up using has
to be discovered from state alone. `l2`'s near-zero initialization keeps the ODE
flat at the very first step while `l1` still receives real gradient, the same
cold-start pattern other reaction modules in this gallery use.

Each of the three output rates also gets one small multiplicative term,
`prod_a(h0) * prod_b(h1)`, added on top of `l2`'s sum. `h0` and `h1` are the
first two of `l1`'s eight hidden units, picked arbitrarily since none of them
carry individual meaning. SR-KAN's own bioprocess case study states plainly
that pure summation was not enough to model its kinetics, and needed
multiplicative units alongside additive ones; this term borrows that same
idea. It is worth saying plainly that this gallery's own training data does not
actually need it: every rate this page fits turns out to depend on glucose
alone, through a single saturating curve, with nothing multiplicative in the
real process (see [Recovering the real rate law](#recovering-the-real-rate-law)
below). The term stays in as a demonstration that the model has this option
available, the same way SR-KAN's own model does.

`prod_a` and `prod_b` start at ordinary, non-zero values, unlike every other
weight in `l2`, which starts at zero. A product of two values that both start
at zero can never move away from zero under gradient descent, since nudging
one side still leaves the other at zero: multiplication needs a non-zero
starting point the way addition does not.

`estimate_all_scales` is unchanged from [Tutorial 4](../tutorials/04_your_first_custom_py.md).

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
Image(filename=str(WORK / "run/forward/plots/run_1.png"))
```

R² above 0.99 on all three species: the KAN found a good fit to this page's own
data. Left column: measured points against the integrated trajectory. Right
column: the rates the KAN actually predicted along the way,
`q_biomass`/`q_glucose`/`q_product`, which never enter the loss directly, only
their integral does.

## What Each Edge Learned

```{code-cell} ipython3
:tags: [remove-input]

wrapper, cfg = hxt.model_load(str(WORK / "run"))
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
edge_rows = []
for ax, species in zip(axes, names):
    i = names.index(species)
    vals = df[f"c_{species}"].to_numpy()
    lo, hi = max(float(vals.min()), 0.0), float(vals.max())
    xs_raw = np.linspace(lo, hi, 60)
    xs_scl = jnp.asarray(xs_raw / scale[i])
    curves = [np.asarray(edge_curve(o, i, xs_scl)) for o in range(l1.base_w.shape[0])]
    for o, ys in enumerate(curves):
        m = best_match(xs_raw, ys)
        edge_rows.append((species, o, m["best"], m["r2"]))
    o = int(np.argmax([float(np.ptp(ys)) for ys in curves]))
    match = best_match(xs_raw, curves[o])
    ax.plot(xs_raw, curves[o])
    ax.set_xlabel(f"{species} (g/L)")
    ax.set_title(f"edge: {species} -> hidden {o}\n({match['best']})")
fig.tight_layout()
```

Each panel is one edge, `φ_oi(x_i)`, evaluated directly over that input's own
observed range and picked as the widest-swinging edge from that input (the
most visually informative one, out of `l1`'s `hidden` edges per input), labeled
with its best match against a small shape library (flat, linear, power,
saturating, exponential), scored by fit quality with a threshold below which a
curve is reported as no clean match rather than forced onto the nearest shape.
None of the three panels above are flat or noisy: each shows genuine curvature,
rising or falling then leveling off, the shape a saturating uptake or a fading
effect would actually have.

```{code-cell} ipython3
:tags: [remove-input]

n_clean = sum(1 for row in edge_rows if row[2] != "no clean match")
print(f"{n_clean} of l1's {len(edge_rows)} edges matched one of the 5 shapes cleanly (R2 >= 0.9)")
```

## Recovering the Real Rate Law

This page's training data (`data.json`) comes from
`hybrax/docs/source/_data/generate.py`'s `build_demo_batch`, a real,
known kinetic model (Monod growth, fixed biomass yield plus maintenance,
Luedeking-Piret product formation). hybrax multiplies each of the model's
three predicted numbers by the current biomass automatically, outside the
reaction module entirely, so what the KAN itself needs to learn is a
per-biomass rate. Written that way, all three
true rates reduce to one shape:

    (a fixed number) x S / (Ks + S)

a single saturating (Michaelis-Menten) curve in glucose (`S`) alone, with the
same `Ks = 0.05 g/L` in all three and nothing depending on biomass or
product. This gives a direct check: feed the trained model synthetic
combinations of the three inputs, picked to isolate one input at a time, and see
whether its own behavior matches.

```{code-cell} ipython3
:tags: [remove-input]

TRUE = {"mu_max": 0.45, "Ks": 0.05, "Y_XS": 0.45, "m_s": 0.02, "alpha": 0.08, "beta": 0.006}
TRUE_SCALE = {
    "q_biomass": TRUE["mu_max"],
    "q_glucose": -(TRUE["mu_max"] / TRUE["Y_XS"] + TRUE["m_s"]),
    "q_product": TRUE["alpha"] * TRUE["mu_max"] + TRUE["beta"],
}

def eval_reaction_module(state_raw):
    scl = jnp.asarray(np.asarray(state_raw) / scale)
    h = kan.l1(scl)
    return np.asarray(kan.l2(h) + kan.prod_a(h[0:1]) * kan.prod_b(h[1:2]))

pooled = {n: [] for n in names}
for process in _collection.processes.values():
    for species in names:
        comp = process.reactor_medium.components[species].concentration
        pooled[species].append(np.asarray(comp.values))
pooled = {n: np.concatenate(v) for n, v in pooled.items()}
ranges = {n: (max(float(pooled[n].min()), 0.0), float(pooled[n].max())) for n in names}
fixed = {n: float(np.median(pooled[n])) for n in names}

other_effect, equations = [], []
for out_idx, rate in enumerate(["q_biomass", "q_glucose", "q_product"]):
    sweeps = {}
    for species in names:
        lo, hi = ranges[species]
        xs = np.linspace(lo, hi, 40)
        ys = np.array([
            eval_reaction_module([x if n == species else fixed[n] for n in names])[out_idx]
            for x in xs
        ])
        sweeps[species] = (xs, ys, best_match(xs, ys))
    glucose_spread = float(np.ptp(sweeps["glucose"][1]))
    for species in ("biomass", "product"):
        xs, ys, _ = sweeps[species]
        rel = float(np.ptp(ys)) / glucose_spread if glucose_spread > 1e-12 else 0.0
        other_effect.append(rel)
    fit = sweeps["glucose"][2]
    if fit["closest"] == "saturating":
        a_fit, k_fit, _ = fit["fits"]["saturating"]["params"]
        a_true, k_true = TRUE_SCALE[rate], TRUE["Ks"]
        equations.append(
            f"{rate:10s}  learned: {a_fit:+.3f} * S / ({k_fit:.3f} + S)"
            f"   true: {a_true:+.3f} * S / ({k_true:.3f} + S)"
        )

print(f"Biomass and product together move each rate by at most "
      f"{max(other_effect) * 100:.1f}% of what glucose does.\n")
print("\n".join(equations))
```

Biomass and product barely move any rate, matching the real process. Glucose alone
traces a clean saturating curve for all three rates, the right shape. The learned
scale and half-saturation numbers above land only roughly near the real ones.

That gap is expected. Training sees only 3
processes here (15 timepoints each, `BATCH_RUNS` in `generate.py`), and the
sweep above evaluates the model at input combinations that never occur
together along any single real trajectory, off the narrow path training
actually walked. Forster et al. report a comparable result on a similarly
small, similarly shaped bioprocess dataset: their own symbolic-regression
model picked up an inhibiting effect of product concentration on biomass
growth that the real process did not have, and noted "slight numerical
discrepancies" between its recovered growth surface and the real one, even
after correctly capturing the overall trend
<a href="#ref-forster">[4]</a>. The aim here is to show that a comparable check runs,
end to end, inside `hybrax.train`, at a smaller scale than SR-KAN's own
hyperparameter-tuned pipeline.

## Gotchas

- **Inputs are `tanh`-bounded before hitting the RBF grid.** An untransformed
  input that wandered past the fitted centers' range would extrapolate through
  only the base `SiLU` term, silently losing the spline's local detail.
- **`centers` is a `frozen_field()`, never trained.** The grid is fixed; only
  `base_w` and `spline_c`, the edge functions' actual shape, move under
  gradient descent.
- **`l2`'s near-zero initialization is what keeps the ODE flat at step 0.**
  Removing it does not error, it just starts training from a wild, effectively
  random rate prediction.
- **`hidden` and `grid` are real capacity knobs**, exactly like a network's
  width. Too few edges or basis functions and the model cannot represent the
  trajectory; too many and every edge pays for a wider, mostly redundant
  radial-basis expansion.
- **`prod_a`/`prod_b` break perfectly-flat-at-step-0 for all three rates**,
  since every rate reads through `l2`'s shared sum. They start non-zero on
  purpose (see above), so each rate starts very close to flat rather than
  exactly flat.

## See Also

The full, runnable example (`custom.py`, configs, data) lives in
`examples/gallery_kan/` at the repo root, no docs build required. This page's own
executed run is at `./source/_data/out/runs/gallery_kan/`.

- [The Reaction Module](../train/reaction_module.md): `auxiliary`, and
  everything else a `UserReactionModule` can return.
- [Gaussian process reaction module](gaussian_process.md): another
  reaction-module architecture, with its own way of reading out what it
  learned.
- [Mechanistic Models](mechanistic_rates.md): a reaction module built from
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
4. <a id="ref-forster"></a>Forster, T., Vázquez, D., Müller, C., &
   Guillén-Gosálbez, G. (2024). Machine learning uncovers analytical kinetic
   models of bioprocesses. *Chemical Engineering Science*, 300, 120606.
   [https://doi.org/10.1016/j.ces.2024.120606](https://doi.org/10.1016/j.ces.2024.120606)
