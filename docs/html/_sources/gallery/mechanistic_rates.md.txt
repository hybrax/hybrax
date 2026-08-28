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

# Mechanistic Models

> This page fits mechanistic kinetics instead of a generic neural network, using
> named constants that are easy to interpret directly. It also shows how some of
> those constants trade off against each other during fitting.

Every tutorial so far let an MLP discover the rates. Nothing requires that. A reaction
module is any function from the state to the rates: it can just as easily be the kinetic
law you already believe in, with a handful of trainable scalars instead of a network.

The walkthrough below shows the file in pieces, next to the reasoning for each one.

```{code-cell} ipython3
:tags: [remove-cell]

import json, os, shutil, subprocess, sys
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_structured_rates").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
EXAMPLE = Path("../../../examples/gallery_mechanistic_rates").resolve()
shutil.copy(EXAMPLE / "data.json", WORK / "data.json")
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
```

## The Reaction Module

```{literalinclude} ../../../examples/gallery_mechanistic_rates/custom.py
:language: python
:linenos:
:lines: 21-68
```

Three things worth noting.

**Every constant is `jnp.exp(log_x)`.** An unconstrained optimizer can push a plain
trainable scalar negative, and a negative `Ks` or `Y_xs` makes the kinetics nonsensical.
Training the *log* of each constant is the cheapest way to impose positivity: no
clipping, no penalty term, the constraint is structural.

**Uptake is gated by the same saturation term as growth.** `q_glucose` includes `sigma`
in both its growth-linked and maintenance-linked parts, so uptake tapers smoothly as
glucose depletes rather than being driven to some fixed rate and clipped afterwards. This
is the mechanistic-modeling equivalent of the "concentrations must not go negative"
problem in [Dense losses](dense_loss.md): here it is built into the rate law instead of
enforced by a penalty.

**State indices are read off the assembled ODE, never hard-coded**:
`build_reaction_module` looks up `names.index("glucose")` to find each state's
position. If someone reorders the dataset's components, this still works.

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

hxt_cli("forward", "--config", "forward-config.json",
         "--output-dir", "run/forward", "--overwrite")
from IPython.display import Image
Image(filename=str(WORK / "run/forward/plots/run_1.png"))
```

## Did It Recover the True Parameters?

The dataset was simulated from known kinetics: nobody told the model this while
training.

```{code-cell} ipython3
:tags: [remove-input]

import jax.numpy as jnp
import hybrax.train as hxt

wrapper, cfg = hxt.model_load(str(WORK / "run"))
rm = wrapper.reaction_module
truth = json.loads(Path("../../../examples/gallery_mechanistic_rates/ground_truth.json").read_text())

fitted = {
    "mu_max": float(jnp.exp(rm.log_mu_max)),
    "Ks":     float(jnp.exp(rm.log_Ks)),
    "Y_XS":   float(jnp.exp(rm.log_Yxs)),
    "m_s":    float(jnp.exp(rm.log_ms)),
    "alpha":  float(jnp.exp(rm.log_alpha)),
    "beta":   float(jnp.exp(rm.log_beta)),
}

print(f"{'parameter':10s} {'fitted':>10s} {'true':>10s}")
for name in truth:
    print(f"{name:10s} {fitted[name]:10.4f} {truth[name]:10.4f}")
```

`mu_max` and `Y_XS` (the parameters that dominate the exponential growth phase, where
most of the data's information lives) come back close to their true values. `Ks`, `m_s`,
`alpha` and `beta` land further from their true values, reflecting the identifiability
trade-off explained in the next section.

## Why the Rest Don't Match, and Why the Fit Is Still Good

```{code-cell} ipython3
:tags: [remove-input]

fitted_combo = fitted["alpha"] * fitted["mu_max"] + fitted["beta"]
true_combo = truth["alpha"] * truth["mu_max"] + truth["beta"]
print(f"alpha*mu_max + beta   fitted={fitted_combo:.4f}   true={true_combo:.4f}")
```

During the growth phase (where nearly every measurement sits), glucose is saturating, so
`sigma ≈ 1` and the product rate collapses to `q_product ≈ alpha·mu_max + beta`: a single
number. The data constrains *that combination* tightly; it says almost nothing about how
much of it comes from `alpha` versus `beta` individually. Two different splits that sum to
the same combination fit equally well, so the optimizer finds whichever split its
initialisation happened to favour: this is a textbook **structural identifiability**
problem baked into the data itself.

`Ks` is a milder version of the same story: `demo_batch` never lingers at low, resolving
glucose concentrations (the culture consumes it and moves on) so there is little data
constraining exactly where the saturation curve bends.

That is the practical payoff of naming the rate law's terms explicitly: the ambiguity
shows up as a number you can name, `alpha`, `beta`, `Ks`, rather than staying invisible
inside unlabeled weights. You know exactly which two experiments would resolve it: a
longer low-glucose tail for `Ks`, a run with product measured *after* growth stops for
`alpha`/`beta`.

## Gotchas

- **Positivity via `log`.** A `jnp.clip` on a rate is a dead gradient region;
  the log-parameterisation has none.
- **Multiple valid initializations exist.** Try a few seeds if a parameter estimate looks
  implausible: you may be seeing one identifiability trade-off rather than a wrong fit.
- **Adding a state that *is* well constrained resolves the ambiguity.** If DO or another
  process variable independently informs the split, adding it as a modeled PV changes the
  identifiability picture. See [A Modeled Process Variable](modeled_pv.md) for what
  declaring and training one actually looks like.

## See Also

The full, runnable example (`custom.py`, configs, data) lives in
`examples/gallery_mechanistic_rates/` at the repo root, no docs build required. This
page's own executed run is at `./source/_data/out/runs/gallery_mechanistic_rates/`.

- [The Reaction Module](../train/reaction_module.md): the general SCL/RAW contract this
  module follows.
- [Tutorial 4](../tutorials/04_your_first_custom_py.md): the MLP version of the same
  problem, for comparison.
- [Dense losses](dense_loss.md): bounds and smoothness as an alternative way to encode
  what you know about the biology.
- [Glutamine Degradation](glutamine_decay.md): a single declared rate feeding two coupled
  derivatives at once, the same "did it recover the true parameters" check.
