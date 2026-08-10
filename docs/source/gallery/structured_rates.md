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

# Structured rate laws

> **Demonstrates.** Mechanistic kinetics — Monod growth, Luedeking-Piret product
> formation — instead of a bare MLP, with named, trainable, physically interpretable
> constants. And where those constants trade off against each other.

Every tutorial so far let an MLP discover the rates. Nothing requires that. A reaction
module is any function from the state to the rates — it can just as easily be the kinetic
law you already believe in, with a handful of trainable scalars instead of a network.

```{code-cell} ipython3
:tags: [remove-cell]

import json, os, shutil, subprocess, sys, textwrap
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_structured_rates").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
shutil.copy(Path("../_data/out/demo_batch/data.json").resolve(), WORK / "data.json")
shutil.copy(Path("_files/structured_rates_custom.py").resolve(), WORK / "custom.py")

ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "BP_TRAIN_DEVICES": "1",
       "MPLBACKEND": "Agg"}

def bp_train(*args):
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
      "train": { "epochs": 600, "seed": 0, "learning_rate": 0.02 },
      "output": { "dir": "run" }
    }
    """))
```

## The reaction module

```{literalinclude} _files/structured_rates_custom.py
:language: python
:linenos:
:lines: 21-67
```

Three things worth noting.

**Every constant is `jnp.exp(log_x)`.** An unconstrained optimizer can push a plain
trainable scalar negative, and a negative `Ks` or `Y_xs` is not just wrong, it makes the
kinetics nonsensical. Training the *log* of each constant is the cheapest way to impose
positivity — no clipping, no penalty term, the constraint is structural.

**Uptake is gated by the same saturation term as growth.** `q_glucose` includes `sigma`
in both its growth-linked and maintenance-linked parts, so uptake tapers smoothly as
glucose depletes rather than being driven to some fixed rate and clipped afterwards. This
is the mechanistic-modeling equivalent of the "concentrations must not go negative"
problem in [Dense losses](dense_loss.md) — here it is built into the rate law instead of
enforced by a penalty.

**State indices are read off the assembled ODE, never hard-coded** —
`names.index("glucose")` in `build_reaction_module`, not a bare `1`. If someone reorders
the dataset's components, this still works.

## Training

```{code-cell} ipython3
:tags: [remove-input]

bp_train("prepare", "--config", "prepare-config.json",
         "--output-dir", "prepared", "--overwrite")
out = bp_train("train", "--config", "train-config.json", "--overwrite")
print([l for l in out.splitlines() if "training complete" in l][0])
```

```{code-cell} ipython3
:tags: [remove-input]

from IPython.display import Image
Image(filename=str(WORK / "run/run_1.png"))
```

## Did it recover the true parameters?

The dataset was simulated from known kinetics — nobody told the model this while
training.

```{code-cell} ipython3
:tags: [remove-input]

import jax.numpy as jnp
import bp_train

wrapper, cfg = bp_train.model_load(str(WORK / "run"))
rm = wrapper.reaction_module
truth = json.loads(Path("../_data/out/demo_batch/ground_truth.json").read_text())

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

`mu_max` and `Y_XS` — the parameters that dominate the exponential growth phase, where
most of the data's information lives — come back close to their true values. `Ks`, `m_s`,
`alpha` and `beta` do not, and that is not a bug in the fit.

## Why the rest don't match — and why the fit is still good

```{code-cell} ipython3
:tags: [remove-input]

fitted_combo = fitted["alpha"] * fitted["mu_max"] + fitted["beta"]
true_combo = truth["alpha"] * truth["mu_max"] + truth["beta"]
print(f"alpha*mu_max + beta   fitted={fitted_combo:.4f}   true={true_combo:.4f}")
```

During the growth phase — where nearly every measurement sits — glucose is saturating, so
`sigma ≈ 1` and the product rate collapses to `q_product ≈ alpha·mu_max + beta`: a single
number. The data constrains *that combination* tightly; it says almost nothing about how
much of it comes from `alpha` versus `beta` individually. Two different splits that sum to
the same combination fit equally well, so the optimizer finds whichever split its
initialisation happened to favour — this is a textbook **structural identifiability**
problem, not an optimizer failure.

`Ks` is a milder version of the same story: `demo_batch` never lingers at low, resolving
glucose concentrations — the culture consumes it and moves on — so there is little data
constraining exactly where the saturation curve bends.

This is the actual, practical argument for structured rate laws over an MLP: an MLP would
have absorbed this same ambiguity invisibly, inside weights with no physical meaning. Here
it is visible, in a number you can name, and you know exactly which two experiments would
resolve it — a longer low-glucose tail for `Ks`, a run with product measured *after*
growth stops for `alpha`/`beta`.

## Gotchas

- **Positivity via `log`, not `clip`.** A `jnp.clip` on a rate is a dead gradient region;
  the log-parameterisation has none.
- **Multiple valid initializations exist.** Try a few seeds if a parameter estimate looks
  implausible — you may be seeing one identifiability trade-off rather than a wrong fit.
- **Adding a state that *is* well constrained resolves the ambiguity.** If DO or another
  process variable independently informs the split, adding it as a modeled PV changes the
  identifiability picture.

## See also

- [The reaction module](../train/reaction_module.md) — the general SCL/RAW contract this
  module follows.
- [Tutorial 4](../tutorials/04_your_first_custom_py.md) — the MLP version of the same
  problem, for comparison.
- [Dense losses](dense_loss.md) — bounds and smoothness as an alternative way to encode
  what you know about the biology.
