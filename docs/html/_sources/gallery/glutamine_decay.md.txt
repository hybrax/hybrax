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

# Glutamine Degradation

> This page declares one physical rate that feeds two different derivatives at once: a
> sink in one, a source in the other. Training recovers that single shared value from
> data alone.

This page is inspired by Ulonska et al. (2018) <a href="#ref-ulonska">[1]</a>, whose CHO
cell culture model includes glutamine's own spontaneous, non-enzymatic decomposition to
glutamate and ammonia at a real, fitted first-order rate of 0.0036 1/h (roughly an
eight-day half-life). `r_Gln` below reproduces that rate exactly as this page's synthetic
ground truth, feeding two derivatives at once the same way the paper's own model does.
This page simplifies the paper in three ways: it keeps only the decomposition term in
NH4's balance (the paper's version also has two other, unrelated production terms), it
tracks glutamine and NH4 in mol/L so the same `r_Gln` value drives both derivatives
without a separate yield constant (the paper's own g/L-plus-yield-constant approach is
equally valid), and it replaces the paper's saturating Monod-form uptake rate with a
plain constant rate ([Mechanistic Models](mechanistic_rates.md) already covers Monod-form
kinetics in depth).

The walkthrough below shows the file in pieces, next to the reasoning for each one.

```{code-cell} ipython3
:tags: [remove-cell]

import json, os, shutil, subprocess, sys
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_glutamine_decay").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
EXAMPLE = Path("../../../examples/gallery_glutamine_decay").resolve()
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

import numpy as np
import pandas as pd
import jax.numpy as jnp
import hybrax.format as hxf
import hybrax.train as hxt

_collection = hxf.serialization.load_process_collection(WORK / "data.json")

def r2_by_target(run_dir):
    """Pooled R2: concatenate every process's residuals/variance before
    dividing, so one process's own range can't dominate the score."""
    df = pd.read_csv(WORK / run_dir / "predictions.csv")
    per_target = {s: ([], []) for s in ("biomass", "Gln", "NH4")}
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

## The Rate Law, Declared

Everything this page's kinetics need lives in the dataset's own `reaction_ode`
block. This is exactly what was declared when the dataset was built:

```{code-cell} ipython3
process = _collection.processes["run_1"]

glutamine_ode = hxf.ReactionOde(
    rates={"q_biomass": (None, None), "q_Gln": (None, None),
           "r_Gln": (None, None)},
    derivatives={
        "biomass": "q_biomass * biomass",
        "Gln": "-q_Gln * biomass - r_Gln * Gln",
        "NH4": "r_Gln * Gln",
    },
)
print("matches the real declared reaction_ode:", glutamine_ode == process.reaction_ode)
```

`Gln`'s derivative has two terms: `-q_Gln * biomass`, ordinary uptake tied to growth,
and `-r_Gln * Gln`, the chemical degradation this page is about. `NH4`'s derivative is just
`r_Gln * Gln`, the same `r_Gln` symbol, reused verbatim. There is no wiring connecting
the two beyond that shared name: `hybrax.format` parses each expression independently, so
whatever value training settles on for `r_Gln` has to simultaneously explain
glutamine's own decline *and* NH4's rise, from the same number.

The full assembled right-hand side, biological and physical halves together:

```{code-cell} ipython3
hxf.print_rhs_ode(process)
```

No volume changes at all here, a true batch, so the Feed/Dilution columns are empty
and the Volume block just confirms `V` stays constant.

That reuse is also why the page's units matter: `Gln` and `NH4` both being `mol/L`
means `r_Gln * Gln` is already in the right units for both derivatives, mol/(L·h) for
`Gln`'s loss and mol/(L·h) for `NH4`'s gain. Mixing a `g/L` state and a `mol/L` state
inside one additive expression is rejected outright, *unless* each term is
individually scaled by its own declared rate: `-q_Gln * biomass - r_Gln * Gln` is fine
even though `biomass` is `g/L` and `Gln` is `mol/L`, because `q_Gln` and `r_Gln` are
each trusted to carry whatever unit bridges their own term. See
[The Bioprocess ODE](../format/bioprocess_ode.md#writing-your-own) for the general rule.

## The Reaction Module

```{literalinclude} ../../../examples/gallery_glutamine_decay/custom.py
:language: python
:linenos:
:lines: 19-43
```

Three log-parameterized scalars, no kinetic structure, no state read at all:
`__call__` ignores `inputs` entirely and returns the same three constants every step.
All of this page's actual complexity is in the derivative strings above.

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

Glutamine declines smoothly across the full 120 h window, NH4 rises to match, and
biomass grows independently of both: exactly the shape the declared derivatives
describe.

## Did the Shared Rate Recover Correctly?

```{code-cell} ipython3
:tags: [remove-input]

wrapper, cfg = hxt.model_load(str(WORK / "run"))
rm = wrapper.reaction_module
truth = json.loads((WORK / "ground_truth.json").read_text())

fitted = {
    "q_biomass": float(jnp.exp(rm.log_q_biomass)),
    "q_Gln":     float(jnp.exp(rm.log_q_Gln)),
    "r_Gln":     float(jnp.exp(rm.log_r_Gln)),
}
print(f"{'parameter':10s} {'fitted':>12s} {'true':>12s}")
for name in truth:
    print(f"{name:10s} {fitted[name]:12.6g} {truth[name]:12.6g}")
```

All three come back close to their true values, `r_Gln` included: the same number
that explains glutamine's own decline also correctly predicts NH4's rise, because
training only ever fit the single number declared in `reaction_ode.rates`. That is
the actual demonstration this page exists to make;
everything above it is setup.

## Gotchas

- **Different-unit states can only be added if each is individually scaled by its
  own declared rate.** A bare `biomass - Gln` (no rate on either side) is rejected;
  `-q_Gln * biomass - r_Gln * Gln` is not, because each term's own rate is trusted to
  carry whatever unit bridges it. See [The Bioprocess ODE](../format/bioprocess_ode.md#writing-your-own).
- **A slow rate needs a long enough window to be identifiable.** `r_Gln`'s ~8-day
  half-life would leave almost no visible trace over a 10-15 h batch window; this
  page's 120 h duration exists specifically so the effect is separable from noise,
  the same lesson [OptFed](optfed.md#gotchas)'s Eyring-identifiability Gotcha teaches
  for a different rate.
- **This page's unusual rate shape (one rate feeding two derivatives) needs no custom
  reaction module at all, technically.** `hybrax.train`'s default reaction module sizes
  itself generically from the declared rate vector, so it would train against this
  exact dataset with zero code. This page skips it on purpose: the default module is
  an opaque MLP, with no single fitted `r_Gln` scalar to check against the paper's
  own value, which is the entire verification this page is built around.

## See Also

The full, runnable example (`custom.py`, configs, data) lives in
`examples/gallery_glutamine_decay/` at the repo root, no docs build required. This
page's own executed run is at `./source/_data/out/runs/gallery_glutamine_decay/`.

- [The Bioprocess ODE](../format/bioprocess_ode.md): `reaction_ode`, `rates`,
  `derivatives`, and the unit-consistency rule this page relies on.
- [Mechanistic Models](mechanistic_rates.md): the same "did it recover the true
  parameters" question, asked of a Monod-form rate law instead.
- [OptFed](optfed.md): a rate law with real kinetic structure, and the same
  "identifiability needs the true value inside the sampled range" lesson.
- [The Reaction Module](../train/reaction_module.md): `trainable_field` and
  everything else a `RateModule` can return.

## References

1. <a id="ref-ulonska"></a>Ulonska, S., Kroll, P., Fricke, J., Clemens, C., Voges,
   R., Müller, M. M., & Herwig, C. (2018). Workflow for Target-Oriented
   Parametrization of an Enhanced Mechanistic Cell Culture Model. *Biotechnology
   Journal*, 13(4), 1700395.
   [https://doi.org/10.1002/biot.201700395](https://doi.org/10.1002/biot.201700395)
