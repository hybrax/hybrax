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

# Glutamine degradation

> **Demonstrates.** One physical rate, declared once in `biological_ode.rates`,
> feeding two different derivatives at once, a sink in one, a source in the other,
> and `hybrax.train` recovering that single shared number from data alone.

Inspired by Ulonska, Kroll, Fricke, Clemens, Voges, Müller & Herwig 2018
<a href="#ref-ulonska">[1]</a>, *"Workflow for Target-Oriented Parametrization of an
Enhanced Mechanistic Cell Culture Model,"* whose CHO cell culture model includes
glutamine's own spontaneous, non-enzymatic decomposition to glutamate and ammonia
(Eq. 18/20), at a real, fitted first-order rate: `rNH4,gln = 0.0036 1/h` (Table 1, an
eight-day half-life). This page reproduces that rate exactly: `r_Gln` below is the
paper's own cited value, used as this page's synthetic ground truth, and it appears in
two different derivatives at once, the same coupling the paper's own equations
describe.

Two things are reduced from the paper's own version, disclosed plainly:

- NH4's own balance (Eq. 20) has three source terms: metabolic production tied to
  glutamine consumption (`qNH4 * VCC`, where `qNH4 = YNH4/gln * qgln`, Eq. 10), release
  from a feed component's own degradation, and the chemical decomposition of glutamine
  (`YNH4,gln * rNH4,gln * cgln`). This page keeps only the third: that is the one rate
  this page is actually about, and the other two would add unrelated terms on top of
  the point being made.
- Glutamine and NH4 are tracked in mol/L here, unlike the rest of this site's g/L
  convention, so glutamine's loss and NH4's gain from decomposition can share exactly
  the same literal number, `r_Gln`, with no separate yield constant. The paper's own
  g/L-plus-yield-constant approach (`YNH4,gln = 0.12 g/g`) is equally valid: that
  yield exists purely because glutamine (146 g/mol) and ammonia (17 g/mol) have
  different molar masses despite decomposing 1:1. This page just makes the other
  valid choice for its own demo.
- The paper's own Monod-form `q_Gln` (saturating in glutamine concentration) is
  replaced with a plain constant specific rate: [Mechanistic models](mechanistic_rates.md)
  already covers Monod-form kinetics in depth.

The walkthrough below shows the file in pieces, next to the reasoning for each one. For
the whole thing at once: to copy, diff against your own, or just read top to bottom:

:::{dropdown} Full `custom.py`
```{literalinclude} ../../../examples/gallery_glutamine_decay/custom.py
:language: python
:linenos:
```
:::

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

## The rate law, declared

Everything this page's kinetics need lives in the dataset's own `biological_ode`
block, not in code. This is exactly what was declared when the dataset was built:

```{code-cell} ipython3
process = _collection.processes["run_1"]

glutamine_ode = hxf.BiologicalOde(
    rates={"q_biomass": (None, None), "q_Gln": (None, None),
           "r_Gln": (None, None)},
    derivatives={
        "biomass": "q_biomass * biomass",
        "Gln": "-q_Gln * biomass - r_Gln * Gln",
        "NH4": "r_Gln * Gln",
    },
)
print("matches the real declared biological_ode:", glutamine_ode == process.biological_ode)
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

## The reaction module

```{literalinclude} ../../../examples/gallery_glutamine_decay/custom.py
:language: python
:linenos:
:lines: 19-43
```

Three log-parameterized scalars, no kinetic structure, no state read at all:
`__call__` ignores `inputs` entirely and returns the same three constants every step.
All of this page's actual complexity is in the derivative strings above, not here.

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

## Did the shared rate recover correctly?

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
training never saw two separate numbers to fit, only the one declared in
`biological_ode.rates`. That is the actual demonstration this page exists to make;
everything above it is setup.

## Gotchas

- **Different-unit states can only be added if each is individually scaled by its
  own declared rate.** A bare `biomass - Gln` (no rate on either side) is rejected;
  `-q_Gln * biomass - r_Gln * Gln` is not, because each term's own rate is trusted to
  carry whatever unit bridges it. See [The Bioprocess ODE](../format/bioprocess_ode.md#writing-your-own).
- **A slow rate needs a long enough window to be identifiable.** `r_Gln`'s ~8-day
  half-life would leave almost no visible trace over a 10-15 h batch window; this
  page's 120 h duration exists specifically so the effect is separable from noise,
  the same lesson [OptFed](optfed.md#gotchas)'s temperature-optimum Gotcha teaches
  for a different rate.
- **This page's unusual rate shape (one rate feeding two derivatives) needs no custom
  reaction module at all, technically.** `hybrax.train`'s default reaction module sizes
  itself generically from the declared rate vector, so it would train against this
  exact dataset with zero code. It is not used here on purpose: the default module is
  an opaque MLP, with no single fitted `r_Gln` scalar to check against the paper's
  own value, which is the entire verification this page is built around.

## See also

Run the example yourself at `./source/_data/out/runs/gallery_glutamine_decay/`.

- [The Bioprocess ODE](../format/bioprocess_ode.md): `biological_ode`, `rates`,
  `derivatives`, and the unit-consistency rule this page relies on.
- [Mechanistic models](mechanistic_rates.md): the same "did it recover the true
  parameters" question, asked of a Monod-form rate law instead.
- [OptFed](optfed.md): a rate law with real kinetic structure, and the same
  "identifiability needs the true value inside the sampled range" lesson.
- [The Reaction Module](../train/reaction_module.md): `trainable_field` and
  everything else a `UserReactionModule` can return.

## References

1. <a id="ref-ulonska"></a>Ulonska, S., Kroll, P., Fricke, J., Clemens, C., Voges,
   R., Müller, M. M., & Herwig, C. (2018). Workflow for Target-Oriented
   Parametrization of an Enhanced Mechanistic Cell Culture Model. *Biotechnology
   Journal*, 13(4), 1700395.
   [https://doi.org/10.1002/biot.201700395](https://doi.org/10.1002/biot.201700395)
