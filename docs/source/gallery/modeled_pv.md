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

# A modeled process variable

> **Demonstrates.** A process variable declared as a modeled (uncontrolled) state,
> with its own trained rate `r_glyco_frac`, and why a modeled process variable is
> never diluted by a feed event the way a modeled reactor component is.

Every other page in this gallery that touches a process variable uses it as a
**controlled** input: a known signal, like `temperature` in [OptFed](optfed.md) or
`media_blend_fraction` in [PLS-dFBA](pls_dfba.md), read by the reaction module but
never itself a trained state. This page is about the other kind: a process variable
with `is_controlled=False`, a real dynamic state integrated by the solver alongside
the reactor components, with its own rate the same way a component has `q_<name>`,
just named `r_<name>` instead.

The dataset pairs two independent, unrelated first-order states so that nothing but
the framework's own rules can explain a difference between them:

- `biomass`, an ordinary modeled reactor component, growing as `q_biomass * biomass`.
- `glyco_frac`, a modeled process variable standing in for a glycosylation-fraction
  quality attribute, decaying as `-r_glyco_frac * glyco_frac`. It never depends on
  `biomass`, and `biomass` never depends on it.

One large feed bolus lands midway through every run, big enough to double the working
volume. `biomass` concentration is diluted by it, visibly. `glyco_frac` is not,
because it is a fraction: adding diluent scales both a glycosylated and a
non-glycosylated pool by the same factor, so their ratio is untouched. hybrax.format
encodes exactly that rule, for any modeled process variable, regardless of what it
represents physically.

The walkthrough below shows the file in pieces, next to the reasoning for each one. For
the whole thing at once: to copy, diff against your own, or just read top to bottom:

:::{dropdown} Full `custom.py`
```{literalinclude} ../../../examples/gallery_modeled_pv/custom.py
:language: python
:linenos:
```
:::

```{code-cell} ipython3
:tags: [remove-cell]

import json, os, shutil, subprocess, sys
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_modeled_pv").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
EXAMPLE = Path("../../../examples/gallery_modeled_pv").resolve()
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
    per_target = {"biomass": ([], []), "glyco_frac": ([], [])}
    for name, process in _collection.processes.items():
        proc_df = df[df["process"] == name]
        t_pred = proc_df["t"].to_numpy()
        biomass = process.reactor_medium.components["biomass"].concentration
        glyco = process.process_variables["glyco_frac"].values
        for species, comp in (("biomass", biomass), ("glyco_frac", glyco)):
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

modeled_pv_ode = hxf.BiologicalOde(
    rates={"q_biomass": (0.0, None), "r_glyco_frac": (0.0, None)},
    derivatives={
        "biomass": "q_biomass * biomass",
        "glyco_frac": "-r_glyco_frac * glyco_frac",
    },
)
print("matches the real declared biological_ode:", modeled_pv_ode == process.biological_ode)
```

`glyco_frac` sits in `process.process_variables`, not `process.reactor_medium.components`,
with `is_controlled=False` and a real `TimeSeries` (an uncontrolled process variable
cannot hold a `StaticVariable`: a state with no time axis has nothing for the solver to
integrate against). That single field is the whole declaration; nothing else about
writing its rate law differs from writing a reactor component's.

```{code-cell} ipython3
hxf.print_rhs_ode(process)
```

Both `biomass` and `glyco_frac` show an empty Feed/Dilution column here, and that is
not the demonstration: that column is populated only by a **continuous** controlled or
modeled inflow/outflow, and this dataset's dilution event is a single discrete bolus.
A bolus is not part of the continuous right-hand side at all; it is a state jump the
solver applies between two integration segments, at the bolus's exact time, which is
why the Volume block below the derivatives shows it instead. That jump is where the
RMC/PV distinction actually happens, and it holds regardless of whether the dilution
arrives continuously or as one lump event.

## The reaction module

```{literalinclude} ../../../examples/gallery_modeled_pv/custom.py
:language: python
:linenos:
:lines: 19-40
```

Two log-parameterized scalars, no kinetic structure, no state read at all: `__call__`
ignores `inputs` entirely and returns the same two constants every step. All of this
page's actual complexity is in the derivative strings and the bolus event above, not
here.

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
    print(f"{name:12s} R2 = {value:.4f}")
```

```{code-cell} ipython3
:tags: [remove-input]

hxt_cli("forward", "--config", "forward-config.json",
         "--output-dir", "run/forward", "--overwrite")
from IPython.display import Image
Image(filename=str(WORK / "run/forward/plots/run_1.png"))
```

Read the top-left and middle-left panels together. `biomass` climbs smoothly, drops
sharply the instant the bolus lands (the same amount of biomass, suddenly divided by
twice the volume), then keeps climbing from the new, lower concentration. `glyco_frac`
crosses the exact same timestamp with no visible reaction at all: its own decline
never bends. Same bolus, same instant, two different modeled states, one physical rule
applied to only one of them. The bottom row shows why: `V_real` doubles in a single
step at the bolus, and that step is the only thing that touches `biomass`; `glyco_frac`
never appears in that bookkeeping.

## Did the rates recover correctly?

```{code-cell} ipython3
:tags: [remove-input]

wrapper, cfg = hxt.model_load(str(WORK / "run"))
rm = wrapper.reaction_module
truth = json.loads((WORK / "ground_truth.json").read_text())

fitted = {
    "q_biomass":    float(jnp.exp(rm.log_q_biomass)),
    "r_glyco_frac": float(jnp.exp(rm.log_r_glyco_frac)),
}
print(f"{'parameter':14s} {'fitted':>12s} {'true':>12s}")
for name in truth:
    print(f"{name:14s} {fitted[name]:12.6g} {truth[name]:12.6g}")
```

Both rates come back close to their true values: training saw the same bolus at the
same instant in every state, and only explained `biomass`'s step with a dilution term,
never with `q_biomass` itself, because that dilution term is not something the
reaction module controls at all.

## Gotchas

- **A modeled process variable needs `target_source: "combined"` in `train-config.json`
  once you actually want it trained.** Left on the `"auto"` default, this dataset's own
  `prepare`/`train` step silently picked `process_variables` only, training against
  `glyco_frac` and dropping `biomass` from the loss entirely, with nothing louder than a
  warning. Set `target_source` explicitly the moment a dataset has both kinds of
  targets; see [Configuration](../train/config.md#target_source).
- **A `UserReactionModule` with modeled process variables must return
  `SCALE_modeled_PVs` from `estimate_all_scales`.** It is not required for a
  PV-free dataset (it defaults to an empty scaler), so a reaction module copied from a
  page like [Glutamine decay](glutamine_decay.md) that never had one will raise a
  shape-mismatch error the moment a modeled PV is added; add the axis explicitly, as
  this page's `custom.py` does.
- **Dilution reaching a modeled process variable is not a silent possibility to guard
  against, it is architecturally unreachable.** `hybrax.format`'s continuous feed term
  only ever touches the RMC slice of the derivative vector, and the discrete bolus/sample
  jump in `hybrax.train` only ever touches the RMC slice of the state, with the same
  reasoning either way: a process variable is treated as intensive (a ratio or an
  observable), not as mass dissolved in a volume that can be diluted.
- **Every process still needs a `biomass` reactor component**, even on a page whose
  point is a process variable: `hybrax.format` expects one on every process, so
  `biomass` does double duty here as both the mandatory component and the diluted half
  of the contrast.

## See also

Run the example yourself at `./source/_data/out/runs/gallery_modeled_pv/`.

- [The Bioprocess ODE](../format/bioprocess_ode.md): `biological_ode`, `rates`,
  `derivatives`, and writing your own expressions.
- [Mechanistic models](mechanistic_rates.md): its own Gotchas section raises this
  exact idea (a well-constrained process variable resolving an identifiability
  problem) without building it; this page is that example.
- [The Reaction Module](../train/reaction_module.md): `trainable_field` and
  everything else a `UserReactionModule` can return, including `SCALE_modeled_PVs`.
- [Configuration](../train/config.md): `target_source` and the rest of the config
  schema.
