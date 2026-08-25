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
# 4. Custom Models
<!-- UNLOCK -->

> Replace the two defaults that matter most (the network that predicts rates, and the
> scaling) and measure what it bought you.

Everything you customise in `hybrax.train` lives in one optional file, `custom.py`. `hybrax.train`
looks in it for functions with specific names; anything it does not find falls back to a
default. There is no registration and no base class to inherit for the file itself: it
is just a module.

```{code-cell} ipython3
:tags: [remove-cell]

import os, shutil, subprocess, sys
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/tutorial_04").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
EXAMPLE = Path("../../../examples/tutorial_04_your_first_custom_py").resolve()
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
hxt_cli("prepare", "--config", "prepare-config.json",
         "--output-dir", "prepared", "--overwrite")

import numpy as np
import pandas as pd
import hybrax.format as hxf

_collection = hxf.serialization.load_process_collection(WORK / "data.json")

def r2_by_target(run_dir):
    """Physical-space R^2 per target, averaged over processes. Scale-free, so
    it is the fair way to compare a scaled run against an unscaled one: their
    SCL losses live in different units and are not otherwise comparable."""
    df = pd.read_csv(WORK / run_dir / "predictions.csv")
    per_target = {}
    for name, process in _collection.processes.items():
        proc_df = df[df["process"] == name]
        t_pred = proc_df["t"].to_numpy()
        for species in ("biomass", "glucose", "product"):
            comp = process.reactor_medium.components[species].concentration
            t_meas = np.asarray(comp.times)
            y_meas = np.asarray(comp.values)
            y_pred = np.interp(t_meas, t_pred, proc_df[f"c_{species}"].to_numpy())
            ss_res = np.sum((y_meas - y_pred) ** 2)
            ss_tot = np.sum((y_meas - y_meas.mean()) ** 2)
            per_target.setdefault(species, []).append(1 - ss_res / ss_tot)
    return {k: float(np.mean(v)) for k, v in per_target.items()}
```

## First: SCL and RAW

You cannot write a reaction module without this, and it is where nearly everyone's first
attempt goes wrong.

- **RAW** is physical units: g/L, litres, hours.
- **SCL** is scaled space, where every axis is roughly 1.

The ODE is **integrated in SCL**. A state vector holding biomass (~5), glucose (~10) and
cumulative feed (~0.001) is badly conditioned; gradients through the solve are dominated
by whichever axis happens to be largest. Dividing each axis by a characteristic magnitude
fixes that, and because the scaling is linear the *same* factor converts a value and its
time derivative, so one number per axis handles both states and rates.

:::{admonition} The double-scaling trap
:class: warning

Your network reads `inputs.SCL_*` (already-scaled values) so whatever it emits is
*already in SCL space*. Return it directly.

If you additionally call a `scale_*` helper on the output, it cancels against the
wrapper's unscale step on the way back to physical units, and your rates end up off by
the scale factor. Both conventions appear in the shipped examples depending on what the
network is defined to emit, which is exactly why this catches people. Rule of thumb: **if
the input was SCL, the output is SCL.**
:::

## The file

```{literalinclude} ../../../examples/tutorial_04_your_first_custom_py/custom.py
:language: python
:linenos:
```

Three things to notice.

**`trainable_field()`** is what exposes the MLP weights to the optimizer. This is the
whole partitioning mechanism: tagged fields are trainable, and **untagged array leaves
default to frozen**. There is no separate partition function to write.

**`super().__init__(**scale_kwargs)`** must be called. The reaction module is the single
source of truth for every scale in `hybrax.train` (the wrapper, the trainer and the loss
module all read them off it) so the base class needs them.

**`SCL_modeled_Inflows_rates=jnp.zeros(0)`** and **`SCL_modeled_Outflows_rates=jnp.zeros(0)`**
are required even though `demo_batch` has no modeled feeds or outflows. `ReactionOutputs`
has no default for either; omitting one is a `TypeError` at the first solve, not at import.

### What `estimate_all_scales` is actually doing

State scales are easy: how big does this species get. Rate scales need one step of
thought. The module emits an O(1) number; that number is multiplied by the rate scale to
become a physical rate. You want the resulting *state change over the run* to be about
the size of the state. For a specific rate, `d(c)/dt = q_c · biomass`, so

```
scale(q_c)  ~  scale(c) / (scale(biomass) · duration)
```

which is the one line in the middle of the hook. Get this wrong in the optimistic
direction and the concentrations blow up on the first solve, before any training happens.

## Train it

Point the config at the file: one extra key:

```json
{
  "data": { "prepared": "prepared" },
  "custom_py": "custom.py",
  "train": { "epochs": 300, "seed": 0, "learning_rate": 0.01 },
  "output": { "dir": "run_custom" }
}
```

The learning rate is ten times the default's. That is deliberate, and the point of the
comparison below: it is only *safe* to raise it because the state is scaled. Try the same
learning rate on the unscaled defaults and the solve diverges to `inf` within the first
few steps: conditioning is not a nice-to-have, it is what lets you use a normal
optimizer setting at all.

```{code-cell} ipython3
:tags: [remove-cell]

shutil.copy(EXAMPLE / "train-default.json", WORK / "train-default.json")
shutil.copy(EXAMPLE / "train-custom.json", WORK / "train-custom.json")
shutil.copy(EXAMPLE / "forward-config.json", WORK / "forward-config.json")
```

## Did it help?

Same data, same seed, same epoch budget. Two things differ: this `custom.py`, and the
learning rate it makes safe to use.

Loss values are not the fair comparison here: the default run's loss lives in raw g/L,
the scaled run's in relative units, and those are not the same number. The fair,
scale-free comparison is **R² in physical space**, computed by re-interpolating each
run's predictions back onto the actual measurement times:

```{code-cell} ipython3
:tags: [remove-input]

hxt_cli("train", "--config", "train-default.json", "--overwrite")
hxt_cli("train", "--config", "train-custom.json", "--overwrite")

root = WORK.parents[4]
print(f"run directories: ./{(WORK / 'run_default').relative_to(root)}"
      f" and ./{(WORK / 'run_custom').relative_to(root)}")
print("both self-contained: cp -r either one anywhere and keep working from the copy")

r2_default = r2_by_target("run_default")
r2_custom = r2_by_target("run_custom")

print(f"{'target':10s} {'defaults':>10s} {'custom.py':>10s}")
for name in ("biomass", "glucose", "product"):
    print(f"{name:10s} {r2_default[name]:10.4f} {r2_custom[name]:10.4f}")
```

Biomass and glucose are close either way (this dataset is small enough that the defaults
already fit them well. **Product is where the two diverge.** It is the smallest-magnitude
target (peaking around 0.5 g/L against glucose's 10 g/L), and with no scaling the loss is
implicitly dominated by whichever axis has the largest raw magnitude) glucose. Product's
error barely moves the unscaled loss, so the optimizer has little reason to fit it well.
Once every axis is normalised to O(1), each target pulls equally, and product stops being
the one that got left behind.

That is the actual argument for scaling: not "converges faster" in general, but "every
target gets a fair share of the gradient", which matters most for whichever quantity in
your dataset happens to be smallest.

```{code-cell} ipython3
:tags: [remove-input]

hxt_cli("forward", "--config", "forward-config.json",
         "--output-dir", "run_custom/forward", "--overwrite")
from IPython.display import Image
Image(filename=str(WORK / "run_custom/forward/plots/run_1.png"))
```

## Check what is actually being trained

Before a long run, confirm the optimizer sees what you think it does. A field you forgot
to tag is silently frozen and will simply never move.

```{code-cell} ipython3
:tags: [remove-input]

import hybrax.train as hxt
wrapper, cfg = hxt.model_load(str(WORK / "run_custom"))
hxt.print_trainable_structure(wrapper)
```

The other half of that question (*which array index is which species*) has its own
printer, and it is the fastest way to stop guessing at slicing:

```{code-cell} ipython3
:tags: [remove-input]

hxt.print_reaction_schema(wrapper)
```

That is where `in_size=self.n_modeled_RMCs` and `out_size=self.n_modeled_BiologicalOde_rates`
came from, and it confirms the output order is `q_biomass, q_glucose, q_product`.

:::{admonition} A misspelled hook name is silent
:class: warning
hybrax looks up hooks by name with a plain attribute lookup. `build_reaction_modul`
(one letter short) is not an error: it is a silent fall back to the default MLP, and
your carefully written module never runs. If a change to `custom.py` seems to have had no
effect, check the spelling first. See
[Silent failures](../troubleshooting/silent_failures.md).
:::

## What you learned

- `custom.py` is plain Python; hooks are found by name, and missing ones use defaults.
- The ODE is integrated in **SCL** space; if your input was SCL, your output is SCL.
- `trainable_field()` opts a field in; everything untagged is frozen.
- Scaling is optional, has no error when omitted, and buys you two things: an optimizer
  setting that would otherwise diverge, and a loss where your smallest-magnitude target
  isn't drowned out by your largest.
- Compare runs by **R² in physical space**, not by raw loss: an unscaled and a scaled
  run don't share loss units.

## What's next

You can run the full tutorial at `examples/tutorial_04_your_first_custom_py/run.py`.

- **[Tutorial 5](05_predict.md)**: use the trained model.
- Real kinetics instead of a bare MLP: [Gallery: mechanistic models](../gallery/mechanistic_rates.md).
- Every hook, with signatures: [Customization](../train/hooks_cheatsheet.md).
- More on scales: [Scaling](../train/scaling.md).
