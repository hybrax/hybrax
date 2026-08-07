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

# Tutorial 4: your first `custom.py`

> **In one sentence.** Replace the two defaults that matter most — the network that
> predicts rates, and the scaling — and measure what it bought you.
>
> **You need this if** you want to control the model. **You can skip it if** you are only
> ever going to use the defaults, which on real data you are not.

Everything you customise in bp-train lives in one optional file, `custom.py`. bp-train
looks in it for functions with specific names; anything it does not find falls back to a
default. There is no registration and no base class to inherit for the file itself — it
is just a module.

```{code-cell} ipython3
:tags: [remove-cell]

import os, re, shutil, subprocess, sys, textwrap
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/tutorial_04").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
shutil.copy(Path("../_data/out/demo_batch/data.json").resolve(), WORK / "data.json")
shutil.copy(Path("_files/tutorial_04_custom.py").resolve(), WORK / "custom.py")

ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "BP_TRAIN_DEVICES": "1",
       "MPLBACKEND": "Agg"}

def bp_train(*args):
    proc = subprocess.run([sys.executable, "-m", "bp_train.cli", *args],
                          cwd=WORK, env=ENV, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout + proc.stderr

def losses(text):
    m = re.search(r"first_mean_loss=([\d.eE+-]+) last_mean_loss=([\d.eE+-]+)", text)
    return float(m.group(1)), float(m.group(2))

(WORK / "prepare-config.json").write_text(
    '{ "prepare": { "raw_input": "data.json" } }\n')
bp_train("prepare", "--config", "prepare-config.json",
         "--output-dir", "prepared", "--overwrite")
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
time derivative — so one number per axis handles both states and rates.

:::{admonition} The double-scaling trap
:class: warning

Your network reads `inputs.SCL_*` — already-scaled values — so whatever it emits is
*already in SCL space*. Return it directly.

If you additionally call a `scale_*` helper on the output, it cancels against the
wrapper's unscale step on the way back to physical units, and your rates end up off by
the scale factor. Both conventions appear in the shipped examples depending on what the
network is defined to emit, which is exactly why this catches people. Rule of thumb: **if
the input was SCL, the output is SCL.**
:::

## The file

```{literalinclude} _files/tutorial_04_custom.py
:language: python
:linenos:
```

Three things to notice.

**`trainable_field()`** is what exposes the MLP weights to the optimizer. This is the
whole partitioning mechanism: tagged fields are trainable, and **untagged array leaves
default to frozen**. There is no separate partition function to write.

**`super().__init__(**scale_kwargs)`** must be called. The reaction module is the single
source of truth for every scale in bp-train — the wrapper, the trainer and the loss
module all read them off it — so the base class needs them.

**`SCL_modeled_FVCs_rates=jnp.zeros(0)`** is required even though `demo_batch` has no
modeled feeds. `ReactionOutputs` has no default for it; omitting it is a `TypeError` at
the first solve, not at import.

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

Point the config at the file — one extra key:

```json
{
  "data": { "prepared": "prepared" },
  "custom_py": "custom.py",
  "train": { "epochs": 300, "seed": 0 },
  "output": { "dir": "run_custom" }
}
```

```{code-cell} ipython3
:tags: [remove-cell]

(WORK / "train-default.json").write_text(textwrap.dedent("""\
    {
      "data": { "prepared": "prepared" },
      "train": { "epochs": 300, "seed": 0 },
      "output": { "dir": "run_default" }
    }
    """))
(WORK / "train-custom.json").write_text(textwrap.dedent("""\
    {
      "data": { "prepared": "prepared" },
      "custom_py": "custom.py",
      "train": { "epochs": 300, "seed": 0 },
      "output": { "dir": "run_custom" }
    }
    """))
```

## Did it help?

Same data, same seed, same epoch count — the only difference is this `custom.py`.

```{code-cell} ipython3
:tags: [remove-input]

first_d, last_d = losses(bp_train("train", "--config", "train-default.json",
                                  "--overwrite", "--no-plot"))
first_c, last_c = losses(bp_train("train", "--config", "train-custom.json",
                                  "--overwrite"))

print(f"{'':22s} {'initial loss':>14s} {'final loss':>12s}")
print(f"{'defaults (no scaling)':22s} {first_d:14.4f} {last_d:12.4f}")
print(f"{'with custom.py':22s} {first_c:14.4f} {last_c:12.4f}")
print()
print(f"initial loss is {first_d / first_c:.0f}x smaller before a single step is taken")
print(f"final loss is   {last_d / last_c:.1f}x smaller")
```

The interesting number is the **initial** one. Nothing has been learned yet — that gap is
purely conditioning. The default run spends much of its budget climbing out of a badly
scaled starting point; the scaled run begins near the answer.

```{code-cell} ipython3
:tags: [remove-input]

from IPython.display import Image
Image(filename=str(WORK / "run_custom/run_1.png"))
```

## Check what is actually being trained

Before a long run, confirm the optimizer sees what you think it does. A field you forgot
to tag is silently frozen and will simply never move.

```{code-cell} ipython3
:tags: [remove-input]

import bp_train
wrapper, cfg = bp_train.model_load(str(WORK / "run_custom"))
bp_train.print_trainable_structure(wrapper)
```

The other half of that question — *which array index is which species* — has its own
printer, and it is the fastest way to stop guessing at slicing:

```{code-cell} ipython3
:tags: [remove-input]

bp_train.print_reaction_schema(wrapper)
```

That is where `in_size=self.n_modeled_RMCs` and `out_size=self.n_modeled_BiologicalOde_rates`
came from, and it confirms the output order is `q_biomass, q_glucose, q_product`.

:::{admonition} A misspelled hook name is silent
:class: warning
bp-train looks up hooks by name with a plain attribute lookup. `build_reaction_modul`
(one letter short) is not an error — it is a silent fall back to the default MLP, and
your carefully written module never runs. If a change to `custom.py` seems to have had no
effect, check the spelling first. See
[Silent failures](../troubleshooting/silent_failures.md).
:::

## What you learned

- `custom.py` is plain Python; hooks are found by name, and missing ones use defaults.
- The ODE is integrated in **SCL** space; if your input was SCL, your output is SCL.
- `trainable_field()` opts a field in; everything untagged is frozen.
- Scaling is optional, has no error when omitted, and matters a lot.

## What's next

- **[Tutorial 5](05_predict.md)** — use the trained model.
- Real kinetics instead of a bare MLP: [Gallery: structured rate laws](../gallery/structured_rates.md).
- Every hook, with signatures: [custom.py at a glance](../train/hooks_cheatsheet.md).
- More on scales: [Scaling](../train/scaling.md).
