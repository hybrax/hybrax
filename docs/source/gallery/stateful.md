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

# Stateful reaction modules

> **Demonstrates.** A reaction module with its own memory — a continuous-time LSTM whose
> hidden and cell state are integrated as extra ODE dimensions — and the opt-in that
> guards it.

Every module so far predicts rates from the *current* state alone. That is a real
modeling assumption: it says the biology has no memory beyond what is currently
measured. A **stateful** module relaxes that — it carries its own latent state through
the solve, so the rates can depend on where the process has been, not just where it is.

## Why this is a bigger change than it looks

bp-train does not run a discrete recurrent network beside the ODE solver. It **turns the
recurrent cell into a continuous-time ODE**: the latent state `h` is an extra integrated
dimension, and its derivative is the discrepancy between `h` and whatever the cell would
have jumped to next — `d(h)/dt = cell(input, h) - h`. As training pulls the residual to
zero, `h` tracks the same trajectory the discrete cell would have taken, but now it is a
proper flow that Diffrax can integrate and differentiate through like any other state.

This is exactly how bp-train's own built-in stateful model works —
`DefaultStatefulReactionModule` in `bp_train/defaults.py` uses this trick with a GRU
cell. What follows applies the identical trick to an LSTM, to show it is a general
pattern, not something specific to GRUs.

```{code-cell} ipython3
:tags: [remove-cell]

import os, re, shutil, subprocess, sys, textwrap
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_stateful").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
shutil.copy(Path("../_data/out/demo_batch/data.json").resolve(), WORK / "data.json")
shutil.copy(Path("_files/stateful_custom.py").resolve(), WORK / "custom.py")

ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "BP_TRAIN_DEVICES": "1",
       "MPLBACKEND": "Agg"}

def bp_train(*args, check=True):
    proc = subprocess.run([sys.executable, "-m", "bp_train.cli", *args],
                          cwd=WORK, env=ENV, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout + proc.stderr

(WORK / "prepare-config.json").write_text(
    '{ "prepare": { "raw_input": "data.json" } }\n')
bp_train("prepare", "--config", "prepare-config.json",
         "--output-dir", "prepared", "--overwrite")
```

## The module

```{literalinclude} _files/stateful_custom.py
:language: python
:linenos:
:lines: 29-66
```

An LSTM cell has *two* pieces of memory, not one — the hidden state and the cell state —
so `SCL_latent` here holds both, concatenated. Everything else is the same trick as
above: compute where the cell would jump to, emit the difference as the derivative, and
read the rates out of the *current* hidden state (not the target) so the module stays
consistent with every other input it receives at time `t`.

`ReactionOutputs` gains a field we have not used before, `SCL_latent_derivative`, aligned
with `SCL_latent`. It defaults to an empty array — every non-stateful module in these docs
has been quietly relying on that default.

## The opt-in

Declaring a nonzero `SCALE_latent` is enough to make `n_latent > 0`, and that alone is not
allowed to train silently:

```{code-cell} ipython3
:tags: [remove-cell]

(WORK / "train-no-optin.json").write_text(textwrap.dedent("""\
    {
      "data": { "prepared": "prepared" },
      "custom_py": "custom.py",
      "train": { "epochs": 5, "seed": 0 },
      "output": { "dir": "run_no_optin" }
    }
    """))
```

```{code-cell} ipython3
:tags: [remove-input]

out = bp_train("train", "--config", "train-no-optin.json", "--overwrite", "--no-plot",
               check=False)
print([l for l in out.splitlines() if "ValueError" in l][-1])
```

That is deliberate: a latent state changes what the model *is* — it is no longer a pure
function of the physical state — and that is a large enough change in what "the model"
means that bp-train wants it to be a decision, not a side effect of adding a field.

```json
{ "train": { "allow_stateful_models": true } }
```

## Training it

```{code-cell} ipython3
:tags: [remove-cell]

(WORK / "train.json").write_text(textwrap.dedent("""\
    {
      "data": { "prepared": "prepared" },
      "custom_py": "custom.py",
      "train": { "epochs": 600, "seed": 0, "learning_rate": 0.01,
                "allow_stateful_models": true },
      "output": { "dir": "run" }
    }
    """))
```

```{code-cell} ipython3
:tags: [remove-input]

out = bp_train("train", "--config", "train.json", "--overwrite")
print([l for l in out.splitlines() if "training complete" in l][0])
```

```{code-cell} ipython3
:tags: [remove-input]

from IPython.display import Image
Image(filename=str(WORK / "run/run_1.png"))
```

## Checking the latent dimension actually registered

```{code-cell} ipython3
:tags: [remove-input]

import bp_train
wrapper, cfg = bp_train.model_load(str(WORK / "run"))
print("n_latent =", wrapper.reaction_module.n_latent)
bp_train.print_trainable_structure(wrapper)
```

`n_latent` is `2 * n_hidden` — hidden and cell state together — and both the LSTM's gates
and the readout head show up as trainable, exactly like any other reaction module.

## When this is worth the extra machinery

Not on `demo_batch` — a memoryless module already fits this data well, because nothing
about a simple batch culture actually depends on history beyond the current state. A
latent state earns its cost when the biology genuinely has memory the current
measurements don't capture: induction that takes hours to take effect, a metabolic shift
that depends on how long a culture has been glucose-limited, product formation that lags
behind the growth that caused it. If a memoryless module systematically mis-fits in a way
that correlates with elapsed time rather than with the current state, that is the signal
to reach for this.

## Gotchas

- **`allow_stateful_models: true` is required**, or training raises before it starts.
- **Read out from the current latent, not the target** — using `h_new` instead of `h` in
  the rate head quietly changes the model's causal structure (the rate would depend on
  information from the *next* step).
- **`SCL_latent_derivative` has a default of empty.** Forgetting to set it on a stateful
  module is a shape mismatch, not a silent zero.
- **This adds real parameters and real integration cost.** Confirm a memoryless module
  actually underfits before reaching for this — see [Tutorial 3](../tutorials/03_train.md)
  and [4](../tutorials/04_your_first_custom_py.md) for the memoryless baseline.

## See also

- [The reaction module](../train/reaction_module.md) — the general contract this module
  follows.
- [Structured rate laws](structured_rates.md) — the opposite direction: less capacity,
  more structure.
- `DefaultStatefulReactionModule` in `bp_train/defaults.py` — the built-in GRU version of
  this same pattern.
