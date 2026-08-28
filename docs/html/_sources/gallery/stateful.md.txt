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

# Stateful Models

> Hybrax's built-in continuous-time LSTM integrates its hidden and cell states
> as extra ODE dimensions, guarded by an explicit stateful-model opt-in.

Every module so far predicts rates from the *current* state alone. That is a real
modeling assumption: it says the biology has no memory beyond what is currently
measured. A **stateful** module relaxes that: it carries its own latent state through
the solve, so the rates can depend on the whole history of where the process has been.

## Why This Is a Bigger Change Than It Looks

`hybrax.train` does not run a discrete recurrent network beside the ODE solver. It **turns the
recurrent cell into a continuous-time ODE**: the latent state `h` is an extra integrated
dimension, and its derivative is the discrepancy between `h` and whatever the cell would
have jumped to next: `d(h)/dt = cell(input, h) - h`. As training pulls the residual to
zero, `h` tracks the same trajectory the discrete cell would have taken, but now it is a
proper flow that Diffrax can integrate and differentiate through like any other state.

`hybrax.train` provides this pattern directly: `DefaultGruReactionModule` and
`DefaultLstmReactionModule` are built-in stateful defaults. The latter stores
`[hidden | cell]` in `SCL_latent`; its `hidden_width` constructor argument sets
the LSTM width, so the integrated latent width is `2 * hidden_width`. The legacy
`DefaultStatefulReactionModule` name remains an alias for the GRU default.

The walkthrough below shows the example's configuration hook and explains the
stateful behavior.

```{code-cell} ipython3
:tags: [remove-cell]

import os, re, shutil, subprocess, sys
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_stateful").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
EXAMPLE = Path("../../../examples/gallery_stateful").resolve()
shutil.copy(EXAMPLE / "data.json", WORK / "data.json")
shutil.copy(EXAMPLE / "custom.py", WORK / "custom.py")

ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "HYBRAX_TRAIN_DEVICES": "1",
       "MPLBACKEND": "Agg"}

def hxt_cli(*args, check=True):
    proc = subprocess.run([sys.executable, "-m", "hybrax.train.cli", *args],
                          cwd=WORK, env=ENV, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout + proc.stderr

shutil.copy(EXAMPLE / "prepare-config.json", WORK / "prepare-config.json")
hxt_cli("prepare", "--config", "prepare-config.json",
         "--output-dir", "prepared", "--overwrite")
```

## The Module

```{literalinclude} ../../../examples/gallery_stateful/custom.py
:language: python
:pyobject: build_reaction_module
```

An LSTM cell has *two* pieces of memory, hidden and cell state, so `SCL_latent`
holds both concatenated. The built-in computes its discrete target, emits the
difference as the derivative, and reads rates from the *current* hidden state
(not the target).

`ReactionOutputs` gains a field we have not used before, `SCL_latent_derivative`, aligned
with `SCL_latent`. It defaults to an empty array: every non-stateful module in these docs
has been quietly relying on that default.

## The Opt-in

Declaring a nonzero `SCALE_latent` is enough to make `n_latent > 0`, and that alone is not
allowed to train silently:

```{code-cell} ipython3
:tags: [remove-cell]

shutil.copy(EXAMPLE / "train-no-optin.json", WORK / "train-no-optin.json")
```

```{code-cell} ipython3
:tags: [remove-input]

out = hxt_cli("train", "--config", "train-no-optin.json", "--overwrite",
               check=False)
print([l for l in out.splitlines() if "ValueError" in l][-1])
```

That is deliberate: a latent state changes what the model *is* (it is no longer a pure
function of the physical state) and that is a large enough change in what "the model"
means that `hybrax.train` wants it to be a deliberate decision on your part.

```json
{ "train": { "allow_stateful_models": true } }
```

## Training It

```{code-cell} ipython3
:tags: [remove-cell]

shutil.copy(EXAMPLE / "train.json", WORK / "train.json")
shutil.copy(EXAMPLE / "forward-config.json", WORK / "forward-config.json")
```

```{code-cell} ipython3
:tags: [remove-input]

out = hxt_cli("train", "--config", "train.json", "--overwrite")
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

## Checking the Latent Dimension Actually Registered

```{code-cell} ipython3
:tags: [remove-input]

import hybrax.train as hxt

wrapper, cfg = hxt.model_load(str(WORK / "run"))
print("n_latent =", wrapper.reaction_module.n_latent)
hxt.print_trainable_structure(wrapper)
```

`n_latent` is `2 * hidden_width` (hidden and cell state together); both the LSTM gates
and the rate head show up as trainable, exactly like any other reaction module.

## When This Is Worth the Extra Machinery

Not on `demo_batch`: a memoryless module already fits this data well, because nothing
about a simple batch culture actually depends on history beyond the current state. A
latent state earns its cost when the biology genuinely has memory the current
measurements don't capture: induction that takes hours to take effect, a metabolic shift
that depends on how long a culture has been glucose-limited, product formation that lags
behind the growth that caused it. If a memoryless module systematically mis-fits in a way
that correlates with elapsed time rather than with the current state, that is the signal
to reach for this.

## Gotchas

- **`allow_stateful_models: true` is required**, or training raises before it starts.
- **The built-in reads from the current latent, not the target.** Custom recurrent
  modules should do the same: using `h_new` instead of `h` in the rate head quietly
  changes the model's causal structure.
- **`SCL_latent_derivative` has a default of empty.** Forgetting to set it on a stateful
  module raises a shape mismatch.
- **This adds real parameters and real integration cost.** Confirm a memoryless module
  actually underfits before reaching for this: see [Tutorial 3](../tutorials/03_train.md)
  and [4](../tutorials/04_your_first_custom_py.md) for the memoryless baseline.

## See Also

The full, runnable example (`custom.py`, configs, data) lives in
`examples/gallery_stateful/` at the repo root, no docs build required. This page's own
executed run is at `./source/_data/out/runs/gallery_stateful/`.

- [The Reaction Module](../train/reaction_module.md): the general contract this module
  follows.
- [Mechanistic Models](mechanistic_rates.md): a reaction module built from explicit
  kinetics, no latent state at all.
- `DefaultGruReactionModule` and `DefaultLstmReactionModule` in
  `hybrax/train/defaults.py`: built-in recurrent defaults. The former name
  `DefaultStatefulReactionModule` remains a GRU alias.
