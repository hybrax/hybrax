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

# Augmentation

> This example generates synthetic sibling processes from a single run with
> `prepare.augmentation`, producing an automatic diagnostic plot. `augment_state_values`
> then gives per-state control over what gets generated.

`demo_fedbatch` is one run. That is a real, common situation: a single mammalian
fed-batch campaign, expensive to repeat, with too little data on its own to train
anything but the simplest model. Augmentation trades that for more (synthetic, noisy)
training signal, without pretending you have more real experiments than you do.

The walkthrough below shows the file in pieces, next to the reasoning for each one.

```{code-cell} ipython3
:tags: [remove-cell]

import json, os, shutil, subprocess, sys, textwrap, warnings
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_augmentation").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
EXAMPLE = Path("../../../examples/gallery_augmentation").resolve()
shutil.copy(EXAMPLE / "data.json", WORK / "data.json")
shutil.copy(EXAMPLE / "custom.py", WORK / "custom.py")

ENV = {**os.environ, "JAX_PLATFORMS": "cpu",
       "MPLBACKEND": "Agg"}

def hxt_cli(*args):
    proc = subprocess.run([sys.executable, "-m", "hybrax.train.cli", *args],
                          cwd=WORK, env=ENV, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout + proc.stderr

shutil.copy(EXAMPLE / "prepare-config.json", WORK / "prepare-config.json")
```

Everything below runs in `WORK`, printed at the end: inspect, copy or modify the real
files it produced.

## Configuration

`prepare.augmentation` is most of it: how many synthetic children per parent, how many
resampled timepoints each has, and an additive Gaussian noise standard deviation
per state, expressed in that state's units. Here, `0.05` means 0.05 g/L.

```json
{
  "n_children_per_process": 5,
  "n_time_points": 11,
  "noise_std": {"biomass": 0.05, "glucose": 0.05, "lactate": 0.05, "product": 0.05}
}
```

Only states named in `noise_std` receive noise. Other modeled states with fitted
splines are resampled without noise; states without splines are copied unchanged.
`n_time_points` need not match the parent's own sampling.

## Splines First

```{literalinclude} ../../../examples/gallery_augmentation/custom.py
:language: python
:linenos:
:pyobject: transform_process_collection
```

Every state named in `noise_std` needs a fitted spline. This example's raw
`data.json` has none, so the hook fits them before augmentation. Files can also
contain splines saved earlier. Without this hook, `prepare` fails fast with
`"modeled state 'biomass' requires a spline"`. Set `custom_py` at the top level
of the config to load the hook.

```{code-cell} ipython3
:tags: [remove-input]

out = hxt_cli("prepare", "--config", "prepare-config.json",
               "--output-dir", "prepared", "--overwrite")
for line in out.splitlines():
    if "UserWarning: " in line:
        print("UserWarning:", line.split("UserWarning: ", 1)[1])
```

## What Got Generated

```{code-cell} ipython3
:tags: [remove-input]

from IPython.display import Image
Image(filename=str(WORK / "prepared/augmented-data.png"))
```

`hybrax prepare` writes this diagnostic automatically whenever `augmentation` is
configured: grey dots are the synthetic children, the blue line is the fitted spline,
red dots the real measurements. Worth checking before training on it: noise that looks
wrong here will look wrong in every downstream fit too.

`glucose` crashes to near zero and stays there for most of the run (the same shape as
[Fed-batch](fed_batch.md)), and the warning above is exactly that: the fitted spline
dips slightly negative between two closely-spaced near-zero measurements. Augmentation
clips resampled reactor-medium values to `≥ 0` before `augment_state_values`.
This example's running maximum preserves nonnegativity; replacement hooks must
preserve it themselves. Check the diagnostic plot before training.

## A Custom Shape Constraint

```{literalinclude} ../../../examples/gallery_augmentation/custom.py
:language: python
:linenos:
:pyobject: augment_state_values
```

For this hook demonstration, we impose a nondecreasing `product` trajectory on the
resampled observations with a running maximum. Product concentration is not
generally monotone in fed-batch: feeding can dilute it. The hook receives the
noised values after clipping and applies this chosen shape constraint.

```{code-cell} ipython3
:tags: [remove-input]

import hybrax.format as hxf
import hybrax.train as hxt
from hybrax.train import augmentation, prepare, run_config

import importlib.util
spec = importlib.util.spec_from_file_location("custom", str(WORK / "custom.py"))
custom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom)

raw = prepare.load_raw_collection(str(WORK / "data.json"))
raw = custom.transform_process_collection(raw, None)
cfg = run_config.RunConfig(prepare=run_config.PrepareConfig(
    raw_input=str(WORK / "data.json"),
    augmentation=run_config.AugmentationConfig(
        n_children_per_process=5, n_time_points=11,
        noise_std={"biomass": 0.05, "glucose": 0.05, "lactate": 0.05, "product": 0.05},
    ),
))
with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    augmented = augmentation.augment_process_collection(
        raw, cfg, custom.augment_state_values
    )
assert caught_warnings
for warning in caught_warnings:
    print(f"{warning.category.__name__}: {warning.message}")

children = [n for n, p in augmented.processes.items() if hasattr(p, "parent_process")]
print(f"{len(children)} synthetic children from 1 parent")
import numpy as np
for name in children[:3]:
    values = np.asarray(
        augmented.processes[name].reactor_medium.components["product"]
        .concentration.values)
    print(f"  {name}: parent={augmented.processes[name].parent_process}"
          f"  product monotone={bool(np.all(np.diff(values) >= 0))}")
```

## Training on the Enlarged Dataset

```{code-cell} ipython3
:tags: [remove-cell]

(WORK / "train-config.json").write_text(textwrap.dedent("""\
    {
      "data": { "prepared": "prepared" },
      "custom_py": "custom.py",
      "train": { "epochs": 300, "seed": 0, "learning_rate": 0.02 },
      "output": { "dir": "run" }
    }
    """))
```

```{code-cell} ipython3
:tags: [remove-input]

out = hxt_cli("train", "--config", "train-config.json", "--overwrite")
lines = [l for l in out.splitlines() if "training complete" in l]
print(lines[0] if lines else "training complete")
```

Nothing about training changes: augmentation happens entirely at `prepare` time, and by
the time `train` runs, the synthetic children are just more processes in the store.

```{code-cell} ipython3
:tags: [remove-input]

root = WORK.parents[5]
print(f"run directory: ./{(WORK / 'run').relative_to(root)}")
print(f"prepared augmentation diagnostic: ./{(WORK / 'prepared/augmented-data.png').relative_to(root)}")
```

## Gotchas

- **Every state named in `noise_std` needs a fitted spline.** This example's raw
  `data.json` has none, so `transform_process_collection` fits them. Loading a file
  with saved splines restores them.
- **`custom_py` is a top-level config key.** Nesting it inside `prepare` raises
  a validation error for an extra field.
- **Only states named in `noise_std` receive noise.** Other modeled states with
  fitted splines are resampled without noise; states without splines are copied
  unchanged. Resampled reactor-medium values are clipped before the hook.
- **`augment_state_values` runs after the default noise**: it receives `augmented_values`
  already populated, so it adjusts rather than replaces them outright.
- **A spline can dip below zero between measurements**, especially near sharp features
  like a near-zero glucose plateau. Reactor-medium values are clipped to `≥ 0`
  before the hook. Replacement hooks must preserve nonnegativity themselves.
- **Cross-validation must stay group-aware.** A parent and its synthetic children carry
  the same information; splitting them across train and holdout leaks the answer. Use
  `hybrax.train`'s LOO ([worked example](loo.md)), which handles this for you.
- **This resamples and perturbs what one run already told you.** It cannot substitute
  for an experiment you have not run.

## See Also

The full, runnable example (`custom.py`, configs, data) lives in
`examples/gallery_augmentation/` at the repo root, no docs build required. This page's
own executed run is at `./docs/source/_data/out/runs/gallery_augmentation/`.

- [Prepare](../train/prepare.md#augmentation): the config reference.
- [Cross-Validation, Worked](loo.md): why augmented data needs group-aware folds.
- [Fed-batch](fed_batch.md): the un-augmented version of this same dataset.
