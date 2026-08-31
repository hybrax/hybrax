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

ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "HYBRAX_TRAIN_DEVICES": "1",
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
resampled timepoints each has, and a relative noise level per state.

```json
{
  "n_children_per_process": 5,
  "n_time_points": 11,
  "noise_std": {"biomass": 0.05, "glucose": 0.05, "lactate": 0.05, "product": 0.05}
}
```

Only states named in `noise_std` are touched. `n_time_points` need not match the
parent's own sampling: children are resampled onto new timepoints.

## Splines First

```{literalinclude} ../../../examples/gallery_augmentation/custom.py
:language: python
:linenos:
:lines: 21-27
```

Augmentation resamples each modeled state onto new timepoints, and that needs a fitted
spline. A freshly loaded hybrax.format file has none:
without this hook, `prepare` fails fast with `"modeled state 'biomass' requires a
spline"` rather than guessing. `custom_py` must be set at the top level of the config
(not inside `prepare`) for `prepare` to pick this hook up at all.

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
clips every reactor-medium child value to `≥ 0` afterward, so this does not produce a
physically invalid concentration, but it is a real sign to look at the plot rather than
trust the config blindly.

## Fixing What Default Noise Gets Wrong

```{literalinclude} ../../../examples/gallery_augmentation/custom.py
:language: python
:linenos:
:lines: 30-36
```

`product` accumulates and should never decrease, but independent Gaussian noise per
timepoint does not know that. The hook receives the noised values *after* the default
transform and can repair them: here, a running maximum.

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

root = WORK.parents[4]
print(f"run directory: ./{(WORK / 'run').relative_to(root)}")
print(f"prepared augmentation diagnostic: ./{(WORK / 'prepared/augmented-data.png').relative_to(root)}")
```

## Gotchas

- **Augmentation needs a fitted spline on every state it touches.** Fit one in
  `transform_process_collection` before augmenting; a freshly loaded file has none.
- **`custom_py` is a top-level config key.** Nesting it inside `prepare` silently means
  no hooks are found (`prepare hooks default: transform_process_collection,
  augment_state_values` in the log is the tell).
- **Only states named in `noise_std` are generated with noise**; everything else is
  copied from the parent's resampled trajectory unchanged.
- **`augment_state_values` runs after the default noise**: it receives `augmented_values`
  already populated, so it adjusts rather than replaces them outright.
- **A spline can dip below zero between measurements**, especially near sharp features
  like a near-zero glucose plateau. Reactor-medium children are clipped to `≥ 0`
  afterward, but check the diagnostic plot for any state where this matters.
- **Cross-validation must stay group-aware.** A parent and its synthetic children carry
  the same information; splitting them across train and holdout leaks the answer. Use
  `hybrax.train`'s LOO ([worked example](loo.md)), which handles this for you.
- **This resamples and perturbs what one run already told you.** It cannot substitute
  for an experiment you have not run.

## See Also

The full, runnable example (`custom.py`, configs, data) lives in
`examples/gallery_augmentation/` at the repo root, no docs build required. This page's
own executed run is at `./source/_data/out/runs/gallery_augmentation/`.

- [Prepare](../train/prepare.md#augmentation): the config reference.
- [Cross-Validation, Worked](loo.md): why augmented data needs group-aware folds.
- [Fed-batch](fed_batch.md): the un-augmented version of this same dataset.
