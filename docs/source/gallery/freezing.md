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

# Freezing parameters

> **Demonstrates.** `trainable_field()` / `frozen_field()`: splitting a reaction module
> into a part the optimizer must not touch and a part it should, checked before training
> with `print_trainable_structure`, and what freezing actually costs.

Every module so far has been entirely trainable. Sometimes you want the opposite: a
piece you trust and do not want the optimizer to move, alongside a small piece you
actually want to fit. hybrax.train's answer is two field tags, not a separate optimizer
mechanism: `trainable_field()` and `frozen_field()` on the module's own attributes.
`partition_trainable` (what the optimizer actually sees) reads these tags straight off
the module.

The walkthrough below shows the file in pieces, next to the reasoning for each one. For
the whole thing at once: to copy, diff against your own, or just read top to bottom:

:::{dropdown} Full `custom.py`
```{literalinclude} _files/freezing_custom.py
:language: python
:linenos:
```
:::

```{code-cell} ipython3
:tags: [remove-cell]

import os, shutil, subprocess, sys, textwrap
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_freezing").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
shutil.copy(Path("../_data/out/demo_batch/data.json").resolve(), WORK / "data.json")
shutil.copy(Path("_files/freezing_custom.py").resolve(), WORK / "custom.py")

ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "HYBRAX_TRAIN_DEVICES": "1",
       "MPLBACKEND": "Agg"}

def hxt_cli(*args):
    proc = subprocess.run([sys.executable, "-m", "hybrax.train.cli", *args],
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
(WORK / "forward-config.json").write_text(
    '{ "models": ["run"], "output": { "predictions": "parents", "plots": true } }\n')

hxt_cli("prepare", "--config", "prepare-config.json",
         "--output-dir", "prepared", "--overwrite")
```

The real files this page produces (data, config, `custom.py`, the trained run) are
printed at the end, under "Everything this produced": inspect, copy or modify them.

## The split

```{literalinclude} _files/freezing_custom.py
:language: python
:linenos:
:lines: 26-40
```

`encoder` is a 2-layer MLP tagged `frozen_field()`. `head` is a single trainable
`Linear` reading the encoder's hidden features. Nothing else about writing a reaction
module changes: this is still an ordinary `__call__` over `ReactionInputs`.

## Training

```{code-cell} ipython3
:tags: [remove-input]

out = hxt_cli("train", "--config", "train-config.json", "--overwrite")
lines = [l for l in out.splitlines() if "training complete" in l]
print(lines[0] if lines else "training complete")
```

```{code-cell} ipython3
:tags: [remove-input]

hxt_cli("forward", "--config", "forward-config.json",
         "--output-dir", "run/forward", "--overwrite")
from IPython.display import Image
Image(filename=str(WORK / "run/forward/forward-results/plots/run_1.png"))
```

## Checking the split actually took

Field tags are easy to get backwards silently, so check the trained model, not just the
source:

```{code-cell} ipython3
:tags: [remove-input]

import hybrax.train as hxt

wrapper, _cfg = hxt.model_load(str(WORK / "run"))
hxt.print_trainable_structure(wrapper)
```

Every `reaction_module.encoder.*` leaf reads `frozen`; every `reaction_module.head.*`
leaf reads `trainable`. This is the same printer used in
[Stateful reaction modules](stateful.md), and worth running on any custom split before
trusting a long run to it.

## What freezing costs

The encoder above was never trained: it is a fixed random projection. For comparison,
here is the identical architecture with `encoder` also tagged `trainable_field()`,
otherwise byte-for-byte the same file, same seed, same epochs:

```{code-cell} ipython3
:tags: [remove-cell]

unfrozen_src = (WORK / "custom.py").read_text().replace(
    "encoder: eqx.nn.MLP = frozen_field()",
    "encoder: eqx.nn.MLP = trainable_field()",
)
(WORK / "custom_unfrozen.py").write_text(unfrozen_src)
(WORK / "train-config-unfrozen.json").write_text(textwrap.dedent("""\
    {
      "data": { "prepared": "prepared" },
      "custom_py": "custom_unfrozen.py",
      "train": { "epochs": 600, "seed": 0, "learning_rate": 0.02 },
      "output": { "dir": "run_unfrozen" }
    }
    """))
```

```{code-cell} ipython3
:tags: [remove-input]

out = hxt_cli("train", "--config", "train-config-unfrozen.json", "--overwrite")
lines = [l for l in out.splitlines() if "training complete" in l]
print(lines[0] if lines else "training complete")
```

```{code-cell} ipython3
:tags: [remove-input]

import pandas as pd

def final_loss(run_dir):
    return float(pd.read_csv(WORK / run_dir / "metrics.csv")["mean_loss"].iloc[-1])

print(f"frozen encoder    final mean_loss = {final_loss('run'):.5f}")
print(f"trainable encoder final mean_loss = {final_loss('run_unfrozen'):.5f}")
```

A ~100x gap, from the same architecture and the same budget. A frozen, untrained random
projection is a real constraint on capacity, not a free simplification. Freezing pays
off when what you freeze is already informative: parameters carried over from a
previous run, a sub-model you have independently validated, constants a domain expert
already trusts (as in [Mechanistic models](mechanistic_rates.md), which trains what this
page freezes). Freezing arbitrary, never-trained weights just removes capacity.

## Everything this produced

```{code-cell} ipython3
:tags: [remove-input]

root = WORK.parents[4]
print(f"run directory: ./{(WORK / 'run').relative_to(root)}")
print(f"comparison run: ./{(WORK / 'run_unfrozen').relative_to(root)}")
```

## Gotchas

- **An explicit tag on a parent field wins over every descendant's own tag**, and
  applies to the whole subtree. Tagging the wrapper itself would override both
  `encoder` and `head`.
- **Untagged array leaves default to frozen.** Forgetting `trainable_field()` on a new
  attribute does not raise; it silently stops training that piece.
- **Check the split before training**, not after: `print_trainable_structure` is cheap,
  a wasted training run is not.
- **Freezing costs capacity.** Only freeze parts you have reason to trust; an untrained
  frozen sub-module is not a shortcut.

## See also

Run the example yourself at `./source/_data/out/runs/gallery_freezing/`.

- [The Reaction Module](../train/reaction_module.md): the general contract every
  reaction module follows, including the field-tag rules in full.
- [Mechanistic models](mechanistic_rates.md): trainable, physically meaningful
  constants instead of a frozen random projection.
- [Stateful reaction modules](stateful.md): another use of
  `print_trainable_structure` to check a module before training it.
