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

# Freezing Parameters

> This page splits a reaction module into a frozen part the optimizer must not touch and
> a trainable part it should, using `trainable_field()` and `frozen_field()`. It checks
> the split before training with `print_trainable_structure` and shows what freezing
> actually costs.

Every module so far has been entirely trainable. Sometimes you want the opposite: a
piece you trust and do not want the optimizer to move, alongside a small piece you
actually want to fit. `hybrax.train`'s answer is two field tags on the module's own
attributes: `trainable_field()` and `frozen_field()`.
`partition_trainable` (what the optimizer actually sees) reads these tags straight off
the module.

The walkthrough below shows the file in pieces, next to the reasoning for each one.

```{code-cell} ipython3
:tags: [remove-cell]

import os, shutil, subprocess, sys
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_freezing").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
EXAMPLE = Path("../../../examples/gallery_freezing").resolve()
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
shutil.copy(EXAMPLE / "train-config.json", WORK / "train-config.json")
shutil.copy(EXAMPLE / "forward-config.json", WORK / "forward-config.json")

hxt_cli("prepare", "--config", "prepare-config.json",
         "--output-dir", "prepared", "--overwrite")
```

The real files this page produces (data, config, `custom.py`, the trained run) are
printed at the end, under "Everything This Produced": inspect, copy or modify them.

## The Split

```{literalinclude} ../../../examples/gallery_freezing/custom.py
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
Image(filename=str(WORK / "run/forward/plots/run_1.png"))
```

## Checking the Split Actually Took

Field tags are easy to get backwards silently, so check the trained model as well as the
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

## What Freezing Costs

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
shutil.copy(EXAMPLE / "train-config-unfrozen.json", WORK / "train-config-unfrozen.json")
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
projection is a real constraint on capacity. Freezing pays
off when what you freeze is already informative: parameters carried over from a
previous run, a sub-model you have independently validated, constants a domain expert
already trusts (as in [Mechanistic Models](mechanistic_rates.md), which trains what this
page freezes). Freezing arbitrary, never-trained weights just removes capacity.

## Everything This Produced

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
- **Check the split before training.** `print_trainable_structure` is cheap; a wasted
  training run is expensive.
- **Freezing costs capacity.** Only freeze parts you have reason to trust: an untrained
  frozen sub-module still costs capacity.

## See Also

The full, runnable example (`custom.py`, configs, data) lives in
`examples/gallery_freezing/` at the repo root, no docs build required. This page's own
executed run is at `./source/_data/out/runs/gallery_freezing/`.

- [The Reaction Module](../train/reaction_module.md): the general contract every
  reaction module follows, including the field-tag rules in full.
- [Mechanistic Models](mechanistic_rates.md): trainable, physically meaningful
  constants instead of a frozen random projection.
- [Stateful reaction modules](stateful.md): another use of
  `print_trainable_structure` to check a module before training it.
