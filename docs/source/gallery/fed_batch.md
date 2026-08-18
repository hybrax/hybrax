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
# Feeds, boluses and samples
<!-- UNLOCK -->

> **Demonstrates.** A continuous feed, two boluses and sampling events in one run, and a
> reaction module that reads the feed rate and a controlled process variable as real
> biological inputs: not just transport bookkeeping.

The tutorials used a pure batch: no feeds, no boluses, no sampling volume. This dataset
has all three at once, which is the normal case for real fermentation data.

The walkthrough below shows the file in pieces, next to the reasoning for each one. For
the whole thing at once: to copy, diff against your own, or just read top to bottom:

:::{dropdown} Full `custom.py`
```{literalinclude} _files/fed_batch_custom.py
:language: python
:linenos:
```
:::

```{code-cell} ipython3
:tags: [remove-cell]

import os, shutil, subprocess, sys, textwrap
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_fed_batch").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
shutil.copy(Path("../_data/out/demo_fedbatch/data.json").resolve(), WORK / "data.json")
shutil.copy(Path("_files/fed_batch_custom.py").resolve(), WORK / "custom.py")

ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "BP_TRAIN_DEVICES": "1",
       "MPLBACKEND": "Agg"}

def bp_train_cli(*args):
    proc = subprocess.run([sys.executable, "-m", "bp_train.cli", *args],
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
      "train": { "epochs": 2000, "seed": 0, "learning_rate": 0.01 },
      "output": { "dir": "run" }
    }
    """))
(WORK / "forward-config.json").write_text(
    '{ "models": ["run"], "output": { "predictions": "parents", "plots": true } }\n')
```

## The dataset

```{code-cell} ipython3
import bp_format as bp

collection = bp.serialization.load_process_collection(WORK / "data.json")
process = collection.processes["fedbatch_1"]

for name, vc in process.volume.volume_changes.items():
    print(f"{name:15s} {type(vc).__name__:20s} continuous={vc.is_continuous}")
```

One continuous feed starting at t = 6 h, two boluses at t = 10 h and t = 16 h, and
sampling at every offline measurement. See the assembled ODE:

```{code-cell} ipython3
bp.print_rhs_ode(process)
```

Compare this to a batch process's `print_rhs_ode` output: every feed and sample now
contributes a real transport term, generated for you.

## The reaction module

The interesting change from the tutorials is not really about feeds: it is that the
model now has real controlled inputs beyond the state:

```{literalinclude} _files/fed_batch_custom.py
:language: python
:linenos:
:lines: 25-51
```

`SCL_controlled_FVCs_rates` (the feed) and `SCL_controlled_PVs` (dissolved oxygen) are
concatenated onto the state before the network sees it. Nothing about the *mechanics* of
writing a reaction module changed (you still emit rates in SCL space) but the module
can now respond to what is happening to the process, not just its own current
concentrations.

## Scaling with real controlled axes

```{literalinclude} _files/fed_batch_custom.py
:language: python
:linenos:
:lines: 58-64
```

`runtime_data.controls_store` is always reachable from `estimate_all_scales`, no
special-cased argument needed. The batch tutorials never used it, because a batch
process has no controlled feed or PV axes to estimate. See
[Scaling](../train/scaling.md#writing-one).

## Training

```{code-cell} ipython3
:tags: [remove-input]

bp_train_cli("prepare", "--config", "prepare-config.json",
         "--output-dir", "prepared", "--overwrite")
out = bp_train_cli("train", "--config", "train-config.json", "--overwrite")
print([l for l in out.splitlines() if "training complete" in l][0])
print(f"run directory: ./{(WORK / 'run').relative_to(WORK.parents[4])}")
```

```{code-cell} ipython3
:tags: [remove-input]

bp_train_cli("forward", "--config", "forward-config.json",
         "--output-dir", "run/forward", "--overwrite")
from IPython.display import Image
Image(filename=str(WORK / "run/forward/forward-results/plots/fedbatch_1.png"))
```

Look at the bottom-right panel (`volume_changes`) before anything else. It plots every
feed, bolus and sample bp-format extracted from the description, on one axis. This is
the fastest way to confirm your event bookkeeping is what you think it is, on real data,
before trusting anything about the fit above it.

In the top two rows, the glucose boluses are visible as sharp jumps, both in the
concentration itself and in the inferred `q_glucose`: the model has to represent a
discontinuity twice in one run, which is a meaningfully harder fit than the smooth batch
case. Biomass is the hardest target here (a real, honest R² in the low 0.9s at this
epoch budget); glucose and product are markedly easier.

## What made this example different from the tutorials

- **A feed medium with every species declared**, including the zeros: see
  [Volume, feeds and events](../format/volume_feeds_events.md).
- **Sample-then-bolus ordering** at the coincident timestamps was handled for you by the
  solve; you never wrote it.
- **A reaction module that reads controlled inputs**, not just the state.
- **`estimate_all_scales` reading `runtime_data.controls_store`**, for the controlled axes.

## See also

Run the example yourself at `./source/_data/out/runs/gallery_fed_batch/`.

- [Volume, feeds and events](../format/volume_feeds_events.md): the concepts behind this
  dataset.
- [Time series and splines](../format/time_series_and_splines.md): the pseudobatch
  transform, which this dataset is exactly the motivating case for.
- [The reaction module](../train/reaction_module.md): `ReactionInputs` in full.
