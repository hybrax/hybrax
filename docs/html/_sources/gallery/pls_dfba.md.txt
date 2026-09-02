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

# PLS-dFBA

> [FBA-Hyb](fba_hyb.md)'s surrogate, extended with a real PLS-shaped component: a
> linear, low-rank latent-variable regression that reads media composition alongside
> state. A controllable recipe choice measurably shifts the predicted metabolic
> corridor.

Inspired by Negahban et al. 2026 <a href="#ref-negahban">[1]</a>, *"Run-to-run
optimization of CHO cell culture media using high-throughput microscale bioreactor
system and a hybrid modeling approach,"* whose hybrid PLS-dFBA model predicts kinetic
rate constraints from a Partial Least Squares regression fed by both state and media
blend composition, then uses those constraints inside a dynamic FBA. This page
reproduces that structural idea, a data-driven regression aware of recipe as well as
physiology shaping the FBA solution, built on the same frozen surrogate as
[FBA-Hyb](fba_hyb.md) and fit by ordinary gradient descent through the whole ODE
trajectory against synthetic data, rather than the paper's own fitting procedure
against real bioreactor data. The PLS component below reproduces PLS's actual
structural form: predictors compressed into a handful of latent components, then
linearly regressed onto outputs, with no nonlinearity anywhere, trained here by
gradient descent rather than the classical NIPALS algorithm, the one disclosed
algorithmic difference from a textbook PLS fit.

The walkthrough below shows the file in pieces, next to the reasoning for each one.

```{code-cell} ipython3
:tags: [remove-cell]

import os, shutil, subprocess, sys
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_pls_dfba").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
EXAMPLE = Path("../../../examples/gallery_pls_dfba").resolve()
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hybrax.format as hxf
import hybrax.train as hxt

_collection = hxf.serialization.load_process_collection(WORK / "data.json")

def r2_by_target(run_dir):
    """Pooled R2: concatenate every process's residuals/variance before
    dividing, so one narrow-range process can't make an otherwise-good fit
    look catastrophic."""
    df = pd.read_csv(WORK / run_dir / "predictions.csv")
    per_target = {s: ([], []) for s in ("biomass", "glucose", "acetate", "succinate")}
    for name, process in _collection.processes.items():
        proc_df = df[df["process"] == name]
        t_pred = proc_df["t"].to_numpy()
        for species in per_target:
            comp = process.reactor_medium.components[species].concentration
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

## A Controlled Process Variable for Recipe

`demo_ecoli_blend` has four runs, each forward-simulated from the same surrogate
under a different `media_blend_fraction` in `[0, 1]`: higher fractions push more
carbon toward succinate (the deliberate product here) at a modest cost to growth,
the same real trade-off Negahban et al. 2026 optimize for with three real media.
`ReactionInputs` has no "which recipe produced this run" field, by design; a
constant-valued controlled process variable does the job instead, the same mechanism
[Knowledge Transfer](knowledge_transfer.md) uses for product identity.
[The data generator](../_data/generate.py)'s `build_demo_ecoli_blend()` attaches it
directly, once, when it builds each process:

```{literalinclude} ../_data/generate.py
:language: python
:linenos:
:start-at: processes[name].process_variables["media_blend_fraction"]
:end-before: collection = hxf.BioProcessCollection
:dedent: 8
```

`media_blend_fraction` ships as a `StaticVariable`-valued controlled process variable
in `data.json` itself, the same pattern [Knowledge Transfer](knowledge_transfer.md)
uses for `is_new_product`.

## The PLS Component

```{literalinclude} ../../../examples/gallery_pls_dfba/custom.py
:language: python
:linenos:
:lines: 214-239
```

`W` compresses `n_in` predictors (state + `media_blend_fraction`) down to
`n_components = 3` latent scores; `Q` regresses those scores back out to the 5
surrogate inputs. No activation function anywhere in `__call__`: real PLS is linear
end to end, which is the whole reason it needs a low-rank bottleneck to handle
collinear predictors in the first place, rather than just more capacity.

```{literalinclude} ../../../examples/gallery_pls_dfba/custom.py
:language: python
:linenos:
:lines: 242-287
```

The rest of the reaction module (surrogate call, unit conversion, scaling) is
unchanged from [FBA-Hyb](fba_hyb.md): only what predicts the surrogate's inputs
changed, from an MLP to this PLS component, and the component now reads
`SCL_controlled_PVs` (the blend fraction) alongside state.

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
    print(f"{name:10s} R2 = {value:.4f}")
```

```{code-cell} ipython3
:tags: [remove-input]

hxt_cli("forward", "--config", "forward-config.json",
         "--output-dir", "run/forward", "--overwrite")
from IPython.display import Image
Image(filename=str(WORK / "run/forward/plots/blend_67.png"))
```

Every predictor passes through only three latent scores before the final linear
regression: that low-rank bottleneck is PLS's actual structural signature. The R²
values above are what that bottleneck achieves against this page's dummy data, which
is the question this page sets out to answer.

## Gotchas

- **`n_components` is the whole tuning knob for PLS's linear bottleneck.** Too few
  and the bottleneck cannot represent the real relationship between recipe and
  metabolism; too many and the "compress collinear predictors" idea PLS exists for
  stops doing anything.
- **The surrogate itself is unchanged from [FBA-Hyb](fba_hyb.md)**: only what
  predicts its inputs differs. This page's predictions are only ever as good as
  that shared, frozen fit.

## See Also

The full, runnable example (`custom.py`, configs, data) lives in
`examples/gallery_pls_dfba/` at the repo root, no docs build required. This page's own
executed run is at `./source/_data/out/runs/gallery_pls_dfba/`.

- [FBA-Hyb](fba_hyb.md): the surrogate and the base reaction-module shape this
  builds on.
- [Knowledge Transfer](knowledge_transfer.md): the same controlled-PV-as-recipe
  trick, for product identity instead of a continuous blend fraction.
- [The Reaction Module](../train/reaction_module.md): `auxiliary`, and everything
  else a `RateModule` can return.

## References

1. <a id="ref-negahban"></a>Negahban, Z., Ghodba, A., Richelle, A., McCready, C.,
   Ward, V., & Budman, H. (2026). Run-to-run optimization of CHO cell culture media
   using high-throughput microscale bioreactor system and a hybrid modeling
   approach. *Journal of Biotechnology*, 417, 274-286.
   [https://doi.org/10.1016/j.jbiotec.2026.06.011](https://doi.org/10.1016/j.jbiotec.2026.06.011)
