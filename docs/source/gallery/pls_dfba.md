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

> **Demonstrates.** [FBA-Hyb](fba_hyb.md)'s surrogate, extended with a real
> PLS-shaped component (a linear, low-rank latent-variable regression, not a neural
> network) that reads media composition alongside state, so a controllable recipe
> choice measurably shifts the predicted metabolic corridor.

Inspired by Negahban et al. 2026 <a href="#ref-negahban">[3]</a>, *"Run-to-run
optimization of CHO cell culture media using high-throughput microscale bioreactor
system and a hybrid modeling approach,"* whose hybrid PLS-dFBA model predicts kinetic
rate constraints from a Partial Least Squares regression fed by both state and media
blend composition, then uses those constraints inside a dynamic FBA. This page
reproduces that structural idea (a data-driven regression, aware of recipe as well as
physiology, shaping the FBA solution), built on the same real, frozen surrogate as
[FBA-Hyb](fba_hyb.md). It is not a replication of their method: their model is
identified via a bi-level NMSE fit plus a gradient-correction run-to-run optimization
loop against real bioreactor data; this page trains by ordinary gradient descent
through the whole ODE trajectory, once, against synthetic data.

**The one piece worth being explicit about**: PLS itself is a specific algorithm
(linear, latent-variable regression, fit via NIPALS). The component below reproduces
its actual *structural form*: predictors compressed into a handful of latent
components, then linearly regressed onto outputs, no nonlinearity anywhere. It is
trained by gradient descent, not NIPALS, which is the one disclosed algorithmic
substitution; the shape itself is real PLS, not a relabeled neural network.

The walkthrough below shows the file in pieces, next to the reasoning for each one. For
the whole thing at once: to copy, diff against your own, or just read top to bottom:

:::{dropdown} Full `custom.py`
```{literalinclude} _files/pls_dfba_custom.py
:language: python
:linenos:
```
:::

```{code-cell} ipython3
:tags: [remove-cell]

import os, shutil, subprocess, sys, textwrap
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_pls_dfba").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
shutil.copy(Path("../_data/out/demo_ecoli_blend/data.json").resolve(), WORK / "data.json")
shutil.copy(Path("_files/pls_dfba_custom.py").resolve(), WORK / "custom.py")

ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "BP_TRAIN_DEVICES": "1",
       "MPLBACKEND": "Agg"}

def bp_train_cli(*args):
    proc = subprocess.run([sys.executable, "-m", "bp_train.cli", *args],
                          cwd=WORK, env=ENV, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout + proc.stderr

(WORK / "prepare-config.json").write_text(
    '{ "prepare": { "raw_input": "data.json" }, "custom_py": "custom.py" }\n')
(WORK / "train-config.json").write_text(textwrap.dedent("""\
    {
      "data": { "prepared": "prepared" },
      "custom_py": "custom.py",
      "train": { "epochs": 800, "seed": 0, "learning_rate": 0.01 },
      "output": { "dir": "run" }
    }
    """))
(WORK / "forward-config.json").write_text('{ "models": ["run"] }\n')

import csv
import numpy as np
import matplotlib.pyplot as plt
import bp_format as bp
import bp_train

_collection = bp.serialization.load_process_collection(WORK / "data.json")

def r2_by_target(run_dir):
    rows_by_process = {}
    with (WORK / run_dir / "predictions.csv").open() as fh:
        for row in csv.DictReader(fh):
            rows_by_process.setdefault(row["process"], []).append(row)
    per_target = {s: ([], []) for s in ("biomass", "glucose", "acetate", "succinate")}
    for name, process in _collection.processes.items():
        rows = rows_by_process[name]
        t_pred = np.array([float(r["t"]) for r in rows])
        for species in per_target:
            comp = process.reactor_medium.components[species].concentration
            t_meas, y_meas = np.asarray(comp.times), np.asarray(comp.values)
            y_pred = np.interp(t_meas, t_pred,
                               np.array([float(r[f"c_{species}"]) for r in rows]))
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

## A controlled process variable for recipe

`demo_ecoli_blend` has four runs, each forward-simulated from the same surrogate
under a different `media_blend_fraction` in `[0, 1]`: higher fractions push more
carbon toward succinate (the deliberate product here) at a modest cost to growth,
the same real trade-off Negahban et al. 2026 optimize for with three real media.
`ReactionInputs` has no "which recipe produced this run" field, by design; a
constant-valued controlled process variable does the job instead, the same mechanism
[Knowledge transfer](knowledge_transfer.md) uses for product identity:

```{literalinclude} _files/pls_dfba_custom.py
:language: python
:linenos:
:lines: 46-56
```

## The PLS component

```{literalinclude} _files/pls_dfba_custom.py
:language: python
:linenos:
:lines: 83-109
```

`W` compresses `n_in` predictors (state + `media_blend_fraction`) down to
`n_components = 3` latent scores; `Q` regresses those scores back out to the 5
surrogate inputs. No activation function anywhere in `__call__`: real PLS is linear
end to end, which is the whole reason it needs a low-rank bottleneck to handle
collinear predictors in the first place, rather than just more capacity.

```{literalinclude} _files/pls_dfba_custom.py
:language: python
:linenos:
:lines: 112-118
```

The rest of the reaction module (surrogate call, unit conversion, scaling) is
unchanged from [FBA-Hyb](fba_hyb.md): only what predicts the surrogate's inputs
changed, from an MLP to this PLS component, and the component now reads
`SCL_controlled_PVs` (the blend fraction) alongside state.

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

r2 = r2_by_target("run")
for name, value in r2.items():
    print(f"{name:10s} R2 = {value:.4f}")
```

```{code-cell} ipython3
:tags: [remove-input]

bp_train_cli("forward", "--config", "forward-config.json",
         "--output-dir", "run/forward", "--overwrite")
from IPython.display import Image
Image(filename=str(WORK / "run/forward/blend_67.png"))
```

Every predictor passes through only three latent scores before the final linear
regression: that low-rank bottleneck is PLS's actual structural signature, not an
implementation detail. The R² values above are what that bottleneck achieves against
this page's dummy data, which is the question this page sets out to answer.

## Gotchas

- **`prepare-config.json` needs `custom_py` at the top level**, not just the train
  config. Omit it and `transform_process_collection` silently never runs: no error,
  no warning, `media_blend_fraction` just never gets attached. See
  [Prepare](../train/prepare.md#configuration).
- **`n_components` is the whole tuning knob for PLS's linear bottleneck.** Too few
  and the bottleneck cannot represent the real relationship between recipe and
  metabolism; too many and the "compress collinear predictors" idea PLS exists for
  stops doing anything.
- **The surrogate itself is unchanged from [FBA-Hyb](fba_hyb.md)**: only what
  predicts its inputs differs. This page's predictions are only ever as good as
  that shared, frozen fit.

## See also

Run the example yourself at `./source/_data/out/runs/gallery_pls_dfba/`.

- [FBA-Hyb](fba_hyb.md): the surrogate and the base reaction-module shape this
  builds on.
- [Knowledge transfer](knowledge_transfer.md): the same controlled-PV-as-recipe
  trick, for product identity instead of a continuous blend fraction.
- [The reaction module](../train/reaction_module.md): `auxiliary`, and everything
  else a `UserReactionModule` can return.

## References

1. <a id="ref-fbahyb"></a>Gotsmy, M., & Guillén-Gosálbez, G. (2026). Integrating
   metabolic networks into hybrid bioprocess models. *bioRxiv*.
   [https://doi.org/10.64898/2026.04.22.720062v1](https://www.biorxiv.org/content/10.64898/2026.04.22.720062v1)
2. <a id="ref-ecolicore"></a>Orth, J. D., Fleming, R. M. T., & Palsson, B. Ø.
   (2010). Reconstruction and use of microbial metabolic networks: the core
   *Escherichia coli* metabolic model as an educational guide. *EcoSal Plus*, 4(1).
   [https://doi.org/10.1128/ecosalplus.10.2.1](https://doi.org/10.1128/ecosalplus.10.2.1)
3. <a id="ref-negahban"></a>Negahban, Z., Ghodba, A., Richelle, A., McCready, C.,
   Ward, V., & Budman, H. (2026). Run-to-run optimization of CHO cell culture media
   using high-throughput microscale bioreactor system and a hybrid modeling
   approach. *Journal of Biotechnology*, 417, 274-286.
   [https://doi.org/10.1016/j.jbiotec.2026.06.011](https://doi.org/10.1016/j.jbiotec.2026.06.011)
