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

# Continuous Culture with Controlled Overflow

> This example follows one process through batch, a pause, fed-batch filling, and
> continuous culture, with matched continuous inflow and outflow. The same growth law
> is then learned two ways: as two Monod parameters or as a small neural network.

A continuous process is an ordinary physical-space process in Hybrax whose volume
description contains continuous inflows and outflows. This example makes that concrete,
then asks how much kinetics can be learned from just 21 noiseless biomass measurements
and 21 noiseless glucose measurements.

```{code-cell} ipython3
:tags: [remove-cell]

import json, os, shutil, subprocess, sys
from pathlib import Path
%matplotlib inline

WORK = Path("../_data/out/runs/gallery_continuous_overflow").resolve()
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
EXAMPLE = Path("../../../examples/gallery_continuous_overflow").resolve()
for name in (
    "custom.py", "data.json", "ground_truth.json", "prepare-config.json", "run.py",
    "train-monod-early.json", "train-monod.json", "train-ann-early.json",
    "train-ann.json",
):
    shutil.copy(EXAMPLE / name, WORK / name)

ENV = {**os.environ, "JAX_PLATFORMS": "cpu", "HYBRAX_TRAIN_DEVICES": "1",
       "MPLBACKEND": "Agg"}
```

## From Batch to Continuous Operation

The run starts with 0.5 L containing 0.5 g/L biomass and 5 g/L glucose. Its four phases
are:

1. **Batch:** no liquid crosses the vessel boundary.
2. **Feed delay:** after a simple exponential-growth estimate of the batch end, the
   feed remains off for another hour. Bioreaction continues during this interval.
3. **Fed-batch:** a 0.1 L/h feed containing 50 g/L glucose fills the vessel to 1 L.
4. **Continuous culture:** an outflow starts at 0.1 L/h, matching the feed and holding
   the vessel at 1 L.

```{code-cell} ipython3
import hybrax.format as hxf

collection = hxf.serialization.load_process_collection(WORK / "data.json")
process = collection.processes["continuous_1"]
for name, change in process.volume.volume_changes.items():
    print(f"{name:10s} {type(change).__name__:8s} continuous={change.is_continuous}")
```

Both volume changes are **controlled**: their cumulative-volume traces are known inputs.
The outflow is the realized trace of a passive overflow. Hybrax does not infer when a
dip tube begins spilling from the simulated volume. Here the start time is known, and
equal prescribed rates make the level balance exact after the vessel reaches 1 L.

`process_type="continuous"` is useful metadata, while the `Inflow`, `Outflow`, and their
time series are what actually create this behavior.

The biological ODE declares one specific rate, growth (`mu`). Transport terms for feed,
outflow, and changing volume are assembled separately from the process description:

```{code-cell} ipython3
hxf.print_rhs_ode(process)
```

## Synthetic Measurements

The ground truth is a Monod growth law,

$$
\mu(S) = \mu_{\max}\frac{S}{K_S + S},
\qquad
\mu_{\max}=0.5\;\mathrm{h}^{-1},
\qquad
K_S=0.5\;\mathrm{g\,L}^{-1}.
$$

with a fixed biomass yield of 0.5 g/g. The dataset stores biomass and glucose every 2 h
from 0 to 40 h: 21 points per trace, 42 scalar observations in total. The observations
are noiseless so this example isolates model structure and optimization from measurement
noise.

Run the complete example. It prepares the data, trains both candidate reaction modules,
retains epochs 1, 50, and 200, reconstructs dense physical trajectories with
`solve_physical_states`, and writes both figures used below.

```{code-cell} ipython3
:tags: [remove-input]

proc = subprocess.run(
    [sys.executable, "run.py"], cwd=WORK, env=ENV, capture_output=True, text=True
)
if proc.returncode != 0:
    raise RuntimeError(proc.stdout + proc.stderr)
print(proc.stdout)
```

```{code-cell} ipython3
:tags: [remove-input]

from IPython.display import Image
Image(filename=str(WORK / "process.png"))
```

The gray band is the one-hour feed delay. The green vertical line starts the feed; the
red line starts the overflow. Volume rises linearly during fed-batch and remains exactly 1 L
once the two flow rates match. The dashed horizontal biomass and glucose lines are the
analytic chemostat steady state. At 40 h the process is close, but not yet exactly at
that asymptote.

## Two Representations of the Same Rate

The mechanistic candidate has only two trainable values, `mu_max` and `Ks`. Their logs
are trained so both physical parameters remain positive. They start at 1.0 h⁻¹ and
1.0 g/L, deliberately away from the ground truth.

```{literalinclude} ../../../examples/gallery_continuous_overflow/custom.py
:language: python
:linenos:
:pyobject: FittedMonodModule
```

The neural candidate sees only scaled glucose and emits the same declared growth rate.
Its `1 → 4 → 4 → 1` MLP has 33 trainable parameters. It does not receive time, biomass,
feed rate, outflow rate, or volume.

```{literalinclude} ../../../examples/gallery_continuous_overflow/custom.py
:language: python
:linenos:
:pyobject: AnnGrowthModule
```

`build_reaction_module` selects between these candidates from the config. The process,
training targets, ODE wrapper, and physical transport terms remain unchanged.

```{literalinclude} ../../../examples/gallery_continuous_overflow/custom.py
:language: python
:linenos:
:pyobject: build_reaction_module
```

## What Training Learns

```{code-cell} ipython3
:tags: [remove-input]

Image(filename=str(WORK / "training.png"))
```

The upper panels inspect the learned rate itself, separately from its integrated
trajectory. That distinction matters: a trajectory can look plausible while its local
rate law is wrong. The lower panels show how those rates accumulate into biomass over all four
operating phases. Black lines are dense simulator references; training still sees only
the 21 timestamps per measured trace.

The two-parameter Monod model moves toward the correct curve in an interpretable way.
The ANN begins with rates that can be negative or much too large because no mechanistic
shape constraint was imposed. By epoch 200, both reproduce the observed trajectory and
the ANN closely approximates the true rate over the glucose range visited by this run.
Unlike the Monod law, the ANN does not force exactly zero growth at zero glucose; the
small endpoint offset remains visible.

```{code-cell} ipython3
results = json.loads((WORK / "results.json").read_text())
print(f"fitted mu_max: {results['fitted_mu_max']:.4f} 1/h")
print(f"fitted Ks:     {results['fitted_Ks']:.4f} g/L")
print(f"ANN rate RMSE: {results['ann_rate_rmse']:.4f} 1/h")
```

Do not read the ANN curve beyond the observed glucose range as validated extrapolation.
The mechanistic law carries a saturation assumption outside the data; the ANN does not.
That is the central tradeoff here: more structural prior and directly interpretable
parameters versus a more flexible learned rate.

## Practical Takeaways

- A controlled continuous `Outflow` is supported in physical-space training and
  prediction just like a controlled `Inflow`.
- Equal prescribed inflow and outflow rates are the simplest representation when the
  realized overflow trace is already known.
- `process_type` records intent; the volume changes define dynamics.
- Inspect learned rates as well as trajectories. Integration can conceal a poor local
  law.
- Sparse, noiseless data are enough for this small demonstration. Real experiments need
  noise-aware validation across independent processes.

## See Also

The full, runnable example (`custom.py`, configs, data) lives in
`examples/gallery_continuous_overflow/` at the repo root, no docs build required. This
page's own executed run is at `./source/_data/out/runs/gallery_continuous_overflow/`.

- [Volume, feeds and events](../format/volume_feeds_events.md): how inflows, outflows,
  feed composition, and cumulative-volume traces are represented.
- [The Reaction Module](../train/reaction_module.md): the scaled input/output contract
  used by both candidates.
- [Mechanistic Models](mechanistic_rates.md): more structured rate laws and parameter
  identifiability.
- [Feeds, boluses and samples](fed_batch.md): controlled inputs in a fed-batch process.
