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
mystnb:
  execution_timeout: 7200
---

# OptFed with neural kinetics

The [OptFed mechanistic model](optfed.md) page fits the reduced OptFed model
(Schlögel et al. \[[1](#ref-optfed-neural)\]) to twelve published fed-batch processes.
This page uses the same data and ODE structure, but it replaces OptFed's mechanistic
equations and fitted kinetic parameters with a small neural network.

The OptFed reduced model defines equations for the specific substrate (glycerol)
uptake rate and for the fraction of substrate used for maintenance and product
formation (the remainder is available for growth). In the example here we follow
the same structure, but we replace the equations with an MLP that predicts the
glycerol uptake capacity and two allocation fractions (fraction of total uptake
used for maintenance and fraction of remainder used for product). Using OptFed's
fixed yields (`0.627` g biomass per g glycerol and `0.652` g product per g
glycerol) these are then converted into specific rates for growth and product
formation.

> Note: As in the original OptFed model we use a fixed Michaelis-Menten
> constant of `K_G = 0.001 g/kg`.

This structure ensures that all rates and concentrations stay non-negative.

The model also tracks the cumulative generation count as a latent state,
using OptFed's equation. This means that the hybrid model tracks an additional
state for internal use that is neither a concentration nor measured. Hybrax
requires an explicit opt-in for such a state, which is why the training config
below sets `allow_stateful_models`. OptFed uses this count to allow the model to
describe production inhibition later in the run and we pass it to the network as
input as well.

## How the pieces fit

The example lives in `examples/gallery_optfed_ffnn/`:

1. **Data** (`../gallery_optfed/data.json`) is the imported OptFed collection
   of the mechanistic example, used unchanged, including its `ReactionOde`. See
   the [OptFed example page](optfed.md#load-the-published-data) for how it was
   built.
2. **Prepare** (`prepare-config.json`) points to that file.
3. **Train** (`train-config.json` + `custom.py`) uses the neural reaction
   module below, returned by `build_reaction_module`. `build_solver` and
   `estimate_all_scales` are the same as in the mechanistic example.
4. **Forward** (`forward-config.json`) runs the model in `run/`. In the
   bundled run, that is the selected checkpoint.

The OptFed repository does not specify a license, so its data and the files on
this page derived from it are not covered by Hybrax's MIT license; all rights
remain with the OptFed authors. `examples/gallery_optfed_ffnn/README.md` lists
which files that covers.

```{code-cell} ipython3
:tags: [remove-cell]

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

from hybrax.format.serialization import load_process_collection
from hybrax.train import model_load

# All reloaded checkpoints belong to the same run.
logging.getLogger("hybrax.train.serialization").setLevel(logging.ERROR)

# Mirror the examples/ layout so the relative data path in prepare-config.json works.
RUNS = Path("../_data/out/runs").resolve()
WORK = RUNS / "gallery_optfed_ffnn"
OPTFED = RUNS / "gallery_optfed"
EXAMPLES = Path("../../../examples").resolve()
for folder, names in (
    (WORK, ("custom.py", "run.py", "prepare-config.json", "train-config.json",
            "forward-config.json")),
    (OPTFED, ("data.json", "run.py", "original_predictions.csv",
              "original_sampling_predictions.csv")),
):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copyfile(EXAMPLES / folder.name / name, folder / name)
ENV = os.environ | {"JAX_PLATFORMS": "cpu", "MPLBACKEND": "Agg"}

def hxt_cli(*arguments):
    result = subprocess.run(
        [sys.executable, "-m", "hybrax.train.cli", *arguments],
        cwd=WORK, env=ENV, capture_output=True, text=True, check=True,
    )
    return result.stdout + result.stderr

# Use the same scoring functions as the mechanistic example.
sys.path.insert(0, str(OPTFED))
from run import normalized_scores, sampling_predictions, scores, select_checkpoint
```

## Define the model

The network is an Equinox MLP with five inputs, two hidden layers of four
`tanh` units and three outputs, 59 trainable parameters in total. Its inputs are
the scaled biomass, glycerol and product concentrations, the scaled
temperature, and the generation count. Its outputs become the uptake capacity
(through a softplus, so it stays positive) and the maintenance and product
fractions (through a sigmoid, so they stay between 0 and 1).

```{literalinclude} ../../../examples/gallery_optfed_ffnn/custom.py
:language: python
:start-at: class NeuralReactionModule
:end-before:     def __call__
```

The last layer starts with small weights, so the initial outputs barely depend
on the inputs. Its biases set those starting outputs: an uptake capacity of
`0.1 g glycerol/(g active biomass h)` and both fractions at `0.5`.

`__call__` turns the network outputs into the three rates declared in the
`ReactionOde` and the derivative of the generation count:

```{literalinclude} ../../../examples/gallery_optfed_ffnn/custom.py
:language: python
:pyobject: NeuralReactionModule.__call__
```

The dataset prescribes feeds and sampling, so the module returns empty modeled
inflow and outflow rates. The generation count uses a latent scale of 1,
because it already stays of order one over these processes.

````{dropdown} Complete custom.py
```{literalinclude} ../../../examples/gallery_optfed_ffnn/custom.py
:language: python
```
````

## Prepare and train

Train with Adam for 500 epochs at a constant learning rate of `0.03`. Omitting
`batch_size` uses all twelve processes in each update, so 500 epochs give 500
updates. Save a checkpoint every 10 epochs. The solver (Kvaerno5 with
`rtol=1e-4`, `atol=1e-6`), the state scales that weight the default
concentration loss, and the control scales all match the
[mechanistic example](optfed.md#prepare-and-fit-concentrations).
Because the model has a latent state, the generation count, the config sets
`allow_stateful_models`.

```{literalinclude} ../../../examples/gallery_optfed_ffnn/train-config.json
:language: json
```

The example bundles the selected checkpoint in `examples/gallery_optfed_ffnn/run/`.
To refit the model, switch to the example directory, prepare the data, and
train. Training takes about 10 minutes on one CPU device. The `--overwrite` flag
replaces the bundled run with the new fit:

```bash
cd examples/gallery_optfed_ffnn
hybrax prepare --config prepare-config.json --output-dir prepared
hybrax train --config train-config.json --overwrite
```

```{code-cell} ipython3
:tags: [remove-input]

hxt_cli("prepare", "--config", "prepare-config.json",
        "--output-dir", "prepared", "--overwrite")
output = hxt_cli("train", "--config", "train-config.json", "--overwrite")
print(next(line for line in output.splitlines() if "training complete" in line))
print("Run directory: docs/source/_data/out/runs/gallery_optfed_ffnn/run/")
```

As in the mechanistic example, evaluate each saved checkpoint on the measured
concentrations and select the one with the lowest combined NRMSE. The NRMSE
divides each species' error by its largest observed concentration. The
`select_checkpoint` function in `examples/gallery_optfed/run.py` does this, and
this example's `run.py` calls it. This is selection on training data, not
validation on held-out processes.

The output prints the ordinary R² for each species and for the
product-to-biomass ratio P/X, computed on raw concentrations. Its last line is
the lowest predicted concentration, which shows whether any prediction dips
below zero.

```{code-cell} ipython3
:tags: [remove-input]

collection = load_process_collection(OPTFED / "data.json")
checkpoint = select_checkpoint(WORK / "run", collection)
print(f"Selected checkpoint: {checkpoint.name}")
predictions = pd.read_csv(WORK / "run/selected-predictions.csv")
original = pd.read_csv(OPTFED / "original_predictions.csv")
points = sampling_predictions(collection, predictions)
for name, value in scores(points).items():
    print(f"{name:10s} R2 = {value:.4f}")

model, _ = model_load(checkpoint)
weights = jax.tree.leaves(eqx.filter(model.reaction_module.network, eqx.is_array))
assert sum(leaf.size for leaf in weights) == 59
metrics = pd.read_csv(WORK / "run/metrics.csv")
assert np.isfinite(metrics.grad_norm).all()
assert metrics.n_failed_samples.sum() == 0
species = ["c_biomass", "c_glycerol", "c_product"]
print(f"Lowest predicted concentration: {predictions[species].min().min():.3g} g/kg")
```

```{code-cell} ipython3
:tags: [remove-cell]

reference = pd.read_csv(OPTFED / "original_sampling_predictions.csv").rename(
    columns={"G_est": "predicted_glycerol", "X_est": "predicted_biomass",
             "P_est": "predicted_product"}
)
reference = points.drop(columns=[
    column for column in points if column.startswith("predicted_")
]).merge(reference, on=["process", "t"], validate="one_to_one")
assert len(reference) == len(points)
reference["predicted_P_X"] = reference.predicted_product / reference.predicted_biomass

comparison = []
for name, frame in (("Original OptFed", reference), ("Neural kinetics", points)):
    total_r2, total_nrmse = normalized_scores(frame, ["glycerol", "biomass", "product"])
    ratio_r2, ratio_nrmse = normalized_scores(frame, ["P_X"])
    comparison.append({
        "Model": name, "Total R²": total_r2, "Total NRMSE": total_nrmse,
        "X R²": normalized_scores(frame, ["biomass"])[0],
        "P R²": normalized_scores(frame, ["product"])[0],
        "P/X R²": ratio_r2, "P/X NRMSE": ratio_nrmse,
    })
comparison = pd.DataFrame(comparison).set_index("Model")
```

## Compare with OptFed

The table scores the fitted network (labelled "Neural kinetics") and the
original OptFed model on the same 83 measurement rows, using the score
definitions of the
[mechanistic example](optfed.md#prepare-and-fit-concentrations). The X, P and P/X
R² of the original model are the fit R² values of 0.83, 0.82 and 0.73 reported in
Fig. 2b of the OptFed paper \[[1](#ref-optfed-neural)\]. The other scores of the
original model are computed from predictions that we obtained by running the
authors' fitting script (`code/01_fit.py`) from the OptFed repository.

The [mechanistic example](optfed.md#compare-with-optfed) fits the same data with
its eleven parameters and reaches similar results.

```{code-cell} ipython3
:tags: [remove-input]

display(comparison.style.set_uuid("optfed_neural_comparison").format("{:.3g}"))
```

The network fits the three concentrations more closely than original OptFed,
while its product-to-biomass ratio is slightly less accurate. The loss compares
each species separately and never sees the ratio, so a lower concentration
error does not have to carry over to P/X. Original OptFed was fitted to rates
estimated from the measurements. The network is trained and selected on the
concentrations themselves, so the concentration scores favour the network.

These are the training processes, so the scores measure goodness of fit, not
performance on unseen processes. With 59 parameters against original OptFed's
eleven fitted ones, a closer fit to the training processes is expected.

## Inspect the fitted trajectories

The plot compares the fitted network, the original OptFed model and the
measurements. Original OptFed curves are semi-transparent solid lines; the
network's curves are dashed.

```{code-cell} ipython3
:tags: [remove-cell]

fig, axes = plt.subplots(12, 3, figsize=(12, 24), layout="constrained")
for row, (name, observed) in zip(axes, points.groupby("process", sort=False)):
    trajectory = predictions[
        (predictions.process == name) & (predictions.t <= observed.t.max())
    ]
    published = original[(original.process == name) & (original.t <= observed.t.max())]
    for ax, field, column in zip(
        row, ("glycerol", "biomass", "product"), ("G", "X", "P")
    ):
        ax.plot(published.t, published[column], color="tab:blue", alpha=0.45,
                label="Original OptFed")
        ax.plot(trajectory.t, trajectory[f"c_{field}"], color="tab:orange",
                linestyle="--", label="Neural kinetics")
        ax.scatter(observed.t, observed[field], color="black", s=15,
                   label="Measurements")
        ax.set(title=f"{name}: {field}", ylabel="Concentration [g/kg]")
        if field == "glycerol":
            # Keep solver roundoff around zero from filling the panel.
            upper = max(observed[field].max(), trajectory.c_glycerol.max(),
                        published.G.max(), 0.1)
            ax.set_ylim(0, 1.05 * upper)
for ax in axes[-1]:
    ax.set_xlabel("Time [h]")
axes[0, 0].legend()
fig.savefig(WORK / "run/concentration_fit.png", dpi=140)
plt.close(fig)
```

```{code-cell} ipython3
:tags: [remove-input]

display(Image(filename=str(WORK / "run/concentration_fit.png")))
```

## Run the example locally

The prepare and train commands above write to `prepared/` and `run/` inside
`examples/gallery_optfed_ffnn/`. To simulate the bundled model, use the same
directory:

```bash
hybrax forward --config forward-config.json --output-dir run/forward
```

The bundled forward configuration loads `run`, which points to its latest
checkpoint. After a refit, that is the final epoch. To simulate a specific
checkpoint, set `models` to its directory, for example
`["run/checkpoints/step_00500"]`.

The optional `run.py` script prepares, trains, selects the checkpoint with the
lowest concentration NRMSE, and runs that checkpoint:

```bash
python run.py
```

This page's executed run is in
`docs/source/_data/out/runs/gallery_optfed_ffnn/run/`.

(ref-optfed-neural)=
## References

1. Schlögel, G., Lück, R., Kittler, S., Spadiut, O.,
   Kopp, J., Zanghellini, J., & Gotsmy, M. (2024). Optimizing bioprocessing
   efficiency with OptFed: Dynamic nonlinear modeling improves
   product-to-biomass yield. *Computational and Structural Biotechnology
   Journal*, 23, 3651-3661.
   [doi:10.1016/j.csbj.2024.09.024](https://doi.org/10.1016/j.csbj.2024.09.024).
