---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python3
  name: python3
mystnb:
  execution_timeout: 21600
---

# HEK293: PCA reaction modes with an LSTM

Mammalian process models often track many more extracellular states than
microbial models, including amino acids and other metabolites. If a hybrid
model's neural network has to learn a specific rate for every one of these
states, it becomes large and hard to train.

Ramos et al. [<a href="#ref-ramos">1</a>] addressed this with a smart trick:
they converted measured concentrations into cumulative reacted amounts, applied
principal component analysis (PCA) across all cultures, and extracted a few of
the most important principal components (which we will call "reaction modes"
here). Each mode is a pattern of correlated net production and consumption. In
their experimental study, seven modes captured more than 99% of the variation in
the normalized cumulative reacted amounts of the 27 measured extracellular
species. Therefore, instead of having the neural network predict 27 separate
reaction rates (one for each state), it only needs to predict seven mode rates
without substantially compromising the fit. A fixed matrix derived from the PCA
loadings then maps these seven mode rates back onto the reaction terms for all
27 states.

The authors' experimental measurements are not in their public repository, so
this example uses the synthetic data they provide instead: nine HEK293 cultures
with 25 extracellular states [<a href="#ref-code">2</a>]. Here four PCA modes
capture 99.66% of the normalized reacted-amount squared magnitude, so the ANN
learns four rates that drive all 25 concentration balances.

This page is an illustrative example of how to implement a model similar to the
authors' in Hybrax. For the sake of simplicity, we do not reproduce their custom
LSTM architecture exactly because this would increase the code complexity
without adding much value for the reader. Instead, we build on Hybrax's built-in
LSTM reaction module. The two networks differ in some regards, but total
parameter count and final accuracy are comparable to the authors' saved MATLAB
model.

## How the pieces fit

This example follows the standard Hybrax pipeline. Each step has its own file
in `examples/gallery_hek293_lstm/`:

> Note: If this is your first Hybrax model, work through the
> [tutorials](../tutorials/01_your_first_dataset.md) first;
> [Tutorial 4](../tutorials/04_your_first_custom_py.md) introduces `custom.py`.
> [Stateful models](stateful.md) explains recurrent reaction modules in more
> depth.

1. **Import** (`import_from_mat.py`; creates `data.json`, `train-config.json`).
   The importer reads the authors' MAT file and converts the nine cultures into
   a Hybrax `BioProcessCollection` with concentrations, reactor volume, feed and
   feed composition. It computes the PCA reaction matrix and stores it in every
   process as a `ReactionOde`. It also writes the training configuration,
   because the config carries statistics computed from all nine cultures (see
   [Validate the data and train the model](#validate-the-data-and-train-the-model)
   for why).
2. **Prepare** (`prepare-config.json`; creates `prepared/prepared.json`).
   Prepare supports data transformation and augmentation, but neither is
   required here. It also validates the collection, including the reaction
   equations and feed controls, and records preparation metadata before writing
   the training-ready artifact. Training reads this artifact (rather than
   `data.json`).
3. **Train** (`train-config.json`, `custom.py`; creates `run/metrics.csv`,
   `run/losses.csv`, model files in `run/model/`, and checkpoints in
   `run/checkpoints/`). The config points to `custom.py`, and Hybrax looks up
   its hook functions by name. Here they build the LSTM reaction module, the
   input scaling, and a loss that uses the same max-normalized concentration
   errors as MATLAB. Training writes losses and checkpoints to `run/`. The
   config requests 20,000 epochs with validation every 10 epochs.
4. **Results** (`plot_data.py`, `results.py`; creates results CSVs and plot
   PNGs). `plot_data.py` plots the source data before training. `results.py`
   selects the checkpoint with the lowest validation loss and compares it with
   the authors' saved MATLAB predictions.

The data therefore flow from `source-data.mat` through the importer to
`data.json`, through `hybrax prepare` to `prepared/`, and through
`hybrax train` to `run/`. The `run.py` script runs all four steps in order.

```{code-cell} ipython3
:tags: [remove-cell]

import os
import shutil
import subprocess
import sys
from pathlib import Path
from pprint import pprint

from IPython.display import Image, display

import hybrax.format as hxf

WORK = Path("../_data/out/runs/gallery_hek293_lstm").resolve()
EXAMPLE = Path("../../../examples/gallery_hek293_lstm").resolve()
WORK.mkdir(parents=True, exist_ok=True)
# For a local fast rebuild, explicitly reuse a completed run copied into WORK.
# Four-unit LSTM, hidden-only linear head, all weights initialized at 0.1x.
REUSE_RUN = os.environ.get("HYBRAX_HEK293_REUSE_RUN") == "1"
for filename in (
    "import_from_mat.py", "plot_data.py", "custom.py", "results.py",
    "source-data.mat", "feed.json",
    "matlab-reference.npz", "provenance.json", "prepare-config.json",
    "train-config.json",
):
    shutil.copyfile(EXAMPLE / filename, WORK / filename)
ENV = os.environ | {
    "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1", "HYBRAX_TRAIN_DEVICES": "1",
    "MPLBACKEND": "Agg",
}

def hxt_cli(command, config):
    with (WORK / f"{command}.log").open("w") as log:
        subprocess.run(
            [sys.executable, "-m", "hybrax.train.cli", command,
             "--config", config, "--overwrite"],
            cwd=WORK, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True,
        )

sys.path.insert(0, str(WORK))
from results import (
    evaluate_checkpoints, plot_loss_curves, plot_trajectories,
    score_selected_checkpoint, select_checkpoint,
)

subprocess.run([sys.executable, "import_from_mat.py"], cwd=WORK, env=ENV, check=True)
collection = hxf.serialization.load_process_collection(WORK / "data.json")
reaction_ode = collection.processes["Br1"].reaction_ode

```

## Import the synthetic cultures

The bundled
{download}`source-data.mat <../../../examples/gallery_hek293_lstm/source-data.mat>`
holds the authors' nine synthetic fed-batch cultures. Each one covers ten days
of biomass growth, mAb production, and 23 nutrient and metabolite
concentrations, sampled every 12 h. All cultures start at 0.27 L with the same
feed composition, but each receives a different feed profile.

The importer turns each culture into one Hybrax `BioProcess`, named Br1 to Br9
in file order. Each process gets one `TimeSeries` per state, the total reactor
volume, and a single feed. The feed is a controlled, continuous `Inflow`, which
means it is a known input rather than something the model predicts. Its
cumulative added volume is $V(t) - V(0)$: the importer treats every volume
increase as feed and does not model sample removal (just like the authors'
MATLAB code). The feed composition comes from {download}`feed.json
<../../../examples/gallery_hek293_lstm/feed.json>`. The only unit conversion is
volume, from mL to L. Concentrations are used as they are.

```{literalinclude} ../../../examples/gallery_hek293_lstm/import_from_mat.py
:language: python
:start-at: processes = {}
:end-at: hxf.serialization.save_process_collection
```

The `save_process_collection` call writes `data.json`, which stores the
measurements, units, time axes, feed definition, and reaction equations
together. Load it from Python with the same serialization API:

```{code-cell} ipython3
collection = hxf.serialization.load_process_collection(WORK / "data.json")
print(f"Loaded {len(collection.processes)} cultures: {', '.join(collection.processes)}")
```

````{dropdown} What the MAT file contains
The file holds one MATLAB struct per culture. All nine share the same time
grid: 21 samples, every 12 h from 0 to 240 h. The importer uses four fields of
each struct:

- `time`: the 21 sample times in hours.
- `conc`: a 25×21 array of concentrations, one row per extracellular state.
- `vol`: the reactor volume in mL at each sample. Every culture starts at
  270 mL, but the feed profiles differ, so final volumes range from about
  0.46 L (Br7) to 4.9 L (Br8).
- `m_r`: a 25×21 array of cumulative reacted amounts, computed by the authors'
  preprocessing. It is zero at 0 h. Only the PCA step uses these values.

The 25 states are biomass, the mAb product, glucose, pyruvate, lactate,
ammonium, and 19 amino acids. The importer renames only `Xv` to `biomass`,
which Hybrax validation requires, and keeps the other MAT-file identifiers,
such as `mAb`, `Glc` and `Gln`. It assigns g/L to biomass, mg/L to mAb, and mM
to the other 23 states. The struct also has fields the importer does not read
(`accum`, `val`, `std` and a batch ID).
````

````{dropdown} Biomass: from cell counts to dry weight
`Xv` already contains dry-weight-equivalent viable biomass in g/L, and the
importer uses these values directly. The authors' preprocessing derived them
from the workbook's `X` column, the viable cell density in million cells/mL,
with an assumed dry mass per cell: 216.1 pg before 100 h and 287.496 pg from
100 h onward. This converts a cell count to a dry-weight concentration. It is
not a wet-to-dry biomass conversion.
````

````{dropdown} Source files and provenance
The inputs come from the authors' repository at a pinned commit
[<a href="#ref-code">2</a>]:

- {download}`source-data.mat <../../../examples/gallery_hek293_lstm/source-data.mat>`
  is `data/data.mat` from that commit.
- {download}`feed.json <../../../examples/gallery_hek293_lstm/feed.json>` is
  extracted from the `synthetic DoE` sheet of the repository's workbook.
- {download}`matlab-reference.npz <../../../examples/gallery_hek293_lstm/matlab-reference.npz>`
  holds the concentration predictions of the authors' saved MATLAB model
  (`hybrid_LSTM_1.mat`). It contains no network weights, and training never
  reads it. The comparison plots and scores use it.
- {download}`provenance.json <../../../examples/gallery_hek293_lstm/provenance.json>`
  records the source commit, the file hashes, and the workbook cells behind
  `feed.json`.

`feed.json` lists 25 feed concentrations in the same units as the reactor
states. Biomass, mAb, lactate, ammonium and several amino acids have a feed
concentration of zero. The importer checks that the feed names after biomass
match the MAT file's state order.

```{literalinclude} ../../../examples/gallery_hek293_lstm/feed.json
:language: json
:caption: feed.json
```
````

````{dropdown} Full importer: import_from_mat.py
```{literalinclude} ../../../examples/gallery_hek293_lstm/import_from_mat.py
:language: python
:caption: import_from_mat.py
```
````

### Train, validation and test cultures

The split follows the authors' code:

| Role | Cultures | Used for |
| --- | --- | --- |
| Training | Br1, Br2, Br3, Br4, Br9 | gradient updates |
| Validation | Br7 | checkpoint selection |
| Test | Br5, Br6, Br8 | final comparison only |

All nine cultures go into `data.json`. The split lives in the training
configuration: `data.processes` lists the training cultures and
`train.holdout_processes` names Br7. This makes Hybrax simulate Br7 with the
current model and evaluate the same loss used for training at each checkpoint
boundary. Br7 is not used for gradient updates. Here `checkpoint.every=10` means
this happens every 10 epochs. With `checkpoint.select_by="holdout_loss"` and
`checkpoint.keep_best=20`, Hybrax retains the 20 checkpoints with the lowest Br7
losses, plus the final checkpoint.

The test cultures appear in neither list, so Hybrax never fits or validates on
their trajectories. The one place they enter is the preprocessing statistics,
discussed in
[Validate the data and train the model](#validate-the-data-and-train-the-model).

### Inspect the culture trajectories

The overview shows biomass, product, glucose and glutamine alongside reactor
volume and feed rate. The lines connect the 21 data points of each culture;
they are not model predictions. The feed rate is $\Delta V / \Delta t$ for each
12-hour interval, plotted as steps to match the piecewise-linear cumulative
feed that Hybrax uses. Black-to-grey shades mark training cultures, orange
marks Br7, and red-to-pink shades mark test cultures. Within each set, shades
and line patterns identify the same culture in both figures.

```{code-cell} ipython3
:tags: [remove-input]

from plot_data import plot_data

# Only Xv is renamed to biomass; plots retain the source-data labels.
plot_data(WORK)
display(Image(filename=str(WORK / "synthetic-overview.png")))
```

Br7 ends with a smaller volume than the training cultures. Its feed stops at
96 h on this measurement grid, while several other cultures keep receiving
feed. A concentration can change through reactions or through dilution, so the
model needs the feed history as well as the concentrations.

````{dropdown} All 25 extracellular states across the nine cultures
```{image} ../_data/out/runs/gallery_hek293_lstm/synthetic-all-states.png
:alt: All 25 concentration trajectories for the nine synthetic cultures.
```
````

## The model

Hybrax splits each state's derivative into a reaction part and a transport part.
Users declare the reaction part and Hybrax handles the transport part (i.e. it
generates feed and dilution from the volume description). In this example the
reaction part is biomass concentration times the PCA reaction matrix and the
four learned mode rates predicted by the ANN.

### Map four learned rates to 25 balances

The importer builds the reaction matrix from the reacted amounts `m_r` in the
MAT file, so the authors' preprocessing carries over unchanged. It divides each
reacted-amount column by its maximum absolute value and computes an uncentered
singular-value decomposition. Multiplying the first four loadings by those
maxima gives the fixed reaction matrix:

```{literalinclude} ../../../examples/gallery_hek293_lstm/import_from_mat.py
:language: python
:start-at: N_RATES = 4
:end-at: reaction_matrix = amount_scale[:, None] * coefficients
```

```{dropdown} How does the PCA reaction matrix work?
**Start with reaction amounts, not raw concentrations.** A concentration
can fall because cells consume a compound or because feed dilutes it. The source
preprocessing multiplies concentrations by reactor volume, subtracts initial and
fed amounts, and accounts for removed material. The resulting `m_r` describes
the cumulative amount produced or consumed by reactions. Our importer reads
these precomputed values from the MAT file.

**Find a few shared reaction patterns.** Stack measurement times from all nine
cultures into one matrix, with 25 columns for the extracellular states. Scale each
column by its maximum absolute reacted amount, then apply an uncentered SVD.
The first four directions describe patterns of production and consumption that
occur together. Restoring the column scales gives the fixed 25×4 matrix `M`.
Its columns carry the corresponding amount units; they are not dimensionless
stoichiometric coefficients.

**Use the same patterns in concentration balances.** If cumulative reaction
amounts are approximated by `A(t) ≈ M z(t)`, their instantaneous rates are
approximated by `dA/dt ≈ M dz/dt`, because `M` is fixed. Dividing every component
by the same reactor volume changes the magnitude, but preserves these four
directions. The LSTM learns four instantaneous, biomass-specific coefficients
`r`, giving the concentration reaction term `X M r`, where `X` is biomass
concentration. It does not predict the cumulative PCA scores `z`.
Hybrax integrates concentrations and adds feed and dilution separately.

**Keep the approximation limits in mind.** The four directions capture 99.66%
of the normalized cumulative-amount squared magnitude. This does not guarantee
equally accurate instantaneous reaction rates.
```

The network learns the rates of these four modes, while the PCA matrix stays
fixed. For concentration vector $C$, biomass $X$, reaction matrix $M$, and four
learned rates $r$, the reaction contribution is

$$
\left.\frac{dC}{dt}\right|_{\mathrm{reaction}} = X M r.
$$

The reacted amounts `m_r` only shape `M`. Training never uses them, or any
rates derived from them, as targets: the loss compares simulated and measured
concentrations directly. Hybrax does this out of the box, so we do not have
to estimate rates from the data before training.

### Declare the balances as a `ReactionOde`

A `ReactionOde` declares named rates and the reaction part of each state's
derivative as an expression in those rates and the states. The importer writes
one expression per state, `biomass * (M[state, 0] * r_pc1 + ... )`, and
attaches the same `ReactionOde` to every culture:

```{literalinclude} ../../../examples/gallery_hek293_lstm/import_from_mat.py
:language: python
:start-at: rate_names =
:end-before: feed =
```

The rates have no bounds. The reaction module defined in `custom.py` must
return four values, one per name in `rates`. Expand the output below to see the
rate declarations and all 25 equations:

```{code-cell} ipython3
:tags: [hide-output]

pprint(reaction_ode)
```

```{note}
You might have noticed that all four coefficients in the mAb equation are
negative. This is OK here: the network predicts signed mode rates, so a
negative coefficient times a negative rate produces mAb. The sign of each
PCA mode is arbitrary, and only the sum over all four modes sets the
net reaction term.
```

### Feed, dilution and discrete events

The `ReactionOde` contains no feed terms. Hybrax adds feed and dilution from
the controlled `feed` inflow, its feed medium, and the reactor volume.

In this synthetic dataset, the reactor volume is only known on the 12-hour grid.
Linear interpolation of the cumulative feed gives a constant feed rate within
each interval, and that rate can jump at every interior sample time. The
importer declares the 19 times from 12 to 228 h as `DiscreteEvents`. Hybrax
passes them to the solver's step-size controller, so the solver steps exactly
onto each jump instead of smoothing over it.

## Network, scaling and loss

`train-config.json` sets `custom_py` to `custom.py`. Hybrax imports that file
and looks up hook functions by their names; any hook we do not define keeps
its default. This example defines three:

- `build_reaction_module` builds the LSTM that predicts the four mode rates.
- `estimate_all_scales` sets how states are standardized.
- `build_loss_module` defines the training and validation loss.

For more details, the [Customization](../train/hooks_cheatsheet.md) docs list
all hooks and their signatures.

### The LSTM as extra ODE states

An LSTM is a discrete recurrence. Each step takes the current inputs plus a
hidden state $h$ and a cell state $c$, and returns updated states. A Hybrax
model is an ODE, so the built-in `DefaultLstmReactionModule` turns the
recurrence into extra ODE states. With four hidden units, $h$ and $c$ add eight
latent states, which start at zero and are integrated alongside the 25
concentrations. Their derivative is `next_state - current_state`, the
difference between what the LSTM cell would return and the current state.
They follow the cell's update continuously, rather than once per 12-hour
interval as in Ramos et al.'s MATLAB code.

Latent states give the model memory of the path the culture has taken. Hybrax
rejects a reaction module with latent states unless `train.allow_stateful_models`
is `true`, and the generated config sets it.

`InputLstm` subclasses `DefaultLstmReactionModule` with `hidden_width=4` and
changes its inputs. For this dataset, the default cell reads 28 inputs: the
25 concentrations, volume, cumulative feed and feed rate. The MATLAB network
reads the 25 concentrations, volume and time instead. Setting `input_size=27`
lets the built-in constructor initialize the cell. The subclass replaces the
default output head with a 4-to-4 linear layer that reads only the hidden
state. It retains the default head's hidden-state weight columns and biases.
After construction, the hook multiplies the LSTM input weights, recurrent
weights and head weights by 0.1, leaving biases unchanged. The model has 532
trainable parameters.

The `input_features` override assembles those 27 standardized features:

```{literalinclude} ../../../examples/gallery_hek293_lstm/custom.py
:language: python
:pyobject: InputLstm.input_features
```

The custom forward pass supplies these features to the LSTM cell. Only the
current four-dimensional hidden state goes to the linear head, which predicts
the four signed mode rates. Hidden and cell states evolve continuously with
derivatives equal to the LSTM update minus the current state:

```{literalinclude} ../../../examples/gallery_hek293_lstm/custom.py
:language: python
:pyobject: InputLstm.__call__
```

The model has no other layers, dropout, data augmentation or learning-rate
schedule.

````{dropdown} How the Hybrax model differs from MATLAB
- **Recurrent cell.** MATLAB uses a custom LSTM cell, updated once per
  12-hour interval. Hybrax uses a standard LSTM cell whose states follow the
  update continuously inside the ODE.
- **Output head.** Hybrax maps its four hidden-state values to four mode rates
  with a learned linear layer. MATLAB uses its custom recurrent-cell output
  directly as the mode rates.
- **Input normalization.** Both use the means and SDs of all nine cultures. In
  the authors' simulation code (`hybnet/hnetsimul.m`), the concentration input
  is written as `predc(t-1,:)-data.xmean./(data.xstd)`. MATLAB's operator
  precedence evaluates this as $x - (\mathrm{mean}/\mathrm{SD})$. Hybrax uses
  the z-score $(x - \mathrm{mean})/\mathrm{SD}$.
- **Optimization.** The two models differ in initialization, gradients and
  optimizer settings, so similar scores do not imply similar learned rates.
````

### One scaling for the solver and the network

Hybrax solves the ODE in scaled units, so that the solver sees values of
similar size (see [Scaling](../train/scaling.md)). The `estimate_all_scales`
hook chooses the transformation for each kind of quantity. Here it returns an
`AffineScaler` for the 25 concentrations and for the reactor volume. An
`AffineScaler(scale, offset)` maps a physical value $x$ to
$(x - \mathrm{offset}) / \mathrm{scale}$. With the mean as offset and the
sample standard deviation (SD) as scale, every state becomes a z-score:

```{literalinclude} ../../../examples/gallery_hek293_lstm/custom.py
:language: python
:start-at:     return EstimatedScales(
:end-at:         SCALE_V_in_cumulative=
:dedent: 4
```

This one choice does two jobs. Hybrax integrates the ODE in these z-score
coordinates, which keeps the solver well conditioned. It also hands the same
scaled values to the reaction module as `inputs.SCL_modeled_RMCs` and
`inputs.SCL_modeled_V`, so the ANN inputs arrive standardized without any extra
code. Time is not a state, so we have to standardize it ourselves in
`input_features`. The means and SDs come from the `custom` block of the config,
and the hook reorders them by name to match the ODE's species order. All other
scales are ones or empty arrays: the four mode rates and the feed keep their
physical units, and the model has no other inputs.

### MATLAB-normalized concentration errors

The default loss compares simulated and measured states in the scaled units
above, so its residuals are differences of z-scores. The original MATLAB code
instead divides each physical residual by that state's maximum concentration
across all nine cultures, which we will reproduce here.

Recall that the mean cancels when two z-scores are subtracted and multiplying
the result by `SD / maximum` therefore gives the physical residual divided by
the maximum:

```{literalinclude} ../../../examples/gallery_hek293_lstm/custom.py
:language: python
:pyobject: MatlabLoss
```

```{literalinclude} ../../../examples/gallery_hek293_lstm/custom.py
:language: python
:pyobject: build_loss_module
```

The loss excludes the first measurement, which prescribes the initial condition.
The normalized mean squared error (NMSE) averages the squared normalized
residuals over all 25 states and the 20 post-initial measurements. Every state
has equal weight. NRMSE is its square root. Training, Br7 checkpoint selection
and the reported comparisons all use this normalization.

There is no penalty for negative concentrations. Predictions can become negative,
as can the authors' saved MATLAB predictions. The fixed PCA reaction modes do
not enforce positivity.

````{dropdown} Complete custom.py
```{literalinclude} ../../../examples/gallery_hek293_lstm/custom.py
:language: python
:caption: custom.py
```
````

## Validate the data and train the model

Before training, we run `hybrax prepare` to validate the imported data and create
a training-ready artifact. This step takes a separate configuration file. Here,
`prepare-config.json` tells Hybrax to read `data.json`, apply strict format
validation,[^prepare-validation] and write the prepared artifact to `prepared/`:

[^prepare-validation]: Validation checks time-series shapes and timestamps,
    initial measurements, feed and volume-change definitions, and whether
    measurements respect their declared bounds. It also checks reaction-equation
    definitions and consistency of species names and units across cultures.
    Unit checks concern consistency of declared units, not dimensional analysis
    of the equations. With strict validation enabled, failed checks stop
    preparation.

```{literalinclude} ../../../examples/gallery_hek293_lstm/prepare-config.json
:language: json
:caption: prepare-config.json
```

```{code-cell} ipython3
:tags: [remove-cell]

if not REUSE_RUN:
    hxt_cli("prepare", "prepare-config.json")
```

The importer generates `train-config.json`. This includes the training and
validation culture selections, training settings, checkpoint retention settings,
and the path to `custom.py`. Its `custom` block also
contains the means and sample standard deviations of the 27 network inputs (25
concentrations, reactor volume, and time), and the maximum of each
concentration. Means and standard deviations are used to z-score the ANN inputs;
the concentration and volume scales also define the coordinates used by the ODE
solver. The concentration maxima are required to reproduce MATLAB's loss
normalization, which divides each physical concentration residual by its maximum
rather than its standard deviation. Following the original MATLAB
implementation, the importer also computes these statistics from all nine
cultures (even though this constitutes data leakage; best practice would be to
fit PCA and all other preprocessing on the training cultures only).

```{literalinclude} ../../../examples/gallery_hek293_lstm/train-config.json
:language: json
:caption: train-config.json
```

The rest of the config sets up the run. Training uses the five training cultures
in every update for 20,000 epochs, with seed 0 and Adam at a constant learning
rate of `5e-5`. Hybrax evaluates Br7 at every checkpoint, which here means every
10 epochs. It keeps the 20 checkpoints with the lowest Br7 losses plus the final
one, and does not copy the prepared data into each checkpoint [^prepared-checkpoints].
The test cultures are not part of training at all.

[^prepared-checkpoints]: Copying the same prepared data into every checkpoint
    would be an unnecessary waste of disk space here. However, setting
    `checkpoint.bundle_prepared=false` means the checkpoint directories are not
    fully self-contained. When sharing a checkpoint with other researchers in
    this case, include the prepared data as `prepared.json.gz` inside the
    checkpoint directory.

To run the steps yourself, switch to the example directory, regenerate the
dataset and config, prepare, and train:

```bash
cd examples/gallery_hek293_lstm
python import_from_mat.py
hybrax prepare --config prepare-config.json
hybrax train --config train-config.json
```

```{code-cell} ipython3
:tags: [remove-input]

if not REUSE_RUN:
    hxt_cli("train", "train-config.json")
print("Run directory: ../_data/out/runs/gallery_hek293_lstm/run")
```

## Loss curves

```{code-cell} ipython3
:tags: [remove-stderr]

curves = evaluate_checkpoints(WORK)
plot_loss_curves(WORK, curves)
display(Image(filename=str(WORK / "loss-curves.png")))
```

The curves show training and Br7 NMSE on a logarithmic axis. Dashed lines show
the authors' saved MATLAB model, scored the same way. Training loss is recorded
before each update and Br7 loss after it. The marker shows the lowest Br7 loss,
which selects the checkpoint. The selected minimum is near the end of the
20,000-epoch run.

## Compare with MATLAB

The selected Hybrax checkpoint is comparable to the authors' saved MATLAB
model, using a four-unit LSTM and a small linear output head:

```{code-cell} ipython3
:tags: [remove-stderr]

selected_epoch = select_checkpoint(WORK, curves)
comparison = score_selected_checkpoint(WORK, selected_epoch)
display(comparison.style.set_uuid("hek_comparison").format("{:.5f}"))
```

````{dropdown} How the test scores are calculated
After selecting the checkpoint using Br7, `score_selected_checkpoint` reloads
that model and simulates the training and test cultures with its parameters
and normalization fixed. It reuses the checkpoint's saved Br7 predictions.
The test NRMSE pools squared, max-normalized concentration errors across all
25 states and all three test cultures before taking the square root, excluding
the initial condition. The function writes all three splits' scores for Hybrax
and MATLAB to `comparison-scores.csv`:

```{literalinclude} ../../../examples/gallery_hek293_lstm/results.py
:language: python
:pyobject: score_selected_checkpoint
```
````

Both models are scored with the NRMSE defined
[above](#matlab-normalized-concentration-errors),
on the same measurement times. Checkpoint selection follows MATLAB's criterion:
the lowest Br7 NMSE. For Hybrax, this selects epoch 19,970. The code below
calculates validation NRMSE from the selected checkpoint's saved Br7 predictions,
avoiding an additional simulation. The test scores come from simulating the
selected checkpoint on the three test cultures. Both the Hybrax
and MATLAB checkpoints are selected using Br7, so their validation scores
reflect model selection rather than an independent test.

### All 25 states on Br7

```{code-cell} ipython3
selected_epoch = select_checkpoint(WORK, curves)
scores = plot_trajectories(WORK, selected_epoch)
print(f"Checkpoint: epoch {selected_epoch:,}")
display(scores[["All 25"]].style.set_uuid("hek_scores").format("{:.5f}"))
display(Image(filename=str(WORK / "Br7-trajectories.png")))
```

Each panel compares one state on Br7, and its title gives both NRMSE values.
A grey background means Hybrax has the lower NRMSE for that state; white means
MATLAB's is lower or equal. Open circles are synthetic measurements, solid
lines are Hybrax predictions, and dashed lines are the authors' saved MATLAB
predictions. Both lines connect predictions at the 21 measurement times. The
vertical dotted line marks the end of feeding at 96 h.

The two models fit individual states differently even though their aggregate
errors are close. Neither model constrains concentrations to stay
non-negative, so both can predict negative values.

## Run the example locally

From `examples/gallery_hek293_lstm/`, run the whole pipeline with:

```bash
python run.py
```

The script regenerates `data.json` and `train-config.json`, plots the source
data, prepares the collection, trains the model, and runs `results.py`. It
replaces any existing `prepared/` and `run/`. It writes `loss-curves.csv`,
`loss-curves.png`, `selected-checkpoint.txt`, `comparison-scores.csv`,
`Br7-scores.csv`, and `Br7-trajectories.png` to the example directory. Everything
runs from the bundled inputs, without MATLAB or network access. This page's
executed run is in `docs/source/_data/out/runs/gallery_hek293_lstm/run/`.

````{dropdown} run.py
```{literalinclude} ../../../examples/gallery_hek293_lstm/run.py
:language: python
:caption: run.py
```
````

````{dropdown} Plot the source data: plot_data.py
```{literalinclude} ../../../examples/gallery_hek293_lstm/plot_data.py
:language: python
:caption: plot_data.py
```
````

````{dropdown} Checkpoint selection, scoring and plotting: results.py
```{literalinclude} ../../../examples/gallery_hek293_lstm/results.py
:language: python
:caption: results.py
```
````

## Conclusion

This example imported nine synthetic HEK293 cultures from Ramos et al.
[<a href="#ref-ramos">1</a>] into a reusable Hybrax collection, with measurements,
units, feed and reaction equations stored together in `data.json`. PCA on the
cumulative reacted amounts reduced the number of rates the network needs to
learn to four, which are mapped to the reaction terms in all 25 concentration
balances using the rescaled PCA loadings. A continuous-time LSTM with four
hidden units and a small linear head was trained directly against measured
concentrations using `hybrax prepare`, `hybrax train`, and three hooks in
`custom.py`.

The resulting fit is comparable to the authors' saved MATLAB model, despite
differences in the recurrent cells, output heads and optimization. Validation
and test NRMSE are 0.05207 and 0.03492, compared with 0.05761 and 0.03757 for
MATLAB. This is a synthetic benchmark: preprocessing uses all nine cultures,
including the held-out cultures (same as the MATLAB code), and neither model
guarantees nonnegative concentrations. Adding a soft penalty for negative
concentrations would be a straightforward extension; see the
[custom loss example](dense_loss.md) for details.

## References

(ref-ramos)=
1. Ramos, J. R. C., Pinto, J., Poiares-Oliveira, G., Peeters, L., Dumas, P., &
   Oliveira, R. (2024). Deep hybrid modeling of a HEK293 process: Combining long
   short-term memory networks with first principles equations.
   *Biotechnology and Bioengineering*, 121(5), 1554-1568.
   [doi:10.1002/bit.28668](https://doi.org/10.1002/bit.28668).

(ref-code)=
2. [Authors' MATLAB code and synthetic data](https://github.com/jrcramos/Hybrid-modeling-of-bioreactor-with-LSTM/tree/d4197ed84ba46bfb53c3e4f42cfe00f44916edb5),
   frozen at commit `d4197ed84ba46bfb53c3e4f42cfe00f44916edb5`.
