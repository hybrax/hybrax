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
  execution_timeout: 3600
---

# OptFed mechanistic model

Hybrax was primarily designed for hybrid models but fully mechanistic models are
also supported. Here we illustrate this using the reduced OptFed model (Schlögel
et al. \[[1](#ref-optfed)\]) as example. This means that, instead of training a
neural network to predict rates, we define the rate equations of the mechanistic
model with Hybrax and fit its eleven free parameters on the concentration data
directly.

The kinetics of the model are determined by three rate equations: for substrate (glycerol)
uptake, substrate spent on maintenance and substrate spent on product formation.
Temperature affects product formation through an
Eyring rate law. Accumulated generations, or cell doublings since induction,
inhibit production. The published model has twelve kinetic parameters; eleven
are fitted here, while the glycerol saturation constant remains fixed at
`0.001 g/kg`, as in the authors' final ODE implementation. The per-substrate
yields of biomass and product also remain fixed.

This example uses the twelve experimental fed-batch processes published with
OptFed, originating from Kittler et al. \[[2](#ref-kittler)\].
The raw data can be found on [GitHub](https://github.com/gschloegel/OptFed/tree/7811f7b9ef76d624f53b852c03d06cb1b5cc96a3/data).
The model fits them jointly with one shared set of eleven parameters.

## How the pieces fit

This example follows the standard Hybrax pipeline. Each step has its own file
in `examples/gallery_optfed/`:

1. **Data** (`import_data.py` → `data.json`). The importer converts the
   published CSVs into a Hybrax `BioProcessCollection`: measurements, feeds,
   sampling withdrawals, and temperature for each process. Each process also
   carries a `ReactionOde`, which declares the named rates and the reaction
   part of each state's derivative. The bundled `data.json` is the output of
   this step, so you only need to rerun the importer to regenerate it.
2. **Prepare** (`hybrax prepare`, `prepare-config.json`). Prepare validates
   the collection, checks that all processes share the same state and control
   layout, and writes `prepared/prepared.json`. Training reads this artifact
   rather than `data.json`.
3. **Train** (`hybrax train`, `train-config.json` + `custom.py`). The config
   points to `custom.py`, and Hybrax looks up its hook functions by name.
   Here `build_reaction_module` returns the mechanistic `RateModule` that
   computes the named rates, and the other hooks choose the solver, the
   learning-rate schedule, and the scales.
4. **Forward** (`hybrax forward`, `forward-config.json`). Forward simulates the
   fitted model and writes trajectories and plots.

If this is your first Hybrax model, work through the
[tutorials](../tutorials/01_your_first_dataset.md) first;
[Tutorial 4](../tutorials/04_your_first_custom_py.md) introduces `custom.py`.
[The reaction module](../train/reaction_module.md) and
[Customization](../train/hooks_cheatsheet.md) document the interfaces used
below.

```{code-cell} ipython3
:tags: [remove-cell]

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

from hybrax.format.serialization import load_process_collection
from hybrax.train import model_load

# All reloaded checkpoints belong to the same run.
logging.getLogger("hybrax.train.serialization").setLevel(logging.ERROR)

WORK = Path("../_data/out/runs/gallery_optfed").resolve()
WORK.mkdir(parents=True, exist_ok=True)
EXAMPLE = Path("../../../examples/gallery_optfed").resolve()
for name in (
    "data.json", "import_data.py", "custom.py", "run.py", "prepare-config.json",
    "train-config.json", "forward-config.json", "original_predictions.csv",
    "original_sampling_predictions.csv",
):
    shutil.copyfile(EXAMPLE / name, WORK / name)
ENV = os.environ | {
    "JAX_PLATFORMS": "cpu", "HYBRAX_TRAIN_DEVICES": "1", "MPLBACKEND": "Agg",
}

def hxt_cli(*arguments):
    result = subprocess.run(
        [sys.executable, "-m", "hybrax.train.cli", *arguments],
        cwd=WORK, env=ENV, capture_output=True, text=True, check=True,
    )
    return result.stdout + result.stderr

# Use the same scoring functions as the standalone example.
sys.path.insert(0, str(WORK))
from run import normalized_scores, sampling_predictions, scores, select_checkpoint
from custom import BOUNDS
```

## Load the published data

The bundled {download}`data.json <../../../examples/gallery_optfed/data.json>`
is a Hybrax `BioProcessCollection`. It is generated from the
four CSVs from the [OptFed GitHub repo](https://github.com/gschloegel/OptFed/tree/7811f7b9ef76d624f53b852c03d06cb1b5cc96a3/data).
The full importer code is in the collapsible box at the end of this section.

Each process contains the measured glycerol, total biomass, and product
concentrations, the carbon and base feeds, the sampling withdrawals, and the
measured temperature. Time zero is induction. OptFed uses a mass-based balance,
so `volume` stores reactor mass in kg, feed amounts are in kg, and
concentrations use g/kg. Temperature is in Kelvin. The importer smooths the
cumulative feeds, and Hybrax uses their derivatives as feed rates.

````{dropdown} How the importer converts the raw CSVs

- `sampling_points.csv` supplies initial reactor mass and raw glycerol, total
  biomass, and product concentrations. As in OptFed's
  original fitting script, the row with index 40 in `sampling_points.csv`,
  from `DoE1_R3` at 10.35 h, is excluded as an erroneous measurement (but the
  removed sample volume remains in the mass balance).
- `volume_flow_rates.csv` supplies cumulative carbon and base feeds.
  The base feed controls pH and adds mass without glycerol. Each feed
  starts at zero added mass at induction. Downward
  reactor-mass jumps estimate individual withdrawals, including extra draws
  between concentration measurements.
- `temperatures.csv` supplies the measured temperature control in Kelvin.
- `c_feed.csv` supplies glycerol concentration in the carbon feed (for the
  individual processes). The importer uses OptFed's density correction to
  convert this value from g/L to g/kg.

The reactor density is set to 1 kg/L; it does not enter this mass-based
calculation.

The importer fits smooth, non-decreasing cubic splines to cumulative carbon and
base feeds. Both endpoints are preserved, keeping the total added feed
unchanged. Hybrax evaluates their derivatives as feed rates. Smoothing reduces
noise and abrupt rate changes that an adaptive ODE solver can miss. Temperature
uses a cubic interpolating spline with `s=0`, matching OptFed's preprocessing.
The importer fits complete traces first, then keeps the curve from induction
through the last concentration sample. The retained curve is unchanged by this
trimming.

````

```{code-cell} ipython3
:tags: [remove-input]

collection = load_process_collection(WORK / "data.json")
summary = pd.DataFrame([
    {
        "Process": name,
        "Measurements": len(
            process.reactor_medium.components["biomass"].concentration.times
        ),
        "Duration [h]": round(process.time_axis.end, 2),
        "Max glycerol [g/kg]": float(
            process.reactor_medium.components["glycerol"].concentration.values.max()
        ),
    }
    for name, process in collection.processes.items()
])
display(summary)
print(f"{len(summary)} processes, {summary.Measurements.sum()} measurement rows")
```

````{dropdown} Full importer code
```{literalinclude} ../../../examples/gallery_optfed/import_data.py
:language: python
```
````

## Define the model

After importing the data into the Hybrax format, we can define the model. Hybrax
handles the transport terms of the ODE (feed, dilution, sampling) itself and we
can focus on the biological dynamics. The importer script defined their backbone
with a `ReactionOde` in each process:

```{literalinclude} ../../../examples/gallery_optfed/import_data.py
:language: python
:start-at:             reaction_ode=hxf.ReactionOde(
:end-at:             ),
:dedent: 12
```

`derivatives` declares the non-transport derivatives of the three states.
`algebraic` defines intermediate quantities for these expressions. Measured
biomass $X$ includes intracellular product $P$, so `active_biomass` is $X-P$.
Active biomass drives uptake and production, and the `biomass` derivative
contains both the biomass and the product formation rate.

`rates` declares the names `q_biomass`, `q_glycerol`, and `q_product`. Their
bounds are `(None, None)`, so no lower or upper rate bounds are specified here.
When setting up the model in `custom.py` later we have to define a `RateModule`
that supplies these rates. The `RateModule` can wrap a neural network (or many
other things), but here we use the OptFed mechanistic model.

The `RateModule` holds the trainable `raw_parameters`. These parameters are
fields of the module, not symbols in `reaction_ode`. Each raw parameter is
unconstrained, and a sigmoid maps it into its physical bounds. For every
parameter except `equilibrium_temperature`, the sigmoid output interpolates
between the logarithms of the bounds (because the bounds span many orders of
magnitudes). Equilibrium temperature interpolates linearly between 300 and 315
K:

```{literalinclude} ../../../examples/gallery_optfed/custom.py
:language: python
:start-at: BOUNDS = {
:end-before: NEUTRAL_INITIAL = {
```

The module combines these parameters with the current states and temperature.
For example, `uptake_max` sets the uptake ceiling, and glycerol saturation and
inhibition reduce it to `uptake`. The module then allocates uptake to
maintenance, growth, and product formation, and converts the glycerol
allocations into biomass and product formation rates. It returns
`[q_biomass, uptake, q_product]` in the order of `reaction_ode.rates`, so
`q_glycerol` receives `uptake`.

Hybrax solves the ODE in scaled units: each quantity is divided by a
characteristic magnitude so that values in the solve are roughly 1 (see
[Scaling](../train/scaling.md)). Names starting with `SCL_` hold scaled values.
The module therefore receives scaled inputs and converts them back to physical
units with its `unscale_*` helpers:

- `unscale_modeled_RMCs` for the reactor medium components (RMCs): biomass,
  glycerol, and product.
- `unscale_controlled_PVs` for the controlled process variables (PVs): here
  only temperature.
- `unscale_latent` for the generation count.

It computes the rates in physical units and converts them back to scaled units
with `scale_modeled_ReactionOde_rates`. In this example all reaction-rate scales are
1, so this last step does not change the values, but it keeps the module
correct if those scales change.

````{dropdown} OptFed model equations

The reduced OptFed model uses active biomass $A=X-P$ and the following
specific glycerol allocation rates:

$$
\begin{aligned}
g &= g_{\max}\frac{G}{K_G+G}\frac{1}{1+G/K_I}, \\
g_m &= \min\!\left[m_0(1+a_m g)(1+b_m P/X),\;g\right], \\
g_a &= g-g_m, \\
g_P &= \min\!\left[p_T(T)\frac{g_a}{K_p+g_a}
                  \frac{1}{1+n/K_n},\;g_a\right], \\
p_T(T) &= 3600\,p_0\frac{k_B T}{h}
\frac{\exp[-E_a/(RT)]}
{1+\exp\!\left[\frac{\Delta H}{R}
                  \left(\frac{1}{T_{\mathrm{eq}}}-\frac{1}{T}\right)\right]}.
\end{aligned}
$$

Here $g$ is uptake, $g_m$ is maintenance demand, and $g_P$ is glycerol allocated
to product formation. All three are positive consumption magnitudes in
$\mathrm{g_G/(g_A\,h)}$. The symbols correspond to the parameters in `BOUNDS`:

- $g_{\max}$ and $K_I$ are `uptake_max` and `glycerol_inhibition`.
- $m_0$, $a_m$, and $b_m$ are `maintenance`, `uptake_maintenance`, and
  `product_maintenance`.
- $p_0$, $K_p$, and $K_n$ are `production_max`,
  `production_half_saturation`, and `generation_inhibition`.
- $E_a$, $\Delta H$, and $T_{\mathrm{eq}}$ are `activation_energy`,
  `inactivation_enthalpy`, and `equilibrium_temperature`.

Our $g$, $g_m$, and $g_P$ correspond to the paper's $\gamma^\circ$,
$\gamma^\alpha$, and $\gamma^\pi$; $K_G$ and $K_I$ correspond to its
$K_m^\circ$ and $K_G^\circ$. Here $g_a$ denotes uptake remaining after maintenance.
Generation inhibition follows the authors' code and Eq. 4b, using $n/K_n$;
Eq. 15c instead prints $nK_n$ \[[1](#ref-optfed)\]. The allocation caps follow
the authors' ODE implementation.

$K_G=0.001\,\mathrm{g/kg}$ is fixed. Temperature $T$ is in Kelvin, $k_B$ is
Boltzmann's constant, $h$ is Planck's constant, and $R$ is the gas constant.
The factor 3600 converts seconds to hours. The reaction module supplies

$$
q_{\mathrm{biomass}}=0.627\,(g_a-g_P),\qquad
q_{\mathrm{glycerol}}=g,\qquad
q_{\mathrm{product}}=0.652\,g_P.
$$

For completeness, the concentration and reactor-mass balances between
withdrawals are

$$
\begin{aligned}
\dot X &= (q_{\mathrm{biomass}}+q_{\mathrm{product}})A-DX, \\
\dot G &= -gA+F_c c_f/M-DG, \\
\dot P &= q_{\mathrm{product}}A-DP, \\
\dot M &= F_c+F_b,\qquad D=(F_c+F_b)/M, \\
\dot n &= \frac{(q_{\mathrm{biomass}}+q_{\mathrm{product}})A}{X\ln 2}.
\end{aligned}
$$

$M$ is reactor mass in kg, $F_c$ and $F_b$ are carbon and base feed rates in
kg/h, and $c_f$ is glycerol concentration in the carbon feed in g/kg.
The latent generation count starts at $n=0$. A well-mixed withdrawal reduces
reactor mass and component amounts proportionally, leaving concentrations and
$n$ unchanged. Hybrax adds transport and withdrawal effects to the reaction
terms declared in `ReactionOde`.

````

The mechanistic rate calculation is implemented in this `RateModule`:

```{literalinclude} ../../../examples/gallery_optfed/custom.py
:language: python
:pyobject: OptFedReactionModule
```

Hybrax inserts these rates into `reaction_ode.derivatives`, where each specific
rate is multiplied by active biomass. The following sections explain the
uptake, maintenance, and production laws behind them.

### Allocate glycerol uptake

For every process, uptake follows
$g_{\max}G/(0.001+G)/(1+G/K_I)$, where $G$ is reactor glycerol in g/kg
and $K_I$ is `glycerol_inhibition` in g/kg.

```{literalinclude} ../../../examples/gallery_optfed/custom.py
:language: python
:start-at:         uptake =
:end-before:         maintenance =
:dedent: 8
```

### Allocate maintenance

Maintenance increases with uptake and the product-to-biomass ratio. It cannot
consume more glycerol than uptake supplies. Specific glycerol uptake and
maintenance have units g glycerol per g active biomass per hour.

```{literalinclude} ../../../examples/gallery_optfed/custom.py
:language: python
:start-at:         maintenance =
:end-before:         available =
:dedent: 8
```

### Allocate production and growth

Product formation uses glycerol left after maintenance. Its production ceiling
depends on temperature through the Eyring equation. The activation energy,
inactivation enthalpy, equilibrium temperature, and prefactor are trainable.

```{literalinclude} ../../../examples/gallery_optfed/custom.py
:language: python
:pyobject: _temperature_ceiling
```

Temperature is in Kelvin, and both energies are in J/mol. The factor `3600`
converts the Eyring rate from seconds to hours.
Uptake has two parameters, maintenance has three, and production has six,
for eleven trainable scalars in total.

Product formation saturates in the remaining uptake and decreases with
accumulated generations. Its glycerol demand is capped by available uptake.
The fixed yields convert the remaining glycerol allocations into active biomass
and product formation rates.

```{literalinclude} ../../../examples/gallery_optfed/custom.py
:language: python
:start-at:         available =
:end-before:         return ReactionOutputs(
:dedent: 8
```

If the production ceiling starts above available uptake, the cap clips
it and its parameters receive zero gradient. The small initial prefactor
of `1e-9` avoids this at initialization.

The generation count is a single latent state initialized at zero. Its derivative
is $\mu/\ln 2$, where
$\mu=(q_{\mathrm{biomass}}+q_{\mathrm{product}})(X-P)/X$ is the biological
specific growth rate of total biomass. Integrating growth keeps this count independent
of dilution and the volume removed during sampling. It has no extra fitted
parameters.

The module declares this latent state in its constructor. The length of
`SCALE_latent` sets the number of latent states, and a scale of 1 keeps the
count in generations, so the derivative can be returned without rescaling.
Hybrax rejects models with latent states unless `train.allow_stateful_models`
is `true` in `train-config.json`.

```{literalinclude} ../../../examples/gallery_optfed/custom.py
:language: python
:start-at:     def __init__(self, **kwargs):
:end-at:         super().__init__(SCALE_latent=jnp.ones(1), **kwargs)
:dedent: 4
```

```{literalinclude} ../../../examples/gallery_optfed/custom.py
:language: python
:start-at:             SCL_latent_derivative=
:end-at:             SCL_latent_derivative=
:dedent: 12
```

### Build the reaction module

Hybrax creates the module by calling the `build_reaction_module` hook at
training setup, after `estimate_all_scales`. The hook receives keyword
arguments: the training collection, the run config, the target and process
names, a random seed, and the scales as `SCALE_*` keywords. A mechanistic model
needs no random initialization, so this hook discards the arguments it does not
use. It looks up the positions of biomass, glycerol, product, and temperature
in the state and control vectors, converts the starting values in
`NEUTRAL_INITIAL` to raw parameters, and passes the scales on to the module.
The module holds the scales for the rest of the run.

```{literalinclude} ../../../examples/gallery_optfed/custom.py
:language: python
:pyobject: build_reaction_module
```

`raw_parameters` is declared with `trainable_field()`, so the optimizer updates
it. Array fields without this tag stay frozen. The index fields are static
integers rather than arrays, so they are not trained either.

The complete implementation, including the other hooks, is shown below.

````{dropdown} Complete custom.py
```{literalinclude} ../../../examples/gallery_optfed/custom.py
:language: python
```
````

## Prepare and fit concentrations

The scales come from the `estimate_all_scales` hook in `custom.py`. Without
this hook every scale is 1 and Hybrax integrates the physical values directly,
without a warning. The returned `EstimatedScales` has one field per kind of
quantity. Fields for quantities this model does not have are empty arrays
(`jnp.zeros(0)`).

The state scale is each species' largest observed concentration across all
processes. Because the default loss compares scaled concentrations, state
scaling also weights the loss, so glycerol does not dominate simply because
its concentrations are larger than product concentrations.

```{literalinclude} ../../../examples/gallery_optfed/custom.py
:language: python
:start-at:         SCALE_modeled_RMCs=jnp.asarray(
:end-before:         SCALE_V_in_cumulative=
:dedent: 8
```

Control traces, such as feed rates and temperature, are scaled by their largest
absolute value, with a floor of `0.01`. These maxima come from the logged
readings, not from the fitted splines: feed-rate scales use finite differences
of the raw cumulative feed readings. Temperature is therefore divided by its
maximum in Kelvin. The reaction-rate scales are fixed at 1.

Use Adam for 200 epochs, with gradient clipping configured at `0.05` and linear
learning-rate decay from `0.05` to `0.0001`, supplied by the
`build_learning_rate` hook in `custom.py`. The default loss compares
predicted concentrations with the supplied measurements. No rate targets are
constructed.
Omitting `batch_size` uses all twelve processes in each update, so
200 epochs give 200 updates. Save a checkpoint every 10 epochs.
Training and evaluation use Kvaerno5 with `rtol=1e-7` and `atol=1e-9`.
The `build_solver` hook in `custom.py` selects this solver:

```{literalinclude} ../../../examples/gallery_optfed/custom.py
:language: python
:pyobject: build_solver
```

The model needs an implicit solver such as Kvaerno5 because the glycerol balance
is stiff. With the half-saturation constant fixed at $K_G=0.001\,\mathrm{g/kg}$,
uptake drops from its maximum to zero within a tiny glycerol range, so near
depletion glycerol changes on time scales far shorter than the process. An
explicit solver would need very small steps there.
Before training, Hybrax checks whether each continuous feed integrates to its
known cumulative increment within 0.1%. The smoothed feeds pass this preflight
with these solver settings. This diagnostic does not guarantee coupled-model
accuracy, so we also check reactor mass against the feed and withdrawal balance.
Allow several minutes for the full fit on CPU. The training command streams
progress to the terminal.

```{literalinclude} ../../../examples/gallery_optfed/train-config.json
:language: json
```

A selected pretrained checkpoint is bundled in `examples/gallery_optfed/run/`.
To simulate it immediately, skip to the forward command in
[Run the example locally](#run-the-example-locally).

To refit the model, switch to the example directory, prepare the data, and train.
The `--overwrite` flag replaces the bundled run with the new fit:

```bash
cd examples/gallery_optfed
hybrax prepare --config prepare-config.json --output-dir prepared
hybrax train --config train-config.json --overwrite
```

```{code-cell} ipython3
:tags: [remove-input]

hxt_cli("prepare", "--config", "prepare-config.json",
        "--output-dir", "prepared", "--overwrite")
output = hxt_cli("train", "--config", "train-config.json", "--overwrite")
print(next(line for line in output.splitlines() if "training complete" in line))
print("Run directory: docs/source/_data/out/runs/gallery_optfed/run/")
```

Evaluate each saved checkpoint on the measured concentrations and select the
one with the lowest combined NRMSE: the root mean squared error over all
measurements, with each species divided by its largest observed concentration
so that the three species are on a comparable scale. This is selection on
training data, not validation on held-out processes. All following plots and
parameter values use that selected checkpoint. `run/predictions.csv` contains
the final epoch's predictions; `run/selected-predictions.csv` contains the
selected model's curves.

The output below prints the ordinary R² for each species and for the
product-to-biomass ratio P/X, computed on raw concentrations.

````{dropdown} Score definitions

- **Combined NRMSE** is the root mean squared normalized error across all 249
  concentration values in the 83 retained measurement rows, including initial
  conditions. Each species is divided by its maximum observed concentration
  across the twelve processes.
- **Total R²** uses the same scaling and centers each species on its own mean.
- **P/X NRMSE** divides the RMSE of the ratio by its maximum observed value;
  **P/X R²** is computed the same way as total R², on the ratio alone.
- **Per-species R²**, printed below, uses raw concentrations and each
  species on its own.

The comparison table in [Compare with OptFed](#compare-with-optfed)
uses the pooled, scaled scores.

````

```{code-cell} ipython3
:tags: [remove-input]

checkpoint = select_checkpoint(WORK / "run", collection)
print(f"Selected checkpoint: {checkpoint.name}")
predictions = pd.read_csv(WORK / "run/selected-predictions.csv")
original = pd.read_csv(WORK / "original_predictions.csv")
points = sampling_predictions(collection, predictions)
for name, value in scores(points).items():
    print(f"{name:10s} R2 = {value:.4f}")
model, _ = model_load(checkpoint)
```

The table below shows the fitted values and their bounds. In the units,
$G$ denotes glycerol, $A$ active biomass, $X$ total biomass, and $P$ product.

```{code-cell} ipython3
:tags: [remove-input]

units = {
    "uptake_max": "g_G/(g_A h)",
    "glycerol_inhibition": "g/kg",
    "maintenance": "g_G/(g_A h)",
    "uptake_maintenance": "g_A h/g_G",
    "product_maintenance": "g_X/g_P",
    "production_max": "g_G/g_A",
    "production_half_saturation": "g_G/(g_A h)",
    "generation_inhibition": "generations",
    "activation_energy": "J/mol",
    "inactivation_enthalpy": "J/mol",
    "equilibrium_temperature": "K",
}
rows = [
    {
        "Parameter": name, "Unit": units[name], "Lower": BOUNDS[name][0],
        "Fitted": float(value), "Upper": BOUNDS[name][1],
    }
    for name, value in model.reaction_module.parameters.items()
]
numbers = ["Lower", "Fitted", "Upper"]
display(
    pd.DataFrame(rows).style.set_uuid("optfed_parameters").format(
        "{:.3g}", subset=numbers
    )
)
```

```{code-cell} ipython3
:tags: [remove-input]

parameter_count = sum(
    value.size for value in jax.tree.leaves(model.reaction_module.raw_parameters)
)
assert parameter_count == 11
metrics = pd.read_csv(WORK / "run/metrics.csv")
assert np.isfinite(metrics.grad_norm).all()
assert metrics.n_failed_samples.sum() == 0
mass_errors = []
for name, process in collection.processes.items():
    predicted = predictions[predictions.process == name]
    times = predicted.t.to_numpy()
    expected_mass = np.full_like(times, process.volume.initial_volume)
    for feed_name in ("carbon_feed", "base_feed"):
        feed = process.volume.volume_changes[feed_name].values
        expected_mass += np.asarray(feed.evaluate_many(times))
    draws = process.volume.volume_changes["sampling"].values
    expected_mass += (
        (times[:, None] >= np.asarray(draws.times)[None, :])
        * np.asarray(draws.values)[None, :]
    ).sum(axis=1)
    np.testing.assert_allclose(
        predicted.V_real, expected_mass, rtol=1e-3, atol=1e-5
    )
    mass_errors.extend(np.abs(predicted.V_real - expected_mass))
print(f"Maximum reactor-mass error at exported times: {max(mass_errors):.6g} kg")
```

```{code-cell} ipython3
:tags: [remove-cell]

reference = pd.read_csv(WORK / "original_sampling_predictions.csv").rename(
    columns={"G_est": "predicted_glycerol", "X_est": "predicted_biomass",
             "P_est": "predicted_product"}
)
reference = points.drop(columns=[
    column for column in points if column.startswith("predicted_")
]).merge(reference, on=["process", "t"], validate="one_to_one")
assert len(reference) == len(points)
reference["predicted_P_X"] = (
    reference.predicted_product / reference.predicted_biomass
)


comparison = []
for name, frame in (("Original OptFed", reference), ("Hybrax", points)):
    total_r2, total_nrmse = normalized_scores(
        frame, ["glycerol", "biomass", "product"]
    )
    ratio_r2, ratio_nrmse = normalized_scores(frame, ["P_X"])
    comparison.append({
        "Model": name, "Total R²": total_r2, "Total NRMSE": total_nrmse,
        "P/X R²": ratio_r2, "P/X NRMSE": ratio_nrmse,
    })
comparison = pd.DataFrame(comparison).set_index("Model")
```

## Compare with OptFed

Fitting concentrations directly, Hybrax reproduces the measurements about as
well as the published OptFed model. The table scores both models on the same
measurements. These are the processes used for training, so the scores measure
goodness of fit rather than performance on unseen processes.

```{code-cell} ipython3
:tags: [remove-input]

display(comparison.style.set_uuid("optfed_comparison").format("{:.3g}"))
```

OptFed minimised error in derived rates, while Hybrax fits concentrations
directly and its checkpoint is selected on pooled concentration NRMSE, so this
comparison favours the concentration-fitting objective. Similar pooled errors
can also hide differences between processes, so the next section compares the
individual trajectories. A good fit also does not establish unique parameters:
the Eyring prefactor and activation energy, for example, can compensate for
each other over a narrow temperature range.

````{dropdown} Comparison details

- The comparison uses all 83 retained measurement rows, including the initial
  conditions. The scores use the same per-species scaling as the Hybrax
  training loss.
- The training loss weights processes equally; pooled NRMSE weights
  measurement rows equally.
- The two fits also differ in optimizer, feed splines, and sampling treatment.
  Differences in their parameters or trajectories therefore do not establish
  weak identifiability on their own.
- These are measurement-based scores, not quoted cross-validation scores from
  the paper. The reference scores use measurement-time predictions exported
  from archived OptFed analysis outputs, rather than interpolation from the
  plotted curves. These archives are not included in the pinned GitHub commit;
  the bundled CSV makes the comparison reproducible locally.

````

## Inspect the fitted trajectories

The following plot compares the fitted Hybrax model, the original OptFed model,
and the experimental measurements. Original OptFed curves are semi-transparent
solid lines; Hybrax curves are dashed. The original curves come from the authors'
global reduced-model fit to derived rates, published in the
[OptFed repository](https://github.com/gschloegel/OptFed/tree/7811f7b9ef76d624f53b852c03d06cb1b5cc96a3).
The bundled {download}`reference curves
<../../../examples/gallery_optfed/original_predictions.csv>` and
{download}`measurement-time predictions
<../../../examples/gallery_optfed/original_sampling_predictions.csv>` let this
comparison run without downloading or refitting OptFed.

```{code-cell} ipython3
:tags: [remove-cell]

fig, axes = plt.subplots(12, 3, figsize=(12, 24), layout="constrained")
for row, (name, observed) in zip(axes, points.groupby("process", sort=False)):
    trajectory = predictions[(predictions.process == name) & (
        predictions.t <= observed.t.max()
    )]
    reference = original[(original.process == name) & (
        original.t <= observed.t.max()
    )]
    for ax, species, field in zip(
        row, ("glycerol", "biomass", "product"), ("G", "X", "P")
    ):
        ax.plot(reference.t, reference[field], color="tab:blue", alpha=0.45,
                label="Original OptFed")
        ax.plot(trajectory.t, trajectory[f"c_{species}"], color="tab:orange",
                linestyle="--", label="Hybrax")
        ax.scatter(observed.t, observed[species], color="black", s=15,
                   label="Measurements")
        ax.set(title=f"{name}: {species}", ylabel="Concentration [g/kg]")
        if species == "glycerol":
            # Keep solver roundoff around zero from filling the panel.
            upper = max(observed[species].max(), trajectory.c_glycerol.max(),
                        reference.G.max(), 0.1)
            ax.set_ylim(0, 1.05 * upper)
for ax in axes[-1]:
    ax.set_xlabel("Time [h]")
axes[0, 0].legend()
fig.suptitle(
    "Glycerol, biomass, product: pooled measurement scores\n"
    + "\n".join(
        f"{name}: R²={row['Total R²']:.3f}, NRMSE={row['Total NRMSE']:.4f}"
        for name, row in comparison.iterrows()
    ),
    fontsize=11,
)
fig.savefig(WORK / "run/concentration_fit.png", dpi=140)
plt.close(fig)
```

```{code-cell} ipython3
:tags: [remove-input]

display(Image(filename=str(WORK / "run/concentration_fit.png")))
```

## Run the example locally

The prepare and train commands above write to `prepared/` and `run/` inside
`examples/gallery_optfed/`. To generate trajectories and plots, run from that
same directory:

```bash
hybrax forward --config forward-config.json --output-dir run/forward
```

The bundled forward configuration loads `run`. To evaluate a specific checkpoint,
set its `models` entry to that checkpoint directory, for example
`["run/checkpoints/step_00200"]`. Use the checkpoint selected by concentration
NRMSE to reproduce the comparison above.

The optional `run.py` script automates preparation, training, checkpoint
selection, and forward simulation:

```bash
python run.py
```

Fitting the bundled dataset does not need network access. To regenerate it from
the four published CSVs, run `python import_data.py` in the example directory.
This page's executed run is in
`docs/source/_data/out/runs/gallery_optfed/run/`.

See [Mechanistic models](mechanistic_rates.md) for a smaller rate law and
[The reaction module](../train/reaction_module.md) for the `RateModule` interface.

(ref-optfed)=
## References

1. Schlögel, G., Lück, R., Kittler, S., Spadiut, O.,
   Kopp, J., Zanghellini, J., & Gotsmy, M. (2024). Optimizing bioprocessing
   efficiency with OptFed: Dynamic nonlinear modeling improves
   product-to-biomass yield. *Computational and Structural Biotechnology
   Journal*, 23, 3651-3661.
   [doi:10.1016/j.csbj.2024.09.024](https://doi.org/10.1016/j.csbj.2024.09.024).

(ref-kittler)=
2. Kittler, S., Ebner, J., Besleaga, M., Larsbrink, J., Darnhofer, B.,
   Birner-Grünberger, R., Schobesberger, S., Akhgar, C. K., Schwaighofer, A.,
   Lendl, B., & Spadiut, O. (2022). Recombinant protein L: Production,
   purification and characterization of a universal binding ligand.
   *Journal of Biotechnology*, 359, 108-115.
   [doi:10.1016/j.jbiotec.2022.10.002](https://doi.org/10.1016/j.jbiotec.2022.10.002).
