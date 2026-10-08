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

Hybrax is designed primarily for hybrid models that combine mechanistic balances
with learned rates. It also fits fully mechanistic models. Here we fit the
11-parameter reduced OptFed model directly to glycerol, biomass, and product
concentrations, with no neural network and no intermediate rate estimation.

The kinetic structure follows Schlögl et al. [1](#ref-optfed). Uptake supplies
maintenance, product formation, and growth. Temperature affects product
formation through an Eyring rate law, and accumulated generations inhibit
production. The yields and glycerol saturation constant remain fixed.

This example uses the twelve experimental fed-batch processes published with
OptFed [2](#ref-optfed-data), originating from Kittler et al. [3](#ref-kittler).
Nine are substrate-limited:
when glycerol remains near zero, its mass balance determines uptake from the
feed supply. OptFed's authors likewise inferred uptake from the glycerol balance
in their rate-fitting stage [1](#ref-optfed). We use that feed-implied uptake for
these processes and fit kinetic uptake for the other three. The model fits all
twelve processes jointly with one shared set of eleven parameters.

```{code-cell} ipython3
:tags: [remove-cell]

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
from run import sampling_predictions, scores
from custom import BOUNDS
```

## Load the published data

The bundled {download}`data.json <../../../examples/gallery_optfed/data.json>`
is a Hybrax `BioProcessCollection`. Training and docs
builds use this local file without downloading data. To regenerate it from the
four CSVs at a pinned OptFed commit, run:

```bash
uv run python examples/gallery_optfed/import_data.py
```

The {download}`importer <../../../examples/gallery_optfed/import_data.py>` maps
the published data as follows:

- `sampling_points.csv` supplies initial reactor mass and raw glycerol, total
  biomass, and product concentrations. Time zero is induction. As in OptFed's
  fitting script, row 40
  of `DoE1_R3` is excluded as an erroneous measurement. Its withdrawal event
  remains in the mass balance.
- `volume_flow_rates.csv` supplies cumulative carbon and base feeds. Each feed
  starts at zero added mass at induction. Downward
  reactor-mass jumps estimate individual withdrawals, including extra draws
  between concentration measurements.
- `temperatures.csv` supplies the measured temperature control in Kelvin.
- `c_feed.csv` supplies glycerol concentration in the carbon feed. The importer
  uses OptFed's density correction to convert this value from g/L to g/kg.

OptFed uses a mass-based balance. Here `volume` stores reactor mass in kg,
feed amounts are in kg, and concentrations use g/kg. The reactor-density field
is set to 1 kg/L; it does not enter this mass-based calculation. Feed and
temperature traces use Hybrax's default piecewise-linear interpolation. The
concentration targets remain the actual measurements, without spline smoothing.

```{code-cell} ipython3
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
        "Uptake": "Feed-implied" if process.process_variables[
            "supply_limited_uptake"
        ].values.values[0] else "Kinetic",
    }
    for name, process in collection.processes.items()
])
display(summary)
print(f"{len(summary)} processes, {summary.Measurements.sum()} measurement rows")
```

## Define the rate laws

Measured biomass \(X\) includes intracellular product \(P\). The model uses
\(X-P\) as active biomass, which drives uptake and production. Both concentrations
are in g/kg. Hybrax adds the feed and sampling balances to these reaction terms:

```{literalinclude} ../../../examples/gallery_optfed/import_data.py
:language: python
:lines: 129-140
:dedent: 12
```

Write the mechanistic rates in `custom.py`. Parameters are fitted in unconstrained
space and mapped into physical bounds with a sigmoid. Positive parameters use
logarithmic bounds; the equilibrium temperature uses linear bounds.

```{literalinclude} ../../../examples/gallery_optfed/custom.py
:language: python
:start-at: BOUNDS = {
:end-before: NEUTRAL_INITIAL = {
```

### Allocate glycerol uptake

The per-process flag `supply_limited_uptake` selects the uptake rule. It is stored
as a controlled process variable so Hybrax passes it to the reaction module.
The summary above shows the reason for the split: glycerol accumulates in
`DoE1_R1`, `DoE2_R1`, and `DoE3_R2`, whereas it stays near zero in the other nine.

For substrate-limited processes, the carbon-feed rate and composition arrive
through Hybrax's controls interface. Their product gives glycerol supplied per
hour. Dividing by reactor mass and active biomass gives specific uptake:

\[
g = \frac{F_C G_{\mathrm{feed}}}{V(X-P)}.
\]

Here \(F_C\) is carbon-feed mass flow in kg/h, \(G_{\mathrm{feed}}\) is its
glycerol concentration in g/kg, and \(V\) is reactor mass in kg.
For the other processes, uptake follows
\(g_{\max}G/(0.001+G)/(1+G/K_I)\), where \(G\) is reactor glycerol in g/kg
and \(K_I\) is `glycerol_inhibition` in g/kg.

```{literalinclude} ../../../examples/gallery_optfed/custom.py
:language: python
:start-at:         kinetic_uptake =
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
for eleven trainable scalars in total. The initial production prefactor is small
enough to keep the rate below its substrate cap, where its parameters have
nonzero gradients.

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

The generation count is a single latent state initialized at zero. Its derivative
is \(\mu/\ln 2\), where
\(\mu=(q_{\mathrm{biomass}}+q_{\mathrm{product}})(X-P)/X\) is the biological
specific growth rate of total biomass. Integrating growth keeps this count independent
of dilution and the volume removed during sampling. It has no extra fitted
parameters. Set `allow_stateful_models` to enable this latent state.

## Prepare and fit concentrations

State and control scales come from the training data. State scaling also weights
the default concentration loss, so glycerol does not dominate simply because
its concentrations are larger than product concentrations.

```{literalinclude} ../../../examples/gallery_optfed/custom.py
:language: python
:start-at:         SCALE_modeled_RMCs=jnp.asarray(
:end-before:         SCALE_V_in_cumulative=
:dedent: 8
```

Use Adam for 500 epochs, with gradient clipping configured at `0.05` and a
warmup followed by cosine learning-rate decay, supplied by the
`build_learning_rate` hook in `custom.py`. The default loss compares
predicted concentrations with the supplied measurements. No rate targets are
constructed.
Each update uses one process in a fixed order. An epoch visits all twelve, so
500 epochs give 6,000 updates. Solver tolerances are `rtol=1e-5` and `atol=1e-7`.
Allow tens of minutes for the full fit on CPU. The standalone runner below
streams training progress; this notebook prints the final status.

```{literalinclude} ../../../examples/gallery_optfed/train-config.json
:language: json
```

```{code-cell} ipython3
hxt_cli("prepare", "--config", "prepare-config.json",
        "--output-dir", "prepared", "--overwrite")
output = hxt_cli("train", "--config", "train-config.json", "--overwrite")
print(next(line for line in output.splitlines() if "training complete" in line))
print("Run directory: docs/source/_data/out/runs/gallery_optfed/run/")
```

```{code-cell} ipython3
predictions = pd.read_csv(WORK / "run/predictions.csv")
original = pd.read_csv(WORK / "original_predictions.csv")
points = sampling_predictions(collection, predictions)
for name, value in scores(points).items():
    print(f"{name:10s} R2 = {value:.4f}")
model, _ = model_load(WORK / "run")
```

The table below shows the fitted values and their bounds. In the units,
\(G\) denotes glycerol, \(A\) active biomass, and \(P\) product.

```{code-cell} ipython3
units = {
    "uptake_max": ["g_G/(g_A h)"],
    "glycerol_inhibition": ["g/kg"],
    "maintenance": ["g_G/(g_A h)"],
    "uptake_maintenance": ["g_A h/g_G"],
    "product_maintenance": ["g biomass/g_P"],
    "production_max": ["g_G/g_A"],
    "production_half_saturation": ["g_G/(g_A h)"],
    "generation_inhibition": ["generations"],
    "temperature": ["J/mol", "J/mol", "K"],
}
temperature_names = [
    "activation_energy", "inactivation_enthalpy", "equilibrium_temperature"
]
rows = []
for name, values in model.reaction_module.parameters.items():
    for i, (value, bounds, unit) in enumerate(zip(values, BOUNDS[name], units[name])):
        rows.append({
            "Parameter": temperature_names[i] if name == "temperature" else name,
            "Unit": unit, "Lower": bounds[0],
            "Fitted": float(value), "Upper": bounds[1],
        })
display(pd.DataFrame(rows))
```

```{code-cell} ipython3
:tags: [remove-cell]

parameter_count = sum(
    value.size for value in jax.tree.leaves(model.reaction_module.raw_parameters)
)
assert parameter_count == 11
metrics = pd.read_csv(WORK / "run/metrics.csv")
assert np.isfinite(metrics.grad_norm).all()
assert metrics.n_failed_samples.sum() == 0
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


def normalized_scores(frame, fields):
    observed = frame[fields].to_numpy()
    predicted = frame[[f"predicted_{name}" for name in fields]].to_numpy()
    scales = np.abs(observed).max(axis=0)
    normalized = observed / scales
    residual = (predicted - observed) / scales
    rss = np.sum(residual ** 2)
    tss = np.sum((normalized - normalized.mean(axis=0)) ** 2)
    return 1 - rss / tss, np.sqrt(np.mean(residual ** 2))


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

## Inspect the fitted trajectories

The following plots compare the fitted Hybrax model, the original OptFed model,
and the experimental measurements. Original OptFed curves are semi-transparent
solid lines; Hybrax curves are dashed. The original curves come from the authors'
global reduced-model fit to derived rates, exported from their saved predictions.
The bundled {download}`reference curves
<../../../examples/gallery_optfed/original_predictions.csv>` and
{download}`measurement-time predictions
<../../../examples/gallery_optfed/original_sampling_predictions.csv>` let this
comparison run without downloading or refitting OptFed. Hybrax fits
concentrations directly.
The reported scores describe the same processes used for training, so they
measure goodness of fit rather than performance on unseen processes.

```{code-cell} ipython3
fig, axes = plt.subplots(4, 3, figsize=(12, 12), layout="constrained")
for ax, (name, observed) in zip(axes.flat, points.groupby("process", sort=False)):
    trajectory = predictions[predictions.process == name]
    reference = original[(original.process == name) & (
        original.t <= collection.processes[name].time_axis.end
    )]
    ax.plot(reference.t, reference.P / reference.X,
            color="tab:blue", alpha=0.45, label="Original OptFed")
    ax.plot(trajectory.t, trajectory.c_product / trajectory.c_biomass,
            color="tab:orange", linestyle="--", label="Hybrax")
    ax.scatter(observed.t, observed.P_X, color="black", s=20,
               label="Measurements")
    ax.set(title=name, xlabel="Time [h]", ylabel="Product / biomass [g/g]")
axes.flat[0].legend()
fig.suptitle(
    "Product / biomass: pooled measurement scores\n"
    + "\n".join(
        f"{name}: R²={row['P/X R²']:.3f}, NRMSE={row['P/X NRMSE']:.4f}"
        for name, row in comparison.iterrows()
    ),
    fontsize=11,
)
fig.savefig(WORK / "run/PX_fit.png", dpi=140)
plt.close(fig)
display(Image(filename=str(WORK / "run/PX_fit.png")))
```

Inspect the three fitted concentrations as well. A similar product-to-biomass
ratio can conceal different biomass and product predictions.

```{code-cell} ipython3
fig, axes = plt.subplots(12, 3, figsize=(12, 24), layout="constrained")
for row, (name, observed) in zip(axes, points.groupby("process", sort=False)):
    trajectory = predictions[predictions.process == name]
    reference = original[(original.process == name) & (
        original.t <= collection.processes[name].time_axis.end
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
display(Image(filename=str(WORK / "run/concentration_fit.png")))
```

## Compare fitting objectives

Direct concentration fitting of these experimental measurements can achieve
aggregate fit quality comparable to the original OptFed results, while yielding
different parameter estimates and trajectories.
Similar goodness of fit therefore does not guarantee recovery of the original
parameters. This suggests weak parameter identifiability, but does not establish
it: OptFed fitted derived rates, and some uptake assumptions also differ. This
comparison uses the same measured processes, with different fitting objectives.
The following comparison uses all 83 retained measurement rows, including the
initial conditions. Combined NRMSE is the root mean squared error across all
249 concentration values, after dividing each species by its maximum observed
concentration across the twelve processes. Total R² uses the same scaled
concentrations, with each species centered on its own mean. P/X NRMSE divides
ratio residuals by the maximum observed P/X. Both models use the same scales.
These metrics differ from the optimizer's training loss.

```{code-cell} ipython3
display(comparison)
```

Hybrax's combined NRMSE is about 6% higher: 0.0935 versus 0.0885. Its pooled
product-to-biomass \(R^2\) is 0.742 versus 0.734. These scores describe comparable
aggregate agreement with measurements, despite the differences in individual
trajectories. They are our measurement-based comparison, not quoted
cross-validation scores from the paper. The reference scores use the authors'
saved predictions at the exact measurement times, rather than interpolation
from the plotted curves.

In the nine feed-implied processes, uptake balances glycerol feed by construction.
Their predicted glycerol therefore only dilutes from its initial value. Glycerol
fit quality in these processes does not validate uptake kinetics. Only the three
kinetic processes inform `uptake_max` and `glycerol_inhibition`.
The Eyring prefactor and activation energy can also compensate for each other
over a narrow temperature range. A good fit does not establish unique parameters.

## Gotchas

- Temperature must be in Kelvin. The published temperature trace uses Kelvin.
- Feed-implied uptake assumes negligible glycerol accumulation. Set
  `supply_limited_uptake` only for processes where this assumption is appropriate.

## Run the example locally

From the repository root, run:

```bash
uv run python examples/gallery_optfed/run.py
```

The standalone run writes to `examples/gallery_optfed/run/`. This page's
executed run is in `docs/source/_data/out/runs/gallery_optfed/run/`.
Regenerate the bundled dataset with
`uv run python examples/gallery_optfed/import_data.py`. This step downloads the
four published CSVs; fitting the bundled dataset does not need network access.

See [Mechanistic models](mechanistic_rates.md) for a smaller rate law and
[The reaction module](../train/reaction_module.md) for the `RateModule` interface.

(ref-optfed)=
## References

1. Schlögl, G., Lück, R., Kittler, S., Spadiut, O.,
   Kopp, J., Zanghellini, J., & Gotsmy, M. (2024). Optimizing bioprocessing
   efficiency with OptFed: Dynamic nonlinear modeling improves
   product-to-biomass yield. *Computational and Structural Biotechnology
   Journal*, 23, 3651-3661.
   [doi:10.1016/j.csbj.2024.09.024](https://doi.org/10.1016/j.csbj.2024.09.024)

(ref-optfed-data)=
2. [OptFed experimental CSVs and analysis code](https://github.com/gschloegel/OptFed/tree/7811f7b9ef76d624f53b852c03d06cb1b5cc96a3).

(ref-kittler)=
3. Kittler et al. (2022). Recombinant protein L: production, purification and
   characterization of a universal binding ligand. *Journal of Biotechnology*,
   359, 108-115.
   [doi:10.1016/j.jbiotec.2022.10.002](https://doi.org/10.1016/j.jbiotec.2022.10.002)
