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
# 1. Import your Data


> **In one sentence.** Turn a CSV of experimental measurements into a bp-format
> `BioProcessCollection` you can save, share and train on.


`Hybrax` only requires you to import your training data once into its JSON format. Everything downstream 
is derived from your data entered in this step.
Here, we build a simple **batch** run with three measured species and no volume
changes. However, the `Hybrax` data format is designed to accomodate a diverse set of bioprocess conditions, for example
* [feeds, boluses and sampling volume changes](../gallery/fed_batch.md),
<!-- UNLOCK -->
* [chemical decay rates](../gallery/glutamine_decay.md), and
* [modeled process variables](../gallery/<placeholder! please fix once the glu lands>).


## 1.1 Example data

```{code-cell} ipython3
:tags: [remove-input]

from pathlib import Path
RAW = Path("../_data/out/demo_batch/raw/offline.csv").resolve()
print("".join(RAW.read_text().splitlines(keepends=True)[:6]), end="")
print("...")
```

One row per sample, one column per assay. Three runs share the file; we build `run_1`.

```{code-cell} ipython3
import pandas as pd

df = pd.read_csv(RAW)
run_1 = (
    df[df["run"] == "run_1"]
    .sort_values("time_h")
    .rename(columns={"time_h": "time", "biomass_gL": "biomass",
                      "glucose_gL": "glucose", "product_gL": "product"})
)

print(len(run_1), "samples from", run_1["time"].iloc[0], "to", run_1["time"].iloc[-1], "h")
```

## 1.2 Measurements become `TimeSeries`

Every time-varying quantity in bp-format is a `TimeSeries` which pairs `times` and `values`.

```{code-cell} ipython3
import numpy as np
import bp_format as bp

biomass = bp.TimeSeries(
    times=run_1["time"].to_numpy(),
    values=run_1["biomass"].to_numpy(),
)
biomass
```

<!-- LOCK -->
:::{admonition} Notes
:class: note
- Each measured species is its own `TimeSeries` naturally enabling irregular sampling.
- The `times` vector must be **strictly increasing**.
- For a quantity that genuinely does not change, use `bp.StaticVariable(value)` instead.
:::


## 1.3 Concentrations become `ReactorMediumComponents`
<!-- UNLOCK -->

The **reactor medium** is what is in the vessel. Each component carries its own unit,
because bp-format never converts units for you: it only checks that you were
consistent.

```{code-cell} ipython3
components = {
    name: bp.ReactorMediumComponent(
        name=name,
        unit="g/L",
        concentration=bp.TimeSeries(
            times=run_1["time"].to_numpy(),
            values=run_1[name].to_numpy(),
        ),
        bounds=(0.0, None),      # metadata: a concentration cannot be negative
    )
    for name in ("biomass", "glucose", "product")
}

reactor_medium = bp.ReactorMedium(
    name="defined_medium",
    density=1.0,
    density_unit="kg/L",
    components=components,
)
list(reactor_medium.components)
```

:::{admonition} Notes
:class: note
- The `bounds` are metadata, not constraints. Nothing in `Hybrax` enforces bounds per default. They are recorded so downstream consumers (bp-train's loss module, for instance) can build soft penalties from them.
- The units are explicitly required. This avoids ambiguity and enables validation downstream.
:::

## 1.4 The clock and the vessel

```{code-cell} ipython3
time_axis = bp.TimeAxis(
    unit="h",
    start=0.0,
    end=14.0,
    time_reference="inoculation",   # what t = 0 means
)

volume = bp.Volume(initial_volume=1.0, unit="L")   # batch: nothing moves volume
```

`time_reference` is what makes runs from different sources alignable: "t = 0 is
inoculation" and "t = 0 is first feed" are different clocks, and later you will want to
know which one you had.

<!-- LOCK -->
`Volume` with no `volume_changes` is a true batch. Note what this claims: no feeds, no
boluses, and no sampling volume. That is a real assertion about the simulated experiment. If
your real world data contains volume changes, see
[Volume, feeds and events](../format/volume_feeds_events.md).
<!-- UNLOCK -->

## 1.5 Assemble the ``BioProcess``

```{code-cell} ipython3
process = bp.BioProcess(
    metadata=bp.BioProcessMetadata(
        name="run_1",
        process_type="batch",
        notes="Simulated E. coli batch culture on glucose.",
    ),
    time_axis=time_axis,
    volume=volume,
    reactor_medium=reactor_medium,
)
```

That is the whole process. Notice what you did **not** write: any dynamics. bp-format
generated a default biological ODE for you:

```{code-cell} ipython3
print("rates      :", list(process.biological_ode.rates))
print("derivatives:", process.biological_ode.derivatives)
```

Each species gets a specific rate `q_<species>`, and its derivative is
`q_<species> * biomass`. Those rates are exactly what a model will later predict.

:::{admonition} This is why a component must be called `biomass`
:class: warning
Auto-generation makes every rate *specific* (per unit biomass) so it needs to know
which component the biomass is. Without one, constructing the `BioProcess` raises
immediately. If your data has no biomass, or you want different dynamics, write
`biological_ode` yourself: [The Bioprocess ODE](../format/bioprocess_ode.md).
:::

## 1.6 Collect and save

Setting `case_id`/`organism`/`citation` on a `BioProcessCollection` marks it as a full
case study, one publication or campaign. `case_id` is also the natural grouping for
cross-validation later. Leaving them unset (the default) is fine too: that is raw or
intermediate data with no case-study identity yet, the same container either way.

```{code-cell} ipython3
import contextlib
import io

collection = bp.BioProcessCollection(
    case_id="my_first_dataset",
    organism="Escherichia coli",
    citation="Simulated data: tutorial only.",
    processes={"run_1": process},
)

out = Path("../_data/out/runs/tutorial_01").resolve()
out.mkdir(parents=True, exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):
    bp.serialization.save_process_collection(collection, out / "data.json")
print(f"./{(out / 'data.json').relative_to(out.parents[4])}")
```

Note the import path: the save/load functions live on `bp.serialization`, not on the
package root.

## 1.7 Check the round trip

```{code-cell} ipython3
reloaded = bp.serialization.load_process_collection(out / "data.json")
run = reloaded.processes["run_1"]

print("runs        :", list(reloaded.processes))
print("components  :", list(run.reactor_medium.components))
print("first 3 t   :", np.asarray(run.reactor_medium.components["biomass"].concentration.times)[:3])
print("first 3 X   :", np.asarray(run.reactor_medium.components["biomass"].concentration.values)[:3])
```

## What you learned

- Every time-varying quantity is a `TimeSeries`; every constant is a `StaticVariable`.
- Components are grouped by **physical role**: reactor medium, volume, process variables.
- `bounds` and `time_reference` are metadata, and both matter later.
- bp-format writes the default biology for you, in terms of specific rates.

## What's next

Run the tutorial yourself at `./source/_data/out/runs/tutorial_01/`.

- **[Tutorial 2](02_look_at_it.md)**: check that the package understood your data the
  way you meant it.
- Measuring something that is not a concentration (pH, DO, off-gas)? That is a
  **process variable**: [The data model](../format/data_model.md).
- Fed-batch? [Volume, feeds and events](../format/volume_feeds_events.md), then
  [Gallery: fed-batch](../gallery/fed_batch.md).
