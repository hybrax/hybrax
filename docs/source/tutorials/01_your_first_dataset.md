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

# Tutorial 1: your first dataset

> **In one sentence.** Turn a CSV of offline measurements into a bp-format
> `CaseStudy` you can save, share and train on.
>
> **You need this if** you have your own data. **You can skip it if** someone already
> handed you a bp-format file.

This is the one genuinely manual step in the whole stack, and it is worth doing
carefully: everything downstream is derived from what you write here. We build the
simplest possible thing: a **batch** run with three measured species and no volume
changes at all. Feeds, boluses and sampling come later, in the
[gallery](../gallery/fed_batch.md).

## The data you have

```{code-cell} ipython3
:tags: [remove-input]

from pathlib import Path
RAW = Path("../_data/out/demo_batch/raw/offline.csv").resolve()
print("".join(RAW.read_text().splitlines(keepends=True)[:6]), end="")
print("...")
```

One row per sample, one column per assay. Three runs share the file; we build `run_1`.

```{code-cell} ipython3
import csv
from collections import defaultdict

rows = defaultdict(list)
with RAW.open() as fh:
    for row in csv.DictReader(fh):
        if row["run"] == "run_1":
            rows["time"].append(float(row["time_h"]))
            rows["biomass"].append(float(row["biomass_gL"]))
            rows["glucose"].append(float(row["glucose_gL"]))
            rows["product"].append(float(row["product_gL"]))

print(len(rows["time"]), "samples from", rows["time"][0], "to", rows["time"][-1], "h")
```

## Step 1: measurements become `TimeSeries`

Every time-varying quantity in bp-format is a `TimeSeries`: paired `times` and `values`.

```{code-cell} ipython3
import numpy as np
import bp_format as bp

biomass = bp.TimeSeries(
    times=np.asarray(rows["time"]),
    values=np.asarray(rows["biomass"]),
)
biomass
```

Two rules worth knowing now:

- `times` must be **strictly increasing**.
- A `TimeSeries` needs discrete samples, or a fitted spline, or both, but not neither.

For a quantity that genuinely does not change, use `bp.StaticVariable(value)` instead.

## Step 2: species become reactor medium components

The **reactor medium** is what is in the vessel. Each component carries its own unit,
because bp-format never converts units for you: it only checks that you were
consistent.

```{code-cell} ipython3
components = {
    name: bp.ReactorMediumComponent(
        name=name,
        unit="g/L",
        concentration=bp.TimeSeries(
            times=np.asarray(rows["time"]),
            values=np.asarray(rows[name]),
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

:::{admonition} `bounds` are metadata, not constraints
:class: note
Nothing in bp-format or the ODE solver enforces bounds. They are recorded so downstream
consumers (bp-train's loss module, for instance) can build soft penalties from them.
:::

## Step 3: the clock and the vessel

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

`Volume` with no `volume_changes` is a true batch. Note what this claims: no feeds, no
boluses, **and no sampling volume**. That is a real assertion about the experiment. If
your samples removed a non-negligible volume, say so: see
[Volume, feeds and events](../format/volume_feeds_events.md).

## Step 4: assemble the run

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

That is the whole run. Notice what you did **not** write: any dynamics. bp-format
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

## Step 5: group and save

A `CaseStudy` is one publication or campaign. Its `case_id` is also the natural grouping
for cross-validation later.

```{code-cell} ipython3
case_study = bp.CaseStudy(
    case_id="my_first_dataset",
    organism="Escherichia coli",
    citation="Simulated data: tutorial only.",
    processes={"run_1": process},
)

out = Path("../_data/out/runs/tutorial_01").resolve()
out.mkdir(parents=True, exist_ok=True)
bp.serialization.save_case_study(case_study, out / "data.json")
```

Note the import path: the save/load functions live on `bp.serialization`, not on the
package root.

## Check the round trip

```{code-cell} ipython3
reloaded = bp.serialization.load_case_study(out / "data.json")
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

- **[Tutorial 2](02_look_at_it.md)**: check that the package understood your data the
  way you meant it.
- Measuring something that is not a concentration (pH, DO, off-gas)? That is a
  **process variable**: [The data model](../format/data_model.md).
- Fed-batch? [Volume, feeds and events](../format/volume_feeds_events.md), then
  [Gallery: fed-batch](../gallery/fed_batch.md).
