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

# The data model

> **In one sentence.** Which object holds which measurement, and why the split is where
> it is.
>
> **You need this if** you are building a dataset and unsure where something goes.
> **You can skip it if** your data is already in bp-format.

Exhaustive field lists live in the [API reference](../autoapi/bp_format/dataclasses/index).
This page is about the *decisions* — the ones that are hard to reverse later.

## The one question that decides everything

**What physical role does this measurement play?** Not what instrument produced it, not
whether it is dense or sparse — its role in the vessel.

| Your measurement | Goes in | Why |
|---|---|---|
| A concentration of something in the broth | `reactor_medium.components` | It participates in the mass balance and gets diluted. |
| Something that moved liquid in or out | `volume.volume_changes` | It *causes* dilution. |
| Anything else measured | `process_variables` | pH, temperature, DO, off-gas, optical signals. |

Biomass, glucose, product, acetate, ammonium → reactor medium. Feed pump, boluses,
sample draws → volume. pH, DO, temperature, off-gas CO₂ → process variables.

The distinction that matters: a **reactor medium component is subject to dilution**. Add
half a litre of feed and its concentration drops even if the cells did nothing. A process
variable is not treated that way. If you put pH in the reactor medium, bp-format will
faithfully dilute your pH, which is nonsense.

## The nesting

```
CaseStudy(case_id, organism, citation)     one publication or campaign
  └── processes: {name: BioProcess}
BioProcessCollection(metadata)             the loose alternative
  └── processes: {name: BioProcess}

BioProcess
  ├── metadata          BioProcessMetadata(name, process_type, notes)
  ├── time_axis         TimeAxis(unit, start, end, time_reference)     ← required
  ├── volume            Volume(initial_volume, unit, volume_changes)   ← required
  ├── reactor_medium    ReactorMedium(name, density, …, components)    ← required
  ├── process_variables {name: ProcessVariable}                          optional
  ├── biological_ode    BiologicalOde(algebraic, rates, derivatives)     auto-generated
  ├── discrete_events   DiscreteEvents                                   optional
  └── pseudobatch_transform                                              built later
```

Four positional arguments are required on `BioProcess`: `metadata` (may be `None`),
`time_axis`, `volume`, `reactor_medium`. Everything else has a default.

:::{admonition} Why dicts keyed by name, not lists
:class: note
`reactor_medium.components["glucose"]` is O(1), produces readable JSON, and keeps the
biological name attached to the data. The cost is that names are load-bearing — see the
collision rules in [The mechanistic ODE](mechanistic_ode.md).
:::

### `CaseStudy` or `BioProcessCollection`?

Both hold the same `BioProcess` objects.

- **`CaseStudy`** requires `case_id`, `organism` and `citation`. Use it for a finished
  dataset: one publication, one campaign. `case_id` is the natural grouping for
  cross-validation.
- **`BioProcessCollection`** requires nothing but the processes. Use it for raw or
  intermediate data that is not a case study yet.

They are not interchangeable at every API boundary — notably `model_predict` wants a
collection. Converting is one line:

```{code-cell} ipython3
import bp_format as bp

case_study = bp.serialization.load_case_study("../_data/out/demo_batch/data.json")
collection = bp.BioProcessCollection(processes=case_study.processes)
print(type(collection).__name__, "with", len(collection.processes), "processes")
```

## Values: `TimeSeries` or `StaticVariable`

Anywhere a value could be constant, both are accepted.

```{code-cell} ipython3
import numpy as np

measured = bp.TimeSeries(times=np.array([0.0, 1.0, 2.0]),
                         values=np.array([0.1, 0.4, 1.1]))
known    = bp.StaticVariable(400.0)      # e.g. a feed concentration
```

`TimeSeries` invariants, all enforced at construction:

- `times` strictly increasing.
- `times` and `values` supplied **together** or not at all.
- At least one of {samples, spline} must be present — an empty `TimeSeries` is an error.
- float64. Importing `bp_format` turns on JAX's x64 mode; float32 input raises rather
  than silently upcasting.

More on the spline half in [Time series and splines](time_series_and_splines.md).

## `is_controlled`: the modeled/controlled switch

Every `ProcessVariable` and every `VolumeChange` carries it.

```{code-cell} ipython3
temperature = bp.ProcessVariable(
    name="temperature", unit="degC", is_controlled=True,      # a known input
    values=bp.TimeSeries(times=np.array([0.0, 10.0]),
                         values=np.array([37.0, 37.0])),
)
```

`is_controlled=True` means *read this from the data*; `False` means *the model must
produce a derivative for it*. A modeled quantity therefore needs a time axis:

:::{admonition} A `StaticVariable` cannot be modeled
:class: warning
`ProcessVariable(values=StaticVariable(...), is_controlled=False)` is rejected when the
process ordering is built — a state with no time axis cannot be integrated. Either mark
it controlled, or give it real dynamics.
:::

## Bounds

Every component, process variable and volume can carry `bounds=(lo, hi)`, with `None`
meaning unbounded.

```{code-cell} ipython3
component = bp.ReactorMediumComponent(
    name="glucose", unit="g/L",
    concentration=measured,
    bounds=(0.0, None),
)
```

**Bounds are metadata.** Nothing in bp-format enforces them, and no solver clips to them.
They exist so downstream consumers — bp-train's loss module in particular — can build
soft penalties from a declaration you made once, in the data, instead of duplicating it
in every training config.

## Units

Units are **free-form strings**. There is no unit engine, no parsing and no conversion.

They are used for exactly two things: checking that quantities you *add together* in a
`biological_ode` expression share a unit, and checking that processes in one case study
agree with each other. `"g/L"` and `"g/l"` are different strings and will be reported as
a mismatch. Pick a spelling and stay with it.

## Where the biology goes

There are no biological flags on components. There is no `is_intracellular` switch. If a
species behaves unusually, you write it out in `biological_ode` — including the algebraic
relationships:

```python
BiologicalOde(
    algebraic={"X_active": "biomass - product_intracellular"},
    rates={"q_biomass": (None, None), "q_glucose": (None, None)},
    derivatives={"biomass": "q_biomass * X_active", ...},
)
```

This is deliberate: one place to look for what the model does, rather than behaviour
scattered across boolean fields. See [The mechanistic ODE](mechanistic_ode.md).

## Gotchas

- **`AugmentedBioProcess` exists but nothing in bp-format produces one.** It is a fixed
  shape so bp-train's augmentation and LOO grouping can rely on it. Ignore it unless you
  are using augmentation.
- **`DiscreteEvents` is a convenience mirror.** The authoritative source of events is
  always `volume.volume_changes` entries with `is_continuous=False`.
- **`metadata` on `BioProcess` may be `None`**, but it is positional — you must pass
  something.

## See also

- [Volume, feeds and events](volume_feeds_events.md) — the part with the most sharp edges.
- [The mechanistic ODE](mechanistic_ode.md) — what gets derived from all of this.
- [Tutorial 1](../tutorials/01_your_first_dataset.md) — building one step by step.
- [API reference](../autoapi/bp_format/dataclasses/index) — every field.
