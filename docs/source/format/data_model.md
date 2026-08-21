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

# The Data Model

> Which object holds which measurement, and why the split is where it is.

This page gives an overview of the `hybrax-format` data structure and the *decision* you have to make to import your bioprocess data into it. Exhaustive field lists for all data objects live in the [API reference](../autoapi/bp_format/dataclasses/index).

## The Data Structure

```
BioProcessCollection
  ├── case_id              Optional[str] = None
  ├── organism             Optional[str] = None
  ├── citation             Optional[str] = None
  ├── metadata             Optional[dict] = None
  └── processes            {name: BioProcess}

BioProcess
  ├── metadata              Optional[BioProcessMetadata(name, process_type, notes)]
  ├── time_axis             TimeAxis(unit, start, end, time_reference)
  ├── volume                Volume(initial_volume, unit, volume_changes)
  ├── reactor_medium        ReactorMedium(name, density, …, components)
  ├── process_variables     {name: ProcessVariable} = {}
  ├── discrete_events       Optional[DiscreteEvents] = None
  ├── biological_ode        Optional[BiologicalOde(algebraic, rates, derivatives)] = None
  └── pseudobatch_transform Optional[PseudobatchTransform] = None
```

`BioProcess` has four arguments with no default: `metadata`, `time_axis`, `volume`,
`reactor_medium`. You must supply all four, but `metadata` accepts `None` as its value:
"required" here means *you must pass something*, not *it must be non-null*. Everything
else on `BioProcess` has a real default. `BioProcessCollection.metadata` above is a
different field entirely: a free-form dict for provenance, unrelated to
`BioProcess.metadata`.

## Where Goes What?

Every time series you import gets asked the same three questions, in order: what is it
(its role), how is it shaped (a real time series or a fixed value), and who drives it
(read from data, or produced by the ODE).

<!-- LOCK -->
### The Physical Role

The `hybrax-format` data format categorizes your measured process data depending on its physical role in the bioreactor. There are three major roles:

| Measurement | Role | Why | Examples |
|---|---|---|---|
| a concentration in the reactor | `reactor_medium.components` | affected by the [ODE](bioprocess_ode) *and* volume change effects | biomass, substrate, product |
| feed and sampling volumes | `volume.volume_changes` | *cause* volume change (and dilution) | (bolus) substrate feed, sampling
| Anything else measured | `process_variables` | affected by the [ODE](bioprocess_ode) but *independent* of volume change | pH, temperature, DO, product quality |

:::{admonition} The role you assign changes the physics
:class: note
A reactor medium component is diluted by every feed addition. A biomass
concentration assigned this way correctly drops when a liter of feed goes in.
A process variable is not diluted. Assign something like product quality as a
reactor medium component by mistake, and the same feed addition dilutes it
too, even though product quality is not a real concentration in the medium.
Register it as a process variable instead.
:::

<!-- UNLOCK -->

### Static or Time-Dependent

Anywhere a value could be constant, both are accepted.

```{code-cell} ipython3
import numpy as np
import bp_format as bp

c_reactor = bp.TimeSeries(times=np.array([0.0, 1.0, 2.0]),
                          values=np.array([0.1, 0.4, 1.1]))
c_feed    = bp.StaticVariable(400.0)
```

`StaticVariable` means constant *within one process*, not across the whole collection.
Two processes in the same `BioProcessCollection` can carry different `StaticVariable`
values for the same field: two different feed concentrations, or two different fixed
temperatures, are both ordinary uses, not an inconsistency.

These rules are checked every time a `TimeSeries` is constructed:
- `times` strictly increasing.
- `times` and `values` arrays have the same length.
- `times` and `values` require `float64` arrays.

:::{admonition} A length-1 `TimeSeries` is not a `StaticVariable`
:class: note
If you know only the initial point value of a quantity, use `TimeSeries`, not `StaticVariable`, so it can still be marked
`is_controlled=False` and modeled. A custom reaction module, like the one in
[FBA-Hyb](../gallery/fba_hyb.md), can still predict a real trajectory for it.
:::


The `TimeSeries` is a powerful dataclass that has even more to offer. You can find all details in [Time series and splines](time_series_and_splines.md).

### Controlled or Modeled

Every `ProcessVariable` and every `VolumeChange` carries `is_controlled = Bool` field. 
During training a quantity set to `is_controlled=True` means it is *read from the data* and *input to the bioprocess model*. Conversly `is_controlled = False` means *it is an model output*, i.e., the model must
produce a derivative for it.


```{code-cell} ipython3
temperature = bp.ProcessVariable(
    name="temperature", unit="degC", is_controlled=True,      # a known input
    values=bp.TimeSeries(times=np.array([0.0, 10.0]),
                         values=np.array([37.0, 37.0])),
)
```


:::{admonition} A `ReactorMediumComponent` is never controlled
:class: note
`ReactorMediumComponent` has no such field at all. Nothing in the medium is a true
control input: you cannot hold a biomass or glucose concentration fixed and have the
bioreactor obey. Every reactor medium component is dynamic, governed by the ODE. The
only real control input into the reactor's contents crosses its boundary: an `Inflow`
or an `Outflow`. That is why `is_controlled` lives on `VolumeChange` and
`ProcessVariable`, never on a medium component.
:::


:::{admonition} A `StaticVariable` cannot be modeled
:class: warning
`ProcessVariable(values=StaticVariable(...), is_controlled=False)` is rejected when the
process ordering is built: a state with no time axis cannot be integrated. Either mark
it controlled, or give it real dynamics.
:::

## Bounds

Every component, process variable and volume can carry `bounds=(lo, hi)`, with `None`
meaning unbounded.

```{code-cell} ipython3
component = bp.ReactorMediumComponent(
    name="glucose", unit="g/L",
    concentration=c_reactor,
    bounds=(0.0, None),
)
```

**Bounds are metadata.** Nothing in bp-format enforces them, and no solver clips to them.
They exist so downstream consumers (`hybrax-train`'s loss module in particular) can build
soft penalties from a declaration you made once, in the data, instead of duplicating it
in every training config.

## Units

Units are *free-form strings* which we require for data completeness. There is no unit engine, no parsing and no conversion.

They are used for exactly two things: checking that quantities you *add together* in a
`biological_ode` expression share a unit, and checking that processes in one case study
agree with each other. `"g/L"` and `"g/l"` are different strings and will be reported as
a mismatch. Pick a spelling and stay with it.

## Where the Biology Goes

`hybrax-format` guesses standard biology by default. If a species needs something more, an intracellular pool, a chemical decay, any mechanism the default doesn't cover, you write it out yourself in `biological_ode`, algebraic relationships included:


```python
BiologicalOde(
    algebraic={"X_active": "biomass - product_intracellular"},
    rates={"q_biomass": (None, None), "q_glucose": (None, None)},
    derivatives={"biomass": "q_biomass * X_active", ...},
)
```

See [The Bioprocess ODE](bioprocess_ode.md) page for more details.

## The Rest of the Fields

The remaining `BioProcess` fields, straightforward enough that they don't need their own
section:

- **`metadata`** (`BioProcessMetadata`): `name`, `process_type`
  (`"batch"`/`"fed_batch"`/`"continuous"`), optional `notes`. A static description of the
  run, not a measurement. No dedicated page. Field list is in the
  [API reference](../autoapi/bp_format/dataclasses/index).
- **`time_axis`** (`TimeAxis`): `unit`, `start`, `end`, `time_reference` (e.g.
  `"inoculation"`, `"first_feed"`, `"operator_defined"`): what `t=0` means for this
  process. Also no dedicated page.
- **`discrete_events`** (`DiscreteEvents`): a convenience mirror, not the source of truth.
  The real events are always the `volume.volume_changes` entries with
  `is_continuous=False`. See [Volume, feeds and events](volume_feeds_events.md).
- **`pseudobatch_transform`**: not something you construct. `hybrax-format` builds it from
  `reactor_medium` and `volume` once you run the transform. See
  [The pseudobatch transform](pseudobatch_transform.md).

## Gotchas

- **`AugmentedBioProcess` exists but nothing in bp-format produces one.** It is a fixed
  shape so bp-train's augmentation and LOO grouping can rely on it. Ignore it unless you
  are using augmentation.

## See also

- [Volume, feeds and events](volume_feeds_events.md): the part with the most sharp edges.
- [The pseudobatch transform](pseudobatch_transform.md): what pseudobatch_transform is built from, and why.
- [The Bioprocess ODE](bioprocess_ode.md): what gets derived from all of this.
- [Tutorial 1](../tutorials/01_your_first_dataset.md): building one step by step.
- [API reference](../autoapi/bp_format/dataclasses/index): every field.
