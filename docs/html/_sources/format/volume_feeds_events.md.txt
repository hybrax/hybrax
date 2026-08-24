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

# Volume, feeds and events

> Everything that moves liquid, and why volume is its own category rather than a state
> or a control. Read the last section even for a pure batch: sampling counts too.

This is where most real datasets go wrong, and where hybrax.format saves you the most work.

## Why volume is its own thing

Volume is not a state variable and not a control input. It is separate because:

- **Many operations affect it**: continuous feeds, boluses and sampling, each with
  different composition and timing.
- **It enters the ODE differently**: every feed stream `k` contributes a dilution term
  `(f_k / V) · (C_in[k,i] − c_i)` to *every* species `i`, and volume itself evolves as
  `dV/dt = Σ f_k`.
- **It carries composition**: a feed is not just litres, it is litres *of something*.

Get the volume description right and every one of those terms is written for you. Get it
wrong and they are all wrong together, quietly.

## The three things that move volume

```{code-cell} ipython3
import numpy as np
import hybrax.format as hxf

cs = hxf.serialization.load_process_collection("../_data/out/demo_fedbatch/data.json")
process = cs.processes["fedbatch_1"]

for name, vc in process.volume.volume_changes.items():
    print(f"{name:15s} {type(vc).__name__:20s} continuous={vc.is_continuous}")
```

| | Type | `is_continuous` | Values are |
|---|---|---|---|
| Pump running | `Inflow` | `True` | a **cumulative volume** trace |
| Bolus | `Inflow` | `False` | one **signed delta per event** |
| Sample draw | `Outflow` | `False` | one **negative delta per event** |

Two rules that catch most import bugs:

**Volume changes are stored in the volume unit (litres, kilograms) never as a rate.**
A continuous feed is the *cumulative* volume delivered; hybrax.format differentiates it to
get the flow. If your control software exported a flow rate, integrate it first.

**Sign is fixed by the type.** Feeds are non-negative, samples non-positive. The type
system encodes the convention and a validator enforces it.

```{code-cell} ipython3
feed = process.volume.volume_changes["glucose_feed"]
print("cumulative volume delivered:",
      np.asarray(feed.values.values)[[0, -1]], feed.unit)

bolus = process.volume.volume_changes["glucose_bolus"]
print("bolus times :", np.asarray(bolus.values.times))
print("bolus deltas:", np.asarray(bolus.values.values))

sampling = process.volume.volume_changes["sampling"]
print("sample deltas:", np.asarray(sampling.values.values)[:3], "...")
```

## Feed composition

Every `Inflow` carries a `FeedMedium`. This is chemistry, not decoration: it is
what makes the difference between "add 100 mL" and "add 100 mL containing 40 g glucose".

```{code-cell} ipython3
medium = feed.feed_medium
for name, comp in medium.components.items():
    print(f"{name:10s} {comp.concentration.value:8.1f} {comp.unit}")
```

:::{admonition} State every reactor species, including the zeros
:class: warning

A species missing from a feed medium is **ambiguous**: it could mean genuinely absent,
or simply not recorded. hybrax.format will not guess, and `validate_volume_change_states`
flags it.

Write the zeros explicitly, as above. It matters: a feed that contains no biomass still
*dilutes* biomass, and that dilution term only appears if the zero is declared.
:::

:::{admonition} Feed concentrations must be constant
:class: important
A `FeedMediumComponent` concentration is schema-wise allowed to be a `TimeSeries`, but
`build_rhs_ode` raises `NotImplementedError` for it. In practice: use
`hxf.StaticVariable`. A feed whose composition genuinely changed over time needs to be
split into separate feed streams. See [Limits and gotchas](limits_and_gotchas.md).
:::

## Sampling is not optional bookkeeping

Every offline measurement came from a physical sample, and every physical sample removed
volume. If you have offline data, you had sample draws.

A sample is a **well-mixed removal**: amount and volume drop together, so concentrations
are *unchanged* at the instant of sampling. What changes is everything afterwards: a
smaller vessel dilutes differently.

Two things to get right:

- **One draw, many assays.** Glucose, biomass and product measured from the same sample
  are one volume removal, not three.
- **Whole-broth versus filtered.** A whole-broth sample removes cells and liquid; a
  filtered sample removes liquid and dissolved species but leaves the cells behind. Only
  the first should remove biomass.

If sample volumes are genuinely unknown, that is under-specified metadata, not zero.
Recording them as absent is defensible; recording them as `0.0` asserts something false.

## Event ordering at a shared timestamp

When a sample and a bolus share a timestamp (which happens constantly, because you
sample then feed) the order is fixed:

1. **Sample first.** The offline row describes the pre-feed reactor state.
2. **Bolus second.** It dilutes from the post-sample volume, then adds its mass.

hybrax.format and hybrax.train both apply this order. It is why a measurement timestamped
strictly *before* a bolus is unaffected by that bolus, and why the alignment validator
cares about measurements landing just *after* a sample.

## Checking your bookkeeping

```{code-cell} ipython3
import json
truth = json.loads(open("../_data/out/demo_fedbatch/ground_truth.json").read())
ok, message, _ = hxf.validate_volume_consistency(
    process, final_volume=truth["final_volume"])
print(message)
```

Run this whenever you build a fed-batch dataset. A per-stream balance sheet against the
volume you actually measured localises the error to one stream immediately.

## A transport-only sanity check

The strongest test available, and it costs nothing: set every biological rate to zero and
integrate. With no biology, concentrations may only change through feed composition,
dilution and sampling. If something moves that shouldn't (or a `pseudobatch_concentration`
trace jumps at a pure sampling event) the volume accounting is wrong, and you have found
it before fitting anything.

## Gotchas

- **Continuous feed values must be cumulative and non-decreasing.** A trace that dips
  implies negative flow.
- **An `Inflow` with no `feed_medium`** raises when the process ordering is
  built.
- **A feed naming a species that is not in `reactor_medium.components`** also raises: 
  hybrax.format will not invent a state for it.
- **Modeled feeds exist.** `is_controlled=False` on a feed means the *model* predicts the
  flow rate. That is an advanced case; see [the Reaction Module](../train/reaction_module.md).

## See also

- [Gallery: fed-batch](../gallery/fed_batch.md): all of this in one worked example.
- [Time series and splines](time_series_and_splines.md): the pseudobatch transform,
  which exists precisely because of dilution.
- [The Bioprocess ODE](bioprocess_ode.md): the terms generated from this description.
