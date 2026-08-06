# Data Model

Source: `bp_format/dataclasses.py`

## Purpose

The data model describes a bioprocess experiment as nested Python dataclasses,
from a single measurement up to a whole published case study. Everything here is
a plain `@dataclass`; the one exception is `TimeSeries`, which lives in
`bp_format/time_series/` and is an `eqx.Module` so it can cross a JAX JIT
boundary (see [Design Rationale §1](01_design_rationale.md#1-jax-first-but-only-where-it-matters)).

All classes are re-exported from the package root: `bp.TimeAxis`,
`bp.BioProcess`, and so on.

## Key choices

- **Dicts keyed by name, not lists.** `reactor_medium.components["glucose"]` is
  O(1), makes clean JSON, and keeps iteration explicit.
- **Volume is separate from states and controls** — see
  [Design Rationale §3](01_design_rationale.md#3-volume-is-its-own-category).
- **Feed and sample are different types.** `FeedVolumeChange` and
  `SampleVolumeChange` fix the sign convention at the type level, and only feeds
  carry a `FeedMedium`.
- **`TimeSeries | StaticVariable` everywhere a value could be constant.** A
  measured concentration is a `TimeSeries`; a known feed concentration is a
  `StaticVariable`.
- **Biology lives in `BiologicalOde`, not in flags on components.** There is no
  `is_intracellular` switch. If a species accumulates inside cells you write it
  out: `algebraic={"X_active": "biomass - product"}` plus the matching
  derivatives.

## Low-level structures

### `TimeAxis`

```python
@dataclass
class TimeAxis:
    unit: str            # "h", "days"
    start: float
    end: float
    time_reference: str  # "inoculation", "first_feed", "operator_defined"
```

`time_reference` records what `t = 0` means, which is what makes runs from
different sources alignable.

### `StaticVariable`

```python
@dataclass
class StaticVariable:
    value: float
```

One time-independent scalar. Used for feed concentrations, fixed setpoints, and
any process variable that never changes.

### `TimeSeries`

The carrier for anything that varies over time. Full treatment in
[06_time_series.md](06_time_series.md).

```python
class TimeSeries(eqx.Module):
    times: jnp.ndarray | None                    # strictly increasing
    values: jnp.ndarray | None                   # same length as times
    breaks: jnp.ndarray | None                   # fitted spline: breakpoints
    coeffs: jnp.ndarray | None                   # fitted spline: (n_pieces, 4)
    segment_start_piece_idx: jnp.ndarray | None  # fitted spline: segment starts
    jump_times: jnp.ndarray                      # known discontinuities
    derived: bool                                # computed, not measured
    continuity_side: str                         # "left" or "right"
    metadata: Any                                # free-form dict
```

At least one of {samples, spline} must be present. All float fields are float64.

### `DiscreteEvents`

```python
@dataclass
class DiscreteEvents:
    times: jnp.ndarray          # sorted, unique
    labels: Optional[list]
    metadata: Optional[dict]
```

An optional convenience record of event times. The authoritative source of
events is always `volume.volume_changes` with `is_continuous=False`.

### `Bounds`

```python
Bounds = Tuple[Optional[float], Optional[float]]   # (lower, upper)
```

`None` on either side means unbounded; `(None, None)` is the default.

**Bounds are metadata only.** They are never enforced by `RhsOde` or by any
integrator. Downstream consumers — bp-train's loss module, for instance — read
them off the process to build soft penalties such as "this concentration cannot
go negative".

## Medium components

### `ReactorMediumComponent`

One species measured in the reactor.

```python
@dataclass
class ReactorMediumComponent:
    name: str                                                 # "glucose", "biomass"
    unit: str                                                 # "g/L", "mM"
    concentration: TimeSeries | StaticVariable                # real, as measured
    c_star_concentration: TimeSeries | StaticVariable | None  # pseudobatch trace
    bounds: Bounds = (None, None)
```

`concentration` is *always* the real reactor concentration in physical units.
When a pseudobatch transform has been built, the derived `c*` trace goes in
`c_star_concentration` and the shared parts (ADF, feed corrections) go on
`BioProcess.pseudobatch_transform`. Loaders reject a `c*` trace with no matching
transform bundle, and reject feeding an already-transformed concentration back
into the transform builder.

### `FeedMediumComponent`

One species in a feed stream.

```python
@dataclass
class FeedMediumComponent:
    name: str
    unit: str
    concentration: TimeSeries | StaticVariable   # in practice: StaticVariable
    is_controlled: bool = False
```

> The mechanistic code currently requires `StaticVariable` here. A `TimeSeries`
> feed concentration raises `NotImplementedError` in `build_rhs_ode` and in the
> pseudobatch feed correction.

### `ReactorMedium` and `FeedMedium`

```python
@dataclass
class ReactorMedium:
    name: str
    density: float                  # often 1.0 for aqueous media
    density_unit: str               # typically "kg/L"
    components: Dict[str, ReactorMediumComponent]

@dataclass
class FeedMedium:
    name: str
    density: float
    density_unit: str
    components: Dict[str, FeedMediumComponent]
```

## Process variables

```python
@dataclass
class ProcessVariable:
    name: str
    unit: str                                # "°C", "%", "L/h"
    is_controlled: bool
    values: TimeSeries | StaticVariable
    bounds: Bounds = (None, None)
```

Anything that is not a concentration and not a volume change: pH, temperature,
dissolved oxygen, off-gas, stirrer speed.

`is_controlled` decides how the mechanistic module treats it:

| `is_controlled` | Role | In `BiologicalOde` |
|---|---|---|
| `True` | Known input, driven by recorded data | may appear *inside* expressions |
| `False` | Modeled state with its own `d/dt` | must appear as a **key** in `derivatives` |

A `StaticVariable`-valued process variable must have `is_controlled=True` — a
state with no time axis cannot be integrated, and `get_process_ordering` raises
if it finds one.

## Volume operations

```python
@dataclass
class VolumeChange:              # base
    name: str
    unit: str                    # "L", "m3", "kg" — never "L/h"
    is_controlled: bool          # True = operator-set, False = modeled
    is_continuous: bool          # True = flow profile, False = discrete events
    values: TimeSeries           # cumulative volume, or per-event deltas

@dataclass
class FeedVolumeChange(VolumeChange):
    feed_medium: FeedMedium      # values >= 0

@dataclass
class SampleVolumeChange(VolumeChange):
    pass                         # values <= 0
```

How `values` is read depends on `is_continuous`:

- **`is_continuous=True`** — a cumulative volume trace. Its spline *derivative*
  is the flow rate used in the ODE.
- **`is_continuous=False`** — one signed delta per event time. Boluses and
  sampling.

Sampling removes broth at whatever concentrations it currently has, so it needs
no medium definition.

```python
@dataclass
class Volume:
    initial_volume: float
    unit: str
    volume_changes: Dict[str, VolumeChange]
    total_volume: TimeSeries | None = None
    bounds: Bounds = (None, None)         # e.g. (0, max_working_volume)
```

`total_volume` is the full reactor-volume trace. It may be online measurement
data, or it may be reconstructed from `initial_volume` plus the volume changes —
`build_pseudobatch_transform` fills it in if it is still `None`.

## The biological ODE

```python
@dataclass
class BiologicalOde:
    algebraic: Dict[str, str]      # name -> expression
    rates: Dict[str, Bounds]       # rate symbol -> (lower, upper)
    derivatives: Dict[str, str]    # state name -> biological dc/dt
```

This block describes **only the biological part** of `dc/dt`. Feed inflow,
dilution, sample outflow, and `dV/dt` are added on top by bp-format from the
`Volume` machinery — you never write them yourself.

- **`algebraic`** — intermediate quantities recomputed every RHS call, e.g.
  `{"X_active": "biomass - product"}`. Must be acyclic.
- **`rates`** — names of the abstract specific rates the runtime supplies. This
  dict's **insertion order is the rate-vector layout** that downstream code
  passes in, so it is deliberately not sorted. The `Bounds` values are metadata
  for loss generators.
- **`derivatives`** — one entry per dynamic state (every reactor component and
  every uncontrolled process variable). Use `"0"` for "no biological dynamics".
  The entry must exist even when zero, so the choice is visible.

Every free symbol in any expression must resolve to a state name, a controlled
process-variable name, an `algebraic` name, or a `rates` name. See
[`validate_biological_ode`](04_validation.md#validate_biological_odeprocess).

### Auto-generation

If you leave `biological_ode` as `None`, `BioProcess.__post_init__` fills it in
with the standard template. That requires a reactor component named `biomass`
(case-insensitive) and produces:

- `algebraic = {}`
- `rates` — `q_<biomass>` first, then `q_<other components>` in insertion order,
  then `r_<dynamic process variable>` in insertion order
- `derivatives` — `"<component>": "q_<component> * <biomass>"` for reactor
  components, `"<pv>": "r_<pv>"` for dynamic process variables

Static process variables are skipped: they have no biological derivative.

**A process with reactor components but no `biomass` raises at construction
time.** Either name a component `biomass` or supply your own `biological_ode`.

## Process level

```python
@dataclass
class BioProcessMetadata:
    name: str
    process_type: str            # "batch", "fed_batch", "continuous"
    notes: Optional[str] = None

@dataclass
class BioProcess:
    metadata: Optional[BioProcessMetadata]
    time_axis: TimeAxis
    volume: Volume
    reactor_medium: ReactorMedium
    process_variables: Dict[str, ProcessVariable] = {}
    discrete_events: Optional[DiscreteEvents] = None
    biological_ode: Optional[BiologicalOde] = None       # auto-filled
    pseudobatch_transform: Optional[PseudobatchTransform] = None
```

### `PseudobatchTransform`

```python
@dataclass
class PseudobatchTransform:
    adf: TimeSeries                              # shared accumulated dilution factor
    feed_corrections: Dict[str, TimeSeries]      # per species
    sample_compensation: Optional[TimeSeries]    # diagnostic
    accumulated_feeds: Dict[str, TimeSeries]     # diagnostic, per feed stream
```

Only the parts shared across species live here. Per-species `c*` stays on
`ReactorMediumComponent.c_star_concentration`. Details in
[07_splines.md](07_splines.md).

### `AugmentedBioProcess`

A synthetic variant of a real run, stored beside its parent.

```python
@dataclass(kw_only=True)
class AugmentedBioProcess(BioProcess):
    parent_process: str      # key of the real BioProcess in the same container
```

Because it subclasses `BioProcess`, every container, validator, serializer, and
mechanistic builder accepts it unchanged.

Rules:

- `parent_process` must name a key in the same container that resolves to a
  **non-augmented** `BioProcess`. Augmented-of-augmented is rejected by
  `validate_augmented_parent_refs`.
- A child inherits its parent's structure — same state and control schema, same
  units.
- **Parent and children are one unit for splitting.** An augmented sibling on
  the train side while its parent is held out is data leakage. The validator
  exposes the parent → children grouping so consumers can honour it.

In JSON, augmented processes carry `"__type__": "AugmentedBioProcess"` plus
`parent_process`.

> No code in bp-format produces `AugmentedBioProcess` objects — the shape is
> fixed so that consumers (bp-train's augmentation and LOO orchestrator) can
> rely on it.

## Mechanistic ordering

### `ProcessOrdering`

The single place that decides the layout of every state, control, and rate
vector. Built by `bp_format.mechanistic.get_process_ordering(process)` and
consumed by every other mechanistic factory.

```python
@dataclass(frozen=True)
class ProcessOrdering:
    name_modeled_rates: Tuple[str, ...]      # BiologicalOde.rates insertion order
    name_modeled_algebraic: Tuple[str, ...]  # topo-sorted
    name_modeled_RMCs: Tuple[str, ...]       # alphabetical
    name_modeled_PVs: Tuple[str, ...]        # alphabetical, is_controlled=False
    name_modeled_FVCs: Tuple[str, ...]       # continuous + uncontrolled
    name_modeled_SVCs: Tuple[str, ...]       # continuous + uncontrolled
    name_controlled_PVs: Tuple[str, ...]     # alphabetical, is_controlled=True
    name_controlled_FVCs: Tuple[str, ...]    # continuous + controlled
    name_controlled_SVCs: Tuple[str, ...]    # continuous + controlled
```

Ordering rules:

- `name_modeled_rates` keeps the user's insertion order. Sorting it would
  silently reshuffle every rate vector downstream.
- `name_modeled_algebraic` is topologically sorted so each expression's
  dependencies are already computed; alphabetical within a level.
- Everything else is alphabetical. **Biomass has no reserved index.**

Resulting layouts:

```
state    c = [ modeled_RMCs... | modeled_PVs... | V ]
control  u = [ controlled_FVCs... | controlled_SVCs... | controlled_PVs... ]
```

See [08_mechanistic.md](08_mechanistic.md) for what the factories do with them.

## Top-level containers

```python
@dataclass
class CaseStudy:
    case_id: str
    organism: str            # "E. coli", "CHO"
    citation: str
    processes: Dict[str, BioProcess]

@dataclass
class BioProcessCollection:
    metadata: Optional[Dict] = None
    processes: Dict[str, BioProcess] = {}
```

One file on disk holds one of these. `CaseStudy` is the strict,
publication-linked form; `BioProcessCollection` is the loose form for raw or
intermediate data.

## Examples

### A minimal batch process

```python
import bp_format as bp
import jax.numpy as jnp

process = bp.BioProcess(
    metadata=bp.BioProcessMetadata(name="batch_001", process_type="batch"),
    time_axis=bp.TimeAxis(unit="h", start=0.0, end=24.0,
                          time_reference="inoculation"),
    volume=bp.Volume(initial_volume=1.0, unit="L"),
    reactor_medium=bp.ReactorMedium(
        name="minimal_medium", density=1.0, density_unit="kg/L",
        components={
            "biomass": bp.ReactorMediumComponent(
                name="biomass", unit="g/L",
                concentration=bp.TimeSeries(
                    times=jnp.array([0.0, 6.0, 12.0, 18.0, 24.0]),
                    values=jnp.array([0.5, 1.2, 3.1, 7.5, 12.0]),
                ),
            ),
            "glucose": bp.ReactorMediumComponent(
                name="glucose", unit="g/L",
                concentration=bp.TimeSeries(
                    times=jnp.array([0.0, 6.0, 12.0, 18.0, 24.0]),
                    values=jnp.array([20.0, 17.5, 12.0, 4.0, 0.1]),
                ),
            ),
        },
    ),
)

# biological_ode was auto-filled:
process.biological_ode.rates        # {"q_biomass": (None, None), "q_glucose": (None, None)}
process.biological_ode.derivatives  # {"biomass": "q_biomass * biomass",
                                    #  "glucose": "q_glucose * biomass"}
```

### Adding a bolus feed and sampling

```python
feed_medium = bp.FeedMedium(
    name="glucose_feed", density=1.05, density_unit="kg/L",
    components={
        # every reactor species should appear — 0.0 means "truly absent"
        "glucose": bp.FeedMediumComponent(
            name="glucose", unit="g/L", concentration=bp.StaticVariable(500.0)),
        "biomass": bp.FeedMediumComponent(
            name="biomass", unit="g/L", concentration=bp.StaticVariable(0.0)),
    },
)

process.volume = bp.Volume(
    initial_volume=1.0, unit="L",
    volume_changes={
        "glucose_bolus": bp.FeedVolumeChange(
            name="glucose_bolus", unit="L",
            is_controlled=True, is_continuous=False,
            values=bp.TimeSeries(times=jnp.array([12.0]),
                                 values=jnp.array([0.1])),      # >= 0
            feed_medium=feed_medium,
        ),
        "sampling": bp.SampleVolumeChange(
            name="sampling", unit="L",
            is_controlled=True, is_continuous=False,
            values=bp.TimeSeries(times=jnp.array([6.0, 18.0]),
                                 values=jnp.array([-0.005, -0.005])),   # <= 0
        ),
    },
)
```

### A custom biological ODE

For an intracellular product, biomass measurements include the product, so the
active biomass driving the rates is the difference:

```python
process.biological_ode = bp.BiologicalOde(
    algebraic={"X_active": "biomass - product"},
    rates={
        "q_growth":  (0.0, None),     # growth cannot be negative
        "q_product": (0.0, None),
        "q_glucose": (None, 0.0),     # uptake cannot be positive
    },
    derivatives={
        "biomass": "(q_growth + q_product) * X_active",
        "product": "q_product * X_active",
        "glucose": "q_glucose * X_active",
    },
)
```

### Reading nested fields

```python
glucose = process.reactor_medium.components["glucose"].concentration
glucose.times, glucose.values

for name, vc in process.volume.volume_changes.items():
    print(name, type(vc).__name__, "continuous" if vc.is_continuous else "discrete")
```

## See also

- [Design Rationale](01_design_rationale.md)
- [TimeSeries](06_time_series.md)
- [Serialization](03_serialization.md)
- [Validation](04_validation.md)
- [Mechanistic](08_mechanistic.md)
