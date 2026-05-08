# Data Model

Source: `bp_format/dataclasses.py`

## Purpose

The data model defines a hierarchical set of Python dataclasses that describe bioprocess experiments from raw measurements up to multi-study benchmark datasets. Every class uses standard `@dataclass` decorators (not `eqx.Module`) because the outer containers hold `Dict[str, ...]` fields that are manipulated outside the JAX JIT boundary. The one exception is `TimeSeries`, which lives in the `time_series/` subpackage and is an `eqx.Module`.

## Design Rationale

- **Dict keyed by name, not lists:** Components are stored as `Dict[str, Component]` (e.g., `reactor_medium.components["glucose"]`). This gives O(1) lookup, produces clean JSON keys, and makes iteration order explicit.
- **Volume is separate from states and controls:** Volume is affected by multiple operations (feeds, sampling, evaporation) and enters the ODE differently from states. See [Design Rationale: Volume](01_design_rationale.md#3-volume-as-a-first-class-concept).
- **Intracellular accumulation lives in `BiologicalOde`:** When a species accumulates inside cells (e.g., inclusion bodies), the user encodes the active-biomass relationship explicitly via `BiologicalOde.algebraic` (e.g., `{"X_active": "biomass - product"}`) and the corresponding derivatives. There is no flag on `ReactorMediumComponent`; `BioProcess.__post_init__` auto-fills a minimal block with `dc/dt = q · biomass` for each reactor component when the user does not supply one, and biology that departs from that template (intracellular bookkeeping, dead-cell pools, custom algebraics) is a user-supplied `BiologicalOde` block consumed by the same `RhsOde`.
- **Feed/Sample subtypes:** `FeedVolumeChange` and `SampleVolumeChange` enforce sign conventions at the type level and only feeds carry a `FeedMedium` reference (sampling removes reactor contents at current concentrations).
- **`TimeSeries | StaticVariable` union:** Concentrations and process variables can be either time-varying (measured) or constant (known). The union type handles both cases cleanly.

## Class Reference

### Low-Level Structures

#### `TimeAxis`
Defines the time domain for a bioprocess.

```python
@dataclass
class TimeAxis:
    unit: str           # e.g. "h", "days"
    start: float        # process start time
    end: float          # process end time
    time_reference: str  # "inoculation", "first_feed", or "operator_defined"
```

The `time_reference` field documents what `t=0` corresponds to, which is critical for aligning data across processes.

#### `StaticVariable`
A single time-independent scalar value.

```python
@dataclass
class StaticVariable:
    value: float
```

Used for constant feed concentrations, fixed parameters, etc.

#### `TimeSeries`
The canonical carrier for both sampled trajectories and optional spline state.

```python
class TimeSeries(eqx.Module):
    times: jnp.ndarray
    values: jnp.ndarray
    jump_times: jnp.ndarray
    continuity_side: str
    breaks: Optional[jnp.ndarray]
    coeffs: Optional[jnp.ndarray]
    segment_start_piece_idx: Optional[jnp.ndarray]
    metadata: Optional[dict]
```

`TimeSeries` is the only continuous-value carrier in the active model. Raw
measurements live in `times` / `values`, while fitted spline state lives
directly on the same object via `breaks`, `coeffs`, and
`segment_start_piece_idx`. Pseudobatch transforms are also stored on
`TimeSeries.metadata`.

#### `DiscreteEvents`
Stores discrete event times (bolus feeds, sampling, volume jumps).

```python
@dataclass
class DiscreteEvents:
    times: jnp.ndarray          # sorted, unique event times
    labels: Optional[list]       # optional labels for each event
    metadata: Optional[dict]     # additional event metadata
```

### Medium Components

#### `ReactorMediumComponent`
A single species (biomass, substrate, product) measured in the reactor.

```python
@dataclass
class ReactorMediumComponent:
    name: str                                    # e.g. "glucose", "biomass"
    unit: str                                    # e.g. "g/L", "mM"
    concentration: TimeSeries | StaticVariable   # measured concentration over time
    bounds: Bounds = (None, None)                # optional metadata: (lo, hi); None on either side = unbounded
```

Active biomass and other derived quantities (e.g. `X_active = biomass - product`) are declared on `BiologicalOde.algebraic`, not on the component itself. The auto-generated block uses `dc/dt = q · biomass` uniformly; any process whose biology departs from that template supplies a custom `BiologicalOde` block consumed by the same `RhsOde`.

The `bounds` field is **metadata only**: never plumbed into `RhsOde` / integrator. Downstream consumers (e.g. `bp-train`'s loss generator) read it off the process to build soft-constraint penalties such as "concentrations cannot be negative".

#### `FeedMediumComponent`
A single species in a feed stream.

```python
@dataclass
class FeedMediumComponent:
    name: str                                    # e.g. "glucose", "ammonium"
    unit: str                                    # e.g. "g/L"
    concentration: TimeSeries | StaticVariable   # feed concentration (usually static)
    is_controlled: bool                          # whether concentration is operator-controlled
```

#### `ReactorMedium`
The reactor contents: a collection of components with density information.

```python
@dataclass
class ReactorMedium:
    name: str
    density: float        # often 1.0 kg/L for aqueous solutions
    density_unit: str     # typically "kg/L"
    components: Dict[str, ReactorMediumComponent]
```

#### `FeedMedium`
A feed stream: composition and density.

```python
@dataclass
class FeedMedium:
    name: str
    density: float
    density_unit: str
    components: Dict[str, FeedMediumComponent]
```

### Process Variables

#### `ProcessVariable`
Non-concentration signals such as pH, temperature, dissolved oxygen, or off-gas measurements.

```python
@dataclass
class ProcessVariable:
    name: str                                    # original name from paper
    unit: str                                    # e.g. "°C", "%", "L/h"
    is_controlled: bool                          # True for controls (pH, DO), False for states (off-gas)
    values: TimeSeries | StaticVariable
    bounds: Bounds = (None, None)                # optional metadata: (lo, hi); None on either side = unbounded
```

The `is_controlled` flag determines whether this variable is treated as a known input (control) or an observed output (state) in the mechanistic module. Under a user-defined `BiologicalOde`:

- Uncontrolled PVs are *dynamic states* — they may appear as keys in `BiologicalOde.derivatives` (with a user-written derivative).
- Controlled PVs are *time-varying inputs* — they may appear as symbols inside expressions but never as keys.

`bounds` is metadata only (see `ReactorMediumComponent`).

### Volume Operations

#### `VolumeChange` (base)
Base class for volume change events.

```python
@dataclass
class VolumeChange:
    name: str
    unit: str             # "L", "m3", "kg" — NOT rates like "L/h"
    is_controlled: bool   # True if operator-controlled
    is_continuous: bool   # True for continuous flows, False for bolus/discrete events
    values: TimeSeries    # cumulative volume change over time
```

Volume changes store cumulative volumes (not rates), because rates are usually derived quantities.

#### `FeedVolumeChange(VolumeChange)`
Inflow with associated feed medium composition. All delta values must be >= 0.

```python
@dataclass
class FeedVolumeChange(VolumeChange):
    feed_medium: FeedMedium
```

#### `SampleVolumeChange(VolumeChange)`
Outflow from sampling. All delta values must be <= 0.

```python
@dataclass
class SampleVolumeChange(VolumeChange):
    pass
```

Sampling removes reactor contents at current concentrations, so no separate medium definition is needed.

#### `Volume`
Container aggregating initial volume and all volume change operations.

```python
@dataclass
class Volume:
    initial_volume: float
    unit: str                                      # "L", "m3", "kg"
    volume_changes: Dict[str, VolumeChange]        # keyed by operation name
    bounds: Bounds = (None, None)                  # optional metadata on V (e.g. (0, V_max_reactor))
```

### Process Level

#### `BioProcessMetadata`
Static metadata about a process run.

```python
@dataclass
class BioProcessMetadata:
    name: str                    # process identifier
    process_type: str            # "batch", "fed_batch", or "continuous"
    notes: Optional[str]         # free-text notes
```

#### `BioProcess`
A single experimental bioprocess run. This is the central object in bp-format.

```python
@dataclass
class BioProcess:
    metadata: Optional[BioProcessMetadata]
    time_axis: TimeAxis
    volume: Volume
    reactor_medium: ReactorMedium
    process_variables: Dict[str, ProcessVariable]
    discrete_events: Optional[DiscreteEvents]
    biological_ode: Optional[BiologicalOde] = None  # user-defined per-state biological RHS
```

When `biological_ode` is `None` (default), the mechanistic module auto-generates the RHS as `q_i * c_biomass + r_i + feed_dilution` per reactor state, uniformly across components. When set, the user-defined block takes precedence — intracellular bookkeeping, dead-cell pools, and other custom biology are expressed there. See `BiologicalOde` below and the [Mechanistic Module](08_mechanistic.md) page for full semantics.

#### `BiologicalOde`
User-defined per-state biological RHS expressions. Describes only the *biological* part of `dc/dt`; physical contributions (feed, dilution, sample, dV) continue to be added by bp-format from the existing `VolumeChange` machinery.

```python
@dataclass
class BiologicalOde:
    derived: Dict[str, str]            # name -> algebraic expression string
    rates: Dict[str, RateDecl]         # rate-symbol name -> per-rate metadata
    derivatives: Dict[str, str]        # state name -> dc/dt expression (biological part)
```

Validation (see `validate_biological_ode`) requires:

- Every dynamic state (reactor component or uncontrolled PV) appears as a key in `derivatives`. Use `"0"` to declare *no biological dynamics for this state* — the entry must be present, even when zero, so every choice is deliberate.
- Every free symbol in any expression resolves to one of: a state name, a controlled-PV name (input), a `derived` name, or a `rates` name.
- Derived-variable dependencies are acyclic (topo-sorted at build time).
- Rate names are disjoint from state, derived, and controlled-PV names.

#### `RateDecl`
Per-rate metadata under `BiologicalOde.rates`.

```python
@dataclass
class RateDecl:
    bounds: Bounds = (None, None)      # metadata only
```

Rates are abstract placeholders: their values are supplied at call time by the runtime — today by spline-fitting from concentrations, later by a neural network in `bp-train`. `len(BiologicalOde.rates)` is the rate-vector dimension (and therefore `bp-train`'s NN output dimension when training).

#### `AugmentedBioProcess`
A synthetic variant of a real `BioProcess` that lives next to its parent in
the same `BioProcessCollection` / `CaseStudy`. Same fields as `BioProcess`,
plus a mandatory `parent_process` string referencing the parent's key.

```python
@dataclass(kw_only=True)
class AugmentedBioProcess(BioProcess):
    parent_process: str        # key of the real BioProcess this was derived from
```

**What it represents (current and future).** Augmented processes are
intended to capture *data augmentation* outputs — for example pseudo-batch
transformations, noise-injected replicates, spline-resampled variants, or
counterfactual reconstructions of an experiment. Today the class is a
placeholder: bp-format only defines the data shape and validates parent
references, while the actual augmentation pipelines live in downstream
packages (`bp-train prepare` is the planned producer).

The structural contract is intentionally narrow:

- Augmented children must share the parent's structural identity (same
  control/state schema, medium semantics, units). Validation reuses the
  cross-process consistency checks that already apply to real processes
  in a `CaseStudy`.
- `parent_process` must resolve to a non-augmented `BioProcess` in the
  same container. Chained augmentation (augmented-of-augmented) is
  rejected in v1; if a future use case demands it, lift the restriction
  in `validate_augmented_parent_refs`.
- Because `AugmentedBioProcess` is a `BioProcess` subclass, every existing
  `Dict[str, BioProcess]` container, the mechanistic RHS path, and any
  validation/serialization hook that already accepts `BioProcess`
  transparently accepts the augmented variant.

**Why grouping matters.** Downstream consumers must treat the parent and
its augmented children as one indivisible unit when constructing
train/eval splits, holdout sets, or cross-validation folds. An augmented
sibling that ends up on the train side while its parent is held out
constitutes data leakage. `validate_augmented_parent_refs` lets those
consumers discover the parent → children grouping deterministically.

**Serialization.** Augmented processes are tagged in JSON with
`"__type__": "AugmentedBioProcess"` plus the `parent_process` field;
loaders reconstruct them via `load_process_collection_json` /
`load_dataset_json`. Plain `BioProcess` payloads are unchanged.

### Collection Level

#### `BioProcessCollection`
A lightweight wrapper for multiple processes without full case-study metadata.

```python
@dataclass
class BioProcessCollection:
    metadata: Optional[Dict]
    processes: Dict[str, BioProcess]
```

#### `CaseStudy`
Processes from one publication or experimental campaign.

```python
@dataclass
class CaseStudy:
    case_id: str           # unique identifier
    organism: str          # e.g. "E. coli", "S. cerevisiae", "CHO"
    citation: str          # publication reference
    processes: Dict[str, BioProcess]
```

#### `BenchmarkDataset`
Top-level container for cross-study benchmarking.

```python
@dataclass
class BenchmarkDataset:
    metadata: Dict[str, str]                    # name, version, description, etc.
    case_studies: Dict[str, CaseStudy]
```

## Examples

### Constructing a Minimal Batch Process

```python
import bp_format as bp
import jax.numpy as jnp

# Time axis
time_axis = bp.TimeAxis(unit="h", start=0.0, end=24.0, time_reference="inoculation")

# Reactor medium with biomass and glucose
reactor_medium = bp.ReactorMedium(
    name="minimal_medium",
    density=1.0,
    density_unit="kg/L",
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
)

# Volume (batch = no changes)
volume = bp.Volume(initial_volume=1.0, unit="L")

# Assemble the process
process = bp.BioProcess(
    metadata=bp.BioProcessMetadata(name="batch_001", process_type="batch"),
    time_axis=time_axis,
    volume=volume,
    reactor_medium=reactor_medium,
)
```

### Constructing a Fed-Batch Process with a Bolus Feed

```python
import bp_format as bp
import jax.numpy as jnp

time_axis = bp.TimeAxis(unit="h", start=0.0, end=48.0, time_reference="inoculation")

# Feed medium (glucose solution)
feed_medium = bp.FeedMedium(
    name="glucose_feed",
    density=1.05,
    density_unit="kg/L",
    components={
        "glucose": bp.FeedMediumComponent(
            name="glucose", unit="g/L",
            concentration=bp.StaticVariable(value=500.0),
        ),
        "biomass": bp.FeedMediumComponent(
            name="biomass", unit="g/L",
            concentration=bp.StaticVariable(value=0.0),
        ),
    },
)

# Volume with a bolus feed at t=12h and sampling at t=6h, t=18h
volume = bp.Volume(
    initial_volume=1.0,
    unit="L",
    volume_changes={
        "glucose_bolus": bp.FeedVolumeChange(
            name="glucose_bolus", unit="L",
            is_controlled=True, is_continuous=False,
            values=bp.TimeSeries(
                times=jnp.array([12.0]),
                values=jnp.array([0.1]),
            ),
            feed_medium=feed_medium,
        ),
        "sampling": bp.SampleVolumeChange(
            name="sampling", unit="L",
            is_controlled=True, is_continuous=False,
            values=bp.TimeSeries(
                times=jnp.array([6.0, 18.0]),
                values=jnp.array([-0.005, -0.005]),
            ),
        ),
    },
)

# ... add reactor_medium and assemble as above
```

### Accessing Nested Fields

```python
# Get glucose concentration time series
glucose_ts = process.reactor_medium.components["glucose"].concentration
print(glucose_ts.times)   # jnp.array([0.0, 6.0, 12.0, ...])
print(glucose_ts.values)  # jnp.array([20.0, 17.5, 12.0, ...])

# Iterate over all volume changes
for name, vc in process.volume.volume_changes.items():
    print(f"{name}: {type(vc).__name__}, continuous={vc.is_continuous}")

# Check process type
print(process.metadata.process_type)  # "batch" or "fed_batch"
```

## See Also

- [Design Rationale](01_design_rationale.md) -- cross-cutting decisions behind this model
- [TimeSeries](06_time_series.md) -- the `TimeSeries` class in detail
- [Serialization](03_serialization.md) -- how these structures are saved/loaded as JSON
- [Validation](04_validation.md) -- integrity checks on these structures
