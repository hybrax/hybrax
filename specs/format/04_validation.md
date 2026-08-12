# Validation

Source: `bp_format/validate.py`

## Purpose

Catch the errors that bioprocess data reliably contains — wrong signs, missing
species, misaligned timestamps — before they turn into silent numerical
nonsense during ODE integration.

Every validator returns `(bool, str)` (or `(bool, list[str])` for the
aggregates) instead of raising, so one pass collects **all** problems into a
report rather than stopping at the first. Structural impossibilities still raise
— see [Design Rationale §6](01_design_rationale.md#6-check-the-data-then-fail-loudly).

All 19 validators are exported from the package root: `bp.validate_process(...)`.

## Individual validators

### `validate_mapping_names(process)`

Dictionary keys match their embedded object `name` for reactor-medium
components, process variables, volume changes, and feed-medium components.
Collection process keys are identifiers and are not compared with
`process.metadata.name`.

### `validate_discrete_events(process)`

When discrete events are present, requires their timestamps to be one-dimensional,
strictly increasing and unique, and within the inclusive process time axis.
Optional event labels must have the same length as the timestamps.

### `validate_timeseries_shape(ts, name="")`

`times` and `values` are both 1-D, the same length, and `times` is strictly
increasing (no duplicates). Fails if the series has no discrete samples at all.
`validate_process` applies this to reactor-medium concentrations (including
`c_star_concentration`), process variables, volume changes, and measured total
volume.

### `validate_time_axis(process)`

Requires `process.time_axis.start <= process.time_axis.end`. Equal bounds are
valid.

### `validate_timestamp_bounds(process)`

Every timestamp falls inclusively between `process.time_axis.start` and
`process.time_axis.end`. This covers reactor-medium concentrations (including
`c_star_concentration`), process variables, volume changes, and measured total
volume. Timestamps are assumed to use `process.time_axis.unit`; no conversion is
performed. This per-process check is separate from cross-process time-axis
consistency.

### `validate_volume_change_sign(volume_change)`

- `FeedVolumeChange`: all values ≥ 0 (inflow)
- `SampleVolumeChange`: all values ≤ 0 (outflow)
- Unknown type: values must be purely positive or purely negative, never mixed

Uses a 1e-12 tolerance so exact zeros and float noise pass.

### `validate_volume_units(process)`

Every volume change uses exactly the same unit string as `process.volume.unit`.
Units are not parsed or converted, and no dimensional analysis is performed.

### `validate_volume_change_states(process)`

For every volume change that adds volume, checks that its `feed_medium` defines
a concentration for **every dynamic species** in the reactor medium (any
component whose concentration is a `TimeSeries`) and uses the same unit string.
Units are compared exactly; they are not parsed or converted, and no dimensional
analysis is performed. Feed concentrations may be static even when the reactor
concentration is dynamic.

A missing entry is ambiguous — did the feed really contain none of that species,
or was it just not recorded? The mass balance needs the answer, so state it
explicitly with `StaticVariable(0.0)` when the species is genuinely absent.

### `validate_biomass_in_reactor_medium(process)`

The reactor medium contains a component named `biomass` (case-insensitive).

This matters because the auto-generated `BiologicalOde` builds every derivative
as `q_<species> * biomass` and cannot do so without it. If your process has no
biomass component, supply your own `biological_ode` — the check will still
report, but the process is usable.

Biomass has **no reserved position** in the state vector; reactor components are
ordered alphabetically.

### `validate_measurement_sampling_alignment(process, rel_threshold=1e-4)`

Flags concentration measurements timestamped *slightly after* a sampling event —
strictly between the sample time and `sample_time + rel_threshold ·
process_length` (0.01 % of the run by default).

An offline measurement describes the broth **as drawn**, i.e. the pre-sample
state. A timestamp a few seconds late makes the pseudobatch transform pick the
post-sample volume, which corrupts the accumulated dilution factor and every
spline built on it. Move such timestamps onto the sampling time exactly.

Measurements *exactly at* a sampling time are correct and are not flagged.

### `validate_bounds(process)`

Every `Bounds` tuple on the process — `ReactorMediumComponent.bounds`,
`ProcessVariable.bounds`, `Volume.bounds` — has `lower <= upper` when both are
set. (Bounds on `BiologicalOde.rates` are checked by `validate_biological_ode`.)
This checks the tuple itself, not the data — see `validate_bounds_against_data`
below for that.

### `validate_bounds_against_data(process)`

For every `ReactorMediumComponent.concentration`, `ProcessVariable.values`, and
(when present) `Volume.total_volume` that carries a *set* `Bounds` tuple
(either side non-`None`), compares the actual measured value(s) — a scalar for
`StaticVariable`, the full array for `TimeSeries` — against `(lower, upper)`
and reports how many datapoints violate the bound, with the observed min/max.

`ReactorMediumComponent.bounds` defaults to `(0.0, None)` even when never set
explicitly, so this check catches negative concentrations by default, not just
on RMCs with an explicit bound. Out of scope: `BiologicalOde.rates` bounds — no
rate-inversion machinery exists in bp-format to compute a measured rate value
to check those against.

### `validate_biological_ode(process)`

The aggregate check for `process.biological_ode`. No-op when it is `None`.

- Every dynamic state (reactor component or uncontrolled process variable) has
  an entry in `derivatives`; `"0"` counts. Every `derivatives` key *is* a
  dynamic state — no extras.
- Every expression parses with sympy.
- Every free symbol resolves to a state, a controlled process variable, an
  `algebraic` name, or a `rates` name.
- The `algebraic` dependency graph is acyclic.
- Rate names collide with nothing; algebraic names collide with nothing.
- Every `rates` bounds tuple has `lower <= upper`.
- **Unit consistency in sums.** Any `Add` node that combines two or more state
  symbols requires those states to share a unit. `biomass - product` is fine at
  matching units and rejected when one is `g/L` and the other `mg/L` — the
  subtraction would be meaningless.

### `validate_biological_ode_equivalence(container)`

Given a `BioProcessCollection`, checks that every process exposes an identical
`BiologicalOde` (same `algebraic`, `rates`, and `derivatives`). One model has to
describe every run in a study for the benchmark to mean anything. Containers
with 0 or 1 process pass trivially.

Not part of `validate_for_publication`; call it directly, or let
[`print_rhs_ode`](05_inspection.md) run it for you.

### `validate_cross_process_consistency(collection)`

Given a `BioProcessCollection`, checks that every process shares identical
structure against the first process:

- same reactor components, each with the same value type and unit
- same process variables, each with the same value type and unit
- same volume unit
- same volume-change names and units
- same time-axis unit and time reference

All units and the time reference are compared as exact strings. Units are not
parsed or converted, and no dimensional analysis is performed. Time-axis start
and end may differ.

Collections with 0 or 1 process pass trivially. This is the check
`validate_for_publication` composes to build its `"__consistency__"` report
entry; call it directly for just the structural-consistency signal.

### `validate_volume_consistency(process, final_volume)`

Initial volume plus all volume changes should land within 5 % of the measured
final volume. Returns `(bool, str, float)` — the third element is the net volume
change. This numeric check assumes compatible units; call `validate_volume_units`
first (or use `validate_process`) to check exact unit-string coherence.

Continuous changes contribute `values[-1] - values[0]` (cumulative); discrete
changes contribute `sum(values)`. Only the endpoints are used, because this runs
*before* any spline fitting.

> `final_volume` has no usable default — pass the measured value.

### `validate_augmented_parent_refs(container)`

For every `AugmentedBioProcess` in a `BioProcessCollection`: `parent_process`
names a key in the same container, and that key resolves to a
**non-augmented** `BioProcess`. Chained augmentation is rejected.

Returns `(bool, list[str])`, always with at least one summary line. Called
automatically by `validate_for_publication`.

## Aggregate validators

### `validate_process(process) -> (bool, list[str])`

Runs, in order:

1. `validate_mapping_names`
2. `validate_discrete_events`
3. `validate_timeseries_shape` on reactor components (including
   `c_star_concentration`), process variables, volume changes, and measured total
   volume carrying a `TimeSeries`
4. `validate_time_axis`
5. `validate_timestamp_bounds`
6. `validate_volume_units`
7. `validate_volume_change_sign` on every volume change
8. `validate_volume_change_states`
9. `validate_biomass_in_reactor_medium`
10. `validate_measurement_sampling_alignment`
11. `validate_bounds`
12. `validate_bounds_against_data`
13. `validate_biological_ode`

Returns one message per check — including the passing ones, so the output reads
as a checklist. Raises `TypeError` if given something that is not a `BioProcess`.

`validate_volume_consistency` is **not** included (it needs a `final_volume`
argument you have to supply).

### `validate_for_publication(collection) -> (bool, dict[str, list[str]])`

`validate_process` on every process, plus `validate_cross_process_consistency`
and `validate_augmented_parent_refs`. This is bp-format's own concern — is this
collection well-formed and internally coherent enough to store or publish as a
case study — distinct from bp-train's training-readiness concern
(`bp_train.validation.validate_for_training`, which composes the same
`validate_cross_process_consistency` check rather than duplicating it).

Results are keyed by process name, with cross-process findings under
`"__consistency__"` and augmented-parent findings under `"__augmented__"`.

## Examples

### One process

```python
import bp_format as bp

collection = bp.serialization.load_process_collection("data.json")
process = collection.processes["run_1"]

ok, messages = bp.validate_process(process)
for msg in messages:
    print(("  " if ok else "  ! ") + msg)
```

### A whole case study

```python
ok, report = bp.validate_for_publication(collection)
if not ok:
    for key, messages in report.items():
        print(f"\n{key}:")
        for msg in messages:
            print(f"  - {msg}")
```

### Volume balance against a measured endpoint

```python
ok, msg, delta = bp.validate_volume_consistency(process, final_volume=1.85)
print(msg)           # full balance table
print(f"net change: {delta:+.3f} {process.volume.unit}")
```

## Common failures

| Message | Cause | Fix |
|---------|-------|-----|
| `FeedVolumeChange contains negative values` | Sign convention flipped | Feeds are ≥ 0, samples ≤ 0 |
| `does not contain a 'biomass' component` | Missing or renamed biomass | Rename it, or supply your own `biological_ode` |
| `times length does not match values length` | Arrays misaligned during parsing | Rebuild the `TimeSeries` |
| `missing feed components for state variable(s)` | Feed medium omits a reactor species | Add it, with the real concentration — `0.0` only if truly absent |
| `measurement at t=… is … after sampling` | Offline timestamp nudged past the sample | Snap it onto the sampling time |
| `derivatives missing entries for dynamic state(s)` | A state has no `d/dt` | Add an entry; `"0"` if there is no biology |
| `references undeclared symbol(s)` | Typo, or a symbol that is not a state / control / algebraic / rate | Declare it or fix the name |
| `combined additively with mismatched units` | Summing `g/L` with `mg/L` | Convert to one unit |

## See also

- [Data Model](02_data_model.md) — what is being validated
- [Serialization](03_serialization.md) — validate after loading
- [Design Rationale §6](01_design_rationale.md#6-check-the-data-then-fail-loudly)
