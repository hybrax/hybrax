# Validation

Source: `bp_format/validate.py`

## Purpose

Catch the errors that bioprocess data reliably contains — wrong signs, missing
species, misaligned timestamps — before they turn into silent numerical
nonsense during ODE integration.

Every validator returns `(bool, str)` (or `(bool, list[tuple[bool, str]])` for
the aggregates) instead of raising, so one pass collects **all** problems into
a report rather than stopping at the first. Structural impossibilities still
raise — see [Design Rationale §5](01_design_rationale.md#5-check-the-data-then-fail-loudly).

The message always follows one template: `"<VERDICT> <check_name>: <detail>"`,
where `VERDICT` is `PASS`, `FAIL`, or `SKIP` (a skipped check — one that does
not apply to this process — still counts as `ok=True`). `check_name` is the
validator's name minus its `validate_` prefix, e.g. `validate_bounds` reports
as `bounds`. This is built by one shared helper (`_check_result` in
`bp_format/validate.py`) so every check's message is unambiguous about what
was checked and why it failed, without having to parse prose.

All 20 validators are exported from the package root: `bp.validate_process(...)`.

Mapping keys are canonical identifiers. Embedded object names may instead be
human-facing or source labels, so validation does not require keys and names to
match.

## Individual validators

### `validate_discrete_events(process)`

When discrete events are present, requires their timestamps to be one-dimensional,
strictly increasing and unique, and within the inclusive process time axis.
Bounds comparisons allow a relative tolerance of `1e-7` to accommodate legacy
float32 timestamps widened during deserialization. Optional event labels must
have the same length as the timestamps.

`SKIP discrete_events: process.discrete_events is None — timing/label checks skipped`

### `validate_timeseries_shape(ts, name="", *, allow_empty=False)`

`times` and `values` are both 1-D, the same length, and `times` is strictly
increasing (no duplicates). Empty matching arrays are valid only when
`allow_empty=True`.

`validate_process` requires nonempty reactor-medium concentrations, process
variables, continuous volume changes, and measured total volume. Empty discrete
volume-change series are valid and mean that no events of that type occur in the
process.

`PASS timeseries_shape: TimeSeries 'biomass' — 1-D, equal-length, strictly increasing times`

### `validate_time_axis(process)`

Requires `process.time_axis.start <= process.time_axis.end`. Equal bounds are
valid.

`PASS time_axis: [0.0, 14.0] h`

### `validate_timestamp_bounds(process)`

Every timestamp falls inclusively between `process.time_axis.start` and
`process.time_axis.end`. This covers reactor-medium concentrations, process
variables, volume changes, and measured total volume. Bounds comparisons allow
`1e-7 * max(1, abs(start), abs(end))` to
accommodate legacy float32 timestamps widened during deserialization. Timestamps
are assumed to use `process.time_axis.unit`; no conversion is performed. This
per-process check is separate from cross-process time-axis consistency. An
inverted time axis skips this policy-dependent check; the preceding
`validate_time_axis` check reports the structural error.

`PASS timestamp_bounds: all timestamps within [0.0, 14.0] h`
`SKIP timestamp_bounds: time_axis is invalid — bounds check skipped`

### `validate_volume_change_sign(volume_change)`

- `Inflow`: all values ≥ 0 (inflow)
- `Outflow`: all values ≤ 0 (outflow)
- Unknown type: values must be purely positive or purely negative, never mixed

Uses a 1e-12 tolerance so exact zeros and float noise pass.

`FAIL volume_change_sign: 'feed' (Inflow) contains negative values; Inflows must have all values >= 0`

### `validate_volume_units(process)`

Every volume change uses exactly the same unit string as `process.volume.unit`.
Units are not parsed or converted, and no dimensional analysis is performed.

`PASS volume_units: volume changes use volume unit 'L'`

### `validate_volume_change_states(process)`

For every volume change that adds volume, checks that its `feed_medium` is
present and that each explicitly declared feed component corresponding to a
dynamic reactor-medium species uses the same unit string. Omitted reactor
components mean zero concentration in the feed and are valid. Units are compared
exactly; they are not parsed or converted, and no dimensional analysis is
performed. Feed concentrations may be static even when the reactor concentration
is dynamic.

`SKIP volume_change_states: no dynamic state variables found in reactor medium`

### `validate_outflow_retention(process)`

For every `Outflow`, requires each `retention` key to name a reactor-medium
component and each retention fraction to fall within the inclusive range
`[0.0, 1.0]`. A nonempty retention mapping is valid only for continuous
outflows; discrete outflows remove reactor medium through state jumps and do not
apply retention factors.

`PASS outflow_retention: all Outflow retention values are valid`

### `validate_biomass_in_reactor_medium(process)`

The reactor medium contains a component named `biomass` (case-insensitive).

This matters because the auto-generated `BiologicalOde` builds every derivative
as `q_<species> * biomass` and cannot do so without it. If your process has no
biomass component, supply your own `biological_ode` — the check will still
report, but the process is usable.

Biomass has **no reserved position** in the state vector; reactor components are
ordered alphabetically.

`PASS biomass_in_reactor_medium: biomass found in reactor medium as 'biomass'`

### `validate_initial_state_alignment(process)`

Every dynamic (`TimeSeries`-valued) reactor-medium component and process
variable must have a measurement exactly at `process.time_axis.start`.
`StaticVariable`-valued states are exempt (constant by definition, defined
everywhere).

This matters because `process.volume.initial_volume` is implicitly anchored
at `time_axis.start`, and bp-train seeds the ODE's initial condition from
whichever timestamp is the earliest one shared by every training target — not
necessarily `time_axis.start`. If a process's earliest actual measurement
comes later (e.g. an unsampled pre-culture period), training would silently
combine a state vector measured at that later point with a volume value that
was only ever valid at `time_axis.start`. This check catches that misalignment
at the data level, before it reaches training.

`PASS initial_state_alignment: all dynamic reactor-medium components and process variables have a measurement at time_axis.start=0, consistent with volume.initial_volume`

### `validate_measurement_sampling_alignment(process, rel_threshold=1e-4)`

Flags concentration measurements timestamped *slightly after* a sampling event —
strictly between the sample time and `sample_time + rel_threshold ·
process_length` (0.01 % of the run by default).

An offline measurement describes the broth **as drawn**, i.e. the pre-sample
state. A timestamp a few seconds late makes the pseudobatch transform pick the
post-sample volume, which corrupts the accumulated dilution factor and every
spline built on it. Move such timestamps onto the sampling time exactly.

Measurements *exactly at* a sampling time are correct and are not flagged.

`SKIP measurement_sampling_alignment: no sampling events found`

### `validate_bounds(process)`

Every `Bounds` tuple on the process — `ReactorMediumComponent.bounds`,
`ProcessVariable.bounds`, `Volume.bounds` — has `lower <= upper` when both are
set. (Bounds on `BiologicalOde.rates` are checked by `validate_biological_ode`.)
This checks the tuple itself, not the data — see `validate_bounds_against_data`
below for that.

`PASS bounds: lo <= hi holds for reactor components, process variables, volume`
`FAIL bounds: reactor component 'glucose' has lo=5.0 > hi=2.0 (lo must be <= hi)`

### `validate_bounds_against_data(process)`

For every `ReactorMediumComponent.concentration`, `ProcessVariable.values`, and
(when present) `Volume.total_volume` that carries a *set* `Bounds` tuple
(either side non-`None`), compares the actual measured value(s) — a scalar for
`StaticVariable`, the full array for `TimeSeries` — against `(lower, upper)`
and reports how many datapoints violate the bound, with the observed min/max.

`ReactorMediumComponent.bounds` defaults to `(0.0, None)` even when never set
explicitly, so this check catches negative concentrations by default, not just
on RMCs with an explicit bound. Out of scope: `BiologicalOde.rates` bounds — no
rate-inversion machinery exists in hybrax.format to compute a measured rate value
to check those against.

`PASS bounds_against_data: all measured datapoints with declared bounds fall within them`

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

`SKIP biological_ode: process.biological_ode is None — structural checks skipped`
`PASS biological_ode: derivatives/algebraic/rates parse, resolve, and are acyclic with consistent units`

### `validate_biological_ode_equivalence(container)`

Given a `BioProcessCollection`, checks that every process exposes an identical
`BiologicalOde` (same `algebraic`, `rates`, and `derivatives`). One model has to
describe every run in a study for the benchmark to mean anything. Containers
with 0 or 1 process pass trivially.

Not part of `validate_for_publication`; call it directly, or let
[`print_rhs_ode`](05_inspection.md) run it for you.

Returns `(bool, str)` — a single aggregate check, not a per-process list.
`PASS biological_ode_equivalence: identical across 3 processes`

### `validate_cross_process_consistency(collection)`

Given a `BioProcessCollection`, checks that every process shares identical
structure against the first process:

- same reactor components, each with the same value type and unit
- same process variables, each with the same value type and unit
- same volume unit
- same time-axis unit and time reference

Volume-change names may differ because processes in one case study can use
different feed and sampling strategies. Each process's volume-change units are
checked against its own volume unit by `validate_volume_units`.

All compared units and the time reference are exact strings. Units are not
parsed or converted, and no dimensional analysis is performed. Time-axis start
and end may differ.

Collections with 0 or 1 process pass trivially. This is the check
`validate_for_publication` composes to build its `"__consistency__"` report
entry; call it directly for just the structural-consistency signal.

Returns `(bool, list[tuple[bool, str]])`, always with at least one entry: one
`FAIL cross_process_consistency: ...` per mismatch, or a single
`PASS cross_process_consistency: ...` entry when consistent — never an empty
list.

### `validate_volume_consistency(process, final_volume)`

Initial volume plus all volume changes should land within 5 % of the measured
final volume. Returns `(bool, str, float)` — the third element is the net volume
change. This numeric check assumes compatible units; call `validate_volume_units`
first (or use `validate_process`) to check exact unit-string coherence.

Continuous changes contribute `values[-1] - values[0]` (cumulative); discrete
changes contribute `sum(values)`. Only the endpoints are used, because this runs
*before* any spline fitting.

> `final_volume` has no usable default — pass the measured value.

The message is still the full multi-line balance table, now prefixed with the
verdict and check name, e.g. `PASS volume_consistency: \n  Initial volume ...`.

### `validate_augmented_parent_refs(container)`

For every `AugmentedBioProcess` in a `BioProcessCollection`: `parent_process`
names a key in the same container, and that key resolves to a
**non-augmented** `BioProcess`. Chained augmentation is rejected.

Returns `(bool, list[tuple[bool, str]])`, always with at least one entry: one
`PASS augmented_parent_refs: N augmented process(es) have valid parent references`
summary entry when there are no problems, or one `FAIL augmented_parent_refs: ...`
entry per problem found. Called automatically by `validate_for_publication`.

## Aggregate validators

### `validate_process(process) -> (bool, list[tuple[bool, str]])`

Runs, in order:

1. `validate_discrete_events`
2. `validate_timeseries_shape` on reactor components, process variables, volume
   changes, and measured total volume carrying a `TimeSeries`
3. `validate_time_axis`
4. `validate_timestamp_bounds`
5. `validate_volume_units`
6. `validate_volume_change_sign` on every volume change
7. `validate_volume_change_states`
8. `validate_outflow_retention`
9. `validate_biomass_in_reactor_medium`
10. `validate_initial_state_alignment`
11. `validate_measurement_sampling_alignment`
12. `validate_bounds`
13. `validate_bounds_against_data`
14. `validate_biological_ode`

Returns one `(check_ok, message)` pair per check — including the passing ones,
so the output reads as a checklist and callers can filter to just the failures
with `[msg for ok, msg in results if not ok]` instead of parsing message text.
Raises `TypeError` if given something that is not a `BioProcess`. Required
structure such as `BioProcess.volume` is enforced when a process is created or
deserialized; missing required fields raise rather than becoming validation
results.

`validate_volume_consistency` is **not** included (it needs a `final_volume`
argument you have to supply).

### `validate_for_publication(collection) -> (bool, dict[str, list[tuple[bool, str]]])`

`validate_process` on every process, plus `validate_cross_process_consistency`
and `validate_augmented_parent_refs`. This is hybrax.format's own concern — is this
collection well-formed and internally coherent enough to store or publish as a
case study — distinct from bp-train's training-readiness concern
(`bp_train.validate.validate_for_training`, which composes the same
`validate_cross_process_consistency` check rather than duplicating it).

Results are keyed by process name, with cross-process findings under
`"__consistency__"` and augmented-parent findings under `"__augmented__"` —
both always non-empty (at least one `PASS` entry on a clean collection).

## Examples

### One process

```python
import bp_format as bp

collection = bp.serialization.load_process_collection("data.json")
process = collection.processes["run_1"]

ok, results = bp.validate_process(process)
for check_ok, msg in results:
    print(("  " if check_ok else "  ! ") + msg)
```

### A whole case study

```python
ok, report = bp.validate_for_publication(collection)
if not ok:
    for key, results in report.items():
        print(f"\n{key}:")
        for check_ok, msg in results:
            if not check_ok:
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
| `FAIL volume_change_sign: ... Inflow contains negative values` | Sign convention flipped | Feeds are ≥ 0, samples ≤ 0 |
| `FAIL biomass_in_reactor_medium: ... does not contain a 'biomass' component` | Missing or renamed biomass | Rename it, or supply your own `biological_ode` |
| `FAIL timeseries_shape: ... times length does not match values length` | Arrays misaligned during parsing | Rebuild the `TimeSeries` |
| `FAIL measurement_sampling_alignment: ... measurement at t=… is … after sampling` | Offline timestamp nudged past the sample | Snap it onto the sampling time |
| `FAIL initial_state_alignment: no measurement at time_axis.start=... for: ...` | A dynamic state's first measurement is later than `time_axis.start` | Add a `t=time_axis.start` measurement, or mark the variable a `StaticVariable` |
| `FAIL biological_ode: ... derivatives missing entries for dynamic state(s)` | A state has no `d/dt` | Add an entry; `"0"` if there is no biology |
| `FAIL biological_ode: ... references undeclared symbol(s)` | Typo, or a symbol that is not a state / control / algebraic / rate | Declare it or fix the name |
| `FAIL biological_ode: ... combined additively with mismatched units` | Summing `g/L` with `mg/L` | Convert to one unit |

## See also

- [Data Model](02_data_model.md) — what is being validated
- [Serialization](03_serialization.md) — validate after loading
- [Design Rationale §5](01_design_rationale.md#5-check-the-data-then-fail-loudly)
