# bp-format Documentation

bp-format is a **data model for bioprocess experiments**. It gives fermentation
and cell-culture runs one shape: a Python object hierarchy that describes what
was in the reactor, what was fed, what was sampled, and what was measured — plus
the tools to check that description, save it to JSON, and turn it into a
differentiable ODE right-hand side.

It does not train models and it does not integrate ODEs. That is
[bp-train](../../bp-train/documentation/README.md), which builds on bp-format.

## Getting Started

Read in this order:

1. [Design Rationale](01_design_rationale.md) — the "why" behind the framework.
2. [Data Model](02_data_model.md) — the dataclasses you build and read.
3. [Serialization](03_serialization.md) — save and load datasets as JSON.
4. [Validation](04_validation.md) — check data before modeling it.
5. [Inspection](05_inspection.md) — print and plot what you have.

Building models? Continue with:

6. [TimeSeries](06_time_series.md) — the measurement container.
7. [Splines](07_splines.md) — pseudobatch transform and spline fitting.
8. [Mechanistic](08_mechanistic.md) — building the ODE right-hand side.
9. [Simulation](09_simulation.md) — helpers for generating synthetic ground truth.

## Module Reference

| Module | Source | Docs | What it does |
|--------|--------|------|--------------|
| Data model | `bp_format/dataclasses.py` | [02](02_data_model.md) | The dataclass hierarchy: `CaseStudy` → `BioProcess` → components |
| TimeSeries | `bp_format/time_series/` | [06](06_time_series.md) | Measurements + optional fitted spline, as a JAX pytree |
| Splines | `bp_format/splines.py` | [07](07_splines.md) | Pseudobatch transform, segmented spline fitting, backtransform |
| Mechanistic | `bp_format/mechanistic.py` | [08](08_mechanistic.md) | `ProcessOrdering`, `ControlSplines`, `RhsOde` |
| Serialization | `bp_format/serialization.py` | [03](03_serialization.md) | JSON save/load for the whole hierarchy |
| Validation | `bp_format/validate.py` | [04](04_validation.md) | 12 integrity checks |
| Inspection | `bp_format/inspect.py` | [05](05_inspection.md) | Text summaries and matplotlib plots |
| Simulation | `bp_format/simulation.py` | [09](09_simulation.md) | Event bookkeeping for synthetic datasets |

## Data Structure

```
CaseStudy                        one publication or campaign
 ├─ case_id / organism / citation
 └─ processes: Dict[str, BioProcess]

BioProcessCollection             loose alternative to CaseStudy
 ├─ metadata: Optional[Dict]
 └─ processes: Dict[str, BioProcess]

BioProcess                       one experimental run
 ├─ metadata: BioProcessMetadata          name, process_type, notes
 ├─ time_axis: TimeAxis                   unit, start, end, time_reference
 ├─ reactor_medium: ReactorMedium         what is in the reactor
 │    └─ components: Dict[str, ReactorMediumComponent]
 │         ├─ concentration          TimeSeries | StaticVariable   (real, measured)
 │         ├─ c_star_concentration   optional pseudobatch trace
 │         └─ bounds                 optional (lo, hi) metadata
 ├─ volume: Volume                        every volume-changing operation
 │    ├─ initial_volume / unit / bounds
 │    ├─ total_volume: Optional[TimeSeries]
 │    └─ volume_changes: Dict[str, FeedVolumeChange | SampleVolumeChange]
 ├─ process_variables: Dict[str, ProcessVariable]    pH, temperature, DO, off-gas
 │    └─ is_controlled: bool              True = known input, False = modeled state
 ├─ biological_ode: BiologicalOde         dc/dt expressions (auto-filled if omitted)
 ├─ pseudobatch_transform: Optional[PseudobatchTransform]
 └─ discrete_events: Optional[DiscreteEvents]

AugmentedBioProcess(BioProcess)  synthetic sibling of a real run
 └─ parent_process: str          key of its parent in the same container
```

Full field lists are in the [Data Model](02_data_model.md).

## A typical workflow

```python
import bp_format as bp

# 1. load
collection = bp.serialization.load_process_collection("data.json")
process = collection.processes["run_1"]

# 2. check
ok, messages = bp.validate_process(process)

# 3. look
bp.print_process_structure(process, verbosity=2)
bp.print_rhs_ode(collection)

# 4. build the pseudobatch transform (optional, fed-batch)
process.pseudobatch_transform = bp.splines.build_pseudobatch_transform(process)

# 5. build the ODE pieces — bp-train integrates them
ordering = bp.mechanistic.get_process_ordering(process)
rhs_ode  = bp.mechanistic.build_rhs_ode(process, ordering)
controls = bp.mechanistic.get_control_splines(process, ordering)
```

## See also

- [bp-train documentation](../../bp-train/documentation/README.md) — trains
  hybrid ODE models on bp-format collections.
- [specs/](../specs/README.md) — proposals, roadmaps, and design notes. Not a
  description of current behaviour.
