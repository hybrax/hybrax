# BP-Format: Bioprocess Benchmarking Dataset Structure

A JAX-compatible framework for standardized bioprocess data management and benchmarking across multiple case studies.

## Motivation

Bioprocess modeling research lacks standardized data formats, making it difficult to compare modeling approaches across labs and publications. bp-format provides a common hierarchical data structure with built-in validation, serialization, and mechanistic modeling support. Built on JAX and Equinox, it enables automatic differentiation through ODE integration for gradient-based hybrid model training.

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import bp_format as bp

# Load a process collection from JSON
collection = bp.serialization.load_process_collection("data.json")

# Explore the collection
bp.print_collection_structure(collection, verbosity=1)

# Access a specific process
process = collection.processes["run_1"]

# Validate data integrity
is_valid, messages = bp.validate_process(process)

# Inspect a single process
bp.print_process_structure(process, verbosity=3)

# Plot
fig = bp.plot_process(process)
```

## Modules

| Module | Description |
|--------|-------------|
| [`dataclasses`](documentation/02_data_model.md) | Hierarchical data structures (BioProcessCollection, BioProcess, etc.) |
| [`time_series`](documentation/06_time_series.md) | Time-series container with optional fitted spline coefficients (JAX pytree) |
| [`splines`](documentation/07_splines.md) | Pseudobatch transformation and segmented spline fitting |
| [`mechanistic`](documentation/08_mechanistic.md) | ODE right-hand side and control splines (integration lives in bp-train) |
| [`serialization`](documentation/03_serialization.md) | JSON save/load for the full data hierarchy |
| [`validate`](documentation/04_validation.md) | Data integrity checks (14 validators) |
| [`inspect`](documentation/05_inspection.md) | Text printing and matplotlib visualization |
| [`simulation`](documentation/09_simulation.md) | Event bookkeeping for synthetic datasets |

## Data Structure

### Hierarchy

```
BioProcessCollection
 ├─ case_id: Optional[str]      set marks a published case study
 ├─ organism: Optional[str]
 ├─ citation: Optional[str]
 ├─ metadata: Optional[Dict]
 └─ processes: Dict[str, BioProcess]

BioProcess
 ├─ metadata: BioProcessMetadata
 │    ├─ name: str
 │    ├─ process_type: str (batch, fed_batch, continuous)
 │    └─ notes: Optional[str]
 ├─ time_axis: TimeAxis
 │    ├─ unit: str
 │    ├─ start: float
 │    ├─ end: float
 │    └─ time_reference: str
 ├─ reactor_medium: ReactorMedium
 │    ├─ name: str
 │    ├─ density: float
 │    ├─ density_unit: str
 │    └─ components: Dict[str, ReactorMediumComponent]
 │         ├─ name: str
 │         ├─ unit: str
 │         └─ concentration: TimeSeries | StaticVariable
 ├─ biological_ode: Optional[BiologicalOde]
 │    ├─ algebraic: Dict[str, str]
 │    ├─ rates: Dict[str, tuple[Optional[float], Optional[float]]]
 │    └─ derivatives: Dict[str, str]
 ├─ process_variables: Dict[str, ProcessVariable]
 │    ├─ name: str
 │    ├─ unit: str
 │    ├─ is_controlled: bool  # True for controls, False for states
 │    └─ values: TimeSeries | StaticVariable
 └─ volume: Volume
      ├─ initial_volume: float
      ├─ unit: str
      └─ volume_changes: Dict[str, VolumeChange]
           ├─ name: str
           ├─ unit: str
           ├─ is_controlled: bool
           ├─ is_continuous: bool
           ├─ feed_medium: FeedMedium
           └─ values: TimeSeries

TimeSeries
 ├─ times: jnp.ndarray | None
 ├─ values: jnp.ndarray | None
 └─ breaks / coeffs / segment_start_piece_idx (optional spline state)

StaticVariable
 └─ value: float

FeedMedium
 ├─ name: str
 ├─ density: float
 ├─ density_unit: str
 └─ components: Dict[str, FeedMediumComponent]
      ├─ name: str
      ├─ unit: str
      ├─ concentration: TimeSeries | StaticVariable
      └─ is_controlled: bool
```

## Ecosystem Context

bp-format is the data foundation the other packages build on:

| Package | Purpose | Status |
|---------|---------|--------|
| **bp-format** | Data model, I/O, validation, mechanistic RHS | Active development |
| **bp-train** | Hybrid ODE model training, LOO-CV, augmentation | Active development |
| **bp-bench** | Benchmark database of prepared case studies | Active development |

Further packages sketched in [specs/PRD.md](specs/PRD.md) are ideas, not code.

## Documentation

- [documentation/](documentation/README.md) — module guides and design rationale.
- [specs/](specs/README.md) — proposals, roadmaps, and design notes. Not a
  description of current behaviour.
