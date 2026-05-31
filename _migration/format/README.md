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

# Load a benchmark dataset from JSON
dataset = bp.serialization.load_dataset("examples/00_combined/01_combined_dataset/data.json")

# Explore the dataset
bp.print_dataset_structure(dataset, verbosity=1)

# Access a specific process
case_study = dataset.case_studies["kittler_2022"]
process = case_study.processes["batch_001"]

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
| [`dataclasses`](documentation/02_data_model.md) | Hierarchical data structures (BenchmarkDataset, CaseStudy, BioProcess, etc.) |
| [`time_series`](documentation/06_time_series.md) | Time-series container with optional fitted spline coefficients (JAX pytree) |
| [`splines`](documentation/07_splines.md) | Pseudobatch transformation and segmented spline fitting |
| [`mechanistic`](documentation/08_mechanistic.md) | Auto-generated ODE RHS, control splines, integration |
| [`serialization`](documentation/03_serialization.md) | JSON save/load for the full data hierarchy |
| [`validate`](documentation/04_validation.md) | Data integrity checks (9 validators) |
| [`inspect`](documentation/05_inspection.md) | Text printing and matplotlib visualization |

## Data Structure

### Hierarchy

```
BenchmarkDataset
 ├─ case_studies: Dict[str, CaseStudy]
 └─ metadata: Dict[str, str]

CaseStudy
 ├─ case_id: str
 ├─ organism: str
 ├─ citation: str
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
 │         ├─ concentration: TimeSeries | StaticVariable
 │         └─ c_star_concentration: Optional[TimeSeries | StaticVariable]
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
 ├─ times: jnp.ndarray
 ├─ values: jnp.ndarray
 ├─ breaks / coeffs / segment_start_piece_idx (optional spline state)
 └─ canonical API only (legacy `timepoints` removed)

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

bp-format is the data foundation for a planned ecosystem of bioprocess modeling packages:

| Package | Purpose | Status |
|---------|---------|--------|
| **bp-form** (current bp-format) | Data classes, I/O, validation, basic simulation | Active development |
| **bp-bench** | Pre-processed case study database | Planned |
| **bp-prep** | Web app for preprocessing raw data | Active development |
| **bp-train** | Training utilities (LOO-CV, augmentation) | Active development |
| **bp-sim** | Data generation with DoE support | Planned |
| **bp-design** | Post-training model-based DoE | Planned |
| **bp-control** | Post-training model-based MPC | Planned |

## Documentation

See the [full documentation](documentation/README.md) for detailed module guides, design rationale, and examples.
