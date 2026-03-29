# BPbench: Bioprocess Benchmarking Dataset Structure

A JAX-compatible framework for standardized bioprocess data management and benchmarking across multiple case studies.

## Overview

BPbench provides a hierarchical data structure for organizing bioprocess experiments, enabling:
- **Unified benchmarking**: Compare modeling approaches across multiple case studies
- **JAX compatibility**: Full PyTree integration for automatic differentiation and JIT compilation
- **Serialization**: JSON export for benchmark datasets and process collections
- **Cross-validation utilities**: Built-in support for leave-one-process-out validation

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

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
 │         └─ is_intracellular: bool
 ├─ process_variables: Dict[str, ProcessVariable]
 │    ├─ name: str
 │    ├─ unit: str
 │    ├─ is_controlled: bool  # True for controls, False for states
 │    ├─ values: TimeSeries | StaticVariable
 │    └─ interpolator: Optional[Interpolator]
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

## TimeSeries Migration

- Hard break applied: use `TimeSeries(times=..., values=...)` and `.times`.
- Legacy `timepoints` constructor/property are removed.
- Serialization is canonical-only (`times` + `values` for discrete payloads).
- Spline-only series in pseudobatch workflows currently use spline breakpoints
  as the fallback measurement grid when no discrete sample grid is provided.
  Provide explicit discrete samples when exact experimental sampling times are
  required.
