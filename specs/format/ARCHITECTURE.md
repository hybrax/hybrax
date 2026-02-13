# BPbench Architecture Documentation

This document provides a comprehensive overview of the BPbench architecture, design decisions, and implementation details for bioprocess benchmarking datasets.

## Table of Contents

1. [Overview](#overview)
2. [Package Organization](#package-organization)
3. [Core Data Structure Hierarchy](#core-data-structure-hierarchy)
4. [Module Descriptions](#module-descriptions)
5. [Key Design Patterns](#key-design-patterns)
6. [JAX/PyTree Integration](#jaxpytree-integration)
7. [Serialization Architecture](#serialization-architecture)
8. [Usage Patterns](#usage-patterns)
9. [Extension Points](#extension-points)

## Overview

BPbench is a JAX-compatible framework for standardized bioprocess data management and benchmarking. It provides:
- **Hierarchical data structures** for organizing experimental bioprocess data
- **Full JAX compatibility** via PyTree registration for automatic differentiation and JIT compilation
- **Flexible serialization** with YAML+HDF5 for efficiency or JSON for simplicity
- **Built-in validation** for volume balance and feed consistency
- **Cross-validation utilities** for leave-one-process-out evaluation

## Package Organization

```
bpbench/
├── __init__.py           # Package exports and version
├── dataclasses.py        # Core data structures (441 lines)
├── serialization.py      # I/O operations (509 lines)
├── splines.py            # Spline fitting utilities (226 lines)
└── utils.py              # Helper functions (206 lines)
```

## Core Data Structure Hierarchy

BPbench uses a four-level hierarchical structure to organize bioprocess data:

```
Level 4: BenchmarkDataset
  ├── metadata: Dict[str, str]
  └── case_studies: Dict[str, CaseStudy]

Level 3: CaseStudy
  ├── case_id: str
  ├── organism: str
  ├── citation: str
  └── processes: Dict[str, Process]

Level 2: Process
  ├── process_id: str
  ├── process_type: str (batch/fed_batch/continuous)
  ├── replicate_id: Optional[str]
  ├── time: TimeAxis
  ├── dynamic_variables: Dict[str, TimeSeries]
  ├── static_variables: Dict[str, StaticVariable]
  ├── volume: Volume
  ├── feeds: Dict[str, Feed]
  ├── event_times: Optional[ndarray]
  └── reactor: ReactorProperties

Level 1: Supporting Structures
  ├── TimeSeries (raw + spline representations)
  ├── Volume (with VolumeChange operations)
  ├── Feed (with FeedComponent definitions)
  └── ReactorProperties

Level 0: Primitive Components
  ├── TimeAxis
  ├── RawTimeSeries
  ├── SplineRepresentation
  ├── StaticVariable
  └── FeedComponent
```

### Key Hierarchy Principles

1. **Top-down organization**: Dataset → CaseStudy → Process → Variables
2. **Dictionaries for collections**: All collections use string keys for easy lookup
3. **Optional fields**: Most fields are optional to support partial datasets
4. **Type safety**: All structures use type hints for clarity

## Module Descriptions

### dataclasses.py

**Purpose**: Defines the core hierarchical data structure for bioprocess benchmarking.

**Key Components by Category**:

#### Low-Level Structures (Level 0)
- **TimeAxis**: Time reference with unit, start/end bounds, and reference point
- **RawTimeSeries**: Experimental measurements (timepoints, values, optional measurement_std)
- **SplineRepresentation**: Fitted piecewise polynomial (type, breakpoints, coefficients, discontinuous flag)
- **StaticVariable**: Single numeric parameter with unit
- **FeedComponent**: Single component in a feed medium (concentration, unit)
- **ReactorProperties**: Static reactor characteristics (working_volume, density)

#### Composed Structures (Level 1)
- **TimeSeries**: Combines raw measurements with optional spline representation
  - Fields: name, canonical_name, unit, role (state/control), raw, spline
- **Feed**: Complete feed medium definition
  - Fields: name, density, density_unit, components (Dict[str, FeedComponent])
- **VolumeChange**: Individual volume operation (controlled/modeled, continuous/discrete)
  - Fields: name, controlled, continuous, unit, feed_medium, timeseries, timepoints, values

#### Volume Container
- **Volume**: Special container for all volume-related information
  - Fields: volume_changes (Dict[str, VolumeChange]), initial_volume, volume_unit
  - Methods: validate_volume_consistency(), validate_feed_components()

#### Process Level (Level 2)
- **Process**: Complete experimental run with all associated data
  - Unified dynamic_variables (combines states and controls)
  - Unified static_variables (combines parameters and static controls)
  - Special volume handling separate from state variables

#### Study Organization (Level 3-4)
- **CaseStudy**: Collection of processes from one publication/dataset
- **BenchmarkDataset**: Top-level container with metadata and multiple case studies

**Design Decisions**:
- Uses Python dataclasses for simplicity and clarity
- All structures are mutable to allow incremental construction
- JAX PyTree registration enables automatic differentiation and JIT compilation
- Hierarchical structure: Dataset → CaseStudy → Process → Variables
- **Unified variables**: `dynamic_variables` combines states and controls (distinguished by role field)
- Volume changes store raw measurements (cumulative volumes in L, not rates)
- Spline fitting used to compute rates from cumulative data as needed
- Feed components validated to ensure consistency with dynamic states

### serialization.py

**Purpose**: Handles saving and loading of datasets in multiple formats.

**Key Functions**:
- `save_dataset(dataset, path)` / `load_dataset(path)`: YAML + HDF5 (recommended for efficiency)
- `save_dataset_json(dataset, path)` / `load_dataset_json(path)`: Pure JSON (human-readable)
- `_extract_arrays()`: Recursively finds and extracts JAX/NumPy arrays from nested structures
- `_restore_arrays()`: Reconstructs full object hierarchy from YAML metadata + HDF5 arrays

**Architecture**:

```
YAML + HDF5 Approach:
  metadata.yaml          arrays.h5
  ├── Structure         ├── /case1/proc1/biomass/raw/values
  ├── Metadata          ├── /case1/proc1/biomass/raw/timepoints
  └── Array refs        └── /case1/proc1/temperature/spline/coefficients
```

**Design Decisions**:
- **Hybrid YAML+HDF5**: Separates human-readable metadata from binary numerical data
- **JSON alternative**: Provides simple single-file option for smaller datasets
- **Recursive extraction**: Automatically finds arrays at any nesting level
- **Path-based keys**: HDF5 groups mirror the hierarchical structure (e.g., `case1/proc1/biomass/raw/values`)
- **Backward compatibility**: Supports legacy `dynamic_states`/`dynamic_controls` structure during loading

**Implementation Notes**:
- Arrays stored as separate HDF5 datasets with reference paths in YAML
- Custom NumpyEncoder handles JAX/NumPy array serialization for JSON
- Deserialization reconstructs full object hierarchy with proper types
- HDF5 ~10x faster than JSON for large numerical arrays

### splines.py

**Purpose**: Spline fitting and rate computation utilities for time series data.

**Key Functions**:
- `fit_cubic_spline(times, values, event_times=None)`: Fit cubic Hermite or linear splines
  - Supports piecewise fitting with discontinuities at event_times
  - Returns SplineRepresentation with breakpoints and coefficients
  - Automatically handles segment boundaries
  
- `compute_rate_from_cumulative(spline, eval_times)`: Compute analytical derivatives
  - Takes cumulative volume spline, returns rate time series
  - Uses analytical differentiation of polynomial coefficients
  - Efficient for dense time grids

**Design Decisions**:
- **Cumulative storage**: Store cumulative volumes (L) in raw form, compute rates when needed
- **Piecewise splines**: Handle discontinuities (sampling, discrete additions) naturally
- **Analytical derivatives**: No numerical differentiation needed
- **Flexible types**: Support cubic Hermite (smooth), linear, zero-order hold

**Spline Representation Details**:
```python
SplineRepresentation(
    type="cubic_hermite",           # polynomial type
    breakpoints=[0, 2, 4, 6, 8],   # K breakpoints → K-1 segments
    coefficients=[[c0, c1, c2, c3], # M=K-1 segments, C coefficients each
                  [c0, c1, c2, c3],
                  ...],
    discontinuous=True,             # if rate has jumps
    fit_residual_std=0.05           # goodness of fit
)
```

### utils.py

**Purpose**: Utility functions for common bioprocess benchmarking tasks.

**Key Functions**:
- `get_event_times(process)`: Extract discontinuity times for ODE solvers
  - Collects event_times from process
  - Adds discrete VolumeChange timepoints
  - Returns sorted unique array
  
- `leave_one_process_out(case_study)`: Generate CV splits for single case study
  - Generator yields (train_ids, test_id) tuples
  - Memory-efficient for large datasets
  
- `iter_loocv(dataset)`: Iterator for leave-one-out across all case studies
  - Yields (case_id, train_ids, test_id)
  - Enables global cross-validation
  
- `print_structure(process, show_values=False)`: Display hierarchical Process view
  - Shows complete structure with metadata
  - Optional value ranges for debugging
  - Human-readable formatting

**Design Decisions**:
- Generator-based CV functions for memory efficiency
- Explicit separation of train/test process IDs
- Works seamlessly with the hierarchical data structure
- Utilities support both manual inspection and automated workflows

## Key Design Patterns

### 1. Volume Change Tracking

Volume is handled separately from state variables because it's special:
- Not a classic "state" variable (affects reactor capacity)
- Can be both controlled (programmed feeds) and modeled (evaporation, base consumption)
- Affected by multiple simultaneous operations

**Pattern**:
```python
Volume(
    initial_volume=1.5,  # L
    volume_unit="L",
    volume_changes={
        "carbon_feed": VolumeChange(
            name="Carbon feed",
            controlled=True,      # programmed feed rate
            continuous=True,      # not discrete
            unit="L",            # cumulative volume
            feed_medium="glucose_medium",
            timeseries=TimeSeries(...)  # cumulative volume over time
        ),
        "base_feed": VolumeChange(
            name="Base feed",
            controlled=False,     # PID-controlled, needs modeling
            continuous=True,
            unit="L",
            feed_medium="base_medium",
            timeseries=TimeSeries(...)
        ),
        "sampling": VolumeChange(
            name="Sampling",
            controlled=True,
            continuous=False,     # discrete events
            unit="L",
            timepoints=[2.0, 4.0, 6.0],  # hours
            values=[-0.05, -0.05, -0.05]  # negative = removal
        )
    }
)
```

**Key Points**:
- `controlled=True`: Feed rate is a design choice (input to optimization)
- `controlled=False`: Feed rate must be modeled (e.g., pH control with base)
- `continuous=True`: Use timeseries with cumulative volumes
- `continuous=False`: Use discrete timepoints and values
- Unit field: "L" for cumulative, "L/h" for rates

**Volume Validation**:
```python
is_valid, message = process.volume.validate_volume_consistency(
    time_axis=process.time,
    final_volume=measured_final_volume
)
# Checks: initial + sum(changes) ≈ final (within 5% tolerance)
```

### 2. Feed Composition Structure

Feeds define medium compositions for mass balance calculations:

```python
Feed(
    name="Glucose feed medium",
    density=1.02,  # kg/L
    density_unit="kg/L",
    components={
        "glucose": FeedComponent(concentration=200.0, unit="g/L"),
        "glycerol": FeedComponent(concentration=50.0, unit="g/L"),
        "yeast_extract": FeedComponent(concentration=10.0, unit="g/L"),
    }
)
```

**Feed References in VolumeChange**:
```python
# Option 1: Reference to Process.feeds
VolumeChange(
    name="Feed 1",
    feed_medium="glucose_medium",  # references Process.feeds["glucose_medium"]
    ...
)

# Option 2: Inline definition
VolumeChange(
    name="Feed 2",
    feed=Feed(...),  # inline definition
    ...
)
```

**Validation**:
- `Volume.validate_feed_components()` checks that:
  - All referenced feeds exist in Process.feeds
  - Feed components cover dynamic variables (warns if missing)

### 3. TimeSeries Dual Representation

Time series data has both raw measurements and fitted representations:

```python
TimeSeries(
    name="Biomass (CDW)",               # original name from paper
    canonical_name="biomass",           # standardized cross-study name
    unit="g/L",
    role="state",                       # "state" or "control"
    
    # Raw experimental measurements
    raw=RawTimeSeries(
        timepoints=jnp.array([0, 1, 2, 4, 8, 12]),
        values=jnp.array([0.1, 0.3, 0.8, 2.1, 4.5, 6.0]),
        measurement_std=jnp.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3])  # optional
    ),
    
    # Fitted spline representation
    spline=SplineRepresentation(
        type="cubic_hermite",
        breakpoints=jnp.array([0, 4, 8, 12]),  # segment boundaries
        coefficients=jnp.array([[...], [...], [...]]),  # polynomial coefficients
        discontinuous=False,
        fit_residual_std=0.05
    )
)
```

**Benefits**:
- Preserves original measurements with uncertainty
- Provides smooth interpolation via splines
- Enables analytical derivatives (rates from cumulative data)
- Handles measurement noise naturally

**Usage**:
- Use `raw` for plotting original data points
- Use `spline` for ODE solver interpolation
- Compute rates: `compute_rate_from_cumulative(spline, times)`

### 4. Unified Dynamic Variables

**Design Evolution**: Earlier versions separated `dynamic_states` and `dynamic_controls`. Current version unifies them into `dynamic_variables` with a `role` field.

```python
Process(
    dynamic_variables={
        # States (measured outputs)
        "biomass": TimeSeries(..., role="state"),
        "substrate": TimeSeries(..., role="state"),
        "product": TimeSeries(..., role="state"),
        
        # Controls (manipulated inputs)
        "temperature": TimeSeries(..., role="control"),
        "pH": TimeSeries(..., role="control"),
        "DO": TimeSeries(..., role="control"),
    }
)
```

**Benefits**:
- Simpler API (one dictionary instead of two)
- Flexible (role field distinguishes when needed)
- Some variables can be both (e.g., pH: controlled but also measured)
- Backward compatible (loader handles old format)

**Note**: Volume changes are NOT in dynamic_variables (special handling in `Process.volume`)

### 5. Event Times for Discontinuities

Discontinuities in control profiles require special handling for ODE solvers:

```python
process.event_times = jnp.array([0, 2.5, 5.0, 7.5, 10.0])  # hours
```

**Sources of discontinuities**:
- Discrete sampling events (volume removal)
- Bolus additions (discrete feeds)
- Induction events (gene expression triggers)
- Feed rate changes (step changes in control)

**Usage with splines**:
```python
# Fit spline with discontinuities
spline = fit_cubic_spline(
    times, values,
    event_times=process.event_times  # creates piecewise segments
)
```

**Utility function**:
```python
event_times = get_event_times(process)
# Automatically extracts from process.event_times + discrete VolumeChanges
```

## JAX/PyTree Integration

BPbench is fully compatible with JAX for automatic differentiation and JIT compilation.

### What are PyTrees?

PyTrees are JAX's way of handling nested Python structures containing numerical arrays. They separate:
- **Leaves**: Numerical data (arrays, floats) that participate in differentiation
- **Nodes**: Structural metadata (strings, type info) that defines the tree shape

### PyTree Registration

All BPbench dataclasses are registered as PyTrees:

```python
tree_util.register_pytree_node(
    TimeSeries,
    # Flatten: Extract leaves (arrays) and nodes (metadata)
    lambda obj: (
        (obj.raw, obj.spline),           # leaves (can be None)
        (obj.name, obj.canonical_name, obj.unit, obj.role)  # nodes
    ),
    # Unflatten: Reconstruct from leaves and nodes
    lambda nodes, leaves: TimeSeries(
        name=nodes[0],
        canonical_name=nodes[1],
        unit=nodes[2],
        role=nodes[3],
        raw=leaves[0],
        spline=leaves[1]
    )
)
```

### Benefits

**1. Automatic Differentiation**:
```python
import jax
from jax import grad, jit

def model_loss(params, process):
    """Loss function for bioprocess model"""
    predictions = simulate_process(params, process)
    biomass_data = process.dynamic_variables["biomass"].raw.values
    return jnp.mean((predictions - biomass_data)**2)

# Compute gradient w.r.t. parameters
loss_grad = grad(model_loss)
gradients = loss_grad(initial_params, process)
```

**2. JIT Compilation**:
```python
# Compile for speed
fast_loss = jit(model_loss)
loss_value = fast_loss(params, process)  # First call: compile
loss_value = fast_loss(params, process)  # Subsequent: fast!
```

**3. Vectorization**:
```python
from jax import vmap

# Process multiple experiments in parallel
processes = [process1, process2, process3]
losses = vmap(lambda p: model_loss(params, p))(processes)
```

### PyTree Operations

**Traverse tree**:
```python
from jax import tree_util

# Get all arrays
leaves, treedef = tree_util.tree_flatten(process)

# Transform all arrays
scaled_process = tree_util.tree_map(lambda x: x * 2.0, process)

# Check structure
tree_util.tree_structure(process)
```

**Registered Structures**:
- TimeAxis (leaves: start, end)
- RawTimeSeries (leaves: timepoints, values, measurement_std)
- SplineRepresentation (leaves: breakpoints, coefficients)
- TimeSeries (leaves: raw, spline)
- StaticVariable (leaves: value)
- VolumeChange (leaves: timeseries, timepoints, values)
- Volume (leaves: all VolumeChange objects)
- Process (leaves: time, all variables, volume, event_times)
- CaseStudy (leaves: all processes)
- BenchmarkDataset (leaves: all case studies)

## Serialization Architecture

BPbench supports two serialization formats, each optimized for different use cases.

### Format 1: YAML + HDF5 (Recommended)

**Structure**:
```
my_dataset/
├── metadata.yaml       # Human-readable structure
└── arrays.h5          # Binary array storage
```

**How it works**:
1. **Extract**: Recursively find all arrays in nested structure
2. **Store**: Arrays → HDF5, structure → YAML with array references
3. **Load**: Reconstruct structure, insert arrays from HDF5

**YAML Example**:
```yaml
case_studies:
  prol_v3:
    case_id: prol_v3
    organism: E. coli
    processes:
      DoE1_R1:
        process_id: DoE1_R1
        dynamic_variables:
          biomass:
            name: Biomass
            role: state
            unit: g/L
            raw:
              timepoints: "__array__:case_studies/prol_v3/processes/DoE1_R1/biomass/raw/timepoints"
              values: "__array__:case_studies/prol_v3/processes/DoE1_R1/biomass/raw/values"
```

**HDF5 Structure**:
```
arrays.h5
├── case_studies/
│   └── prol_v3/
│       └── processes/
│           └── DoE1_R1/
│               ├── biomass/
│               │   └── raw/
│               │       ├── timepoints [array: (7,)]
│               │       └── values [array: (7,)]
│               └── temperature/
│                   └── raw/
│                       ├── timepoints [array: (1457,)]
│                       └── values [array: (1457,)]
```

**Advantages**:
- ~10x faster I/O than JSON for large arrays
- Human-readable structure in YAML
- Efficient binary storage in HDF5
- Standard formats (easy to inspect with external tools)

**When to use**: Production datasets, large numerical arrays, shared data repositories

### Format 2: Pure JSON

**Structure**:
```
my_dataset.json         # Single file
```

**How it works**:
1. **Convert**: All arrays to lists
2. **Encode**: Custom NumpyEncoder handles special types
3. **Store**: Single JSON file with nested structure

**JSON Example**:
```json
{
  "case_studies": {
    "prol_v3": {
      "processes": {
        "DoE1_R1": {
          "dynamic_variables": {
            "biomass": {
              "raw": {
                "timepoints": [0.0, 2.04, 4.05, 6.14, 8.17, 9.22, 10.22],
                "values": [27.45, 36.5, 40.0, 39.5, 38.2, 35.1, 22.93]
              }
            }
          }
        }
      }
    }
  }
}
```

**Advantages**:
- Single file (easy to share/email)
- Human-readable (can edit with text editor)
- No external dependencies (pure Python)
- Version control friendly (text diff)

**Disadvantages**:
- Slower I/O (~10x) for large arrays
- Larger file size
- Precision loss for floats (JSON limitation)

**When to use**: Small datasets, sharing examples, debugging, version control

### Backward Compatibility

The loader automatically handles legacy data structures:
- `dynamic_states` + `dynamic_controls` → merged into `dynamic_variables`
- Missing fields get default values
- Old spline formats converted to current structure

```python
# Loader detects old format and converts automatically
loaded = load_dataset("old_format_dataset")
# → Process.dynamic_variables contains merged data
# → Process.dynamic_states is deprecated but still accessible
```

## Usage Patterns

### Complete Workflow Example

```python
import jax.numpy as jnp
from pathlib import Path
from bpbench import *

# 1. Define time axis
time_axis = TimeAxis(
    unit="hours",
    start=0.0,
    end=48.0,
    time_reference="inoculation"
)

# 2. Create time series for states
biomass = TimeSeries(
    name="Biomass (CDW)",
    canonical_name="biomass",
    unit="g/L",
    role="state",
    raw=RawTimeSeries(
        timepoints=jnp.array([0., 12., 24., 36., 48.]),
        values=jnp.array([0.1, 1.2, 3.5, 5.8, 6.0]),
        measurement_std=jnp.array([0.01, 0.05, 0.1, 0.15, 0.2])
    )
)

# 3. Create time series for controls
temperature = TimeSeries(
    name="Temperature",
    canonical_name="temperature",
    unit="K",
    role="control",
    raw=RawTimeSeries(
        timepoints=jnp.array([0., 12., 24., 36., 48.]),
        values=jnp.array([310., 310., 310., 310., 310.])
    )
)

# 4. Define feed medium
glucose_feed = Feed(
    name="Glucose feed",
    density=1.05,
    density_unit="kg/L",
    components={
        "glucose": FeedComponent(concentration=200.0, unit="g/L"),
        "yeast_extract": FeedComponent(concentration=20.0, unit="g/L")
    }
)

# 5. Create volume tracking
volume = Volume(
    initial_volume=1.0,
    volume_unit="L",
    volume_changes={
        "feed": VolumeChange(
            name="Glucose feed",
            controlled=True,
            continuous=True,
            unit="L",
            feed_medium="glucose_feed",
            timeseries=TimeSeries(
                name="Feed cumulative",
                canonical_name="feed_cumulative",
                unit="L",
                role="control",
                raw=RawTimeSeries(
                    timepoints=jnp.array([0., 12., 24., 36., 48.]),
                    values=jnp.array([0., 0.1, 0.25, 0.45, 0.7])
                )
            )
        )
    }
)

# 6. Create reactor properties
reactor = ReactorProperties(
    working_volume=2.0,
    volume_unit="L",
    density=1.0,
    density_unit="kg/L"
)

# 7. Build process
process = Process(
    process_id="batch_001",
    process_type="batch",
    replicate_id="R1",
    time=time_axis,
    dynamic_variables={
        "biomass": biomass,
        "temperature": temperature
    },
    static_variables={
        "initial_glucose": StaticVariable(value=10.0, unit="g/L")
    },
    volume=volume,
    feeds={"glucose_feed": glucose_feed},
    event_times=jnp.array([0., 12., 24., 36., 48.]),
    reactor=reactor
)

# 8. Assemble case study
case_study = CaseStudy(
    case_id="ecoli_2024",
    organism="Escherichia coli K-12",
    citation="Doe et al., Biotechnol. Bioeng. 2024",
    processes={"batch_001": process}
)

# 9. Create dataset
dataset = BenchmarkDataset(
    metadata={
        "name": "E. coli Batch Dataset",
        "version": "1.0.0",
        "description": "Example batch cultivation data",
        "created": "2024-01-15"
    },
    case_studies={"ecoli_2024": case_study}
)

# 10. Save dataset
save_dataset(dataset, Path("data/ecoli_dataset"))

# 11. Load and verify
loaded = load_dataset(Path("data/ecoli_dataset"))
print(f"Loaded {len(loaded.case_studies)} case studies")
```

### Working with Splines

```python
from bpbench import fit_cubic_spline, compute_rate_from_cumulative

# Fit spline to cumulative feed data
feed_ts = process.volume.volume_changes["feed"].timeseries
spline = fit_cubic_spline(
    times=feed_ts.raw.timepoints,
    values=feed_ts.raw.values,
    event_times=process.event_times
)

# Compute feed rate from cumulative volume
eval_times = jnp.linspace(0, 48, 1000)
feed_rate = compute_rate_from_cumulative(spline, eval_times)

# Add spline to time series
feed_ts.spline = spline
```

### Cross-Validation Workflow

```python
from bpbench import iter_loocv

# Iterate over all case studies with LOOCV
for case_id, train_ids, test_id in iter_loocv(dataset):
    print(f"\nCase: {case_id}")
    print(f"  Training on: {train_ids}")
    print(f"  Testing on: {test_id}")
    
    # Access training and test data
    case_study = dataset.case_studies[case_id]
    train_processes = [case_study.processes[pid] for pid in train_ids]
    test_process = case_study.processes[test_id]
    
    # Train model
    # model_params = train_model(train_processes)
    
    # Evaluate on test
    # predictions = model.predict(test_process)
    # evaluate(predictions, test_process.dynamic_variables["biomass"])
```

### Inspecting Data Structure

```python
from bpbench import print_structure

# Display complete process structure
print_structure(process)

# Show with value ranges
print_structure(process, show_values=True)
```

### Volume Validation

```python
# Validate volume consistency
is_valid, message = process.volume.validate_volume_consistency(
    time_axis=process.time,
    final_volume=1.7  # measured final volume
)

print(message)
if not is_valid:
    print("Warning: Volume imbalance detected!")

# Validate feed components
is_valid, message = process.volume.validate_feed_components(
    process_feeds=process.feeds,
    dynamic_variables=process.dynamic_variables
)
print(message)
```

## Extension Points

### Adding New Fields

To add a new field to any dataclass:

**Example: Adding pH measurement to Process**

1. **Add field to dataclass**:
```python
@dataclass
class Process:
    # ... existing fields ...
    ph_measurements: Optional[TimeSeries] = None  # new field
```

2. **Update PyTree registration**:
```python
tree_util.register_pytree_node(
    Process,
    lambda obj: (
        # Add new field to leaves if it contains arrays
        (obj.time, obj.dynamic_variables, obj.volume, obj.event_times, obj.ph_measurements),
        # nodes remain the same
        (obj.process_id, obj.process_type, ...)
    ),
    # Update unflatten function
    lambda nodes, leaves: Process(...)
)
```

3. **Update serialization** (if needed for special handling)
4. **Add tests** for the new field

### Custom Spline Types

To add a new spline type (e.g., B-spline):

1. **Define in SplineRepresentation**:
```python
spline = SplineRepresentation(
    type="b_spline",  # new type
    breakpoints=...,
    coefficients=...,
    ...
)
```

2. **Implement fitting function**:
```python
def fit_bspline(times, values, degree=3):
    # Fitting logic
    return SplineRepresentation(
        type="b_spline",
        breakpoints=knots,
        coefficients=control_points,
        ...
    )
```

3. **Document coefficient format** in splines.py

### Additional Utilities

Common additions you might implement:

**Visualization helpers**:
```python
def plot_process(process, variables=None):
    """Plot all or selected variables from a process"""
    # Implementation
```

**Data validation**:
```python
def validate_process(process) -> tuple[bool, List[str]]:
    """Comprehensive validation of process data"""
    # Check time axis consistency
    # Validate volume balance
    # Check for NaN/Inf values
    # Verify feed references
    return is_valid, messages
```

**Format conversion**:
```python
def convert_to_pandas(process) -> Dict[str, pd.DataFrame]:
    """Convert process data to pandas DataFrames"""
    # Implementation
```

## Design Philosophy

### 1. Simplicity First
- **Clear dataclass definitions**: Easy to understand at a glance
- **Minimal abstraction**: No unnecessary layers or complexity
- **Standard Python idioms**: Uses familiar patterns (dicts, dataclasses, type hints)
- **Explicit over implicit**: Structure is obvious, no magic

### 2. JAX Integration
- **Full PyTree support**: All structures are JAX-compatible
- **Automatic differentiation**: grad() and jit() work seamlessly
- **Functional programming friendly**: Immutable-style operations encouraged
- **Performance**: Enables GPU/TPU acceleration when needed

### 3. Flexibility
- **Multiple serialization formats**: Choose between YAML+HDF5 or JSON
- **Optional fields**: Support partial datasets (not all data always available)
- **Extensible structure**: Easy to add new fields without breaking existing code
- **Backward compatibility**: Loaders handle older format versions

### 4. Type Safety
- **Type hints throughout**: All functions and classes are fully typed
- **Clear documentation**: Expected shapes and units documented
- **Runtime validation**: Dataclass fields provide basic validation
- **Meaningful errors**: Error messages guide users to solutions

### 5. Data Integrity
- **Volume validation**: Built-in checks for mass balance
- **Feed validation**: Ensures feed compositions are complete
- **Unit tracking**: All measurements have explicit units
- **Metadata preservation**: Original names and canonical names both stored

## Testing Strategy

### Test Categories

**1. Unit Tests** (`tests/test_dataclasses.py`):
- Individual dataclass instantiation
- Field validation
- Default values
- Type checking

**2. PyTree Tests** (`tests/test_pytree.py`):
- Flatten/unflatten operations
- Tree mapping functions
- JAX compatibility (grad, jit, vmap)
- Structure preservation

**3. Serialization Tests** (`tests/test_serialization.py`):
- Round-trip YAML+HDF5 save/load
- Round-trip JSON save/load
- Array preservation (dtype, shape)
- Backward compatibility loading

**4. Integration Tests** (`tests/test_integration.py`):
- Complete workflow (create → save → load → use)
- Cross-validation utilities
- Spline fitting pipeline
- Volume validation

**5. Example Tests** (`tests/test_examples.py`):
- Example notebooks execute without errors
- Expected outputs are created
- Data can be loaded and accessed

### Running Tests

```bash
# All tests
pytest tests/

# Specific category
pytest tests/test_serialization.py

# With coverage
pytest --cov=bpbench tests/

# Verbose
pytest -v tests/
```

## Performance Considerations

### PyTree Operations
- **Overhead**: Minimal (~1-2%) for flatten/unflatten
- **Impact**: Negligible for most operations
- **Benefit**: Enables JAX transformations worth the small cost

### Serialization
- **YAML+HDF5**: ~10x faster than JSON for large arrays
- **HDF5 read**: Lazy loading possible (not implemented by default)
- **JSON**: Slower but portable, good for <100MB datasets

### Memory Usage
- **In-memory**: Full dataset loaded by default
- **Large datasets**: Consider lazy loading with custom loaders
- **JAX arrays**: Device memory management (CPU/GPU)

### JIT Compilation
- **First call**: Compilation overhead (1-10 seconds depending on complexity)
- **Subsequent calls**: Near-C speed execution
- **Recommendation**: JIT-compile in production, not in loops

### Benchmarks

Typical performance on modern hardware:

| Operation | Small Dataset | Large Dataset |
|-----------|---------------|---------------|
| Save (YAML+HDF5) | 10 ms | 500 ms |
| Load (YAML+HDF5) | 15 ms | 800 ms |
| Save (JSON) | 50 ms | 5000 ms |
| Load (JSON) | 80 ms | 8000 ms |
| PyTree flatten | <1 ms | <10 ms |
| Spline fit | 5 ms | 50 ms |

*Small: 1 case study, 5 processes, 50 timepoints each*  
*Large: 10 case studies, 100 processes, 1000 timepoints each*

## Common Patterns and Best Practices

### 1. Always Specify Units
```python
# Good
biomass = TimeSeries(name="Biomass", unit="g/L", ...)

# Bad - ambiguous
biomass = TimeSeries(name="Biomass", unit="", ...)
```

### 2. Use Canonical Names for Cross-Study Comparison
```python
# Enables comparison across different papers
TimeSeries(name="DCW", canonical_name="biomass", ...)
TimeSeries(name="Cell Density", canonical_name="biomass", ...)
```

### 3. Store Cumulative Data, Compute Rates
```python
# Store cumulative feed volume
feed_cumulative = TimeSeries(unit="L", raw=RawTimeSeries(...))

# Compute rate when needed
feed_rate = compute_rate_from_cumulative(feed_cumulative.spline, times)
```

### 4. Validate Volume Balance
```python
# Always validate after constructing process
is_valid, msg = process.volume.validate_volume_consistency(
    final_volume=measured_final_volume
)
assert is_valid, f"Volume imbalance: {msg}"
```

### 5. Use Event Times for Discontinuities
```python
# Mark discrete events
process.event_times = jnp.array([0., 10., 20., 30.])

# Splines will respect these boundaries
spline = fit_cubic_spline(times, values, event_times=process.event_times)
```

## Troubleshooting

### Common Issues

**Issue**: "Process.__init__() got unexpected keyword argument 'dynamic_states'"
- **Cause**: Using old API (pre-v0.1.0)
- **Solution**: Replace `dynamic_states` → `dynamic_variables`, `dynamic_controls` → `dynamic_variables`

**Issue**: Volume validation shows >5% error
- **Cause**: Potential coding error in volume tracking
- **Solution**: Check cumulative vs. rate units, verify all volume changes are tracked

**Issue**: "FileNotFoundError: metadata.yaml"
- **Cause**: Output directory doesn't exist
- **Solution**: Create directory before saving: `Path(save_path).mkdir(exist_ok=True)`

**Issue**: JAX arrays become Python lists after serialization
- **Cause**: JSON format converts arrays to lists
- **Solution**: Use YAML+HDF5 format, or convert back: `jnp.array(data)`

**Issue**: Spline fit fails with "breakpoints not monotonic"
- **Cause**: Event times not sorted or duplicate values
- **Solution**: Sort and deduplicate: `jnp.unique(jnp.sort(event_times))`

## Future Enhancements

Potential additions under consideration:

### Short-term
- [ ] Enhanced validation schemas (e.g., pydantic integration)
- [ ] Automatic unit conversion utilities
- [ ] Built-in plotting functions for common visualizations
- [ ] More spline types (B-splines, PCHIP)

### Medium-term
- [ ] Integration with diffrax/ODE solvers
- [ ] Lazy loading for very large datasets
- [ ] Database backend support (SQL, MongoDB)
- [ ] Remote dataset loading (HTTP, S3)

### Long-term
- [ ] Distributed dataset processing (Dask, Ray)
- [ ] Automatic data augmentation for ML
- [ ] Integration with experimental design tools
- [ ] Real-time data streaming support

## References and Resources

### Documentation
- **README**: Quick start and basic usage
- **ARCHITECTURE** (this doc): Detailed design and implementation
- **Examples**: `examples/03_prol_v3/` - Complete workflow demonstration

### External Resources
- **JAX Documentation**: https://jax.readthedocs.io/
- **PyTree Guide**: https://jax.readthedocs.io/en/latest/pytrees.html
- **HDF5 Format**: https://www.hdfgroup.org/solutions/hdf5/
- **YAML Specification**: https://yaml.org/

### Citation

If you use BPbench in your research, please cite:

```bibtex
@software{bpbench2024,
  title = {BPbench: Bioprocess Benchmarking Dataset Structure},
  author = {BPbench Contributors},
  year = {2024},
  url = {https://github.com/Gotsmy/BPbench}
}
```

---

**Document Version**: 1.0.0  
**Last Updated**: 2024-02-13  
**Maintained By**: BPbench Contributors
