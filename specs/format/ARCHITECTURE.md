# BPbench Module Structure

This document provides an overview of the BPbench module organization and design decisions.

## Package Organization

```
bpbench/
├── __init__.py           # Package exports and version
├── dataclasses.py        # Core data structures (400+ lines)
├── serialization.py      # I/O operations (300+ lines)
└── utils.py             # Helper functions (50+ lines)
```

## Module Descriptions

### dataclasses.py

**Purpose**: Defines the core hierarchical data structure for bioprocess benchmarking.

**Key Components**:
- **Low-level structures**: TimeAxis, RawTimeSeries, SplineRepresentation, TimeSeries
- **Medium-level structures**: StaticVariable, FeedComponent, Feed, ReactorProperties
- **High-level structures**: Process, CaseStudy, BenchmarkDataset
- **PyTree registration**: All dataclasses registered for JAX compatibility

**Design Decisions**:
- Uses Python dataclasses for simplicity and clarity
- All structures are immutable by design (frozen could be added if needed)
- JAX PyTree registration enables automatic differentiation and JIT compilation
- Hierarchical structure: Dataset → CaseStudy → Process → TimeSeries

### serialization.py

**Purpose**: Handles saving and loading of datasets in multiple formats.

**Key Functions**:
- `save_dataset()` / `load_dataset()`: YAML + HDF5 (recommended for efficiency)
- `save_dataset_json()` / `load_dataset_json()`: Pure JSON (human-readable)

**Design Decisions**:
- Hybrid YAML+HDF5 approach separates metadata from numerical data
- JSON option provides simple alternative for smaller datasets
- Recursive array extraction/restoration for nested structures
- HDF5 groups handle nested paths automatically

**Implementation Notes**:
- Arrays are stored as separate HDF5 datasets with reference paths in YAML
- Deserialization reconstructs full object hierarchy
- Custom NumpyEncoder handles JAX/NumPy array serialization for JSON

### utils.py

**Purpose**: Utility functions for common bioprocess benchmarking tasks.

**Key Functions**:
- `get_event_times()`: Extract discontinuity times for ODE solvers
- `leave_one_process_out()`: Generate cross-validation splits for single case study
- `iter_loocv()`: Iterator for leave-one-out across all case studies

**Design Decisions**:
- Generator-based CV functions for memory efficiency
- Explicit separation of train/test process IDs
- Works seamlessly with the hierarchical data structure

## Design Philosophy

### 1. **Simplicity First**
- Clear, readable dataclass definitions
- Minimal abstraction layers
- Standard Python idioms

### 2. **JAX Integration**
- All structures are PyTrees
- Supports automatic differentiation
- JIT compilation compatible
- Functional programming friendly

### 3. **Flexibility**
- Multiple serialization formats
- Optional fields for partial datasets
- Extensible structure (easy to add fields)

### 4. **Type Safety**
- Type hints throughout
- Clear documentation of expected shapes
- Runtime validation via dataclass fields

## Usage Patterns

### Creating a Dataset

```python
from bpbench import *

# 1. Define time axis
time = TimeAxis("hours", 0.0, 48.0, "inoculation")

# 2. Create time series
biomass = TimeSeries(
    "Biomass", "biomass", "g/L", "state",
    raw=RawTimeSeries(timepoints, values)
)

# 3. Build process
process = Process(
    "batch_001", "batch",
    time=time, dynamic_states={"biomass": biomass}
)

# 4. Assemble case study
case = CaseStudy("study1", "E. coli", "Citation", {"p1": process})

# 5. Create dataset
dataset = BenchmarkDataset(
    metadata={"name": "My Dataset"},
    case_studies={"cs1": case}
)
```

### Serialization

```python
# Efficient: YAML + HDF5
save_dataset(dataset, Path("data/my_dataset"))
loaded = load_dataset(Path("data/my_dataset"))

# Simple: JSON
save_dataset_json(dataset, Path("data/my_dataset.json"))
loaded = load_dataset_json(Path("data/my_dataset.json"))
```

### Cross-Validation

```python
# Iterate over all CV folds
for case_id, train_ids, test_id in iter_loocv(dataset):
    train_processes = [case.processes[pid] for pid in train_ids]
    test_process = case.processes[test_id]
    # Train and evaluate model...
```

## Extension Points

### Adding New Fields

To add a new field to any dataclass:
1. Add field to dataclass definition
2. Update PyTree registration if field contains arrays
3. Update serialization/deserialization functions
4. Add tests for the new field

### Custom Spline Types

Add new spline types by:
1. Adding type name to SplineRepresentation
2. Implementing fitting function
3. Documenting coefficient format

### Additional Utilities

Common additions:
- Data visualization helpers
- Statistical analysis functions
- Data validation routines
- Format conversion tools

## Testing Strategy

- **Unit tests**: Each dataclass and function
- **Integration tests**: Serialization round-trips
- **JAX compatibility tests**: PyTree operations
- **Example validation**: Full workflow test

All tests use pytest and are located in `tests/` directory.

## Performance Considerations

- **PyTree overhead**: Minimal, ~1-2% for flatten/unflatten
- **HDF5 I/O**: ~10x faster than JSON for large arrays
- **Memory**: Lazy loading possible with custom loaders
- **JIT compilation**: First call compiles, subsequent calls are fast

## Future Enhancements

Potential additions:
- Data validation schemas
- Automatic unit conversion
- Integration with diffrax/ODE solvers
- Plotting utilities
- Database backend support
- Distributed dataset loading
