# BPbench: Bioprocess Benchmarking Dataset Structure

A JAX-compatible framework for standardized bioprocess data management and benchmarking across multiple case studies.

## Overview

BPbench provides a hierarchical data structure for organizing bioprocess experiments, enabling:
- **Unified benchmarking**: Compare modeling approaches across multiple case studies
- **JAX compatibility**: Full PyTree integration for automatic differentiation and JIT compilation
- **Flexible serialization**: YAML+HDF5 for efficiency or JSON for simplicity
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
 ├─ reactor_medium: Dict[str, ReactorMedium]
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
 │    └─ spline: Optional[SplineRepresentation]
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
 ├─ timepoints: jnp.ndarray
 └─ values: jnp.ndarray

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

## Quick Start

### Creating a Dataset

```python
import jax.numpy as jnp
from bpbench import (
    BenchmarkDataset, CaseStudy, BioProcess, BioProcessMetadata,
    TimeSeries, TimeAxis, ProcessVariable, ReactorMedium, ReactorMediumComponent,
    Volume, StaticVariable
)

# Create a simple batch process
time_axis = TimeAxis(
    unit="hours",
    start=0.0,
    end=48.0,
    time_reference="inoculation"
)

# Process variable with time series data
biomass = ProcessVariable(
    name="Biomass",
    unit="g/L",
    is_controlled=False,  # State variable (measured output)
    values=TimeSeries(
        timepoints=jnp.array([0., 12., 24., 36., 48.]),
        values=jnp.array([0.1, 1.2, 3.5, 5.8, 6.0])
    )
)

# Static process variable
temperature = ProcessVariable(
    name="Temperature",
    unit="°C",
    is_controlled=True,  # Control variable
    values=StaticVariable(value=37.0)
)

# Reactor medium with components
reactor = ReactorMedium(
    name="Batch medium",
    density=1.0,
    density_unit="kg/L",
    components={
        "glucose": ReactorMediumComponent(
            name="glucose",
            unit="g/L",
            concentration=TimeSeries(
                timepoints=jnp.array([0., 12., 24., 36., 48.]),
                values=jnp.array([20.0, 15.2, 8.5, 2.1, 0.1])
            ),
            is_intracellular=False
        )
    }
)

process = BioProcess(
    metadata=BioProcessMetadata(
        name="batch_001",
        process_type="batch",
        notes="Example batch process"
    ),
    time_axis=time_axis,
    process_variables={"biomass": biomass, "temperature": temperature},
    reactor_medium={"main": reactor},
    volume=Volume(
        initial_volume=1.0,
        unit="L"
    )
)

case_study = CaseStudy(
    case_id="ecoli_study_2024",
    organism="Escherichia coli",
    citation="Doe et al. 2024",
    processes={"batch_001": process}
)

dataset = BenchmarkDataset(
    metadata={
        "name": "Example Dataset",
        "version": "0.1.0",
        "description": "Example bioprocess dataset"
    },
    case_studies={"ecoli_study": case_study}
)
```

### Serialization

#### YAML + HDF5 (Recommended)

```python
from pathlib import Path
from bpbench import save_dataset, load_dataset

# Save
save_dataset(dataset, Path("data/my_dataset"))

# Load
loaded_dataset = load_dataset(Path("data/my_dataset"))
```

This creates two files:
- `metadata.yaml`: Human-readable structure
- `arrays.h5`: Efficient binary storage for arrays

#### JSON (Alternative)

```python
from bpbench import save_dataset_json, load_dataset_json

# Save
save_dataset_json(dataset, Path("data/my_dataset.json"))

# Load
loaded_dataset = load_dataset_json(Path("data/my_dataset.json"))
```

### Cross-Validation

```python
from bpbench import iter_loocv

# Iterate over all case studies with leave-one-out CV
for case_id, train_ids, test_id in iter_loocv(dataset):
    print(f"Case: {case_id}")
    print(f"  Train processes: {train_ids}")
    print(f"  Test process: {test_id}")
    
    # Access training data
    case_study = dataset.case_studies[case_id]
    train_processes = [case_study.processes[pid] for pid in train_ids]
    test_process = case_study.processes[test_id]
    
    # Your modeling code here...
```

### Inspecting BioProcess Structure

Use `print_structure()` to display a hierarchical view of a BioProcess object:

```python
from bpbench import print_structure, load_dataset

# Load a dataset
dataset = load_dataset(Path("data/my_dataset"))
process = dataset.case_studies["case1"].processes["proc1"]

# Display the complete structure
print_structure(process)

# Show sample data values
print_structure(process, show_values=True)
```

This displays:
- Process metadata (name, type, notes)
- Time axis information
- Process variables (states and controls)
- Reactor medium components
- Volume tracking and feed information

See `examples/demo_print_structure.py` for a complete demonstration.

## Key Features

### JAX Compatibility

All dataclasses are registered as PyTrees, enabling:

```python
import jax
from jax import grad, jit

def process_loss(process_params, process):
    # Your model here
    predictions = model(process_params, process)
    biomass_data = process.process_variables["biomass"].values.values
    return jnp.mean((predictions - biomass_data)**2)

# Automatic differentiation works!
loss_grad = grad(process_loss)

# JIT compilation works!
fast_loss = jit(process_loss)
```

### Time Series Support

TimeSeries and StaticVariable can be used for all measurements:

```python
# Time-varying measurement
biomass_ts = TimeSeries(
    timepoints=jnp.array([0., 12., 24., 36., 48.]),
    values=jnp.array([0.1, 1.2, 3.5, 5.8, 6.0])
)

# Constant value
temperature_static = StaticVariable(value=37.0)

# Use in ProcessVariable
biomass_var = ProcessVariable(
    name="Biomass",
    unit="g/L",
    is_controlled=False,
    values=biomass_ts
)

# Optional spline fitting (placeholder for future processing)
from bpbench import SplineRepresentation
# biomass_var.spline = SplineRepresentation(...)
```

### Feed Definitions

Support for complex feed media and volume changes:

```python
from bpbench import FeedMedium, FeedMediumComponent, VolumeChange

# Define feed medium composition
glucose_feed = FeedMedium(
    name="Glucose feed",
    density=1.1,
    density_unit="kg/L",
    components={
        "glucose": FeedMediumComponent(
            name="glucose",
            unit="g/L",
            concentration=StaticVariable(value=500.0),
            is_controlled=True
        ),
        "NH4Cl": FeedMediumComponent(
            name="NH4Cl",
            unit="g/L",
            concentration=StaticVariable(value=10.0),
            is_controlled=True
        )
    }
)

# Define volume change with feed
feed_volume_change = VolumeChange(
    name="Glucose feed",
    unit="L",
    is_controlled=True,
    is_continuous=True,
    feed_medium=glucose_feed,
    values=TimeSeries(
        timepoints=jnp.array([0., 12., 24., 36., 48.]),
        values=jnp.array([0., 0.05, 0.12, 0.22, 0.35])  # Cumulative volume
    )
)

# Add to volume tracking
volume = Volume(
    initial_volume=1.0,
    unit="L",
    volume_changes={"feed": feed_volume_change}
)
```

## Examples

The `examples/` directory contains sample implementations demonstrating how to use BPbench:

### Legacy Examples (00_old_examples, 01_prol_v1)
These directories contain older example implementations that demonstrate early versions of the data structure. They are kept for reference but may not reflect the current best practices.

### Current Example: PROL v2 (02_prol_v2)

The **examples/02_prol_v2/** directory contains a complete, documented example of loading experimental data into the BPbench format:

- **load_data.ipynb**: Jupyter notebook demonstrating the complete workflow:
  1. Loading CSV data (online measurements, offline measurements, discrete events)
  2. Creating BPbench data structures (TimeAxis, TimeSeries, Volume, FeedMedium, etc.)
  3. Assembling a complete BioProcess object for an E. coli fed-batch process
  4. Validating volume consistency
  5. Visualizing the data
  6. Saving to HDF5 format
  7. Loading and verifying the saved data

- **original_data/**: Contains sample CSV files from an E. coli fed-batch experiment for Protein L production
  - `260212_exp_onl.csv`: Online measurements (Temperature, Volume, Feed rate)
  - `260212_exp_off.csv`: Offline measurements (Biomass, Glycerol, Product)
  - `260212_exp_dsp.csv`: Discrete event times
  - `data_info.md`: Description of the data files

- **bpbench_format/**: Output directory containing the saved dataset
  - `metadata.yaml`: Human-readable structure
  - `arrays.h5`: Efficient binary storage for arrays

To run the example:
```bash
cd examples/02_prol_v2
jupyter notebook load_data.ipynb
```

## Module Structure

```
bpbench/
├── __init__.py          # Package initialization
├── dataclasses.py       # Core data structures
├── serialization.py     # Save/load functionality
└── utils.py             # Helper functions
```

## Requirements

- Python >= 3.8
- JAX >= 0.4.0
- NumPy >= 1.20.0
- PyYAML >= 5.4.0
- h5py >= 3.0.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Citation

If you use BPbench in your research, please cite:

```bibtex
@software{bpbench2024,
  title = {BPbench: Bioprocess Benchmarking Dataset Structure},
  author = {BPbench Contributors},
  year = {2024},
  url = {https://github.com/Gotsmy/BPbench}
}
```

## Acknowledgments

This framework is designed to support reproducible bioprocess modeling research and facilitate comparison of different modeling approaches across diverse datasets.
