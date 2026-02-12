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
 └─ processes: Dict[str, Process]

Process
 ├─ process_id: str
 ├─ process_type: str (batch, fed_batch, continuous)
 ├─ time: TimeAxis
 ├─ dynamic_states: Dict[str, TimeSeries]  # Biomass, substrate, product
 ├─ dynamic_controls: Dict[str, TimeSeries]  # pH, temperature (if varying)
 ├─ static_controls: Dict[str, StaticVariable]  # Temperature (if constant)
 ├─ volume: Volume  # Special handling for volume tracking
 │    └─ volume_changes: Dict[str, VolumeChange]  # Feed, sampling, etc.
 ├─ feeds: Dict[str, Feed]
 ├─ static_parameters: Dict[str, StaticVariable]
 └─ reactor: ReactorProperties

Volume
 ├─ volume_changes: Dict[str, VolumeChange]
 ├─ initial_volume: float
 └─ volume_unit: str

VolumeChange
 ├─ name: str
 ├─ controlled: bool  # True if controlled, False if modeled
 ├─ continuous: bool  # True if continuous, False if discrete
 ├─ unit: str
 ├─ feed_medium: Optional[str]  # Reference to feed name
 ├─ timeseries: Optional[TimeSeries]  # For continuous changes
 ├─ timepoints: Optional[ndarray]  # For discrete changes
 └─ values: Optional[ndarray]  # For discrete changes
```

## Quick Start

### Creating a Dataset

```python
import jax.numpy as jnp
from bpbench import (
    BenchmarkDataset, CaseStudy, Process,
    TimeSeries, RawTimeSeries, TimeAxis,
    ReactorProperties, StaticVariable
)

# Create a simple batch process
time_axis = TimeAxis(
    unit="hours",
    start=0.0,
    end=48.0,
    time_reference="inoculation"
)

biomass = TimeSeries(
    name="Biomass",
    canonical_name="biomass",
    unit="g/L",
    role="state",
    raw=RawTimeSeries(
        timepoints=jnp.array([0., 12., 24., 36., 48.]),
        values=jnp.array([0.1, 1.2, 3.5, 5.8, 6.0])
    )
)

process = Process(
    process_id="batch_001",
    process_type="batch",
    time=time_axis,
    states={"biomass": biomass},
    reactor=ReactorProperties(
        working_volume=1.0,
        volume_unit="L"
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

## Key Features

### JAX Compatibility

All dataclasses are registered as PyTrees, enabling:

```python
import jax
from jax import grad, jit

def process_loss(process_params, process):
    # Your model here
    predictions = model(process_params, process)
    return jnp.mean((predictions - process.states["biomass"].raw.values)**2)

# Automatic differentiation works!
loss_grad = grad(process_loss)

# JIT compilation works!
fast_loss = jit(process_loss)
```

### Time Series Support

- **Raw data**: Experimental measurements with uncertainty
- **Spline representations**: Fitted curves (cubic hermite, linear, zero-order hold)
- **Event times**: Discontinuities for ODE solvers

```python
from bpbench import SplineRepresentation

spline = SplineRepresentation(
    type="cubic_hermite",
    breakpoints=jnp.array([0., 12., 24., 36., 48.]),
    coefficients=jnp.array([[...]]),  # Spline coefficients
    discontinuous=False,
    fit_residual_std=0.05
)

biomass.spline = spline
```

### Feed Definitions

Support for complex feed media:

```python
from bpbench import Feed, FeedComponent

glucose_feed = Feed(
    name="Glucose feed",
    density=1.1,
    density_unit="kg/L",
    components={
        "glucose": FeedComponent(concentration=500.0, unit="g/L"),
        "NH4Cl": FeedComponent(concentration=10.0, unit="g/L")
    }
)

process.feeds["feed1"] = glucose_feed
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
