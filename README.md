# Hybrax

A JAX-compatible framework for standardized bioprocess data management,
mechanistic modeling, and hybrid ODE training.

Hybrax combines bioprocess data interchange and hybrid ODE training in one
package. `hybrax.format` provides the data model, validation, serialization,
and mechanistic right-hand side. `hybrax.train` provides preparation,
augmentation, training, prediction, and leave-one-out cross-validation.

## Motivation

Bioprocess modeling research lacks standardized data formats, making it
difficult to compare modeling approaches across labs and publications.
`hybrax.format` provides a common hierarchical data structure with
built-in validation, serialization, and mechanistic modeling support.
Built on JAX and Equinox, it enables automatic differentiation through
ODE integration for gradient-based hybrid model training.

## Installation

For local development, create the uv environment with the development extras:

```bash
uv sync --extra dev
```

Run tests with `uv run pytest`. Add `-n 12` to use 12 pytest-xdist workers.

## Quick start

```python
import hybrax.format as bp

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

| Module | Docs | Description |
|--------|------|-------------|
| `hybrax.format.dataclasses` | [specs/format/02](specs/format/02_data_model.md) | Hierarchical data structures (`BioProcessCollection`, `BioProcess`, etc.) |
| `hybrax.format.time_series` | [specs/format/06](specs/format/06_time_series.md) | Time-series container with optional fitted spline coefficients (JAX pytree) |
| `hybrax.format.splines` | [specs/format/07](specs/format/07_splines.md) | Pseudobatch transform, segmented spline fitting |
| `hybrax.format.mechanistic` | [specs/format/08](specs/format/08_mechanistic.md) | ODE right-hand side and control splines (integration lives in `hybrax.train`) |
| `hybrax.format.serialization` | [specs/format/03](specs/format/03_serialization.md) | JSON save/load for the full data hierarchy |
| `hybrax.format.validate` | [specs/format/04](specs/format/04_validation.md) | Data integrity checks |
| `hybrax.format.inspect` | [specs/format/05](specs/format/05_inspection.md) | Text printing and matplotlib visualization |
| `hybrax.format.simulation` | [specs/format/09](specs/format/09_simulation.md) | Event bookkeeping for synthetic datasets |
| `hybrax.train` | [specs/train](specs/train/README.md) | Preparation, hybrid ODE training, prediction, and cross-validation |

See the [`hybrax.format` documentation](specs/format/README.md) for the data
structure and the [`hybrax.train` documentation](specs/train/README.md) for the
training workflow.
