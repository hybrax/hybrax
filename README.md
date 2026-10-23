# Hybrax

Bioprocess hybrid models and data management in JAX.

Hybrid models, also known as universal differential equations (UDEs) or
grey-box models, embed learned components such as neural networks in a
mechanistic ODE.

Hybrax combines bioprocess data interchange and hybrid model training in one
package. `hybrax.format` provides the data model, validation, serialization,
and mechanistic right-hand side. `hybrax.train` provides preparation,
augmentation, training, prediction, and leave-one-out cross-validation.

Read the [documentation](https://hybrax.readthedocs.io) for tutorials, gallery
examples, and API reference.

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

| Module | Description |
|--------|-------------|
| `hybrax.format.dataclasses` | Hierarchical data structures (`BioProcessCollection`, `BioProcess`, etc.) |
| `hybrax.format.time_series` | Time-series container with optional fitted spline coefficients (JAX pytree) |
| `hybrax.format.splines` | Pseudobatch transform, segmented spline fitting |
| `hybrax.format.mechanistic` | ODE right-hand side and control splines (integration lives in `hybrax.train`) |
| `hybrax.format.serialization` | JSON save/load for the full data hierarchy |
| `hybrax.format.validate` | Data integrity checks |
| `hybrax.format.inspect` | Text printing and matplotlib visualization |
| `hybrax.format.simulation` | Event bookkeeping for synthetic datasets |
| `hybrax.train` | Preparation, hybrid model training, prediction, and cross-validation |

See the [documentation](https://hybrax.readthedocs.io) for the data structure
and the training workflow.
