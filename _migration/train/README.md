# BP-Train: Bioprocess Hybrid Model Training

Bioprocess hybrid model setup and training with JAX and Diffrax.

## Motivation

bp-train turns a [bp-format](../bp-format) bioprocess collection into a
trainable **hybrid ODE model**: it builds the mechanistic mass balance from the
process definition, lets you plug in your own neural / mechanistic **reaction
module** and **loss module**, and fits them with gradient-based optimization
through adaptive ODE integration. The ODE is integrated in a scaled (O(1)) state
space for numerical stability, solved once per sample for both the rates and the
loss, and trained with optax. A FAIR, self-contained run directory makes every
fit reproducible and resumable.

## Installation

```bash
pip install -e .
```

For development (matplotlib, plotly, pytest, …):
```bash
pip install -e ".[dev]"
```

## Quick Start

### CLI pipeline

```bash
# 1. transform a raw bp-format collection into a prepared dir
#    (writes prepared.json + prepare_config.json + prepare_diagnostics/ into it)
bp-train prepare --config prepare-config.json --output-dir prepared

# 2. fit reaction + loss modules into a run directory
bp-train train   --config train-config.json

# 3. re-simulate a trained model and optionally export predictions
#    (set output.predictions to parents or all; >1 model dir = ensemble)
bp-train forward --config forward-config.json

# 4. (optional) leave-one-process-out cross-validation
bp-train loo     --config loo-config.json
```

A `custom.py` next to the config supplies the hooks (`build_reaction_module`,
`build_loss_module`, `estimate_all_scales`, …); omit it to train the built-in
MLP reaction module with per-target MSE loss. See
[examples/00_e2e_sim/](examples/00_e2e_sim).

### Programmatic

```python
from bp_format.serialization import load_process_collection
from bp_train import model_load, model_predict, print_trainable_structure

# Load a trained model from its run directory. `config` carries the solver
# settings the model was fitted under, so prediction takes no solver arguments.
wrapper, config = model_load("examples/00_e2e_sim/output_all")
print_trainable_structure(wrapper.reaction_module)

# Forward-simulate. The collection may hold processes the model never trained on.
collection = load_process_collection("examples/00_e2e_sim/prepared/prepared.json")
predictions = model_predict(wrapper, config, collection)   # {name: DenseProcessExport}
```

`model_load` addresses a model by path — a run directory, a checkpoint directory,
or a `params.eqx`. To move between checkpoints of the *same* run without paying
for the dataset rebuild again, use `model_reload(path, wrapper)`, which returns the
same `(wrapper, config)` pair.

## Modules

| Module | Documentation |
|--------|---------------|
| CLI, run config & `custom.py` hooks | [documentation/02_cli_and_config.md](documentation/02_cli_and_config.md) |
| Data preparation, scales, layout | [documentation/03_data_preparation.md](documentation/03_data_preparation.md) |
| Reaction & loss modules | [documentation/04_reaction_and_loss.md](documentation/04_reaction_and_loss.md) |
| Training, forward & LOO | [documentation/05_train_forward_loo.md](documentation/05_train_forward_loo.md) |
| Serialization & inspection | [documentation/06_serialization_inspect.md](documentation/06_serialization_inspect.md) |

## Documentation

Full docs — including the design rationale and the **single `custom.py` hooks
reference** — are in [documentation/](documentation/README.md). Start with the
[Design Rationale](documentation/01_design_rationale.md).

[specs/](specs/README.md) holds proposals, roadmaps, and investigations. It is
not a description of current behaviour — several documents there predate the
current API.

bp-train builds on [bp-format](../bp-format) for the data model and the
mechanistic ODE RHS.
