# Serialization & Inspection

Source: [`bp_train/serialization.py`](../bp_train/serialization.py),
[`bp_train/inspect.py`](../bp_train/inspect.py),
[`bp_train/checkpointing.py`](../bp_train/checkpointing.py)

## Purpose

How a trained model is saved, reloaded, and verified for provenance — and the
tooling to inspect which leaves are trainable and how the reaction axes map to
state/rate indices.

## Design Rationale

**Trainable-partition-only persistence.** `save_model` writes *only* the
trainable partition (`params.eqx`). The static half (controls store, `RhsOde`,
indices, `SCALE_*`) is **always rebuilt** from `prepared.json` + `custom.py` at
load time via the single `reconstruct_run` path that training itself uses. This
keeps checkpoints small, sidesteps controls-store shape mismatches, and is
forward-compatible with trainable controls. Every checkpoint dir is
self-contained (it bundles `prepared.json.gz` and `custom.py`). See
[01_design_rationale.md](01_design_rationale.md#8-trainable-partition-only-serialization).

## Public API

### Persistence

```python
save_model(wrapper, path) -> None                    # writes trainable partition (params.eqx)
load_trained_wrapper(path, *, template) -> wrapper   # loads params into a structural template
save_opt_state(opt_state, path) -> None              # optimizer state (resume)
load_opt_state(path, *, template) -> opt_state
```

`load_trained_wrapper` / `load_opt_state` need a structurally identical
`template` (eqx raises on a pytree mismatch) — the template comes from
rebuilding the static half.

### Reconstruction & resumption

```python
load_run(run_dir, *, checkpoint="best", load_opt_state=False) -> LoadedRun
load_params(run_dir, *, into, checkpoint="latest") -> wrapper
reconstruct_run(run_dir, config, document=None) -> (reaction_module, loss_module, store, collection)
```

- `load_run` reconstructs a trained model from a run directory **alone** — the
  one call notebooks/forward/resume use. `checkpoint` is `"best"` | `"latest"` |
  `"step_00300"` (under `checkpoints/`) or `"final"` (the run-root `model/`
  copy). Returns a [`LoadedRun`](../bp_train/serialization.py):

  | Field | Meaning |
  |---|---|
  | `wrapper` | the reconstructed `HybridOdeWrapper` |
  | `collection` | the rebuilt bp-format collection |
  | `store` | the `TrainingDataStore` |
  | `config` | the resolved `RunConfig` |
  | `run_dir` | the source directory |
  | `opt_state` | optimizer state (only if `load_opt_state=True`) |

  `LoadedRun.reload(checkpoint="latest")` refreshes just the weights from another
  checkpoint into the existing wrapper.

- `load_params` refreshes weights into an **already-built** wrapper (no
  dataset/`custom.py` reload).
- `reconstruct_run` is THE single reconstruction path (forward, resume,
  `load_run` all call it): it verifies the prepared `content_hash` against
  `config.json`, then rebuilds `(reaction_module, loss_module, store,
  collection)` exactly as training did.

### Provenance

```python
content_hash(collection) -> str      # sha256 of a collection's canonical content
file_hash(path) -> str               # sha256 of a file
environment_versions() -> dict[str, str]   # JAX / Diffrax / bp-format / … versions
```

These are recorded in the run's `config.json` and the prepared artifact so a
reload can detect a data/code mismatch and fail fast.

## Introspection

Print the two structure tables (the harness prints both at training start):

```python
from bp_train import print_trainable_structure, print_reaction_schema

print_trainable_structure(wrapper.reaction_module, title="UserReactionModule")
print_trainable_structure(wrapper.loss_module, title="UserLossModule")
print_reaction_schema(wrapper)
```

- `format_trainable_structure(module, *, color=False, title=…)` /
  `print_trainable_structure(module, *, color=None, title=…)` — a `(name, shape,
  status)` table where `status` is `trainable` or `frozen`; submodules render as
  `(Module)` rows with nested fields. `color=None` auto-detects a TTY and tints
  trainable rows red. Use it to confirm exactly which leaves the optimizer
  updates.
- `format_reaction_schema(rhs_ode, controls)` /
  `print_reaction_schema(wrapper)` — labeled tables of the `ReactionInputs` and
  `ReactionOutputs` axes (names sourced from `wrapper.rhs_ode` and
  `wrapper.controls`), so you can map each rate/state index to its biological
  name when writing a [reaction module](04_reaction_and_loss.md#the-reaction-module).

## Examples

```python
from bp_train import load_run, print_trainable_structure

run = load_run("examples/00_e2e_sim/output", checkpoint="latest")
print_trainable_structure(run.wrapper.reaction_module)

# re-simulate with the loaded wrapper (see 05_train_forward_loo.md)
from bp_train import forward_from_collection
result = forward_from_collection(run.collection, model_path="examples/00_e2e_sim/output")
```
