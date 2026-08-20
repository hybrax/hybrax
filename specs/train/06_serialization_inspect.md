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
load time via the single `reconstruct_training` path that training itself uses. This
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

### Reconstruction

```python
model_load(path) -> (trained_wrapper, config)
model_reload(path, trained_wrapper) -> (trained_wrapper, config)
model_predict(trained_wrapper, config, collection, *, process_names=None, grid_n=200)
    -> {process_name: DenseProcessExport}
reconstruct_training(run_dir, config=None, document=None, *, custom_module=None,
                     custom_py=None, training_process_names=None) -> ReconstructedTraining
reconstruct_run(run_dir, config, document=None) -> (reaction_module, loss_module, store, collection)
```

**Addressing a model.** All three take a **path**, not a run dir plus a selector
string. `path` may be a run directory, a checkpoint directory, or a `params.eqx`.
A directory resolves its weights in one ordered pass:

```
<dir>/params.eqx  →  <dir>/model/params.eqx  →  <dir>/checkpoints/latest/params.eqx
```

so a run that has not finished (no `model/` yet) still loads from its latest
checkpoint. Name a specific checkpoint by its path —
`model_load(run_dir / "checkpoints" / "step_00300")` — which works because every
checkpoint dir bundles its own `config.json`, `custom.py` and `prepared.json.gz`.
A file that is not named `params.eqx` raises rather than silently falling through
to the run's final weights.

- `model_load` reconstructs a trained model from a run directory **alone**. It
  loads the run's prepared collection to rebuild the **static** half of the wrapper
  (`rhs_ode`, `controls`, every `SCALE_*`, the index arrays); only the trainable
  leaves come from `params.eqx`. That reconstruction is the expensive part of the
  call. Returns `(trained_wrapper, config)`; `config.solver` carries the solver
  settings the model was fitted under.

- `model_reload` refreshes **only** the trainable leaves into a wrapper you already
  hold, skipping the collection entirely — on a 61-process run that is ~0.03 s
  against ~100 s. It returns the same `(trained_wrapper, config)` pair as
  `model_load`, so the two are interchangeable at the call site, and it logs a
  warning on **every** call. See the danger note below.

- `model_predict` forward-solves the model over `collection` in one batched solve.
  The collection may hold processes the model never trained on — every process in a
  collection shares one `RhsOde` layout, and a mismatch fails fast via
  `validate_rhs_ode_compatibility`. Solver settings come from `config.solver`, so
  there is nothing to re-decide at the call site. Two requirements on `collection`:
  every process needs a measurement at its first time for **every** target (that is
  where the ODE initial condition comes from), and the target set must match
  `config.data.targets`.

- `reconstruct_training` is THE single reconstruction path — `model_load`,
  standalone and ensemble `forward`, and notebooks all go through it. It loads the
  run's **own** prepared collection, **requires** and verifies its recorded
  `inputs.prepared_input.content_hash` *before* invoking any hook, narrows the
  hook-visible data to the run's recorded training process selection, and rebuilds
  the reaction module, loss module and deserialisation template exactly as training
  did. A missing, tampered, or stale hash is an error: there is no optional bypass
  and no forward-only variant. Every loadable run record carries that hash —
  `train` writes it, and each LOO fold config inherits the producer-validated one.
  `reconstruct_run` is a thin caller returning only
  `(reaction_module, loss_module, store, collection)`.

  `model_reload` is deliberately **not** on this path: it reuses the caller's
  wrapper structure and never reads a collection at all (see the danger note).

### Danger: `model_reload` keeps the static half

`model_reload` exchanges only the **trainable** leaves. The static half — every
`SCALE_*`, `controls`, `Cin`, `rhs_ode`, `target_state_indices` — is kept from the
wrapper you pass in and is **not** read from the checkpoint, because it was never
written there (see `save_model` above). Equinox only checks that the *trainable*
pytree matches, which for a typical MLP head depends on layer shapes alone.

So passing a wrapper that came from a different run, or one built against a
different collection, loads the weights into a different scaled space and every
prediction is silently wrong — no exception, no NaN, no log line beyond the
standing warning. **Only use `model_reload` to move between checkpoints of the same
run.** When in doubt, pay for `model_load`.

The same mechanism is why `model_predict` takes an already-loaded wrapper rather
than rebuilding one: it reuses the trained `SCALE_*` as-is. `forward_from_collection`
does rebuild them, but always from the model's own recorded training input, so the
collection you hand it is evaluation data only and never re-scales the model.

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
from bp_format.serialization import load_process_collection
from bp_train import model_load, model_predict, print_trainable_structure

wrapper, config = model_load("examples/00_e2e_sim/output_all")
print_trainable_structure(wrapper.reaction_module)

# Re-simulate with the loaded wrapper — no solver arguments; they ride in `config`.
collection = load_process_collection("examples/00_e2e_sim/prepared/prepared.json")
predictions = model_predict(wrapper, config, collection)
```
