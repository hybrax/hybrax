# hybrax.train documentation

hybrax.train fits **hybrid bioprocess ODE models**: it takes a
[`hybrax.format`](../format/README.md) process collection, builds the mechanistic
mass balance, lets you plug in neural / mechanistic reaction and loss modules via
`custom.py` hooks, and runs the prepare → train → forward / loo pipeline on
JAX + Diffrax.

## Getting Started

Read in this order:
1. [Design Rationale](01_design_rationale.md) — the "why": SCL/RAW scaling, the
   shared solve, field-tag partitioning, mean aggregation.
2. [CLI, Config & Hooks](02_cli_and_config.md) — the `hybrax` subcommands, the
   run-config schema, and the **full `custom.py` hooks reference**.
3. [Data Preparation](03_data_preparation.md) — `prepare`, scale estimation,
   state/control layout, target selection.
4. [Reaction & Loss Modules](04_reaction_and_loss.md) — the two pluggable halves
   of a hybrid model (reaction module + loss module).
5. [Training, Forward & LOO](05_train_forward_loo.md) — the training harness,
   forward export, and cross-validation.
6. [Serialization & Inspection](06_serialization_inspect.md) — save/load,
   resumption, provenance, and the structure/schema printers.

## Module Reference

| Module | Source | Documentation | Description |
|--------|--------|---------------|-------------|
| CLI | `src/hybrax/train/cli.py` | [02](02_cli_and_config.md) | `prepare` / `train` / `forward` / `loo` subcommands |
| Run config | `src/hybrax/train/run_config.py` | [02](02_cli_and_config.md) | Pydantic config schema + path resolution |
| Hooks / utils | `src/hybrax/train/utils.py`, `src/hybrax/train/defaults.py` | [02](02_cli_and_config.md) | `custom.py` discovery + default implementations |
| Prepare | `src/hybrax/train/prepare.py` | [03](03_data_preparation.md) | Raw collection → `prepared.json` artifact |
| Training data | `src/hybrax/train/training_data.py` | [03](03_data_preparation.md) | Batch assembly, target selection |
| Controls | `src/hybrax/train/controls_store.py`, `src/hybrax/train/controls.py` | [03](03_data_preparation.md) | Runtime control evaluation + event sources |
| Validation | `src/hybrax/train/validate.py` | [03](03_data_preparation.md) | hybrax.format + prepared-semantics checks |
| Model API | `src/hybrax/train/model_api.py` | [04](04_reaction_and_loss.md) | `RateModule` / `UserLossModule`, scales, field tags |
| Defaults | `src/hybrax/train/defaults.py` | [04](04_reaction_and_loss.md) | `DefaultReactionModule` (MLP) / `DefaultLossModule` (MSE) |
| Wrapper | `src/hybrax/train/wrapper.py` | [04](04_reaction_and_loss.md) | `HybridOdeWrapper` — ODE RHS bridge |
| Dense grids | `src/hybrax/train/dense.py` | [04](04_reaction_and_loss.md) | Union-grid + jump-mask helpers for dense losses |
| Harness | `src/hybrax/train/harness.py` | [05](05_train_forward_loo.md) | Training orchestrator, forward, dense exports |
| Trainer | `src/hybrax/train/trainer.py` | [05](05_train_forward_loo.md) | Single-sample / batched loss evaluation |
| Postprocessing | `src/hybrax/train/postprocessing.py` | [05](05_train_forward_loo.md) | Loss curves + `predictions.csv` export |
| LOO | `src/hybrax/train/loo.py`, `src/hybrax/train/loo_metrics.py` | [05](05_train_forward_loo.md) | Leave-one/some-process-out cross-validation + metrics |
| Checkpointing / logging | `src/hybrax/train/checkpointing.py`, `src/hybrax/train/logging.py` | [05](05_train_forward_loo.md) | Resumable snapshots + telemetry |
| Serialization | `src/hybrax/train/serialization.py` | [06](06_serialization_inspect.md) | Save/load, reconstruction, provenance |
| Inspection | `src/hybrax/train/inspect.py` | [06](06_serialization_inspect.md) | Trainable-structure + reaction-schema tables |

## `custom.py` hooks at a glance

The seven `get_hook` hooks you can define in `custom.py`. A missing hook falls
back to its default. The optional `get_custom_config(raw_custom, config)` setup
adapter is invoked separately before the hooks. Full per-hook write-ups
(signature, behavior, defaults) are in
[02_cli_and_config.md](02_cli_and_config.md#custompy-hooks-reference).

| Hook | Stage | Signature | Default |
|---|---|---|---|
| [`transform_process_collection`](02_cli_and_config.md#transform_process_collection) | prepare | `(collection, config) -> collection` | rename map |
| [`augment_state_values`](02_cli_and_config.md#augment_state_values) | prepare | `(*, parent_name, child_name, state_name, times, base_values, augmented_values, config) -> ndarray` | none |
| [`estimate_all_scales`](02_cli_and_config.md#estimate_all_scales) | train | `(runtime_data, target_names, config) -> EstimatedScales` | none (ones) |
| [`build_reaction_module`](02_cli_and_config.md#build_reaction_module) | train | `(*, target_names, process_names, config, seed, training_parent_collection, **scale_kwargs) -> RateModule` | `DefaultReactionModule` |
| [`build_loss_module`](02_cli_and_config.md#build_loss_module) | train | `(*, target_names, process_names, config, seed, training_parent_collection) -> UserLossModule` | `DefaultLossModule` |
| [`build_learning_rate`](02_cli_and_config.md#build_learning_rate) | train | `(custom_cfg, train_cfg, total_updates) -> float \| optax.Schedule` | none |
| [`build_optimizer`](02_cli_and_config.md#build_optimizer) | train | `(custom_cfg, train_cfg) -> optax.GradientTransformation` | none |

`runtime_data` is a collection-free `RuntimeDataContext` containing the prepared
training and control stores plus the numeric source traces needed by scale hooks.
Constructor hooks receive only the ordered original parents represented by the
selected training processes.

The two base classes the `build_*_module` hooks return are documented in
[04_reaction_and_loss.md](04_reaction_and_loss.md): `RateModule.__call__(t,
inputs) -> ReactionOutputs`, and `UserLossModule` (`loss_names`, `dense_grid_n`,
`__call__(inputs) -> LossOutputs`).

## State / Control layout

The ODE integrates a **scaled (SCL)** state; the reaction module reads it (and
the continuous controls) via
[`ReactionInputs`](04_reaction_and_loss.md#reactioninputs) — all in SCL space —
and returns scaled rates. Layout (see
[03](03_data_preparation.md#state-and-control-layout)):

```
SCL_state (physical)
 ├─ modeled_RMCs                 # species concentrations
 ├─ modeled_PVs                  # dynamic process-variable states
 ├─ V_in_cumulative              # scalar cumulative inflow volume
 ├─ modeled_Inflows_cumulative   # non-negative cumulative inflow
 └─ modeled_Outflows_cumulative  # non-positive cumulative outflow

SCL_integrated_state (solver)
 ├─ SCL_state
 └─ SCL_latent                   # optional module state

SCL_controls (continuous, evaluated at t)
 ├─ controlled Inflows: cumulative | rates | Cin (feed-media composition)
 ├─ controlled Outflows: cumulative | rates | raw retention (0 = removed,
 │  1 = retained)
 └─ controlled PVs: pH, DO, T, …

SCL reaction outputs
 ├─ modeled_ReactionOde_rates
 ├─ modeled_Inflows_rates (≥ 0)
 ├─ modeled_Outflows_rates (≤ 0)
 └─ latent_derivative

Discrete events (applied as state jumps during the solve — not read by module)
 └─ controlled boluses & samples # mass-balance jumps at known event times
```

## See also

- [`hybrax.format` documentation](../format/README.md) — the data model,
  mechanistic RHS, and pseudobatch transform that `hybrax.train` builds on.
