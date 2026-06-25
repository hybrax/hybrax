# bp-train Documentation

bp-train fits **hybrid bioprocess ODE models**: it takes a
[bp-format](../../bp-format) process collection, builds the mechanistic mass
balance, lets you plug in neural / mechanistic reaction and loss modules via
`custom.py` hooks, and runs the prepare → train → forward / loo pipeline on
JAX + Diffrax.

## Getting Started

Read in this order:
1. [Design Rationale](01_design_rationale.md) — the "why": SCL/RAW scaling, the
   shared solve, field-tag partitioning, mean aggregation.
2. [CLI, Config & Hooks](02_cli_and_config.md) — the `bp-train` subcommands, the
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
| CLI | `bp_train/cli.py` | [02](02_cli_and_config.md) | `prepare` / `train` / `forward` / `loo` subcommands |
| Run config | `bp_train/run_config.py` | [02](02_cli_and_config.md) | Pydantic config schema + path resolution |
| Hooks / utils | `bp_train/utils.py`, `bp_train/defaults.py` | [02](02_cli_and_config.md) | `custom.py` discovery + default implementations |
| Prepare | `bp_train/prepare.py` | [03](03_data_preparation.md) | Raw collection → `prepared.json` artifact |
| Training data | `bp_train/training_data.py` | [03](03_data_preparation.md) | Batch assembly, target selection |
| Controls | `bp_train/controls_store.py`, `bp_train/controls.py` | [03](03_data_preparation.md) | Runtime control evaluation + event sources |
| Validation | `bp_train/validation.py` | [03](03_data_preparation.md) | bp-format + prepared-semantics checks |
| Model API | `bp_train/model_api.py` | [04](04_reaction_and_loss.md) | `UserReactionModule` / `UserLossModule`, scales, field tags |
| Defaults | `bp_train/defaults.py` | [04](04_reaction_and_loss.md) | `DefaultReactionModule` (MLP) / `DefaultLossModule` (MSE) |
| Wrapper | `bp_train/wrapper.py` | [04](04_reaction_and_loss.md) | `HybridOdeWrapper` — ODE RHS bridge |
| Dense grids | `bp_train/dense.py` | [04](04_reaction_and_loss.md) | Union-grid + jump-mask helpers for dense losses |
| Harness | `bp_train/harness.py` | [05](05_train_forward_loo.md) | Training orchestrator, forward, dense exports |
| Trainer | `bp_train/trainer.py` | [05](05_train_forward_loo.md) | Single-sample / batched loss evaluation |
| Postprocessing | `bp_train/postprocessing.py` | [05](05_train_forward_loo.md) | Plots + `predictions.csv` export |
| LOO | `bp_train/loo.py`, `bp_train/loo_metrics.py` | [05](05_train_forward_loo.md) | Leave-one-process-out CV + metrics |
| Checkpointing / logging | `bp_train/checkpointing.py`, `bp_train/logging.py` | [05](05_train_forward_loo.md) | Resumable snapshots + telemetry |
| Serialization | `bp_train/serialization.py` | [06](06_serialization_inspect.md) | Save/load, reconstruction, provenance |
| Inspection | `bp_train/inspect.py` | [06](06_serialization_inspect.md) | Trainable-structure + reaction-schema tables |

## `custom.py` hooks at a glance

The single list of every hook you can define in `custom.py`. All are discovered
via `get_hook(custom_module, "<name>", <default>)`; a missing hook falls back to
its default. Full per-hook write-ups (signature, behavior, defaults) are in
[02_cli_and_config.md](02_cli_and_config.md#custompy-hooks-reference).

| Hook | Stage | Signature | Default |
|---|---|---|---|
| [`transform_process_collection`](02_cli_and_config.md#transform_process_collection) | prepare | `(collection, config) -> collection` | rename map |
| [`build_sample_acc_series`](02_cli_and_config.md#build_sample_acc_series) | prepare | `(process, process_name, collection_metadata, config)` | sample-acc source |
| [`estimate_all_scales`](02_cli_and_config.md#estimate_all_scales) | train | `(collection, target_names, config) -> EstimatedScales` | none (ones) |
| [`build_reaction_module`](02_cli_and_config.md#build_reaction_module) | train | `(*, target_names, process_names, config, seed, collection, **scale_kwargs) -> UserReactionModule` | `DefaultReactionModule` |
| [`build_loss_module`](02_cli_and_config.md#build_loss_module) | train | `(*, target_names, process_names, config, seed, collection) -> UserLossModule` | `DefaultLossModule` |
| [`build_learning_rate`](02_cli_and_config.md#build_learning_rate) | train | `(custom_cfg, train_cfg) -> float` | none |
| [`build_optimizer`](02_cli_and_config.md#build_optimizer) | train | `(custom_cfg, train_cfg) -> optax.GradientTransformation` | none |

The two base classes the `build_*_module` hooks return are documented in
[04_reaction_and_loss.md](04_reaction_and_loss.md): `UserReactionModule.__call__(t,
inputs) -> ReactionOutputs`, and `UserLossModule` (`loss_names`, `dense_grid_n`,
`__call__(inputs) -> LossOutputs`).

## State / Control layout

The ODE integrates a **scaled (SCL)** state; the reaction module reads it via
[`ReactionInputs`](04_reaction_and_loss.md#reactioninputs) and returns scaled
rates. Controls are evaluated from the controls store at time `t` (not
integrated). Layout (see [03](03_data_preparation.md#state-and-control-layout)):

```
SCL_state (integrated)
 ├─ modeled_RMCs              # species concentrations            SCALE_modeled_RMCs
 ├─ modeled_PVs              # dynamic process-variable states    SCALE_modeled_PVs
 ├─ V_in_cumulative          # scalar cumulative inflow volume    SCALE_V_in_cumulative
 └─ modeled_FVCs_cumulative  # per modeled feed                   SCALE_modeled_FVCs_cumulative

controls (evaluated at t)
 ├─ controlled_FVCs: cumulative | rates | Cin                     SCALE_controlled_FVCs_*
 ├─ controlled_PVs           # pH, DO, T, …                       SCALE_controlled_PVs
 └─ extras                   # bolus dilution + sample-accumulation

reaction outputs (scaled)
 ├─ modeled_BiologicalOde_rates                                   SCALE_modeled_BiologicalOde_rates
 └─ modeled_FVCs_rates (≥ 0)                                      SCALE_modeled_FVCs_rates
```

## Examples

The `examples/` directory contains end-to-end case studies. Each has a
`train-config.json`, a `custom.py` of hooks, and a prepared/output workflow.

| Directory | Organism / data | Demonstrates |
|-----------|-----------------|--------------|
| [`00_e2e_sim/`](../examples/00_e2e_sim) | Simulated | Custom reaction module learning RMC + modeled-PV rates; `estimate_all_scales` |
| [`01_kittler_2022/vanilla/`](../examples/01_kittler_2022/vanilla) | E. coli fed-batch | `DefaultReactionModule` baseline |
| [`01_kittler_2022/fba_hyb/`](../examples/01_kittler_2022/fba_hyb) | E. coli fed-batch | FBA surrogate reaction module + Kendall uncertainty loss |
| [`11_tub_2026/fba_hyb/`](../examples/11_tub_2026/fba_hyb) | V. natriegens | FBA surrogate with algebraic biomass + bounds-hinge loss |
| [`11_tub_2026/migration/`](../examples/11_tub_2026/migration) | V. natriegens | Migration from legacy bp-format |
| [`12_martens_2025_expanded/vanilla/`](../examples/12_martens_2025_expanded/vanilla) | CHO (simulated) | Default reaction module |
| [`12_martens_2025_expanded/structured/`](../examples/12_martens_2025_expanded/structured) | CHO (simulated) | Dense-grid curvature penalty + between-measurement bounds |
| [`12_martens_2025_expanded/migration/`](../examples/12_martens_2025_expanded/migration) | CHO (simulated) | Migration from legacy bp-format |
| [`13_volume_integration/`](../examples/13_volume_integration) | Synthetic | Volume integration / dilution tracking |

## See also

- [bp-format documentation](../../bp-format/documentation/README.md) — the data
  model, mechanistic RHS, and pseudobatch transform that bp-train builds on.
