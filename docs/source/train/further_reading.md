# Further reading

> **In one sentence.** Where the exhaustive reference and the real example projects live.

These docs cover concepts, runnable examples and gotchas, and stop before the
field-by-field tables, because those change, and two places to look is one too many.

## The three layers

| Layer | Where | Use it for |
|---|---|---|
| **This guide** | you are here | Learning, and remembering how something fits together. |
| **API reference** | [`bp_train`](../autoapi/bp_train/index) | Every signature, field and default. Generated from source, so it cannot drift. |
| **Package documentation** | `bp-train/documentation/*.md` in the repo | The dense reference: full config schema tables, every `LossInputs` field, the complete hook write-ups. Written for agents; correct but heavy going. |

The third layer is deliberately not rendered here. It lives in the package repository,
where its relative links to source files resolve.

## Package documentation index

In `bp-train/documentation/`:

| File | Covers |
|---|---|
| `01_design_rationale.md` | SCL/RAW, the shared solve, field-tag partitioning, mean aggregation: at length. |
| `02_cli_and_config.md` | Every subcommand flag, the full run-config schema, the complete hooks reference. |
| `03_data_preparation.md` | Prepare internals, scale-axis table, SCL state and control layout, target selection. |
| `04_reaction_and_loss.md` | Every `ReactionInputs` / `LossInputs` field, the default modules, dense-grid losses. |
| `05_train_forward_loo.md` | The training harness, forward exports, LOO folds and metrics. |
| `06_serialization_inspect.md` | Save/load, reconstruction, provenance, the inspection printers. |

:::{admonition} `specs/` is not documentation
:class: warning
Its own README says so: it holds plans, some never built, several predating the current
API. Two are genuinely good background: `pseudo_diagnosis.md` on why the pseudobatch
solve was replaced by the bounded physical solve, and `two_tier_integration_grid.md` on
the dense-grid benchmark. Read those for *rationale*, never for behaviour.
:::

## Example projects

`bp-train/examples/`: real configs, real `custom.py` files, committed outputs.

| Example | Worth reading for |
|---|---|
| **`00_e2e_sim/`** | The designated smallest complete set: prepare, train, forward and LOO configs, a well-commented three-hook `custom.py`, and `run_all.sh`. Start here. |
| **`13_volume_integration/`** | Pedagogically the clearest: a no-op reaction module, which isolates transport so you can check the physics alone. |
| **`01_kittler_2022/structured/`** | Structured rate laws instead of a bare MLP, with a README explaining the reasoning. |
| **`01_kittler_2022/fba_hyb/`** | Kendall uncertainty weighting in a loss module. |
| **`11_tub_2026/fba_hyb/`** | A bounds-hinge loss on a real dataset. |
| **`12_martens_2025_expanded/vanilla/`** | Learning-rate schedules, and a three-line `get_custom_config`. |

Several have committed `.log` files next to their run scripts: real console output,
useful as "what you should see".

`bp-train/tests/fixtures/martens_single/` is the smallest complete fixture: config,
`custom.py` and data that run prepare → train → forward. It makes a good blank-project
template.

:::{admonition} Do not copy from `output_*` directories
:class: warning
`examples/*/output_*/custom.py` and the copies under `checkpoints/` are frozen provenance
snapshots. Some carry stale signatures from older versions of the hook API. Always take
the top-level `custom.py`.
:::

## Background

- **JAX**: autodiff and JIT. **Equinox**: the module system; `eqx.Module`, `filter_jit`,
  `tree_at`. **Diffrax**: the ODE solver and adjoint. **optax**: optimizers, schedules,
  `masked` / `multi_transform`.
- `bp-train/diffrax_callbacks/` has three standalone runnable scripts for the discrete-event
  layer, if you want to understand how bolus and sample jumps are applied.

## See also

- [bp-format guide](../format/index.md): the other half of the stack.
- [Design rationale](../under_the_hood/design_rationale.md).
