# Further Reading

> Where the exhaustive reference and the real example projects live.

These docs cover concepts, runnable examples and gotchas, and stop before the
field-by-field tables, because those change, and two places to look is one too many.

## The three layers

| Layer | Where | Use it for |
|---|---|---|
| **This guide** | you are here | Learning, and remembering how something fits together. |
| **API reference** | [`hybrax.train`](../autoapi/hybrax/train/index) | Every signature, field and default. Generated from source, so it cannot drift. |
| **Package documentation** | `hybrax/specs/train/*.md` in the repo | The dense reference: full config schema tables, every `LossInputs` field, the complete hook write-ups. Written for agents; correct but heavy going. |

The third layer is deliberately not rendered here. It lives in the package repository,
where its relative links to source files resolve.

## Package documentation index

In `hybrax.train/documentation/`:

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

Research examples inside `hybrax` hold real project configs and `custom.py` files from
actual research work. They are not documentation: several are pinned to older hook
signatures, none are guaranteed to run against the current API, and they are being
superseded by this site's [Tutorials](../tutorials/01_your_first_dataset.md) and
[Gallery](../gallery/index.md), which cover the same patterns as runnable, verified
examples kept in sync with every release. Prefer those.

## Background

- **JAX**: autodiff and JIT. **Equinox**: the module system; `eqx.Module`, `filter_jit`,
  `tree_at`. **Diffrax**: the ODE solver and adjoint. **optax**: optimizers, schedules.
- `hybrax.train.diffrax_callbacks` has three standalone runnable scripts for the
  discrete-event layer, if you want to understand how bolus and sample jumps are
  applied.

## See also

- [hybrax.format guide](../format/index.md): the other half of the stack.
- [Design rationale](../under_the_hood/design_rationale.md).
