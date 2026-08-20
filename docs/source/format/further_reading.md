# Further reading

> Where the exhaustive reference lives, now that you have the human-sized version.

These docs are deliberately not exhaustive. They cover concepts, runnable examples and
gotchas, and stop before the field-by-field tables, because those rot, and because two
places to look is one too many.

## The three layers

| Layer | Where | Use it for |
|---|---|---|
| **This guide** | you are here | Learning, and remembering how something fits together. |
| **API reference** | [`bp_format`](../autoapi/bp_format/index) | Every signature, every field, every default. Generated from source, so it cannot drift. |
| **Package documentation** | `bp-format/documentation/*.md` in the repo | The dense reference: exhaustive field semantics, design arguments in full, per-module write-ups. Written for agents; correct but heavy going. |

The third layer is intentionally not rendered here. It is in the package repository, and
its relative links resolve when you read it there.

## Package documentation index

In `bp-format/documentation/`:

| File | Covers |
|---|---|
| `01_design_rationale.md` | The design arguments, at length. |
| `02_data_model.md` | Every dataclass, every field, with worked examples. |
| `03_serialization.md` | Full JSON payload shapes, including rejected legacy forms. |
| `04_validation.md` | All twelve validators individually, plus a failure table. |
| `05_inspection.md` | Verbosity levels, plot behaviour, annotated `print_rhs_ode` output. |
| `06_time_series.md` | `TimeSeries` invariants, arithmetic semantics, helper modules. |
| `07_splines.md` | The pseudobatch identity, segmentation policy, backtransform derivation. |
| `08_mechanistic.md` | `ProcessOrdering`, `ControlSplines`, `RhsOde` argument shapes. |
| `09_simulation.md` | The `Simulation` helpers for synthetic ground truth. |

:::{admonition} `specs/` is not documentation
:class: warning
The `specs/` directory in the repository says so itself. It contains plans, some never
built, several predating the current API. Mine it for *rationale* if you are curious why
something is the way it is: never for behaviour.
:::

## Background

- **Pseudobatch transform**: Hesselberg-Thomsen et al. (2024). The method behind
  `pseudobatch_concentration` and the accumulated dilution factor.
- **JAX** and **Equinox**: the array and module libraries everything is built on.
  Worth twenty minutes if you are going to write a reaction module.

## See also

- [bp-train guide](../train/index.md): the other half of the stack.
- [Design rationale](../under_the_hood/design_rationale.md): the architectural decisions.
