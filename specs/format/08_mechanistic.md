# Mechanistic ODE Module

Source: `bp_format/mechanistic.py`

## Purpose

Build JAX/Equinox-compatible ODE components from a `BioProcess`:

- `ProcessOrdering`: canonical name ordering across every derived module
  (states, controls, rates, algebraic, FVCs, SVCs).
- `ControlSplines`: controlled inputs (controlled FVCs/SVCs/PVs) over time.
- `RhsOde`: mechanistic RHS evaluating user-supplied biological expressions
  for each dynamic state, with bp-format adding feed/dilution and volume
  dynamics on top.
- `extract_discrete_events`, `build_state_splines`, `build_algebraic_func`:
  helpers for events, state splines, and algebraic-variable observables.

The legacy spline-based rate-inversion helpers (`build_q_func`,
`build_rates_func`, `estimate_specific_rates`, `integrate_process_pseudospace`)
were removed in the P3 refactor. See
[`_analytical_rates_spec.md`](_analytical_rates_spec.md) for the full
description of their behavior and the planned replacement
`build_rates_func_analytical`. Forward integration of the process
(`integrate_process`) lives in `bp-train` and is not part of bp-format.

## ProcessOrdering — single source of truth

`get_process_ordering(process) -> ProcessOrdering` collects every name
group consumed by every other factory. Sub-group ordering rules:

- `name_modeled_rates`: preserve user-supplied insertion order of
  `BiologicalOde.rates` (downstream consumers pass rate vectors in this
  order).
- `name_modeled_algebraic`: topo-sorted by inter-algebraic dependencies,
  ties broken alphabetically.
- All other tuples are alphabetical within their sub-group.

Layout invariants:

```
c = [name_modeled_RMCs... | name_modeled_PVs... | V]
u = [name_controlled_FVCs... | name_controlled_SVCs... | name_controlled_PVs...]
```

The first `len(FVCs)+len(SVCs)` entries of `u` are flow rates (spline
derivatives). FVC flow rates are non-negative; SVC flow rates carry their
storage sign (non-positive) so the feed-dilution machinery treats them as
signed outflows. The remaining entries of `u` are direct PV values.

`get_process_ordering` validates:

- Every continuous `FeedVolumeChange` defines `feed_medium`, with every
  feed component existing in `reactor_medium.components`.
- Every non-controlled `ProcessVariable` carries a `TimeSeries` value
  (static PVs must be `is_controlled=True`).
- The `BiologicalOde.algebraic` graph is acyclic.
- All names across every group are unique (no shared names between
  states, rates, algebraic, controlled PVs, FVCs, SVCs).

## State and Rate Model

### Dynamics partition

For each dynamic state `s`, the `RhsOde` evaluates a user-supplied
biological derivative expression over the symbol table

```
{state names} ∪ {controlled-PV names} ∪ {algebraic names} ∪ {rate names}
```

and bp-format adds the physical contribution on top. For reactor (RMC)
states this is feed inflow (FVCs add species) and dilution from all flows
(FVCs and SVCs); for PV states it is biological-only (no feed/dilution);
for volume it is `dV/dt = total_FVC_inflow - total_SVC_outflow_magnitude`.

The biological derivative expressions live in
`process.biological_ode.derivatives`. When the user does not supply a
`biological_ode` block, `BioProcess.__post_init__` auto-generates a
minimal one keyed by reactor-medium component names and dynamic PV names.

## Public API

### Factory functions

- `get_process_ordering(process) -> ProcessOrdering`
- `get_control_splines(process, ordering=None) -> ControlSplines`
- `get_rhs_ode(process, ordering=None) -> RhsOde`
- `build_rhs_ode(process, ordering=None) -> RhsOde` — equivalent to
  `get_rhs_ode`; raises if `process.biological_ode` is unset.
- `build_algebraic_func(process, ordering=None) -> Callable` — evaluator for
  `BiologicalOde.algebraic` quantities, e.g. `X_active(t)` as an observable.
- `extract_discrete_events(process, ordering) -> list[dict]`
- `build_state_splines(process, ordering) -> dict[str, callable]`

`extract_discrete_events` and `build_state_splines` take a
`ProcessOrdering` rather than a compiled `RhsOde` — they only need name
tuples, not lambdified callables.

### `RhsOde` call signature

```python
rhs_ode(c, rates, u, f_modeled_FVCs, f_modeled_SVCs) -> dc_dt
```

Argument shapes:

- `c`: `(len(name_modeled_RMCs) + len(name_modeled_PVs) + 1,)` —
  `[RMCs..., PVs..., V]`.
- `rates`: `(len(name_modeled_rates),)` aligned with
  `rhs_ode.name_modeled_rates`.
- `u`: full control vector from `ControlSplines.__call__(t)` —
  `[FVC_flows | SVC_flows | PV_values]`.
- `f_modeled_FVCs`: `(len(name_modeled_FVCs),)` — uncontrolled FVC flow
  rates (non-negative); pass `jnp.zeros(0)` when none.
- `f_modeled_SVCs`: `(len(name_modeled_SVCs),)` — uncontrolled SVC flow
  rates (non-positive, signed); pass `jnp.zeros(0)` when none.

Return shape: same as `c`.

Evaluation order inside `__call__`:

1. Compute algebraic variables (e.g. `X_active`) in topo-sorted order.
2. Evaluate the per-state biological RHS expression.
3. Add feed/dilution contributions on the reactor block (PV states are
   biological-only).
4. Append `dV/dt` from FVC inflow + SVC outflow.

### Fields on `RhsOde`

- Names: `name_modeled_rates`, `name_modeled_algebraic`,
  `name_modeled_RMCs`, `name_modeled_PVs`, `name_modeled_FVCs`,
  `name_modeled_SVCs`, `name_controlled_PVs`, `name_controlled_FVCs`,
  `name_controlled_SVCs`.
- Compiled callables: `algebraic_funcs`, `derivative_funcs`.
- Feed compositions: `Cin_controlled_FVCs`, `Cin_modeled_FVCs`.

All sizes derive from the lengths of the name tuples (`len(...)`); there
are no separate sizing fields.

### Boundary: biological vs. physical

User-written expressions describe only the *biological* part of `dc/dt`.
bp-format unconditionally adds, on top of the biological derivatives:

- Feed inflow + dilution on reactor states from FVC flows and the
  `Cin_*` matrices.
- Dilution from SVC outflows on reactor states.
- `dV/dt = total_FVC_inflow - total_SVC_outflow_magnitude`.

Process-variable states receive *no* physical contribution — their
dynamics are entirely encoded in the user expressions.

### Bounds

`bounds` on reactor components, process variables, volume, and per-rate
are **metadata only** — never plumbed into `RhsOde`. Downstream
consumers (e.g. `bp-train`'s loss generator) read them off the process
to build soft-constraint penalties (concentrations ≥ 0, quality
attributes in [0, 1], etc.).

## Example

```python
import jax.numpy as jnp
import bp_format as bp

process = ...
ordering = bp.mechanistic.get_process_ordering(process)
ctrl = bp.mechanistic.get_control_splines(process, ordering)
rhs_ode = bp.mechanistic.get_rhs_ode(process, ordering)

t = jnp.array(5.0)
u = ctrl(t)
c = jnp.zeros(len(rhs_ode.name_modeled_RMCs) + len(rhs_ode.name_modeled_PVs) + 1)
rates = jnp.zeros(len(rhs_ode.name_modeled_rates))
dc_dt = rhs_ode(c, rates, u, jnp.zeros(0), jnp.zeros(0))
```

Forward integration is provided by `bp-train`; consume `RhsOde` and
`ControlSplines` from there.

## See Also

- [Data Model](02_data_model.md) — `ProcessOrdering` listing
- [Splines](07_splines.md)
- [Analytical rate inversion spec](_analytical_rates_spec.md) — planned
  `build_rates_func_analytical`
