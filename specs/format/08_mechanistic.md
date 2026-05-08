# Mechanistic ODE Module

Source: `bp_format/mechanistic.py`

## Purpose

Build JAX/Equinox-compatible ODE components from a `BioProcess`:

- `ControlSplines`: controlled inputs (controlled feeds, controlled PVs) over time.
- `RhsOde`: mechanistic RHS evaluating user-supplied biological expressions
  for each dynamic state, with bp-format adding feed/dilution and volume
  dynamics on top.
- `integrate_process`: full hybrid ODE integration with discrete event handling.
- `extract_discrete_events`, `build_state_splines`, `build_algebraic_func`:
  helpers for events, state splines, and algebraic-variable observables.

The legacy spline-based rate-inversion helpers (`build_q_func`,
`build_rates_func`, `estimate_specific_rates`, `integrate_process_pseudospace`)
were removed in the P3 refactor. See
[`_analytical_rates_spec.md`](_analytical_rates_spec.md) for the full
description of their behavior and the planned replacement
`build_rates_func_analytical`.

## State and Rate Model

### State layout

The ODE state is always:

```text
c = [reactor_component_states..., process_variable_states..., V]
```

- Reactor block order: `rhs_ode.reactor_component_state_names` (biomass always
  index 0, by construction in `_build_process_metadata`).
- PV block order: `rhs_ode.process_variable_state_names` (uncontrolled
  process variables only).
- Volume index: `rhs_ode.volume_idx`.

### Dynamics partition

For each dynamic state `s`, the `RhsOde` evaluates a user-supplied
biological derivative expression over the symbol table

```
{state names} ∪ {controlled-PV names} ∪ {algebraic names} ∪ {rate names}
```

and bp-format adds the physical contribution on top. For reactor states
this is feed inflow and dilution; for PV states it is biological-only
(no feed/dilution); for volume it is `dV/dt = sum(u_flow) + sum(f_modeled)`.

The biological derivative expressions live in
`process.biological_ode.derivatives`. When the user does not supply a
`biological_ode` block, `BioProcess.__post_init__` auto-generates a minimal
one:

- For each reactor-medium component `c`: `dc/dt = q_<c> * <biomass>`.
- For each dynamic (non-controlled, non-static) PV `p`: `dp/dt = r_<p>`.
- Static PVs are skipped (no rate symbol, derivative implicitly zero).

The auto-generated rate names are `q_<rmc_biomass_first>...` followed by
`r_<dynamic_pv>...`, in that exact insertion order — this layout is the
load-bearing invariant for any caller-supplied rates_func that must produce
a flat array aligned with `rhs_ode.rate_names`.

## Public API

### Factory functions

- `get_control_splines(process) -> ControlSplines`
- `get_rhs_ode(process) -> RhsOde`
- `build_rhs_ode(process) -> RhsOde` — alias used internally; raises if
  `process.biological_ode` is unset (which only happens for processes whose
  reactor medium has no components, e.g. shape-only fixtures).
- `build_algebraic_func(process) -> Callable` — evaluator for
  `BiologicalOde.algebraic` quantities, e.g. `X_active(t)` as an observable.

`get_rhs_ode` returns the same `RhsOde` regardless of whether the process's
`biological_ode` block is auto-generated or user-supplied — the dispatch
distinction from before the P3 refactor is gone.

The factory validates strictly: unknown feed-medium components and malformed
`biological_ode` blocks raise `ValueError`.

### `RhsOde` call signature

```python
rhs_ode(c, rates, u_flow, f_modeled, ctrl_pv_values) -> dc_dt
```

Argument shapes:

- `c`: `(rhs_ode.c_size,)` = `[reactor..., pv..., V]`
- `rates`: `(rhs_ode.rate_size,)` — flat user-supplied rate vector, aligned
  with `rhs_ode.rate_names` (= the insertion order of
  `process.biological_ode.rates`)
- `u_flow`: `(rhs_ode.u_flow_size,)` — controlled continuous flow rates
- `f_modeled`: `(rhs_ode.f_modeled_size,)` — modeled (uncontrolled) continuous
  flow rates; pass `jnp.zeros(0)` when none
- `ctrl_pv_values`: `(rhs_ode.n_controlled_pv,)` — controlled-PV values at the
  current time, aligned with `rhs_ode.controlled_pv_names`; pass `jnp.zeros(0)`
  when there are no controlled PVs

Return shape: `(rhs_ode.output_size,) == (rhs_ode.c_size,)`.

Evaluation order inside `__call__`:

1. Compute algebraic variables (e.g. `X_active`) in topo-sorted order.
2. Evaluate the per-state biological RHS expression.
3. Add feed/dilution contributions on the reactor block (PV states are
   biological-only).
4. Append `dV/dt` from the volume changes.

### Important metadata on `RhsOde`

- `reactor_component_state_names`, `process_variable_state_names`,
  `controlled_pv_names`
- `n_reactor_states`, `n_pv_states`, `n_controlled_pv`
- `reactor_indices`, `pv_indices`, `volume_idx`, `static_pv_indices`
- `flow_names`, `modeled_flow_names`
- `name_modeled_algebraic`, `rate_names`, `rate_size`
- `Cin`, `Cin_modeled`

`biomass_idx` was removed; biomass is identified by its name in
`reactor_component_state_names` (always at index 0 by construction).

### Boundary: biological vs. physical

User-written expressions describe only the *biological* part of `dc/dt`.
bp-format unconditionally adds, on top of the biological derivatives:

- Feed inflow + dilution on reactor states from `VolumeChange` flows and
  the Cin matrices.
- Sample outflow.
- `dV/dt = sum(u_flow) + sum(f_modeled)`.

Process-variable states receive *no* physical contribution — their dynamics
are entirely encoded in the user expressions.

### Bounds

`bounds` on reactor components, process variables, volume, and per-rate are
**metadata only** — they are never plumbed into `RhsOde` or the integrator.
Downstream consumers (e.g. `bp-train`'s loss generator) read them off the
process to build soft-constraint penalties (concentrations ≥ 0, quality
attributes in [0, 1], etc.).

## Integration

`integrate_process(process, ctrl, rhs_ode, rates_func, t_eval, ...) -> dict`
runs the full hybrid ODE integration segment-by-segment using
`jax.lax.scan` over segments separated by discrete events. The entire scan
is JIT-compiled once via `eqx.filter_jit`; subsequent calls reuse the
compiled code. Between events the ODE is solved with `diffrax.Tsit5`. At
event boundaries, discrete state updates (sampling, bolus feeds) are applied.

The `rates_func` callback signature is

```python
rates_func(t, state, controls) -> jnp.ndarray  # shape (rhs_ode.rate_size,)
```

aligned with `rhs_ode.rate_names`.

## Example

```python
import jax.numpy as jnp
import bp_format as bp

process = ...
ctrl = bp.mechanistic.get_control_splines(process)
rhs_ode = bp.mechanistic.get_rhs_ode(process)

def rates_func(t, state, controls):
    del t, state, controls
    return jnp.zeros(rhs_ode.rate_size)

t_eval = jnp.linspace(
    process.time_axis.start, process.time_axis.end, 100, dtype=float
)
result = bp.mechanistic.integrate_process(
    process, ctrl, rhs_ode, rates_func, t_eval
)
```

## See Also

- [Splines](07_splines.md)
- [Data Model](02_data_model.md)
- [Analytical rate inversion spec](_analytical_rates_spec.md)
