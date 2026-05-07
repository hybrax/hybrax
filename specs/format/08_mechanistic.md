# Mechanistic ODE Module

Source: `bp_format/mechanistic.py`

## Purpose

This module builds JAX/Equinox-compatible ODE components from a `BioProcess`:

- `ControlSplines`: controlled inputs over time.
- `RhsOde`: mechanistic RHS with reactor-state kinetics, physical rates, feed/dilution,
  and volume dynamics.
- Integration and inversion helpers (`integrate_process*`, `build_q_func`,
  `estimate_specific_rates`).

## State and Rate Model

### State layout

The ODE state is always:

```text
c = [reactor_component_states..., process_variable_states..., V]
```

- Reactor block order: `mb.reactor_component_state_names` (biomass always index 0).
- PV block order: `mb.process_variable_state_names` (uncontrolled process variables).
- Volume index: `mb.volume_idx`.

### Dynamics partition

For reactor states `c_r` other than biomass:

```text
dc_i/dt = q_i(t) * X_active + r_i(t) + feed_dilution(c_i, V, u_flow, f_modeled)
```

The biomass entry additionally absorbs the intracellular accumulation rates so
that mass balance `X_measured = X_active + Σ P_intracellular` holds:

```text
dc_biomass/dt = (q_biomass(t) + Σ_{j ∈ intracellular} q_j(t)) * X_active
              + r_biomass(t) + feed_dilution(c_biomass, V, u_flow, f_modeled)
```

When no component is marked `is_intracellular=True` the sum is empty and the
biomass equation collapses to the same form as the other states. `q_biomass`
is therefore the specific growth rate of *active* biomass, not the apparent
specific rate of measured biomass; the inversion in `build_q_func` reflects
the same convention by subtracting the intracellular concentration
derivatives before dividing by `X_active`.

For PV states `c_pv`:

```text
dc_pv/dt = r_pv(t)
```

For volume:

```text
dV/dt = sum(u_flow) + sum(f_modeled)
```

Where:

- `q` is reactor-only (`mb.q_size == mb.n_reactor_states`).
- `r` is additive on all non-volume states (`mb.r_size == mb.n_reactor_states + mb.n_pv_states`).
- Feed and dilution are applied only to the reactor block.
- Static uncontrolled PVs are forced to zero dynamics.

## Public API

### Factory functions

- `get_control_splines(process) -> ControlSplines`
- `get_rhs_ode(process) -> RhsOde | UserDefinedRhsOde` *(dispatching)*
- `build_user_defined_rhs_ode(process) -> UserDefinedRhsOde` *(force user-defined path; raises if `process.biological_ode` is unset)*
- `build_algebraic_func(process) -> Callable` *(evaluator for `BiologicalOde.algebraic` quantities, e.g. `X_active(t)` as an observable)*

`get_rhs_ode` dispatches based on `process.biological_ode`:

- When `None` (default), it returns the auto-generated `RhsOde` with the dynamics described above.
- When set, it returns a `UserDefinedRhsOde` built from the user's per-state biological RHS expressions, algebraic variables, and abstract rate placeholders. The dilution / feed / volume contributions are still added by bp-format on top of the user's biological RHS — the boundary is what makes the block named `biological_ode` and not just `ode`.

Both factories validate strictly: unknown feed-medium components and malformed `biological_ode` blocks raise `ValueError`.

### `RhsOde` call signature

```python
mb(c, q, u_flow, f_modeled, r) -> dc_dt
```

Argument shapes:

- `c`: `(mb.c_size,)` = `[reactor..., pv..., V]`
- `q`: `(mb.q_size,)` reactor-only
- `u_flow`: `(mb.u_flow_size,)`
- `f_modeled`: `(mb.f_modeled_size,)`
- `r`: `(mb.r_size,)` for all non-volume states

Return shape: `(mb.output_size,) == (mb.c_size,)`.

### Important metadata on `RhsOde`

- `reactor_component_state_names`
- `process_variable_state_names`
- `n_reactor_states`, `n_pv_states`
- `reactor_indices`, `pv_indices`, `volume_idx`
- `flow_names`, `modeled_flow_names`
- `Cin`, `Cin_modeled`

### `UserDefinedRhsOde` call signature

```python
mb(c, rates, u_flow, f_modeled, ctrl_pv_values) -> dc_dt
```

Argument shapes:

- `c`: `(mb.c_size,)` = `[reactor..., pv..., V]`
- `rates`: `(mb.rate_size,)` — user-declared rate vector, aligned with `mb.rate_names`
- `u_flow`: `(mb.u_flow_size,)`
- `f_modeled`: `(mb.f_modeled_size,)`
- `ctrl_pv_values`: `(mb.n_controlled_pv,)` — controlled-PV values at the current time, aligned with `mb.controlled_pv_names`. Pass `jnp.zeros(0)` when there are no controlled PVs.

Return shape: `(mb.output_size,) == (mb.c_size,)`.

Evaluation order inside `__call__`:

1. Compute algebraic variables (e.g. `X_active`) in topo-sorted order.
2. Evaluate the per-state biological RHS expression.
3. Add the standard feed/dilution contribution on the reactor block (PV states are biological-only).
4. Append `dV/dt` from the volume changes.

### Additional metadata on `UserDefinedRhsOde`

- `controlled_pv_names`, `n_controlled_pv`
- `name_modeled_algebraic`
- `rate_names`, `rate_size` (replaces `q_size`; not pinned to `n_reactor_states`)

### Boundary: biological vs. physical

User-written expressions describe only the *biological* part of `dc/dt`. bp-format unconditionally adds:

- Feed inflow + dilution on reactor states from `VolumeChange` flows + Cin matrices.
- Sample outflow.
- `dV/dt = sum(u_flow) + sum(f_modeled)`.

Process-variable states get *no* physical contribution — their dynamics are entirely user-defined.

### Bounds

`bounds` on reactor components, process variables, volume, and per-rate are **metadata only** — they are never plumbed into `RhsOde` / `UserDefinedRhsOde` / `build_q_func` / integrator. Downstream consumers read them off the process to build soft-constraint penalties (e.g. concentrations ≥ 0, quality attributes in [0, 1]).

## Integration and inversion

### Integration

Both integration paths now require a mixed-state `rates_func`:

- `integrate_process(..., rates_func, t_eval, ...)`
- `integrate_process_pseudospace(..., rates_func, t_eval, ...)`

```python
rates_func(t, state, controls) -> (q, r)
```

Where:

- `q` has shape `(mb.q_size,)` and covers reactor-component specific rates.
- `r` has shape `(mb.r_size,)` and covers additive physical rates on all
  non-volume states.
- In `build_rates_func(..., r_func=None)` default mode, reactor-component `r`
  entries are zero and process-variable `r` entries are inferred from
  process-variable spline derivatives.

The older public `q_func` / `r_func` integration inputs have been removed.

### Inversion (`build_q_func`)

`build_q_func` returns reactor-only `q(t)` and supports q/r partitioning on reactor
states:

- default: all reactor states treated as q-states,
- explicit partition via `q_state_indices` and `r_state_indices`,
- when supplying `r_func`, explicit partition indices are required,
- overlapping q/r indices require `r_func` so overlap `r` can be subtracted.

## Example

```python
import jax.numpy as jnp
import bp_format as bp

process = ...
ctrl = bp.mechanistic.get_control_splines(process)
mb = bp.mechanistic.get_rhs_ode(process)

def rates_func(t, state, controls):
    del t, state, controls
    q = jnp.zeros(mb.q_size)
    r = jnp.zeros(mb.r_size)
    return q, r

state = jnp.zeros(mb.c_size)
u = ctrl(10.0)
u_flow = u[jnp.array(ctrl.flow_indices)]
f_modeled = jnp.zeros(mb.f_modeled_size)
q, r = rates_func(10.0, state, u)
dc_dt = mb(state, q, u_flow, f_modeled, r)
```

## See Also

- [Splines](07_splines.md)
- [Data Model](02_data_model.md)
