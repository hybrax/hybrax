# Mechanistic ODE Module

Source: `bpbench/mechanistic.py`

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

For reactor states `c_r`:

```text
dc_r/dt = q(t) * X_active + r_r(t) + feed_dilution(c_r, V, u_flow, f_modeled)
```

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
- `get_rhs_ode(process) -> RhsOde`

`get_rhs_ode` validates feed-media components strictly: unknown feed component names
raise `ValueError`.

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
import bpbench as bp

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
