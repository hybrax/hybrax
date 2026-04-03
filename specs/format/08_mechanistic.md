# Mechanistic ODE Module

Source: `bpbench/mechanistic.py`

## Purpose

This module generates JAX/Equinox-compatible ODE right-hand-side modules directly from a `BioProcess` object. It automates the construction of mass balance equations, control signal interpolation, and discrete event handling, so users only need to define the specific-rate model `q(t)` (the biological "black box") while BPbench handles the physics.

## Design Rationale

### Why Auto-Generated ODE RHS?

Every fed-batch bioprocess shares the same mass balance structure:

```
dc_i/dt = q_i * X_active + sum_k (f_k / V) * (C_in[k,i] - c_i)
dV/dt   = sum_k f_k
```

The species count, feed streams, and composition vary between processes, but the equation structure is the same. Rather than having users manually write these equations (error-prone and repetitive), `get_rhs_ode()` inspects a `BioProcess` and constructs the correct `RhsOde` module automatically.

### Why ControlSplines as a Separate Module?

Control signals (feed flow rates, pH setpoints, temperature profiles) are *known inputs* measured during the experiment. Specific rates `q` are the *unknowns* to be modeled. Separating these into `ControlSplines` and `RhsOde` makes the interface clean:

- `ControlSplines` evaluates all known controls at time `t` (one spline evaluation call).
- `RhsOde` takes the state `c`, specific rates `q`, and control flows `u_flow` and returns `dc/dt`.
- The user plugs in their model for `q` (neural network, Monod kinetics, etc.) between these two.

### Why the (c, q, u_flow, f_modeled) Interface?

The `RhsOde.__call__` signature separates four roles:

| Argument | Role | Source |
|----------|------|--------|
| `c` | State vector `[c_species..., V]` | ODE solver |
| `q` | Specific rates (one per species) | User model |
| `u_flow` | Controlled flow rates | `ControlSplines` |
| `f_modeled` | Uncontrolled (modeled) flow rates | User model |

This separation means:
- The user model only needs to output `q` (and optionally `f_modeled`).
- Controlled feeds are handled automatically.
- The same `RhsOde` works for any user model.

### Why the Cin Matrix?

The feed composition matrix `Cin` (shape `(n_flows, n_species)`) pre-computes inlet concentrations for all feed streams and species. This avoids dictionary lookups inside the JIT boundary and enables efficient vectorized computation of the dilution terms.

### Why Discrete Events Are Handled Separately?

Bolus feeds and sampling events are not smooth functions -- they are instantaneous state changes. The ODE solver cannot integrate through a discontinuity. Instead:
1. Integration runs until the next event time.
2. The state vector is updated with the discrete jump (dilution from feed, concentration preservation from sampling).
3. Integration resumes from the new state.

This is implemented in `integrate_process()` using diffrax.

## Public API

### Factory Functions

#### `get_control_splines(process) -> ControlSplines`

Builds a JIT-compatible module that evaluates all controlled signals at time `t`.

- **Continuous feed flows** are included as *flow rates* (derivative of the cumulative volume spline).
- **Controlled process variables** (pH, temperature) are included as direct values.
- **Discrete volume changes** (bolus feeds, sampling) are *excluded* -- they are handled as state discontinuities.

#### `get_rhs_ode(process) -> RhsOde`

Builds a JIT-compatible module implementing the generalized fed-batch ODE RHS.

- Inspects reactor medium components to determine species ordering and intracellular flags.
- Inspects volume changes to build feed composition matrices (`Cin`, `Cin_modeled`).
- Biomass is always at index 0 in the state vector.

### Module Classes

#### `ControlSplines(eqx.Module)`

| Attribute | Type | Description |
|-----------|------|-------------|
| `control_names` | `tuple[str, ...]` | Ordering of all controlled signals |
| `flow_indices` | `tuple[int, ...]` | Indices corresponding to flow rates |
| `ctrl_indices` | `tuple[int, ...]` | Indices corresponding to process variables |

**`__call__(t) -> jnp.ndarray`**: Returns shape `(n_controls,)`. Flow rate entries are derivatives of cumulative volume splines; process variable entries are direct spline values.

#### `RhsOde(eqx.Module)`

| Attribute | Type | Description |
|-----------|------|-------------|
| `c_size` | `int` | `n_species + 1` (species + volume) |
| `q_size` | `int` | `n_species` (number of specific rates) |
| `u_flow_size` | `int` | Number of controlled flow streams |
| `f_modeled_size` | `int` | Number of modeled flow streams |
| `species_names` | `tuple[str, ...]` | Species ordering (biomass first) |
| `flow_names` | `tuple[str, ...]` | Controlled flow stream ordering |
| `modeled_flow_names` | `tuple[str, ...]` | Modeled flow stream ordering |
| `biomass_idx` | `int` | Always 0 |
| `intracellular_indices` | `tuple[int, ...]` | Indices of intracellular species |
| `Cin` | `jnp.ndarray` | Feed composition matrix `(n_flows, n_species)` |
| `Cin_modeled` | `jnp.ndarray` | Modeled feed composition matrix |

**`__call__(c, q, u_flow, f_modeled) -> jnp.ndarray`**: Computes `dc/dt` as:

```
X_active = c[biomass] - sum(c[intracellular_i])
dc_i/dt  = q_i * X_active + sum_k (f_k/V) * (Cin[k,i] - c_i)
dV/dt    = sum(f_k)
```

Returns shape `(c_size,)` = `[dc_species/dt..., dV/dt]`.

### Spline and Rate Estimation Functions

| Function | Description |
|----------|-------------|
| `build_conc_splines(process)` | Build concentration splines from process data for all reactor medium components. |
| `build_q_func(process, ctrl, mb, conc_splines)` | Construct a JIT-compilable `q(t)` callable by inverting the ODE RHS using known concentrations and controls. |
| `estimate_specific_rates(process, ctrl, mb, conc_splines, t_eval)` | Convenience wrapper: evaluate `q(t)` at specified times. |

### Integration Functions

| Function | Description |
|----------|-------------|
| `integrate_process(process, ctrl, mb, q_func, t_eval)` | Full hybrid ODE + discrete event integration in real concentration space. |
| `integrate_process_pseudospace(process, ctrl, mb, q_func, t_eval)` | Same integration but in pseudobatch space (c*). |
| `extract_discrete_events(process, mb)` | Extract discrete events (sampling, bolus feeds) with their state-vector deltas. |

### State Vector Layout

The state vector `c` has a fixed layout:

```
c = [c_biomass, c_species_1, c_species_2, ..., c_species_n, V]
     ^^^^^^^^^                                                ^
     index 0                                                 last
     (always biomass)                                        (volume)
```

Species ordering matches `mb.species_names`. Intracellular species are at the indices given by `mb.intracellular_indices`.

## Examples

### Minimal Workflow: Build Modules and Evaluate

```python
import bpbench as bp
import equinox as eqx
import jax.numpy as jnp

# Load a process
dataset = bp.serialization.load_dataset("data.json")
process = dataset.case_studies["kittler_2022"].processes["fed_batch_001"]

# Build modules
ctrl = bp.mechanistic.get_control_splines(process)
mb = bp.mechanistic.get_rhs_ode(process)

# Evaluate controls at t=10.0
u = eqx.filter_jit(ctrl)(10.0)
print(f"Control names: {ctrl.control_names}")
print(f"Control values at t=10: {u}")

# Extract flow rates
u_flow = u[jnp.array(ctrl.flow_indices)]

# Compute dc/dt with dummy specific rates
q = jnp.ones(mb.q_size) * 0.1  # placeholder
c = jnp.ones(mb.c_size)        # placeholder state
f_modeled = jnp.zeros(mb.f_modeled_size)

dc_dt = eqx.filter_jit(mb)(c, q, u_flow, f_modeled)
print(f"State vector size: {mb.c_size}")
print(f"Species: {mb.species_names}")
```

### Estimating Specific Rates from Data

```python
import bpbench as bp
import jax.numpy as jnp

process = ...  # loaded process
ctrl = bp.mechanistic.get_control_splines(process)
mb = bp.mechanistic.get_rhs_ode(process)

# Build concentration splines
conc_splines = bp.mechanistic.build_conc_splines(process)

# Estimate specific rates at evaluation times
t_eval = jnp.linspace(process.time_axis.start, process.time_axis.end, 50)
q_values = bp.mechanistic.estimate_specific_rates(
    process, ctrl, mb, conc_splines, t_eval
)
# q_values shape: (len(t_eval), mb.q_size)
```

### Defining a Custom Rate Model

```python
import equinox as eqx
import jax.numpy as jnp

class MonodKinetics(eqx.Module):
    """Simple Monod kinetics for biomass growth."""
    mu_max: float
    K_s: float

    def __call__(self, c, u):
        """Return specific rates q given state c and controls u."""
        X = c[0]         # biomass
        S = c[1]         # substrate
        mu = self.mu_max * S / (self.K_s + S)
        q_biomass = mu
        q_substrate = -mu / 0.5  # yield coefficient
        return jnp.array([q_biomass, q_substrate])

# Use with RhsOde:
# dc_dt = mb(c, model(c, u), u_flow, f_modeled)
```

## See Also

- [Splines](07_splines.md) -- spline fitting that feeds into this module
- [Data Model](02_data_model.md) -- `BioProcess` structure consumed by factory functions
- [Design Rationale](01_design_rationale.md#3-volume-as-a-first-class-concept) -- why volume is special in the ODE
