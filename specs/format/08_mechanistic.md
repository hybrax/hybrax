# Mechanistic ODE Module

Source: `bp_format/mechanistic.py`

## Purpose

Turn a `BioProcess` into the pieces of an ODE:

- **`ProcessOrdering`** — the canonical layout of every state, control, and rate
  vector.
- **`ControlSplines`** — all controlled inputs as one function of time.
- **`RhsOde`** — `dc/dt`, combining the biology you wrote with the physics
  bp-format adds.
- Helpers for discrete events, state trajectories, and algebraic observables.

Everything here is `eqx.Module` and JIT-safe.

**bp-format does not integrate.** It builds the right-hand side; running a
solver over it is [bp-train](../../bp-train/documentation/README.md)'s job.

## `ProcessOrdering` — one layout, decided once

```python
ordering = bp.mechanistic.get_process_ordering(process)
```

Every other factory takes this same object, so a rate vector built for one is
valid for all of them. Without it, each factory would need its own sorting rule
and they would drift.

Ordering rules:

- `name_modeled_rates` — **the insertion order of `BiologicalOde.rates`**, kept
  as written. Sorting it would silently permute every rate vector downstream.
- `name_modeled_algebraic` — topologically sorted so each expression's
  dependencies are computed first; alphabetical within a level.
- Everything else — alphabetical. Biomass has no reserved index.

Resulting layouts:

```
c = [ modeled_RMCs... | modeled_PVs... | V ]
u = [ controlled_Inflows... | controlled_Outflows... | controlled_PVs... ]
```

In `u`, the first `len(Inflows) + len(Outflows)` entries are **flow rates** (spline
derivatives of the cumulative-volume traces); the rest are direct process
variable values. Feed flows are non-negative; sample flows keep their stored
negative sign, and the mass balance treats them as signed.

`get_process_ordering` raises if:

- a continuous `Inflow` has no `feed_medium`, or names a species that
  is not in `reactor_medium.components`;
- an uncontrolled `ProcessVariable` holds a `StaticVariable` (a state with no
  time axis cannot be integrated — mark it `is_controlled=True`);
- the `algebraic` graph has a cycle;
- any name appears in two groups (a state that is also a rate, and so on).

Field list: [02_data_model.md](02_data_model.md#processordering).

## `ControlSplines`

```python
controls = bp.mechanistic.get_control_splines(process, ordering)
u = controls(t)            # shape (n_controlled_Inflows + n_controlled_Outflows + n_controlled_PVs,)
```

Evaluates every controlled signal at time `t`, in `ProcessOrdering` layout.
Volume-change entries are differentiated (`nu=1`) because they are stored
cumulatively and the ODE needs a rate; process variables are evaluated directly.

Stored splines are reused as-is, never refitted. A series with no spline state
gets a cubic fit on the spot; a `StaticVariable` process variable becomes a
constant piece over the time axis.

## `RhsOde`

```python
rhs_ode = bp.mechanistic.build_rhs_ode(process, ordering)
dc_dt   = rhs_ode(c, rates, u, f_modeled_Inflows, f_modeled_Outflows)
```

| Argument | Shape | Meaning |
|----------|-------|---------|
| `c` | `(n_RMCs + n_PVs + 1,)` | State: `[RMCs…, PVs…, V]` |
| `rates` | `(len(name_modeled_rates),)` | Rate values, in declaration order |
| `u` | full control vector | Output of `ControlSplines(t)` |
| `f_modeled_Inflows` | `(len(name_modeled_Inflows),)` | Uncontrolled feed flow rates, ≥ 0 |
| `f_modeled_Outflows` | `(len(name_modeled_Outflows),)` | Uncontrolled sample flow rates, ≤ 0 |

Returns `dc/dt` with the same shape as `c`. Pass `jnp.zeros(0)` for the modeled
flow vectors when there are none.

`rates` is a single flat array, not a tuple — one argument, one layout, aligned
with `rhs_ode.name_modeled_rates`.

### What it computes

1. **Algebraic quantities**, in topological order. Each is a sympy expression
   compiled to a JAX callable over `[RMCs | PVs | controlled PVs | algebraic |
   rates]`.
2. **Biological derivatives**, one compiled expression per dynamic state,
   verbatim from `biological_ode.derivatives`. A state with no entry gets `0`.
3. **Physical contributions on reactor states only**:

   ```
   total_in  = Σ controlled feed flows + Σ modeled feed flows        (≥ 0)
   total_out = −(Σ controlled sample flows + Σ modeled sample flows) (≥ 0)
   retained_out_per_rmc = Σ_Outflows retention · |q_outflow|          (per RMC)

   dilution  = −(total_in − retained_out_per_rmc) / V · c_RMCs
   addition  =  Σ_k flow_k · Cin[k, :] / V
   ```

   A component leaving with a well-mixed Outflow at the reactor's own bulk
   concentration does not, on its own, change that concentration — removing
   a sample doesn't alter what's left behind. Only feeding (which adds mass
   at a *different* concentration, `Cin`) dilutes. An Outflow's per-RMC
   `retention` (σ ∈ [0, 1], default 0 on every Outflow) inverts this for the
   retained fraction: as volume drops around what's retained, its
   concentration rises — σ=1 reproduces exact mass conservation as `V`
   shrinks (perfusion bleed retaining biomass; evaporation retaining
   solutes). `retention` is only implemented for continuous Outflows; a
   discrete Outflow is required to have empty `retention`.
4. **Volume**: `dV/dt = total_in − total_out`.

**Process-variable states get no physical term.** Their dynamics are entirely
what you wrote. A dissolved-oxygen state is not "diluted" by feeding, and
pretending otherwise would be wrong — if a process variable really does need a
transport term, write it into its derivative expression.

Volume is guarded: `eqx.error_if` aborts the solve if `V` reaches `1e-10` or
below, rather than dividing by nearly zero.

### The biological / physical boundary

This split is the core contract of the module.

| You write | bp-format adds |
|-----------|----------------|
| `biological_ode.derivatives`, `algebraic`, `rates` | feed inflow, dilution, sample outflow, `dV/dt` |

Feed and sample flow rates are deliberately **not** in the expression symbol
table. That keeps mass balance out of user code and makes it impossible to
double-count a dilution term. (It is also why perfusion and evaporation are not
expressible today — see [specs/](../specs/README.md).)

`print_rhs_ode` renders the two halves side by side; use it to check what you
actually built.

### Fields

Name tuples (all static): `name_modeled_rates`, `name_modeled_algebraic`,
`name_modeled_RMCs`, `name_modeled_PVs`, `name_modeled_Inflows`,
`name_modeled_Outflows`, `name_controlled_PVs`, `name_controlled_Inflows`,
`name_controlled_Outflows`.

Compiled callables: `algebraic_funcs`, `derivative_funcs`.

Feed composition matrices: `Cin_controlled_Inflows`, `Cin_modeled_Inflows`, each
`(n_feeds, n_RMCs)` of static feed concentrations, zero where a feed does not
carry a species.

There are no separate size fields — every dimension is `len(...)` of a name
tuple.

### Bounds are not enforced here

`bounds` on components, process variables, volume, and rates never reach
`RhsOde`. They are metadata for downstream loss generators. A rate bounded
`(0, None)` is not clipped during a solve; it is up to the training loss to
penalize violations.

## Helpers

### `build_algebraic_func(process, ordering=None)`

```python
f = bp.mechanistic.build_algebraic_func(process, ordering)
f(state_values, ctrl_pv_values, rates)   # -> {"X_active": ..., ...}
```

Evaluates the `algebraic` quantities on their own, keyed by name in topological
order. Useful for plotting or for a loss that targets a derived quantity such as
active biomass. `state_values` is `[RMCs… | PVs…]` without volume.

### `extract_discrete_events(process, ordering)`

Returns a list of dicts, sorted by time with **samples before boluses** at the
same timestamp:

| Key | Value |
|-----|-------|
| `t` | event time |
| `kind` | `"sample"` or `"bolus_feed"` |
| `dV` | signed volume change |
| `Cin` | feed composition aligned with `ordering.name_modeled_RMCs`, or `None` for samples |
| `source` | name of the originating volume change |

Zero-magnitude deltas (`< 1e-15`) are dropped. At most one sample and one bolus
per timestamp — more raises, because the outcome would depend on an
undefined ordering.

### `build_state_splines(process, ordering)`

Returns `{state_name: callable}` for every non-volume state, giving the measured
trajectory as a continuous function.

- Pseudobatch-transformed reactor components return a `BacktransformSpline`, so
  the callable yields **real-space** concentration.
- Everything else returns the stored `PPoly` directly; a `StaticVariable`
  becomes a constant piece.

The pseudobatch bundle is validated first: a `c*` trace without a matching
`feed_corrections` entry (or the reverse) raises.

## Example

```python
import jax.numpy as jnp
import bp_format as bp

ordering = bp.mechanistic.get_process_ordering(process)
controls = bp.mechanistic.get_control_splines(process, ordering)
rhs_ode  = bp.mechanistic.build_rhs_ode(process, ordering)

n_states = len(ordering.name_modeled_RMCs) + len(ordering.name_modeled_PVs) + 1

t     = jnp.array(5.0)
c     = jnp.ones(n_states)
rates = jnp.zeros(len(ordering.name_modeled_rates))

dc_dt = rhs_ode(
    c, rates, controls(t),
    jnp.zeros(len(ordering.name_modeled_Inflows)),
    jnp.zeros(len(ordering.name_modeled_Outflows)),
)
```

Under JIT:

```python
import equinox as eqx

step = eqx.filter_jit(rhs_ode)
dc_dt = step(c, rates, controls(t), jnp.zeros(0), jnp.zeros(0))
```

## Limitations

- **Feed composition must be static.** A `TimeSeries` feed concentration raises
  `NotImplementedError`.
- **Well-mixed CSTR only.** Every species leaves at the same rate through a
  `Outflow`, so perfusion with cell retention and evaporation (where
  solutes stay) cannot be expressed.
- **No rate inversion.** Recovering rate values from state splines is not
  implemented.

Design notes on the last two live in [specs/](../specs/README.md); neither has
code behind it.

## See also

- [Data Model](02_data_model.md) — `BiologicalOde`, `ProcessOrdering`
- [Inspection](05_inspection.md) — `print_rhs_ode`
- [Splines](07_splines.md) — the state trajectories consumed here
- [bp-train](../../bp-train/documentation/README.md) — integrates all of this
