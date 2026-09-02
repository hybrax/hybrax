# diffrax-callbacks

**Julia-style callback system for [Diffrax](https://github.com/patrick-kidger/diffrax): differentiable discrete event handling for Neural ODEs in JAX.**

Brings `ContinuousCallback`, `StopConditionCallback`, `DiscreteCallback`, `PresetTimeCallback`, `PeriodicCallback`, `ManifoldProjection`, and `CallbackSet` to JAX/Diffrax — with full differentiability through discrete events.

## Why?

Neural ODEs in JAX can model continuous dynamics, but real systems have **discrete events**: a bioreactor gets fed, a valve opens, a sample is taken. These events change the system state instantaneously, and we need to:

1. **Detect** when events occur (state-triggered or scheduled)
2. **Modify** the state at the event (add substrate, remove volume, etc.)
3. **Differentiate** through the entire trajectory — including events — to learn dynamics and optimize control

Julia's [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/) has had this for years. Diffrax can detect events but cannot modify state and continue. **This library bridges that gap.**

## Installation

```bash
pip install diffrax-callbacks
```

Or from source:
```bash
pip install -e ".[dev]"
```

**Requirements:** JAX >= 0.4, Diffrax >= 0.5, Equinox >= 0.11, Optimistix >= 0.0.6

## Quick Start

```python
import jax.numpy as jnp
import diffrax
from diffrax_callbacks import ContinuousCallback, diffeqsolve_with_callbacks

# Your ODE
def ode(t, y, args):
    X, S = y[0], y[1]
    mu = 0.4 * S / (2.0 + S)
    return jnp.array([mu * X, -mu * X / 0.5])

# Feed when substrate drops below 1.0 g/L
callback = ContinuousCallback(
    condition_fn=lambda y, t, args: y[1] - 1.0,  # triggers at zero-crossing
    affect_fn=lambda y, t, args: jnp.array([      # modify state
        y[0] * y[1] / (y[1] + 0.1),               # dilute biomass
        (y[1] * 1.0 + 100.0 * 0.1) / 1.1,         # add substrate
    ]),
    direction="down",  # trigger on S crossing 1.0 from above
)

sol = diffeqsolve_with_callbacks(
    diffrax.ODETerm(ode),
    diffrax.Tsit5(),
    t0=0.0, t1=48.0, dt0=0.01,
    y0=jnp.array([0.5, 20.0]),
    args=None,
    callbacks=callback,
    max_events=20,
)

print(f"Events: {sol.event_count}")
print(f"Final state: {sol.y_final}")
```

**This is fully differentiable:**
```python
import jax

def objective(feed_volume):
    cb = ContinuousCallback(
        condition_fn=lambda y, t, args: y[1] - 1.0,
        affect_fn=lambda y, t, args: apply_feed(y, feed_volume),
        direction="down",
    )
    sol = diffeqsolve_with_callbacks(...)
    return sol.y_final[0]  # maximize biomass

grad = jax.grad(objective)(0.1)  # gradient through events!
```

## Callback Types

### ContinuousCallback

Triggers when a continuous condition function crosses zero. Uses root-finding for exact event time detection.

```python
ContinuousCallback(
    condition_fn,       # (y, t, args) -> scalar. Triggers on zero-crossing.
    affect_fn,          # (y, t, args) -> new_y. Applied when triggered.
    direction="down",   # "up", "down", or "both"
    root_finder=...,    # optimistix root finder (default: Newton)
    repeat_nudge=1e-6,  # min time before re-triggering (prevents loops)
)
```

**Julia equivalent:** `ContinuousCallback(condition, affect!; direction)`

### StopConditionCallback

Stops the complete solve when a boolean condition is true at the initial state or
after an accepted solver step. It applies no effect and is not added to the
state-changing callback event log. If a stop condition and continuous callback fire
within the same accepted step, the stop condition wins even when the continuous root
is earlier in that step.

```python
StopConditionCallback(
    condition_fn,  # (y, t, args) -> bool
)
```

### DiscreteCallback

Evaluated at every segment boundary (between events). Useful for clamping, validation, or corrections.

```python
DiscreteCallback(
    condition_fn,  # (y, t, args) -> bool. Checked at each boundary.
    affect_fn,     # (y, t, args) -> new_y. Applied when True.
)
```

**Julia equivalent:** `DiscreteCallback(condition, affect!)`

### PresetTimeCallback

Triggers at predetermined times. The solver steps exactly to each time.

```python
PresetTimeCallback(
    times=jnp.array([6.0, 12.0, 24.0]),  # trigger times
    affect_fn=lambda y, t, args: ...,      # state modification
)
```

**Julia equivalent:** `PresetTimeCallback(tstops, affect!)`

### PeriodicCallback

Triggers every Δt time units. Convenience wrapper around PresetTimeCallback.

```python
PeriodicCallback(
    dt=4.0,                           # interval
    affect_fn=lambda y, t, args: y,   # e.g., log state (no-op)
    t_end=48.0,                       # last possible trigger
)
```

**Julia equivalent:** `PeriodicCallback(dt, affect!)`

### ManifoldProjection

Projects state onto a constraint manifold after each event.

```python
ManifoldProjection(
    project_fn=lambda y, t, args: jnp.maximum(y, 0.0),  # non-negative
)
```

**Julia equivalent:** `ManifoldProjection(g)`

### CallbackSet

Combines multiple callbacks with priority handling:
1. StopConditionCallbacks — terminate without applying an effect
2. ContinuousCallbacks — earliest event wins
3. PresetTimeCallbacks — at exact times
4. DiscreteCallbacks / ManifoldProjection — at every segment boundary

```python
callbacks = CallbackSet(
    ContinuousCallback(...),   # feed on low substrate
    StopConditionCallback(...), # stop on an invalid state
    ContinuousCallback(...),   # bleed on high biomass
    PresetTimeCallback(...),   # scheduled samples
    PeriodicCallback(...),     # periodic logging
    ManifoldProjection(...),   # enforce constraints
)
```

**Julia equivalent:** `CallbackSet(cb1, cb2, ...)`

## The Solver

```python
sol = diffeqsolve_with_callbacks(
    terms,                  # diffrax.ODETerm(ode_fn)
    solver,                 # e.g., diffrax.Tsit5(), diffrax.Kvaerno5()
    t0, t1, dt0,            # time span and initial step
    y0,                     # initial state
    args,                   # passed to ODE, condition_fn, and affect_fn
    callbacks=...,          # single callback or CallbackSet
    max_events=20,          # callback-event budget (fixed for JIT)
    stepsize_controller=..., # adaptive stepping
    max_steps=4096,
    adjoint=...,            # diffrax adjoint method
)
```

### How it works

The solver wraps `diffrax.diffeqsolve` in a `jax.lax.scan` loop:

1. **Solve** until the next callback, stop condition, or final time
2. **Stop** if a `StopConditionCallback` fired
3. **Apply** a callback's `affect_fn` to modify state otherwise
4. **Run** DiscreteCallbacks / ManifoldProjection at the boundary
5. **Restart** from the modified state and repeat up to `max_events` times

Boolean stop conditions and numeric `ContinuousCallback`s share one composed
Diffrax event, so they can be used together.

Everything is JIT-compiled. State-changing callbacks remain differentiable via
JAX's autodiff; the boolean decision to stop is not differentiable.

### The Solution Object

```python
sol.y_final           # final state
sol.t_final           # final time reached
sol.terminated_by_event  # whether a StopConditionCallback ended the solve early
sol.event_count       # number of state-changing callbacks triggered
sol.event_times       # (max_events,) when each event fired
sol.event_types       # (max_events,) which callback triggered (-1 = unused)
sol.event_states_before  # state just before each event
sol.event_states_after   # state just after each affect

# Filter by callback
times, before, after = sol.get_events(callback_index=0)
n_feeds = sol.count_by_type(0)

# Pretty-print
sol.print_events(
    state_names=["X", "S", "P", "V"],
    callback_names={0: "feed", 1: "bleed"},
)
```

## Differentiability

Gradients flow through:
- **ODE dynamics** (standard Diffrax autodiff)
- **Event detection** (root-finding for exact crossing time)
- **State jumps** (the `affect_fn` itself)
- **Event timing** (changing parameters changes when events fire)

This means you can optimize:
- Feed amounts and concentrations
- Event trigger thresholds
- Neural network parameters in the ODE, condition, or affect
- All simultaneously, end-to-end

```python
# Optimize feed volume, concentration, AND trigger threshold
def objective(params):
    feed_vol, feed_conc, threshold = params
    cb = ContinuousCallback(
        condition_fn=lambda y, t, args: y[1] - threshold,
        affect_fn=lambda y, t, args: apply_feed(y, feed_vol, feed_conc),
        direction="down",
    )
    sol = diffeqsolve_with_callbacks(...)
    return sol.y_final[2] * sol.y_final[3]  # total product

grads = jax.grad(objective)(params)  # all three gradients
```

### Learnable Callbacks

The `affect_fn` can be a neural network. Pass it through `args`:

```python
class FeedController(eqx.Module):
    mlp: eqx.nn.MLP
    def __call__(self, y):
        sig = jax.nn.sigmoid(self.mlp(y))
        return 0.05 + sig[0] * 0.45, 50 + sig[1] * 250  # vol, conc

# args = (dynamics_model, feed_controller, threshold)
cb = ContinuousCallback(
    condition_fn=lambda y, t, args: y[1] - args[2],
    affect_fn=lambda y, t, args: apply_feed(y, *args[1](y)),
    direction="down",
)
sol = diffeqsolve_with_callbacks(..., args=args, callbacks=cb)
```

## Examples

| Example | Description |
|---------|-------------|
| [01_basic_fed_batch.py](examples/01_basic_fed_batch.py) | State-triggered feeding + gradient-based optimization |
| [02_multiple_callbacks.py](examples/02_multiple_callbacks.py) | 5 callback types: feed, bleed, sample, log, projection |
| [03_hybrid_node_control.py](examples/03_hybrid_node_control.py) | Full pipeline: learn dynamics + optimize neural controller |

## Comparison with Julia

| Feature | Julia SciML | diffrax-callbacks |
|---------|------------|-------------------|
| ContinuousCallback | `ContinuousCallback(cond, affect!)` | `ContinuousCallback(cond_fn, affect_fn)` |
| DiscreteCallback | Per solver step | Per segment boundary |
| PresetTimeCallback | `PresetTimeCallback(tstops, affect!)` | `PresetTimeCallback(times, affect_fn)` |
| VectorContinuousCallback | `VectorContinuousCallback(cond, affect!, n)` | Multiple ContinuousCallbacks in CallbackSet |
| CallbackSet | `CallbackSet(cb1, cb2)` | `CallbackSet(cb1, cb2)` |
| ManifoldProjection | `ManifoldProjection(g)` | `ManifoldProjection(project_fn)` |
| State modification | In-place: `integrator.u .= ...` | Functional: `affect(y, t, args) -> new_y` |
| Differentiation | All adjoint methods | RecursiveCheckpointAdjoint (default) |
| Dynamic event count | Unlimited | Fixed budget (`max_events`) |
| JIT compilation | N/A (Julia is compiled) | Full `jax.jit` support |
| GPU support | Via CUDA.jl | Via `jax.devices()` |

### Key differences from Julia

1. **Functional, not imperative.** Julia's `affect!(integrator)` mutates state in-place. Ours returns a new state: `affect(y, t, args) -> new_y`. This is a JAX requirement.

2. **Fixed event budget.** `jax.lax.scan` requires a fixed iteration count. Set `max_events` high enough — unused slots have negligible cost.

3. **DiscreteCallback runs at segment boundaries**, not every solver step. This covers most practical use cases (clamping, validation, logging). For per-step evaluation, you'd need to modify Diffrax internals.

## Citation

If you use this in your research, please cite:

```bibtex
@software{diffrax_callbacks,
  title={diffrax-callbacks: Julia-style callback system for Diffrax},
  author={Buehler, Marco},
  year={2026},
  url={https://github.com/marcobuehler/diffrax-callbacks},
}
```

## License

MIT
