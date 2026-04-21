# Splines and Pseudobatch Transform

Source: `bp_format/splines.py`

## Purpose

This module provides the pseudobatch transformation pipeline and spline fitting infrastructure for converting discrete concentration measurements into continuous-time representations. It bridges the gap between raw experimental data and the smooth, differentiable functions needed for ODE integration and gradient-based optimization.

The module handles three related tasks:
1. **Pseudobatch normalization:** Transforms fed-batch concentrations to batch-equivalent pseudo-concentrations, removing dilution artifacts.
2. **Segmented spline fitting:** Fits piecewise cubic splines to time-series data, respecting discontinuities at discrete events.
3. **Backtransformation:** Reconstructs real concentrations from pseudo-concentrations using the ADF and feed correction terms, in a JIT-compatible way.

## Design Rationale

### Why Pseudobatch Normalization?

In fed-batch processes, observed concentrations are affected by both biological activity and physical dilution from feeds. A substrate being consumed might appear to have a *rising* concentration simply because the feed adds more substrate than the cells consume. This makes it difficult to:
- Fit smooth splines to the raw concentration data
- Compare fed-batch and batch processes
- Extract meaningful biological rates

The pseudobatch transform (Hesselberg-Thomsen et al., 2024) removes dilution effects:

```
c*(t) = c(t) * ADF(t) - feed_correction(t)
```

where `ADF(t)` is the accumulative dilution factor and `feed_correction(t)` accounts for mass added by feeds. The resulting pseudo-concentrations `c*` are smooth and well-suited for spline approximation.

### Why Segmented Splines?

Even after pseudobatch transformation, bolus feed events create discontinuities in the time series. A single global spline would smooth over these jumps, producing incorrect intermediate values. Instead:

1. **Event detection** identifies discrete events (bolus feeds, sampling) from the volume changes.
2. **Segmentation** splits the time axis at event boundaries.
3. **Per-segment fitting** fits an independent spline to each segment.
4. **Padded storage** pads segments to fixed array shapes for JAX compatibility.

### Why Step-Aware ADF and `linear_plus_step` Feed Correction?

For discrete bolus events:
- `ADF` is represented as a left-continuous step function.
- `feed_correction` is represented as `linear_base + instantaneous_jumps`
  (`linear_plus_step` mode).

Plain linear interpolation across event anchors creates non-physical ramps.
Step-aware evaluation preserves event discontinuities.

### The Epsilon Convention

For discrete bolus events at `t_event`, helper grids still use `t_event ± _EPS`
(`_EPS = 1e-4`) for robust event bookkeeping. Runtime backtransform semantics do
not rely on linear ramps across that interval.

### Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `_EPS` | `1e-4` | Pre-event offset for discrete bolus handling |
| `DEFAULT_MAX_SEGMENTS` | `16` | Maximum number of segments in padded storage |
| `DEFAULT_MAX_CTRL_POINTS` | `128` | Maximum control points per segment |
| `SMOOTHING_THRESHOLD` | `100` | Switch from interpolation to smoothing spline above this many points |

## Public API

### Core Spline Infrastructure

#### Fitting

| Function | Description |
|----------|-------------|
| `fit_timeseries_spline(ts, boundaries, max_segments, max_ctrl_points)` | Fit a segmented cubic spline to a TimeSeries. Returns an `Interpolator`. |
| `choose_spline_kind(n_points)` | Returns `"smoothing_bspline"` (> 100 pts) or `"cubic_interp"` (otherwise). |
| `make_interpax_spline(t, y, bc_type)` | Create an `interpax.CubicSpline` from arrays. |
| `make_pchip_spline(t, y)` | Create a monotone PCHIP spline from arrays. |
| `make_constant_spline(value, t_start, t_end)` | Create a constant-valued spline over a time range. |

#### Evaluation

| Function | Description |
|----------|-------------|
| `build_interpax_spline(rep)` | Reconstruct an `interpax.CubicSpline` from an `Interpolator`. |
| `evaluate_spline_at(rep, t)` | Evaluate an `Interpolator` at a single time point. |

#### Segmentation

| Function | Description |
|----------|-------------|
| `detect_discrete_state_events(process)` | Extract discrete event times from a BioProcess. Returns `DiscreteEvents`. |
| `make_segment_boundaries(t_min, t_max, event_times)` | Build segment breakpoints from event times. |
| `split_timeseries(ts, boundaries)` | Partition a TimeSeries into segments at the given boundaries. |

### Pseudobatch Pipeline

| Function | Description |
|----------|-------------|
| `build_pseudobatch_inputs(process, species_name)` | Extract the inputs needed for pseudobatch transformation (biomass, substrate, feed rate, etc.) for a given species. Returns a dict. |
| `build_splines(inputs, process=None, species_name=None)` | Build per-species runtime spline payload from `build_pseudobatch_inputs` output. |
| `evaluate_real_concentration(rep, t)` | Evaluate backtransformed real concentration using step-aware ADF and feed correction metadata. |
| `to_interpolator(...)` | Convert scipy/interpax spline output to a padded `Interpolator` with metadata. |

### JAX-Compatible Backtransform Classes

#### `BacktransformSpline`

An `eqx.Module` that evaluates the inverse pseudobatch transform at any time:

```python
c(t) = (c*(t) + feed_correction(t)) / ADF(t)
```

Fields:
- `c_star_spline` -- interpax CubicSpline for c*
- `adf_times`, `adf_values` -- step function data for ADF
- `fc_spline`, `fc_times`, `fc_values` -- feed correction baseline
- `fc_jump_times`, `fc_jump_values` -- discrete feed-correction jumps (linear_plus_step mode)
- `is_constant` -- bypass flag for constant-concentration species
- `constant_value` -- returned directly when `is_constant=True`

Methods:
- `__call__(t)` -- evaluate backtransformed concentration
- `derivative()` -- return a callable for dc/dt

Build with `build_backtransform_spline(rep)` from a stored `Interpolator`.

#### `BatchedBacktransformSpline`

Stacks multiple `BacktransformSpline` objects for vectorized evaluation across species. Used inside JIT-compiled ODE solvers.

Build with `build_batched_conc_splines(conc_splines, species_names, t_start, t_end)`.

## Examples

### Detecting Discrete Events

```python
import bp_format as bp

dataset = bp.serialization.load_dataset("data.json")
process = dataset.case_studies["kittler_2022"].processes["batch_001"]

events = bp.splines.detect_discrete_state_events(process)
print(events.times)   # array of event times
print(events.labels)  # optional labels
```

### Fitting Splines to a Process

```python
import bp_format as bp

# Load a process
dataset = bp.serialization.load_dataset("data.json")
process = dataset.case_studies["kittler_2022"].processes["fed_batch_001"]

# Build pseudobatch inputs + runtime spline payload for one species
inputs = bp.splines.build_pseudobatch_inputs(process, "glucose")
splines = bp.splines.build_splines(inputs, process, "glucose")
interpolator = bp.splines.to_interpolator(inputs, splines, "glucose")
```

### Evaluating Backtransformed Concentrations

```python
import jax.numpy as jnp
from bp_format.splines import build_backtransform_spline

# Build a JIT-compatible backtransform module
bt = build_backtransform_spline(interpolator)

# Evaluate at a single time (works inside jax.jit)
c_glucose = bt(5.0)

# Evaluate at many times
t_dense = jnp.linspace(0.0, 48.0, 500)
c_dense = jax.vmap(bt)(t_dense)
```

### Using BacktransformSpline in a JIT-Compiled Function

```python
import equinox as eqx
import jax

@eqx.filter_jit
def eval_concentrations(bt_spline, times):
    return jax.vmap(bt_spline)(times)

c_values = eval_concentrations(bt, jnp.linspace(0.0, 48.0, 100))
```

## See Also

- [TimeSeries](06_time_series.md) -- the underlying data container
- [Mechanistic](08_mechanistic.md) -- consumes splines for ODE integration
- [Data Model](02_data_model.md) -- `Interpolator` and `DiscreteEvents` dataclasses
- [Design Rationale](01_design_rationale.md#5-pseudobatch-normalization) -- mathematical background
