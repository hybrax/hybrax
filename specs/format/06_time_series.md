# TimeSeries

Source: `bp_format/time_series/`

## Purpose

The `TimeSeries` class is the fundamental measurement container in bp-format. It is an `eqx.Module` (JAX pytree) that stores measured data (`times`/`values` arrays) and optionally carries fitted spline coefficients (`breaks`/`coeffs`/`segment_start_piece_idx`) for continuous-time evaluation.

`TimeSeries` itself does not decide whether the data is continuous or discontinuous -- that semantic comes from the parent object. For example, a `VolumeChange` with `is_continuous=True` holds a TimeSeries representing a continuous flow, while `is_continuous=False` means discrete events (bolus feeds, sampling). Spline fitting is only meaningful for continuous data.

## Design Rationale

- **Why `eqx.Module`:** TimeSeries instances are passed into JIT-compiled functions (ODE solvers, loss functions). Being an Equinox module means JAX can automatically trace through TimeSeries objects, enabling `jax.jit`, `jax.grad`, and `jax.vmap`.

- **Why power-basis storage:** Spline coefficients are stored as `[a, b, c, d]` per piece, where the polynomial is `a + b*dt + c*dt^2 + d*dt^3` with `dt = t - t_break`. Horner-form evaluation (`a + dt*(b + dt*(c + dt*d))`) is simple, fast, and maps cleanly to JAX operations. B-spline basis evaluation would require recursive knot-vector lookups.

- **Why `jump_times`:** Fed-batch processes have discontinuities at bolus feed events. The `jump_times` field tracks where these occur so that downstream code (spline segmentation, event detection) can handle them correctly.

- **Why `continuity_side`:** At breakpoints where the spline transitions between pieces, the value can be taken from the left or right piece. For fed-batch processes with discrete events, right-continuity (`"right"`) means the post-event value is used at the event time.

- **Why `derived` flag:** Marks a TimeSeries as computed (not directly measured). Downstream code can distinguish derived series (e.g., specific rates, pseudobatch concentrations) from raw experimental data.

## Public API

### `TimeSeries` Class

```python
class TimeSeries(eqx.Module):
    # Measured data (optional -- but at least one of data or spline must be present)
    times: jnp.ndarray | None     # strictly increasing 1D array
    values: jnp.ndarray | None    # same shape as times

    # Fitted spline coefficients (optional, all three required if any present)
    breaks: jnp.ndarray | None              # breakpoints (n_pieces + 1)
    coeffs: jnp.ndarray | None              # shape (n_pieces, 4), power-basis [a, b, c, d]
    segment_start_piece_idx: jnp.ndarray | None  # maps segments to piece indices

    # Metadata (static fields, not JAX-traced)
    derived: bool                  # default False
    continuity_side: str           # "left" or "right", default "right"
    jump_times: jnp.ndarray | None # event times for discontinuities
    metadata: Any                  # arbitrary metadata dict
    dtype: jnp.dtype               # JAX dtype for arrays, default float64
```

**Construction invariants:**
- Must provide discrete samples (times + values) and/or spline state (breaks + coeffs + segment_start_piece_idx).
- If discrete samples are provided, both `times` and `values` are required, must be 1D, same length, and `times` must be strictly increasing.
- If spline state is provided, all three fields are required. `coeffs` must be `(n_pieces, 4)`, `breaks` must have `n_pieces + 1` entries.

**Key methods:**

| Method | Description |
|--------|-------------|
| `evaluate(t, *, side=None)` | Evaluate spline at a single time point. Returns scalar. |
| `evaluate_many(ts, *, side=None)` | Evaluate spline at multiple time points. Returns 1D array. |
| `deriv(order=1)` | Return a new TimeSeries with derivative coefficients. |
| `integrate(a, b)` | Compute the definite integral from `a` to `b`. |
| `to_dict()` | Serialize to a dict (for JSON). |
| `from_dict(data)` | Class method: deserialize from a dict. |

**Arithmetic operators:**
TimeSeries supports `+`, `-`, `*`, `/` with other TimeSeries or scalars.

- **Exact (add/sub with splines):** When both operands have splines, breakpoints are merged and coefficients are added/subtracted directly. No approximation error.
- **Approximate (mul/div with splines):** Multiplication and division of splines cannot be done exactly in the power basis. These operations fit a new smoothing spline to the result.
- **Discrete fallback:** When one or both operands lack splines, operations fall back to interpolating discrete samples onto a merged time grid.
- **Scalar operations:** `ts * 2.0` or `ts / 3.0` scales coefficients directly (exact).

### `spline_ops` Module

Low-level spline evaluation primitives.

| Function | Description |
|----------|-------------|
| `piece_index(t, breaks, side)` | Find which piece contains time `t`. |
| `evaluate_piece(coeff_row, dt)` | Evaluate one cubic piece using Horner's method. |
| `evaluate_scalar(t, breaks, coeffs, side)` | Evaluate the full piecewise spline at one time. |
| `evaluate_many(ts, breaks, coeffs, side)` | Vectorized evaluation via `jax.vmap`. |
| `rebase_piece(coeff_row, shift)` | Rebase coefficients from `x0` to `x0 + shift`. |
| `rebase_to_breaks(old_breaks, old_coeffs, new_breaks)` | Rebase an entire spline onto a new breakpoint grid. |
| `derivative_coeffs(coeffs, order)` | Compute polynomial derivative coefficients. |
| `integrate_definite(breaks, coeffs, a, b)` | Definite integral over `[a, b]`. |
| `merge_breaks(breaks_a, breaks_b)` | Union of two breakpoint arrays. |
| `merge_segment_starts(...)` | Merge segment start indices for two splines. |

### `grid_utils` Module

Time-grid operations for merging and interpolating discrete samples.

| Function | Description |
|----------|-------------|
| `merge_times_with_tolerance(a, b)` | Merge two sorted time arrays, deduplicating within tolerance (`TIME_DEDUP_ATOL`, `TIME_DEDUP_RTOL`). |
| `linear_interpolate_samples(times, values, target_times)` | Linearly interpolate discrete samples onto a target grid. |
| `synthesize_binary_samples(t_a, v_a, t_b, v_b, op)` | Merge discrete samples for binary operations. |

### `io` Module

Serialization/deserialization for TimeSeries.

| Function | Description |
|----------|-------------|
| `timeseries_to_dict(ts)` | Convert TimeSeries to canonical dict format. |
| `timeseries_from_dict(cls, data)` | Reconstruct TimeSeries from dict. |
| `timeseries_from_process_state(cls, process_state, variable)` | Load from metadata.hybrax format. |
| `timeseries_from_input_dict(cls, input_data, process_key, variable)` | Load from full input JSON. |

### `constants` Module

Numerical tolerance defaults:

| Constant | Value | Purpose |
|----------|-------|---------|
| `TIME_DEDUP_ATOL` | `1e-8` | Absolute tolerance for time deduplication |
| `TIME_DEDUP_RTOL` | `1e-7` | Relative tolerance for time deduplication |
| `APPROX_ABS_FLOOR` | `1e-8` | Absolute floor for approximate operations |
| `APPROX_REL_ERR_TARGET` | `1e-3` | Target relative error for smoothing spline fits |
| `APPROX_INITIAL_S` | `1.0` | Initial smoothing parameter |
| `APPROX_S_REDUCTION_FACTOR` | `0.5` | Smoothing parameter reduction per refit attempt |
| `APPROX_MAX_REFIT_ATTEMPTS` | `8` | Maximum refit attempts for approximate operations |
| `DIVISION_NEAR_ZERO_THRESHOLD` | `1e-8` | Threshold for near-zero division detection |

## Examples

### Creating a TimeSeries from Experimental Data

```python
import jax.numpy as jnp
from bp_format import TimeSeries

# Discrete-only (no spline)
ts = TimeSeries(
    times=jnp.array([0.0, 2.0, 4.0, 6.0, 8.0]),
    values=jnp.array([1.0, 2.5, 5.1, 4.2, 3.0]),
)

print(ts.times)   # [0. 2. 4. 6. 8.]
print(ts.values)  # [1.  2.5 5.1 4.2 3. ]
```

### Evaluating a Spline-Backed TimeSeries

```python
# After spline fitting (e.g., via bp_format.splines.fit_timeseries_spline),
# the TimeSeries will have breaks, coeffs, and segment_start_piece_idx populated.

# Evaluate at a single time
value_at_3 = ts_with_spline.evaluate(3.0)

# Evaluate at multiple times
t_dense = jnp.linspace(0.0, 8.0, 100)
values_dense = ts_with_spline.evaluate_many(t_dense)
```

### Arithmetic on TimeSeries

```python
# Add two time series (exact if both have splines)
combined = ts_biomass + ts_product

# Scale by a constant
doubled = ts_biomass * 2.0

# Compute a ratio (approximate if both have splines)
yield_ratio = ts_product / ts_biomass
```

### Computing Derivatives

```python
# Get the first derivative (rate of change)
rate = ts_with_spline.deriv(order=1)

# Evaluate the derivative at specific times
rate_values = rate.evaluate_many(jnp.array([1.0, 3.0, 5.0]))
```

### Serialization Round-Trip

```python
# To dict (for JSON serialization)
data = ts.to_dict()

# From dict
ts_restored = TimeSeries.from_dict(data)
```

## See Also

- [Data Model](02_data_model.md) -- where TimeSeries fits in the hierarchy
- [Splines](07_splines.md) -- spline fitting and pseudobatch transformation
- [Serialization](03_serialization.md) -- full dataset JSON I/O
- [Design Rationale](01_design_rationale.md#4-timeseries-structure) -- why this structure and optional spline state
