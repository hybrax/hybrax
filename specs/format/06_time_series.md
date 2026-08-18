# TimeSeries

Source: `bp_format/time_series/`

## Purpose

`TimeSeries` is the container for everything that varies over time: measured
concentrations, cumulative feed traces, and process signals. It holds
**discrete samples, a fitted spline, or both**.

It is an `eqx.Module`, so it passes through `jax.jit` / `jax.grad` / `jax.vmap`
untouched — an ODE solver can evaluate a stored spline inside a compiled step
function.

`TimeSeries` does not know whether its data is continuous. That comes from the
parent object: a `VolumeChange` with `is_continuous=True` holds a flow profile,
one with `is_continuous=False` holds discrete events where a spline would be
meaningless.

## Design notes

- **Power-basis coefficients.** A piece is `[a, b, c, d]`, evaluated as
  `a + h(b + h(c + h·d))` with `h = t − t_break`. Horner form, a handful of
  fused multiply-adds, trivially vectorized. B-spline evaluation would need
  recursive knot lookups.
- **`jump_times`** records where the signal genuinely discontinues (bolus
  events), so segmentation and event handling can find them without re-deriving
  them from the volume changes.
- **`continuity_side`** decides which piece wins exactly at a breakpoint.
  `"right"` (default) gives the post-event value; `"left"` gives the pre-event
  value — useful for any series that must read as its pre-event value exactly
  at an event time, with the jump applying immediately after.
- **`derived`** marks a series as computed rather than measured, so downstream
  code can tell a fitted rate from raw experimental data.
- **float64 only.** Constructing a `TimeSeries` from float32 arrays raises
  rather than upcasting silently — see
  [Design Rationale §1](01_design_rationale.md#1-jax-first-but-only-where-it-matters).

## The class

```python
class TimeSeries(eqx.Module):
    times: jnp.ndarray | None                    # strictly increasing, 1-D
    values: jnp.ndarray | None                   # same length as times
    breaks: jnp.ndarray | None                   # spline: n_pieces + 1
    coeffs: jnp.ndarray | None                   # spline: (n_pieces, 4)
    segment_start_piece_idx: jnp.ndarray | None  # spline: segment start pieces
    jump_times: jnp.ndarray                      # default: empty
    derived: bool = False                        # static
    continuity_side: str = "right"               # static, "left" | "right"
    metadata: Any = None                         # static
```

All constructor arguments are keyword-only.

**Invariants, enforced at construction:**

- At least one of {samples, spline} must be present.
- Samples: `times` and `values` both given, both 1-D, equal length, `times`
  strictly increasing.
- Spline: `breaks`, `coeffs`, and `segment_start_piece_idx` all given.
  `coeffs` is `(n_pieces, 4)` with `n_pieces >= 1`; `len(breaks) == n_pieces + 1`;
  `breaks` strictly increasing.
- `segment_start_piece_idx` starts at `0`, is strictly increasing, and stays in
  range. It marks which piece each independently-fitted segment begins at.
- Passing a `poly=` argument instead of `breaks`/`coeffs` is allowed; its
  `continuity_side` must match.

## Methods and properties

| Member | Description |
|--------|-------------|
| `poly` | The spline as a `PPoly`, or `None` if no spline is stored. |
| `dtype` | Floating dtype of the arrays (always float64). Read-only. |
| `evaluate(t, *, side=None)` | Evaluate the spline at one time. |
| `evaluate_many(ts, *, side=None)` | Evaluate the spline on a 1-D grid. |
| `lin_interp(t)` | Linear interpolation of the **samples**, ignoring any spline. |
| `deriv(order=1)` | New `TimeSeries` holding the derivative spline. |
| `integrate(a, b)` | Definite integral of the spline over `[a, b]`. |
| `to_pd_series()` | Samples as a pandas `Series` indexed by time. |
| `to_dict()` / `TimeSeries.from_dict(d)` | Canonical dict round trip. |
| `TimeSeries.from_process_state(state, variable)` | Build from a `metadata.hybrax.process_state` payload. |
| `TimeSeries.from_input_dict(data, process_key, variable)` | Same, from a full input JSON. |

`evaluate*` raise if there is no spline; `lin_interp` and `to_pd_series` raise
if there are no samples.

## Arithmetic

`TimeSeries` supports `+`, `-`, `*`, `/` with another `TimeSeries` or with a
scalar. Which path runs depends on what the operands carry:

| Case | Behaviour |
|------|-----------|
| `+` / `-`, both have splines | **Exact.** Breakpoints are merged and coefficients added — a sum of cubics is a cubic. |
| `*` / `/`, both have splines | **Approximate.** A product of cubics is degree 6, so the result is re-fitted as a cubic on a merged grid, tightening the smoothing parameter until the mean relative error is within `APPROX_REL_ERR_TARGET`. Raises if it cannot converge in `APPROX_MAX_REFIT_ATTEMPTS`. |
| Either operand lacks a spline | **Discrete.** Both are linearly interpolated onto a merged time grid; the result has samples only. Warns if exactly one operand had a spline, since that spline is being discarded. |
| Scalar `*` / `/` | **Exact.** Coefficients and values are scaled directly. |

Division raises `ZeroDivisionError` if the denominator reaches or crosses zero
(checked both on the sample values and across the spline pieces). Results are
marked `derived=True` and carry `metadata["source"]` recording which path ran.

Both operands must share a `continuity_side` and dtype.

## `PPoly`

`bp_format.time_series.PPoly` is the bare spline evaluator — an `eqx.Module`
with `breaks` and `coeffs` and no measurement data.

| Member | Description |
|--------|-------------|
| `PPoly(breaks, coeffs, continuity_side="right")` | Construct directly. |
| `PPoly.from_scipy_ppoly(ppoly, ...)` | Convert a SciPy `PPoly`; pads lower-degree pieces to cubic and drops zero-width pieces. |
| `PPoly.from_samples_pchip(t, y, ...)` | Build via SciPy's PCHIP interpolator. |
| `__call__(t, nu=0, side=None)` | Evaluate; `nu` is the derivative order. |
| `derivative(order=1)` | New `PPoly` of the derivative. |

`TimeSeries.poly` returns one of these; `ControlSplines` stores them.

## Helper modules

### `spline_ops` — spline primitives

| Function | Description |
|----------|-------------|
| `piece_index(t, breaks, side)` | Which piece contains `t`. |
| `evaluate_piece(coeff_row, dt)` | One cubic piece, Horner form. |
| `evaluate_scalar` / `evaluate_many` | Full spline at one time / a grid. |
| `rebase_piece(coeff_row, shift)` | Shift a piece's local origin. |
| `rebase_to_breaks(old_breaks, old_coeffs, new_breaks)` | Re-express a spline on a finer breakpoint grid, exactly. |
| `derivative_coeffs(coeffs, order)` | Differentiate coefficients. |
| `integrate_definite(breaks, coeffs, a, b)` | Definite integral. |
| `merge_breaks(a, b)` | Union of two breakpoint arrays. |
| `merge_segment_starts(...)` | Merge segment boundaries of two splines. |
| `has_near_zero_piece_value(breaks, coeffs, threshold)` | Whether any piece approaches zero — the division guard. |
| `validate_side(side)` | Reject anything other than `"left"` / `"right"`. |

### `grid_utils` — time-grid operations

| Function | Description |
|----------|-------------|
| `merge_times_with_tolerance(a, b)` | Merge sorted time arrays, deduplicating near-identical timestamps. |
| `linear_interpolate_samples(times, values, target)` | Linear interpolation, clamped at both ends. |
| `synthesize_binary_samples(t_a, v_a, t_b, v_b, op)` | Sample grid for a binary operation. |

### `io` — dict conversion

`timeseries_to_dict`, `timeseries_from_dict`, `timeseries_from_process_state`,
`timeseries_from_input_dict`. Reached through the `TimeSeries` class methods
above; see [03_serialization.md](03_serialization.md) for the full dataset I/O.

### `constants` — numerical defaults

| Constant | Value | Purpose |
|----------|-------|---------|
| `TIME_DEDUP_ATOL` | `1e-8` | Absolute tolerance when merging time grids |
| `TIME_DEDUP_RTOL` | `1e-7` | Relative tolerance when merging time grids |
| `APPROX_ABS_FLOOR` | `1e-8` | Error-scale floor for approximate refits |
| `APPROX_REL_ERR_TARGET` | `1e-3` | Accepted mean relative error of an approximate refit |
| `APPROX_INITIAL_S` | `1.0` | Starting smoothing parameter |
| `APPROX_S_REDUCTION_FACTOR` | `0.5` | Smoothing reduction per retry |
| `APPROX_MAX_REFIT_ATTEMPTS` | `8` | Retries before giving up |
| `DIVISION_NEAR_ZERO_THRESHOLD` | `1e-8` | Denominator floor for division |

## Examples

### From experimental data

```python
import jax.numpy as jnp
from bp_format import TimeSeries

ts = TimeSeries(
    times=jnp.array([0.0, 2.0, 4.0, 6.0, 8.0]),
    values=jnp.array([1.0, 2.5, 5.1, 4.2, 3.0]),
)
ts.lin_interp(3.0)      # works: samples are present
ts.evaluate(3.0)        # raises: no spline yet
```

### Fitting and evaluating

```python
from bp_format.splines import fit_timeseries_spline

fitted = fit_timeseries_spline(ts)          # smoothing_s=0 -> interpolating
fitted.evaluate(3.0)
fitted.evaluate_many(jnp.linspace(0.0, 8.0, 100))
fitted.integrate(0.0, 8.0)
```

### Derivatives

```python
rate = fitted.deriv(order=1)                # derived=True
rate.evaluate_many(jnp.array([1.0, 3.0, 5.0]))
```

### Arithmetic

```python
total    = ts_biomass + ts_product          # exact if both have splines
doubled  = ts_biomass * 2.0                 # exact
specific = ts_product / ts_biomass          # approximate refit
specific.metadata["source"]                 # "approx_binary_op"
```

### Round trip

```python
restored = TimeSeries.from_dict(ts.to_dict())
```

## See also

- [Data Model](02_data_model.md) — where `TimeSeries` sits in the hierarchy
- [Splines](07_splines.md) — segmented spline fitting
- [Serialization](03_serialization.md) — dataset-level JSON I/O
