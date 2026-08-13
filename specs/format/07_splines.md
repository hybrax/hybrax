# Splines

Source: `bp_format/splines.py`

## Purpose

Turn a sparse, irregularly-sampled `TimeSeries` (measured concentrations, feed
rates, process-variable readings) into a smooth, differentiable function of
time, without pretending away discontinuities from discrete events (sampling,
bolus feeds). Two jobs:

1. **Discrete-event detection and segmentation** — find event times from a
   process's non-continuous volume changes, and split a `TimeSeries` at those
   times so the fit never draws a smooth curve through a jump.
2. **Spline fitting** — fit an interpolating or smoothing cubic to each
   segment, stored as `TimeSeries` spline state (`breaks`/`coeffs`, power-basis
   form).

## Discrete-event detection and segmentation

| Function | Description |
|----------|-------------|
| `detect_discrete_state_events(process)` | Event times from every non-continuous (`is_continuous=False`) volume change, as `DiscreteEvents`. |
| `make_segment_boundaries(t_min, t_max, event_times)` | `[t_min, …interior events…, t_max]`, strictly increasing. |
| `split_timeseries(ts, boundaries)` | Split a `TimeSeries` into segments defined by `boundaries`; a point exactly on a boundary belongs to both neighbouring segments. |

```python
events = bp.splines.detect_discrete_state_events(process)
bounds = bp.splines.make_segment_boundaries(
    process.time_axis.start, process.time_axis.end, events.times
)
fitted = bp.splines.fit_timeseries_spline(raw_series, boundaries=bounds)
```

## Spline fitting

`fit_timeseries_spline(ts, *, boundaries=None, smoothing_s=0.0)` is the main
entry point. It fits each segment independently, so a fit never smooths across
an event boundary:

| Points in a segment | Method |
|---------------------|--------|
| ≥ 4 | SciPy smoothing B-spline (`make_splrep`, cubic). `smoothing_s=0` makes it interpolating. |
| 2–3 | Interpolating natural `CubicSpline` — a cubic smoothing B-spline needs 4 samples. |
| 1 | Constant piece over a `1e-6`-wide interval. |

The chosen strategy per segment is recorded in `metadata["fit_strategies"]`
(`"mixed"` in `metadata["fit_strategy"]` if segments disagree). Fitted
segments are flattened into one set of `breaks`/`coeffs` on the returned
`TimeSeries`, with `segment_start_piece_idx` marking where each segment's
pieces begin.

Two smaller standalone builders:

| Function | Description |
|----------|-------------|
| `make_constant_spline(value, t_start, t_end)` | A constant spline-backed `TimeSeries` over `[t_start, t_end]`, tagged `metadata["is_constant"]`. |
| `make_cubic_ppoly(t, y, bc_type="natural")` | A cubic `PPoly` directly from raw arrays; sorts and deduplicates knots. Used internally by `mechanistic.py` for state-spline construction outside the segmented-fitting path. |

## Example: fit a measured concentration series

```python
import bp_format as bp

fitted = bp.splines.fit_timeseries_spline(process.reactor_medium.components["glucose"].concentration)
t = jnp.linspace(process.time_axis.start, process.time_axis.end, 500)
c = fitted.evaluate_many(t)
```

## See also

- [TimeSeries](06_time_series.md) — the container being fitted
- [Mechanistic](08_mechanistic.md) — consumes these splines as state trajectories
