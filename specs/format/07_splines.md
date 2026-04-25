# Splines and Pseudobatch Transform

Source: `bp_format/splines.py`

## Purpose

This module provides the pseudobatch transformation pipeline and spline fitting infrastructure for converting discrete concentration measurements from fed-batch / sample-driven bioprocesses into continuous-time representations. It bridges the gap between raw experimental data and the smooth, differentiable functions needed for ODE integration and gradient-based optimization.

The module handles three related tasks:

1. **Pseudobatch normalization** — transforms fed-batch concentrations to batch-equivalent pseudo-concentrations by removing dilution and feed-addition effects.
2. **Smooth spline fitting** — fits a cubic or PCHIP spline to the resulting pseudo-concentrations (which are smooth by design).
3. **Backtransformation** — reconstructs real-space concentrations from the pseudo-concentration spline, the accumulated dilution factor (ADF), and the feed-correction trajectory.

## Mathematical background

### The pseudobatch identity

For a fed-batch bioreactor under mass balance, the real-space concentration `c(t)` of a species is related to a "pseudobatch" concentration `c*(t)` via

```
c(t) = ( c*(t) + fc(t) ) / ADF(t)
```

with the inverse transform

```
c*(t) = c(t) · ADF(t) − fc(t)
```

where

- **`ADF(t)` — Accumulated Dilution Factor.** A multiplicative factor that
  re-normalises species concentrations to a common reference volume so that
  physical dilution of the broth is cancelled out. In the active
  implementation it is represented as a `TimeSeries` with a piecewise-linear
  baseline plus instantaneous bolus jumps: continuous feed changes ADF
  smoothly, bolus feed adds true jumps, and pure sampling leaves ADF flat.
- **`fc(t)` — feed correction.** The cumulative mass of the species that has been added by all feeds (continuous + bolus) up to time `t`, discounted by the ADF at each addition and normalised by the reactor volume:
  ```
  fc(t) = cumsum_streams( ADF · Δaccumulated_feed · c_in_feed / V_reactor )
  ```
  `fc` grows smoothly due to continuous feed and jumps instantaneously at each bolus event.
- **`c*(t)` — pseudobatch-space concentration.** The "as-if it were a batch with no feed and no sampling" concentration. By construction `c*` is **smooth at every event** (continuous feed, bolus, sample) provided ADF satisfies the smoothness invariant below. This is what makes `c*` a good target for spline fitting — unlike `c(t)`, which has discontinuities at bolus and sample events, `c*` is continuous.

### The pseudobatch smoothness invariant

At any event at time `t_event`, the pseudobatch framework requires

```
ADF_post / ADF_pre  =  V_reactor_post / V_reactor_before_bolus
```

where `V_reactor_before_bolus` is the reactor volume *after any simultaneous sampling but before the bolus addition*. When this invariant holds, `c*(t)` is continuous across every event and can be fitted with a smooth spline. When it is violated, the cubic spline averages hidden discontinuities and the reconstructed `c(t)` picks up spurious sample-time steps or mis-sized bolus jumps.

### Computing ADF — the sample-compensation factor `S(t)`

Sampling removes volume but does **not** change concentration, so ADF must be *unchanged* across sample events. At the same time, any future bolus ratio must reflect the now-smaller reactor volume. This is achieved with a sample-compensation factor `S(t)`:

```
S(t)    = ∏  over samples at time ≤ t of  V_before_sample / V_after_sample
V_eff(t) = V_reactor_actual(t) · S(t)
ADF(t)  = V_eff(t) / V_init
```

Event-by-event behaviour:

| Event | `V_reactor` | `S(t)` | `V_eff` / ADF | ratio |
|-------|-------------|--------|---------------|-------|
| continuous feed | grows | unchanged | grows proportionally | `V_post/V_pre` (smoothness invariant) |
| sample only | drops by `V_s` | multiplied by `V_pre/(V_pre−V_s)` | unchanged → ADF held | 1 |
| bolus only | grows by `V_b` | unchanged | grows | `(V_pre+V_b)/V_pre` |
| simultaneous sample + bolus (sample first, then bolus) | net `V_b−V_s` | multiplied by `V_pre/(V_pre−V_s)` | steps by `(V_pre−V_s+V_b)/(V_pre−V_s)` | correct physics |

`S(t)` is exposed as `inputs["sample_compensation_dense"]` from `build_pseudobatch_inputs` and is also plotted alongside ADF in the `examples/00_combined/03_pseudobatch_splines/` scripts.

### Dense time grid

Volume, feeds, ADF and `fc` are evaluated on a **dense time grid** that includes:
- every measurement time,
- every event time plus `t_event ± _EPS` knots (for sharp event representation),
- every reference time of every `TimeSeries` in `volume_changes`,
- a background `linspace` of at least 500 evenly-spaced knots across `[t_start, t_end]`.

The background densification is required because linear interpolation of ADF between sparse knots would otherwise render smooth continuous-feed growth as a huge apparent jump right after each event.

## Implementation pipeline

1. **`build_pseudobatch_inputs(process, species_name)`** — builds the dense grid, computes `reactor_volume_dense`, `accumulated_feed_dense`, `sample_volume_dense`, `concentration_in_feed`, then derives `adf_dense` via the sample-compensation formula, `adf_at_meas`, `feed_corr_at_meas`, `feed_corr_dense`, and `c_star = meas_conc · adf − cumsum(feed_term)` at the measurement times. Returns a dict.
2. **`build_splines(inputs, process, species_name)`** — fits a cubic spline (or PCHIP, see below) through `(meas_times, c_star)`; calibrates the dense feed-correction trajectory to anchor exactly at `feed_corr_at_meas` at every measurement index; extracts deterministic bolus jumps at each event's ε-pair; and splits both feed correction and ADF into a smooth baseline plus explicit jump metadata. Returns a runtime payload dict with `spline_cstar`, `adf_base_*`, `adf_jump_*`, `feed_corr_base_*`, `feed_corr_jump_*`.
3. **`to_timeseries(inputs, splines, species_name)`** — converts the runtime pseudobatch payload into the canonical transformed `TimeSeries` carrier with nested `metadata["transform"]["series"]` payloads for `adf_ts` and `feed_corr_ts`.
4. **`evaluate_real_concentration(t_eval, splines)`** — evaluates `c(t) = (cs(t) + fc(t)) / ADF(t)` with both ADF and discrete feed correction in `linear_plus_step` form (piecewise-linear base + instantaneous jumps).

### Cubic vs PCHIP fallback

`build_splines` starts with an `interpax.CubicSpline` through `(meas_times, c_star)`. It then checks a dense evaluation of the spline and switches to a PCHIP (monotonicity-preserving) spline when either:

- the cubic goes negative on any dense point while the raw `c_star` values are all non-negative, or
- the cubic overshoots the raw `c_star` range by more than 5 % of the data span.

The overshoot check catches processes with stepwise `c_star` (e.g. bolus-only processes with no continuous feed, where `c_star = meas · ADF` is essentially piecewise constant). In such cases a natural cubic spline exhibits Gibbs-style oscillation between knots; PCHIP traces the staircase monotonically.

## Design rationale

### Why separate `c*` and `fc` in the spline representation?

The key insight of pseudobatch is that `c*(t)` is smooth while `c(t)` is not. By splining `c*` (smooth) and reconstructing `c = (c* + fc) / ADF` with `fc` and `ADF` computed analytically on the dense grid, we avoid ever fitting a spline to data that contains discontinuities. Bolus jumps and sample-time "batch equivalence" shifts are carried exactly in the `fc_jump_*` and `ADF` components; the spline only has to cover the smooth pseudobatch dynamics.

### Why anchor `fc` to meas values?

The pseudobatch invariant `c(t_meas) = meas_conc` requires that `c_star_spline(t_meas) + fc(t_meas) = meas_conc · ADF(t_meas)` exactly. The module enforces this by rescaling the dense `fc` trajectory within each inter-meas interval so that its endpoints match `feed_corr_at_meas` at the anchor times. Between anchors the calibrated trajectory inherits the curvature of the physical cumsum — which captures continuous-feed growth accurately, even for dual-fed species (same species in both continuous and bolus feed streams).

### The epsilon convention

For every discrete event at `t_event`, the dense grid contains `t_event ± _EPS` knots (`_EPS = 1e-4`). Bolus jumps are extracted as `fc_dense[i_post] − fc_dense[i_pre]`. On the dense grid, ADF and `fc` transition across the `2·_EPS`-wide interval in a single step that `jnp.interp` renders indistinguishable from a true discontinuity at the plotting resolution.

### Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `_EPS` | `1e-4` | Pre/post-event offset for discrete event ε-pair knots |
| `DEFAULT_MAX_SEGMENTS` | `16` | Maximum number of segments in padded storage |
| `DEFAULT_MAX_CTRL_POINTS` | `128` | Maximum control points per segment |
| `SMOOTHING_THRESHOLD` | `100` | Switch from interpolation to smoothing spline above this many points |
| Background densification | ≥ 500 pts | Minimum number of linspace knots added to the dense grid |
| PCHIP overshoot threshold | 5 % of data range | Cubic→PCHIP switchover for near-stepwise `c_star` |

## Public API

### Core spline infrastructure

| Function | Description |
|----------|-------------|
| `fit_timeseries_spline(ts, boundaries, max_segments, max_ctrl_points)` | Fit segmented spline state onto a `TimeSeries`. Returns a spline-backed `TimeSeries`. |
| `choose_spline_kind(n_points)` | Returns `"smoothing_bspline"` (> `SMOOTHING_THRESHOLD` pts) or `"cubic_interp"` otherwise. |
| `make_interpax_spline(t, y, bc_type)` | Create an `interpax.CubicSpline` from arrays. |
| `make_pchip_spline(t, y)` | Create a monotonicity-preserving PCHIP spline from arrays. |
| `make_constant_spline(value, t_start, t_end)` | Create a constant-valued spline over a time range. |

### Evaluation

| Function | Description |
|----------|-------------|
| `build_interpax_spline(rep)` | Reconstruct per-segment `interpax.CubicSpline` objects from a spline-backed `TimeSeries`. |
| `evaluate_spline_at(rep, t)` | Evaluate a spline-backed `TimeSeries` at a single time point. |
| `evaluate_left_continuous_step(t, step_times, step_values)` | Left-continuous step function lookup. |
| `evaluate_linear_plus_step(t, base_times, base_values, jump_times, jump_values)` | Piecewise-linear base plus left-continuous instantaneous jumps. |

### Segmentation

| Function | Description |
|----------|-------------|
| `detect_discrete_state_events(process)` | Extract discrete event times from a `BioProcess`. Returns `DiscreteEvents`. |
| `make_segment_boundaries(t_min, t_max, event_times)` | Build segment breakpoints from event times. |
| `split_timeseries(ts, boundaries)` | Partition a `TimeSeries` into segments at the given boundaries. |

### Pseudobatch pipeline

| Function | Description |
|----------|-------------|
| `build_pseudobatch_inputs(process, species_name)` | Build the dense grid, volumes, ADF (via the sample-compensation factor), feed-correction and `c_star` needed for pseudobatch normalisation. Returns a dict including `dense_times`, `adf_dense`, `sample_compensation_dense`, `c_star`, `feed_corr_at_meas`, `feed_corr_dense`. |
| `build_splines(inputs, process=None, species_name=None)` | Build runtime spline payload from the `build_pseudobatch_inputs` output: cubic/PCHIP `spline_cstar`, `feed_corr_base_*` + `feed_corr_jump_*`, and `adf_base_*` + `adf_jump_*`. |
| `to_timeseries(inputs, splines, species_name)` | Convert the in-memory pseudobatch payload to the canonical transformed `TimeSeries` carrier used for serialization and runtime backtransform. |
| `evaluate_real_concentration(t_eval, splines)` | Evaluate the backtransformed real concentration using linear-plus-step ADF and feed-correction semantics. |

### JAX-compatible backtransform classes

#### `BacktransformSpline`

An `eqx.Module` that evaluates the inverse pseudobatch transform at any time:

```python
c(t) = (c*(t) + feed_correction(t)) / ADF(t)
```

Fields:
- `c_star_spline` — interpax `CubicSpline` or `PchipInterpolator` for `c*`
- `adf_times`, `adf_values` — ADF baseline knots/values
- `adf_jump_times`, `adf_jump_values` — instantaneous ADF bolus jumps
- `fc_spline`, `fc_times`, `fc_values` — feed correction baseline
- `fc_jump_times`, `fc_jump_values` — discrete feed-correction jumps (linear_plus_step mode)
- `is_constant` — bypass flag for constant-concentration species
- `constant_value` — returned directly when `is_constant = True`

Methods:
- `__call__(t)` — evaluate backtransformed concentration
- `derivative()` — return a callable for `dc/dt`

Build with `build_backtransform_spline(rep)` from a transformed `TimeSeries`.

#### `BatchedBacktransformSpline`

Stacks multiple `BacktransformSpline` objects for vectorised evaluation across species. Used inside JIT-compiled ODE solvers.

Build with `build_batched_conc_splines(conc_splines, species_names, t_start, t_end)`.

## Examples

### Detecting discrete events

```python
import bp_format as bp

dataset = bp.serialization.load_dataset("data.json")
process = dataset.case_studies["kittler_2022"].processes["batch_001"]

events = bp.splines.detect_discrete_state_events(process)
print(events.times)   # array of event times
print(events.labels)  # optional labels
```

### Fitting splines to a process

```python
import bp_format as bp

dataset = bp.serialization.load_dataset("data.json")
process = dataset.case_studies["martens_2025_f"].processes["run_1"]

# Build pseudobatch inputs + runtime spline payload for one species
inputs = bp.splines.build_pseudobatch_inputs(process, "glucose")
splines = bp.splines.build_splines(inputs, process, "glucose")
series = bp.splines.to_timeseries(inputs, splines, "glucose")
```

### Inspecting ADF and `S(t)`

```python
import matplotlib.pyplot as plt

inputs = bp.splines.build_pseudobatch_inputs(process, "glucose")

fig, ax1 = plt.subplots()
ax1.plot(inputs["dense_times"], inputs["adf_dense"], label="ADF")
ax2 = ax1.twinx()
ax2.plot(inputs["dense_times"], inputs["sample_compensation_dense"],
         color="tab:green", label="S(t)")
ax1.set_xlabel("time"); ax1.set_ylabel("ADF"); ax2.set_ylabel("S(t)")
```

### Evaluating backtransformed concentrations

```python
import jax
import jax.numpy as jnp
from bp_format.splines import build_backtransform_spline

bt = build_backtransform_spline(series)

# Evaluate at a single time (works inside jax.jit)
c_glucose = bt(5.0)

# Evaluate at many times
t_dense = jnp.linspace(0.0, 48.0, 500)
c_dense = jax.vmap(bt)(t_dense)
```

### Using `BacktransformSpline` in a JIT-compiled function

```python
import equinox as eqx
import jax

@eqx.filter_jit
def eval_concentrations(bt_spline, times):
    return jax.vmap(bt_spline)(times)

c_values = eval_concentrations(bt, jnp.linspace(0.0, 48.0, 100))
```

## Accuracy notes

On the three real datasets `10_martens_2025_f`, `12_martens_expanded`, and `02_gotsmy_2023`, the end-to-end backtransform is accurate to:

- `abs@meas ≤ 1e-5` at every measurement point (pseudobatch-math invariant).
- `rel@gt_p90 < 1 %` between measurements on dense ground-truth CSVs for the fed species (`glucose`, `glutamine`) on all martens datasets.
- `rel@gt_p99 < 5 %` for all species on dense continuous-feed datasets (`12_martens_expanded`).

Outliers beyond this range (e.g. `10_martens_2025_f` glucose at its sharpest peak) are a sampling-density limitation: 9 sparse meas points cannot fully constrain a cubic/PCHIP spline through rapid transitions. They are not pipeline bugs.

## See also

- [TimeSeries](06_time_series.md) — the underlying data container
- [Mechanistic](08_mechanistic.md) — consumes splines for ODE integration
- [Data Model](02_data_model.md) — `TimeSeries`, `VolumeChange`, and `DiscreteEvents`
- [Design Rationale](01_design_rationale.md#5-pseudobatch-normalization) — mathematical background
- [`pseudobatch`](https://github.com/viktorht/pseudobatch) — upstream library (`accumulated_dilution_factor`, `pseudobatch_transform`)
