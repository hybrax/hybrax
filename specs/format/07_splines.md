# Splines and Pseudobatch Transform

Source: `bp_format/splines.py`

## Purpose

This module provides the pseudobatch transformation pipeline and spline fitting infrastructure for converting discrete concentration measurements from fed-batch / sample-driven bioprocesses into continuous-time representations. It bridges the gap between raw experimental data and the smooth, differentiable functions needed for ODE integration and gradient-based optimization.

The module handles three related tasks:

1. **Pseudobatch normalization** — transforms fed-batch concentrations to batch-equivalent pseudo-concentrations by removing dilution and feed-addition effects.
2. **Smooth spline fitting** — fits smoothing `TimeSeries` splines to the resulting pseudo-concentrations (which are smooth by design).
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

`S(t)` is exposed as `pseudobatch_transform.sample_compensation` when the
process-level pseudobatch bundle is materialized. Compatibility plotting arrays
may be present in lower-level helper payloads, but runtime evaluation uses
`TimeSeries.evaluate`.

### Exact TimeSeries break grid

Volume, feeds, ADF, and `fc` are represented on an exact `TimeSeries` break grid
that includes:

- every measurement time,
- every event time,
- every reference time of every `TimeSeries` in `volume_changes`,

There are no `t_event ± eps` knots. Discontinuities are represented by local
polynomial pieces plus `continuity_side="left"` semantics.

## Implementation pipeline

1. **`build_pseudobatch_inputs(process, species_name)`** — builds lower-level
   `TimeSeries` objects for reactor volume, accumulated feed, sample
   compensation, ADF, and feed correction. It computes `c_star = meas_conc ·
   adf − feed_corr` at measurement times.
2. **`build_pseudobatch_transform(process, species_names)`** — materializes the
   JSON-facing schema: shared `pseudobatch_transform.adf`, species-keyed
   `pseudobatch_transform.feed_corrections`, optional helper traces, derived
   `volume.total_volume`, and each component's `c_star_concentration`. Raw
   real concentration remains in `component.concentration`.
3. **`build_splines(inputs, process, species_name)`** — builds a lower-level
   runtime backtransform payload for `(meas_times, c_star)`. Stored
   `TimeSeries` carriers use the common smoothing-B-spline policy: segments
   with at least four points use SciPy cubic smoothing B-splines
   (`smoothing_s=0` is exact/interpolating), while shorter segments fall back
   to interpolating `CubicSpline`.
4. **`evaluate_pseudobatch_transform(process, component, times)`** — evaluates
   `c(t) = (c*(t) + fc(t)) / ADF(t)` from the stored component-level c* and the
   process-level transform bundle.

## Design rationale

### Why separate `c*` and `fc` in the spline representation?

The key insight of pseudobatch is that `c*(t)` is smooth while `c(t)` is not.
By splining `c*` and reconstructing `c = (c* + fc) / ADF` with canonical
`TimeSeries` ADF/feed-correction objects, we avoid fitting the concentration
spline through discontinuities. Bolus jumps and sample-time batch-equivalence
shifts are encoded in the `TimeSeries` pieces; the `c*` spline covers only the
smooth pseudobatch dynamics.

### Feed correction invariant

Feed correction is built from the simplified physical invariant:

```python
dFC = S(t) * dF * C_feed / V_init
```

For continuous feeds this is integrated piecewise. For boluses, the exact jump
uses the sample-first value of `S(t)` at the event timestamp.

### Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `DEFAULT_MAX_SEGMENTS` | `16` | Maximum number of segments in padded storage |
| Smoothing B-spline minimum samples | `4` | Segments with fewer samples fall back to interpolating `CubicSpline` |

## Public API

### Core spline infrastructure

| Function | Description |
|----------|-------------|
| `fit_timeseries_spline(ts, boundaries, smoothing_s)` | Fit segmented spline state onto a `TimeSeries`. Segments with at least four points use SciPy cubic smoothing B-splines; `smoothing_s=0` gives an exact/interpolating fit. Shorter segments fall back to interpolating `CubicSpline`. |
| `make_cubic_ppoly(t, y, bc_type)` | Create an owned `PPoly` cubic spline from arrays. |
| `make_constant_spline(value, t_start, t_end)` | Create a constant-valued spline over a time range. |

### Evaluation

| Function | Description |
|----------|-------------|
| `TimeSeries.evaluate(t)` | Evaluate a spline-backed `TimeSeries` at a single time point. |
| `TimeSeries.evaluate_many(t)` | Evaluate a spline-backed `TimeSeries` on a 1-D time grid. |

### Segmentation

| Function | Description |
|----------|-------------|
| `detect_discrete_state_events(process)` | Extract discrete event times from a `BioProcess`. Returns `DiscreteEvents`. |
| `make_segment_boundaries(t_min, t_max, event_times)` | Build segment breakpoints from event times. |
| `split_timeseries(ts, boundaries)` | Partition a `TimeSeries` into segments at the given boundaries. |

### Pseudobatch pipeline

| Function | Description |
|----------|-------------|
| `build_pseudobatch_inputs(process, species_name)` | Build canonical pseudobatch `TimeSeries` objects and measurement-level `c_star`, `adf_at_meas`, and `feed_corr_at_meas`. |
| `build_pseudobatch_transform(process, species_names, cstar_smoothing_s)` | Build process-level pseudobatch storage: `adf`, `feed_corrections`, optional helper traces, `volume.total_volume`, and component-level `c_star_concentration`. |
| `build_splines(inputs, process=None, species_name=None, cstar_smoothing_s=0.0)` | Build lower-level runtime spline payloads from `build_pseudobatch_inputs`; mainly useful for tests/internal pipelines. |
| `to_timeseries(inputs, splines, species_name, cstar_smoothing_s=0.0)` | Convert lower-level pseudobatch samples to a transformed `TimeSeries` carrier. |
| `evaluate_pseudobatch_transform(process, component, times)` | Evaluate stored c* as real concentration using `component.c_star_concentration`, `pseudobatch_transform.adf`, and `feed_corrections[component]`. |

### JAX-compatible backtransform classes

#### `BacktransformSpline`

An `eqx.Module` that evaluates the inverse pseudobatch transform at any time:

```python
c(t) = (c*(t) + feed_correction(t)) / ADF(t)
```

Fields:
- `c_star_spline` — owned `PPoly` view of the stored `TimeSeries` c* spline
- ADF and feed-correction `TimeSeries` — internal canonical transform fields
  sourced from `pseudobatch_transform.adf` and `feed_corrections[species]`
- derivative `TimeSeries` for smooth RHS terms
- `is_constant` — bypass flag for constant-concentration species
- `constant_value` — returned directly when `is_constant = True`

Methods:
- `__call__(t)` — evaluate backtransformed concentration
- `derivative()` — return a callable for `dc/dt`

Build with `build_backtransform_spline(process, species_name)` from a process
that has `pseudobatch_transform` and component-level `c_star_concentration`.

#### `BatchedBacktransformSpline`

Stacks multiple `BacktransformSpline` objects for vectorised evaluation across species. Used inside JIT-compiled ODE solvers.

Build with `build_batched_conc_splines(process, species_names)`.

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

# Build JSON-facing pseudobatch storage for one species
transform = bp.splines.build_pseudobatch_transform(process, ["glucose"])
process.pseudobatch_transform = transform
series = process.reactor_medium.components["glucose"].c_star_concentration
```

### Inspecting ADF and `S(t)`

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

times = jnp.linspace(process.time_axis.start, process.time_axis.end, 500)
transform = process.pseudobatch_transform

fig, ax1 = plt.subplots()
ax1.plot(times, transform.adf.evaluate_many(times), label="ADF")
ax2 = ax1.twinx()
if transform.sample_compensation is not None:
    ax2.plot(
        times,
        transform.sample_compensation.evaluate_many(times),
        color="tab:green",
        label="S(t)",
    )
ax1.set_xlabel("time"); ax1.set_ylabel("ADF"); ax2.set_ylabel("S(t)")
```

### Evaluating backtransformed concentrations

```python
import jax.numpy as jnp
from bp_format.splines import evaluate_pseudobatch_transform

# Evaluate at many times from stored c*, feed correction, and ADF.
t_dense = jnp.linspace(0.0, 48.0, 500)
c_dense = evaluate_pseudobatch_transform(process, "glucose", t_dense)
```

### Using `BacktransformSpline` in a JIT-compiled function

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from bp_format.splines import build_backtransform_spline

bt = build_backtransform_spline(process, "glucose")

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

Outliers beyond this range (e.g. `10_martens_2025_f` glucose at its sharpest peak) are a sampling-density limitation: 9 sparse measurement points cannot fully constrain a cubic spline through rapid transitions. They are not pipeline bugs.

## See also

- [TimeSeries](06_time_series.md) — the underlying data container
- [Mechanistic](08_mechanistic.md) — consumes splines for ODE integration
- [Data Model](02_data_model.md) — `TimeSeries`, `VolumeChange`, and `DiscreteEvents`
- [Design Rationale](01_design_rationale.md#5-pseudobatch-normalization) — mathematical background
- [`pseudobatch`](https://github.com/viktorht/pseudobatch) — upstream library (`accumulated_dilution_factor`, `pseudobatch_transform`)
