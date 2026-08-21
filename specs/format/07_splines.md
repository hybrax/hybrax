# Splines and the Pseudobatch Transform

Source: `src/hybrax/format/splines.py`

## Purpose

Turn sparse offline concentration measurements from a fed-batch run into smooth,
differentiable functions of time. Three jobs:

1. **Pseudobatch transform** — strip dilution and feed addition out of the
   measured concentrations.
2. **Spline fitting** — fit a smooth cubic to what is left, which is now
   actually smooth.
3. **Backtransform** — reconstruct the real concentration (and its derivative)
   from that spline at any time.

## The problem

A measured concentration in a fed-batch reactor moves for two unrelated reasons:
the cells consumed or produced something, and the broth was diluted or sampled.
Fitting a spline directly to `c(t)` therefore means fitting through instantaneous
jumps at every bolus — the spline rings, and the implied `dc/dt` is wrong
everywhere near the event.

## The pseudobatch identity

```
c*(t) = c(t) · ADF(t) − fc(t)          forward
c(t)  = (c*(t) + fc(t)) / ADF(t)       inverse
```

- **`ADF(t)`** — *accumulated dilution factor*. Rescales concentrations to the
  initial reactor volume, cancelling physical dilution.
- **`fc(t)`** — *feed correction*. The species mass delivered by all feeds up to
  `t`, expressed on the same normalized basis.
- **`c*(t)`** — the concentration this species *would* have had in a batch
  reactor with identical biology. Continuous across feeds, boluses, and samples,
  so it is a well-behaved spline target.

The discontinuities have not disappeared — they moved into `ADF` and `fc`, which
are represented exactly rather than fitted.

## How `ADF` is built

Sampling removes volume but does **not** change any concentration, so `ADF` must
stay flat across a sample. At the same time, a later bolus must be sized against
the now-smaller reactor. A sample-compensation factor `S(t)` reconciles the two:

```
S(t)     = ∏  over samples up to t  of  V_before_sample / V_after_sample
V_eff(t) = V_reactor(t) · S(t)
ADF(t)   = V_eff(t) / V_init
```

| Event | `V_reactor` | `S(t)` | `ADF` |
|-------|-------------|--------|-------|
| continuous feed | grows | unchanged | grows smoothly |
| sample only | drops by `V_s` | × `V_pre / (V_pre − V_s)` | **unchanged** |
| bolus only | grows by `V_b` | unchanged | jumps by `(V_pre + V_b) / V_pre` |
| sample + bolus at the same time | net `V_b − V_s` | × `V_pre / (V_pre − V_s)` | jumps by `(V_pre − V_s + V_b) / (V_pre − V_s)` |

The last row is why event order matters: **sample first, then bolus**. The
offline measurement describes the broth as drawn, and the bolus dilutes what
remains.

`S(t)` is kept as `pseudobatch_transform.sample_compensation` for inspection.

## Scope: Inflow and discrete Outflow only

The `ADF(t) = V(t) · S(t) / V_init` identity above is only a valid solution to
the required growth-rate ODE (`dADF/dt = ADF · Fin(t) / V(t)`) because `V(t)`
is built from continuous **Inflows** alone (`dV/dt = Fin`) — plugging that same
numerator back in makes the identity self-consistent. A continuous **Outflow**
(perfusion, continuous harvest/bleed) changes the real reactor volume too
(`dV/dt = Fin − Fout`, matching `mechanistic._apply_feed_dilution`), and the
required ADF growth rate would then need `1/V(t)` for a genuine cubic `V(t)`
— this does not integrate to a polynomial (or any simple closed form) in
general, regardless of `Outflow.retention`.

Rather than silently produce wrong `ADF`/`c*` values for such a process,
`build_pseudobatch_transform` and `build_pseudobatch_inputs` raise
`NotImplementedError` for any process containing a continuous Outflow. Only
Inflow and discrete Outflow (sampling) volume changes are supported. This is
orthogonal to `Outflow.retention` — a discrete Outflow can never carry
retention in the first place (`_check_outflow_retention` rejects it at
construction time), so the restriction is about `is_continuous`, not about
whether retention is zero.

## How `fc` is built

Each addition contributes

```
Δfc = S(t) · ΔV_feed · C_feed / V_init
```

integrated piecewise for continuous feeds and applied as an exact jump for
boluses, using the sample-first value of `S(t)` at that timestamp.

## Representation: exact pieces, no epsilon knots

`ADF`, `fc`, reactor volume, and accumulated feeds are all built on one
breakpoint grid containing the process start and end, every measurement time,
and every timestamp or breakpoint of every volume change.

They are stored as `TimeSeries` with **`continuity_side="left"`** and exact local
polynomial coefficients. There are no `t_event ± ε` knots: evaluating at an event
time returns the pre-event value, and the jump takes effect immediately after.

`ADF` and `fc` are **not** step functions — continuous feed makes both vary
smoothly between events. Only boluses cause true jumps, recorded in `jump_times`
(with magnitudes in `metadata["jump_values"]`).

## Storage layout

After building the transform, one process holds:

| Where | What |
|-------|------|
| `component.concentration` | The real measured concentration. **Never overwritten.** |
| `component.pseudobatch_concentration` | The fitted `c*` spline, tagged `metadata["transform"]["name"] == "pseudo_batch"` |
| `process.pseudobatch_transform.adf` | Shared `ADF` — one per process |
| `process.pseudobatch_transform.feed_corrections[species]` | Per-species `fc` |
| `process.pseudobatch_transform.sample_compensation` | `S(t)`, diagnostic |
| `process.pseudobatch_transform.accumulated_feeds[feed]` | Per-stream cumulative feed, diagnostic |
| `process.volume.total_volume` | Reconstructed reactor-volume trace, filled in if it was `None` |

`ADF` and `S(t)` are species-independent, so they are stored once. The builder
recomputes them per species and **asserts they came out identical** — a mismatch
means the physical bookkeeping diverged and it raises.

The schema is checked before use: a `c*` trace with no matching
`feed_corrections` entry (or vice versa) raises, as does a `c*` trace on a
process with no transform bundle at all.

## Spline fitting policy

`fit_timeseries_spline(ts, *, boundaries=None, smoothing_s=0.0)`:

| Points in a segment | Method |
|---------------------|--------|
| ≥ 4 | SciPy smoothing B-spline (`make_splrep`, cubic). `smoothing_s=0` makes it interpolating. |
| 2–3 | Interpolating natural `CubicSpline` — a cubic smoothing B-spline needs 4 samples |
| 1 | Constant piece over a `1e-6`-wide interval |

`boundaries` splits the series into independently fitted segments; the resulting
piece indices are recorded in `segment_start_piece_idx`. Use
`make_segment_boundaries` with event times to break the fit at discontinuities.
The chosen strategy per segment is recorded in `metadata["fit_strategies"]`.

## Public API

### Pseudobatch pipeline

| Function | Description |
|----------|-------------|
| `build_pseudobatch_transform(process, species_names=None, *, cstar_smoothing_s=0.0)` | The one you normally call. Builds the whole bundle. `species_names=None` means every reactor component with a `TimeSeries` concentration. |
| `evaluate_pseudobatch_transform(process, component, times)` | Evaluate stored `c*` back to real concentration. `component` may be a name or the object. |
| `build_backtransform_spline(process, species_name)` | A JIT-compatible `BacktransformSpline` for the same thing. |

`build_pseudobatch_transform` **mutates the process**: it sets each component's
`pseudobatch_concentration` and fills `volume.total_volume` if empty. It *returns* the
bundle — assign it yourself:

```python
process.pseudobatch_transform = bp.splines.build_pseudobatch_transform(process)
```

It raises if `component.concentration` itself carries pseudobatch `c*`
metadata — that means the real concentration was overwritten by a transform at
some point, and there is nothing raw left to transform.

Re-running it on a process that already has a transform is **not** an error: the
`pseudobatch_concentration` fields and the returned bundle are simply recomputed from
the (untouched) real concentrations. Note that `volume.total_volume` is only
filled when it is `None`, so it keeps whatever was there.

### Spline infrastructure

| Function | Description |
|----------|-------------|
| `fit_timeseries_spline(ts, *, boundaries=None, smoothing_s=0.0)` | Fit segmented spline state onto a `TimeSeries`. |
| `make_cubic_ppoly(t, y, bc_type="natural")` | A cubic `PPoly` from raw arrays; sorts and deduplicates knots. |
| `make_constant_spline(value, t_start, t_end)` | A constant spline-backed `TimeSeries`. |

### Segmentation

| Function | Description |
|----------|-------------|
| `detect_discrete_state_events(process)` | Event times from all non-continuous volume changes, as `DiscreteEvents`. |
| `make_segment_boundaries(t_min, t_max, event_times)` | `[t_min, …interior events…, t_max]`. |
| `split_timeseries(ts, boundaries)` | Split into segments; boundary points belong to both neighbours. |

### Lower-level pipeline

Used by tests and by the transform builder itself; you rarely need them
directly.

| Function | Description |
|----------|-------------|
| `build_pseudobatch_inputs(process, species_name)` | Per-species `ADF`, `fc`, volume, accumulated-feed series plus `c_star` at measurement times. |
| `build_splines(inputs, ...)` | Runtime payload dict from those inputs. |
| `to_timeseries(inputs, splines, species_name, ...)` | The `c*` carrier from that payload. |
| `evaluate_real_concentration(t_eval, splines)` | Backtransform on a `build_splines` payload. |

### `BacktransformSpline`

An `eqx.Module` evaluating `c(t) = (c*(t) + fc(t)) / ADF(t)` inside JIT.

| Member | Description |
|--------|-------------|
| `__call__(t)` | Backtransformed concentration. |
| `derivative()` | A callable returning `dc/dt`. |
| `c_star_spline` | `PPoly` view of the stored `c*`. |
| `adf_ts`, `feed_corr_ts` | The canonical `ADF` / `fc` series. |
| `dadf_ts`, `dfc_ts` | Their derivative series. |
| `is_constant`, `constant_value` | Bypass for near-constant species. |

**The derivative is not just `dc*/dt / ADF`.** By the quotient rule,

```
dc/dt = ( dc*/dt + dfc/dt − c · dADF/dt ) / ADF
```

The `−c · dADF/dt` term is non-zero whenever continuous feed makes `ADF` vary
between events. Dropping it puts a systematic bias into every rate inferred from
the spline.

For species whose measured concentration is effectively constant (and which have
no discrete feed), `is_constant` short-circuits the whole thing and returns the
stored value — a cubic through flat data oscillates, and the backtransform would
amplify that.

## Failure modes

The pipeline raises rather than producing plausible-looking wrong numbers:

| Condition | Why |
|-----------|-----|
| A continuous Outflow anywhere in `process.volume.volume_changes` | Exact closed-form `ADF` is only representable when volume grows from Inflows alone — see "Scope" above |
| Reactor volume ≤ `1e-10` at any breakpoint, before or after any event, or anywhere inside a volume piece | Physically impossible; `ADF` would blow up |
| `|ADF| ≤ 1e-12` at a division | Same |
| A feed concentration given as a `TimeSeries` | Time-varying feed composition is not implemented |
| `c*` present without a `feed_corrections` entry, or vice versa | Half a transform cannot be inverted |
| `component.concentration` already carries `c*` metadata | The real concentration was overwritten; nothing raw left to transform |
| Species-independent series differ between species | The volume bookkeeping is inconsistent |

## Examples

### Build and use the transform

```python
import hybrax.format as bp
import jax.numpy as jnp

process.pseudobatch_transform = bp.splines.build_pseudobatch_transform(process)

t = jnp.linspace(process.time_axis.start, process.time_axis.end, 500)
c = bp.splines.evaluate_pseudobatch_transform(process, "glucose", t)
```

### Inspect `ADF` and `S(t)`

```python
import matplotlib.pyplot as plt

transform = process.pseudobatch_transform
fig, ax = plt.subplots()
ax.plot(t, transform.adf.evaluate_many(t), label="ADF")
if transform.sample_compensation is not None:
    ax.plot(t, transform.sample_compensation.evaluate_many(t), label="S(t)")
ax.set_xlabel(f"time [{process.time_axis.unit}]")
ax.legend()
```

Flat `ADF` across a sampling event and a step at a bolus is the signature of a
correct transform. If `ADF` steps at a *pure sampling* event, the volume
accounting is wrong.

### Inside a JIT-compiled function

```python
import equinox as eqx
import jax

bt = bp.splines.build_backtransform_spline(process, "glucose")

@eqx.filter_jit
def concentrations(spline, times):
    return jax.vmap(spline)(times)

@eqx.filter_jit
def rates(spline, times):
    return jax.vmap(spline.derivative())(times)
```

### Segmenting a fit at events

```python
events = bp.splines.detect_discrete_state_events(process)
bounds = bp.splines.make_segment_boundaries(
    process.time_axis.start, process.time_axis.end, events.times
)
fitted = bp.splines.fit_timeseries_spline(raw_series, boundaries=bounds)
```

## Accuracy

At every measurement point the backtransform reproduces the measurement to
`≤ 1e-5` absolute — that is the pseudobatch identity closing on itself, and a
larger error means a bug.

*Between* measurements the accuracy is bounded by sampling density, not by the
pipeline. On continuously fed datasets the 99th-percentile relative error
against dense ground truth stays under ~5 %; the worst cases are sharp peaks
sampled fewer than ten times, where no cubic can follow the transition. Those
are a data limitation.

## See also

- [TimeSeries](06_time_series.md) — the container being fitted
- [Mechanistic](08_mechanistic.md) — consumes these splines as state trajectories
- [Design Rationale §5](01_design_rationale.md#5-pseudobatch-normalization)
- Hesselberg-Thomsen, V., Groves, T., McCubbin, T., Martínez-Monge, I.,
  de Mas, I. M., & Nielsen, L. K. (2024). Pseudo batch transformation: a novel
  method to correct for mass removal through sample withdrawal of fed-batch
  fermentations. *bioRxiv*.
- [`pseudobatch`](https://github.com/viktorht/pseudobatch) — the upstream
  reference library
