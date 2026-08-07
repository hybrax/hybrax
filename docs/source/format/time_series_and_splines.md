---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Time series and splines

> **In one sentence.** How a measurement becomes something evaluable at any `t`, and how
> fed-batch dilution is removed before you fit to it.
>
> **You need this if** you need continuous interpolation or your process has feeds.
> **You can skip it if** you only ever touch measured values at measured times.

## `TimeSeries` holds up to two things

A `TimeSeries` can carry **discrete samples**, a **fitted spline**, or both.

```{code-cell} ipython3
import numpy as np
import bp_format as bp

samples = bp.TimeSeries(times=np.array([0.0, 2.0, 4.0, 6.0, 8.0]),
                        values=np.array([0.1, 0.3, 0.9, 2.4, 4.1]))
print("has samples:", samples.times is not None)
print("has spline :", samples.breaks is not None)
```

The two halves have different jobs, and keeping both is the point:

- **Samples are the experimental ground truth.** The loss is computed against them, and
  validation checks them.
- **The spline is for evaluation during integration.** An ODE solver needs a value at
  whatever `t` it lands on, and re-interpolating at every step would be both slow and
  non-differentiable in the way JAX needs.

A `TimeSeries` may be spline-only — that happens in pseudobatch workflows where the
original samples no longer mean anything on their own.

### Evaluating

```{code-cell} ipython3
print("linear interp on samples:", float(samples.lin_interp(3.0)))
```

`lin_interp` works off the samples. `evaluate` works off the spline and requires one to
be fitted.

### Fitting one

```{code-cell} ipython3
from bp_format.splines import fit_timeseries_spline

fitted = fit_timeseries_spline(samples, smoothing_s=0.0)
print("has spline now:", fitted.breaks is not None,
      "| pieces:", None if fitted.coeffs is None else fitted.coeffs.shape[0])
print("spline at t=3:", float(fitted.evaluate(3.0)))
```

`smoothing_s=0.0` interpolates exactly; larger values smooth. For **controlled** signals
— a noisy pump trace, an online pH — smoothing is usually what you want, because you are
going to differentiate the result and noise differentiates badly.

:::{admonition} Why power-basis polynomials
:class: note
Splines are stored as piecewise cubics in power basis: `c₀h³ + c₁h² + c₂h + c₃` with
`h = t − t_break`. Horner's method evaluates that in a few multiply-adds, which
vectorises cleanly under JAX. A B-spline basis would need recursive knot lookups, which
does not.
:::

## Segmentation at events

A real fed-batch trace is not smooth: a bolus makes it jump. Fitting one polynomial
across a discontinuity produces a spline that is wrong on both sides.

So splines are fitted **per segment**, with boundaries at discrete events. The helpers
`detect_discrete_state_events`, `make_segment_boundaries` and `split_timeseries` build
those boundaries from the process's own volume changes, and `fit_timeseries_spline`
accepts them via `boundaries=`.

## The pseudobatch transform

In a fed-batch run a measured concentration moves for two reasons — the cells did
something, and the volume changed. The pseudobatch transform removes the second.

```
c*(t) = c(t) · ADF(t) − feed_correction(t)
```

where **ADF** is the accumulated dilution factor `V(t)/V(0)` and the feed correction
accounts for mass the feed added. The result is what the concentration *would have been*
in a batch run with identical biology.

```{code-cell} ipython3
from bp_format.splines import build_pseudobatch_transform

cs = bp.serialization.load_case_study("../_data/out/demo_fedbatch/data.json")
process = cs.processes["fedbatch_1"]

bundle = build_pseudobatch_transform(process)
process.pseudobatch_transform = bundle        # <- you must assign it yourself

glucose = process.reactor_medium.components["glucose"]
print("t       :", np.asarray(glucose.concentration.times)[:6])
print("measured:", np.round(np.asarray(glucose.concentration.values)[:6], 2))
print("c*      :", np.round(np.asarray(glucose.c_star_concentration.values)[:6], 2))
```

Read those last two rows across. They agree until the feed starts at t = 6 h, and diverge
afterwards: the measured glucose is propped up by feeding, while `c*` keeps falling
because that is what the cells are actually doing to it.

Three uses:

1. **Smoother curves**, so cubic splines fit better.
2. **Comparability** — batch and fed-batch runs become directly comparable.
3. **Segmentation** at bolus discontinuities becomes meaningful.

:::{admonition} The transform mutates, and also returns
:class: warning
`build_pseudobatch_transform(process)` writes `c_star_concentration` onto every component
in place, **but does not set `process.pseudobatch_transform`**. You have to assign the
returned bundle yourself, as above. It also fills `volume.total_volume` only when that is
currently `None`.
:::

:::{admonition} Fit control splines *before* transforming
:class: important
If the continuous-feed `TimeSeries` have no spline yet, the transform falls back to one
polynomial piece per raw sample. On a densely logged online trace that means tens of
thousands of pieces and seconds per species. Fitting the control splines first is roughly
a hundredfold difference in wall time.
:::

### Going back

`build_backtransform_spline(process, species)` returns a `BacktransformSpline` that maps
`c*` back to real concentration — including the derivative, via the quotient rule — and
is JIT-safe, so a model can be trained in pseudobatch space and evaluated in physical
space.

### The step-interpolation rule

ADF and the feed correction are **piecewise constant**, and must be evaluated with step
(nearest-neighbour) interpolation, not linear. Interpolating them linearly smears each
discontinuity across the interval and breaks the backtransform near events. bp-format
does this correctly; it matters if you build your own.

## Gotchas

- **`PPoly` is not root-exported.** `from bp_format.time_series import PPoly`.
- **Re-running the transform is safe**, but feeding an already-transformed `c*` back in
  raises. Same principle as the loader rejecting `c*` carriers.
- **A `c*` trace that jumps at a pure sampling event means your volume accounting is
  wrong.** Sampling is a well-mixed removal, so the dilution-corrected trace should be
  smooth through it. This is one of the best diagnostics in the package.
- **`TimeSeries` arithmetic exists** (`ts_a - ts_b`) with exact and approximate paths
  depending on whether the operands share breaks. Useful, but read the API reference
  before relying on the approximate path.

## See also

- [Volume, feeds and events](volume_feeds_events.md) — where the dilution comes from.
- [Gallery: fed-batch](../gallery/fed_batch.md) — the transform on a real process.
- [API reference](../autoapi/bp_format/splines/index).
- Hesselberg-Thomsen et al. (2024) for the pseudobatch method itself.
