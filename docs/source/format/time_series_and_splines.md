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

> How a measurement becomes something evaluable at any `t`.

## `TimeSeries` holds up to two things

A `TimeSeries` can carry **discrete samples**, a **fitted spline**, or both.

```{code-cell} ipython3
import numpy as np
import hybrax.format as hxf

samples = hxf.TimeSeries(times=np.array([0.0, 2.0, 4.0, 6.0, 8.0]),
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

A `TimeSeries` may be spline-only, that happens in [pseudobatch
workflows](pseudobatch_transform.md) where the original samples no longer mean anything
on their own.

### Evaluating

```{code-cell} ipython3
print("linear interp on samples:", float(samples.lin_interp(3.0)))
```

`lin_interp` works off the samples. `evaluate` works off the spline and requires one to
be fitted.

### Fitting one

```{code-cell} ipython3
from hybrax.format.splines import fit_timeseries_spline

fitted = fit_timeseries_spline(samples, smoothing_s=0.0)
print("has spline now:", fitted.breaks is not None,
      "| pieces:", None if fitted.coeffs is None else fitted.coeffs.shape[0])
print("spline at t=3:", float(fitted.evaluate(3.0)))
```

`smoothing_s=0.0` interpolates exactly; larger values smooth. For **controlled** signals
 (a noisy pump trace, an online pH) smoothing is usually what you want, because you are
going to differentiate the result and noise differentiates badly.

### Monotone fits for cumulative signals

A cumulative signal, such as the mass a scale reports as having left a feed bottle, can
only move in one direction. Its derivative is the rate the ODE sees, so a fit that wiggles
backwards would feed a negative rate into the model. `fit_monotone_timeseries_spline`
fits a smooth cubic whose derivative never changes sign. Instead of a smoothing
parameter you give an error budget in the units of the values, and it places knots where
the data need them:

```{code-cell} ipython3
from hybrax.format.splines import fit_monotone_timeseries_spline

t = np.linspace(0.0, 10.0, 601)
rng = np.random.default_rng(1)
# A pump that doses in small pulses, starting at t=2 and speeding up at t=6.
rate = np.where(t < 2.0, 0.0, np.where(t < 6.0, 0.02, 0.05))
pulses = rng.random(t.size) < 0.4
increments = np.r_[0.0, np.where(pulses[1:], rate[1:] * np.diff(t) / 0.4, 0.0)]
cumulative = hxf.TimeSeries(times=t, values=np.cumsum(increments))

feed = fit_monotone_timeseries_spline(cumulative, rms_tol=3e-3, increasing=True)
info = feed.metadata["monotone_fit"]
print("RMS error:", round(info["rms_error"], 5), "| max error:", round(info["max_error"], 5))
print("interior knots:", info["interior_knots"], "| direction:", info["direction"])
rates = np.asarray(feed.deriv().evaluate_many(np.array([1.0, 4.0, 8.0])))
print("fitted rate at t=1, 4, 8:", rates.round(3), "| true: [0, 0.02, 0.05]")
```

`rms_tol` bounds the root-mean-square error at the samples. Both endpoints are kept exactly,
so the derivative integrates to the measured net change and the cumulative error never
drifts. For the chosen knots, the fit numerically minimizes curvature among curves with
ordered B-spline coefficients that meet the budget, so it usually uses all of the budget.
The error at individual samples can be larger than `rms_tol`: a short pump pause or a
reading that dips below the previous one is smoothed over rather than followed. Every
sample counts equally, so densely logged stretches weigh more towards `rms_tol`. Use
`increasing=False` for decreasing series; the endpoints must agree with
the requested direction. Equal endpoints give a constant fit for either direction if it
meets the budget. Flat stretches get a tiny nonzero rate.

Larger `rms_tol` gives a smoother curve and rate; it is the smoothing knob. Choose it
from the scale of the whole process, for example a small fraction of the total feed or
reactor volume, and at least about 1.5 times the measurement noise standard deviation.
At about the noise standard deviation, the fit is unstable: it can switch between
few knots, the cap and an error. A budget below the noise can need a knot for nearly
every sample and give a jagged rate, as can data that are not monotone enough to fit
within the budget. The fit stops at `max_knots` (by default a tenth of the samples, at
least 16) and raises an error.

The budget, monotonicity, continuity and endpoints are checked on the stored curve: the
fit either meets them or raises an error. Smoothness is not checked. With strongly
clustered samples, such as bursts of fast logging, the smoothing can fail or return a
curve rougher than necessary. Thinning redundant burst readings helps, but keep enough
readings to resolve short additions; the budget then covers the remaining samples.

Between two samples the fit only knows that the curve rises from one reading to the
next, not when within that interval the pump actually delivered. A large step between
two adjacent readings therefore becomes a smooth rise across that interval and its
neighbours.

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

## Gotchas

- **`PPoly` is not root-exported.** `from hybrax.format.time_series import PPoly`.
- **`TimeSeries` arithmetic exists** (`ts_a - ts_b`) with exact and approximate paths
  depending on whether the operands share breaks. Useful, but read the API reference
  before relying on the approximate path.
- **Monotone fits ignore segment boundaries.** `fit_monotone_timeseries_spline` fits one
  continuous curve and rejects series with jumps; fit each continuous stretch separately.
  Pass time since the start of the process, not absolute timestamps: times more than 1000
  durations from zero are rejected.

## See also

- [The pseudobatch transform](pseudobatch_transform.md): fed-batch dilution correction,
  built on the spline machinery above.
- [Volume, feeds and events](volume_feeds_events.md): where the dilution comes from.
- [API reference](../autoapi/hybrax/format/splines/index).
