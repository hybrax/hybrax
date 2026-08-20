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
from bp_format.splines import fit_timeseries_spline

fitted = fit_timeseries_spline(samples, smoothing_s=0.0)
print("has spline now:", fitted.breaks is not None,
      "| pieces:", None if fitted.coeffs is None else fitted.coeffs.shape[0])
print("spline at t=3:", float(fitted.evaluate(3.0)))
```

`smoothing_s=0.0` interpolates exactly; larger values smooth. For **controlled** signals
 (a noisy pump trace, an online pH) smoothing is usually what you want, because you are
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

## Gotchas

- **`PPoly` is not root-exported.** `from bp_format.time_series import PPoly`.
- **`TimeSeries` arithmetic exists** (`ts_a - ts_b`) with exact and approximate paths
  depending on whether the operands share breaks. Useful, but read the API reference
  before relying on the approximate path.

## See also

- [The pseudobatch transform](pseudobatch_transform.md): fed-batch dilution correction,
  built on the spline machinery above.
- [Volume, feeds and events](volume_feeds_events.md): where the dilution comes from.
- [API reference](../autoapi/bp_format/splines/index).
