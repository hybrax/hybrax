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

# The pseudobatch transform

> How fed-batch dilution is removed from a measured concentration before you fit to it.

In a fed-batch run a measured concentration moves for two reasons: the cells did
something, and the volume changed. The pseudobatch transform removes the second.

```
pseudobatch_concentration(t) = c(t) · ADF(t) − feed_correction(t)
```

where **ADF** is the accumulated dilution factor `V(t)/V(0)` and the feed correction
accounts for mass the feed added. The result is what the concentration *would have been*
in a batch run with identical biology.

This is a direct implementation of Hesselberg-Thomsen et al. 2024
<a href="#ref-pseudobatch">[1]</a>, *"Pseudo batch transformation: A novel method to
correct for mass removal through sample withdrawal of fed-batch fermentations,"* whose
accumulated-dilution-factor and feed-correction formulas `build_pseudobatch_transform`
reproduces exactly.

```{code-cell} ipython3
import numpy as np
import hybrax.format as hxf
from hybrax.format.splines import build_pseudobatch_transform

cs = hxf.serialization.load_process_collection("../_data/out/demo_fedbatch/data.json")
process = cs.processes["fedbatch_1"]

bundle = build_pseudobatch_transform(process)
process.pseudobatch_transform = bundle        # <- you must assign it yourself

glucose = process.reactor_medium.components["glucose"]
print("t                        :", np.asarray(glucose.concentration.times)[:6])
print("measured                 :", np.round(np.asarray(glucose.concentration.values)[:6], 2))
print("pseudobatch_concentration:",
      np.round(np.asarray(glucose.pseudobatch_concentration.values)[:6], 2))
```

Read those last two rows across. They agree until the feed starts at t = 6 h, and diverge
afterwards: the measured glucose is propped up by feeding, while `pseudobatch_concentration`
keeps falling because that is what the cells are actually doing to it.

Three uses:

1. **Smoother curves**, so cubic splines fit better.
2. **Comparability**: batch and fed-batch runs become directly comparable.
3. **Segmentation** at bolus discontinuities becomes meaningful.

:::{admonition} The transform mutates, and also returns
:class: warning
`build_pseudobatch_transform(process)` writes `pseudobatch_concentration` onto every component
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

## Going back

`build_backtransform_spline(process, species)` returns a `BacktransformSpline` that maps
`pseudobatch_concentration` back to real concentration (including the derivative, via
the quotient rule) and is JIT-safe, so a model can be trained in pseudobatch space and
evaluated in physical space.

## The step-interpolation rule

ADF and the feed correction are **piecewise constant**, and must be evaluated with step
(nearest-neighbour) interpolation, not linear. Interpolating them linearly smears each
discontinuity across the interval and breaks the backtransform near events. hybrax.format
does this correctly; it matters if you build your own.

## Gotchas

- **Re-running the transform is safe**, but feeding an already-transformed
  `pseudobatch_concentration` back in raises. Same principle as the loader rejecting
  `pseudobatch_concentration` carriers.
- **A `pseudobatch_concentration` trace that jumps at a pure sampling event means your
  volume accounting is wrong.** Sampling is a well-mixed removal, so the
  dilution-corrected trace should be smooth through it. This is one of the best
  diagnostics in the package.
- **Scoped to `Inflow` and discrete `Outflow`.** A continuous `Outflow` (e.g. perfusion)
  raises `NotImplementedError`: exact ADF requires volume to only grow via inflows. See
  [Limits and gotchas](limits_and_gotchas.md).

## See also

- [Time series and splines](time_series_and_splines.md): the general spline-fitting
  machinery this transform builds on.
- [Volume, feeds and events](volume_feeds_events.md): where the dilution comes from.
- [Gallery: pseudobatch splines](../gallery/pseudobatch_splines.md): recovering a curve
  through a jump from just 5 measurements, checked against a known ground truth.
- [Gallery: fed-batch](../gallery/fed_batch.md): the transform on a real process.
- [API reference](../autoapi/hybrax/format/splines/index).

## References

1. <a id="ref-pseudobatch"></a>Hesselberg-Thomsen, V., Groves, T., McCubbin, T.,
   Martínez-Monge, I., de Mas, I. M., & Nielsen, L. K. (2024). Pseudo batch
   transformation: A novel method to correct for mass removal through sample
   withdrawal of fed-batch fermentations. *bioRxiv*.
   [https://doi.org/10.1101/2024.05.27.596043](https://doi.org/10.1101/2024.05.27.596043)
