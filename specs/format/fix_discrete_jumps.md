# Discrete-Jump Pseudobatch Semantics

This note documents the current post-`Interpolator` pseudobatch design.

## Current schema

Transformed reactor-component concentrations are stored as `TimeSeries`.
Pseudobatch metadata lives under:

```python
ts.metadata["transform"] = {
    "name": "pseudo_batch",
    "species": "...",
    "feed_corr_interp": "linear_plus_step" | "cubic",
    "series": {
        "adf_ts": ...,
        "feed_corr_ts": ...,
    },
}
```

Both nested entries are canonical serialized `TimeSeries` payloads, not raw
array blobs.

## Physical semantics

The implementation preserves these invariants:

- continuous feed changes ADF smoothly
- bolus feed creates instantaneous ADF and feed-correction jumps
- pure sampling does not create an artificial concentration jump
- pseudobatch `c*` stays smooth across events when the ADF/sample-compensation
  construction is correct

## ADF representation

ADF is now a `TimeSeries` with:

- `continuity_side="left"`
- a piecewise-linear baseline stored in `times` / `values`
- instantaneous bolus jumps stored via `jump_times` and
  `metadata["jump_values"]`

This is intentionally not a pure global step function. A pure step ADF would
be physically wrong for continuous-feed growth.

## Feed-correction representation

For discrete-fed species, feed correction also uses the `linear_plus_step`
representation:

- smooth baseline in `times` / `values`
- jump locations in `jump_times`
- jump increments in `metadata["jump_values"]`

For purely continuous-feed cases, feed correction may stay cubic.

## Runtime consumers

The nested `TimeSeries` schema is the primary path used by:

- `bp_format.splines.build_backtransform_spline`
- `bp_format.mechanistic._build_pseudobatch_transforms`

Older flat metadata keys such as `adf_times`, `adf_values`,
`feed_corr_times`, and `feed_corr_values` remain only as compatibility
fallbacks for legacy payloads.

## Validation targets

Relevant tests cover:

- bolus jumps appear immediately after the event
- sampling remains continuous
- same-time sample+bolus handling matches the segmented/reference path
- mechanistic reintegration still reproduces measurements to the current error
  budget
