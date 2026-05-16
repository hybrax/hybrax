# Discrete-Jump Pseudobatch Semantics

This note documents the current post-`Interpolator` pseudobatch design.

## Current schema

Raw real reactor-component concentrations remain in
`ReactorMediumComponent.concentration`. Optional pseudobatch transformed
concentrations are stored as component-level `TimeSeries` values in
`ReactorMediumComponent.c_star_concentration`.

Lightweight c* provenance metadata lives under:

```python
ts.metadata["transform"] = {
    "name": "pseudo_batch",
    "component": "...",
    "is_constant": False,
    "constant_value": None,
}
```

The shared process-level transform bundle lives in
`BioProcess.pseudobatch_transform`:

```python
PseudobatchTransform(
    adf=...,                    # TimeSeries
    feed_corrections={...},      # component name -> TimeSeries
    sample_compensation=...,     # optional TimeSeries
    accumulated_feeds={...},     # feed/change name -> TimeSeries
)
```

The full reactor-volume trace, when stored, lives at
`Volume.total_volume`. It may be raw online data or derived from volume
changes.

## Physical semantics

The implementation preserves these invariants:

- continuous feed changes ADF smoothly
- bolus feed creates instantaneous ADF and feed-correction jumps
- pure sampling does not create an artificial concentration jump
- pseudobatch `c*` stays smooth across events when the ADF/sample-compensation
  construction is correct

## ADF representation

ADF is a `TimeSeries` with:

- `continuity_side="left"`
- exact local polynomial pieces stored in `breaks` / `coeffs`
- left-continuous jump times stored in `jump_times` for inspection

This is intentionally not a pure global step function. A pure step ADF would
be physically wrong for continuous-feed growth.

## Feed-correction representation

Feed correction is also a canonical `TimeSeries`:

- smooth continuous-feed pieces in `breaks` / `coeffs`
- exact bolus jumps via adjacent pieces and `continuity_side="left"`
- optional `metadata["jump_values"]` as derived inspection data

## Runtime consumers

The process/component schema is the primary path used by:

- `bp_format.splines.evaluate_pseudobatch_transform`
- `bp_format.splines.build_backtransform_spline`
- `bp_format.mechanistic.build_state_splines`

Older flat metadata keys such as `adf_times`, `adf_values`,
`feed_corr_times`, and `feed_corr_values` are no longer runtime carriers.

## Validation targets

Relevant tests cover:

- bolus jumps appear immediately after the event
- sampling remains continuous
- same-time sample+bolus handling matches the segmented/reference path
- mechanistic reintegration still reproduces measurements to the current error
  budget
