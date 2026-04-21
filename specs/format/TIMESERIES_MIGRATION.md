# TimeSeries Migration Notes

## Canonical API (hard break)

- Constructor: `TimeSeries(times=..., values=...)`
- Field access: `series.times`, `series.values`
- Optional spline state: `breaks`, `coeffs`, `segment_start_piece_idx`

Legacy compatibility has been removed:
- `timepoints=...` constructor argument is no longer supported.
- `series.timepoints` property alias is no longer available.

## Serialization

- Canonical-only schema for `TimeSeries` payloads.
- Discrete payloads must include both `times` and `values`.
- Spline-only payloads are supported via spline fields (`breaks`, `coeffs`,
  `segment_start_piece_idx`) with `values` optional.

## Pseudobatch spline-only fallback

- In pseudobatch workflows, when a `TimeSeries` is spline-only (no discrete
  samples), bp_format uses spline `breaks` as the fallback measurement grid.
- This keeps workflows numerically coherent but may differ from true
  experimental sampling times.
- If sampling-time semantics matter, provide explicit discrete samples.
