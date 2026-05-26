# Pseudobatch transform prototypes

This directory contains prototype/reference scripts for implementing pseudobatch
normalization with bp-format `TimeSeries` objects.

## Files

- `pseudobatch_transform_timeseries_splines.py`
  - compares two ways to populate `process.pseudobatch_transform`:
    - exact discrete-value reconstruction from clean cumulative feeds and trusted events
    - spline-backed reconstruction from noisy cumulative feed-scale traces plus trusted sampling/bolus events
  - uses ex14 generated data as a concrete fixture
  - includes plots for raw `TimeSeries` values vs spline evaluations

## Run

From the repository root:

```bash
pixi run python documentation/pseudobatch-transform/pseudobatch_transform_timeseries_splines.py
```

The script is intentionally educational/prototype code. It should stay readable
and formula-oriented, not become a production abstraction layer.
