# BP-bench

Bioprocess hybrid model setup and training with Jax and Diffrax

## Event overlap semantics

In V1, event-like controls are represented as finite-width piecewise-linear
segments (sampling ramps and bolus triangles), not instantaneous jumps.
If a sampling event and a bolus event occur within the same `min_dt` window,
their support intervals overlap in time and both contributions are active over
that overlap. All event boundaries are merged into one per-process `step_ts`
sequence and forwarded to the solver as `jump_ts` hints.
Training/evaluation loss is still sampled only at measurement timestamps, so a
measurement strictly before a bolus event (`t_sample < t_bolus`) is unaffected
by that bolus.
