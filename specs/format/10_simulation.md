# Simulation

Source: `bp_format/simulation.py`

The Simulation API is a small helper layer for deterministic example
simulations used as mechanistic ground truth.

It is not part of `BioProcessCollection` serialization. Simulations write
realistic CSV outputs and return dense arrays in memory; parsers then build
normal bp-format objects from those CSVs.

Example simulations should keep rate laws physically well behaved. In
particular, substrate uptake should taper at depletion instead of relying on
output clipping, and substrate-limited uptake should be mirrored in dependent
growth/product rates; otherwise event replay can mix bolus feeds into different
raw pre-event states during reintegration or the simulated biology becomes
physically inconsistent.

## Runtime Contract

Subclasses implement:

```python
evaluate_rates(self, t, state, controls=None) -> tuple[q, r]
```

The return values match `bp_format.mechanistic.integrate_process`:

- `q`: specific rates for reactor-component states.
- `r`: additive rates for non-volume states, reactor states first and
  uncontrolled process-variable states after them.
- `controls` is optional. Standalone simulations may evaluate controls from
  `t`; reintegration can pass the controls vector from `ControlSplines`.

`Simulation.as_rates_func()` returns a wrapper with the
`rates_func(t, state, controls)` signature expected by mechanistic integration.

## Event Semantics

The base class owns generic realized sampling and bolus mechanics:

- at most one sample and one bolus per process timestamp,
- sample before bolus at shared timestamps,
- sampling changes volume only,
- bolus changes volume and reactor concentrations by mixing feed mass,
- process-variable states are not changed by sample or bolus events.

## CSV Outputs

`Simulation.write_csvs(...)` writes:

- `simulation_dense_output.csv`
- `events.csv`

`simulation_dense_output.csv` uses `row_type`:

- `online`: regular dense sensor-grid row, typically every 30 s or 1 min,
- `offline`: sample measurement row,
- `pre-event`: state before all events at that time,
- `post-event`: state after all events at that time.

Rows are ordered by time, then `online`, `offline`, `pre-event`, `post-event`.
At a sample event, `offline` and `pre-event` carry the same state values.

`events.csv` has one row per realized event operation with explicit
`event_order`, event type, signed volume delta, feed id, and feed-component
concentrations for bolus rows.
