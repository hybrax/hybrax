# Simulation 1: Intracellular Simulation

This simulation is a deterministic CHO fed-batch fixture. It is inspired by the
Martens-derived examples, especially `12_martens_expanded`, but it is not meant
to reproduce one Martens run quantitatively. The model is intentionally small:
it keeps only the biology and data-model assumptions needed to test
intracellular components, uncontrolled process variables, dense online data,
sampling, bolus feeding, and event replay.

## Lineage

The starting point is the simulated Martens family:

- `05_martens_2025_a` through `10_martens_2025_f` contain the original
  Martens-inspired CHO virtual-lab examples.
- `12_martens_expanded` adds nutrient continuous feed, base-feed volume,
  sampling events, event-aware outputs, and mechanistic reintegration checks.
- `sim_1_intracellular` keeps those structural ideas but changes the product
  biology to exercise hybrax.format intracellular state handling.

Sim 1 uses one compact rate law with fixed setpoints and two default processes.

## State Layout

The mechanistic state order is:

1. reactor states: `biomass`, `product_extracellular`,
   `product_intracellular`, `dead_cells`, `glucose`, `glutamine`, `lactate`,
   `ammonia`
2. uncontrolled process variable: `intracellular_product_ratio`
3. volume state: `volume`

`biomass`, `dead_cells`, and product pools use `mg/L`. `biomass` means measured
viable biomass, like dried cell-pellet mass, so it includes intracellular
product mass. Active biomass is only the rate basis:

```text
X_active = biomass - product_intracellular
```

The biomass ODE therefore uses:

```text
d_biomass/dt = q_biomass * X_active
```

`q_biomass` is the measured-biomass specific rate. It includes active-biomass
growth, death, and intracellular product accumulation:

```text
q_biomass = mu - death_rate + q_product_intracellular
```

So `q_biomass` can be negative if death is larger than growth plus
intracellular product accumulation.

## Biology

### pH and Temperature

The original Martens virtual lab multiplies growth by pH and temperature
response factors. Sim 1 keeps that idea but simplifies it to symmetric Gaussian
penalties around fixed nominal setpoints:

- pH nominal: `7.05`
- temperature nominal: `36.8 degC`

The default controls are constant at those setpoints, so the factor is normally
one. Keeping the dependency in the rate law is useful because downstream
mechanistic integration passes controls through the same interface that real
controlled variables use.

### Glucose and Glutamine

Growth and product formation are multiplied by glucose and glutamine limitation
terms:

- glucose: `glucose / (K_G + glucose)`
- glutamine: `glutamine / (K_Q + glutamine)`

Substrate uptake uses the same limitation terms so uptake tapers near depletion.
If either substrate is depleted, product formation stops and the biomass rate is
dominated by death.

### Product Formation

Sim 1 has intracellular and extracellular product pools:

- early in the run, all newly formed product is secreted extracellularly
- over time, the intracellular retained fraction relaxes toward about `0.5`
- `intracellular_product_ratio` stores that split

`product_intracellular` is a reactor-medium component marked intracellular in
the parsed `BioProcess`. There is no product degradation in sim 1.

### Death and Byproducts

Sim 1 uses a constant specific death rate. Dead-cell, lactate, and ammonia
formation are specific rates scaled by `X_active`. They are simple byproduct
channels, not fitted biology.

## Feeding And Events

Default generation writes two processes:

- `sim_1_run_1`: continuous nutrient feed, base feed, and bolus feed
- `sim_1_run_2`: bolus-only nutrient feed, no base feed, with a similar final
  reactor volume

The feed medium carries glucose and glutamine at the same concentrations used
in ex12:

- glucose: `500 mmol/L`
- glutamine: `50 mmol/L`

Continuous nutrient feed and base feed use standard well-mixed dilution terms.
Base feed is pure dilution; it is included because ex12 adds base volume as a
proxy for pH-control addition.

Sampling events reduce volume only. Bolus events increase volume and mix feed
mass into the reactor. When sampling and bolus occur at the same timestamp, the
sample is recorded first and the bolus is applied second.

## Outputs

Generate the canonical simulation artifacts with:

```bash
pixi run python tests/e2e_tests/sim_1/simulation.py --output-dir tests/e2e_tests/sim_1/sim_results --plot-dir tests/e2e_tests/sim_1/sim_plots
```

`simulation.py` writes:

- `sim_results/simulation_dense_output.csv`: dense online rows,
  offline rows, event-boundary rows, controls, diagnostics, and simulated rate
  columns (`q_*`, `r_*`).
- `sim_results/simulation_events.csv`: event rows keyed by `process_id`.
- `sim_plots/*.png`: simulation plots.

Online sensor rows are generated every five minutes. Event timestamps also get
explicit event-boundary rows in `simulation_dense_output.csv`:

- `online`: regular dense online sensor row
- `offline`: sample measurement row
- `pre-event`: state before events at that timestamp
- `post-event`: state after events at that timestamp
- `fermentation_end`: explicit end-of-fermentation event row

At sampling times, `offline` and `pre-event` have the same state values. At a
shared sampling/bolus timestamp, both are pre-bolus.

## Target Layout

The canonical parsed collection is:

- `sim_results/process_collection.json`: `BioProcessCollection` with
  `sim_1_run_1` and `sim_1_run_2`.

It is generated by parsing `sim_results/simulation_dense_output.csv` and
`sim_results/simulation_events.csv` with `load_utils.py`.

The dense CSV preserves simulator row semantics (`online`, `offline`,
`pre-event`, `post-event`). It is not hybrax.format measurement data and should not
be interpreted as dense physical offline sampling.

## What This Simulation Tests

Sim 1 is mainly a verification fixture for:

- real-space mechanistic RHS reintegration against a known synthetic ground truth
- explicit reactor/process-variable/volume state ordering
- intracellular component subtraction from active biomass as rate basis
- measured-biomass state and `q_biomass` semantics including intracellular mass
- controlled pH/temperature plumbing
- dense online rows plus sparse offline sample rows from one CSV
- sampling and bolus event replay with sample-before-bolus semantics
- physical volume and substrate mass balance around events
