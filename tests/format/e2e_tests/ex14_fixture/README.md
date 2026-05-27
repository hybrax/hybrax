# Example 14: Intracellular Simulation

This example is a deterministic CHO fed-batch simulation that exercises
`bp_format.mechanistic.RhsOde`. (Forward integration lives in `bp-train`.)
It is inspired by the
Martens-derived examples, especially `12_martens_expanded`, but it is not meant
to reproduce one Martens run quantitatively. The model is intentionally small:
it keeps the biological and data-model assumptions needed to test intracellular
components, uncontrolled process variables, dense online data, sampling, bolus
feeding, and event replay.

## Lineage

The starting point is the simulated Martens family:

- `05_martens_2025_a` through `10_martens_2025_f` contain the original
  Martens-inspired CHO virtual-lab examples.
- `12_martens_expanded` modifies that setup to add nutrient continuous feed,
  base-feed volume, sampling events, event-aware outputs, and mechanistic
  reintegration checks.
- `14_simulation_intracellular` keeps those structural ideas but changes the
  product biology to exercise bp-format intracellular state handling.

The major simplification is that ex14 uses one compact rate law with fixed
setpoints and two default processes, instead of the larger celltype/fidelity
parameterization from the Martens virtual lab.

## State Layout

The mechanistic state order is:

1. reactor states: `biomass`, `product_extracellular`,
   `product_intracellular`, `dead_cells`, `glucose`, `glutamine`, `lactate`,
   `ammonia`
2. uncontrolled process variable: `intracellular_product_ratio`
3. volume state: `volume`

`biomass`, `dead_cells`, and product pools use `mg/L`. Martens-style examples
mostly use cell counts, but the intracellular component has to be subtracted
from measured biomass by `bp_format.mechanistic`, so the biomass and
intracellular product units must be compatible. The conversion used here is
`200 mg / 1e9 cells`.

Active biomass is:

```text
X_active = biomass - product_intracellular
```

This is the quantity used by the mechanistic RHS for specific rates. The
reported `biomass` state therefore represents measured viable biomass including
intracellular product mass.

## Biology

### pH and Temperature

The original Martens virtual lab multiplies growth by pH and temperature
response factors. Those are Gaussian-like factors centered around an optimal pH
and temperature. Ex14 keeps that idea but simplifies it to symmetric Gaussian
penalties around fixed nominal setpoints:

- pH nominal: `7.05`
- temperature nominal: `36.8 degC`

The default controls are constant at those setpoints, so the factor is normally
one. Keeping the dependency in the rate law is still useful because the
mechanistic interface (and downstream forward integration in `bp-train`)
passes controls through the same interface that real controlled variables use.

### Glucose and Glutamine

The Martens growth model uses Monod-like glucose and glutamine terms. Ex12 also
introduced Monod-gated maintenance so substrate uptake goes to zero at
depletion instead of driving negative concentrations.

Ex14 uses the same physical principle:

- glucose uptake is gated by `glucose / (K_G + glucose)`
- glutamine uptake is gated by `glutamine / (K_Q + glutamine)`
- growth and product formation are also multiplied by both limitation terms

This means depleting either substrate stops product formation and leaves only
the death term in the biomass rate. That coupling is important: reducing uptake
without reducing growth would create biomass/product from unavailable substrate.

### Product Formation

Ex12 models glycosylated and non-glycosylated product, with a time-varying
glycosylation split. Ex14 replaces that idea with an intracellular/extracellular
split:

- early in the run, all newly formed product is secreted extracellularly
- over time, the intracellular retained fraction relaxes toward about `0.5`
- the process variable `intracellular_product_ratio` stores that split

`product_intracellular` is a reactor-medium component marked intracellular in
the parsed `BioProcess`. This is the feature ex14 is mainly designed to test.

There is no product degradation in ex14. Product can be diluted by feed volume,
but it should not disappear through a degradation term.

### Death and Byproducts

The Martens virtual lab contains more detailed death and lysis dynamics. Ex14
uses a constant specific death rate to keep the test case small. Dead-cell,
lactate, and ammonia formation are specific rates scaled by active biomass.
They are simple byproduct channels, not fitted biology.

## Feeding And Events

Default generation writes two processes:

- `ex14_run_1`: continuous nutrient feed, base feed, and bolus feed
- `ex14_run_2`: bolus-only nutrient feed, no base feed, with a similar final
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

Generate the canonical simulation artifacts first:

```bash
pixi run python 00_simulation/run_simulation.py
```

`00_simulation/run_simulation.py` writes exactly two source artifacts:

- `00_simulation/simulation_dense_output.csv`: one wide dense CSV with
  `process_id`, time, row type, simulated states, volume, controls,
  diagnostics, and simulated rate columns (`q_*`, `r_*`).
- `00_simulation/events.csv`: event rows keyed by `process_id`.

Online sensor rows are generated every five minutes. Event timestamps also get
explicit event-boundary rows in `simulation_dense_output.csv`:

- `online`: regular dense online sensor row
- `offline`: sample measurement row
- `pre-event`: state before events at that timestamp
- `post-event`: state after events at that timestamp

At sampling times, `offline` and `pre-event` have the same state values. At a
shared sampling/bolus timestamp, both are pre-bolus.

There is no separate canonical parsing step. The target-layout loaders parse
the existing `00_simulation` CSVs through the shared `load_utils.py` module.
Reintegration coverage will be rebuilt in later e2e pipeline milestones, not by
a fixture-local validation script.

## Target Layout

The migrated target-layout loaders are:

```bash
pixi run python 01_single_process/load_single_process.py
pixi run python 02_all_processes/load_all_processes.py
```

They consume the existing `00_simulation/simulation_dense_output.csv` and
`00_simulation/events.csv`; they do not rerun the simulation. They write:

- `01_single_process/output/data.json`: `BioProcessCollection` with
  `ex14_run_1` only;
- `02_all_processes/output/data.json`: `BioProcessCollection` with
  `ex14_run_1` and `ex14_run_2`.

The dense CSV preserves simulator row semantics (`online`, `offline`,
`pre-event`, `post-event`). It is not bp-format measurement data and should not
be interpreted as dense physical offline sampling.

## What This Example Tests

Ex14 is mainly a verification fixture for:

- `Simulation.evaluate_rates(t, state, controls)` matching
  `mechanistic.RhsOde` rate semantics
- explicit reactor/process-variable/volume state ordering
- intracellular component subtraction from active biomass
- controlled pH/temperature plumbing
- dense online rows plus sparse offline sample rows from one CSV
- sampling and bolus event replay with sample-before-bolus semantics
- physical volume and substrate mass balance around events
