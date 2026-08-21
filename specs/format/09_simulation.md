# Simulation

Source: `src/hybrax/format/simulation.py`

## Purpose

A small helper layer for building **synthetic datasets with known ground truth**.
It owns the fiddly part of simulating a real run: applying sampling and bolus
events in the right order, and emitting rows that look like what a real
bioreactor logs.

It is not part of the `BioProcess` data model and nothing here is serialized.
The intended flow is:

```
your simulation script  ->  Simulation helpers  ->  CSV rows  ->  parser  ->  BioProcess
```

Writing the CSV and parsing it back is deliberate: the parser then gets
exercised on data whose true answer you know.

## What this module does and does not do

**Does:** group events, enforce event ordering, apply mass-balance jumps to a
state vector, and lay out dense-output and event rows.

**Does not:** integrate anything, define rate laws, or write files. You bring the
solver and the biology; `Simulation` handles the bookkeeping around the events.
`SimulationResult` gives you `dense_rows` / `event_rows` as lists of dicts —
write them with `csv.DictWriter` or pandas.

## Constants

Row types, in the order rows appear at a shared timestamp:

```python
ROW_TYPE_ONLINE            = "online"             # dense sensor-grid row
ROW_TYPE_OFFLINE           = "offline"            # the sample measurement
ROW_TYPE_PRE_EVENT         = "pre-event"          # state before events at this time
ROW_TYPE_POST_EVENT        = "post-event"         # state after events at this time
ROW_TYPE_FERMENTATION_END  = "fermentation_end"   # end-of-run marker
```

Event types:

```python
EVENT_TYPE_SAMPLE           = "sample"
EVENT_TYPE_BOLUS            = "bolus"
EVENT_TYPE_FERMENTATION_END = "fermentation_end"
```

## Data classes

### `SimulationEvent`

```python
@dataclass(frozen=True)
class SimulationEvent:
    process: str
    time: float
    event_type: str                                   # "sample" | "bolus" | "fermentation_end"
    delta_volume: float                               # signed
    feed_id: str | None = None
    feed_concentrations: Mapping[str, float] = {}     # per reactor species, for boluses
```

Sign rules are enforced on construction of any group: samples must have
`delta_volume < 0`, boluses `> 0`, and `fermentation_end` exactly `0`.

### `SimulationResult`

```python
@dataclass(frozen=True)
class SimulationResult:
    process: str
    times: np.ndarray                     # solver time grid
    states: np.ndarray                    # (len(times), len(state_names))
    state_names: tuple[str, ...]
    reactor_state_names: tuple[str, ...]  # the subset a bolus mixes into
    dense_rows: list[dict]
    event_rows: list[dict]
    row_columns: tuple[str, ...]
    event_columns: tuple[str, ...]
```

## Event semantics

The rules the module enforces, all of them physical:

- **At most one sample and one bolus per process per timestamp.** More than one
  would make the outcome depend on an ordering nobody specified.
- **Sample first, then bolus.** The offline measurement describes the broth as
  drawn; the feed then dilutes what is left.
- **Sampling changes volume only.** Removing well-mixed broth reduces amounts
  and volume proportionally, so concentrations do not move.
- **A bolus changes volume and mixes in feed mass:**

  ```
  c_new = (c_old · V_old + c_feed · ΔV) / (V_old + ΔV)
  ```

- **Process-variable states are untouched by events.** Only species in
  `reactor_state_names` are mixed.
- **Volume must stay positive.** An event that would take it to zero or below
  raises.

## Methods

| Method | Description |
|--------|-------------|
| `group_events(events)` | Group by `(process, time)`, validate signs and multiplicity, sort sample → bolus → end. |
| `apply_events(state, events, *, state_names, reactor_state_names, volume_state_name="volume")` | Apply one timestamp's events to a state vector; returns the post-event state. |
| `build_dense_rows(...)` | Ordered dense-output rows. |
| `build_event_rows(events, reactor_state_names)` | One row per event operation. |
| `event_columns(reactor_state_names)` | Column tuple for the event rows. |
| `build_result(...)` | Assemble a `SimulationResult` from a solved trajectory plus events. |

`build_result` needs a state value at **every** row time — every online time and
every event time must be present in `state_times`, or it raises. Extra
diagnostic columns can be passed via `extra_columns`, one value per entry in
`state_times`.

## Row layout

`dense_rows` columns:

```
process_id, time, row_type, <state_names...>, cum_bolus_feed, <extra_columns...>
```

`event_rows` columns:

```
process_id, time, event_order, event_type, delta_volume, feed_id, feed_<species>...
```

At one timestamp, rows appear in this order:

1. `online` — if that time is on the sensor grid
2. `offline` — if a sample happens there
3. `pre-event` — the state before events (identical values to the `offline` row)
4. `post-event` — the state after all events at that time

A `fermentation_end` event emits `offline` then `fermentation_end` and no
pre/post pair, since it changes nothing.

`cum_bolus_feed` accumulates bolus volume and is written *after* the events at
that timestamp are applied, so a `post-event` row already includes the bolus that
just happened.

The `offline` / `pre-event` split matters for parsing: the offline row is the
measurement a lab would report, the pre/post pair is what a reintegration needs
to replay the jump.

## Writing physically sensible simulations

The module cannot check your rate laws, and these are the mistakes that make a
synthetic dataset unusable:

- **Taper uptake at depletion.** Let the rate go to zero as the substrate runs
  out, rather than clipping negative concentrations after the fact. Clipping is
  not the same as limiting: it silently creates mass.
- **Propagate the limitation.** If substrate uptake is limited, growth and
  product formation must be limited too — otherwise the simulation makes biomass
  from nothing.
- **Emit offline rows only where a sample is drawn,** and give that sample a
  volume. An offline measurement with no volume removal is not something a real
  run can produce.

Getting this wrong shows up later as event replay mixing a bolus into the wrong
pre-event state during reintegration.

## Example

```python
import numpy as np
from hybrax.format import Simulation, SimulationEvent

sim = Simulation()

events = [
    SimulationEvent(process="run_1", time=6.0,  event_type="sample",
                    delta_volume=-0.005),
    SimulationEvent(process="run_1", time=6.0,  event_type="bolus",
                    delta_volume=0.05, feed_id="glucose_feed",
                    feed_concentrations={"glucose": 500.0, "biomass": 0.0}),
]

# ... solve your ODE, stopping and restarting at each event time ...

result = sim.build_result(
    process="run_1",
    state_times=times,                 # must include every event and online time
    states=states,                     # (len(times), len(state_names))
    online_times=online_grid,
    state_names=("biomass", "glucose", "volume"),
    reactor_state_names=("biomass", "glucose"),
    events=events,
)

import csv
with open("dense_output.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=result.row_columns)
    w.writeheader()
    w.writerows(result.dense_rows)
```

## See also

- [Data Model](02_data_model.md) — what a parser builds from these rows
- [Mechanistic](08_mechanistic.md) — `extract_discrete_events` applies the same
  sample-before-bolus rule on the `BioProcess` side
