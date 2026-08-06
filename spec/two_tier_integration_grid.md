# Two-tier integration grid

Status: implemented (`38a5261`), benchmarked. This document records what the change is,
what it costs and buys, and how to predict which of the two it does on a given dataset.

---

## Background: why output times were segment boundaries

`physical_solve.solve_physical_states` runs the ODE as a `lax.scan` of per-segment
`diffrax.diffeqsolve` calls, applying bolus/sample state jumps between segments
(`diffrax_callbacks.diffeqsolve_with_callbacks`). Each segment saved with
`SaveAt(t1=True)`, and the trajectory was read back out of the **event log** by an argmin
gather over `event_states_before`.

The event log being the only output channel is the entire reason measurement, dense and
prediction times had to be preset nodes: a value could only be observed where the solver
stopped. Physically only bolus and sample events need a split, because only they
discontinuously jump the state.

The consequence was that asking for a finer trajectory *subdivided the integration*
rather than just recording more of it. Measured on 2023_bayer/run_01: 10 segments / 38
ODE steps on the measurement grid, and **208 segments / 426 steps** once a 200-point
export grid was requested. It also made `fail_time` — and therefore which samples were
masked out of the loss — depend on how finely the horizon happened to be chopped.

---

## The change

### Tier A — segment boundaries

`preset_times` is now `bolus ∪ sample` only. `max_events = n_presets + 1` (one stop per
node plus the final leg to `t1`). Vector-field kinks (control-spline knots,
`BioProcess.discrete_events`) are unchanged: they were never boundaries, only
`PIDController(jump_ts=...)` hints.

### Tier B — output times

`diffeqsolve_with_callbacks` takes `output_times` and each segment saves its share with
`SaveAt(t1=True, ts=...)`. Probed on diffrax 0.7.2: `SaveAt(ts=...)` adds **no** solver
steps (`num_steps` is 15 for `n_save ∈ {2, 5, 50, 500}`) — it is pure interpolation.

**Ownership is half-open `(t_current, segment_t1]`.** A time landing exactly on an event
is owned by the segment that *ends* there, so it reports the PRE-jump state, matching the
old `event_states_before` semantics. The post-sample volume correction and the `t0` patch
are unchanged.

### The per-segment window

Without a bound, diffrax writes **every** output slot in **every** segment (out-of-range
times pin onto the segment endpoints), making the work `O(n_segments × n_output)` —
measured **10× slower** than a boundary save on a 160-event process under `vmap`. Each
segment therefore sees only a `dynamic_slice` window of `K` output times.

Per-segment cost scales with `K`, so it must be tight (161 segments, `vmap`×8, float64):

| `K` | µs/segment | vs `SaveAt(t1=True)` |
|---|---|---|
| none (`t1` only) | 10.6 | 1.00× |
| 1 | 14.0 | 1.32× |
| 10 | 57.7 | 5.47× |
| 25 | 68.5 | 6.49× |
| 100 | 107.4 | 10.18× |
| 200 | 208.8 | 19.78× |

`K` is exact, not estimated:

```
K = ceil(f · n_linspace) + G + 2
```

- **`f`** — largest inter-event gap as a fraction of the measurement window.
- **`G`** — largest measurement count in one gap, on the *padded* grid.
- **`n_linspace`** — `dense_grid_n + prediction_grid_n`, a static int the trainer holds.

`f` and `G` are collection-wide constants on `ControlsStore` (`_output_window_bounds`).
The bound is provable: an `N`-point linspace over `[t0, t1]` has spacing
`h = (t1−t0)/(N−1)`, so a gap of length `f·(t1−t0)` holds at most `floor(f·(N−1)) + 1` of
its points; two blocks give `f·n_linspace + 2`. `CallbackSolution.output_overflow` is an
assertion that the derivation holds, not a safety net.

**`is_controlled` process variables are excluded from `G`.** They are pH, temperature,
gas flow — RHS inputs logged at 1000+ points that `training_data` rejects as targets. In
an earlier revision counting them put `G` at 288/317/451 instead of 1/15/6 on the three
examples, pushing `K` past the grid so the window clamped and did nothing.

Resulting `K`:

| example | `f` | `G` | K (training) | K (export, +200) |
|---|---|---|---|---|
| 00_e2e_sim | 0.200 | 1 | 3 | 43 |
| 01_kittler_2022 | 0.187 | 15 | 17 | 55 |
| 11_tub_2026/fba_hyb | 0.274 | 6 | 8 | 63 |

### One step budget

`max_steps_per_segment` and `_MAX_STEPS_PER_SEGMENT` (512) are deleted. `solver.max_steps`
now bounds both each segment's inner solve and the running total. The per-segment cap was
a pmap latency guard predating the trajectory budget; with few long segments it silently
became the binding constraint.

### Two bugs fixed on the way

- **`done` lanes were not collapsed**, only `terminated` ones. Every scan iteration past
  the last event ran a tolerance-length (not zero-length) segment costing ≥1 step, and
  those steps consumed `max_steps` — ~200 of a 2048 budget on an export grid.
- **An empty preset array reached `jnp.argmin`** (`has_presets` counted callbacks, not
  times). Zero bolus *and* zero sample events is legitimate data; 2025_digink has it.

### Deliberately rejected

Deriving a finer `fail_time` from the reached save slots. Implemented and reverted: on a
blow-up the solver *does* reach output points past the last good node, so a later cutoff
stops masking 1e11 values and presents them as real predictions. The conservative
node-level cutoff is the point of `fail_time`.

---

## Benchmarks

A/B of `92b5848` (before) vs `38a5261` (after), 100 epochs, `devices: max`, identical
configs and data. Mean epoch time over **clean epochs only** — rows where any sample hit
a failed ODE segment are dropped, because a failed lane collapses its segments and does
far less work.

### Dense-grid loss regime (kittler, 12 processes, non-negativity hinge on the dense grid)

| `dense_grid_n` | before | after | speedup |
|---|---|---|---|
| off | 444.4 ms | 379.5 ms | 1.17× |
| 32 | 502.6 ms | 367.2 ms | 1.37× |
| 128 | 900.6 ms | 390.0 ms | **2.31×** |
| 256 | 1436.8 ms | 416.7 ms | **3.45×** |

Before-time grows ~3.9 ms per dense point; after-time ~0.15 ms per dense point — a ~26×
difference in slope. This is the regime the change exists for, and the gain keeps growing
with grid density.

### Augmented measurement density (kittler, 30 processes, 45 points/process, dense OFF)

| before | after | speedup |
|---|---|---|
| 1827.0 ms | 652.6 ms | **2.80×** |

Purely from measurement times no longer splitting the integration. This is the shipped
augmentation workflow (`prepare-config.json → augmentation.n_time_points`), just denser.

### Stock examples

| example | before | after | speedup |
|---|---|---|---|
| `11_tub_2026/fba_hyb` (single) | 139.9 ms | 131.9 ms | 1.06× |
| `00_e2e_sim` | 178.3 ms | 196.8 ms | **0.91×** |

### Isolated solves (export grid, float64, GRU latent-ODE RHS)

Forward 1.2–11.7×, gradient 2.1–26×, largest on event-free processes (2022_roell
84.7 → 4.2 ms gradient). Scaling with output points, single lane:

| n_out | gradient before → after | ratio |
|---|---|---|
| 11 | 5.72 → 5.05 ms | 1.13× |
| 111 | 24.54 → 7.24 ms | 3.39× |
| 411 | 92.63 → 12.15 ms | 7.62× |

---

## How to predict the effect on a dataset

The single quantity that decides it: **how many output times are not already events.**

| dataset | measurements on an event | segments before → after | result |
|---|---|---|---|
| 00_e2e_sim | 4 / 6 | 8 → 7 | 0.91× |
| fba_hyb | 16 / 19 | 117 → 115 | 1.06× |
| kittler45 (augmented) | 0 / 7 | 15 → 9 | 2.80× |

An offline measurement requires a sample draw, and a sample draw is a
`SampleVolumeChange` — an event. So on datasets where measurements coincide with
sampling, those times were **already** boundaries: tier A removes nothing while
`SaveAt(ts=...)` still costs its per-segment overhead. e2e_sim removes one segment of
eight and pays overhead on the rest; its losses are bit-identical, confirming no output
value changed.

Augmentation is the opposite: it resamples measurement times onto a random grid but
leaves sampling events alone, so none coincide and every measurement was a gratuitous
boundary.

**Rule of thumb.** Gains scale with `n_output / n_events`. Dense-grid losses, augmented
or resampled data, and forward/export grids gain 2–3.5×; datasets whose measurements sit
on their sample draws gain nothing and lose ~10%.

---

## Secondary observations

- **Fewer failed segments.** On kittler45, before had **20** failure epochs vs after's
  **2**. Plausibly the `done`-lane fix returning ~200 steps of the 2048 budget that were
  being spent integrating nothing — not isolated, so treat as an observation.
- **Grid-independence.** `fail_time`, and therefore which samples are masked out of the
  loss, no longer moves when a denser output grid is requested. This was a real defect.
- **Numerics change.** Output values at non-node times are now Tsit5's 4th-order dense
  interpolant rather than 5th-order step endpoints. `k_128` first-epoch loss is 203229
  (before) vs 481.16 (after). At tight tolerance the two solvers agree to **6.8e-12**, so
  the readout is equivalent; the difference is discretisation. `test_stateful_convergence`
  was restated as the contraction it was checking (gaps 1.8e-3 → 1.2e-5 → 2.4e-6);
  production `rtol=1e-5` is inside the converged regime.

---

## Benchmarking pitfall (worth recording)

The first round of end-to-end A/B numbers was **invalid**: every "before" run silently
imported the *new* code. Two independent mechanisms:

1. `bp_train` is an **editable install**, which registers a `MetaPathFinder`.
   `sys.meta_path` is consulted *before* `sys.path`, so `PYTHONPATH=<archive>` loses.
2. The runner did `cd $REPO`, putting `''` (the repo) at `sys.path[0]`.

The tell was bit-identical losses across 500 optimizer steps, which should have been
implausible on its face. Any future A/B against an archived commit must (a) run from a
neutral cwd, (b) strip the editable finder via `sitecustomize.py`, and (c) **assert** the
imported tree before running — the harness in `scratchpad/ab/run.sh` does all three, and
the assertion caught a recurrence later in the same session.

---

## Verification

625 tests pass. New coverage:

- `tests/test_output_grid.py` — the window is bitwise identical to the whole grid; the
  bound covers every gap (brute-forced); an undersized window is a loud `output_overflow`;
  controlled PVs do not inflate the bound while measured PVs do; answers are independent
  of output-grid size; a zero-event process solves.
- `tests/diffrax_callback/test_done_and_empty_presets.py` — `done` lanes take zero steps
  and do not consume the budget; empty preset times solve as a single segment.

Constraints: `pytest -n 4` (never `-n auto`), `BP_TRAIN_DEVICES=1`. The 132-process
kittler at `devices: max` aborts XLA CPU on a 19 GB box — on the pre-change code too, so
it is a machine limit, not a regression.
