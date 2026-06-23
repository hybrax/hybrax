# Pseudobatch training divergence — diagnosis

**Date:** 2026-06-13
**Code under test:** `e641709` ("pseudobatch" reformulation) vs `54d344f` (previous
"continuous-volume" formulation, one commit earlier).
**Case study:** `examples/11_tub_2026` (`tub_2026_all`, 37 V. natriegens fed-batch
processes; the runs live in `migration/` and were copied to `migration_pseudo/`
and `migration_pmap/`).

## TL;DR

The pseudobatch reformulation is **faster but its training diverges to `inf`**. The
root cause is **not** the forward model (which is correct) — it is that the
**reverse-mode gradient (adjoint) is catastrophically ill-conditioned** in the
pseudobatch state space. At identical weights the two formulations produce the same
forward trajectory but the pseudobatch gradient is up to ~10¹⁴× larger than the true
gradient. This corrupts the optimiser step, which pushes the model deeper into the
bad regime, which inflates the gradient further → runaway → `inf`, after which a
`NaN`-poisoning bug in the optimiser chain makes the run unrecoverable.

`stop_gradient(ADF)` does **not** help (ADF is param-independent). Lower LR does not
help (the gradient *direction* is wrong, not just the magnitude). `float64` and
cutting the C→ANN feedback each remove a large factor but leave a biased, still-huge
gradient. The clean fix is a **bounded physical-state reformulation**.

**Resolved** (see [Resolution](#resolution--what-actually-shipped-physical_solvepy-via-diffrax_callbacks)
below). Shipped as `bp_train/physical_solve.py`: the physical `[C | V | modeled_cum]`
state is integrated directly and boluses/samples are applied as **truly-discrete state
jumps between segment solves** via `diffrax_callbacks`, giving a correct adjoint
(9-digit). Full 1500-step training: **best loss 0.043** (beats the continuous pmap run's
0.048), **0.54 s/step** (fastest of the three), **0 `inf`** — vs pseudobatch's `inf`.
Validated on a second model (`fba_hyb`): callbacks ≈ triangles in stability (both reach
0.1114, 0 `inf`) and is faster on both models; final volume is exact to ~1e-8 (see the
cross-model section).

## What was compared

Same model, same data tensors (byte-identical; the prepared-file size delta is pure
provenance metadata), same `seed=42`, `lr=1e-2`, `grad_clip_norm=1.0`, `steps=1500`,
solver `max_steps=4096 rtol=1e-4 atol=1e-5`, both sharded 37-way. The **only**
material difference is the bp_train commit:

- **pmap run** = `54d344f` (continuous-volume): state is `[RMC concentrations | V_cum
  | modeled_FVC_cum]`; sampling/feeding/dilution are continuous ODE terms.
  → converged smoothly to total loss ≈ 0.050.
- **pseudo run** = `e641709` (pseudobatch): augmented state `[rmc_star | v_cont |
  modeled_cum | feed_corr | v0 | sample_dummy]`; boluses/samples applied
  algebraically via an Accumulated Dilution Factor (ADF); `jump_ts` reduced to event
  times only.
  → ~8% faster per healthy step, but **diverged to `inf`** at step ~1374 and never
  recovered.

## Symptom timeline (pseudo run)

Trained fine to ~step 1000 (loss ≈ 0.08). Destabilised ~1050–1175 (spikes to 0.45,
7.9), oscillated, then blew up over steps 1366–1373
(`15.9 → 45 → 283 → 153575`, driven by **acetate and glucose**) → step 1374 onward
all targets `inf`. Per-step `dt` crept 0.5 → 0.85 → 1.6 s as it degraded, then 3.65
s once `inf`. `sulfate` loss is 0.0 in both runs throughout (degenerate target).

## Root cause — the same-weights test

Load one shared set of ANN weights into **both** wrapper formulations (the
`reaction_module` is structurally identical across the two commits, so it serialises
from one and deserialises into the other), evaluate the same process, and compare the
forward glucose trajectory **and** the gradient of the loss w.r.t. the ANN weights:

| shared weights | glucose min | **PSEUDO ‖∇‖** | **CONTINUOUS ‖∇‖ (=true)** | ratio |
|---|---|---|---|---|
| step 500  (healthy)  | −4.7  | 2.31      | 2.17  | ~1× |
| step 1000 (healthy)  | −7.0  | 0.37      | 0.46  | ~1× |
| step 1100            | −4.8  | 46.8      | 1.70  | **28×** |
| step 1200            | −6.8  | 3.50      | 3.52  | ~1× |
| step 1300            | −35.6 | **4.08e16** | 83.8 | **~5e14×** |
| step 1350            | −29.0 | **7.43e6**  | 28.8 | **~2.6e5×** |

Two conclusions:

1. **The forward trajectories match at every checkpoint** (glucose min −29.0 vs −29.3
   at step 1350; losses within ~4%). So the pseudobatch *math is correct* — there is
   **no forward bug**, and the large-negative glucose is identical in both
   formulations. The continuous formulation gives the trustworthy *true* gradient
   (it matches pseudo in the healthy regime).
2. **The gradient is the destabiliser, and the explosion is episodic** — usually fine
   (~1×) but it spikes to 28× / 2.6e5× / 5e14× exactly when the model wanders into a
   bad-parameter configuration (large-negative recovered glucose). This matches the
   spiky training-loss curve.

## Mechanism

The recovered concentration is `C = (rmc_star + bolus_corr) / ADF`. The adjoint
gradient `dC/dθ = (1/ADF)·d(rmc_star)/dθ` where `rmc_star(t) = rmc_star(0) +
∫ ADF·bio(C,θ) dt` is the integrated **biological accumulator**. The sensitivity
`S = d(rmc_star)/dθ` obeys

```
dS/dt = (∂bio/∂C)·S + ADF·(∂bio/∂θ)
```

Compared with the continuous formulation `dS_c/dt = (∂bio/∂C − dilution)·S_c +
∂bio/∂θ`, the pseudobatch sensitivity ODE (a) **drops the stabilising `−dilution`
damping term** (it was folded into ADF algebraically, which does not damp the
*sensitivity*), and (b) **amplifies the forcing by ADF**. When the model over-consumes
(glucose → −29), `∂bio/∂C` is large/positive (the ANN extrapolating out of
distribution), so `S` grows without damping. The *forward* value survives via
cancellation (`rmc_star + bolus_corr` → small net), but the *adjoint* does not — its
huge intermediate sensitivities then lose all precision to float32 round-off.

### Two distinct contributions, isolated by experiment (all at step-1350 weights)

| intervention | ‖∇‖ | note |
|---|---|---|
| baseline (float32) | 7.43e6 | corrupted |
| **float64** | 1.56e4 | round-off accounts for ~470× |
| **`stop_gradient(ADF)`** | 7.43e6 | **no-op** — ADF is param-independent |
| **detach C→ANN feedback** (float32) | 1.13e4 | cuts `(∂bio/∂C)·S`; biased; round-off-insensitive |
| detach C→ANN feedback + float64 | 1.12e4 | ≈ detach alone → residual is genuine, not round-off |
| true gradient (continuous) | 28.8 | — |

So the 2.6e5× has two parts: **float32 round-off** (~470×, fixed by float64) **and a
genuine ~400× ill-conditioning** from the unbounded `rmc_star` accumulator path
(neither float64 nor the feedback-detach removes it; the detach merely returns a
*biased* gradient that is still ~400× too large).

## Why the obvious mitigations don't work

- **`stop_gradient(ADF)`**: proven no-op. ADF (and `bolus_corr`, `sample_factor`) are
  pure functions of the fixed feed/sample schedule — the gradient does not flow
  through the discrete-event quantities at all; it flows through the continuous
  `rmc_star` accumulator.
- **Lower LR**: tested by the user — avoids `inf` but plateaus at high loss. The
  gradient *direction* is corrupted/biased and `clip_by_global_norm` normalises the
  magnitude, so the optimiser takes wrong steps at any LR.
- **`float64`**: removes the round-off factor (~470×) but leaves the ~400× genuine
  ill-conditioning. Partial mitigation, ~10× slower, not a cure.
- **Detaching the C→ANN feedback** (a `stop_gradient` on the concentration the rate
  network sees): removes the recursive `(∂bio/∂C)·S` term but yields a **biased**
  gradient that is *still* ~400× too large — it neither converges correctly nor
  tames the magnitude.

## The irreversibility (separate, secondary bug)

Once a step produces a non-finite loss, the run can never recover: the optimiser chain
is `optax.chain(zero_nans(), clip_by_global_norm(1.0), adam(lr))`
([harness.py `_build_optimizer`](../bp_train/harness.py)). `zero_nans` runs *first* and
does not catch `inf`; `clip_by_global_norm` then computes `inf/inf = NaN`, permanently
poisoning the params and the Adam moment buffers. There is no `isfinite`/skip-step
guard anywhere in the train step. Fix ordering (clip before zero_nans, or add
`optax.apply_if_finite` / a non-finite skip-guard) so a single bad step is survivable.

## The fix

**Bounded pseudo-state reformulation.** Apply the bolus/sample events to the
*integrated state* as jumps at the event times (which are already in `jump_ts`, so no
speed loss), instead of accumulating a biology-only `rmc_star` and re-adding the feed
algebraically in the recovery. Then:

- the integrated state stays O(physical concentration) instead of growing as the
  difference of two large accumulators, and
- applying a sample (volume drop) to the state **re-introduces the per-event
  sensitivity damping** the continuous formulation has — a sample that removes 10% of
  volume damps the sensitivity by 10%.

This keeps the reduced-`jump_ts` speed benefit while making the adjoint
well-conditioned **and unbiased** (unlike any `stop_gradient` band-aid). It is a
segmented/event-stepped solve over the handful of event times.

## The missing test (why this shipped)

The suite asserts only forward/structural properties and runs ≤8 training steps with
small fixed rates (0.0–0.4); nothing reaches the large-rate / large-accumulator
regime, and nothing compares pseudobatch vs continuous. **Add a gradient-equivalence
regression test**: at *degraded* weights, assert the pseudobatch gradient ≈ the
continuous gradient (and that `max|rmc_star|` stays O(C·ADF)). Forward-equivalence
alone sails right past this.

## Resolution — what actually shipped (`physical_solve.py` via `diffrax_callbacks`)

The naive bounded-state reformulation looked dead because this data has **98–208
bolus events per process** (median 164, a feed every ~5 min — quasi-continuous), so
applying each as a state jump needs ~114–228 segments. The trilemma is **fast +
bounded physical state (correct gradient) + exact discrete boluses** — pick three.

### Approaches tried (on `exp_23083`, 114 events), and why they were rejected

| approach | gradient | verdict |
|---|---|---|
| single `diffeqsolve` (jump_ts), pseudobatch | **broken** | unbounded pseudo-state → ill-conditioned adjoint (this whole doc) |
| N separate `diffeqsolve`s | correct | 2.4× steps — each restarts the PID controller (warm `dt0` doesn't fix it) |
| custom `solver.step` loop that mutates state *inside* `step` | **broken** | FD ratio ≈ 0 — modifying state inside a solver step silently severs the diffrax adjoint |
| **`diffrax_callbacks` (scan of segments, jumps *between* segments)** | **correct (9-digit)** | **shipped** |

### The shipped design

`bp_train/physical_solve.py::solve_physical_states` integrates the **physical** state
`[C | V | modeled_cum]` directly (biology-only RHS, `wrapper.physical_rhs`) and applies
boluses/samples as **discrete state jumps at their known event times** using
[`diffrax_callbacks`](../../diffrax_callbacks) (`PresetTimeCallback` +
`diffeqsolve_with_callbacks`). Because the jumps happen *between* segment solves — never
inside `solver.step` — the adjoint is the standard `RecursiveCheckpointAdjoint` and is
**correct** (verified: gradient through a preset jump matches the analytic value to 9
digits). Preset times are `bolus ∪ sample ∪ measurement` times; inactive/padded slots
are parked at `t1 + 1e6` so they never trigger; the per-event `affect_fn` does the
mass-conserving mix `C2 = (C·V + Σ Cin·dv) / max(V+Σdv, min_V)`, then `V2 = V+Σdv − Σsample_dv`.

### Fixes required to make it work

1. **float32 event tolerances** in the package (`_solve.py`): the upstream
   `1e-10`/`1e-12` "is this a preset time?" tolerances are below float32 ULP at t≈15, so
   events silently failed to fire and the solve froze. Relaxed to
   `1e-5·(1+|t|)` / `1e-7·(1+|t|)`.
2. **`max_steps_per_segment`** — the per-segment step ceiling. Originally an early
   worry that a global `max_steps` buffer would be multiplied across *every* segment →
   OOM. Measurement disproved that: the segments run in a sequential `scan` (one solve
   live at a time) and `RecursiveCheckpointAdjoint` keeps only ~`O(log max_steps)`
   checkpoints, so RAM is **flat in the number of events** (see RAM note) and a high
   ceiling is free (8096 vs 512 → identical compile/speed/loss). So `solve_physical_states`
   now wires `--solver-max-steps` straight to the per-segment ceiling (`None` ⇒ use
   `max_steps`) — it was previously accepted but ignored. It is a *safety* bound (gives a
   stiff segment headroom), not a budget to ration across segments.
3. **equinox closure-convert leniency patch** (`_ad.py`, `.bak` saved): under `pmap`,
   the backward pass tripped `_check_closure_convert_input` on a `RESULTS` enum that is
   tracedness-mismatched but shape/dtype-identical. Patched to only raise on real
   shape/dtype differences. **This is `pmap`-specific**; the patch-free `shard_map`/GSPMD
   alternatives were investigated but are blocked or too slow (see the Sharding section),
   so `pmap` + this patch remains the fast default.
4. **`lr=3e-3`** (not `1e-2`): the truly-discrete loss has a steeper gradient; `1e-2`
   was stable for the first ~800 steps then drove into a stiff regime (dt 0.55→10.9 s,
   loss → 17). `3e-3` stays out of it.
5. **dense-grid feed double-count** (`physical_solve.py`): `affect_fn` applies *every*
   feed/sample within `_EVENT_EPS` (1e-4 h) of the current event node. When a
   measurement/output node lands within `eps` of a feed, that feed is applied a second
   time at the measurement node → the volume/concentrations drift by ~one bolus from
   there on. Invisible on the measurement (loss) grid — no point coincides with a feed —
   but it showed up in the 200-pt prediction grid for ~2/37 processes (max ΔV 2.0e-5 L).
   Fixed by parking any measurement/output node within `eps` of a feed/sample node
   (the gather still recovers its state from the feed node). Predictions now volume-exact
   (max ΔV **1.96e-8** L, all processes); training loss bit-identical (no coincidences
   there). See the volume-precision check below.

### Result — three-way comparison (full 1500-step runs)

| | **callbacks (truly discrete)** | pmap (continuous, triangles) | pseudobatch |
|---|---|---|---|
| best loss | **0.0431** (step 1477) | 0.0483 | 0.0746 |
| final loss | 0.153¹ | 0.052 | **`inf`** (diverged) |
| `inf` steps | **0 / 1500** | 0 | 127 |
| speed | **0.54 s/step** | 0.77 s/step | 1.06 s/step |
| total wall | **~13.7 min** | ~20.6 min | ~28 min |
| boluses | **truly discrete** | smoothed (triangle ramps) | discrete, broken adjoint |

¹ mild late oscillation (best 0.043 ↔ final 0.15, driven by plasmid); not divergence.
An LR schedule / early-stop-on-best locks in 0.043.

The truly-discrete method is **the most accurate** (0.043 < pmap 0.048; fitting the
boluses as the discontinuities they are beats smoothing them) **and the fastest**
(0.54 s/step), with stable training and zero `inf`.

### RAM — measured

The reason `max_steps_per_segment` exists: `RecursiveCheckpointAdjoint` stores a
checkpoint stack sized by `max_steps`. The segments run in a `lax.scan` (sequential), so
only **one** segment's `SaveAt` buffer is live at a time — RAM is `O(max_steps_per_segment
× n_state)`, **flat in the number of events**, not `O(global_max_steps × n_segments)`.
The earlier WSL crashes came from the *opposite* layout (a large global `max_steps`
reserved once and replayed/multiplied per segment).

**Measured** (full 37-process step, `--devices max` → sharded 37-way on a 128-core box,
`max_steps_per_segment=512`): **peak RSS 2.34 GB** — dominated by the XLA runtime +
compilation + 37 device buffers (~63 MB/device), *not* the solver buffers. It holds
steady at ~0.54 s/step across 1500 steps with no growth. Verdict: **not bad** —
comfortable on WSL, with headroom to drop the per-segment cap further if needed.

### Forward accuracy — verified vs the triangle (continuous) forward

At **identical weights** (pmap step-1500 `reaction_module` transplanted into the
callbacks wrapper), compared to the triangle `predictions.csv` on every state × 200
dense points × 37 processes:

- **Volume is exact.** `|callbacks_V − (V0 + Σboluses≤t − Σsamples≤t)|` ≤ **1.96e-8 L**
  (float32 noise on V≈0.013, after fix #5) — the callbacks volume *is* the true discrete
  step function from the controls, and is more faithful than the triangle's smoothed
  `V_real`. (Before fix #5, ~2/37 processes hit 2.0e-5 from the dense-grid double-count.)
- **Concentrations** agree with the triangle to within a **zero-mean sawtooth bounded by
  one bolus** — the intended discrete-vs-continuous difference. Worst case (glucose, the
  primary feed): signed-mean Δ = −0.023 g/L (no drift), RMS 0.20 g/L ≈ one-bolus bump
  (0.30 g/L); where there is no feed the agreement is ~1e-4. Initial conditions match to
  9.5e-7. No mass leak, no integration drift.

### Sharding: pmap vs shard_map vs GSPMD (jax 0.10.1)

Goal was to drop fix #3 (the equinox `pmap` patch) by moving to `shard_map`. Outcome —
**not currently possible without a big speed loss**; `pmap` stays the default. Three
paths tried (`bp_train/harness.py`):

| path | works? | patch-free? | speed | why |
|---|---|---|---|---|
| **`pmap`** (default) | ✅ | ❌ (needs fix #3) | **0.53 s/step** | the only fast option |
| `shard_map` (manual) | ❌ crash | — | — | `assert not hlo_sharding.is_manual()` in diffrax's nested `eqx.filter_eval_shape` solver loop |
| Explicit-sharding `jit` | ❌ crash | — | — | `ShardingTypeError` on diffrax's internal `select`s (sharding-in-types) |
| **GSPMD** Auto-mesh `jit` (`BP_GSPMD=1`) | ✅ | ✅ | **~34 s/step** | correct (loss trace *identical* to pmap) but XLA can't partition the data-dependent ODE solve — it replicates per-device work (~60× slower). Adding the equinox-tutorial `eqx.filter_shard` / `with_sharding_constraint` hints (params replicated, batch + per-sample loss sharded) **did not change it** — still 34 s/step; XLA replicates the diffrax `while_loop`s regardless |

The GSPMD step (`_make_gspmd_step`) is kept as a documented, **patch-free** opt-in
(`BP_GSPMD=1`): batch sharded over an `AxisType.Auto` mesh, params replicated, one
`jit`'d vmap-grad+update, XLA auto-reduces — no manual `psum`. It becomes the clean path
the day XLA partitions vmapped ODE solves (or diffrax composes with `shard_map`'s manual
mode). Until then, **`pmap` + the equinox patch is the fast path**; `shard_map` is blocked
upstream (jax 0.10 manual-mode × diffrax `filter_eval_shape`).

### Remaining

- **LR schedule / early-stop** to hold the 0.043 best instead of the 0.15 final.
- Revisit `shard_map` when jax/diffrax fix the manual-mode `filter_eval_shape` compose,
  or pin **jax 0.9.2** (where sharding worked patch-free per the env notes) if dropping
  the monkeypatch matters more than staying on latest jax.

## Cross-model validation: `fba_hyb` (callbacks vs triangles)

A second model (`examples/11_tub_2026/fba_hyb`, FBA-hybrid, 37 processes) was used to
check the callbacks formulation head-to-head against the **triangle (continuous)**
formulation under matched conditions: same raw data, *identical* `custom.py`, seed 42,
`lr=1e-3`, same solver settings, both pmap-sharded 37-way. The triangle run was produced
from the `54d344f` worktree (continuous code); step-1 loss is bit-identical (4.5443 both),
confirming a fair comparison. Each formulation prepared its data with its own native
pipeline.

### Stability — both stable to 1500 steps

| | callbacks (discrete) | triangles (continuous) |
|---|---|---|
| step 250 / 500 / 1000 | 0.243 / 0.142 / 0.1165 | 0.357 / 0.142 / 0.1218 |
| **final (1500)** | **0.1114** | **0.1114** |
| best | 0.1083 @1453 | 0.1104 @1486 |
| back-half max (750–1500) | 0.169 @909 | 0.157 @1250 |
| **`inf`** | **0** | **0** |
| speed (pmap 37-way) | **0.50 s/step** | 0.58 s/step |

Both formulations are **equivalently stable** — they converge to the *identical* final
loss (0.1114), each rides out one mild recoverable back-half wobble, and neither produces
a single `inf`. Both also share a transient spike at step ~101 (max ~61 in the 500-step
run) that recovers — a model/LR/seed artifact, not a formulation effect. Callbacks is
marginally smoother and ~14 % faster per step. So the truly-discrete formulation is a
**safe drop-in** for triangles here — unlike the original pseudobatch, which diverged.

### Speed — both models (37 processes, pmap 37-way)

| model | callbacks (discrete) | triangles / continuous | pseudobatch |
|---|---|---|---|
| migration (`tub_2026`) | **0.54 s/step** | 0.77 s/step | 1.06 s/step → `inf` |
| `fba_hyb` | **0.50 s/step** | 0.58 s/step | — |

Callbacks is the fastest formulation on **both** models (1.4× vs triangles on migration,
1.16× on `fba_hyb`). (Migration callbacks used `lr=3e-3`, the others `lr=1e-2`; `fba_hyb`
all used `lr=1e-3`. The single-device triangle run was ~16× slower — 7.4 s/step — until
`--devices 37` engaged pmap; `54d344f`'s old `batch >= n_devices` gate silently disabled
sharding under `--devices max` because 128 cores > 37 batch.)

#### Single-device (`--devices 1`) — the gap widens to 3× (`fba_hyb`)

| `fba_hyb` | callbacks | triangles | callbacks advantage |
|---|---|---|---|
| 37-way pmap | 0.50 s/step | 0.58 s/step | 1.16× |
| **single device** | **2.42 s/step** | **7.34 s/step** | **3.0×** |
| sharding speedup | 4.8× | 12.7× | — |

On one CPU device the discrete formulation is **~3× faster** than triangles, because the
triangle (continuous) per-process solve is intrinsically heavier — it resolves the triangle
ramps plus the long `jump_ts` list with many small adaptive steps, whereas callbacks takes
only a few steps per inter-event segment. With 37-way pmap that per-process cost is hidden
behind device parallelism (triangles benefits more from sharding, 12.7× vs 4.8×), which
compresses the visible gap to 1.16×. So on a normal few-core machine — no 37-way sharding —
callbacks wins decisively.

### Why per-step time is robust to the usual knobs (measured)

On the callbacks path the per-step time is **insensitive** to every per-call/per-step knob
(fba_hyb, 37-way): `--no-jump-ts` 0 %, `max_steps_per_segment` 512→32 0 %, solver tolerance
1e-4→3e-3 (30×) 0 %. The cost is **structural** — a fixed `lax.scan` over ~247 event slots
per solve (208 boluses + 20 samples + 19 measurements, padded to the batch max), each a
small `diffeqsolve` + adjoint. The work *inside* a segment is negligible; the *count* is the
cost, and it is set by the number of feed events (inherent to the truly-discrete approach).
`jump_ts` is computed and threaded to the loss fn every step but never reaches
`solve_physical_states` — dead weight, but free to carry. The only structural headroom is
cutting segment count: e.g. saving the ~19 measurement times via dense interpolation *within*
a segment instead of as event nodes (~8 %), which would require a `diffrax_callbacks` change.

### Volume precision — does the final V really equal `V0 + Σfeeds − Σsamples`?

Yes — both formulations land on the analytic step volume, to high precision (final V per
process vs analytic, all 37 processes):

| | callbacks (discrete, after fix #5) | triangles (continuous) |
|---|---|---|
| median \|V_final − analytic\| | **1.0e-8 L** | 1.45e-6 L |
| max \|V_final − analytic\| | **1.96e-8 L** | 2.67e-6 L |

The difference is mechanistic: **callbacks is exact by construction** — V is literally
`V0 + cumsum(feeds) − cumsum(samples)` applied as discrete jumps, so the error is pure
float32 roundoff (median 1e-8). **Triangles *integrates* the feed rate** through the ODE
solver, so it lands within solver tolerance (~1.5e-6, more uniform but looser at the
median). Both are far below any physically meaningful threshold (< 0.02 % of V≈0.013 L)
and agree with each other to ~1e-6. (The callbacks max was 2.0e-5 before fix #5 — the
dense-grid feed double-count; see the Fixes list.)

## Reproduction

Worktree at the pseudobatch commit (so the install tracking the main checkout stays
put): `git worktree add --detach <wt> e641709`. The probe loads a shared
`reaction_module` into each formulation and prints loss + ANN gradient norm; run with
`PYTHONPATH=<worktree>` for pseudo and `PYTHONPATH=<main checkout>` for continuous,
`JAX_PLATFORMS=cpu`, env `PREP/MODEL/CUSTOM/PROC`. Set `X64=1` for the float64 run.
Scripts used: `/tmp/sameweights.py` (forward+grad, both formulations) and
`/tmp/pgrad.py` (single-formulation grad, optional float64). Env: jax 0.9.2,
optax 0.2.8, diffrax 0.7.2.
