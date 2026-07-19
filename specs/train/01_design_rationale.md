# Design Rationale

This document explains the cross-cutting design decisions behind bp-train.
Individual topic docs reference these sections for context.

bp-train turns a [bp-format](../../bp-format) `BioProcessCollection` into a
trainable hybrid ODE model: it builds the mechanistic mass balance from the
process definition, lets you plug a neural / mechanistic **reaction module** and
a **loss module** in via `custom.py` hooks, and runs the
prepare → train → forward / loo pipeline on JAX + Diffrax.

## 1. Built on bp-format, JAX, Diffrax, Equinox, optax

bp-train consumes the data structures and the mechanistic RHS from bp-format and
adds the training machinery:

- **bp-format** owns the data model and `build_rhs_ode(process)` →
  `RhsOde`, which formalizes which species/feeds/PVs are *modeled* (dynamic
  states whose rates the reaction module predicts) vs *controlled* (driven by
  the recorded control signals). bp-train never re-derives layout — `RhsOde` is
  the single source of truth for axis names and ordering.
- **JAX** provides autodiff + JIT for the repeated forward solves.
- **Diffrax** integrates the ODE with an adaptive solver and adjoint
  backprop.
- **Equinox** (`eqx.Module`) makes the reaction module, loss module, controls
  store, and wrapper JAX pytrees that pass through `eqx.filter_jit` cleanly.
- **optax** supplies the optimizer (`adam`/`sgd`) and gradient clipping.

## 2. Scaled (SCL) vs physical (RAW) space

Neural-ODE training is numerically fragile when state magnitudes span orders of
magnitude (biomass ~g/L, volume ~L, cumulative feed ~L). bp-train integrates the
ODE in **scaled space (SCL)** so every axis is O(1), which keeps gradients
well-conditioned, then converts to **physical space (RAW)** only where the
chemistry needs real units.

Scaling is a single linear factor per semantic axis. The 11 data-derived
`SCALE_*` axes (see [`EstimatedScales`](../bp_train/model_api.py)) cover states,
rates, cumulative volumes, feed compositions, and process variables. Stateful
reaction modules additionally own `SCALE_latent`. The physical SCL state is

```
SCL_state = [ modeled_RMCs | modeled_PVs | V_in_cumulative | modeled_FVCs_cumulative ]
SCL_integrated_state = [ SCL_state | SCL_latent ]
```

with `SCALE_state` and `SCALE_integrated_state` the matching concatenations (see
[`UserReactionModule.SCALE_state`](../bp_train/model_api.py)). Because scaling is
linear, the same factor converts both a value and its time-derivative
(`d(x/k)/dt = (dx/dt)/k`), so the `scale_*` / `unscale_*` helpers work for states
and rates identically.

**Single source of truth:** every `SCALE_*` vector lives on the
`UserReactionModule` (frozen fields). The wrapper reads
`reaction_module.SCALE_*`; the trainer reads `SCALE_state` to convert
measurements to SCL space; the loss module reaches them via
`inputs.reaction_module.SCALE_*`. Scales are never duplicated onto inputs.

## 3. Bounded physical-state solve

The ODE is solved with a **bounded physical-state** integrator
([`physical_solve.py`](../bp_train/physical_solve.py)) that applies discrete
jumps at control-event times. This replaced an earlier single pseudobatch solve:
the pseudobatch accumulator is unbounded over a long fed-batch, which corrupted
the reverse-mode adjoint and produced unstable gradients. Solving the bounded
physical state with explicit event jumps keeps the adjoint well-behaved.

## 4. One shared ODE solve feeds both reaction and loss

Each sample is solved **once**. The solver saves states/rates at the measurement
times (and, when a loss module opts in, at a dense grid — see
[Dense-grid losses](04_reaction_and_loss.md#dense-grid-losses)). The reaction
module produces the rates *inside* that solve; the loss module reads the saved
outputs *after* it. Adding dense save points costs extra `SaveAt` evaluations,
not extra solver steps.

## 5. Trainable partition via field tags

What gets optimized is declared with field metadata, not a partition function:

- [`trainable_field()`](../bp_train/model_api.py) marks an `eqx.Module` field's
  array leaves as trainable; [`frozen_field()`](../bp_train/model_api.py) marks
  them frozen. **Untagged array leaves default to frozen.**
- [`partition_trainable(module)`](../bp_train/model_api.py) splits any module
  (including the whole wrapper) into `(trainable, static)` pytrees. The
  inheritance rule is *first explicit tag on the path wins*, so an untagged
  container lets its children's own tags through.

This is the single mechanism — there is no per-module override. The loss module
is partitioned by the same rule, so trainable loss parameters (e.g. Kendall
uncertainty weights) are optimized alongside the reaction module. For sub-field
control (freeze some MLP layers), use the
[`build_optimizer`](02_cli_and_config.md#build_optimizer) hook with
`optax.masked` / `optax.multi_transform`.

## 6. Mean loss aggregation

A loss module returns a dict of **named scalar losses**; the total for backprop
is `mean(named_losses.values())`, not the sum.

bp-train clips the **raw** gradient (`clip_by_global_norm`) *before* Adam. Mean
keeps the gradient magnitude independent of the term count, so a tuned
`grad_clip_norm` keeps behaving the same as you add named terms — the clip stays
dormant in normal training. Sum would scale the gradient by the term count, push
it past the clip threshold, and (because the clip sits before Adam) hold the
step size large near the optimum → overshoot / divergence on stiff neural-ODE
problems. For weighted-sum behavior, scale the individual terms inside
`__call__` and retune `grad_clip_norm`.

## 7. Discrete events as differentiable state jumps

Controlled boluses and samples are applied at their known event times as
discrete, differentiable **state jumps** — not as continuous ramps in the ODE
RHS. The bounded solve (§3) runs as a sequence of segment solves with the jumps
applied *between* segments via `diffrax_callbacks` (`PresetTimeCallback`), so the
adjoint stays standard and correct (a gradient through a preset jump matches the
analytic value to ~9 digits).

At a coincident timestamp the jump is ordered **sample first** (well-mixed
removal: concentrations unchanged, volume drops) **then bolus** (dilute from the
post-sample volume and add the fed mass). These state jumps are handled by the
callback, not solver `jump_ts` hints. `jump_ts` instead contains genuine
vector-field discontinuities from `BioProcess.discrete_events` (toggle with
`solver.jump_ts`). Training/evaluation loss is still sampled only at measurement
timestamps, so a measurement strictly before a bolus (`t_sample < t_bolus`) is
unaffected by it. Continuous controlled feeds are not events — they are
piecewise-linear signals evaluated inside the RHS at each `t`.

## 8. Trainable-partition-only serialization

`save_model` writes **only the trainable partition** (`params.eqx`). The static
half (controls store, `RhsOde`, indices, `SCALE_*`) is **always rebuilt** from
`prepared.json` + `custom.py` at load time. This avoids shape mismatches on the
controls store, keeps checkpoints small, and is forward-compatible with future
trainable controls. Every checkpoint directory is self-contained (it bundles
`prepared.json.gz` and `custom.py`). See
[06_serialization_inspect.md](06_serialization_inspect.md).

## 9. Opt-in multi-core device pooling

Training can shard the process batch across CPU cores via `pmap` (~N speedup).
This is **opt-in** and resolved *before* JAX initializes (the device count is
fixed at import time): set `train.devices: N` (or `"max"`) in the config, or
`BP_TRAIN_DEVICES=N` (the env var always wins). `"max"` resolves to
`min(n_processes, n_cpus)`. Default is 1 device (unchanged behavior) so bp-train
never competes for cores with other work. No effect on GPU.
