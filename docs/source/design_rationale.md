# Design Rationale

This document explains the cross-cutting design decisions behind **bp-format** and
**bp-train**. Individual module/topic docs reference these sections for context.

The two packages form one stack:

- **bp-format** owns the data model and the mechanistic right-hand side. It turns a
  bioprocess definition into JAX-compatible data structures and a differentiable ODE
  RHS (`build_rhs_ode(process)` → `RhsOde`), formalizing which species / feeds / process
  variables are *modeled* (dynamic states) vs *controlled* (driven by recorded signals).
- **bp-train** consumes those structures and adds the training machinery: it lets you
  plug a neural / mechanistic **reaction module** and a **loss module** in via `custom.py`
  hooks and runs the prepare → train → forward / loo pipeline on JAX + Diffrax + optax.
  bp-train never re-derives layout — `RhsOde` is the single source of truth for axis
  names and ordering.

---

## Shared foundation: JAX-first architecture

Both packages are built on [JAX](https://github.com/google/jax) and
[Equinox](https://github.com/patrick-kidger/equinox) to support:

- **Automatic differentiation (AD):** Gradient-based optimization of hybrid bioprocess
  models requires differentiating through ODE integration, spline evaluation, and loss
  computation. JAX's `jax.grad` / `jax.value_and_grad` enable this end-to-end.
- **JIT compilation:** `jax.jit` (or `eqx.filter_jit`) compiles Python functions to
  optimized XLA code, critical for the repeated forward solves in training loops.
- **Vectorization:** `jax.vmap` allows batching over processes or parameter sets without
  manual loop code.

**Why Equinox?** Equinox provides `eqx.Module`, a frozen dataclass that registers as a
JAX pytree. Objects like `TimeSeries`, `ControlSplines`, `RhsOde`, and (in bp-train) the
reaction module, loss module, controls store, and wrapper can be passed into and out of
JIT-compiled functions without manual pytree registration. `eqx.filter_jit` automatically
separates static (non-differentiable) fields from dynamic (array) leaves.

**Constraints this introduces:**

- Array fields must be JAX arrays (`jnp.ndarray`), not plain NumPy.
- Python dicts with string keys are not natural JAX pytree leaves. The outer container
  classes (`BioProcess`, `CaseStudy`, etc.) use standard Python `@dataclass` rather than
  `eqx.Module` because they hold `Dict[str, ...]` fields manipulated outside the JIT
  boundary. Only the inner numerical objects (e.g., `TimeSeries`) need to be `eqx.Module`.
- Mutation is not allowed on `eqx.Module` instances (they are frozen). Use `eqx.tree_at`
  for functional updates.

---

## bp-format

### 1. Hierarchical data model

Bioprocess experiments are organized in a three-level hierarchy. The top-level
artifact — one file on disk — is either a strict `CaseStudy` or a loose
`BioProcessCollection`:

```
CaseStudy             (one publication / experimental campaign — strict metadata)
  -> BioProcess       (one experimental run)
    -> Components      (reactor medium, volume, process variables)

BioProcessCollection  (raw / intermediate processes — no strict metadata)
  -> BioProcess
    -> Components
```

- **CaseStudy** corresponds to one publication or experimental campaign. It carries
  `organism`, `citation`, and a `case_id`, which is the natural grouping for
  leave-one-process-out cross-validation. Each case study is its own file.
- **BioProcessCollection** is the loose counterpart: a dict of processes plus
  optional free-form metadata, for raw or intermediate data not yet a full-fledged
  case study.
- **BioProcess** is a single fermentation run: time axis, volume operations, reactor
  medium concentrations, and process variables.
- **Components** within a process are organized by physical role: `ReactorMedium`
  (concentration time series — biomass, substrates, products), `Volume` (all
  volume-change operations), and `ProcessVariable` (non-concentration signals: pH,
  temperature, dissolved oxygen, off-gas).

**Why `Dict[str, ...]` keyed by name?** String-keyed dicts provide O(1) lookup by name,
produce readable JSON, and make it easy to iterate over components. Lists would require
linear search and lose the semantic naming.

### 2. Volume as a first-class concept

Volume is not a state variable, not a control input, and not a process variable. It is
its own category because:

- **Multiple operations affect it:** continuous feeds, bolus feeds, and sampling events,
  each with different media compositions and schedules.
- **It interacts with the ODE differently:** each feed stream `k` contributes a dilution
  term `(f_k / V) * (C_in[k,i] - c_i)` for every species `i`, and volume itself evolves
  as `dV/dt = sum(f_k)`.
- **It carries composition metadata:** each `FeedVolumeChange` references a `FeedMedium`
  defining what enters the reactor; `SampleVolumeChange` removes reactor contents at
  current concentrations.

The `Volume` dataclass aggregates `initial_volume` and a dict of `VolumeChange` entries.
Sign conventions are enforced: feeds are non-negative, samples are non-positive.

### 3. TimeSeries structure

The `TimeSeries` class (an `eqx.Module`) stores measured data as `times` and `values`
arrays, and optionally carries fitted spline coefficients (`breaks`, `coeffs`,
`segment_start_piece_idx`) for continuous-time interpolation.

`TimeSeries` itself is agnostic about whether its data represents continuous or discrete
quantities — that semantic is set by the parent object (e.g., a `VolumeChange` with
`is_continuous=True` uses it for a continuous flow profile; `is_continuous=False` means
discrete event data such as bolus feeds or sampling, where fitting splines is not
meaningful).

**Why optional spline coefficients?** The raw `times`/`values` are the experimental
ground truth (needed for loss and validation); spline coefficients enable continuous-time
evaluation during ODE integration without re-interpolating at each solver step. A
`TimeSeries` can be spline-only in pseudobatch workflows where the original samples are no
longer meaningful.

**Why power-basis storage (not B-spline basis)?** Power-basis polynomials
`c[0]*h^3 + c[1]*h^2 + c[2]*h + c[3]` (with `h = t - t_break`) evaluate with Horner's
method in a few multiply-adds — simple, fast, and clean to vectorize in JAX. B-spline
basis evaluation requires recursive knot-vector lookups that are harder to vectorize.

### 4. Pseudobatch normalization

Fed-batch processes change volume over time, so observed concentrations are affected by
dilution in addition to biological activity. The **pseudobatch transform**
(Hesselberg-Thomsen et al., 2024) converts measured concentrations `c(t)` to
pseudo-concentrations `c*(t)` — what concentrations *would have been* in a batch process
with the same biological activity:

```
c*(t) = c(t) * ADF(t) - feed_correction(t)
```

where `ADF(t)` is the accumulative dilution factor (current / initial volume) and
`feed_correction(t)` accounts for mass added by feed streams. This yields smoother curves
(better cubic-spline fits), enables fair comparison across batch and fed-batch processes,
and supports spline segmentation at bolus-event discontinuities. ADF and feed correction
are piecewise-constant and must be evaluated with **step** (nearest-neighbor)
interpolation, not linear, to preserve correct discontinuity behavior in the backtransform.

### 5. Validation-first approach

Bioprocess data comes from diverse sources, with common errors like negative feed volumes,
missing biomass component, mismatched times/values lengths, feed media that omit reactor
species, or measurement times coinciding with sampling events. bp-format validates early
and explicitly. All validation functions return `(bool, str)` tuples, so callers can
collect **all** issues in one pass and present a comprehensive report rather than failing
on the first error. `validate_process()` runs single-process checks; `validate_case_study()`
adds cross-process consistency checks.

---

## bp-train

### 1. Built on bp-format, JAX, Diffrax, Equinox, optax

bp-train consumes the data structures and mechanistic RHS from bp-format and adds training:

- **bp-format** owns the data model and `build_rhs_ode(process)` → `RhsOde` (the single
  source of truth for axis names and ordering; bp-train never re-derives layout).
- **JAX** provides autodiff + JIT for the repeated forward solves.
- **Diffrax** integrates the ODE with an adaptive solver and adjoint backprop.
- **Equinox** makes the reaction module, loss module, controls store, and wrapper JAX
  pytrees that pass through `eqx.filter_jit` cleanly.
- **optax** supplies the optimizer (`adam`/`sgd`) and gradient clipping.

### 2. Scaled (SCL) vs physical (RAW) space

Neural-ODE training is numerically fragile when state magnitudes span orders of magnitude
(biomass ~g/L, volume ~L, cumulative feed ~L). bp-train integrates the ODE in **scaled
space (SCL)** so every axis is O(1) — keeping gradients well-conditioned — then converts
to **physical space (RAW)** only where the chemistry needs real units.

Scaling is a single linear factor per semantic axis (13 `SCALE_*` axes on
`EstimatedScales`, covering states, rates, cumulative volumes, feed compositions, and
process variables). The integrated SCL state vector is

```
SCL_state = [ modeled_RMCs | modeled_PVs | V_in_cumulative | modeled_FVCs_cumulative ]
```

Because scaling is linear, the same factor converts a value and its time-derivative
(`d(x/k)/dt = (dx/dt)/k`), so `scale_*` / `unscale_*` work for states and rates alike.
**Single source of truth:** every `SCALE_*` vector lives on the `UserReactionModule`
(frozen fields); the wrapper, trainer, and loss module all read them there — scales are
never duplicated onto inputs.

### 3. Bounded physical-state solve

The ODE is solved with a **bounded physical-state** integrator (`physical_solve.py`) that
applies discrete jumps at control-event times. This replaced an earlier single pseudobatch
solve: the pseudobatch accumulator is unbounded over a long fed-batch, which corrupted the
reverse-mode adjoint and produced unstable gradients. Solving the bounded physical state
with explicit event jumps keeps the adjoint well-behaved.

### 4. One shared ODE solve feeds both reaction and loss

Each sample is solved **once**. The solver saves states/rates at the measurement times
(and, when a loss module opts in, at a dense grid — see
[Reaction & loss](narrative/bp-train/04_reaction_and_loss.md)). The reaction module
produces rates *inside* that solve; the loss module reads the saved outputs *after* it.
Adding dense save points costs extra `SaveAt` evaluations, not extra solver steps.

### 5. Trainable partition via field tags

What gets optimized is declared with field metadata, not a partition function:
`trainable_field()` marks an `eqx.Module` field's array leaves as trainable;
`frozen_field()` marks them frozen; **untagged array leaves default to frozen**.
`partition_trainable(module)` splits any module (including the whole wrapper) into
`(trainable, static)` pytrees, with the rule *first explicit tag on the path wins*. This
is the single mechanism — no per-module override. The loss module is partitioned by the
same rule, so trainable loss parameters (e.g. Kendall uncertainty weights) are optimized
alongside the reaction module. For sub-field control (e.g. freezing some MLP layers), use
the `build_optimizer` hook with `optax.masked` / `optax.multi_transform`.

### 6. Mean loss aggregation

A loss module returns a dict of **named scalar losses**; the total for backprop is
`mean(named_losses.values())`, not the sum. bp-train clips the **raw** gradient
(`clip_by_global_norm`) *before* Adam. Mean keeps gradient magnitude independent of the
term count, so a tuned `grad_clip_norm` behaves the same as you add named terms. Sum would
scale the gradient by the term count, push it past the clip threshold, and (because the
clip sits before Adam) hold the step size large near the optimum → overshoot / divergence
on stiff neural-ODE problems. For weighted-sum behavior, scale individual terms inside
`__call__` and retune `grad_clip_norm`.

### 7. Discrete events as differentiable state jumps

Boluses and samples are applied at their known event times as **discrete, differentiable
state jumps** — not continuous ramps in the RHS. The bounded solve (§3) runs as a sequence
of segment solves with jumps applied *between* segments via `diffrax_callbacks`
(`PresetTimeCallback`), so the adjoint stays standard and correct (gradient through a
preset jump matches the analytic value to ~9 digits). At a coincident timestamp the jump
is ordered **sample first** (well-mixed removal: concentrations unchanged, volume drops)
**then bolus** (dilute from the post-sample volume and add the fed mass). Loss is sampled
only at measurement timestamps, so a measurement strictly before a bolus is unaffected by
it. Continuous controlled feeds are not events — they are piecewise-linear signals
evaluated inside the RHS at each `t`.

### 8. Trainable-partition-only serialization

`save_model` writes **only the trainable partition** (`params.eqx`). The static half
(controls store, `RhsOde`, indices, `SCALE_*`) is **always rebuilt** from `prepared.json`
+ `custom.py` at load time. This avoids shape mismatches on the controls store, keeps
checkpoints small, and is forward-compatible with future trainable controls. Every
checkpoint directory is self-contained (it bundles `prepared.json.gz` and `custom.py`).
See [Serialization & inspection](narrative/bp-train/06_serialization_inspect.md).

### 9. Opt-in multi-core device pooling

Training can shard the process batch across CPU cores via `pmap` (~N speedup). This is
**opt-in** and resolved *before* JAX initializes (the device count is fixed at import
time): set `train.devices: N` (or `"max"`) in the config, or `BP_TRAIN_DEVICES=N` (the env
var always wins). `"max"` resolves to `min(n_processes, n_cpus)`. Default is 1 device, so
bp-train never competes for cores with other work. No effect on GPU.
