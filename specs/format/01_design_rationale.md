# Design Rationale

Why bp-format looks the way it does. The other pages describe *what* the code
does; this one explains the choices behind it.

## 1. JAX-first, but only where it matters

The numerical objects in bp-format are built on
[JAX](https://github.com/google/jax) and
[Equinox](https://github.com/patrick-kidger/equinox) so that downstream training
code can differentiate straight through them:

- **Autodiff.** Fitting a hybrid model means taking gradients through ODE
  integration and spline evaluation. `jax.grad` handles that end-to-end.
- **JIT.** `eqx.filter_jit` compiles the repeated forward solves in a training
  loop down to XLA.
- **Vectorization.** `jax.vmap` batches over processes or parameter sets
  without hand-written loops.

`eqx.Module` is a frozen dataclass that JAX already knows how to flatten into a
pytree, so a `TimeSeries`, `ControlSplines`, or `RhsOde` can cross a JIT
boundary untouched.

**Only the numerical leaves are Equinox modules.** `TimeSeries`, `PPoly`,
`ControlSplines`, and `RhsOde` are `eqx.Module`.
Everything else — `BioProcess`, `BioProcessCollection`, `ReactorMedium`, … —
is a plain `@dataclass`. Those containers hold `Dict[str, ...]` fields that
are edited by name outside any JIT boundary; making them pytrees would buy
nothing and cost mutability.

**Consequences to know about:**

- Array fields must be `jnp.ndarray`, not NumPy.
- `eqx.Module` instances are immutable. Use `dataclasses.replace` or
  `eqx.tree_at` for functional updates.
- Plain dataclasses (`BioProcess` and friends) *are* mutable — pipeline steps
  assign fields like `process.biological_ode = ...` in place.

**float64 everywhere.** Importing `bp_format` sets `JAX_ENABLE_X64=true` before
JAX loads. Spline fitting and differentiation, and the mechanistic RHS's ODE
integration, both compound floating-point error over many steps; float32 loses
too much there. A `TimeSeries` constructed from float32 arrays raises rather
than silently upcasting, so precision loss cannot enter through the data.

**Importing is cheap.** `bp_format/__init__.py` resolves its exports lazily via
`__getattr__`, so `import bp_format` does not pull in JAX, sympy, or matplotlib
until you touch something that needs them.

## 2. Two levels: collection → run → components

```
BioProcessCollection  one collection of processes, one file on disk
  └─ BioProcess       one experimental run
       └─ components  reactor medium, volume, process variables
```

- **`BioProcessCollection`** carries optional `case_id`, `organism`, and
  `citation` fields alongside a free-form `metadata` dict. Set (all
  non-empty), they mark the collection as a full, publication-linked case
  study — the natural unit for leave-one-process-out cross-validation. Left
  `None` (the default), the collection is raw or intermediate data that is
  not yet a published case study.
- **`BioProcess`** is a single fermentation run and holds everything needed to
  simulate it: time axis, volume operations, reactor concentrations, process
  variables, and the biological ODE.
- **Components** are grouped by physical role: `ReactorMedium` for
  concentrations, `Volume` for anything that changes the working volume,
  `ProcessVariable` for everything else (pH, temperature, DO, off-gas).

**Why dicts keyed by name?** O(1) lookup, readable JSON keys, and obvious
iteration. Lists would mean linear search and lose the naming.

## 3. Volume is its own category

Volume is not a state, not a control, and not a process variable:

- **Many operations move it.** One run can have continuous feeds, bolus feeds,
  and sampling, each on its own schedule.
- **It enters the ODE differently.** Every flow dilutes every reactor species by
  `-(inflow + outflow)/V · c`, while feeds additionally *add* mass at `q·Cin/V`.
  Volume itself follows `dV/dt = inflow − outflow`.
- **It carries chemistry.** A `Inflow` references a `FeedMedium` that
  says what enters. A `Outflow` removes broth at current
  concentrations, so it needs no medium.

Signs are enforced by type: feeds store non-negative values, samples store
non-positive ones. Values are stored as **cumulative volumes, never rates** —
rates are a derived quantity (the spline derivative), and storing the primary
measurement avoids baking one differentiation choice into the data.

## 4. TimeSeries carries samples, a spline, or both

`TimeSeries` holds `times`/`values` from the experiment and, optionally, fitted
spline state (`breaks`, `coeffs`, `segment_start_piece_idx`) giving a
continuous-time version of the same signal. At least one of the two must be
present.

- Raw samples are ground truth for loss computation and validation.
- The spline lets an ODE solver evaluate a signal at arbitrary times without
  re-interpolating at every step.
- A spline-only series is legitimate: `make_constant_spline` and
  `make_cubic_ppoly` build spline state directly from known values or arrays,
  with no underlying "measurements".

**`TimeSeries` does not know if its data is continuous.** That comes from the
parent. A `VolumeChange` with `is_continuous=True` means the series is a
continuous flow profile; `is_continuous=False` means the same series holds
discrete bolus or sampling events, where fitting a spline would be meaningless.

**Why power-basis coefficients, not B-splines?** A piece is stored as
`[a, b, c, d]` and evaluated as `a + h(b + h(c + h·d))` with `h = t − t_break` —
a few fused multiply-adds that map directly onto JAX. B-spline evaluation needs
recursive knot-vector lookups that vectorize poorly.

## 5. Check the data, then fail loudly

Bioprocess data arrives from many labs with many conventions. The recurring
problems are always the same: sign confusion on feeds, a missing biomass
component, mismatched array lengths, a feed medium that forgets a species,
measurement timestamps nudged past a sampling event.

bp-format uses **two** mechanisms, deliberately:

**Validators return `(bool, str)`.** Everything in `bp_format/validate.py` reports
rather than raises, so one pass collects every problem into a readable report
instead of stopping at the first. `validate_process()` aggregates the
per-process checks; `validate_cross_process_consistency()` adds cross-process
consistency.

**Constructors and builders raise.** Anything that would produce silently wrong
numbers fails immediately: a `TimeSeries` with unsorted times, a feed medium
naming a species that is not in the reactor, a name used by both a state and a
rate, a cyclic algebraic definition. Inside JIT, `eqx.error_if` guards the same
invariants at runtime — a reactor volume at or below `1e-10` aborts the solve
instead of dividing by nearly zero.

The rule of thumb: *reporting* is for data quality you may knowingly accept;
*raising* is for states from which no correct answer exists.

## See also

- [Data Model](02_data_model.md)
- [Validation](04_validation.md)
- [Splines](07_splines.md)
