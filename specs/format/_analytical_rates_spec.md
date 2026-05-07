# Analytical Rate Inversion from `UserDefinedRhsOde`

Specification for [`build_rates_func_analytical`](../bp_format/mechanistic.py#L1538) — the
sympy-driven, JIT-compilable rate-inversion path that operates on a
[`UserDefinedRhsOde`](../bp_format/mechanistic.py#L567) (i.e. on a `BioProcess`
whose [`biological_ode`](../bp_format/dataclasses.py#L248) block is set).

This is the user-defined-RHS counterpart to the legacy
[`build_q_func`](../bp_format/mechanistic.py#L1296) /
[`build_rates_func`](../bp_format/mechanistic.py#L1811) inversion that operates
on an auto-generated [`RhsOde`](../bp_format/mechanistic.py).

## 1. Goal

Given

- a process with a `biological_ode` block declaring `algebraic`, `rates`,
  and per-state biological derivative expressions, and
- splines that reconstruct each non-volume state trajectory from data,

produce a callable

```python
rates_func(t) -> jnp.ndarray  # shape (rate_size,)
```

that returns the values of the user-declared rate symbols at time `t`,
ordered as in `mb.rate_names`.

The mapping is purely *inversion*: spline-implied biological `dc/dt` →
algebraic solve for the rate vector.

## 2. Inputs

| arg | type | role |
|---|---|---|
| `process` | `BioProcess` | must have `biological_ode is not None` |
| `ctrl` | `ControlSplines` | evaluates controlled-PV symbols at `t` |
| `mb` | `UserDefinedRhsOde` | provides ordering of states / rates / algebraic / controlled PVs |
| `state_splines` | `Dict[str, spline]` (optional) | reactor entries should be `BacktransformSpline`; built via [`build_state_splines`](../bp_format/mechanistic.py#L1200) when omitted |

Output: `Callable[[float], jnp.ndarray]` of shape `(mb.rate_size,)`.

## 3. Mathematical model

### 3.1 Per-state biological RHS

For each non-volume state `s` (reactor-component or process-variable), the
user supplies a biological-RHS expression

```
f_s(c_state, u_ctrl, d(c_state, u_ctrl, rates), rates)
```

over the symbol table

```
{state names} ∪ {controlled-PV names} ∪ {algebraic names} ∪ {rate names}
```

where `d` is the vector of `algebraic` quantities (acyclic, recomputed every
call). After topo-sorting `algebraic` and inlining every entry into the
state derivatives, each per-state expression depends only on
`(c_state, u_ctrl, rates)`:

```
f_s(c_state, u_ctrl, rates)
```

### 3.2 Linearity requirement

`build_rates_func_analytical` requires that every per-state expression be
**linear** in the declared rate symbols. After inlining `algebraic`, each
must factor as

```
f_s(c_state, u_ctrl, rates) =
    Σ_k A_{s,k}(c_state, u_ctrl) · rate_k  +  b0_s(c_state, u_ctrl)
```

This is verified symbolically:

```
A_row = [sympy.diff(expr, rate_k) for rate_k in rate_syms]
b0    = expr.subs({rate_k: 0 for rate_k in rate_syms})
residual = sympy.expand(expr - Σ_k A_row[k]·rate_k - b0)
assert residual == 0   # else: ValueError("... non-linear in rates ...")
```

Non-linearity (rate products, rate-rate divisions, transcendental functions
of rates, etc.) raises `ValueError` at build time.

### 3.3 Biological `dc/dt` observation from splines

The inversion is performed in **pseudo-batch (`c*`) coordinates** for reactor
components — the `c*` transform absorbs feed/dilution exactly, so the
biological `dc/dt` is recovered without recomputing `Cin`, `V`, `u_flow`,
or sample/feed terms inside the inversion:

- **reactor state with `BacktransformSpline`** (carries `c_star_spline`,
  `adf_times`, `adf_values`):

  ```
  dc/dt_biol(t) = (dc*/dt)(t) / adf(t)        # adf clamped to ≥1e-12
  ```

  where `adf(t) = jnp.interp(t, adf_times, adf_values)` is the accumulated
  dilution factor used during the original pseudo-batch transform.

- **reactor state without `c_star_spline`** (fallback): the spline is
  treated as already living in the biological coordinate, i.e.

  ```
  dc/dt_biol(t) = spline.derivative()(t)      # identity transform
  ```

  This is correct only when `adf ≡ 1` (no dilution); the function does not
  warn but the result is wrong otherwise.

- **process-variable state**: identity transform.

  ```
  dc/dt_biol(t) = spline.derivative()(t)
  ```

### 3.4 Linear inversion

Stack the per-state observations into `bio_dc ∈ R^{n_state}` and the
intercepts into `b0 ∈ R^{n_state}`. Define `b = bio_dc - b0`. Then

```
A · rates = b
```

Two solve paths:

1. **Diagonal fast path** — when there are `n_rates` rate symbols and each
   one appears in exactly one row (and each such row is unique), the system
   is permuted-diagonal and is solved per-row:

   ```
   rate_k = b[row(k)] / A[row(k), k]
   ```

   No `jnp.linalg.solve` call.

2. **Square dense solve** — otherwise the inversion is restricted to the
   set of rows that constrain at least one rate (`nonzero_rows`). The
   system must be square: `len(nonzero_rows) == n_rates`. Solve

   ```
   rates = jnp.linalg.solve(A[nonzero_rows, :], b[nonzero_rows])
   ```

   If `len(nonzero_rows) != n_rates` and the diagonal fast path was not
   matched, build raises:

   ```
   ValueError(f"biological_ode has {n_rates} rate symbols but
               {len(nonzero_rows)} states constrain rates; analytical
               inversion requires a square system.")
   ```

### 3.5 Edge case: zero rates

If `mb.rate_size == 0` the runtime returns `jnp.zeros(0)` without
attempting any solve.

## 4. Build-time pipeline (executed once)

1. Validate inputs: `process.biological_ode is not None`,
   `isinstance(mb, UserDefinedRhsOde)`. Build `state_splines` if missing.
2. Collect orderings: `state_names = reactor + pv`, `ctrl_pv_names`,
   `rate_names`, `name_modeled_algebraic`. These are static under JIT.
3. Build a sympy symbol table covering states, controlled PVs, algebraic
   names, and rates. `sympify` every `algebraic` and `derivatives` expression
   under that table.
4. Topo-sort `algebraic` ([`_topo_sort_algebraic`](../bp_format/mechanistic.py#L545));
   inline each algebraic expression into all later expressions and into every
   per-state derivative; `sympy.expand` the result.
5. For each per-state expression, compute the row of the rate Jacobian
   `A_row` and the rate-zero intercept `b0`; check linearity (Section 3.2).
6. Lambdify each `A[i,k]` and each `b0[i]` over the flat argument vector

   ```
   args = concat(state_values, ctrl_pv_values)
   ```

   using [`_lambdify_with_array_arg`](../bp_format/mechanistic.py#L524) with
   `modules="jax"`. The wrapper accepts a single `args` tensor (positional
   `*args` is unsafe under `jax.jit`).
7. Decide diagonal vs. dense solve and freeze the `nonzero_rows` index
   array. Pre-pull each reactor spline's `c_star_spline.derivative()` and
   `(adf_times, adf_values)` arrays; pre-pull each PV spline derivative.
8. Cache `ctrl.ctrl_indices` as a static index array for `ctrl(t)`.

## 5. Runtime pipeline (`rates_func(t)`)

Per-call, all loops over names/indices are Python-static (the lengths and
keys are known at build time); only `t` and the spline data are traced.

1. **State concentrations.** Evaluate each reactor spline and each PV
   spline at `t`; concatenate to `c_state ∈ R^{n_state}`.
2. **Biological `dc/dt`.** Per Section 3.3 — `c*`-domain derivative divided
   by `adf(t)` for reactor states with a `BacktransformSpline`, plain
   spline derivative otherwise.
3. **Controlled PVs.** `u = ctrl(t); ctrl_pv = u[ctrl_indices]`.
4. **Args & intercepts.** `args = concat(c_state, ctrl_pv); b0_vec[i] =
   b0_funcs[i](args); b = bio_dc - b0_vec`.
5. **Solve.** Diagonal fast path or `jnp.linalg.solve` over `nonzero_rows`,
   per Section 3.4.
6. **Return** `rates ∈ R^{rate_size}`, ordered as `mb.rate_names`.

## 6. Failure modes (build-time `ValueError`)

| condition | message |
|---|---|
| `process.biological_ode is None` | requires biological_ode to be set |
| `mb` not `UserDefinedRhsOde` | requires UserDefinedRhsOde from `get_rhs_ode` |
| derivative non-linear in rates | "biological_ode.derivatives[s] is non-linear in rates" |
| `n_rates != len(nonzero_rows)` and not diagonal | "analytical inversion requires a square system" |
| cyclic `algebraic` | from `_topo_sort_algebraic` |

## 7. Differences vs. the previous implementation

The previous inversion path is [`build_q_func`](../bp_format/mechanistic.py#L1296)
plus its wrapper [`build_rates_func`](../bp_format/mechanistic.py#L1811).
That path is still in the codebase and still operates on the
auto-generated `RhsOde` (the `q · X_active + r + feed` form). The
analytical path is a parallel inversion specialized for the user-defined
RHS branch.

| aspect | `build_q_func` / `build_rates_func` (auto path) | `build_rates_func_analytical` (user-defined path) |
|---|---|---|
| RHS form | hard-coded `dc/dt = q·X_active + r + feed_term`; biomass row corrected for intracellular accumulation via `_subtract_intracellular_from_biomass_q` | arbitrary user-supplied expressions, must be linear in declared rates; mass balance must be encoded in the expressions themselves |
| Rate-vector dimension | `q.shape == (n_reactor_states,)` — one specific rate per reactor component | `rates.shape == (rate_size,)` — set by `len(biological_ode.rates)`; independent of state count |
| Rate semantics | always *biomass-specific* | abstract user-declared symbols; no fixed biological meaning |
| State scope of inversion | reactor-component states only; PV states get `r_pv` from spline derivatives in `build_rates_func` | all non-volume states (reactor + PV); a rate that only appears in PV derivatives is still recoverable |
| Feed / dilution handling | reconstructs `V(t)` from `V0` + cumulative-volume splines + discrete events; subtracts `feed_term = Σ_k (f_k/V)·(C_in_k - c)` in real space | uses pseudo-batch coordinates: `dc/dt_biol = (dc*/dt) / adf(t)`; the `c*` transform absorbs feed, dilution, sampling, and `dV` exactly — no `V`, `Cin`, `u_flow` reconstruction inside the inversion |
| Volume / discrete-event plumbing | `extract_discrete_events`, `cum_splines_ctrl`, `cum_splines_mod`, `_batch_splines`, `ev_dV_cum`, `searchsorted` | none — handled by the `BacktransformSpline` upstream |
| `q` vs `r` partitioning | explicit `q_state_indices` / `r_state_indices` / `r_func` API with overlap-only-with-`r_func` rules | n/a — biological vs. physical split is encoded in the `biological_ode` block; everything outside that block is physical and added by bp-format on top |
| Linearity assumption | implicit (`q · X_active` is linear by construction) | verified symbolically (`sympy.diff` / `sympy.expand` residual check) and rejected at build time if violated |
| Solve | per-row scalar division: `q_i = (dc/dt - feed_term - r_overlap) / X_active`; biomass corrected | diagonal fast path (per-row scalar) or `jnp.linalg.solve(A, b)` over non-zero rows |
| Symbolic toolchain | none; pure numerical formulas | sympy: `sympify`, `_topo_sort_algebraic`, `diff`, `expand`, `subs`, `lambdify(modules="jax")` |
| Dependence on `ctrl` | only through downstream callers (the inversion itself ignores `ctrl`) | required: controlled-PV values are bound into `args` via `ctrl(t)[ctrl_indices]` |
| Reactor-spline contract | calls `state_splines[s](t)` and `state_splines[s].derivative()(t)` directly in real space | requires `BacktransformSpline` (with `c_star_spline`, `adf_times`, `adf_values`) for the `c*`-domain inversion; gracefully falls back to identity transform if absent (correct only when `adf ≡ 1`) |
| Output | wrapped by `build_rates_func` into `(q, r)` for the integration callback signature `rates_func(t, state, controls)` | returns rates only; signature is `rates_func(t)`. Integration of a `UserDefinedRhsOde` consumes this rate vector via the `is_user_defined` branch in [`_build_segment_rhs`](../bp_format/mechanistic.py#L1975) |
| Active-biomass handling | `X_active = c[biomass] - Σ c[intra]`, clamped to `1e-6` | not handled here; the user expresses `X_active` as an `algebraic` entry and writes it into the relevant per-state expressions explicitly |

### Inputs newly required

- A validated `biological_ode` block (`algebraic`, `rates`, `derivatives`)
  with every dynamic state present and every expression sympy-parseable
  over the typed symbol table.
- Reactor-component splines built as `BacktransformSpline` (the default
  output of `build_state_splines` when an interpolator with a `transform`
  metadata field is present).
- Linearity of every `derivatives[s]` in the declared rate symbols.
- A square inversion system: `n_rates` rate symbols matched by `n_rates`
  states whose derivatives depend on at least one rate.
- A `ControlSplines` instance whose `ctrl_indices` align with
  `mb.controlled_pv_names`.

### Inputs newly *not* required

- `q_state_indices` / `r_state_indices` / `r_func` partitioning.
- An `r_func`-style external physical-rate callback (physical contributions
  are added by bp-format from `VolumeChange` data outside this function).
- Reconstruction of `V(t)`, `Cin`, `u_flow`, or discrete-event volume
  jumps inside the inversion — the `c*`-coordinate input absorbs them.

## 8. JIT / autodiff notes

- All Python-level iteration is over static names/indices fixed at build
  time; only `t` and array data are traced.
- `_lambdify_with_array_arg` wraps `sympy.lambdify(..., modules="jax")` so
  the produced callables consume a single concatenated `args` tensor and
  are safe under `jax.jit` / `eqx.filter_jit`.
- `nonzero_rows` is materialized as a `jnp.array` once at build time and
  used as a static gather index at runtime.
- `adf` is clamped at `1e-12` to keep the division differentiable through
  zero-dilution edge cases.

## 9. Provenance

The function lands together with the `biological_ode` data-model block in
commit `c7419b4` ("add user-defined biological ODE block and fix
intracellular mass balance") and is exercised by the dual-path
equivalence tests against the auto-generated `RhsOde` inversion on the
martens fixtures (with and without intracellular product). It is the
inversion-side analogue of the `is_user_defined` integration branch in
[`_build_segment_rhs`](../bp_format/mechanistic.py#L1975).
