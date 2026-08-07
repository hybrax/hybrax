# Reaction & Loss Modules

Source: [`bp_train/model_api.py`](../bp_train/model_api.py),
[`bp_train/defaults.py`](../bp_train/defaults.py),
[`bp_train/bounds_loss.py`](../bp_train/bounds_loss.py),
[`bp_train/wrapper.py`](../bp_train/wrapper.py),
[`bp_train/dense.py`](../bp_train/dense.py)

## Purpose

The two user-pluggable halves of a hybrid model. The **reaction module**
(`UserReactionModule`) predicts the biological rates that drive the ODE; the
**loss module** (`UserLossModule`) maps the resulting trajectory to a dict of
named scalar losses. You supply each via a `custom.py` hook, or take the default.

## Design Rationale

- Both are `eqx.Module`s, partitioned into trainable/static by field tags — one
  mechanism for both (see
  [01_design_rationale.md](01_design_rationale.md#5-trainable-partition-via-field-tags)).
- The reaction module is the **single source of truth for every `SCALE_*`
  axis**; the loss module reads scales through `inputs.reaction_module`.
- A sample is **solved once**; the reaction module runs inside the solve, the
  loss module reads its saved outputs after.

---

## The reaction module

### The `build_reaction_module` hook

Define this in `custom.py` to supply a reaction module; omit it for the default
MLP.

```python
def build_reaction_module(*, target_names, process_names, config, seed,
                          collection, **scale_kwargs):
    return MyReactionModule(key=jax.random.key(seed), **scale_kwargs)
```

`scale_kwargs` carries the promoted `SCALE_*` scalers from
[`estimate_all_scales`](02_cli_and_config.md#estimate_all_scales); pass them to
`super().__init__(**scale_kwargs)`. Bare hook arrays become `LinearScaler`;
`AffineScaler(scale, offset)` opts one value axis into affine scaling. The hook
is discovered the same way as the
loss hook ([`get_hook`](02_cli_and_config.md#custompy-hooks-reference), falling
back to `default_build_reaction_module`).

### `UserReactionModule`

Override `__call__`; everything else is inherited.

```python
def __call__(self, t: jax.Array, inputs: ReactionInputs) -> ReactionOutputs
```

- Inputs arrive in SCL space; return rates in SCL space. Use `self.unscale_*`
  when you need RAW physical **values** (chemistry / FBA / kinetic laws), then
  `self.scale_*` on the way back out. All helpers except the three `*_rates`
  pairs have value semantics, so an affine value transform subtracts/adds the
  offset. The rate pairs use offset-free derivative semantics
  (`d((RAW-b)/s)/dt = dRAW/dt/s`); call the scaler's explicit
  `scale_derivative` / `unscale_derivative` for any other derivative. In
  particular, `ReactionOutputs.SCL_latent_derivative` is a derivative, not a
  value.
- The 12 stored `SCALE_*` fields (including `SCALE_latent`) are inherited frozen
  `Scaler` fields; **do not redeclare** them — pass values to
  `super().__init__()`. Rate-axis scalers cannot have non-zero offsets. Axis
  dimensions are available as properties (`n_modeled_RMCs`, `n_modeled_PVs`,
  `n_modeled_BiologicalOde_rates`, `n_modeled_FVCs`, `n_controlled_FVCs`,
  `n_controlled_PVs`, `n_latent`) so you size MLPs after `super().__init__()`
  without threading `n_*` kwargs.

Three further methods exist for stateful modules; the defaults are fine for a
stateless one:

| Member | Default | Override when |
|---|---|---|
| `initial_latent(RAW_phys_y0)` | zeros of length `n_latent` | the latent state should be seeded from the initial physical state |
| `latent_observables` | `()` | some observable needs live latent state *during* the solve |
| `observe(states)` | identity | the physical state needs a post-hoc observation map |

### `ReactionInputs`

Built by the wrapper at each RHS evaluation; all in SCL space. Read only the
axes you need — unused fields cost nothing under JIT.

| Field | Shape | Meaning |
|---|---|---|
| `SCL_modeled_RMCs` | `(n_RMC,)` | species concentrations (UNCLIPPED — gradient flow survives negative excursions) |
| `SCL_modeled_PVs` | `(n_modeled_PV,)` | modeled (dynamic) process-variable states |
| `SCL_modeled_V` | scalar | real reactor volume at `t` (incl. `min_V` floor) |
| `SCL_modeled_FVCs_cumulative` | `(n_modeled_FVC,)` | per-modeled-feed cumulative volume |
| `SCL_controlled_FVCs_cumulative` | `(n_ctrl_FVC,)` | per-controlled-feed cumulative volume |
| `SCL_controlled_FVCs_rates` | `(n_ctrl_FVC,)` | per-controlled-feed flow rate |
| `SCL_controlled_FVCs_Cin` | `(n_ctrl_FVC, n_RMC)` | controlled-feed composition |
| `SCL_controlled_PVs` | `(n_ctrl_PV,)` | controlled PV signals (pH, DO, T, …) |
| `SCL_modeled_FVCs_Cin` | `(n_modeled_FVC, n_RMC)` | modeled-feed composition |
| `SCL_latent` | `(n_latent,)` | integrated latent state (empty for stateless modules) |

### `ReactionOutputs`

| Field | Shape | Meaning |
|---|---|---|
| `SCL_modeled_BiologicalOde_rates` | `(n_rates,)` | rates aligned with `rhs_ode.name_modeled_rates`; **not** 1:1 with RMCs (algebraic rates like `q_X_active` live here) |
| `SCL_modeled_FVCs_rates` | `(n_modeled_FVC,)` | modeled-feed flow rates; **must be ≥ 0** — apply your own positivity transform (e.g. softplus) before scaling |
| `SCL_latent_derivative` | `(n_latent,)` | derivative aligned with `SCL_latent` (empty for stateless modules) |
| `auxiliary` | `dict[str, array] \| None` | optional model-defined observables saved at solver times (see [Using auxiliary](#using-auxiliary)) |

### `DefaultReactionModule`

An `eqx.nn.MLP` over `[SCL_modeled_RMCs | SCL_modeled_PVs]` →
`SCL_modeled_BiologicalOde_rates` (includes any `r_<pv>` PV rates). It ignores
controls and emits zero-valued modeled-feed rates. The defaults remain two hidden
layers and width `max(8, 2 * max(n_inputs, n_outputs))`; pass `depth` and
`width_size` to override them. Networks through three hidden layers use tanh with
Glorot-uniform weights; deeper networks use SiLU with He-uniform weights. Biases
are zero, and the output weights use `0.01 × Glorot uniform` to start the ODE
rates near zero. The single trainable leaf is the MLP
(`model: eqx.nn.MLP = trainable_field()`).

### `DefaultStatefulReactionModule`

Import it from `bp_train.defaults` — unlike `DefaultReactionModule` it is not
re-exported at the package top level. Using any module with `n_latent > 0` also
requires `train.allow_stateful_models: true` in the config; otherwise training
fails fast.

`DefaultStatefulReactionModule` is a latent ODE with a standard GRU:
`dh/dt = GRUCell(x, h) - h`. Its cell input `x` contains the scaled physical
and control inputs only; the latent state is passed only as the GRU hidden
argument. The reset and keep gates use sigmoid and the candidate uses tanh.

Each reset/keep/candidate input-kernel block starts from independent
Glorot-uniform weights, and each recurrent block from independent orthogonal
weights. The GRU's trainable `bias` and `bias_n` both start at zero. Its signed
biological-rate head uses `0.01 × Glorot` weights and a zero bias, giving a
near-zero initial output. When modeled feeds exist, their Softplus head also
uses `0.01 × Glorot` weights, but its trainable bias is calibrated so a zero
readout emits `0.01` in SCL derivative units. That nonzero feed-head bias is
separate from, and does not change, the zero initialization of internal GRU
biases.

### Field tagging

Declare trainable leaves with [`trainable_field()`](../bp_train/model_api.py),
frozen leaves with `frozen_field()`; untagged array leaves default to frozen.
[`partition_trainable(module)`](../bp_train/model_api.py) splits the module into
`(trainable, static)`. The harness partitions the whole wrapper this way, so
reaction- and loss-module trainable leaves are optimized together. Details in
[01_design_rationale.md](01_design_rationale.md#5-trainable-partition-via-field-tags).

### The `HybridOdeWrapper` bridge

[`HybridOdeWrapper`](../bp_train/wrapper.py) is the `eqx.Module` that joins your
reaction module, the controls store, the `SCALE_*` axes, and the bp-format
`RhsOde`. Its `__call__` is the ODE RHS: it assembles `ReactionInputs`, calls the
reaction module, unscales the rates, and feeds the physical mass balance;
`physical_save_outputs` produces the states/rates the loss module reads.
`validate_rhs_ode_compatibility` checks the reaction module's axes against the
`RhsOde` at construction. You rarely build it directly — the harness does.

---

## The loss module

bp-train computes the training loss through a user-defined `UserLossModule`, the
loss-side twin of `UserReactionModule`. You write one class that maps a
`LossInputs` bundle to a dict of **named scalar losses**; the harness averages
them for backprop, names every plot/log panel by the dict keys, and optimizes
any `trainable_field()` you declare — all from the single shared ODE solve. There
is no separate "default loss" callback: the default *is* a `UserLossModule`
(`DefaultLossModule`), and you subclass or replace it.

### The `build_loss_module` hook

Define this in your `custom.py` to supply a loss module; omit it to get the
default per-target MSE.

```python
def build_loss_module(*, target_names, process_names, config, seed, collection):
    return MyLossModule(...)
```

- `target_names` — the loss target-column labels: measured species followed by
  cumulative modeled-feed columns (`B_<feed>_cum`). These name the columns of
  `LossInputs.SCL_target_pred`, so a per-target module emits one term per label.
- `process_names`, `config` (your `CONFIG` dict), `seed`, `collection` — same as
  `build_reaction_module`.

The hook is discovered the same way as `build_reaction_module`
([`get_hook`](02_cli_and_config.md#custompy-hooks-reference), falling back to
`DefaultLossModule`). It must return a `UserLossModule` instance.

### `LossInputs` — what your `__call__` receives

One per sample, evaluated on the measurement-time grid. Predicted trajectories
come in both SCL and RAW space (scaling is a cheap elementwise broadcast); pick
whichever you need.

| Field | Shape | Meaning |
|---|---|---|
| `SCL_states` / `RAW_states` | `(n_meas, n_state)` | integrated state, scaled / physical |
| `SCL_modeled_BiologicalOde_rates` / `RAW_…` | `(n_meas, n_rates)` | reaction rates over time |
| `SCL_modeled_FVCs_rates` / `RAW_…` | `(n_meas, n_modeled_FVCs)` | modeled-feed flow rates |
| `SCL_V` / `RAW_V` | `(n_meas,)` | reactor volume |
| `auxiliary` | `dict[str, (n_meas, …)]` | model-defined observables (see below); empty dict if none |
| `SCL_target_pred` | `(n_meas, n_target)` | predicted target columns (`SCL_states[:, target_state_indices]`) |
| `SCL_target_measured` | `(n_meas, n_target)` | ground-truth measurements, SCL-scaled |
| `mask_measured` | `(n_meas, n_target)` | **per-cell** validity (see below) |
| `mask_measured_any` | `(n_meas,)` | **per-row** validity, float — `any(mask_measured, axis=1)` |
| `t_measured` | `(n_meas,)` | measurement times |
| `n_measured` | scalar | unpadded row count |
| `reaction_module` | — | the `UserReactionModule` (single source of `SCALE_*`) |
| `step` | scalar | training step (−1 in forward eval) |
| `jump_ts` | `(n_jump_ts,)` or `None` | genuine vector-field discontinuity times (`controls.active_jump_ts`, sourced from `BioProcess.discrete_events`); use to mask dense points / triples near jumps |

**Dense-grid view** — populated iff the loss module declares
`dense_grid_n: int` (see [Dense-grid losses](#dense-grid-losses)); otherwise all
dense fields are `None`. Same dtypes and column layout as the measurement-grid
fields above, leading dim `n_dense`:

| Field | Shape | Mirror of |
|---|---|---|
| `dense_t` | `(n_dense,)` | `t_measured` (but evenly spaced inside `[t_start, t_end]`) |
| `dense_SCL_states` / `dense_RAW_states` | `(n_dense, n_state)` | `SCL_states` / `RAW_states` |
| `dense_SCL_modeled_BiologicalOde_rates` / `dense_RAW_…` | `(n_dense, n_rates)` | rates pair |
| `dense_SCL_modeled_FVCs_rates` / `dense_RAW_…` | `(n_dense, n_modeled_FVCs)` | feed-rates pair |
| `dense_SCL_V` / `dense_RAW_V` | `(n_dense,)` | volume pair |
| `dense_auxiliary` | `dict[str, (n_dense, …)]` | `auxiliary` |
| `dense_valid_time` | `(n_dense,)` bool | which dense rows are real predictions — see below |

#### Failed solves

If the ODE solve bailed partway (a stiff segment hit the step cap), every point
past the failure time is dropped **before** `LossInputs` is built:

- On the measurement grid, `mask_measured` / `mask_measured_any` are already
  zeroed on those rows.
- On the dense grid, `dense_valid_time` marks the still-valid rows. It is
  all-`True` when the solve succeeded, and `None` when the dense grid is off.

Every predicted trajectory here — measurement and dense — is guaranteed finite:
post-failure rows carry a finite fallback, never `inf`/`nan`. So the
`penalty * mask` idiom is safe (no `0 * inf`). **Gate dense penalties by
`dense_valid_time`** the way measurement terms use `mask_measured`; the dense
grid has no other failure mask.

#### Masks (sparse, unaligned measurements)

When species are sampled on different time grids, the data is built on the
*union* grid and padded per batch. Two masks express what is real:

- `mask_measured[t, j]` is True iff target `j` has a real measurement at time
  `t`. It is False for padding rows **and** for targets not sampled at that
  time. Always zero out / skip masked-off cells in a loss term.
- `mask_measured_any[t]` is True iff *any* target is real at time `t`. Multiply
  trajectory-wide penalties (e.g. a bounds hinge that applies at every real
  timestep regardless of which species was sampled) by this float mask.

Scales live only on `reaction_module` — read them via
`inputs.reaction_module.SCALE_*` or its `scale_*` / `unscale_*` helpers. They
are never duplicated onto `LossInputs`.

### `LossOutputs` and aggregation

```python
return LossOutputs(named_losses={"biomass": ..., "glucose": ..., "lwr_bnd/q_glc": ...})
```

- The **total** loss for backprop is `mean(named_losses.values())`.
- Why mean and not sum: bp-train clips the **raw** gradient
  (`clip_by_global_norm`) *before* Adam. Mean keeps the gradient magnitude in
  the same range regardless of how many named terms you add, so a tuned
  `grad_clip_norm` keeps behaving the same — the clip stays dormant in normal
  training. Sum would scale the gradient by the term count, push it past the
  clip threshold, and (because the clip sits before Adam) hold the step size
  large near the optimum → overshoot / divergence on stiff neural-ODE problems.
  If you genuinely want sum-style weighting, scale the individual terms inside
  `__call__` and retune `grad_clip_norm` accordingly.

### `loss_names` — required property

Declare the term names up front; they must equal the keys your `__call__`
returns, in order:

```python
@property
def loss_names(self) -> tuple[str, ...]:
    return ("biomass", "glucose", "lwr_bnd/q_glc")
```

The harness reads `loss_names` once at setup to build the console table,
`metrics.csv`, and `loss_curve.png` panels, and to size/validate the loss
vector. A mismatch between `loss_names` and the returned keys is a fail-fast
error.

### Choosing the loss type (MAE / MSE / Huber)

`DefaultLossModule.residual_reduction(residual, mask)` is the per-target
reduction (default: masked mean-squared error). Override it to switch:

```python
class MAELossModule(DefaultLossModule):
    def residual_reduction(self, residual, mask):
        masked = jnp.where(mask, jnp.abs(residual), 0.0)
        n_active = jnp.maximum(jnp.sum(mask, axis=0), 1)
        return jnp.sum(masked, axis=0) / n_active
```

### Adding bounds penalties

`BoundsViolationLossModule` retains the default measurement MSE and adds one
squared-hinge term for each finite bp-format bound on a modeled reactor-medium
component, modeled process variable, reactor volume, or `BiologicalOde` rate:

```python
from bp_train import BoundsViolationLossModule


def build_loss_module(*, target_names, collection, config, **_):
    return BoundsViolationLossModule(
        target_names=target_names,
        collection=collection,
        weight=config.custom.bounds_weight,
    )
```

Bounds default to `(None, None)`, so constructing this module does not invent
physical constraints. Bounds for the same quantity must agree across all
processes in the collection. Violations are computed in RAW physical space,
then normalized with the corresponding offset-free derivative scale. Reactor-
volume bounds use the integrated volume before the reaction model's `min_V`
safety floor. This is safe for both linear and affine value scalers. `weight`
multiplies every bounds term and must be finite and nonnegative. At zero, the
bound terms remain in the stable loss schema and evaluate to zero; because total
loss averages all named
terms, select `DefaultLossModule` instead when no bounds terms should exist.

The penalties use the existing measurement grid and `mask_measured_any`; the
module does not request a denser solve. Each finite side is reported separately
as `lwr_bnd/<state>`, `upr_bnd/<state>`, `lwr_bnd/rate/<rate>`, or
`upr_bnd/rate/<rate>`. If a modeled state is named `V`, reactor-volume terms use
the `volume/V` segment. Reconstruction targets share the loss-name namespace;
any remaining actual collision is rejected during module construction. See
[the TUB migration hook](../examples/11_tub_2026/migration/custom.py).

### Adding other custom loss terms

Subclass a built-in loss module and add named entries; they show up
automatically as new panels and log columns. `loss_names` must include the base
module's names and every added term.

### Trainable loss parameters

Declare `trainable_field()` leaves and they are optimized alongside the reaction
module (the harness partitions the whole wrapper by field tags). Example —
Kendall uncertainty weighting:

```python
class KendallLossModule(DefaultLossModule):
    log_sigma: jax.Array = trainable_field()

    def __init__(self, *, target_names):
        super().__init__(target_names=target_names)
        self.log_sigma = jnp.zeros(len(target_names))

    def __call__(self, inputs):
        base = super().__call__(inputs).named_losses
        weighted = {
            name: jnp.exp(-self.log_sigma[i]) * base[name] + self.log_sigma[i]
            for i, name in enumerate(self.target_names)
        }
        return LossOutputs(named_losses=weighted)
```

At training start the harness prints two structure tables —
`UserReactionModule` and `UserLossModule` — so you can verify exactly which
leaves are trainable vs frozen (see
[06_serialization_inspect.md](06_serialization_inspect.md#introspection)).
Trainability is declared solely through field tags; there is no custom
`partition_trainable()` override. For advanced sub-field control (e.g. freezing
some MLP layers), use the
[`build_optimizer`](02_cli_and_config.md#build_optimizer) hook with
`optax.masked` / `optax.multi_transform` rather than a second partition
mechanism.

### Using `auxiliary`

Emit observables from the reaction module:

```python
return ReactionOutputs(
    SCL_modeled_BiologicalOde_rates=...,
    SCL_modeled_FVCs_rates=...,
    auxiliary={"q_glucose_signed": some_scalar},
)
```

They are saved at every measurement time and arrive stacked in
`inputs.auxiliary["q_glucose_signed"]` (shape `(n_meas,)`), ready to drive a
loss term.

### Dense-grid losses

Some loss terms need values *between* measurement points — e.g. a smoothness
penalty on rate time-derivatives (finite differences need >3 well-spaced
points), or a bounds hinge that should hold everywhere, not only when the
plate was sampled. Opt in by declaring a `dense_grid_n` property:

```python
class CurvatureLossModule(DefaultLossModule):
    @property
    def dense_grid_n(self):
        return 32  # any int N > 0
```

When set, the trainer solves **once** on `union(t_measured, linspace(t_start,
t_end, N))` — the same single ODE solve, just with more `SaveAt` points (no
extra solver steps) — and populates the `dense_*` fields on `LossInputs`
alongside the existing measurement-grid fields. `jump_ts` is populated
unconditionally so any loss (measurement or dense) can locate controls
discontinuities.

Three helpers in `bp_train` cover the typical needs (lifted from the structured
example, exported for reuse):

```python
from bp_train import (
    build_union_time_grid,
    dense_point_mask_away_from_jumps,
    dense_triple_mask_away_from_jumps,
)
```

- `build_union_time_grid` — the same routine the trainer uses internally;
  exposed in case a downstream tool wants the index mapping.
- `dense_point_mask_away_from_jumps(dense_t, jump_ts, eps)` — per-point
  mask; rejects dense points within `eps` of any jump.
- `dense_triple_mask_away_from_jumps(dense_t, jump_ts, eps)` — per-triple
  mask `(i-1, i, i+1)`; rejects triples whose span crosses a jump. Use this
  for finite-difference curvature so the second derivative is never measured
  across a discontinuity.

#### Example 1 — rate-curvature penalty

```python
class CurvatureLossModule(DefaultLossModule):
    """DefaultLossModule + second-derivative penalty on selected SCL rates."""

    rate_indices: tuple = eqx.field(static=True)
    weight: float = eqx.field(static=True)
    jump_eps_h: float = eqx.field(static=True)

    @property
    def dense_grid_n(self):
        return 32  # tune to your dynamics; see practical note below

    @property
    def loss_names(self):
        return tuple(self.target_names) + tuple(
            f"curvature/{i}" for i in self.rate_indices
        )

    def __call__(self, inputs):
        base = super().__call__(inputs).named_losses
        rates = inputs.dense_SCL_modeled_BiologicalOde_rates       # (n_dense, n_rates)
        t = inputs.dense_t                                          # (n_dense,)
        triple = dense_triple_mask_away_from_jumps(
            t, inputs.jump_ts, self.jump_eps_h
        ).astype(rates.dtype)                                       # (n_dense - 2,)
        dt = jnp.maximum(t[1:] - t[:-1], 1e-6)
        slopes = (rates[1:] - rates[:-1]) / dt[:, None]
        mid_dt = jnp.maximum(0.5 * (dt[1:] + dt[:-1]), 1e-6)
        curv = (slopes[1:] - slopes[:-1]) / mid_dt[:, None]         # (n_dense - 2, n_rates)
        denom = jnp.maximum(jnp.sum(triple), 1.0)
        extras = {
            f"curvature/{i}": self.weight * (jnp.sum(jnp.square(curv[:, i]) * triple) / denom)
            for i in self.rate_indices
        }
        return LossOutputs(named_losses={**base, **extras})
```

The same module can compose metadata-driven `lwr_bnd/<state>` measurement
terms with the curvature penalty.

#### Example 2 — between-measurement bounds

`BoundsViolationLossModule` intentionally checks only measurement times. The
RMC/PV example below keeps the same RAW-first, offset-free normalization and
also masks points after a solve failure. `dense_RAW_states` contains the floored
volume, so a dense physical-volume bound would first need an unclamped dense-
volume input.

```python
# inside __call__ of a BoundsViolationLossModule subclass with dense_grid_n set
values = inputs.dense_RAW_states[:, idx]  # modeled RMC/PV state
violation = jax.nn.relu(threshold_RAW - values)  # lower-bound example
scaler = inputs.reaction_module.SCALE_state[idx]
normalized = scaler.scale_derivative(violation)
mask = dense_point_mask_away_from_jumps(
    inputs.dense_t, inputs.jump_ts, jump_eps_h
) & inputs.dense_valid_time
dense_penalty = self.weight * (
    jnp.sum(jnp.square(normalized) * mask)
    / jnp.maximum(jnp.sum(mask), 1)
)
```

#### Practical note on `dense_grid_n` size

**A dense time is not a segment boundary.** `physical_solve` splits segments only
at bolus/sample events and reads the output grid off `SaveAt(ts=…)` *inside* each
segment — interpolation, costing no solver steps. So a finer dense grid does not
subdivide the integration, does not move `fail_time`, and does not change which
samples bail.

(This was not always true. When output times *were* segment boundaries, one
240 h / 11-measurement / 10-sample-event process went from 10 segments and 38 ODE
steps at `dense_grid_n=None` to 108 segments and 229 steps at `dense_grid_n=100`.
See [specs/two_tier_integration_grid.md](../specs/two_tier_integration_grid.md).)

It is still not free: each dense point costs one interpolant evaluation per
segment it lands in, and the `dense_*` arrays it populates are real arrays inside
the loss. Pick the smallest N your finite differences actually need.

### Where terms show up

Every named term flows, by its key, to:

- the per-step console table,
- `metrics.csv` (the per-step loss history in the run directory),
- the checkpoint `loss_curve.png` (one panel per term, plus a `total` panel),
- the per-process fit plot: each species/feed subplot is annotated with its
  named term (when one matches by name) plus R²; the process's total loss is in
  the figure title, and non-species terms (penalties, aux) are listed there.
  Terms with no matching subplot simply don't annotate one — never an error.
