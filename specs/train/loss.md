# Custom loss: `UserLossModule`

bp-train computes the training loss through a user-defined `UserLossModule`,
the loss-side twin of `UserReactionModule`. You write one class that maps a
`LossInputs` bundle to a dict of **named scalar losses**; the harness sums them
for backprop, names every plot/log panel by the dict keys, and optimizes any
`trainable_field()` you declare on the module — all from the single shared ODE
solve. There is no separate "default loss" callback to wrestructure: the
default *is* a `UserLossModule` (`DefaultLossModule`), and you subclass or
replace it.

## The `build_loss_module` hook

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
(`get_hook`, falling back to `DefaultLossModule`). It must return a
`UserLossModule` instance.

## `LossInputs` — what your `__call__` receives

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
| `jump_ts` | `(n_step_ts,)` or `None` | controls-discontinuity times (`controls.active_step_ts`); use to mask dense points / triples near jumps |

**Dense-grid view** — populated iff the loss module declares
`dense_grid_n: int` (see [Dense-grid losses](#dense-grid-losses-rate-curvature-between-measurement-constraints)
below); otherwise all dense fields are `None`. Same dtypes and column layout as
the measurement-grid fields above, leading dim `n_dense`:

| Field | Shape | Mirror of |
|---|---|---|
| `dense_t` | `(n_dense,)` | `t_measured` (but evenly spaced inside `[t_start, t_end]`) |
| `dense_SCL_states` / `dense_RAW_states` | `(n_dense, n_state)` | `SCL_states` / `RAW_states` |
| `dense_SCL_modeled_BiologicalOde_rates` / `dense_RAW_…` | `(n_dense, n_rates)` | rates pair |
| `dense_SCL_modeled_FVCs_rates` / `dense_RAW_…` | `(n_dense, n_modeled_FVCs)` | feed-rates pair |
| `dense_SCL_V` / `dense_RAW_V` | `(n_dense,)` | volume pair |
| `dense_auxiliary` | `dict[str, (n_dense, …)]` | `auxiliary` |

### Masks (sparse, unaligned measurements)

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

## `LossOutputs` and aggregation

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

## `loss_names` — required property

Declare the term names up front; they must equal the keys your `__call__`
returns, in order:

```python
@property
def loss_names(self) -> tuple[str, ...]:
    return ("biomass", "glucose", "lwr_bnd/q_glc")
```

The harness reads `loss_names` once at setup to build the console table, CSV,
JSONL, and `loss_curve.png` panels, and to size/validate the loss vector. A
mismatch between `loss_names` and the returned keys is a fail-fast error.

## Choosing the loss type (MAE / MSE / Huber)

`DefaultLossModule.residual_reduction(residual, mask)` is the per-target
reduction (default: masked mean-squared error). Override it to switch:

```python
class MAELossModule(DefaultLossModule):
    def residual_reduction(self, residual, mask):
        masked = jnp.where(mask, jnp.abs(residual), 0.0)
        n_active = jnp.maximum(jnp.sum(mask, axis=0), 1)
        return jnp.sum(masked, axis=0) / n_active
```

## Adding custom loss terms

Subclass and add named entries — they show up automatically as new panels and
log columns:

```python
class BoundsHingeLossModule(DefaultLossModule):
    bound_records: tuple = eqx.field(static=True)
    weight: float = eqx.field(static=True)

    def __init__(self, *, target_names, collection, weight):
        super().__init__(target_names=target_names)
        self.bound_records = tuple(_collect_bounds(collection, ...))
        self.weight = float(weight)

    @property
    def loss_names(self):
        return tuple(self.target_names) + tuple(l for l, *_ in self.bound_records)

    def __call__(self, inputs):
        base = super().__call__(inputs).named_losses          # measurement terms
        rm = inputs.reaction_module
        penalties = {}
        for label, source, idx, side, threshold_RAW in self.bound_records:
            values = inputs.SCL_states[:, idx]
            scl_threshold = threshold_RAW / rm.SCALE_state[idx]
            penalties[label] = self.weight * _hinge_sq(
                values, scl_threshold, side, inputs.mask_measured_any
            )
        return LossOutputs(named_losses={**base, **penalties})
```

See [examples/11_tub_2026/fba_hyb/custom.py](../examples/11_tub_2026/fba_hyb/custom.py)
for the full bounds-hinge module.

## Trainable loss parameters

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
leaves are trainable vs frozen. Trainability is declared solely through field
tags; there is no custom `partition_trainable()` override. For advanced
sub-field control (e.g. freezing some MLP layers), use the `build_optimizer`
hook with `optax.masked` / `optax.multi_transform` rather than a second
partition mechanism.

## Using `auxiliary`

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

## Dense-grid losses (rate curvature, between-measurement constraints)

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

### Example 1 — rate-curvature penalty

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

The full version (with `nonneg/<target>` measurement terms alongside the
curvature) is in
[tests/fixtures/martens_single/custom.py](../tests/fixtures/martens_single/custom.py)
— it runs end-to-end through `prepare -> train -> forward -> losses.csv`.

### Example 2 — between-measurement bounds

Today's `BoundsHingeLossModule` only enforces bounds at measurement times.
Swapping `inputs.SCL_states` → `inputs.dense_SCL_states` (and
`inputs.mask_measured_any` → a dense-point mask) makes the same hinge fire on
every dense point — the bound is then enforced *everywhere* the solver
reports state, not only when the plate was sampled:

```python
# inside __call__ of a DefaultLossModule subclass with dense_grid_n set
values = inputs.dense_SCL_states[:, idx]
mask = dense_point_mask_away_from_jumps(
    inputs.dense_t, inputs.jump_ts, jump_eps_h
).astype(values.dtype)
penalties[label] = self.weight * _hinge_sq(values, scl_threshold, side, mask)
```

### Practical note on `dense_grid_n` size

The added cost is just more `SaveAt` evaluations of `wrapper.save_outputs`;
the underlying solver steps are adaptive and unchanged. In practice, very
large `dense_grid_n` (on the order of the number of internal solver steps)
can stress the JIT'd graph during the backward pass on some setups; on the
`martens_single` fixture `dense_grid_n=32` is comfortable, `dense_grid_n>=64`
started to bite at the default solver tolerances (segfault inside the JIT'd
train step on the current JAX/diffrax). Pick the smallest N your finite
differences need; if you hit a hard crash at large N, drop it and/or loosen
the solver tolerances.

## Where terms show up

Every named term flows, by its key, to:

- the per-step console table,
- `run.csv` and `run.jsonl`,
- the checkpoint `loss_curve.png` (one panel per term, plus a `total` panel),
- the per-process fit plot: each species/feed subplot is annotated with its
  named term (when one matches by name) plus R²; the process's total loss is in
  the figure title, and non-species terms (penalties, aux) are listed there.
  Terms with no matching subplot simply don't annotate one — never an error.
