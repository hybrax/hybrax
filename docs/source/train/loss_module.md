# The loss module

> **In one sentence.** Turns one solved trajectory into a dict of **named** scalar losses,
> whose **mean** is what gets differentiated.
>
> **You need this if** plain MSE on every target is not what you want: weighting,
> penalties, physical constraints. **You can skip it if** it is.

## What it is

An `eqx.Module` with two things:

```python
from bp_train import UserLossModule, LossInputs, LossOutputs

class MyLossModule(UserLossModule):
    target_names: tuple[str, ...] = eqx.field(static=True)

    @property
    def loss_names(self):
        return (*self.target_names, "smoothness")

    def __call__(self, inputs: LossInputs) -> LossOutputs:
        ...
        return LossOutputs(named_losses={...})
```

`loss_names` is declared up front and `__call__` must return **exactly** those keys, in
that order. Mismatch is a hard error: the names are what label the plot panels and the
metrics columns, so they cannot vary between steps.

## The hook

**Fires:** at training setup, last.
**Signature:** `(*, target_names, process_names, config, seed, runtime_context) -> UserLossModule`
**Default:** `DefaultLossModule`: one MSE term per measured target.
**Type-checked:** yes.

## Mean, not sum

The total loss for backprop is `mean(named_losses.values())`.

This is deliberate and it matters. bp-train clips the **raw** gradient before Adam, so
with a sum, adding a loss term scales the gradient by the term count, pushes it past the
clip threshold, and (because the clip sits before Adam) holds the step size large near
the optimum. On a stiff neural ODE that overshoots and diverges.

With a mean, a `grad_clip_norm` you tuned once keeps behaving the same as you add terms.
If you want weighted-sum behaviour, scale the individual terms inside `__call__` and
retune the clip.

## Changing the reduction only

The cheapest useful customisation. `DefaultLossModule` factors its per-target reduction
into an overridable method, so MAE or Huber is a subclass:

```python
import jax.numpy as jnp
from bp_train.defaults import DefaultLossModule

class MAELossModule(DefaultLossModule):
    def residual_reduction(self, residual, mask):
        absolute = jnp.where(mask, jnp.abs(residual), 0.0)
        n_active = jnp.maximum(jnp.sum(mask, axis=0), 1)
        return jnp.sum(absolute, axis=0) / n_active

def build_loss_module(*, target_names, **kwargs):
    return MAELossModule(target_names=list(target_names))
```

`residual` and `mask` are `(n_meas, n_target)`; you return `(n_target,)`. Note the
per-column normalisation: each target is divided by *its own* active-cell count, so a
sparsely measured species is not diluted by padding rows.

## What you get in `LossInputs`

Predictions arrive in **both** SCL and RAW space: the conversion is a cheap elementwise
broadcast, so you pick whichever the term needs. Fit residuals belong in SCL, where
targets are comparable; physical penalties belong in RAW, where the units mean something.

| Field | What it is |
|---|---|
| `SCL_target_pred` | Predictions sliced to the targets. The convenience path. |
| `SCL_target_measured` | Ground truth. |
| `SCL_states` / `RAW_states` | Full trajectories. |
| `SCL_modeled_BiologicalOde_rates` / `RAW_…` | The rates over time. |
| `SCL_V` / `RAW_V` | Volume. |
| `mask_measured` | `(n_meas, n_target)`: is this cell a real measurement? |
| `mask_measured_any` | `(n_meas,)` float: is this *row* real? Multiply trajectory-wide penalties by it. |
| `t_measured`, `n_measured` | Times, and the unpadded row count. |
| `reaction_module` | The scales, via `inputs.reaction_module.SCALE_*`. |
| `step` | Training step, or −1 outside training. For annealing. |
| `jump_ts` | Control-discontinuity times, for masking near events. |
| `auxiliary` | Whatever the reaction module passed along. |

### The masks are not optional

Real datasets measure different species at different times, and batches are padded to a
common length. A term that ignores `mask_measured` is fitting padding.

```python
residual = inputs.SCL_target_pred - jnp.where(
    inputs.mask_measured, inputs.SCL_target_measured, 0.0)
squared = jnp.where(inputs.mask_measured, jnp.square(residual), 0.0)
per_target = jnp.sum(squared, axis=0) / jnp.maximum(
    jnp.sum(inputs.mask_measured, axis=0), 1)
```

For a penalty that applies to the whole trajectory rather than to specific measurements,
multiply by `mask_measured_any`.

:::{admonition} `penalty * mask` is safe
:class: note
If a solve bailed partway (a stiff segment hitting the step cap) every point past the
failure is masked out *before* `LossInputs` is built, and the trajectory values carry a
finite fallback rather than `inf`/`nan`. So the `penalty * mask` idiom cannot produce
`0 * inf`. Dense penalties are the exception: gate those by `dense_valid_time`.
:::

## Adding a physical penalty

A bounds hinge, using the `bounds` you declared once in the data:

```python
def __call__(self, inputs):
    residual = inputs.SCL_target_pred - jnp.where(
        inputs.mask_measured, inputs.SCL_target_measured, 0.0)
    fit = self.residual_reduction(residual, inputs.mask_measured)

    # Concentrations must not go negative: in RAW space, where it means something.
    below = jnp.clip(-inputs.RAW_states, a_min=0.0)
    hinge = jnp.mean(jnp.square(below) * inputs.mask_measured_any[:, None])

    return LossOutputs(named_losses={
        **{name: fit[i] for i, name in enumerate(self.target_names)},
        "negativity": hinge,
    })
```

Remember to add `"negativity"` to `loss_names`.

## Trainable loss parameters

The loss module is partitioned by the same field tags as the reaction module, so a
parameter tagged `trainable_field()` here is optimized alongside the model. That is how
learned uncertainty weighting (Kendall-style) works:

```python
class KendallLoss(UserLossModule):
    log_sigma: jax.Array = trainable_field()   # one per target, learned
```

## The dense grid

By default a loss only sees measurement times. Declaring `dense_grid_n` asks the trainer
to also save on a dense linspace, and populates the `dense_*` mirror fields:

```python
@property
def dense_grid_n(self):
    return 200
```

This is how you penalise behaviour *between* measurements: smoothness, curvature, bounds
across the whole trajectory.

:::{admonition} Dense points are cheap, but not free
:class: note
A dense time is **not** a segment boundary. The solve still splits only at bolus and
sample events and reads the grid off `SaveAt` inside each segment: interpolation, not
extra solver steps. So a finer grid does not subdivide the integration or change which
samples bail. It does cost one interpolant evaluation per point per segment, and the
`dense_*` arrays are real arrays in the loss.
:::

Gate every dense penalty by `dense_valid_time`, which marks the rows before a failed
solve's bail point. The measurement masks do not cover the dense grid.

See [Gallery: dense losses](../gallery/dense_loss.md).

## Gotchas

- **`loss_names` must equal the returned keys, in order.** Fail-fast, by design.
- **A misspelled `build_loss_module`** silently uses per-target MSE.
- **Do not put scales on your loss module.** Read them from
  `inputs.reaction_module`.
- **`step` is −1 outside training** (in `forward`). Guard any annealing schedule that
  reads it.
- **The loss runs under `jit` and `grad`.** No Python control flow on traced values.

## See also

- [The reaction module](reaction_module.md): the other half.
- [Gallery: dense losses](../gallery/dense_loss.md): a full custom loss.
- [Design rationale](../under_the_hood/design_rationale.md): the mean-versus-sum argument.
- [API reference](../autoapi/bp_train/model_api/index).
