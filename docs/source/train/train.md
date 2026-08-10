# Training

> **In one sentence.** The loop, the optimizer, the two hooks that shape it, and how to
> use more than one core.
>
> **You need this if** you are tuning a run rather than just launching one. **You can
> skip it if** the defaults are converging.

```bash
bp-train train --config train-config.json [--overwrite] [--epochs N] [--no-plot]
```

## What one step does

For each process in the batch:

1. Solve the ODE **once**, from `t_start` to `t_end`, in SCL space with a bounded
   physical state and discrete jumps applied at event times.
2. Save states and rates at the measurement times (and on a dense grid, if the loss module
   asked for one).
3. Hand those to the loss module; take the **mean** of its named losses.
4. Differentiate the whole thing (solver steps, event jumps, spline evaluations) with
   respect to the trainable parameters.
5. Clip the **raw** gradient by global norm, then apply the optimizer.

One solve per sample serves both the reaction module (which runs *inside* it) and the
loss module (which reads its saved outputs *after*). Adding dense save points costs extra
interpolant evaluations, not extra solver steps.

## The knobs that matter

```json
{
  "train": {
    "epochs": 2000,
    "learning_rate": 3e-4,
    "grad_clip_norm": 10.0,
    "batch_size": 8,
    "shuffle": true,
    "seed": 0,
    "optimizer": "adam",
    "devices": "max"
  },
  "solver": { "max_steps": 4096, "rtol": 1e-5, "atol": 1e-7 }
}
```

**`grad_clip_norm`** defaults to 1000, which is effectively off. Once your scales are
right, a real value (1 to 10) is usually what stabilises a stiff run. Check
`grad_norm_curve.png` to pick it: clip somewhere around the bulk of the distribution, not
below it.

**`solver.max_steps`** is the first thing to raise when solves start failing. Failures are
not fatal (points after the bail are masked out of the loss) but a run where most
samples bail is fitting almost nothing. If raising it does not help, the problem is
usually stiffness caused by bad scaling, not the solver.

**`batch_size`** must not exceed the number of selected processes. It is not clamped;
you get a `ValueError`.

## Hook: `build_learning_rate`

**Fires:** first, at setup.
**Signature:** `(custom_cfg, train_cfg, total_updates) -> float | optax.Schedule`
**Default:** the constant `train.learning_rate`.

```python
import optax

def build_learning_rate(custom_cfg, train_cfg, total_updates):
    warmup = max(1, total_updates // 20)
    return optax.join_schedules(
        [
            optax.linear_schedule(0.0, train_cfg.learning_rate, warmup),
            optax.cosine_decay_schedule(train_cfg.learning_rate,
                                        total_updates - warmup, alpha=0.05),
        ],
        boundaries=[warmup],
    )
```

`total_updates` is precomputed for you, so schedules can be defined in terms of the whole
run rather than guessed at.

## Hook: `build_optimizer`

**Fires:** second, at setup.
**Signature:** `(custom_cfg, train_cfg) -> optax.GradientTransformation`
**Default:** `clip_by_global_norm(train.grad_clip_norm)` then `adam` or `sgd`.

Replace it when you need per-parameter treatment that field tags cannot express: the
usual case being "train the output layer, freeze the rest":

```python
import equinox as eqx
import jax
import optax

def build_optimizer(custom_cfg, train_cfg):
    base = optax.chain(
        optax.clip_by_global_norm(train_cfg.grad_clip_norm),
        optax.adam(train_cfg.learning_rate),
    )

    def label(path, _leaf):
        name = jax.tree_util.keystr(path)
        return "train" if "layers[2]" in name else "freeze"

    return optax.multi_transform(
        {"train": base, "freeze": optax.set_to_zero()},
        param_labels=lambda params: jax.tree_util.tree_map_with_path(label, params),
    )
```

:::{admonition} Order of operations
:class: note
The clip is applied to the **raw** gradient, before Adam. That is what makes the
mean-not-sum loss aggregation matter, and it is why a clip value tuned once stays valid
as you add loss terms. If you replace the optimizer, keep the clip first unless you know
why you are moving it.
:::

Note that `build_learning_rate` and `build_optimizer` are independent: if you build the
optimizer yourself, wire the schedule in yourself too.

## Running on more than one core

Training can shard the process batch across CPU cores with `pmap`, for roughly an N×
speedup.

```json
{ "train": { "devices": 4 } }
```

or `"devices": "max"`, which resolves to `min(n_processes, n_cpus)`.

:::{admonition} The device count is fixed before JAX initializes
:class: important
JAX decides its CPU device count at import. bp-train therefore resolves the setting
*before* that, by scanning the command line and config at import time.

Consequences: **`BP_TRAIN_DEVICES=N` in the environment always wins** over the config
file; and if `XLA_FLAGS` already sets `xla_force_host_platform_device_count`, the whole
bootstrap is skipped and your value stands.
:::

The default is **1**: bp-train never quietly takes over your machine. `"max"`
deliberately does not mean "all cores": surplus idle devices can deadlock the `pmap`
rendezvous on an AllReduce timeout, so it is capped at the process count. Requesting more
devices than you have cores is capped, with a warning to stderr.

None of this affects GPU.

:::{admonition} Do not fan out training runs in parallel
:class: warning
Several JAX processes each claiming cores will oversubscribe and, on constrained
machines, get OOM-killed. Run one at a time, or shard within one run using `devices`.
:::

## Checkpoints and resuming

```json
{ "checkpoint": { "every": 100 } }
```

Each checkpoint directory is **self-contained**: parameters, optimizer state, config,
`custom.py`, the prepared data, and a `predictions.csv` for that step. You can point
`forward` at a checkpoint exactly as at a run directory.

That self-containment costs time: every checkpoint re-exports predictions. On a fast run
the checkpointing can dominate the wall clock, so set `every` to something coarse.

See [Saving, loading and predicting](save_load_predict.md).

## Reading the output

| File | Use it for |
|---|---|
| `metrics.csv` | Per-epoch loss and gradient norm. The source of truth. |
| `loss_curve.png` | Is it converging? |
| `grad_norm_curve.png` | Raw gradient norm: is the clip active all the time? |
| `<process>.png` | Fit and inferred rates, per process. |
| `predictions.csv` | Dense trajectories at the end of training. |
| `config.json`, `custom.py` | Exactly what was run. |

**Judge the fit by the rates**, not only the trajectories. A model can match
concentrations beautifully with rates that are physically impossible: growth and death
both far too high, or uptake compensating for a transport error. Compensating errors are
invisible in the left column and obvious in the right one.

## Gotchas

- **`--overwrite` is required** to reuse a run directory.
- **`--epochs` overrides the config**, which is what you want while iterating.
- **`batch_size` greater than the process count** raises rather than clamping.
- **Stateful modules need `train.allow_stateful_models: true`.**
- **`BP_GSPMD=1`** switches sharding to GSPMD auto-sharding. It is correct but roughly
  sixty times slower: a debugging tool, not an option.
- **x64 is on globally.** Importing the packages enables JAX double precision.

## See also

- [Scaling](scaling.md): fix this before tuning anything here.
- [Forward](forward.md): what to do with the result.
- [Cross-validation](loo.md), whether it generalises.
- [Errors](../troubleshooting/errors.md).
