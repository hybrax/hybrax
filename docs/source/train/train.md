# Training

> The loop, the optimizer, the two hooks that shape it, and how to use more than one
> core.

```bash
hybrax train --config train-config.json [--overwrite] [--epochs N]
```

## Setup, in order

Before the first step, training resolves five hooks from `custom.py`:
`build_learning_rate` → `build_optimizer` → `estimate_all_scales` →
`build_reaction_module` → `build_loss_module`. Two more, `transform_process_collection`
and `augment_state_values`, already ran during `prepare`. See
[Customization](hooks_cheatsheet.md) for the full table, with every signature.

This page covers `build_learning_rate` and `build_optimizer` below: they are purely a
training concern. The other three are substantial enough to need their own page each:
[The Reaction Module](reaction_module.md), [The Loss Module](loss_module.md),
[Scaling](scaling.md).

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

Replace it to change the optimizer itself (a different `optax` transform, extra
gradient transforms in the chain). For **"train this part, freeze that part"**, do not
reach for this hook: split the module into separate fields and tag them with
`trainable_field()` / `frozen_field()` instead. That is a single-path mechanism that
works for any module, is checked before training with `print_trainable_structure`, and
needs no `optax` plumbing. See [Freezing parameters](../gallery/freezing.md) for a full
worked example.

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
JAX decides its CPU device count at import. `hybrax.train` therefore resolves the setting
*before* that, by scanning the command line and config at import time.

Consequences: **`HYBRAX_TRAIN_DEVICES=N` in the environment always wins** over the config
file; and if `XLA_FLAGS` already sets `xla_force_host_platform_device_count`, the whole
bootstrap is skipped and your value stands.
:::

The default is **1**: `hybrax.train` never quietly takes over your machine. `"max"`
deliberately does not mean "all cores": surplus idle devices can deadlock the `pmap`
rendezvous on an AllReduce timeout, so it is capped at the process count. Requesting more
devices than you have cores is capped, with a warning to stderr.

None of this affects GPU.

:::{admonition} Do not fan out training runs in parallel
:class: warning
Several JAX processes each claiming cores will oversubscribe and, on constrained
machines, get OOM-killed. Run one at a time, or shard within one run using `devices`.
:::

## Checkpoints and retention

```json
{
  "train": { "holdout_processes": ["Br7"] },
  "checkpoint": {
    "every": 10,
    "keep_best": 3,
    "select_by": "holdout_loss"
  }
}
```

`every` is measured in epochs and controls both checkpoint writes and holdout
checks. Omit it or use `null` for an automatic cadence of at least five epochs
and at most 20 checkpoints. Use `0` to disable periodic writes. The final step
always writes a checkpoint, even when it is also a periodic boundary.

Each checkpoint directory is self-contained: parameters, optimizer state,
config, `custom.py`, and prepared data. Set `bundle_prepared: false` to omit
prepared data from checkpoints. Checkpoints also contain measurement-grid
holdout predictions when original holdout processes are available. Use `forward`
for additional predictions or plots.

### Limit disk use

`keep_best` controls retention:

- Omitted or `null`: retain every checkpoint, the default. Omit `select_by`.
- `0`: retain latest only. Omit `select_by`; no ranking metric is needed.
- Positive N: retain the best N scored checkpoints plus latest. Set `select_by`.
  When latest is already among the best N, it is stored only once.

For example, `{ "every": 10, "keep_best": 0 }` replaces the previous checkpoint
at every boundary. With `keep_best: 3`, at most four `step_*` directories remain
once pruning completes. The new checkpoint is fully written and `latest` updated
before old checkpoints are deleted, so allow space for one temporary extra
checkpoint. On filesystems without symlinks, `latest` is a separate copy and uses
another checkpoint's worth of space.

Every boundary still writes a full checkpoint. Retention limits stored disk
space, not serialization or export work. Loss history retains scores for pruned
steps. `checkpoints/retention.json` lists surviving best checkpoints in ascending
score order and identifies latest. Each checkpoint's `train_state.json` records
its selection metric and score. Equal scores retain the older checkpoint.
Nonfinite scores produce a warning and cannot enter the best N, but their
checkpoint remains while it is latest.

`latest` always identifies the newest checkpoint. The completed run's `model/`
and default loading from the run directory use the final model, even if another
checkpoint scores better. To load a best checkpoint, use its directory from
`retention.json`; an explicit path to a pruned checkpoint no longer exists.

With retention enabled, an output directory containing old `step_*` checkpoints
is rejected before training. Choose a fresh output directory to preserve the
previous attempt, or pass `--overwrite` to clear it and start fresh. This also
applies to failed runs; restarting training does not resume old optimizer state
or combine checkpoint rankings. An incomplete output directory with no
checkpoints can be reused.

### Choose a selection metric

All supported metrics minimize loss and reuse existing results. They add no
extra prediction passes over training data:

- `holdout_loss`: evaluates the updated model saved in the checkpoint, using
  the same loss module as training. Requires holdout processes.
- `train_loss`: the last batch's logged loss before its optimizer update.
  It lags the saved parameters by one update and can be noisy with minibatches.
- `epoch_mean_loss`: averages logged batch losses over the epoch as parameters
  change. Every checkpoint boundary must land at an epoch end. A fractional
  `every` that produces a mid-epoch boundary is rejected; no stale score is used.

Other metric names are rejected. Custom selection hooks are not supported.

### Hold out processes in ordinary training

`train.holdout_processes` selects processes from the prepared data for evaluation
at checkpoint boundaries. If `data.processes` is omitted, training uses all
remaining processes, excluding each holdout's entire augmentation group, its
parent and children. An explicit `data.processes` selection that overlaps those
groups raises an error. Holdout groups are excluded before estimating scales
and constructing modules from training parents.

Holdout names must exist in the prepared dataset and be unique; the holdout list
must not be empty. Only the named holdouts are evaluated, even though their whole
augmentation groups are excluded from training.

For the example above, you can explicitly set
`data.processes` to `["Br1", "Br2", "Br3", "Br4", "Br9"]` and leave Br7 in the
prepared dataset. Final results include training and holdout losses for the
final model. The saved config records the resolved training selection so loading
rebuilds scales and modules from the same training parents. No additional custom
scoring hook is needed.

`train.holdout_processes` is for `train` only. LOO folds define their own holdouts
and reject this setting. Retention works separately within each fold. Using
`holdout_loss` to select a fold's best checkpoint uses that fold's test data;
LOO's reported results continue to use the final model.

See [The Python API](save_load_predict.md).

## Reading the output

| File | Use it for |
|---|---|
| `metrics.csv` | Per-epoch loss and gradient norm. The source of truth. |
| `loss_curve.png` | Is it converging? |
| `grad_norm_curve.png` | Raw gradient norm: is the clip active all the time? |
| `config.json`, `custom.py` | Exactly what was run. |

Trajectories, rates and per-process figures are not part of `train`'s own output:
point [`forward`](forward.md) at the run directory (`output.predictions` and
`output.plots`) to get those. **Judge the fit by the rates**, not only the
trajectories, once you have them: a model can match concentrations beautifully with
rates that are physically impossible: growth and death both far too high, or uptake
compensating for a transport error. Compensating errors are invisible in the
concentration plot and obvious in the rate one.

## Gotchas

- **`--overwrite` is required** to reuse a completed run directory, or an
  incomplete one containing checkpoints when retention is enabled. It deletes
  the previous output before starting fresh.
- **`--epochs` overrides the config**, which is what you want while iterating.
- **`batch_size` greater than the process count** raises rather than clamping.
- **Stateful modules need `train.allow_stateful_models: true`.**
- **`HYBRAX_GSPMD=1`** switches sharding to GSPMD auto-sharding. It is correct but roughly
  sixty times slower: a debugging tool, not an option.
- **x64 is on globally.** Importing hybrax enables JAX double precision.

## See also

- [Scaling](scaling.md): fix this before tuning anything here.
- [Forward](forward.md): what to do with the result.
- [Cross-Validation](loo.md), whether it generalises.
- [Errors](../troubleshooting/errors.md).
