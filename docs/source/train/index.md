# bp-train guide

> **In one sentence.** bp-train takes a bp-format dataset, lets you plug in a model for
> the biology, and fits it by differentiating through an ODE solve.
>
> **You need this if** you want to fit anything. **You can skip it if** you only handle
> data — that is [bp-format](../format/index.md).

## The pipeline

```
data.json                  a bp-format CaseStudy or BioProcessCollection
    │
    │  bp-train prepare        + custom.py: transform_process_collection
    ▼                                       augment_state_values
prepared/                  the training problem: layouts, control splines, targets
    │
    │  bp-train train         + custom.py: estimate_all_scales
    ▼                                      build_reaction_module
run/                                       build_loss_module
    ├── model/params.eqx                   build_learning_rate
    ├── metrics.csv                        build_optimizer
    ├── predictions.csv
    └── <process>.png
    │
    ├── bp-train forward      re-simulate, export dense trajectories, ensemble
    └── bp-train loo          cross-validate (wraps train + forward per fold)
```

Four commands. Everything you customise happens through one optional `custom.py`, and
every hook in it has a working default.

## What is actually being fitted

```
d(state)/dt  =  biology(rates)                              ← the reaction module
              + transport(feeds, dilution, samples, volume)  ← bp-format, already written
```

Training differentiates the *whole solve* — solver steps, event jumps, spline
evaluations — with respect to the reaction module's parameters, and descends. That is why
the stack is built on JAX and Diffrax, and why numerical conditioning matters more here
than in ordinary supervised learning.

## The pages

**Start here**

| Page | Read it when |
|---|---|
| [Configuration](config.md) | Always. It is the surface you actually touch. |
| [Prepare](prepare.md) | You want to know what happens between data and training. |

**The model — the two halves you write**

| Page | Read it when |
|---|---|
| [The reaction module](reaction_module.md) | You are replacing the default MLP. |
| [Scaling](scaling.md) | Right after. In practice this is not optional. |
| [The loss module](loss_module.md) | Plain MSE on every target is not what you want. |

**Running it**

| Page | Read it when |
|---|---|
| [Training](train.md) | Optimizers, schedules, batching, devices. |
| [Forward](forward.md) | You have a model and want trajectories out of it. |
| [Cross-validation](loo.md) | You need to know whether it generalises. |
| [Saving, loading, predicting](save_load_predict.md) | Checkpoints, resuming, the Python API. |

**Reference**

| Page | |
|---|---|
| [custom.py at a glance](hooks_cheatsheet.md) | Every hook, signature, default, and where it fires. |
| [Further reading](further_reading.md) | The dense reference and the example projects. |

## Defaults, and what replacing them costs

You can train with no `custom.py` at all. These are what you get:

| Piece | Default | Good enough for |
|---|---|---|
| Reaction module | 2-layer MLP over the modeled state | a first look |
| Loss module | per-target mean squared error | a first look |
| **Scales** | **all ones — no scaling** | small, well-behaved datasets only |
| Optimizer | Adam, lr 1e-3, gradient clipping at norm 1000 | most things |
| Learning rate | constant | most things |
| Batching | full batch, shuffled | small process counts |

The one to fix first is scaling. It is optional, it fails silently, and it is the
difference between a model that trains and one that thrashes — see
[Scaling](scaling.md) and [Tutorial 4](../tutorials/04_your_first_custom_py.md), which
measures the gap.

## Two things that will bite on your first real dataset

:::{admonition} Every target needs a measurement at t₀
:class: warning
The most common first-run failure. If a target has no measured value at the first time
point of the union grid, training stops with an explicit error naming the process and the
target. The fix is in the message: supply a t₀ measurement, mark the quantity static, or
drop it from the targets.
:::

:::{admonition} A misspelled hook name is silent
:class: warning
Hooks are found by plain attribute lookup. `build_reaction_modul` is not an error — it is
a silent fall back to the default. If an edit to `custom.py` appears to change nothing,
check the spelling first.
:::

## See also

- [Quickstart](../start/quickstart.md) — the whole pipeline in three commands.
- [Concepts and vocabulary](../start/concepts.md) — SCL, RAW, targets, folds.
- [Design rationale](../under_the_hood/design_rationale.md) — why it is built this way.
