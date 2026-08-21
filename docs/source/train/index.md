# bp-train guide

> bp-train takes a bp-format dataset, lets you plug in a model for the biology, and fits
> it by differentiating through an ODE solve. For the data side, see
> [bp-format](../format/index.md).

## The pipeline

<img class="theme-diagram diagram-light" src="../_static/diagram_train_pipeline_light.svg" alt="hybrax-format data (from data.csv or data.xlsx) flows through hybrax prepare (producing data.json, then prepared.json), hybrax train or loo (producing run/) and hybrax forward to produce predictions, rates and metrics. Measured data, process transformation and augmentation, the reaction module, loss module, scales, optimizer and learning rate (all custom.py), and old or new controls (data.json, new_data.json) feed in from the left. Each stage's own script or config file feeds in from the right: import.py, prepare-config.json, train-config.json or loo-config.json, and forward-config.json.">
<img class="theme-diagram diagram-dark" src="../_static/diagram_train_pipeline_dark.svg" alt="hybrax-format data (from data.csv or data.xlsx) flows through hybrax prepare (producing data.json, then prepared.json), hybrax train or loo (producing run/) and hybrax forward to produce predictions, rates and metrics. Measured data, process transformation and augmentation, the reaction module, loss module, scales, optimizer and learning rate (all custom.py), and old or new controls (data.json, new_data.json) feed in from the left. Each stage's own script or config file feeds in from the right: import.py, prepare-config.json, train-config.json or loo-config.json, and forward-config.json.">

Four commands. Everything you customise happens through one optional `custom.py`, and
every hook in it has a working default.

Each command's real output, on disk:

```
prepared/               prepare's output: the training problem
run/                     train's output
└── forward/             forward's output, written inside it
loo_run/                 loo's output
└── folds/<slug>/        one full run/ directory per fold
```

Full listings are on each stage's own page: [Prepare](prepare.md#what-it-writes),
[Saving, loading and predicting](save_load_predict.md#what-is-on-disk),
[Forward](forward.md#what-it-produces), [Cross-validation](loo.md#what-it-produces).

## What is actually being fitted

```
d(state)/dt  =  biology(rates)                              ← the reaction module
              + transport(feeds, dilution, samples, volume)  ← bp-format, already written
```

Training differentiates the *whole solve* (solver steps, event jumps, spline
evaluations) with respect to the reaction module's parameters, and descends. That is why
the stack is built on JAX and Diffrax, and why numerical conditioning matters more here
than in ordinary supervised learning.

## The pages

**Start here**

| Page | Read it when |
|---|---|
| [Configuration](config.md) | Always. It is the surface you actually touch. |
| [Prepare](prepare.md) | You want to know what happens between data and training. |

**The model: the two halves you write**

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
| **Scales** | **all ones: no scaling** | small, well-behaved datasets only |
| Optimizer | Adam, lr 1e-3, gradient clipping at norm 1000 | most things |
| Learning rate | constant | most things |
| Batching | full batch, shuffled | small process counts |

The one to fix first is scaling. It is optional, it fails silently, and it is the
difference between a model that trains and one that thrashes: see
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
Hooks are found by plain attribute lookup. `build_reaction_modul` is not an error: it is
a silent fall back to the default. If an edit to `custom.py` appears to change nothing,
check the spelling first.
:::

## See also

- [Quickstart](../start/quickstart.md): the whole pipeline in three commands.
- [Concepts and vocabulary](../start/concepts.md): SCL, RAW, targets, folds.
- [Design rationale](../under_the_hood/design_rationale.md): why it is built this way.

```{toctree}
:maxdepth: 1
:hidden:
config
prepare
reaction_module
scaling
loss_module
train
forward
loo
save_load_predict
hooks_cheatsheet
further_reading
```
