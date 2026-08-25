# Overview

> hybrax.train takes a hybrax.format dataset, lets you plug in a model for the biology, and fits
> it by differentiating through an ODE solve. For the data side, see
> [hybrax.format](../format/index.md).

## The pipeline

<img class="theme-diagram diagram-light" src="../_static/diagram_train_pipeline_light.svg" alt="hybrax-format data (from data.csv or data.xlsx) flows through hybrax prepare (producing data.json, then prepared.json), hybrax train or loo (producing run/) and hybrax forward to produce predictions, rates and metrics. Measured data, process transformation and augmentation, the reaction module, loss module, scales, optimizer and learning rate (all custom.py), and old or new controls (data.json, new_data.json) feed in from the left. Each stage's own script or config file feeds in from the right: import.py, prepare-config.json, train-config.json or loo-config.json, and forward-config.json.">
<img class="theme-diagram diagram-dark" src="../_static/diagram_train_pipeline_dark.svg" alt="hybrax-format data (from data.csv or data.xlsx) flows through hybrax prepare (producing data.json, then prepared.json), hybrax train or loo (producing run/) and hybrax forward to produce predictions, rates and metrics. Measured data, process transformation and augmentation, the reaction module, loss module, scales, optimizer and learning rate (all custom.py), and old or new controls (data.json, new_data.json) feed in from the left. Each stage's own script or config file feeds in from the right: import.py, prepare-config.json, train-config.json or loo-config.json, and forward-config.json.">

Four commands. Everything you customise happens through one optional `custom.py`, and
every hook in it has a working default.

None of these directory names are automatic beyond a literal fallback. `prepare`,
`train`, and `loo` all fall back to the same literal `output/` if you set neither
`--output-dir` nor `output.dir`, so anything beyond a single throwaway run needs a name
you chose. Only `forward` has a real default: `<first model>/forward`, nested inside the
model's own run directory, unless you point it elsewhere.

Whichever directory a command lands on, `--overwrite` deletes everything already there,
regardless of what put it there, before writing fresh output. There is no partial or
selective overwrite.

A typical layout, one command's output per directory:

```
prepared/                prepare's output
├── prepared.json
├── prepare_config.json
└── prepare_diagnostics/

run/                     train's output
├── config.json
├── custom.py
├── metrics.csv
├── losses.csv
├── model/
└── checkpoints/step_NNNNN/

loo_run/                 loo's output
├── loo-config.json
├── folds/<slug>/        one run/ per fold
└── loo_summary.csv

forward/                 forward's output
├── losses.csv
├── predictions.csv
└── plots/
```

Full listings are on each stage's own page: [Prepare](prepare.md#what-it-writes),
[The Python API](save_load_predict.md#what-is-on-disk),
[Forward](forward.md#what-it-produces), [Cross-Validation](loo.md#what-it-produces).

## What is actually being fitted

$$
\frac{d\,\mathrm{state}}{dt} \;=\; \mathrm{biology}(\mathrm{state}, \mathrm{RATES}) \;+\; \mathrm{transport}(\mathrm{state}, \mathrm{controls})
$$

`biology` and `transport` are both hybrax.format's: fixed the moment your dataset is
prepared, and covered in full on [The Bioprocess ODE](../format/bioprocess_ode.md#the-split).
The one term hybrax.format leaves open is `RATES`, and supplying it is the entire job of
the reaction module:

$$
\mathrm{RATES} = \mathrm{reaction\_module}(t, \mathrm{inputs})
$$

Training differentiates the *whole solve* (solver steps, event jumps, spline
evaluations) with respect to the reaction module's parameters, and descends. That is why
the stack is built on JAX and Diffrax, and why numerical conditioning matters more here
than in ordinary supervised learning.

## The pages

**Start here**

| Page | Read it when |
|---|---|
| [Configuration](config.md) | Always. It is the surface you actually touch. |
| [Customization](hooks_cheatsheet.md) | Every hook, signature, default, and where it fires. |

**Prepare and train**

| Page | Read it when |
|---|---|
| [Prepare](prepare.md) | You want to know what happens between data and training. |
| [Training](train.md) | Optimizers, schedules, batching, devices. |
| [The Reaction Module](reaction_module.md) | You are replacing the default MLP. |
| [The Loss Module](loss_module.md) | Plain MSE on every target is not what you want. |
| [Scaling](scaling.md) | Right after the loss module. In practice this is not optional. |

**Running it**

| Page | Read it when |
|---|---|
| [Forward](forward.md) | You have a model and want trajectories out of it. |
| [Cross-Validation](loo.md) | You need to know whether it generalises. |
| [The Python API](save_load_predict.md) | Loading, predicting and resuming from a script, not the CLI. |

**Reference**

| Page | |
|---|---|
| [Further Reading](further_reading.md) | The dense reference and the example projects. |

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
check the spelling first. Every `prepare`, `train` and `loo` run logs which hooks it found
at startup, `<stage> hooks detected: ...` and `<stage> hooks default: ...`: check that line
before you go looking any further.
:::

## See also

- [Quickstart](../start/quickstart.md): the whole pipeline in three commands.
- [Concepts and vocabulary](../start/concepts.md): SCL, RAW, targets, folds.
- [Design rationale](../under_the_hood/design_rationale.md): why it is built this way.
