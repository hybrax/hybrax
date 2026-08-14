# Bioprocess Modeling with Hybrax

**Get your fermentation data into one shape, then fit models to it without
re-implementing the physics.**

Two packages, one stack:

- **bp-format** is the data model. It describes a bioprocess run (what was in the
  reactor, what was fed, what was sampled, what was measured) and turns that
  description into a differentiable ODE right-hand side. It knows about dilution,
  feed composition, boluses and sampling, so you don't have to write those terms
  again.
- **bp-train** fits models on top of it. You plug in a reaction module (neural,
  mechanistic, or both) and a loss module; it runs `prepare → train → forward / loo`
  on JAX + Diffrax.

If you have measurements and you want rates, this is the stack.

---

## Where to start

| If you… | Go to |
|---|---|
| have never used this before | [Why this exists](start/why.md), then the [Quickstart](start/quickstart.md) |
| want to see it work in 10 minutes | [Quickstart](start/quickstart.md): three commands, no code |
| have your own data to load | [Tutorial 1: your first dataset](tutorials/01_your_first_dataset.md) |
| keep hitting unfamiliar words | [Concepts and vocabulary](start/concepts.md) |
| know what you want but not where it is | [Which page do I need?](start/find.md) |
| got an error you don't understand | [Troubleshooting](troubleshooting/errors.md) |

New readers: the five [tutorials](tutorials/01_your_first_dataset.md) are meant to be
read in order and take about an hour end to end. They deliberately use one small
batch dataset and nothing else: feeds, boluses, cross-validation and custom losses
all wait until the [gallery](gallery/index.md).

:::{note}
Every code block in the Start-here, Tutorials and Gallery sections is **executed when
these docs are built**, against the installed packages. If something on those pages is
out of date, the build fails rather than the page silently lying to you.
:::

```{toctree}
:maxdepth: 1
:caption: Start here
:hidden:
start/why
start/install
start/quickstart
start/concepts
start/find
```

```{toctree}
:maxdepth: 1
:caption: Tutorials
:hidden:
tutorials/01_your_first_dataset
tutorials/02_look_at_it
tutorials/03_train
tutorials/04_your_first_custom_py
tutorials/05_predict
```

```{toctree}
:maxdepth: 1
:caption: bp-format guide
:hidden:
format/index
format/data_model
format/load_and_save
format/validate_and_inspect
format/volume_feeds_events
format/time_series_and_splines
format/bioprocess_ode
format/limits_and_gotchas
format/further_reading
```

```{toctree}
:maxdepth: 1
:caption: bp-train guide
:hidden:
train/index
train/config
train/prepare
train/reaction_module
train/scaling
train/loss_module
train/train
train/forward
train/loo
train/save_load_predict
train/hooks_cheatsheet
train/further_reading
```

```{toctree}
:maxdepth: 1
:caption: Gallery
:hidden:
gallery/index
gallery/fed_batch
gallery/mechanistic_rates
gallery/dense_loss
gallery/stateful
gallery/freezing
gallery/loo
gallery/augmentation
gallery/gaussian_process
gallery/knowledge_transfer
gallery/fba_hyb
gallery/pls_dfba
gallery/kan
```

```{toctree}
:maxdepth: 1
:caption: Troubleshooting
:hidden:
troubleshooting/errors
troubleshooting/silent_failures
```

```{toctree}
:maxdepth: 1
:caption: Under the hood
:hidden:
under_the_hood/design_rationale
```

```{toctree}
:maxdepth: 1
:caption: API reference
:hidden:
autoapi/bp_format/index
autoapi/bp_train/index
```
