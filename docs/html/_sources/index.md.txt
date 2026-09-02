# Bioprocess Modeling with Hybrax

**Import your fermentation data into the right format once, then fit different
mechanistic and hybrid models to it with little extra code.**

One package, two halves:

- **`hybrax.format`** is the data model. It describes a bioprocess run (what was
  in the reactor, what was fed, how much was sampled, what was measured) and
  turns that description into the structure needed for modeling. It knows about
  dilution, feed composition, boluses and sampling, so you don't have to write
  those terms again.
- **`hybrax.train`** fits models to your data. You plug in a reaction module
  (data-driven, mechanistic, or hybrid) that describes the biological part of
  the ODE right-hand side and the library handles the rest. The training pipeline is split into multiple stages:
  - `prepare`: additional data prep (filtering, augmentation, etc.)
  - `train`: train a single model on one or multiple bioprocess runs
  - `loo`: cross-validation (leave-one-out per default but customizable) on a
    set of runs

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
these docs are built**, against the installed package. If something on those pages is
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
:caption: hybrax.format guide
:hidden:
format/index
format/data_model
format/load_and_save
format/validate_and_inspect
format/volume_feeds_events
format/time_series_and_splines
format/pseudobatch_transform
format/bioprocess_ode
format/limits_and_gotchas
format/further_reading
```

```{toctree}
:maxdepth: 1
:caption: hybrax.train guide
:hidden:
train/index
train/config
train/hooks_cheatsheet
train/prepare
train/train
train/reaction_module
train/loss_module
train/scaling
train/forward
train/loo
train/save_load_predict
train/further_reading
```

```{toctree}
:maxdepth: 1
:caption: Gallery
:hidden:
gallery/index
Pseudobatch Splines <gallery/pseudobatch_splines>
Augmentation <gallery/augmentation>
Feeds, Boluses, and Samples <gallery/fed_batch>
Modeled Process Variables <gallery/modeled_pv>
Freezing Parameters <gallery/freezing>
Mechanistic Models <gallery/mechanistic_rates>
Custom Losses <gallery/dense_loss>
Stateful Models <gallery/stateful>
Glutamine Degradation <gallery/glutamine_decay>
Cross-Validation <gallery/loo>
Gaussian Processes <gallery/gaussian_process>
Knowledge Transfer <gallery/knowledge_transfer>
FBA-Hyb <gallery/fba_hyb>
PLS-dFBA <gallery/pls_dfba>
KAN Models <gallery/kan>
OptFed Models <gallery/optfed>
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
contributors
```

```{toctree}
:maxdepth: 1
:caption: API reference
:hidden:
autoapi/hybrax/format/index
autoapi/hybrax/train/index
```
