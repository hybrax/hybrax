# Which page do I need?

> **In one sentence.** Every feature in both packages, indexed by the thing you are
> trying to do.
>
> **You need this if** you know what you want but not where it lives. **You can skip it
> if** you are reading the tutorials in order: they cover the common path already.

This page is deliberately a lookup table, not a reading order. Nothing here needs to be
learned; scan for your row.

## Getting data in

| I want to… | Go to |
|---|---|
| turn my CSVs into a bp-format dataset | [Tutorial 1](../tutorials/01_your_first_dataset.md) |
| know which object to use for which measurement | [The data model](../format/data_model.md) |
| record a feed, a bolus, or a sample draw | [Volume, feeds and events](../format/volume_feeds_events.md) |
| say what a feed contains | [Volume, feeds and events](../format/volume_feeds_events.md) |
| store a value that never changes | `StaticVariable`: [The data model](../format/data_model.md) |
| save or load a dataset, or read one someone sent me | [Loading and saving](../format/load_and_save.md) |
| group runs from one publication | `BioProcessCollection` with `case_id`/`organism`/`citation` set: [The data model](../format/data_model.md) |
| handle raw data that is not a full case study yet | `BioProcessCollection` with those fields left `None`: [The data model](../format/data_model.md) |

## Checking and looking at data

| I want to… | Go to |
|---|---|
| check my dataset before modeling it | [Validating and inspecting](../format/validate_and_inspect.md) |
| see the structure as text | `print_process_structure`: [Validating and inspecting](../format/validate_and_inspect.md) |
| plot every measurement in a run | `plot_process`: [Validating and inspecting](../format/validate_and_inspect.md) |
| see the ODE that was assembled from my data | `print_rhs_ode`: [The Bioprocess ODE](../format/bioprocess_ode.md) |
| check my volume bookkeeping adds up | `validate_volume_consistency`: [Validating and inspecting](../format/validate_and_inspect.md) |

## Modeling

| I want to… | Go to |
|---|---|
| write my own biological ODE instead of the default | [The Bioprocess ODE](../format/bioprocess_ode.md) |
| model a species that accumulates inside cells | `algebraic` expressions: [The Bioprocess ODE](../format/bioprocess_ode.md) |
| interpolate a measurement continuously | [Time series and splines](../format/time_series_and_splines.md) |
| remove dilution from fed-batch concentrations | Pseudobatch: [Time series and splines](../format/time_series_and_splines.md) |
| generate synthetic ground-truth data | [Limits and gotchas](../format/limits_and_gotchas.md) and the `Simulation` helpers |

## Training

| I want to… | Go to |
|---|---|
| train something, anything, right now | [Quickstart](quickstart.md) |
| understand the config file | [Configuration](../train/config.md) |
| know what `prepare` actually does | [Prepare](../train/prepare.md) |
| replace the neural network that predicts rates | [The reaction module](../train/reaction_module.md) |
| put real kinetics in instead of a bare MLP | [Gallery: mechanistic models](../gallery/mechanistic_rates.md) |
| change the loss, or add a penalty term | [The loss module](../train/loss_module.md) |
| penalise something *between* measurements | [Gallery: dense losses](../gallery/dense_loss.md) |
| fix badly conditioned training | [Scaling](../train/scaling.md) |
| choose what is optimized and what is frozen | [The reaction module](../train/reaction_module.md) |
| use a learning-rate schedule | [Training](../train/train.md) |
| use more than one CPU core | [Training](../train/train.md) |
| resume an interrupted run | [Saving, loading and predicting](../train/save_load_predict.md) |

## Evaluating

| I want to… | Go to |
|---|---|
| re-simulate with a trained model | [Forward](../train/forward.md) |
| get dense trajectories and rates as CSV | [Forward](../train/forward.md) |
| cross-validate | [Cross-validation](../train/loo.md), worked: [Gallery](../gallery/loo.md) |
| do a cheap holdout check without a full LOO run | `holdout_processes`: [Gallery: cross-validation](../gallery/loo.md) |
| average several models and get a spread | [Forward](../train/forward.md) |
| load a trained model in Python and predict | [Saving, loading and predicting](../train/save_load_predict.md) |
| see which parameters are actually being trained | `print_trainable_structure`: [The reaction module](../train/reaction_module.md) |
| map array indices back to biological names | `print_reaction_schema`: [The reaction module](../train/reaction_module.md) |

## When it goes wrong

| I want to… | Go to |
|---|---|
| understand an error message | [Errors](../troubleshooting/errors.md) |
| find out why a run that "worked" gives nonsense | [Silent failures](../troubleshooting/silent_failures.md) |
| know what is simply not implemented | [Limits and gotchas](../format/limits_and_gotchas.md) |
| find the exhaustive reference for an API | [API reference](../autoapi/bp_format/index) · [bp-train](../autoapi/bp_train/index) |

## Things the tutorials leave out on purpose

The five tutorials use one batch dataset and nothing else. Everything below is in the
[gallery](../gallery/index.md), each as a self-contained worked example.

| Topic | Page |
|---|---|
| Continuous feed, boluses and sampling in one run | [Fed-batch](../gallery/fed_batch.md) |
| Mechanistic rate laws, partially trainable | [Mechanistic models](../gallery/mechanistic_rates.md) |
| Bounds and smoothness penalties between measurements | [Dense losses](../gallery/dense_loss.md) |
| A reaction module with memory (latent-ODE / LSTM) | [Stateful reaction modules](../gallery/stateful.md) |
| Freezing part of a reaction module, checked before training | [Freezing parameters](../gallery/freezing.md) |
| A holdout check and a full leave-one-out run, executed | [Cross-validation, worked](../gallery/loo.md) |
| A Gaussian process as the reaction module | [Gaussian process reaction module](../gallery/gaussian_process.md) |
| Pooling data across products to help a data-poor new one | [Knowledge transfer](../gallery/knowledge_transfer.md) |
| A frozen surrogate of a real FBA solution, no LP solve during training | [FBA-Hyb](../gallery/fba_hyb.md) |
| A PLS-shaped component reading media composition alongside state | [PLS-dFBA](../gallery/pls_dfba.md) |
| Learnable per-edge functions (a KAN) instead of a neural network's fixed activations | [A KAN model](../gallery/kan.md) |
| A real published rate law with a controlled variable (temperature) feeding the kinetics directly | [OptFed](../gallery/optfed.md) |
| One declared rate feeding two coupled derivatives at once | [Glutamine decay](../gallery/glutamine_decay.md) |
