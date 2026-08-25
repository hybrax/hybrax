# Overview

> Worked examples of the things the tutorials deliberately left out.

The tutorials use one small batch dataset and stop early on purpose: the point there is
the shape of the pipeline, not the ceiling of what it can do. This is where the ceiling
lives. Each entry is a self-contained, executed example, most built on the fed-batch
demo dataset or an extended version of the batch one.

Every entry's `.md` file is itself a real, executable notebook (MyST Markdown via
myst-nb), living in `source/gallery/`. Its runnable `custom.py`, configs and frozen
input data live alongside it in `hybrax`'s own top-level `examples/gallery_<name>/`:
clone `hybrax`, install it, and each page runs top to bottom exactly as shown here,
no rendering required — or run the `examples/` directory directly, no docs build
needed at all.

## Format Demo

| Entry | Demonstrates |
|---|---|
| [Pseudobatch Splines](pseudobatch_splines.md) | Recovering a smooth curve through a discrete feed jump from just 5 measurements, checked against a known ground truth. |

## Prepare Demo

| Entry | Demonstrates |
|---|---|
| [Augmentation](augmentation.md) | Generating synthetic sibling processes from a single run, and controlling what values they carry with `augment_state_values`. |

## Train Demo

| Entry | Demonstrates |
|---|---|
| [Feeds, Boluses, and Samples](fed_batch.md) | Continuous feed **and** boluses **and** sampling in one run; a reaction module that reads the feed and a controlled process variable as real inputs. |
| [Modeled Process Variables](modeled_pv.md) | A process variable with its own trained rate, contrasted against a modeled reactor component: a large feed bolus dilutes the component but leaves the process variable untouched. |
| [Freezing Parameters](freezing.md) | Splitting a reaction module into a frozen part and a trainable part with field tags, checked with `print_trainable_structure`, and what freezing actually costs. |
| [Mechanistic Models](mechanistic_rates.md) | Mechanistic kinetics (Monod, Luedeking-Piret) instead of a bare MLP; recovering physical parameters, and where they trade off against each other. |
| [Custom Losses](dense_loss.md) | A loss module that constrains the trajectory *between* measurements: bounds on states and rates read from hybrax.format's own metadata, plus a rate-smoothness penalty. |
| [Stateful Models](stateful.md) | A continuous-time LSTM (a reaction module with its own memory, integrated as extra ODE latents) and the opt-in that guards it. |
| [Glutamine Decay](glutamine_decay.md) | One declared rate feeding two coupled derivatives at once, a sink in one, a source in the other, recovered from data as a single shared number. |

## Loo Demo

| Entry | Demonstrates |
|---|---|
| [Cross-Validation](loo.md) | A cheap `holdout_processes` check, then a full leave-one-out run: real folds, the corrected config schema, and the files it produces. |

## Paper Demo

| Entry | Demonstrates |
|---|---|
| [Gaussian Processes](gaussian_process.md) | A closed-form sparse-GP posterior, mean and variance, occupying a reaction module's slot instead of a neural network. |
| [Knowledge Transfer](knowledge_transfer.md) | Pooling data across products to help a data-poor new one, using a controlled process variable as a product-identity feature. |
| [FBA-Hyb](fba_hyb.md) | A frozen, pole-free surrogate of a real flux-balance-analysis solution inside a reaction module, so no LP solve ever happens during training. |
| [PLS-dFBA](pls_dfba.md) | FBA-Hyb extended with an actual PLS-shaped component (linear, low-rank, no neural network) that reads media composition alongside state. |
| [KAN Models](kan.md) | Learnable univariate functions on edges, summed at nodes, occupying a reaction module's slot instead of a neural network, with each edge's learned curve read out directly after training. |
| [OptFed Models](optfed.md) | A real, published non-competitive-inhibition Michaelis-Menten rate law with Eyring-equation temperature dependence, `temperature` feeding straight into the kinetics as a controlled process variable. |

## See also

- [Tutorials](../tutorials/01_your_first_dataset.md): start here if you have not yet.
- [Which page do I need?](../start/find.md): the full feature routing table.
