# Gallery

> **In one sentence.** Worked examples of the things the tutorials deliberately left out.

The tutorials use one small batch dataset and stop early on purpose: the point there is
the shape of the pipeline, not the ceiling of what it can do. This is where the ceiling
lives. Each entry is a self-contained, executed example built on the fed-batch demo
dataset or an extended version of the batch one.

Every entry's `.md` file is itself a real, executable notebook (MyST Markdown via
myst-nb), living in `source/gallery/` with its runnable `custom.py` alongside it in
`source/gallery/_files/`: clone the `bp-docs` source, install bp-format/bp-train, and
each page runs top to bottom exactly as shown here, no rendering required.

| Entry | Demonstrates |
|---|---|
| [Fed-batch](fed_batch.md) | Continuous feed **and** boluses **and** sampling in one run; a reaction module that reads the feed and a controlled process variable as real inputs. |
| [Mechanistic models](mechanistic_rates.md) | Mechanistic kinetics (Monod, Luedeking-Piret) instead of a bare MLP; recovering physical parameters, and where they trade off against each other. |
| [Custom losses on the dense grid](dense_loss.md) | A loss module that constrains the trajectory *between* measurements: bounds on states and rates read from bp-format's own metadata, plus a rate-smoothness penalty. |
| [Stateful reaction modules](stateful.md) | A continuous-time LSTM (a reaction module with its own memory, integrated as extra ODE latents) and the opt-in that guards it. |
| [Freezing parameters](freezing.md) | Splitting a reaction module into a frozen part and a trainable part with field tags, checked with `print_trainable_structure`, and what freezing actually costs. |
| [Cross-validation, worked](loo.md) | A cheap `holdout_processes` check, then a full leave-one-out run: real folds, the corrected config schema, and the files it produces. |
| [Augmentation](augmentation.md) | Generating synthetic sibling processes from a single run, and controlling what values they carry with `augment_state_values`. |
| [Gaussian process reaction module](gaussian_process.md) | A closed-form sparse-GP posterior, mean and variance, occupying a reaction module's slot instead of a neural network. |
| [Knowledge transfer](knowledge_transfer.md) | Pooling data across products to help a data-poor new one, using a controlled process variable as a product-identity feature. |

## See also

- [Tutorials](../tutorials/01_your_first_dataset.md): start here if you have not yet.
- [Which page do I need?](../start/find.md): the full feature routing table.
