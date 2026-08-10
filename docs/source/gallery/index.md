# Gallery

> **In one sentence.** Worked examples of the things the tutorials deliberately left out.

The tutorials use one small batch dataset and stop early on purpose: the point there is
the shape of the pipeline, not the ceiling of what it can do. This is where the ceiling
lives. Each entry is a self-contained, executed example built on the fed-batch demo
dataset or an extended version of the batch one.

| Entry | Demonstrates |
|---|---|
| [Fed-batch](fed_batch.md) | Continuous feed **and** boluses **and** sampling in one run; a reaction module that reads the feed and a controlled process variable as real inputs. |
| [Mechanistic models](mechanistic_rates.md) | Mechanistic kinetics (Monod, Luedeking-Piret) instead of a bare MLP; recovering physical parameters, and where they trade off against each other. |
| [Custom losses on the dense grid](dense_loss.md) | A loss module that constrains the trajectory *between* measurements: bounds on states and rates read from bp-format's own metadata, plus a rate-smoothness penalty. |
| [Stateful reaction modules](stateful.md) | A continuous-time LSTM (a reaction module with its own memory, integrated as extra ODE latents) and the opt-in that guards it. |

More reference-quality examples (real published datasets, LOO ensembles, multi-model
comparisons) live in `bp-train/examples/` and are indexed in
[bp-train further reading](../train/further_reading.md).

## See also

- [Tutorials](../tutorials/01_your_first_dataset.md): start here if you have not yet.
- [Which page do I need?](../start/find.md): the full feature routing table.
