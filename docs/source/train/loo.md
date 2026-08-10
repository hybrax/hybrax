# Cross-validation (LOO)

> **In one sentence.** Train one model per fold with some processes held out, then
> aggregate how well each held-out run was predicted.
>
> **You need this if** you want to claim your model generalises. **You can skip it if**
> you are still getting a single run to fit.

```bash
bp-train loo --config loo-config.json [--output-dir DIR] [--overwrite]
bp-train loo --resume RUN_DIR
```

## Why it matters here more than usual

Bioprocess datasets are small (often fewer than ten runs) and the measurements within
one run are strongly correlated. A model can fit every run it was shown and be useless on
the next one, and a train-set loss will not tell you. Leave-one-process-out is the
smallest honest answer available.

The unit of holdout is a **process**, not a timepoint. Holding out individual measurements
from a run the model otherwise saw tests interpolation, not generalisation.

## Configuration

The config is a train config plus a `loo` section:

```json
{
  "data": { "prepared": "prepared" },
  "custom_py": "custom.py",
  "train": { "epochs": 2000, "seed": 0 },
  "loo": { "parallel_folds": 2, "devices_per_fold": 2 }
}
```

With no `per_fold_holdout_sets`, you get classic leave-one-out: one fold per process.
To hold out groups instead: replicates of one condition, say:

```json
{
  "loo": {
    "per_fold_holdout_sets": [["run_1", "run_2"], ["run_3", "run_4"]]
  }
}
```

## What it produces

```
loo_run/
├── loo-config.json      bundled verbatim: this is what --resume reads
├── custom.py
├── prepared.json
├── folds/
│   └── <slug>/          a complete run directory per fold
│       ├── model/ metrics.csv predictions.csv losses.csv …
├── loo_summary.csv      one row per fold
└── loo_aggregate.json   metrics across folds
```

Each fold is a full run directory, so anything you can do to a training run (`forward`,
`model_load`, plotting) you can do to a fold.

Aggregate metrics include R², NMAE, MAE and RMSE on the held-out processes. Read the
**spread across folds**, not just the mean: one fold much worse than the others usually
means that run is different in a way your model does not capture, which is more
informative than the average.

## Parallelism

Folds are independent, and each runs in its own subprocess.

```json
{ "loo": { "parallel_folds": 2, "devices_per_fold": 2 } }
```

That is 2 × 2 = 4 cores. Size it against your machine: the orchestrator deliberately
holds itself to one device so an exported `BP_TRAIN_DEVICES` meant for workers does not
reserve the pool.

:::{admonition} Do not also fan out at the shell level
:class: warning
`parallel_folds` already runs multiple JAX processes. Launching several `bp-train loo`
commands alongside each other oversubscribes the machine and, on constrained systems,
gets processes OOM-killed.
:::

## Resuming

```bash
bp-train loo --resume loo_run
```

Reloads the bundled `loo-config.json` verbatim (**no overrides**) and re-runs only the
folds that have no `losses.csv`. That is what the self-contained run directory buys you:
an interrupted twelve-hour cross-validation picks up where it stopped.

`--resume` and `--config` are mutually exclusive.

## Augmented data

If the dataset was augmented during prepare, synthetic children carry a `parent_process`
reference and LOO is **group-aware**: a parent and all of its children land in the same
fold.

This is not a nicety. Without it, a synthetic sibling of the held-out run sits in the
training set, and your cross-validation score is measuring memorisation. If you use
augmentation, use bp-train's LOO rather than rolling your own splits.

## Holdout without cross-validation

For a quick check without N full trainings, the Python API supports a plain holdout
(`holdout_processes` on `TrainHarnessConfig`). It has no config-file equivalent: 
API-only. `losses.csv` then labels each process `train` or `holdout`.

## Gotchas

- **N folds means N trainings.** Get a single run converging first; LOO multiplies
  everything, including your mistakes.
- **The hidden `--fold` flag is internal** worker dispatch. Do not use it.
- **Fold directories are named by slug**, derived from the held-out process names.
- **`--resume` ignores config changes.** If you edited the config, you want a fresh run,
  not a resume.

## See also

- [Training](train.md): get one fold right first.
- [Forward](forward.md): ensembles, which pair naturally with folds.
- [Prepare](prepare.md): augmentation and why grouping matters.
