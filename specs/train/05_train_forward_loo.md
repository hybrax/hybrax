# Training, Forward & LOO

Source: [`src/hybrax/train/harness.py`](../../src/hybrax/train/harness.py),
[`src/hybrax/train/trainer.py`](../../src/hybrax/train/trainer.py),
[`src/hybrax/train/postprocessing.py`](../../src/hybrax/train/postprocessing.py),
[`src/hybrax/train/loo.py`](../../src/hybrax/train/loo.py),
[`src/hybrax/train/loo_metrics.py`](../../src/hybrax/train/loo_metrics.py),
[`src/hybrax/train/checkpointing.py`](../../src/hybrax/train/checkpointing.py),
[`src/hybrax/train/logging.py`](../../src/hybrax/train/logging.py)

## Purpose

The execution layer behind the `train`, `forward`, and `loo`
[subcommands](02_cli_and_config.md#subcommands): the training harness that fits a
hybrid model, the forward pass that re-simulates and exports a trained model, and
the leave-one-process-out cross-validation that wraps both.

## Design Rationale

- **Collection in, run directory out.** The harness takes a hybrax.format collection
  (or a prepared JSON), builds the wrapper from the hooks, partitions trainable
  leaves, and drives the optax loop — writing a self-contained, resumable
  [run directory](02_cli_and_config.md#run-directory-layout).
- **Prediction grids only serve exported processes.** Losses are evaluated for all
  selected processes, while only exported processes splice a prediction grid
  into the union time grid
  ([`build_union_time_grid`](04_reaction_and_loss.md#dense-grid-losses)).
- **LOO reuses the harness verbatim.** Each fold is an ordinary training run plus
  a forward on the holdout; folds are grouped so augmented children never leave
  their parent. A producer materializes collection-free runtime inputs before
  the worker processes begin.

## Training

### Entry points

```python
train_from_collection(collection, *, config=None, custom_py=None,
                      runtime_config=None, run_config=None,
                      custom_module=None) -> TrainHarnessResult
train_from_prepared_json(prepared_json, *, config=None, custom_py=None,
                        runtime_config=None) -> TrainHarnessResult
```

`config` is a [`TrainHarnessConfig`](../../src/hybrax/train/harness.py) — the flat,
harness-level mirror of the config sections (epochs, batch, optimizer, solver
tolerances, checkpoint cadence, logging format, and optional holdout processes).
Build it from a `RunConfig` with `train_harness_config_from_run_config(...)`, or
let the CLI do it. The CLI path (`hybrax train --config …`) is the normal way
in.

### What the loop does

- **Batching:** each epoch independently shuffles the selected processes, drops
  its incomplete final batch, and trains every full `batch_size` batch once.
- **Optimizer:** `adam`/`sgd` + `clip_by_global_norm(grad_clip_norm)`, built by
  `build_optimizer_for_run`. Override the LR with
  [`build_learning_rate`](02_cli_and_config.md#build_learning_rate) or the whole
  chain with [`build_optimizer`](02_cli_and_config.md#build_optimizer).
- **Gradient clipping** is applied to the **raw** gradient before Adam — this is
  why the loss is mean-aggregated (see
  [01_design_rationale.md](01_design_rationale.md#6-mean-loss-aggregation)).
- **Checkpointing** ([`checkpointing.py`](../../src/hybrax/train/checkpointing.py)):
  `checkpoint_every` is measured in epochs and may be fractional. The default
  automatic cadence is `max(5, ceil(epochs / 20))`, so it writes at most 20
  checkpoints. Explicit cadences are honored, all periodic checkpoints are
  retained, and a final checkpoint is mandatory. Checkpoints contain model and
  optimizer state only.
- **Logging** ([`logging.py`](../../src/hybrax/train/logging.py), `RunLogger`): every update
  writes a console row and `metrics.csv` row with epoch, batch, and sample
  counters. Epoch mean loss and training-only duration appear on epoch-end rows.
  Every checkpoint refreshes the run-level loss and global gradient-norm curves.
- **Holdout set (LOO only):** evaluated whenever a checkpoint is written. It is
  diagnostic and never drives optimizer updates.

### Result

[`TrainHarnessResult`](../../src/hybrax/train/harness.py) carries the `trained_wrapper`,
`mean_loss_by_step`, per-process / per-target loss series, the batch composition
per update, `updates_completed`, and timing (`compile_warmup_seconds`,
`step_time_seconds`).

## Forward evaluation

```python
forward_from_collection(collection, *, model_path, config=None, custom_py=None,
                       training_process_names=None,
                       prediction_process_names=None, run_config=None,
                       custom_module=None,
                       prediction_grid_n=200) -> ForwardResult
```

Loads a trained model and runs one forward ODE pass per selected process (no
gradient steps; `step = -1`). Driven by the CLI `forward` subcommand, which is
[fully config-driven](02_cli_and_config.md#hybrax-forward): the
`forward-config.json` carries a `models` list (one entry = single model, more =
ensemble) plus optional `data` and `output` blocks.

- [`ForwardConfig`](../../src/hybrax/train/harness.py) / [`ForwardResult`](../../src/hybrax/train/harness.py)
  carry the per-process losses and selected dense exports. Losses still cover
  every evaluated process. An omitted `ForwardConfig.target_source` inherits the
  model run's recorded training source; explicitly passing `"auto"` requests
  automatic resolution against the evaluation collection.
- `compute_dense_exports(trained_wrapper, store, process_names, *,
  solver_max_steps, solver_rtol, solver_atol, solver_use_jump_ts,
  prediction_grid_n=200)` runs one batched solve and returns per-process
  [`DenseProcessExport`](../../src/hybrax/train/postprocessing.py) trajectories: time,
  species, volume, cumulative modeled Inflows/Outflows, biological rates,
  separate physical modeled Inflow/Outflow rates, and auxiliary values.
  It is *the* single source of dense predictions for forward evaluation and
  final training exports, so exported predictions always match the training solve.
- All forward artifacts are written directly into `<output-dir>`.
  For an ensemble, per-model exports are averaged (`aggregate_dense_exports`)
  into `predictions.csv` plus a `predictions_std.csv`; each model also keeps its
  own `models/<name>/predictions.csv` and `losses.csv` there.
- `output.predictions` selects `none` (the default), non-augmented `parents`,
  or `all` evaluated processes. `none` skips dense prediction solves. When a
  rerun selects no processes, stale prediction CSVs are removed.
- Outputs are written by `export_predictions_csv` in
  [`postprocessing.py`](../../src/hybrax/train/postprocessing.py). Set `output.plots` to
  `true` to also write
  `<output-dir>/plots/<process>.png` for every exported process.
  Plotting requires `output.predictions` to be `parents` or `all` and is
  best-effort: rendering failures are logged without failing forward.
  Modeled flow columns use `B_<name>_cum` for cumulative values and
  `B_<name>_rate` for physical rates. Per-model and ensemble-mean Inflow
  values/rates remain non-negative; Outflow values/rates remain non-positive.
  `predictions_std.csv` contains non-negative standard deviations.

### Programmatic forward

For a loaded model, `model_predict(wrapper, config, collection)` is the simpler
path — it takes its solver settings from `config` and returns the same
`{process_name: DenseProcessExport}` mapping. See
[06_serialization_inspect.md](06_serialization_inspect.md#reconstruction).

`collection` is **evaluation data only**. The model itself is rebuilt from the
prepared collection *it* trained on, resolved from `model_path`'s run directory and
hash-verified before any hook runs
([`reconstruct_training`](06_serialization_inspect.md#reconstruction)) — so
`estimate_all_scales` never sees the evaluation data, and `training_process_names`
defaults to the selection the run recorded rather than anything derived from what
you pass in. The evaluation collection may hold entirely different processes, but it
must be compatible with the training data (same measured/modeled variables in the
same order) or the call fails with an explicit error. `model_predict` skips
reconstruction altogether and reuses the trained scales as-is.

## Leave-one/some-process-out cross-validation

`loo` is **config-driven**, like `train`: it takes the same run config plus an
optional [`loo`](../../src/hybrax/train/run_config.py) section. The CLI is
`hybrax loo --config loo-config.json` (`--output-dir` overrides `output.dir`);
`hybrax loo --resume <run_dir>` continues an interrupted run.

```jsonc
"loo": {
  // folds: each is a held-out `test` set + optional `train` set
  // (train omitted -> every process not in test) + optional `name` (labels the
  // fold dir / summary row). Omit per_fold_holdout_sets entirely for classic
  // leave-one-out (one fold per process).
  "per_fold_holdout_sets": [
    {"name": "high feed",  "test": ["proc_1", "proc_1b"]},
    {"name": "no feed",    "test": ["proc_2", "proc_3"], "train": ["proc_4"]}
  ],
  // How many folds to train at once (you own the RAM call).
  "parallel_folds": 4,
  // null derives a per-fold CPU-device count.
  "devices_per_fold": null
}
```

- **Folds** ([`resolve_folds`](../../src/hybrax/train/loo.py)): explicit `per_fold_holdout_sets`
  (each `name` becomes the fold's `folds/<slug>/` directory; without one the slug
  is derived from the test process names), or — when omitted — one fold per
  parent group. **Augmentation is respected everywhere**: holding out any member
  of an augmentation group (a parent + its `AugmentedBioProcess` children)
  excludes the whole group from train, so a synthetic child can't leak its parent
  or siblings into the fold (a pinned `train` that does so fails fast).
- **Parallel folds** ([`run_loo_cv`](../../src/hybrax/train/loo.py) → per-fold
  [`run_single_fold`](../../src/hybrax/train/loo.py)): each fold trains as **its own
  subprocess** (the JAX CPU device count is fixed per process). You set
  `parallel_folds` (default `1`, sequential) from what your RAM and CPU budget
  can hold. A JAX CPU device is not one CPU core: XLA may use several threads per
  device, so lower `parallel_folds` to reduce aggregate CPU use. With
  `devices_per_fold=null`, the orchestrator
  ([`compute_parallel_split`](../../src/hybrax/train/loo.py)) derives it as
  `n_cpu // parallel_folds`, capped at the smallest fold's effective batch
  (`min(train size, train.batch_size)`) since a fold cannot expose more host
  devices than its `pmap` batch. Set `devices_per_fold` to a positive integer to
  override that derived count. Worker processes are not CPU-pinned; the OS
  scheduler owns core placement. There is deliberately **no automatic RAM
  sizing**.
- **Holdout evaluation:** each fold uses its `test` set as a diagnostic holdout,
  evaluated at every periodic checkpoint and at the mandatory final checkpoint.
  It never drives the optimizer.
- **Runtime producer and workers:** the orchestrator first starts a short-lived
  producer subprocess. It loads the bundled prepared artifact, resolves folds,
  estimates scales, and writes `<output_dir>/runtime-artifact/`, then exits.
  Fold workers load only this strict, collection-free artifact; they do not load
  prepared JSON or estimate scales. The internal producer/worker CLI modes are
  implementation details, not user-facing commands.
- **Self-contained run dir**: a fresh run atomically claims a nonexistent exact
  `<output_dir>`; an existing directory requires `--overwrite` or `--resume`.
  The orchestrator writes true copies of `custom.py` and the prepared artifact
  plus a loadable `loo-config.json` with paths relative to the run dir. After the
  producer publishes and validates the runtime artifact, top-level `config.json`
  is atomically published with the expected artifact format and identity. The
  manifest is authoritative for runtime inputs and resolved folds. Editing or
  moving the source tree mid-run cannot desync workers.
- **Resume** (`--resume <run_dir>`): reloads the bundled `loo-config.json`
  verbatim (no overrides — `parallel_folds` etc. come from the run dir), validates
  the top-level artifact anchor and manifest, and re-runs folds with missing or
  incomplete completion records. A present malformed or mismatched per-fold
  artifact-identity/fold-ID binding fails loudly before that fold is deleted.
- **Per fold** → [`FoldResult`](../../src/hybrax/train/loo.py): train on the fold's `train`
  set, forward on its `train ∪ test`, write to `<output_dir>/folds/<slug>/` (own
  lightweight model-state checkpoints plus final `trained_wrapper.eqx`,
  `losses.csv`, optional configured predictions, `loss_curve.png`, and
  `grad_norm_curve.png`). The
  default `none` scope skips prediction exports; `parents` includes every evaluated
  original process, including the holdout. Aggregate metrics remain holdout-only.
  Each fold refreshes its run-level loss and gradient-norm plots at every
  checkpoint. Checkpoint directories do not contain prediction exports or plots.
- **Aggregation** ([`LOOResult`](../../src/hybrax/train/loo.py)): the orchestrator reads each
  fold's `losses.csv` back from disk and writes `loo_summary.csv`,
  `loo_aggregate.json`, and `loo_loss_curves.png` (holdout metrics averaged over
  each fold's `test` set; a single-fold run reports `NaN` for the cross-fold
  std). Post-hoc metrics honor the stored prediction scope and do not score
  augmented holdout children omitted by `parents`.

## Recreating predictions after training

Prediction exports are opt-in. Point `forward` at a self-contained run or
checkpoint directory and explicitly select `parents` or `all`:

```json
{
  "models": ["output/checkpoints/latest"],
  "output": {
    "dir": "output/forward",
    "predictions": "parents",
    "plots": true
  }
}
```

```bash
hybrax forward --config forward-config.json --overwrite
```

The checkpoint's bundled configuration, custom module, and prepared data are
reused. Add `data.prepared` to evaluate different prepared data, and
`data.processes` to select processes.

For a completed LOO run, run each held-out process through its fold's final
checkpoint. For example:

```json
{
  "models": ["output_loo/folds/DoE1_R1/checkpoints/latest"],
  "data": {"processes": ["DoE1_R1"]},
  "output": {
    "dir": "output_loo/folds/DoE1_R1/forward",
    "predictions": "parents"
  }
}
```

Repeat this config for each fold. For grouped holdouts, list every held-out
process. Use `"all"` instead of `"parents"` when augmented processes are also
wanted. This recreates fold-specific predictions without treating the fold
models as an ensemble.

With `output.plots` enabled, forward plots measured values and dense predictions
for every modeled reactor-medium component and process variable, plus every
inferred rate and `V_real`. Raw feed, bolus, and sampling events are separate
from cumulative modeled-feed trajectories. Figures include R², named and total
losses, and ensemble standard-deviation bands when available. For custom
figures, plot the exported columns directly; for example, this writes one plot
for a selected variable across all exported processes:

```python
import csv
from collections import defaultdict

import matplotlib.pyplot as plt

column = "c_biomass"
series = defaultdict(list)
with open("output/forward/predictions.csv", newline="") as file:
    for row in csv.DictReader(file):
        series[row["process"]].append((float(row["t"]), float(row[column])))

for process, values in series.items():
    t, y = zip(*values)
    plt.plot(t, y, label=process)
plt.xlabel("t")
plt.ylabel(column)
plt.legend()
plt.savefig(f"output/forward/{column}.png", dpi=150, bbox_inches="tight")
```
