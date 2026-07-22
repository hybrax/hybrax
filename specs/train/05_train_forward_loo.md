# Training, Forward & LOO

Source: [`bp_train/harness.py`](../bp_train/harness.py),
[`bp_train/trainer.py`](../bp_train/trainer.py),
[`bp_train/postprocessing.py`](../bp_train/postprocessing.py),
[`bp_train/loo.py`](../bp_train/loo.py),
[`bp_train/loo_metrics.py`](../bp_train/loo_metrics.py),
[`bp_train/checkpointing.py`](../bp_train/checkpointing.py),
[`bp_train/logging.py`](../bp_train/logging.py)

## Purpose

The execution layer behind the `train`, `forward`, and `loo`
[subcommands](02_cli_and_config.md#subcommands): the training harness that fits a
hybrid model, the forward pass that re-simulates and exports a trained model, and
the leave-one-process-out cross-validation that wraps both.

## Design Rationale

- **Collection in, run directory out.** The harness takes a bp-format collection
  (or a prepared JSON), builds the wrapper from the hooks, partitions trainable
  leaves, and drives the optax loop — writing a self-contained, resumable
  [run directory](02_cli_and_config.md#run-directory-layout).
- **One solve serves training, plotting, and CSV export.** Forward export reuses
  the same batched solve as training, splicing a prediction grid into the union
  time grid ([`build_union_time_grid`](04_reaction_and_loss.md#dense-grid-losses)).
- **LOO reuses the harness verbatim.** Each fold is an ordinary training run plus
  a forward on the holdout; folds are grouped so augmented children never leave
  their parent.

## Training

### Entry points

```python
train_from_collection(collection, *, config=None, custom_py=None,
                      runtime_config=None, run_config=None,
                      custom_module=None) -> TrainHarnessResult
train_from_prepared_json(prepared_json, *, config=None, custom_py=None,
                        runtime_config=None) -> TrainHarnessResult
```

`config` is a [`TrainHarnessConfig`](../bp_train/harness.py) — the flat,
harness-level mirror of the config sections (steps, batch, optimizer, solver
tolerances, checkpoint cadence/retention, logging cadence, monitor processes).
Build it from a `RunConfig` with `train_harness_config_from_run_config(...)`, or
let the CLI do it. The CLI path (`bp-train train --config …`) is the normal way
in.

### What the loop does

- **Batching:** `batch_size` processes per step (default all), shuffled when
  `shuffle_batches` is set; ragged samples are padded per batch and masked.
- **Optimizer:** `adam`/`sgd` + `clip_by_global_norm(grad_clip_norm)`, built by
  `build_optimizer_for_run`. Override the LR with
  [`build_learning_rate`](02_cli_and_config.md#build_learning_rate) or the whole
  chain with [`build_optimizer`](02_cli_and_config.md#build_optimizer).
- **Gradient clipping** is applied to the **raw** gradient before Adam — this is
  why the loss is mean-aggregated (see
  [01_design_rationale.md](01_design_rationale.md#6-mean-loss-aggregation)).
- **Checkpointing** ([`checkpointing.py`](../bp_train/checkpointing.py)): every
  `checkpoint_every` steps writes a self-contained checkpoint dir; retention is
  `best+latest` (prune step dirs) or `all`. Plots render on a background worker
  so they don't block training.
- **Logging** ([`logging.py`](../bp_train/logging.py), `RunLogger`): a per-step
  console table (header re-emitted every `header_every` rows) plus `metrics.csv`
  in the run dir; an optional `metrics_jsonl`. Each named loss term is its own
  column (see [04_reaction_and_loss.md](04_reaction_and_loss.md#where-terms-show-up)).
- **Monitor set (optional):** `monitor_processes` are evaluated every
  `log_every` steps as a diagnostic validation loss — never drives updates.

### Result

[`TrainHarnessResult`](../bp_train/harness.py) carries the `trained_wrapper`,
`mean_loss_by_step`, per-process / per-target loss series, the batch composition
per step, and timing (`compile_warmup_seconds`, `step_time_seconds`).

### Resuming

```bash
bp-train train --resume output --steps 2000
```
`resume_run` reloads `checkpoints/latest` (params + `opt_state.eqx` +
`train_state.json`), appends to `metrics.csv`, and continues; `--steps` may
extend the original target. The optimizer is rebuilt structurally identically via
the shared `build_optimizer_for_run` so the loaded `opt_state` matches.

## Forward evaluation

```python
forward_from_collection(collection, *, model_path, config=None, custom_py=None,
                       runtime_config=None, training_process_names=None,
                       run_config=None, custom_module=None,
                       prediction_grid_n=200) -> ForwardResult
```

Loads a trained model and runs one forward ODE pass per selected process (no
gradient steps; `step = -1`). Driven by the CLI `forward` subcommand with a
`forward_config.json` (`models` list → single model or ensemble) or `--model`.

- [`ForwardConfig`](../bp_train/harness.py) / [`ForwardResult`](../bp_train/harness.py)
  carry the per-process losses and dense exports.
- `compute_dense_exports(trained_wrapper, store, collection, process_names, *,
  solver_max_steps, solver_rtol, solver_atol, solver_use_jump_ts,
  prediction_grid_n=200)` runs one batched solve and returns per-process
  [`DenseProcessExport`](../bp_train/postprocessing.py) trajectories
  (time, species, volume, rates, auxiliary).
- Outputs ([`postprocessing.py`](../bp_train/postprocessing.py)):
  `plot_process_simulations` (per-process fit plots with measurement overlays and
  bolus annotations), `export_predictions_csv` (`predictions.csv` /
  `--timeseries-csv` merged across processes), and a `losses.csv` loss table.

## Leave-one/some-process-out cross-validation

`loo` is **config-driven**, like `train`: it takes the same run config plus an
optional [`loo`](../bp_train/run_config.py) section. The CLI is
`bp-train loo --config loo-config.json` (`--output-dir` overrides `output.dir`);
`bp-train loo --resume <run_dir>` continues an interrupted run.

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
  "parallel_folds": 4,     // how many folds to train at once (you own the RAM call)
  "devices_per_fold": null, // null derives a per-fold CPU-device count
  "monitor_every": 50      // trace each fold's holdout (test) loss every N steps
}
```

- **Folds** ([`resolve_folds`](../bp_train/loo.py)): explicit `per_fold_holdout_sets`
  (each `name` becomes the fold's `folds/<slug>/` directory; without one the slug
  is derived from the test process names), or — when omitted — one fold per
  parent group. **Augmentation is respected everywhere**: holding out any member
  of an augmentation group (a parent + its `AugmentedBioProcess` children)
  excludes the whole group from train, so a synthetic child can't leak its parent
  or siblings into the fold (a pinned `train` that does so fails fast).
- **Parallel folds** ([`run_loo_cv`](../bp_train/loo.py) → per-fold
  [`run_single_fold`](../bp_train/loo.py)): each fold trains as **its own
  subprocess** (the JAX CPU device count is fixed per process). You set
  `parallel_folds` (default `1`, sequential) from what your RAM and CPU budget
  can hold. A JAX CPU device is not one CPU core: XLA may use several threads per
  device, so lower `parallel_folds` to reduce aggregate CPU use. With
  `devices_per_fold=null`, the orchestrator
  ([`compute_parallel_split`](../bp_train/loo.py)) derives it as
  `n_cpu // parallel_folds`, capped at the smallest fold's effective batch
  (`min(train size, train.batch_size)`) since a fold cannot expose more host
  devices than its `pmap` batch. Set `devices_per_fold` to a positive integer to
  override that derived count. Worker processes are not CPU-pinned; the OS
  scheduler owns core placement. There is deliberately **no automatic RAM
  sizing**.
- **Holdout monitoring** (`loo.monitor_every`): each fold automatically uses its
  `test` set as a diagnostic monitor (`monitor_label="holdout"`), evaluated every
  `monitor_every` steps during that fold's training (`null` → the `logging.every`
  cadence). It never drives the optimizer — it just traces the holdout loss while
  the fold trains.
- **Self-contained run dir**: the orchestrator writes true copies of `custom.py`
  and the prepared artifact into `<output_dir>/` plus a loadable
  `loo-config.json` with paths relative to the run dir. Every worker (and
  `--resume`) loads **only** from there, so editing or moving the source tree
  mid-run can't desync folds.
- **Resume** (`--resume <run_dir>`): reloads the bundled `loo-config.json`
  verbatim (no overrides — `parallel_folds` etc. come from the run dir) and
  re-runs only folds missing a `losses.csv`, then re-aggregates.
- **Per fold** → [`FoldResult`](../bp_train/loo.py): train on the fold's `train`
  set, forward on its `train ∪ test`, write to `<output_dir>/folds/<slug>/` (own
  `checkpoints/`, `trained_wrapper.eqx`, `losses.csv`, predictions, plots).
- **Aggregation** ([`LOOResult`](../bp_train/loo.py)): the orchestrator reads each
  fold's `losses.csv` back from disk and writes `loo_summary.csv` +
  `loo_aggregate.json` (holdout metrics averaged over each fold's `test` set;
  a single-fold run reports `NaN` for the cross-fold std).

## Examples

```bash
# train, then re-simulate + export, then cross-validate
bp-train train   --config examples/01_kittler_2022/vanilla/train-config.json
bp-train forward --model examples/01_kittler_2022/vanilla/output \
                 --timeseries-csv predictions.csv
bp-train loo     --config examples/01_kittler_2022/fba_hyb/loo-config.json
```

The FBA-surrogate fold setup (`SRfba` reaction module + Kendall loss) is in
[examples/01_kittler_2022/fba_hyb/](../examples/01_kittler_2022/fba_hyb/); a
structured dense-grid run is in
[examples/12_martens_2025_expanded/structured/](../examples/12_martens_2025_expanded/structured/).
