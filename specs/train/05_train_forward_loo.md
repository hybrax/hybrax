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

## Leave-one-process-out cross-validation

```python
run_loo_cv(collection, *, config: LOOConfig, custom_py=None,
           runtime_config=None) -> LOOResult
```

[`LOOConfig`](../bp_train/loo.py) wraps a `base_train_config`, an `output_dir`,
optional `selected_holdouts` (parent names; `None` = all), and plot/prediction
toggles. The CLI `loo` subcommand drives it (`--holdouts` selects a subset; a
single name runs one fold for cluster fan-out).

- **Fold grouping:** each non-augmented `BioProcess` is a fold group; every
  `AugmentedBioProcess` travels with its parent, so children are never held out
  alone.
- **Per fold** ([`run_loo_fold`](../bp_train/loo.py) → [`FoldResult`](../bp_train/loo.py)):
  train on the other groups, forward on the holdout, write to
  `<output_dir>/folds/<holdout_parent>/` (its own `checkpoints/`,
  `trained_wrapper.eqx`, plots, predictions).
- **Aggregation** ([`LOOResult`](../bp_train/loo.py)): a summary CSV + aggregate
  JSON. Metrics come from
  [`compute_per_process_metrics`](../bp_train/loo_metrics.py) and
  [`compute_aggregated_metrics`](../bp_train/loo_metrics.py); `DEFAULT_METRICS`
  is `{r2, nmae, mae, rmse}`.

## Examples

```bash
# train, then re-simulate + export, then cross-validate
bp-train train   --config examples/01_kittler_2022/vanilla/train-config.json
bp-train forward --model examples/01_kittler_2022/vanilla/output \
                 --timeseries-csv predictions.csv
bp-train loo     --input prepared.json --custom custom.py --holdouts batch_001
```

The FBA-surrogate fold setup (`SRfba` reaction module + Kendall loss) is in
[examples/01_kittler_2022/fba_hyb/](../examples/01_kittler_2022/fba_hyb/); a
structured dense-grid run is in
[examples/12_martens_2025_expanded/structured/](../examples/12_martens_2025_expanded/structured/).
