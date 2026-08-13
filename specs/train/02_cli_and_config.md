# CLI, Config & Hooks

Source: [`bp_train/cli.py`](../bp_train/cli.py),
[`bp_train/run_config.py`](../bp_train/run_config.py),
[`bp_train/utils.py`](../bp_train/utils.py),
[`bp_train/defaults.py`](../bp_train/defaults.py)

## Purpose

Everything you touch from the outside: the `bp-train` subcommands, the JSON run
config that drives them, and the `custom.py` hooks that let you swap in your own
reaction/loss modules, scales, and optimizer. The
[`custom.py` hooks reference](#custompy-hooks-reference) is the single, complete
list of every hook with its signature.

## The pipeline

```
raw bp_format collection
   │  bp-train prepare   (transform + validate + persist)
   ▼
prepare dir (prepared.json, prepare_config.json, prepare_diagnostics/)
   │  bp-train train     (fit reaction + loss modules → run directory)
   ▼
run directory (config.json, custom.py, metrics.csv, checkpoints/, model/)
   │  bp-train forward   (re-simulate, optionally export predictions)
   │  bp-train loo       (leave-one-process-out cross-validation)
   ▼
optional predictions.csv / loss curves / losses.csv / loo summary
```

All subcommands are config-driven (`--config run.json`), with documented CLI
options for selected overrides. See
[03_data_preparation.md](03_data_preparation.md) for prepare and
[05_train_forward_loo.md](05_train_forward_loo.md) for train/forward/loo
internals.

## Subcommands

### `bp-train prepare`

Transform a raw bp-format process collection into a prepared artifact.

| Flag | Required | Meaning |
|---|---|---|
| `--config` | yes | Path to a prepare run config JSON (needs a `prepare` section). |
| `--output-dir` | yes | Output directory; prepare writes `prepared.json`, `prepare_config.json`, `prepare_diagnostics/`, and — when augmentation is configured — `augmented-data.png` into it. |
| `--overwrite` | no | Overwrite an existing `prepared.json` in `--output-dir` (rewrites only prepare's own files; leaves any train/forward artifacts in the dir untouched). |

### `bp-train train`

Train one or more processes from a prepared artifact into a FAIR run directory.

| Flag | Default | Meaning |
|---|---|---|
| `--config` | — | Train run config JSON (required). |
| `--output-dir` | config's `output.dir` | Override the run directory. |
| `--overwrite` | off | Allow re-running into a completed run dir. |
| `--epochs` | config's `train.epochs` | Override the epoch count. |
| `--log-level` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR`. |

Console numeric formatting is config-only. `metrics.csv` records every optimizer
update.

### `bp-train forward`

Load one or more trained models and run one forward ODE pass per selected
process (no training); exports configured predictions and prints a loss table.

**Fully config-driven** — everything that used to be a flag (`--model`,
`--input`, `--process`, `--loss-csv`, `--timeseries-csv`) now lives in the
`forward-config.json`.

| Flag | Meaning |
|---|---|
| `--config` | Required. Path to a `forward-config.json` (schema below). |
| `--output-dir` | Override `output.dir`; default `<first model>/forward`. |
| `--overwrite` | Allow re-running into an output dir that already holds a `losses.csv`. |
| `--log-level` | `DEBUG`/`INFO`/`WARNING`/`ERROR`. |

The `forward-config.json`:

```jsonc
{
  // One or more self-contained run/checkpoint dirs. len 1 = single model,
  // >1 = ensemble. Bare path strings and {"name": ..., "path": ...} objects
  // are interchangeable; `name` defaults to the bundle's output.dir basename.
  "models": ["output_all"],

  // Optional. Forward on data other than each model's bundled prepared.json.
  "data": {
    "prepared": "prepared",        // prepared.json[.gz] or a prepare output dir
    "processes": ["run_1"]         // subset; omit for all
  },

  // Optional. predictions defaults to "none"; also accepts "parents" or "all".
  "output": { "dir": null, "predictions": "none" }
}
```

An ensemble (`len(models) > 1`) **requires** `data.prepared`, so every model
predicts on the same collection and the per-model outputs can be aligned.

Outputs under `--output-dir`:

```
predictions.csv          # selected-process mean; omitted for "none"
predictions_std.csv      # ensemble std; omitted for "none" or one model
losses.csv               # loss table of the first model
models/<name>/           # per model: losses.csv + optional predictions.csv
```

### `bp-train loo`

Leave-one/some-process-out cross-validation.
Config-driven: the run config is the same as `train` plus an optional `loo` section.
Each fold trains as its own subprocess; you choose how many run at once and how many JAX CPU devices each fold exposes. A short-lived producer loads the
prepared collection, resolves folds, estimates scales, and writes a strict,
collection-free `runtime-artifact/`; it exits before fold workers start. Workers
load only that artifact, never the prepared collection.

| Flag | Meaning |
|---|---|
| `--config` | Run config JSON (train schema + a `loo` section). Required unless `--resume`. |
| `--resume` | Continue an interrupted run from its output dir; reloads the bundled `loo-config.json` verbatim and re-runs only folds without a matching completed-fold record. Mutually exclusive with `--config`. |
| `--output-dir` | Override `output.dir` (the LOO run directory). |
| `--overwrite` | Replace an existing LOO output directory. |

The `loo` config section:

| Key | Meaning |
|---|---|
| `per_fold_holdout_sets` | List of `{"name"?: ..., "test": [...], "train"?: [...]}` folds. `train` omitted → every process not in `test`; `name` (optional) labels the `folds/<slug>/` dir. Omit the whole key → classic leave-one-out (one fold per process). Holding out any member of an augmentation group excludes the whole group (parent + children) from `train`. |
| `parallel_folds` | How many folds to train concurrently (default `1`, sequential). Worker processes are not CPU-pinned; the OS scheduler owns core placement. You set concurrency from what your RAM holds — there is no automatic RAM sizing. |
| `devices_per_fold` | Optional positive JAX CPU-device count per fold. Omitted (`null`) → `n_cpu // parallel_folds`, additionally capped at the smallest fold's effective batch (a fold cannot expose more host devices than its `pmap` batch without deadlocking). |
Each fold's holdout loss is evaluated whenever a checkpoint is written, including the mandatory final checkpoint.

A fresh LOO run requires the exact output directory not to exist. Use
`--overwrite` to replace an existing directory or `--resume` to continue it.

Outputs: the self-contained run dir (`loo-config.json`, `custom.py`, prepared
artifact, `config.json`), strict `runtime-artifact/`, per-fold
`folds/<slug>/`, and top-level `loo_summary.csv`, `loo_aggregate.json`, and
`loo_loss_curves.png`.
Top-level `config.json` anchors the expected artifact identity; the artifact
manifest is authoritative for runtime inputs and folds. Per-fold completion
records bind to that identity and fold ID. Missing or incomplete records are
rerun, while present malformed or mismatched identity records fail loudly.

## Run directory layout

A `train` run writes a self-contained FAIR directory at `output.dir`:

```
<output.dir>/
  config.json            # the resolved RunConfig (provenance)
  custom.py              # copied custom hooks (provenance)
  metrics.csv            # per-update loss, epoch, sample, and grad-norm history
  loss_curve.png         # final training/holdout loss history
  predictions.csv        # selected processes; omitted for "none"
  model/                 # final trained bundle
  checkpoints/
    latest/  step_00100/ …
        params.eqx          # trainable partition only
        opt_state.eqx       # optimizer state
        train_state.json    # step counter etc.
        config.json         # resolved config
        prepared.json.gz    # bundled data → self-contained
        custom.py           # bundled hooks
```

## `custom.py` hooks reference

A run can supply a `custom.py` module via `custom_py` in the config. (`forward`
finds it from the model bundle instead: the path recorded in the run's
`config.json` if it still exists on this machine, otherwise the `custom.py`
copied into the run dir.) Hooks are looked up by name with
[`get_hook(custom_module, "<name>", <default>)`](../bp_train/utils.py); a missing
hook falls back to its default (or to a no-op for the `None`-default hooks). The
module is imported with `load_custom_module`. A custom typed config object is
produced by an optional `get_custom_config(raw_custom, config)` and reaches every
hook as `config.custom`.

There are **7 `get_hook` hooks**. The optional `get_custom_config` setup adapter
is invoked separately before them. Stage = when a hook fires (`prepare` vs
`train`).

| Hook | Stage | Default |
|---|---|---|
| [`transform_process_collection`](#transform_process_collection) | prepare | `default_transform_process_collection` |
| [`augment_state_values`](#augment_state_values) | prepare | none |
| [`estimate_all_scales`](#estimate_all_scales) | train | none (no scaling) |
| [`build_reaction_module`](#build_reaction_module) | train | `default_build_reaction_module` |
| [`build_loss_module`](#build_loss_module) | train | `default_build_loss_module` |
| [`build_learning_rate`](#build_learning_rate) | train | none |
| [`build_optimizer`](#build_optimizer) | train | none |

### `transform_process_collection`

```python
def transform_process_collection(collection, config: RunConfig) -> collection
```
Mutate/replace the collection before scales and controls are built — e.g. swap a
fixed `biological_ode` derivative for an `r_<pv>` rate so the reaction module
learns it. Default applies `prepare.process_rename_map`.

### `augment_state_values`

```python
def augment_state_values(
    *,
    parent_name,
    child_name,
    state_name,
    times,
    base_values,
    augmented_values,
    config,
):
    return augmented_values
```

Optionally replace one configured state's generated child values. The hook runs
once per configured state and child after additive noise, initial-value handling,
and built-in reactor-medium nonnegative clipping. It must return a finite,
one-dimensional array with the same shape as `times`; its result is final and may intentionally
override those built-in constraints. `base_values` contains the parent spline on
the child grid, while `augmented_values` contains the fully processed built-in
values. The hook runs only when defined in `custom.py`.

For example, preserve exact-zero parent traces while retaining built-in behavior
for every other trace:

```python
import numpy as np


def augment_state_values(*, base_values, augmented_values, **_):
    if np.all(base_values == 0):
        return base_values
    return augmented_values
```

### `estimate_all_scales`

```python
def estimate_all_scales(
    runtime_data: RuntimeDataContext,
    target_names: list[str],
    config,
) -> EstimatedScales
```
Return the `SCALE_*` axes (as an [`EstimatedScales`](../bp_train/model_api.py))
used to normalize state/rate/control vectors. Bare arrays are promoted to
`LinearScaler` (`SCL = RAW / scale`, the default). Opt into affine scaling for
one value axis by returning `AffineScaler(scale, offset)` for that field:

```python
from bp_train import AffineScaler, EstimatedScales

return EstimatedScales(
    ...,
    SCALE_controlled_PVs=AffineScaler(pv_scale, pv_offset),
)
```

Rate axes (`SCALE_controlled_FVCs_rates`,
`SCALE_modeled_BiologicalOde_rates`, `SCALE_modeled_FVCs_rates`) reject non-zero
offsets. Runs once at train setup; the scalers are baked into the reaction
module. No default hook — when absent, every axis is a unit `LinearScaler` (no
scaling). See [03_data_preparation.md](03_data_preparation.md#scale-estimation).

`runtime_data` is a collection-free
[`RuntimeDataContext`](../bp_train/runtime_context.py). Its `training_data`,
`controls_store`, `rhs_ode`, `process_order`, and typed trace accessors expose
the numeric inputs used by scale hooks without rebuilding or retaining a
`BioProcessCollection`.

### `build_reaction_module`

```python
def build_reaction_module(*, target_names, process_names, config, seed,
                          runtime_context, **scale_kwargs) -> UserReactionModule
```
Construct the reaction module. `runtime_context` contains the prepared
`RuntimeDataContext` as `.data` and the resolved scalers as `.scales`.
`scale_kwargs` carries those same promoted `SCALE_*` scaler instances from
`estimate_all_scales`; pass them unchanged to `super().__init__(**scale_kwargs)`.
Default is `DefaultReactionModule` (a 2-layer MLP). See
[04_reaction_and_loss.md](04_reaction_and_loss.md#the-reaction-module).

### `build_loss_module`

```python
def build_loss_module(*, target_names, process_names, config, seed,
                      runtime_context) -> UserLossModule
```
Construct the loss module. `runtime_context` is the same resolved context passed
to the reaction-module hook. Default is `DefaultLossModule` (per-target MSE). See
[04_reaction_and_loss.md](04_reaction_and_loss.md#the-loss-module).

### `build_learning_rate`

```python
def build_learning_rate(custom_cfg, train_cfg, total_updates) -> float | optax.Schedule
```
Override the learning rate (e.g. a decay schedule). No default — `train.learning_rate`
is used as-is.

### `build_optimizer`

```python
def build_optimizer(custom_cfg, train_cfg) -> optax.GradientTransformation
```
Replace the whole optimizer chain — use `optax.masked` / `optax.multi_transform`
for per-leaf control (e.g. freezing some MLP layers). No default — the standard
`adam`/`sgd` + `clip_by_global_norm` chain is built from `train_cfg`.

## `run_config.json` schema

Top-level keys (unknown keys are rejected):
`data`, `custom_py`, `train`, `solver`, `checkpoint`, `output`, `logging`,
`prepare`, `custom`, `loo`. `prepare` needs the `prepare` section; `train`
needs the `data` section. All paths resolve relative to the config file's
directory.

**`data`** — [`DataConfig`](../bp_train/run_config.py)

| Field | Type / default | Meaning |
|---|---|---|
| `prepared` | path (required) | The prepared collection to train on: a `prepared.json[.gz]` file or a prepare `--output-dir` (resolves `prepared.json` inside). |
| `processes` | tuple\|null = null | Subset of process names; null = all. |
| `targets` | tuple\|null = null | Explicit target names; null = derived. |
| `target_source` | `auto` | `auto`/`process_variables`/`reactor_components`/`combined`. |

**`train`** — [`TrainConfig`](../bp_train/run_config.py)

| Field | Default | Meaning |
|---|---|---|
| `epochs` | 5 (>0) | Full shuffled traversals of selected processes, dropping each epoch's incomplete final batch. |
| `seed` | 0 | Init seed. |
| `optimizer` | `adam` | `adam`/`sgd`. |
| `learning_rate` | 1e-3 (>0) | Base LR. |
| `grad_clip_norm` | 1000.0 (≥0) | `clip_by_global_norm` threshold. |
| `batch_size` | null | Processes per update; null = all. Values larger than the selected process count are invalid. |
| `shuffle` | true | Independently shuffle each epoch. |
| `batch_seed` | null | Batch-index seed. |
| `devices` | 1 | CPU devices to shard over; `"max"` = `min(n_proc, n_cpu)`. |
| `allow_stateful_models` | false | Allow reaction modules with latent state (`n_latent > 0`); otherwise they fail fast. |

**`solver`** — [`SolverConfig`](../bp_train/run_config.py)

| Field | Default | Meaning |
|---|---|---|
| `max_steps` | 2048 (>0) | Max solver steps per solve. |
| `rtol` | 1e-5 (>0) | Relative tolerance. |
| `atol` | 1e-7 (>0) | Absolute tolerance. |
| `jump_ts` | true | Pass `BioProcess.discrete_events` vector-field discontinuities as `jump_ts` hints. |

**`checkpoint`** — [`CheckpointConfig`](../bp_train/run_config.py)

| Field | Default | Meaning |
|---|---|---|
| `every` | null | Periodic checkpoint cadence in epochs. Null selects `max(5, ceil(epochs / 20))`, giving at most 20 checkpoints; explicit fractional values are supported, and 0 disables periodic writes but not the mandatory final checkpoint. |

**`output`** — [`OutputConfig`](../bp_train/run_config.py): `dir` (default
`output`) and `predictions` (`none` by default; also `parents` or `all`).

**`logging`** — [`LoggingConfig`](../bp_train/run_config.py): `decimals` (4).

**`prepare`** — [`PrepareConfig`](../bp_train/run_config.py)

| Field | Default | Meaning |
|---|---|---|
| `raw_input` | path (required) | Raw bp-format `BioProcessCollection` JSON. |
| `augmentation` | null | Persist deterministic synthetic `AugmentedBioProcess` children; see below. |
| `strict_bp_format_validation` | false | Fail on bp-format validation warnings. |
| `required_control_names` | () | Continuous controlled-feed/PV names that must exist (tuple, or per-process dict). |
| `require_consistent_controls` | true | All processes share the same continuous controlled-feed/PV names. |
| `initial_grid_points` | 16 (>0) | Starting dense control-grid resolution. |
| `max_rel_error` | 1e-4 (>0) | Control-grid refinement tolerance. |
| `max_refinement_rounds` | 8 (≥0) | Refinement round cap. |
| `process_rename_map` | {} | Old→new process-name map (used by the default transform). |
| `diagnostics` | true | Write per-process control diagnostic plots into `prepare_diagnostics/`; rendering failures only warn. |

The `prepare.augmentation` object has these fields.

| Field | Default | Meaning |
|---|---|---|
| `seed` | 0 | Deterministic child-grid and value seed. |
| `n_children_per_process` | required (>0) | Number of children per non-augmented parent. |
| `n_time_points` | required (>=2) | Points on each child grid, including exact endpoints. |
| `min_spacing_fraction` | `0.1` (0 < value <= 1) | Fraction of the nominal interval `(end - start) / (n_time_points - 1)` reserved for each child-grid gap; `1` gives an even grid. Requests within four timestamp-resolution steps fail; stored gaps use a small relative rounding tolerance. |
| `noise_std` | required, nonempty mapping | Absolute additive-noise standard deviation per modeled state, in that state's physical units. Values must be finite and nonnegative; `0` performs time resampling without target noise. Mapping order controls plot column order. |
| `initial_value_source` | `measured` | `measured`, `spline`, or `augmented`, applied to every `noise_std` state; alternatively, an exact per-state mapping with the same keys. |

**`custom_py`** — path to `custom.py`. **`custom`** — free-form object passed to
hooks as `config.custom` (typed via `get_custom_config` or a permissive default).

## JSON comments

Config inputs accept whole lines whose first non-whitespace characters are `//`.
Inline comments, block comments, trailing commas, and other JSON5-only syntax
remain invalid. The stdlib parser and generated files retain its
`NaN`/`Infinity` extensions for non-finite floats.

## Device pooling

```json
{ "train": { "devices": "max" } }
```
Use an integer instead of `"max"` to request a fixed device count, or set
`BP_TRAIN_DEVICES=8 bp-train train --config …` (the environment wins). Resolved
before JAX initializes; default 1. See
[01_design_rationale.md](01_design_rationale.md#9-opt-in-multi-core-device-pooling).

## Example: `train-config.json`

From [examples/00_e2e_sim/train-config.json](../examples/00_e2e_sim/train-config.json):

```json
{
  "data": {
    "prepared": "prepared",
    "target_source": "combined"
  },
  "custom_py": "custom.py",
  "train": {
    "epochs": 1,
    "seed": 0,
    "devices": "max"
  },
  "solver": {
    "max_steps": 4096,
    "rtol": 1e-05,
    "atol": 1e-07
  },
  "checkpoint": {
    "every": 1.0
  }
}
```
with [examples/00_e2e_sim/forward-config.json](../examples/00_e2e_sim/forward-config.json):

```json
{
  "models": ["output_all"],
  "output": { "predictions": "parents" }
}
```

Run them with:
```bash
bp-train prepare --config prepare-config.json --output-dir prepared
bp-train train   --config train-config.json
bp-train forward --config forward-config.json
bp-train loo     --config loo-config.json
```
