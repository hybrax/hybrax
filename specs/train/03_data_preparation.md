# Data Preparation

Source: [`bp_train/prepare.py`](../bp_train/prepare.py),
[`bp_train/training_data.py`](../bp_train/training_data.py),
[`bp_train/controls_store.py`](../bp_train/controls_store.py),
[`bp_train/controls.py`](../bp_train/controls.py),
[`bp_train/validation.py`](../bp_train/validation.py),
[`bp_train/model_api.py`](../bp_train/model_api.py)

## Purpose

Turn a raw bp-format collection into a `prepared.json` artifact that training can
consume directly, and define the vector layouts that everything downstream uses:
the scaled state vector, the control vector, and the measured-target selection.
`prepare` does the data-side work once; `train` then estimates scales and
assembles batches from the prepared artifact.

## Design Rationale

- **Prepare once, train many.** Validation, process renaming, and control-source
  construction are deterministic and data-only, so they run in `prepare` and are
  frozen into `prepared.json`. The artifact carries bp-train provenance
  (`metadata["bp-train"]`: bp-format validation report, source/custom hashes,
  environment versions, prepared semantics) so a run is reproducible.
- **Controls are precomputed.** Discrete events (boluses, sampling) and
  continuous feeds are converted into piecewise-linear control sources at
  prepare time, on a refined dense grid, so the solver just evaluates them. See
  [event semantics](#controls-and-event-semantics).
- **Scales are estimated at train setup**, not baked into the data, so the same
  `prepared.json` can be trained with different scaling strategies via
  `estimate_all_scales`.

## Public API

### Loading and preparing

```python
load_raw_collection(input_json, *, case_study=None) -> BioProcessCollection
prepare_artifact(loaded_config: LoadedRunConfig, output_json) -> BioProcessCollection
```

- `load_raw_collection` accepts a bp-format collection file, an in-memory
  `BioProcessCollection`, or a `BenchmarkDataset` (extracts `case_study`, or the
  first one when `None`).
- `prepare_artifact` runs the prepare hooks
  ([`transform_process_collection`](02_cli_and_config.md#transform_process_collection),
  [`build_sample_acc_series`](02_cli_and_config.md#build_sample_acc_series)),
  validates, enforces the control contract (the reserved sample-accumulation
  control name `BP_TRAIN_SAMPLE_ACC_NAME` must be present and not user-supplied),
  and writes the artifact. Normally invoked via `bp-train prepare`.

### Scale estimation

The 13 `SCALE_*` axes (11 stored on the reaction module, plus the derived
`SCALE_state` and `SCALE_modeled_V` properties) normalize every vector so the
ODE integrates in O(1) space. The [`estimate_all_scales`](02_cli_and_config.md#estimate_all_scales)
hook returns them as an [`EstimatedScales`](../bp_train/model_api.py); they are
stored as frozen fields on the reaction module (the single source of truth — see
[01_design_rationale.md](01_design_rationale.md#2-scaled-scl-vs-physical-raw-space)).
Without the hook, every axis defaults to ones (no scaling).

| `SCALE_*` axis | Shape | Scales |
|---|---|---|
| `SCALE_modeled_RMCs` | `(n_RMC,)` | modeled species concentrations |
| `SCALE_modeled_PVs` | `(n_modeled_PV,)` | modeled (dynamic) process-variable states |
| `SCALE_V_in_cumulative` | scalar | cumulative inflow volume (and real volume) |
| `SCALE_modeled_FVCs_cumulative` | `(n_modeled_FVC,)` | per-modeled-feed cumulative volume |
| `SCALE_controlled_FVCs_cumulative` | `(n_ctrl_FVC,)` | per-controlled-feed cumulative volume |
| `SCALE_controlled_FVCs_rates` | `(n_ctrl_FVC,)` | per-controlled-feed flow rate |
| `SCALE_controlled_FVCs_Cin` | `(n_ctrl_FVC, n_RMC)` | controlled-feed composition |
| `SCALE_controlled_PVs` | `(n_ctrl_PV,)` | controlled PV signals (pH, DO, T, …) |
| `SCALE_modeled_FVCs_Cin` | `(n_modeled_FVC, n_RMC)` | modeled-feed composition |
| `SCALE_modeled_BiologicalOde_rates` | `(n_rates,)` | reaction rates |
| `SCALE_modeled_FVCs_rates` | `(n_modeled_FVC,)` | modeled-feed flow rates |

### State and control layout

The integrated **SCL state vector** (what the solver advances):

```
SCL_state = [ modeled_RMCs | modeled_PVs | V_in_cumulative | modeled_FVCs_cumulative ]
            └ species ─────┘└ dyn. PVs ──┘└ scalar ───────┘└ per modeled feed ─────┘
```

`SCALE_state` is the matching concatenation. The reaction module reads each slice
through [`ReactionInputs`](04_reaction_and_loss.md#reactioninputs); modeled vs
controlled membership comes from bp-format's `RhsOde` (`name_modeled_*` /
`name_controlled_*`).

The **control vector** (evaluated from the controls store at time `t`, not
integrated): the continuous controlled feeds (`cumulative`, `rates`, `Cin`),
controlled process variables, and the discrete "extras" — bolus dilution and the
sample-accumulation signal.

### Controls and event semantics

[`ControlsStore`](../bp_train/controls_store.py) /
[`PerProcessControls`](../bp_train/controls_store.py) hold per-process control
accessors built from the bp-format collection. At prepare time
[`controls.py`](../bp_train/controls.py) turns the recorded signals into:

- **continuous feeds** → a refined piecewise-linear dense grid
  (`build_dense_payload`),
- **boluses** → finite-width triangles (`build_bolus_sources`),
- **sampling** → a sample-accumulation ramp (`build_sample_acc_source_default`).

Events within `bolus_run_min_dt` overlap and stay simultaneously active; all
boundaries merge into one per-process `step_ts` array, exposed at runtime as
`controls.active_step_ts` and forwarded to the solver as `jump_ts`. See
[01_design_rationale.md](01_design_rationale.md#7-event-overlap-semantics-v1).

### Target selection

[`TrainingDataStore`](../bp_train/training_data.py) indexes processes, measured
targets, and measurement times; [`PerProcessTrainingData`](../bp_train/training_data.py)
holds one process's `y0`, measurement times, target values, and lengths. The
loss fits whichever targets `target_source` selects (`TARGET_SOURCES`):

| `target_source` | Targets |
|---|---|
| `process_variables` | measured (uncontrolled) process variables |
| `reactor_components` | measured reactor-medium components (species) |
| `combined` | both species and modeled PVs |
| `auto` | pick whichever the process group supports |

`store.name_measured` returns the active target names
(`[name_measured_RMCs | name_measured_PVs]`); these label the loss columns (see
[04_reaction_and_loss.md](04_reaction_and_loss.md#the-loss-module)).

### Validation

- `validate_collection(collection)` — bp-format validation plus post-transform
  checks.
- `ensure_required_controls(...)` — enforce `prepare.required_control_names`.
- `summarize_process_semantics(process)` / `ensure_prepared_training_semantics`
  — structural diagnostics used during prepare; `strict_bp_format_validation`
  promotes warnings to failures.

## Examples

```bash
bp-train prepare --config prepare-config.json --output prepared.json
```

```jsonc
// prepare-config.json
{
  "prepare": {
    "raw_input": "../../bp-format/examples/01_kittler_2022/02_bp_format_data_all/data.json",
    "case_study": "kittler_2022",
    "required_control_names": ["glucose_feed"],
    "bolus_run_min_dt": 0.05
  },
  "custom_py": "custom.py"   // transform_process_collection / build_sample_acc_series
}
```

A `transform_process_collection` hook that turns a fixed PV derivative into a
learnable `r_<pv>` rate is shown in
[examples/00_e2e_sim/custom.py](../examples/00_e2e_sim/custom.py).
