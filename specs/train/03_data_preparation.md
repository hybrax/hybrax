# Data Preparation

Source: [`bp_train/prepare.py`](../bp_train/prepare.py),
[`bp_train/augmentation.py`](../bp_train/augmentation.py),
[`bp_train/training_data.py`](../bp_train/training_data.py),
[`bp_train/controls_store.py`](../bp_train/controls_store.py),
[`bp_train/controls.py`](../bp_train/controls.py),
[`bp_train/validation.py`](../bp_train/validation.py),
[`bp_train/model_api.py`](../bp_train/model_api.py)

## Purpose

Turn a raw bp-format collection into a `prepared.json` artifact that training can
consume directly, and define the state, control, and measured-target layouts that
everything downstream uses. `prepare` does the data-side work once; `train` then
estimates scales, builds runtime controls, and assembles batches from the
prepared artifact.

## Design Rationale

- **Prepare once, train many.** Validation, process renaming, and control-source
  selection are deterministic and data-only, so they run in `prepare` and are
  frozen into `prepared.json`. The artifact carries bp-train provenance
  (`metadata["bp-train"]`: bp-format validation report, source/custom hashes,
  environment versions, prepared semantics) so a run is reproducible.
- **Controls are built at runtime.** `ControlsStore` refines continuous feeds
  into piecewise-linear dense signals the RHS evaluates at each `t`; discrete
  events (controlled boluses, sampling) become event arrays applied as **state
  jumps** during the solve. See [event semantics](#controls-and-event-semantics).
- **Augmentation is persisted.** Each configured synthetic variant is stored as a complete `AugmentedBioProcess` child, so repeated training uses the same observations and provenance.
- **Scales are estimated at train setup**, not baked into the data, so the same
  `prepared.json` can be trained with different scaling strategies via
  `estimate_all_scales`.

## Public API

### Loading and preparing

```python
load_raw_collection(input_json) -> BioProcessCollection
prepare_artifact(loaded_config: LoadedRunConfig, output_dir, *, overwrite=False) -> BioProcessCollection
```

- `load_raw_collection` accepts a bp-format `BioProcessCollection` (file or
  in-memory) or a `CaseStudy` (file or in-memory); a `CaseStudy`'s processes
  are wrapped into a collection, with the case identity kept in `metadata`.
- `prepare_artifact` runs the prepare hooks
  ([`transform_process_collection`](02_cli_and_config.md#transform_process_collection),
  [`augment_state_values`](02_cli_and_config.md#augment_state_values)),
  validates, enforces the required and consistent continuous-control contract,
  and writes the artifact. Normally invoked via `bp-train prepare`.

### Prepared augmentation

Set `prepare.augmentation` to generate deterministic synthetic children after `transform_process_collection` and before final validation.
Each child is named `{parent}__aug_{index:03d}` and keeps its parent's controls, volume, events, and other process structure.
It receives an independently sampled measurement grid with the exact parent start and end times. The grid reserves `min_spacing_fraction` (default `0.1`) of the nominal grid interval as the minimum gap between adjacent points, then randomly allocates the remaining duration across the gaps. Preparation fails if the requested minimum is within four coarsest timestamp-resolution steps; stored gaps use a small relative rounding tolerance.

Modeled states follow three rules.

1. Listed spline-backed states are evaluated on the child grid and noised.
2. Unlisted spline-backed states are evaluated on the same grid without noise.
3. Unlisted states without splines keep their original observations and grid.

For listed states, `initial_value_source` selects the child's initial value.
`measured` preserves the parent's real observation at the process start and fails when none exists.
`spline` uses `spline(t0)`, while `augmented` also applies augmentation noise at `t0`.
The setting may be one value for every listed state or an exact per-state mapping.
When `spline` or `augmented` extrapolates before a trace's first observation, augmentation emits a warning.

The second rule implicitly uses `spline(t0)` and emits the same extrapolation warning when its first observation is later.
A state used as a training target should normally be listed, otherwise the children supervise repeated noise-free spline trajectories for that target.
Mixed state grids remain valid because training constructs a union grid and a per-target measurement mask.

Reactor-medium component values are clipped at zero after built-in or custom augmentation.
For additive noise, the plot band shows the nominal Gaussian standard deviation before this clipping, so it can extend below zero.
At the initial time, the band collapses to zero when `initial_value_source` preserves the measured or spline value.
Augmentation warns when a reactor-medium component spline dips below zero over the process interval, or when this happens for a process variable whose observations are mostly nonnegative.
Process-variable values are not clipped by the final reactor-medium safeguard.
The built-in `add` model nevertheless clips every listed state at zero; use `mult` or a custom hook for variables that may legitimately be negative.
The `mult` model scales noise by the mean nonzero absolute spline value and preserves each value's sign.
With `residual_scope: process`, each parent supplies its own spline-residual RMS.
With `residual_scope: variable`, residual squares are pooled across parent observations of a state before taking one shared RMS.
Identically zero observed traces are excluded from that estimate so they do not dilute the error scale, but the resulting shared RMS is still applied to them.
The `mult` model uses a deterministic reference magnitude from the parent spline at its observation times, so every child and the augmentation plot use the same relative noise scale.
Resampled modeled states are stored on children as observation-only series, while their generating splines remain available on the parent.
Controlled-variable splines remain on each child because simulation still needs them.

These resampled points are synthetic training observations.
They are not claims of new physical offline samples, and augmentation does not add sample-removal events or change the reactor volume.

### Scale estimation

The 11 data-derived `SCALE_*` axes normalize physical vectors so the ODE
integrates in O(1) space. The [`estimate_all_scales`](02_cli_and_config.md#estimate_all_scales)
hook returns them as an [`EstimatedScales`](../bp_train/model_api.py); they are
stored as frozen fields on the reaction module (the single source of truth — see
[01_design_rationale.md](01_design_rationale.md#2-scaled-scl-vs-physical-raw-space)).
A stateful reaction module also supplies `SCALE_latent`; `SCALE_state`,
`SCALE_modeled_V`, and `SCALE_integrated_state` are derived properties. Without
the hook, every data-derived axis defaults to ones (no scaling).

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
| `SCALE_latent` | `(n_latent,)` | integrated latent state (set by a stateful module) |

### State and control layout

The physical **SCL state vector**:

```
SCL_state = [ modeled_RMCs | modeled_PVs | V_in_cumulative | modeled_FVCs_cumulative ]
            └ species ─────┘└ dyn. PVs ──┘└ scalar ───────┘└ per modeled feed ─────┘
SCL_integrated_state = [ SCL_state | SCL_latent ]
```

The solver advances `SCL_integrated_state`; stateless modules have an empty
`SCL_latent`. `SCALE_state` and `SCALE_integrated_state` are the matching
concatenations. The reaction module reads each slice
through [`ReactionInputs`](04_reaction_and_loss.md#reactioninputs); modeled vs
controlled membership comes from bp-format's `RhsOde` (`name_modeled_*` /
`name_controlled_*`).

The **continuous controls** the reaction module reads at time `t` (in SCL
space): the continuous controlled feeds (`cumulative`, `rates`, `Cin`),
controlled process variables, and the modeled-feed composition (`modeled_FVCs_Cin`).
Discrete bolus/sample events are **not** part of this vector — they are applied
as state jumps during the solve (below).

### Controls and event semantics

[`ControlsStore`](../bp_train/controls_store.py) /
[`PerProcessControls`](../bp_train/controls_store.py) hold per-process control
accessors built from the bp-format collection, of two kinds:

- **Continuous controlled feeds** → a refined piecewise-linear dense signal
  (`build_dense_payload`) the RHS evaluates at each `t` (rates / cumulative /
  `Cin`).
- **Discrete controlled bolus & sample events** → event arrays
  (`bolus_event_times/volumes/Cin`, `sample_event_times/volumes`) applied as
  **differentiable state jumps** at their event times during the segmented solve
  (`PresetTimeCallback`), sample-first-then-bolus.

`controls.active_jump_ts` comes only from `BioProcess.discrete_events`: genuine
vector-field discontinuities passed to the solver as `jump_ts` hints. Bolus and
sample state jumps are handled by the callback itself. See
[01_design_rationale.md](01_design_rationale.md#7-discrete-events-as-differentiable-state-jumps).

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
bp-train prepare --config prepare-config.json --output-dir prepared
```

`prepare` writes standard-named files into `--output-dir`: `prepared.json` (the
artifact downstream consumes), `prepare_config.json` (resolved config +
provenance), and `prepare_diagnostics/<process>_controls.png` (when
`prepare.diagnostics`). `--overwrite` rewrites only those prepare-owned files, so
the dir can be shared with a `train`/`forward` run. Downstream `data.prepared` /
`forward --input` accept either this dir (resolving `prepared.json[.gz]` inside)
or a plain `prepared.json` file.

```jsonc
// prepare-config.json
{
  "prepare": {
    "raw_input": "../../bp-format/examples/01_kittler_2022/02_bp_format_data_all/data.json",
    "required_control_names": ["glucose_feed"]
  },
  "custom_py": "custom.py"   // transform_process_collection / augment_state_values
}
```

A `transform_process_collection` hook that turns a fixed PV derivative into a
learnable `r_<pv>` rate is shown in
[examples/00_e2e_sim/custom.py](../examples/00_e2e_sim/custom.py).
