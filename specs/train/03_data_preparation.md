# Data Preparation

Source: [`src/hybrax/train/prepare.py`](../../src/hybrax/train/prepare.py),
[`src/hybrax/train/augmentation.py`](../../src/hybrax/train/augmentation.py),
[`src/hybrax/train/training_data.py`](../../src/hybrax/train/training_data.py),
[`src/hybrax/train/controls_store.py`](../../src/hybrax/train/controls_store.py),
[`src/hybrax/train/controls.py`](../../src/hybrax/train/controls.py),
[`src/hybrax/train/validate.py`](../../src/hybrax/train/validate.py),
[`src/hybrax/train/model_api.py`](../../src/hybrax/train/model_api.py)

## Purpose

Turn a raw hybrax.format collection into a `prepared.json` artifact that training can
consume directly, and define the state, control, and measured-target layouts that
everything downstream uses. `prepare` does the data-side work once; `train` then
estimates scales, builds runtime controls, and assembles batches from the
prepared artifact.

## Design Rationale

- **Prepare once, train many.** Validation, process renaming, and control-source
  selection are deterministic and data-only, so they run in `prepare` and are
  frozen into `prepared.json`. The artifact carries hybrax.train provenance
  (`metadata["hybrax.train"]`: hybrax.format validation report, source/custom hashes,
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

- `load_raw_collection` accepts a hybrax.format `BioProcessCollection`, either
  in-memory or as a path to its JSON file. `case_id`/`organism`/`citation`
  (set when the collection is a published case study) are native fields on
  `BioProcessCollection` itself.
- `prepare_artifact` runs
  [`transform_process_collection`](02_cli_and_config.md#transform_process_collection)
  and the optional
  [`augment_state_values`](02_cli_and_config.md#augment_state_values) hook,
  validates, enforces the required and consistent continuous-control contract,
  and writes the artifact. Normally invoked via `hybrax prepare`.

### Prepared augmentation

Set `prepare.augmentation` to generate deterministic synthetic children after `transform_process_collection` and before final validation.
Each child is named `{parent}__aug_{index:03d}` and keeps its parent's controls, volume, events, and other process structure.
It receives an independently sampled measurement grid with the exact parent start and end times. The grid reserves `min_spacing_fraction` (default `0.1`) of the nominal grid interval as the minimum gap between adjacent points, then randomly allocates the remaining duration across the gaps. Preparation fails if the requested minimum is within four coarsest timestamp-resolution steps; stored gaps use a small relative rounding tolerance.

Modeled states follow three rules.

1. States configured in `noise_std` are evaluated on the child grid and receive
   additive Gaussian noise with that absolute standard deviation.
2. Other spline-backed states are evaluated on the same grid without noise.
3. Other states without splines keep their original observations and grid.

A `noise_std` value of zero performs time resampling without adding target noise.
Configured noise applies equally to identically-zero and nonzero parent traces.
Use `augment_state_values` when domain knowledge requires preserving selected
structural-zero traces.
For configured states, `initial_value_source` selects the child's initial value.
`measured` preserves the parent's real observation at the process start and fails when none exists.
`spline` uses `spline(t0)`, while `augmented` also applies augmentation noise at `t0`.
The setting may be one value for every listed state or an exact per-state mapping.
When `spline` or `augmented` extrapolates before a trace's first observation, augmentation emits a warning.

The second rule implicitly uses `spline(t0)` and emits the same extrapolation warning when its first observation is later.
A state used as a training target should normally be listed, otherwise the children supervise repeated noise-free spline trajectories for that target.
Mixed state grids remain valid because training constructs a union grid and a per-target measurement mask.

Reactor-medium component values are clipped at zero after additive noise.
Plots show the central 95% Gaussian interval (`spline ± 1.96 * noise_std`).
Reactor-medium band bounds are clipped at zero; process-variable bands remain
signed. At the initial time, the band collapses to zero width when
`initial_value_source` preserves the measured or spline value.
Augmentation warns when a reactor-medium component spline dips below zero over the
process interval, or when this happens for a process variable whose observations
are mostly nonnegative. Process-variable values are not clipped, so signed variables
remain signed.
Resampled modeled states are stored on children as observation-only series, while their generating splines remain available on the parent.
Controlled-variable splines remain on each child because simulation still needs them.

These resampled points are synthetic training observations.
They are not claims of new physical offline samples, and augmentation does not add sample-removal events or change the reactor volume.

### Scale estimation

The 11 data-derived `SCALE_*` axes normalize physical vectors so the ODE
integrates in O(1) space. The [`estimate_all_scales`](02_cli_and_config.md#estimate_all_scales)
hook returns them as an [`EstimatedScales`](../../src/hybrax/train/model_api.py). Bare
arrays become frozen `LinearScaler` fields (`SCL = RAW / scale`, the default);
a hook may return `AffineScaler(scale, offset)` for a value axis to opt into
`SCL = (RAW - offset) / scale`. Use an offset when an axis varies over a small
range around a much larger baseline, so the variation rather than the absolute
value sets the SCL magnitude. Controlled temperature and pH, or reactor volume
in fed-batch processes, are typical candidates; choose the offset near the
operating baseline and estimate the scale from the centered values.

For integrated-state axes, the adaptive solver applies its error tolerance in
SCL coordinates (`atol + rtol * abs(SCL)`). Centering a state near zero reduces
the relative-tolerance term and can therefore increase solver work; check step
counts and failed segments with the intended offset and tolerances.

The reaction module is the single source of truth (see
[01_design_rationale.md](01_design_rationale.md#2-scaled-scl-vs-physical-raw-space)).
A stateful reaction module also supplies `SCALE_latent`; `SCALE_state`,
`SCALE_modeled_V`, and `SCALE_integrated_state` are derived scaler properties.
Without the hook, every data-derived axis is a unit `LinearScaler` (no scaling).

Offsets are prohibited on the controlled Inflow/Outflow and modeled
biological/Inflow/Outflow rate axes. Derivatives are always transformed
offset-free (`dSCL/dt = dRAW/dt / scale`); using a value transform on a
derivative would spuriously subtract the offset. Integrated-state axes must use
the built-in elementwise `LinearScaler` or `AffineScaler`, because the state
scaler composes their scale/offset arrays in
the exact state layout below. Custom non-affine scalers are supported only on
non-state axes; accepting one on a state axis would silently reinterpret its
transform as affine.

The scale axes are:

- `SCALE_modeled_RMCs` (`n_RMC`): modeled species concentrations.
- `SCALE_modeled_PVs` (`n_modeled_PV`): modeled dynamic process variables.
- `SCALE_V_in_cumulative` (scalar): cumulative inflow volume and real volume.
- `SCALE_modeled_Inflows_cumulative` (`n_modeled_Inflow`): modeled cumulative
  inflow volumes.
- `SCALE_modeled_Outflows_cumulative` (`n_modeled_Outflow`): modeled cumulative
  removals.
- `SCALE_controlled_Inflows_cumulative` (`n_ctrl_Inflow`): controlled cumulative
  inflow volumes.
- `SCALE_controlled_Inflows_rates` (`n_ctrl_Inflow`): controlled inflow rates.
- `SCALE_controlled_Inflows_Cin` (`n_ctrl_Inflow × n_RMC`): feed compositions.
- `SCALE_controlled_Outflows_cumulative` (`n_ctrl_Outflow`): controlled
  cumulative removals.
- `SCALE_controlled_Outflows_rates` (`n_ctrl_Outflow`): controlled outflow
  rates.
- `SCALE_controlled_PVs` (`n_ctrl_PV`): controlled PV signals.
- `SCALE_modeled_Inflows_Cin` (`n_modeled_Inflow × n_RMC`): modeled feed
  compositions.
- `SCALE_modeled_BiologicalOde_rates` (`n_rates`): reaction rates.
- `SCALE_modeled_Inflows_rates` (`n_modeled_Inflow`): modeled inflow rates.
- `SCALE_modeled_Outflows_rates` (`n_modeled_Outflow`): modeled outflow rates.
- `SCALE_latent` (`n_latent`): integrated latent state for stateful modules.

### State and control layout

The physical **SCL state vector**:

```
SCL_state = [ modeled_RMCs | modeled_PVs | V_in_cumulative |
              modeled_Inflows_cumulative | modeled_Outflows_cumulative ]
            [ species | dynamic PVs | scalar volume ]
            [ modeled inflows | modeled outflows ]
SCL_integrated_state = [ SCL_state | SCL_latent ]
```

The solver advances `SCL_integrated_state`; stateless modules have an empty
`SCL_latent`. `SCALE_state` and `SCALE_integrated_state` concatenate both the
per-axis scales and offsets (including zero-width parts). The reaction module
reads each slice
through [`ReactionInputs`](04_reaction_and_loss.md#reactioninputs); modeled vs
controlled membership comes from hybrax.format's `RhsOde` (`name_modeled_*` /
`name_controlled_*`).

The **continuous controls** the reaction module reads at time `t` (in SCL
space) are ordered `[controlled Inflows | controlled Outflows | controlled
PVs]`. Inflow composition comes from feed media. Outflow retention remains raw
with 0 = removed and 1 = retained; it has no scale axis. Modeled Inflow
composition (`SCL_modeled_Inflows_Cin`) is also available.
Discrete bolus/sample events are **not** part of this vector — they are applied
as state jumps during the solve (below).

### Controls and event semantics

[`ControlsStore`](../../src/hybrax/train/controls_store.py) /
[`PerProcessControls`](../../src/hybrax/train/controls_store.py) hold per-process control
accessors built from the hybrax.format collection, of two kinds:

- **Continuous controlled feeds and process variables** → exact process-local
  piecewise-linear signals (`build_linear_payload`) evaluated by the RHS at each
  `t` (rates / cumulative / `Cin`). Valid hybrax.format splines are evaluated
  directly instead.
- **Discrete controlled bolus & sample events** → event arrays
  (`bolus_event_times/volumes/Cin`, `sample_event_times/volumes`) applied as
  **differentiable state jumps** at their event times during the segmented solve
  (`PresetTimeCallback`), sample-first-then-bolus.

`controls.active_jump_ts` comes only from `BioProcess.discrete_events`: genuine
vector-field discontinuities passed to the solver as `jump_ts` hints. Bolus and
sample state jumps are handled by the callback itself. See
[01_design_rationale.md](01_design_rationale.md#7-discrete-events-as-differentiable-state-jumps).

### Target selection

[`TrainingDataStore`](../../src/hybrax/train/training_data.py) indexes processes, measured
targets, and measurement times; [`PerProcessTrainingData`](../../src/hybrax/train/training_data.py)
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

- `validate_for_training(collection)` — hybrax.format per-process validation plus
  cross-process structural consistency (`validate_cross_process_consistency`,
  shared with hybrax.format's own `validate_for_publication`).
- `ensure_required_controls(...)` — enforce `prepare.required_control_names`.
- `summarize_process_semantics(process)` / `ensure_prepared_training_semantics`
  — structural diagnostics used during prepare; `strict_format_validation`
  promotes warnings to failures.

Both report-style checks return
`{"ok": bool, "messages": list[tuple[bool, str]]}` for each process. Each
message follows hybrax.format's
`"<PASS|FAIL|SKIP> <check_name>: <detail>"` convention. See the
[validation reference](../format/04_validation.md). Callers can select failures
with `[message for ok, message in entry["messages"] if not ok]` without parsing
the message text.

## Examples

```bash
hybrax prepare --config prepare-config.json --output-dir prepared
```

`prepare` writes standard-named files into `--output-dir`: `prepared.json` (the
artifact downstream consumes), `prepare_config.json` (resolved config +
provenance), and `prepare_diagnostics/<process>_controls.png` (when
`prepare.diagnostics`). `--overwrite` rewrites only those prepare-owned files, so
the dir can be shared with a `train`/`forward` run. Downstream `data.prepared` /
`forward --input` accept either this dir (resolving `prepared.json[.gz]` inside)
or a plain `prepared.json` file. Raw and prepared JSON inputs may contain
whole-line `//` comments after optional indentation. Generated artifacts use the
stdlib JSON encoder, including its `NaN`/`Infinity` extensions for non-finite
floats.

```jsonc
// prepare-config.json
{
  "prepare": {
    "raw_input": "data.json",
    "required_control_names": ["glucose_feed"]
  },
  // Defines transform_process_collection / augment_state_values.
  "custom_py": "custom.py"
}
```
