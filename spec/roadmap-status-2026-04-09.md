# BP-Train Roadmap & Code Review (as of 2026-04-09)

This document is a section-by-section walk through `spec/v1-detailed-spec.md`,
checking what is actually implemented in the current package, what is partial,
what is missing, what is broken or sloppy, and what tests exist or are missing.
Where the answer is "implemented", a concrete file path / line range is given
so the claim can be re-checked.

Compared to the prior status doc (`roadmap-status-2026-03-30.md`):

- The collection-level training harness, batch index stream, batched JIT
  train-step, and `RunLogger`-based telemetry pipeline now exist and replace
  the "minimal single-process primitive" the prior doc described.
- The `train` CLI subcommand exists and is wired through to
  `train_from_collection`.
- The duplicate `prepared.json` load is fixed (today's CLI loads the
  collection exactly once).
- The new logging redesign (`bp_train/logging.py` + `RunLogger` + `StepRecord`)
  is in place and tested.

Below the per-section walk, two appendices collect:

- **Appendix A**: code issues and design rough edges that this review surfaced
  but which the spec does not directly call out.
- **Appendix B**: a test-coverage map and concrete additional tests worth
  writing.

> **Generality reminder.** `bp-train` is meant to be a *general* bioprocess
> simulation/training package for any dataset that can be expressed in the
> bp_format data format. Anything that hard-codes case-study-specific names,
> shapes, or biology is a bug for V1, not a feature.

---

## 1. Purpose (spec §1)

| Spec | Status | Notes |
|---|---|---|
| Consume bp_format-compatible serialized bioprocess data | ✅ | `prepare.load_raw_collection` accepts a JSON path, an in-memory `BioProcessCollection`, or a `BenchmarkDataset` and pulls a named case study out of the latter. |
| Prepare a training-ready artifact | ✅ | `prepare.prepare_artifact(...)` is the canonical entry point and is exported from `bp_train/__init__.py`. |
| Researcher-facing model API | ✅ | `bp_train/model_api.py` exposes `UserReactionModule`, `ReactionOutputs`, `partition_trainable`. |
| Train hybrid models with standardized controls/feeds/volume/dilution | ✅ | `bp_train/wrapper.py::HybridOdeWrapper` owns this; it delegates the mechanistic step to `bp_format.mechanistic.RhsOde`. |

---

## 2. V1 Goals (spec §2)

| Goal | Status | Where |
|---|---|---|
| Load bp_format `BioProcessCollection` JSON via bp_format deserialization | ✅ | `prepare.load_raw_collection`, `controls_store.ControlsStore.from_json`, `training_data.TrainingDataStore.from_json`. |
| Standardize controls/measured states into a prepared artifact | ✅ | `prepare.prepare_artifact` |
| Hybrid config (JSON artifacts + `custom.py` + optional run JSON) | ✅ | `utils.load_custom_module`, `utils.resolve_config`, prepare/train CLIs both accept `--custom` and `--config`. |
| Globally padded controls representation that avoids recompilation | ✅ | `controls_store.ControlsStore` stacks all processes into `[n_processes, max_grid_length, max_controls]`. |
| Materialize control payloads as JAX arrays with stable collection-wide shapes | ✅ | `controls_store._as_jax_array` + `_pad_payload`. |
| Convert controls to a dense linear-interpolation payload (single solver lookup per experiment) | ✅ | `controls.build_dense_payload` + `controls_store._interp_columns`. |
| Library-owned RHS wrapper handling all dilution and feed transport | ✅ | `wrapper.HybridOdeWrapper.__call__`. The user module never sees dilution. |
| Stateless user models only | ✅ | `model_api.UserReactionModule.__call__` is purely `(t, c, u) → ReactionOutputs`; nothing in the package threads recurrent state. |
| Train against real measurement timestamps only | ✅ | `training_data._timeseries_numpy` reads measured times directly; the harness drives the solver against those times. |
| Strong validation, fail-fast | 🟡 partial | Harness/prepare validation is good; one concrete generality bug is documented in §9 / Appendix A: the prepared-semantics validator hard-codes a "biomass" component. |
| Maintain a running note on bp_format API changes/adapters | ✅ | `spec/bp_format-api-notes.md` exists. |

---

## 3. Non-Goals for V1 (spec §3)

All explicitly deferred items are still deferred in code: no pseudo-batch
integration, no stateful models, no checkpointing, no LOO orchestration, no
data augmentation, no segmented runtime controls API. ✅

---

## 4. Design Principles (spec §4)

| Principle | Status |
|---|---|
| `bp_format` is the semantic source of truth | ✅ — `bp_train` never re-defines bp_format dataclasses; it imports them. |
| Prepared artifact is explicit and persisted | ✅ — `prepare.py` writes `prepared.json` with `metadata["bp_train"]` provenance. |
| Strict prep, fail at prep time | ✅ for control/feed metadata; 🟡 for the hard-coded biomass requirement (Appendix A.1). |
| Library owns generic mechanics | ✅ — wrapper + bp_format split. |
| User code owns case-study-specific semantics | ✅ — `custom.py` with `transform_process_collection`, `build_sample_acc_series`, `build_reaction_module`, `build_learning_rate`, `estimate_all_scales`. |

---

## 5. High-Level Architecture (spec §5)

The Phase A / Phase B / Phase C split is realized:

- **Phase A — raw load**: `prepare.load_raw_collection`, with
  `BenchmarkDataset` adaptation. ✅
- **Phase B — preparation**: `prepare.prepare_artifact` writes `prepared.json`
  with `bp_train` namespaced metadata. ✅
- **Phase C — training**: `harness.train_collection` (collection-level harness)
  is a fully batched `eqx.filter_jit` train step that wraps
  `_batched_measurement_loss_from_batch` under `jax.vmap`. ✅

---

## 6. Artifact Model (spec §6)

### 6.1 Raw artifact
✅ `load_raw_collection` does not assume the raw is training-ready.

### 6.2 Prepared artifact
✅ `prepare_artifact` preserves all original fields and adds
`metadata["bp_train"]` containing prep timestamp, hashes, transform hook
names, control ordering, dynamic-volume flag, semantics provenance, and the
runtime-control build settings. The store reconstructs padded payloads from
this metadata at load time (`controls_store.ControlsStore.from_collection`).

The prepared artifact does **not** persist the padded dense arrays — only the
structural metadata needed to rebuild them. Verified by
`tests/test_prepare.py::test_prepare_artifact_does_not_persist_padded_control_arrays`.

### 6.3 Optional run config
🟡 partial. The CLI `--config` flag loads a JSON dict and threads it through
to the prepare hook (and to `runtime_config` in the train harness path), but
there is no schema for it and the train CLI does not actually expose its
fields (`--config` is currently only used to populate `custom_cfg` for the
train hooks; nothing maps it to `TrainHarnessConfig`). This works but is
under-documented.

---

## 7. Hybrid Configuration Approach (spec §7)

✅ The split is implemented: persisted JSON for data, `custom.py` for code,
JSON config for run settings. `utils.resolve_config` merges
`custom.get_config()` / `custom.CONFIG` with the JSON dict.

---

## 8. `custom.py` Responsibilities (spec §8)

Hook contracts implemented in code:

| Hook | Default | User signature accepted |
|---|---|---|
| `transform_process_collection(collection, config) -> collection` | `defaults.default_transform_process_collection` (supports `process_rename_map`) | ✅ |
| `build_sample_acc_series(process, process_name, collection_metadata, config) -> SignalSource` | `defaults.default_build_sample_acc_series` → `controls.build_sample_acc_source_default` | ✅ |
| `build_reaction_module(*, target_names, process_names, config, seed, collection) -> UserReactionModule` | `defaults.default_build_reaction_module` → `DefaultReactionModule` | ✅ — `collection` was added in the most recent change to avoid double-loading `prepared.json`. |
| `build_learning_rate(config) -> float \| optax.Schedule` | none | ✅ — invoked from `harness.train_from_collection`. |
| `estimate_all_scales(collection, target_names, config) -> (state_scale, controls_scale, q_scale, f_scale)` | none | ✅ — invoked from `harness.train_from_collection`. |
| `partition_trainable(self) -> (trainable, static)` | `model_api._default_partition_trainable` | ✅ |
| `observe(self, states) -> states` | identity | ✅ |

The spec only formally lists `transform_process_collection`. The other hooks
have grown organically and are now load-bearing. The spec should probably be
updated to mention them, but they are reasonable extensions of the principle
"case-study code lives in `custom.py`".

**Issue (Appendix A.2):** the `build_learning_rate` and `estimate_all_scales`
hooks are not documented in the spec and not exercised by any test. If a user
mis-spells one, the harness silently falls back. Worth a unit test.

---

## 9. Preparation Pipeline (spec §9)

### 9.1 Inputs
✅ Raw bp_format JSON, `custom.py`, optional JSON config — all wired through
`prepare.prepare_artifact`.

### 9.2 Processing steps 1–12

| Step | Status |
|---|---|
| 1. Load raw collection | ✅ |
| 2. First-pass `validate_process` | ✅ — `validation.validate_raw_collection` calls `bp_format.validate_process` for each process, with a strict mode. |
| 3. Resolve case-study config in code | ✅ — `utils.resolve_config`. |
| 4. Apply `transform_process_collection` | ✅ |
| 5. Reactor/feed enrichment in code | ✅ — happens inside the user hook. |
| 6. Build default derived controls (`V_sample_acc`, bolus ramps) | ✅ — `controls.build_sample_acc_source_default`, `controls.build_bolus_sources`. |
| 7. Validate control roles, feed semantics, completeness | ✅ — `validation.ensure_prepared_training_semantics`, `_validate_prepared_control_contract`, `validation.ensure_required_controls`. |
| 8. Strict post-transform `bp_format` validation | ✅ — `validate_collection(collection, strict=True)` runs after the transform hook. |
| 9. Persist structural runtime-control metadata | ✅ — `metadata["bp_train"]["processes"][name]` carries `local_control_names`, `control_metadata`, `sample_acc_source` (times/values/step_ts/metadata). |
| 10. Compute scaling stats | 🟡 — happens in user code via `estimate_all_scales`, but is **not persisted** in `prepared.json`. The spec section 13 explicitly says the prep phase may compute and persist these stats. Today they are recomputed at every training run from the in-memory collection. (See Appendix A.3.) |
| 11. Update prep metadata | ✅ |
| 12. Serialize | ✅ `save_process_collection_json(...)`. |

### 9.3 Validation Rules (fail-fast list)

| Rule | Status | Where |
|---|---|---|
| Config-declared control missing | ✅ | `validation.ensure_required_controls` |
| Required `Interpolator` missing | 🟡 partial | `controls._make_source_from_process_variable` falls through to a `TypeError` for unsupported value types, but does not specifically advertise "interpolator missing" as the failure mode. Acceptable but not user-friendly. |
| Feed stream metadata underspecified | ✅ | `validation.ensure_prepared_training_semantics` requires `feed_medium` and component metadata for every feed change. |
| Feed-media coverage invalid for a positive feed stream | ✅ | Same. |
| Reactor/feed component metadata missing | ✅ | Same. |
| **Biomass or other required dynamic species missing** | ❌ generality bug | `validation.py:104, 167-169` literally checks `name.strip().lower() == "biomass"` and raises if no component named "biomass" exists. This is a hard-coded case-study assumption — the bp_format schema does not require the biomass-like component to be called "biomass". A general bioprocess package should let the user *declare* which reactor component plays the biomass role (or relax the requirement). See Appendix A.1. |
| Initial condition cannot be constructed | 🟡 | Implicit — `training_data` raises if a target has no measurements, but does not have a dedicated "missing initial state" error. |
| Control ordering inconsistent | ✅ | `prepare._validate_prepared_control_contract` and `controls_store.ControlsStore.from_collection` both fail fast. |
| Shapes/units irreconcilable | ✅ | `wrapper.validate_rhs_ode_compatibility`. |
| Variable required as both measured state and control without role | 🟡 | `_measurement_targets` rejects controlled targets in `process_variables`, but no symmetric check for the reverse direction. |

---

## 10. Control Semantics (spec §10)

### 10.1 Generic principle
✅ Nothing is tied to `[D, Cf_norm, T]`. Controls are an ordered list discovered
from the prepared process plus the derived `V_sample_acc`.

### 10.2 Control ordering
✅ `controls.select_control_sources` honours `config["control_order"]` (list
or per-process dict), then deterministically appends remaining feed sources
followed by remaining controlled process variables. The chosen ordering is
written into `metadata["bp_train"]["processes"][name]["local_control_names"]`
and re-read by `controls_store._ordered_control_sources`.

### 10.3 Rates vs cumulative
✅ The store keeps the **cumulative** value as the dense control (matching the
bp_format `FeedVolumeChange` traces); the wrapper consumes
`controls.eval_derivative(...)` for feed channels to get the actual flow
rates. Documented in `wrapper.HybridOdeWrapper.__call__` lines 296–308.

### 10.4 Feed streams
✅ Every feed stream is its own control. Controlled feeds come from the
control vector (via `flow_control_indices`); modeled feeds come from
`ReactionOutputs.modeled_feed_rates` aligned to `RhsOde.modeled_flow_names`.

### 10.5 Dynamic volume only / V_real reconstruction
✅ State layout is `[species..., V_cont, B_modeled_cum_0, ...]` with
`V_real = max(V_cont - V_sample_acc, min_real_volume)` reconstructed in the
wrapper. `V_sample_acc` is persisted as a derived control in `prepared.json`
under `metadata["bp_train"]["processes"][name]["sample_acc_source"]`.

### 10.6 Default sampling-control construction
✅ `controls.build_sample_acc_source_default` reads `SampleVolumeChange`
events, accumulates them into a monotone cumulative, and approximates each
event as a short ramp of `get_shortest_time_diff(process)`. Overrideable via
`build_sample_acc_series` hook in `custom.py`.

---

## 11. Control Representation for V1 Runtime (spec §11)

### Padded JAX-tensor canonical layout
Implemented in `controls_store.ControlsStore`:

| Spec field | Code field | Shape |
|---|---|---|
| `dense_grid` | `ControlsStore.dense_grid` | `[n_processes, max_grid_length]` ✅ |
| `control_values` | `ControlsStore.control_values` | `[n_processes, max_grid_length, max_controls]` ✅ |
| `control_derivatives` | `ControlsStore.control_derivatives` | same shape ✅ |
| `step_ts` | `ControlsStore.step_ts` | `[n_processes, max_step_ts_length]` ✅ |
| `grid_lengths` | `ControlsStore.grid_lengths` | `[n_processes]` ✅ |
| `step_ts_lengths` | `ControlsStore.step_ts_lengths` | `[n_processes]` ✅ |

`ControlsStore.from_collection` enforces one shared control ordering across
processes and fails fast on mismatch
(`tests/test_controls_store.py::test_controls_store_rejects_different_control_order`).

### 11.1 Build strategy
✅ `controls.build_dense_payload` starts from source knots + a uniform initial
grid (`initial_grid_points`, default 16), iteratively bisects intervals where
the relative interpolation error against the source exceeds `max_rel_error`
(default 1e-4), normalizes by per-source spread
(`controls.compute_signal_spreads`), and stops at `max_refinement_rounds`
(default 8). `_make_source_from_xy` builds piecewise-linear sources from
xy series, `_make_source_from_process_variable` handles ppoly interpolators
and static variables.

### 11.2 Segment boundaries / events
✅ The runtime store keeps `step_ts` separate from `dense_grid`, populated
from bolus and `V_sample_acc` ramp boundaries. The training step passes a
right-clamped `jump_ts_rows` slice into the diffrax PIDController via
`_clamp_padded_time_rows` (`harness.py:431-434`). `solver_use_jump_ts=True`
is the default.

### 11.3 Bolus approximation
✅ `controls.build_bolus_sources` rejects near-duplicate bolus timestamps
(threshold = `BOLUS_DUPLICATE_THRESHOLD_REL * total_duration`,
1e-4 of duration), and turns each bolus into a triangular ramp of duration
`get_shortest_time_diff(process)`. Sample ramps work the same way.

`step_ts` includes both event start and ramp end. The "tests must verify
integrated bolus additions and integrated sampled-volume removals match
intended amounts" requirement is **partially** covered:

- `tests/test_prepare.py::test_prepare_artifact_builds_sample_acc_amount_correctly`
  checks the cumulative magnitudes,
- `tests/test_wrapper.py::test_wrapper_constant_feed_rate_integrates_volume_correctly`
  is a regression test for continuous-feed volume integration,
- but **no test integrates a bolus through the wrapper end-to-end and checks
  that ∫dV/dt over the ramp equals the intended discrete bolus amount**, and
  no test integrates a sampling ramp through the wrapper and checks that
  V_real drops by the intended amount. Worth adding (Appendix B.4).

### 11.4 Global padding
✅ Padding shapes are stable; the active prefix is exposed via
`PerProcessControls.active_*` properties for debugging/plotting only. The
padded tail is right-clamped in the batched evaluator (`as_batch_controls`)
so an out-of-active-range `t` returns the last active value rather than zero.

### 11.5 Batched controls evaluator
✅ `controls_store.BatchControls` exposes
`eval(process_idx, t)` and `eval_derivative(process_idx, t)` with no
process-name dict lookups in the hot path. `_BatchIndexedControls` in
`trainer.py` is the per-vmap-sample wrapper that the JIT step uses.

---

## 12. Model Abstraction (spec §12)

### 12.1 User reaction module contract
✅ `UserReactionModule` is an `eqx.Module` with `__call__` and
`partition_trainable`, optionally `observe`.

### 12.2 Concrete return contract
✅ `ReactionOutputs(specific_rates, modeled_feed_rates)` is a frozen
`eqx.Module`. Wrapper validates both shapes (`wrapper.py:336-346`) and raises
`ValueError` on mismatch.

### 12.3 Trainable partitioning
✅ `model_api.partition_trainable` defaults to "inexact arrays under
`.model`". Tests cover the default path, the missing-`.model` failure, the
custom override, invalid partition structure, and overlapping partitions
(`tests/test_model_api.py`).

### 12.4 Reaction-only contract
✅ `HybridOdeWrapper.__call__` builds the augmented controls vector (`base
controls ++ flat Cin ++ flat Cin_modeled`), scales inputs, calls the
reaction module, un-scales outputs, then delegates to `RhsOde` for biomass
scaling, transport and dilution. `B_modeled_cum_k` derivatives are appended
manually as `dB_k/dt = F_modeled_k`.

### 12.5 Observations
✅ `observe` defaults to identity. **Not actually called anywhere in the
training loop or postprocessing today** — the spec says "should be called
during integration or trajectory saving rather than as a purely post-hoc
transform", but in practice the trained model is read out as raw state
trajectories and `observe` is a dead default. Acceptable for V1 since no
case study uses it, but worth either using it in
`postprocessing.plot_training_results` or removing the abstraction. (Appendix A.4)

---

## 13. Scaling (spec §13)

✅ Scaling is owned by the wrapper but **parameterized** by frozen
`state_scale / controls_scale / q_scale / f_scale / target_variance` arrays
that are constructed at harness setup time, optionally from the
`estimate_all_scales` hook. The wrapper applies `c_scaled = y[:n_species]`
(integration is in scaled space), `u_scaled = U_augmented / controls_scale`,
and un-scales `q` / `f` after the MLP call (`wrapper.py:320-350`).

🟡 As noted in §9.2 step 10: scales are computed at every run rather than
persisted into `prepared.json`. The spec allows persisting in prep metadata,
which would make runs more reproducible.

---

## 14. Runtime Wrapper Contract (spec §14)

### 14.1 Inputs / 14.2 Responsibilities
✅ Implemented in `wrapper.HybridOdeWrapper.__call__`. The wrapper:

1. Reads all flow controls via `controls.eval_derivative(t)[flow_control_indices]`.
2. Maintains `V_cont` as state index `n_species`.
3. Reads `V_sample_acc` from `controls.eval(t)[sample_acc_control_index]`.
4. Builds `V_real = max(V_cont - V_sample_acc, min_real_volume)`.
5. Builds `U_augmented = [controls_vector, Cin.flat, Cin_modeled.flat]`.
6. Calls `reaction_module(t_arr, c_scaled, u_scaled)`.
7. Delegates to `RhsOde(C_rhs, Q, U_flow, F_modeled)`.
8. Appends `dB_k/dt = F_modeled_k` and re-scales the full derivative.

### 14.3 Modeled feeds
✅ Strict: every feed declared explicitly, ordering pulled from
`RhsOde.modeled_flow_names`. The wrapper raises `ValueError` if
`modeled_feed_rates.shape != (f_modeled_size,)`.

### 14.4 Relationship to bp_format.mechanistic
✅ `bp-train` owns controls/batching/loss/training-loop; `bp_format.mechanistic.RhsOde`
owns mechanistic derivative assembly. The wrapper is a thin adapter.

---

## 15. Training Data Object (spec §15)

✅ `training_data.PerProcessTrainingData` carries the minimum V1 fields
(`process_name`, `t_meas`, `y_meas`, `y0`, `controls`) plus a measurement
mask and active count. `BatchTrainingData` is the gathered batch view.

### 15.1 `y_meas`
- Contains only measured target variables ✅
- Uses config-defined column order by variable name ✅ (or auto-resolved per
  process when no `target_variable_order` is given)
- Subset of all dynamic states ✅
- Does NOT include `V_cont` ✅
- **Does** include cumulative modeled-feed columns
  (`B_modeled_cum_per_modeled_feed`) appended after the species columns —
  this is an extension over the spec text, justified by the in-source
  comment (it gives the MLP a direct training signal for modeled feeds). The
  spec text doesn't forbid it but doesn't mention it either; this is worth
  adding to the spec.
- Padded across experiments ✅

### 15.2 `y0`
- First measured value for observed species ✅
- Appends `V_cont` ✅
- Appends `B_modeled_cum_k(0) = 0` for each modeled flow (extension consistent
  with the y_meas extension)
- `V_cont(0) = process.volume.initial_volume` ✅
- Built at training-data build time from the prepared collection ✅

### 15.3 Training grid
✅ Trains only against real measurement timestamps. The harness uses
`_clamp_padded_time_rows` to right-clamp padded tails so the solver does not
see NaN/zero times. Loss uses `meas_mask` to ignore padded rows.

---

## 16. Solver Behavior (spec §16)

### 16.1 Integration mode
✅ Real-space, dynamic volume, `V_cont` is the entry at index `n_species`,
`V_real` reconstructed in the wrapper.

### 16.2 Step boundaries
✅ Wired through `jump_ts_rows` in `harness._make_batched_step` →
`_measurement_loss_from_arrays` → `_simulate_measurement_states_on_grid` →
`diffrax.PIDController(jump_ts=jump_ts)`.

### 16.3 Expected runtime objects
✅ One stacked array per payload kind, lightweight per-process index metadata.

### 16.4 Batched train-step contract

| Requirement | Status |
|---|---|
| `steps` = number of optimizer updates | ✅ |
| Each update consumes one full batch | ✅ |
| Total sampled indices = `steps * batch_size` | ✅ — `harness._build_batch_index_stream` |
| `batch_size=None` resolves to `len(selected_processes)` | ✅ — `_resolve_effective_batch_size` |
| No `drop_last_batch` | ✅ — explicitly tested by `test_harness_config_has_no_drop_last_batch_field` |
| `process_names=None` → all of `store.process_order` | ✅ — `_ensure_process_names` |
| Unique-and-known process names | ✅ |
| Round-robin base mode | ✅ |
| `shuffle_batches=True` shuffles each cycle | ✅ |
| `batch_seed` controls all batch randomness | ✅ |
| Determinism (`batch_seed is None` → fall back to `seed`) | ✅ — `test_harness_batch_stream_falls_back_to_seed_when_batch_seed_none` |
| Optax backend, `{adam, sgd}`, default adam, lr > 0 | ✅ — `trainer._build_optimizer` |
| Batch loss = mean of per-sample losses | ✅ — `_batched_measurement_loss_from_batch` returns `jnp.mean(per_sample_total)` |
| One JIT step per run under stable shapes | ✅ — `_make_batched_step` builds it once |
| Record JIT input-signature summary | ✅ — `summarize_train_step_input_signature` is on `TrainHarnessResult.train_step_input_signature` |
| Record rebuild count | ✅ — `TrainHarnessResult.train_step_rebuild_count`, also surfaced in `RunLogger.record_rebuild` warnings |
| `log_every` controls cadence | ✅ |
| Per-process losses logged at log steps are for sampled batch members only | ✅ — Today they are read directly from the JIT step's per-sample vector (post-redesign) — no extra solve. |

### 16.5 Batching config and validation contract

The dataclass [`harness.TrainHarnessConfig`](../bp_train/harness.py) has all
required fields plus solver tolerances and the new logging knobs. Fail-fast
errors:

| Rule | Code |
|---|---|
| `steps <= 0` | `_validate_batching_config` ✅ |
| Effective `batch_size <= 0` | `_validate_batching_config` ✅ |
| `learning_rate <= 0` | `_validate_batching_config` ✅ (also enforced in `_build_optimizer`) |
| Unknown process names | `_ensure_process_names` ✅ |
| Duplicate process names | `_ensure_process_names` ✅ |
| Empty selected set | `_ensure_process_names` ✅ |
| Unsupported optimizer | `_validate_batching_config` ✅ |

### 16.6 Batch Telemetry Contract

| Required field | Provided as |
|---|---|
| Batch mean loss by step | `TrainHarnessResult.mean_loss_by_step` ✅ |
| Sampled per-process losses at logging steps | `TrainHarnessResult.sampled_loss_by_process_at_log_steps` ✅ |
| Batch composition by step | `TrainHarnessResult.batch_process_names_by_step` ✅ |
| First compile/warmup time | `TrainHarnessResult.compile_warmup_seconds` ✅ |
| Per-step runtime timings | `TrainHarnessResult.step_time_seconds` ✅ |
| JIT input-signature summary | `TrainHarnessResult.train_step_input_signature` ✅ |
| Train-step rebuild count | `TrainHarnessResult.train_step_rebuild_count` ✅ |
| (extra) Per-process loss by step | `TrainHarnessResult.per_process_loss_by_step` — added in the logging redesign, free because the JIT step already produces it. |

The `RunLogger` redesign also adds optional CSV / JSONL persistence and a
fixed-width tabular console formatter. Tested in `tests/test_logging.py`
(11 tests).

---

## 17. Metadata in `prepared.json` (spec §17)

| Required field | Status |
|---|---|
| Source input path or dataset id | ✅ `metadata["bp_train"]["source_input_path"]` |
| SHA-256 hex of raw input bytes | ✅ `source_input_sha256` |
| Prep timestamp | ✅ `prepared_at` |
| SHA-256 hex of `custom.py` bytes | ✅ `custom_py_sha256` |
| Names of applied transform hooks | ✅ `transform_hooks` |
| Control ordering | ✅ `processes[name]["local_control_names"]` |
| Whether volume is dynamic | ✅ `dynamic_volume: True` |
| Updates to `is_controlled` | 🟡 — provenance is captured under `semantics_provenance`, but there is no dedicated `is_controlled_changes` field. Acceptable in practice. |
| Provenance for derived controls (e.g. `V_sample_acc`) | ✅ `processes[name]["sample_acc_source"]` |
| Scaling metadata | ❌ not persisted — see Appendix A.3 |
| Runtime control-build settings + per-process control/source metadata | ✅ `runtime_controls_config` and `processes[name]["control_metadata"]` |
| Stored under `metadata["bp_train"]` namespace | ✅ |

---

## 18. Suggested File Layout (spec §18)

The package layout closely matches the suggestion. Difference: the spec
suggests one `cli.py`; we have one `cli.py` and the train command lives
inside it as a subparser. Difference: the spec mentions "controls.py" (prep)
and "controls_store.py" (runtime) — both exist in code with the right split.

`bp-train prepare` and `bp-train train` both exist. ✅

| Spec module | Actual module | Notes |
|---|---|---|
| `cli.py` | ✅ `bp_train/cli.py` | Both `prepare` and `train` subcommands |
| `prepare.py` | ✅ | |
| `controls.py` (prep-time) | ✅ | numpy-only adaptive grid |
| `controls_store.py` (runtime) | ✅ | JAX-padded |
| `training_data.py` | ✅ | |
| `wrapper.py` | ✅ | |
| `trainer.py` | ✅ — single-process primitives + helpers | `single_process_train_step`, `simulate_measurement_states`, `single_process_measurement_loss`. |
| `model_api.py` | ✅ | |
| `validation.py` | ✅ | |
| `utils.py` | ✅ | |
| `defaults.py` | ✅ | |
| **(extra)** `harness.py` | ✅ | Collection-level batched harness — replaces what the prior status doc called "missing collection-level orchestrator". |
| **(extra)** `logging.py` | ✅ | `RunLogger` + `StepRecord` + `_ConsoleTableFormatter` |
| **(extra)** `postprocessing.py` | ✅ | `save_model`, `plot_training_results` |

---

## 19. Testing Requirements (spec §19)

### 19.1 Control Prep Tests

| Required | Status | Test |
|---|---|---|
| Prepared metadata is structural, no padded dense arrays | ✅ | `test_prepare_artifact_does_not_persist_padded_control_arrays` |
| Dense-grid linear interpolation error below threshold | ❌ missing | No test asserts that the refined dense grid satisfies `max_rel_error`. Appendix B.1. |
| Global padding shapes stable | 🟡 | `test_controls_store_loads_by_process_name_and_index` and `_eval_matches_prepared_linear_payload` indirectly verify it; no direct shape-stability assertion across heterogeneous-length processes. |
| Config-defined control ordering preserved | ✅ | `test_prepare_artifact_respects_custom_control_order` |
| Missing controls fail fast | ✅ | `test_prepare_artifact_fails_on_missing_required_control` |
| Cumulative-to-rate transform hooks work | ❌ missing — there is no test that exercises a `transform_process_collection` hook that converts a cumulative trace to a rate. Appendix B.2. |
| Default `V_sample_acc` from `SampleVolumeChange` | ✅ | `test_prepare_artifact_builds_sample_acc_amount_correctly` |
| User override of `V_sample_acc` construction | ✅ | `test_controls_store_uses_custom_sample_acc_from_prepared_metadata` |

### 19.2 Event Approximation Tests

| Required | Status | Notes |
|---|---|---|
| Bolus approximation adds intended amount | 🟡 | `test_build_bolus_sources_*` tests cover the *signal*, but no test integrates the wrapper across a bolus and asserts ∫dV/dt = bolus amount. Appendix B.4. |
| Boundary times passed through correctly | ✅ | `step_ts` propagation is verified in `test_controls_store_loads_*` and indirectly via `prepare_artifact_persists_feed_metadata`. |
| Reconstructed `V_real` matches measured volume trace | ❌ missing as a wrapper-integration test. The constant-feed regression test `test_wrapper_constant_feed_rate_integrates_volume_correctly` is the closest. Appendix B.4. |
| Reconstructed `V_real` matches sampled-volume history | ❌ missing |
| Solver does not exhibit excessive rejected steps around events | ❌ missing — no assertion on solver step counts. Appendix B.5. |

### 19.3 Wrapper Tests

| Required | Status | Test |
|---|---|---|
| Dilution and feed transport applied correctly | 🟡 | `test_wrapper_with_modeled_feed_produces_finite_derivative`, `_multiple_controlled_feeds`, and `_constant_feed_rate_integrates_volume_correctly` cover the sunny path. No test explicitly compares dilution vs an analytic CSTR result. |
| Multiple feed streams sum correctly | ✅ | `test_wrapper_multiple_controlled_feeds` |
| Feed-medium composition applied (incl. zero contribution for absent species) | 🟡 | `test_wrapper_with_modeled_feed_produces_finite_derivative` covers a single-species/single-feed case. No multi-species test where one species is absent from the feed medium. Appendix B.6. |
| Dynamic volume state updates correctly | ✅ | `test_wrapper_constant_feed_rate_integrates_volume_correctly` |
| User reaction terms combined correctly with wrapper transport | 🟡 | Implicitly via the harness loss-decreases tests; no isolated test that pins the linear superposition. |

### 19.4 Trainer Tests

| Required | Status |
|---|---|
| Single train step produces gradients | ✅ — `test_single_process_train_step_produces_gradients_and_keeps_frozen_params_static` |
| Only `partition_trainable()` params receive gradients | ✅ — same |
| Frozen params remain gradient-free | ✅ — same |
| Measurement-time loss uses padded arrays + masks | ✅ — `test_measurement_loss_ignores_padded_rows_via_mask` |
| Batch-size/repeat behavior (incl. single-process with `batch_size > 1`) | 🟡 — covered in `test_harness_batch_stream_*` for the index stream, not for the train step itself. |
| Batch index generation deterministic with fixed seed/config | ✅ — `test_harness_batch_stream_shuffle_is_deterministic_and_seeded` |
| Training loss decreases on toy data | ✅ — `test_train_collection_single_process_loss_decreases` and `_multi_process_tracks_per_process_histories` |
| Invalid batching config fails fast | ✅ — `test_harness_phase1_batching_validation_checks_basics` and friends |
| Train-step input signatures stable | ✅ — `test_train_collection_signature_is_stable_and_no_rebuilds` |
| No explicit train-step rebuild path triggered in stable runs | ✅ — same |

---

## 20. Implementation Order (spec §20)

The historical 1–8 list is implemented. The new items beyond what the
2026-03-30 status doc captured:

- **Step 8+: collection-level training harness** — `harness.train_collection`,
  `train_from_collection`, `train_from_prepared_json`, `TrainHarnessConfig`,
  `TrainHarnessResult`, `RunLogger`. ✅
- **Step 8+: train CLI subcommand** — `bp-train train --input ... --custom ...`. ✅
- **Step 8+: end-to-end Kittler example** — `examples/01_kittler_2022/run.sh`,
  `run_single_process.sh`, `custom.py` with all the hooks. ✅

Remaining items (the spec is V1 so these are out of scope, but worth listing
for the next planning round):

- LOO-CV orchestration (still deferred)
- Persisted scaling metadata (Appendix A.3)
- A general "biomass role" declaration to remove the hard-coded name
  (Appendix A.1)

---

## 21. Deferred Items (spec §21)

All explicitly deferred. ✅

---

## 22. Known Limitations (spec §22)

| Limitation | Still true? |
|---|---|
| `Cin` constant at runtime | ✅ true — `wrapper.py:312-317` builds `cin_flat` from `rhs_ode.Cin` and `Cin_modeled`, both fixed at wrapper-construction time. |
| `bp-train` delegates mechanistic RHS to `bp_format.mechanistic.RhsOde` | ✅ true |
| Reaction-module contract is `q + modeled_feed_rates`, can't bypass `q * X_active` | ✅ true |

---

# Appendix A — Code issues this review found

## A.1 — `validation.py` hard-codes a `"biomass"` reactor component

[`bp_train/validation.py:104`](../bp_train/validation.py#L104) does:

```python
has_biomass = any(name.strip().lower() == "biomass" for name in reactor_components)
```

and then [`validation.py:167-169`](../bp_train/validation.py#L167) raises if
`has_biomass` is `False`. This is the most concrete generality violation in
the package: a dataset whose biomass-like reactor component is named `X`,
`cells`, `cell_mass`, `Pichia_pastoris`, `WCW`, `DCW`, `cdw`, … will fail
prep even though everything else about it is well-formed. The bp_format schema
does not require the biomass component to be called "biomass".

**Suggested fix**: replace the hard-coded check with one of:

1. A user-declared role: `config["biomass_component_name"]` (default
   `"biomass"`, but resolvable by the user). Persist the resolved name into
   `metadata["bp_train"]["biomass_component"]`.
2. A heuristic *plus* override: try to discover a biomass-like component
   (heuristic search for the reactor component referenced as `X_active` by
   `bp_format.mechanistic.RhsOde`, since bp_format already needs to know which
   component is biomass to build the mechanistic RHS) and let the user
   override it via `custom.CONFIG`.
3. Drop the requirement entirely from `bp-train` and rely on bp_format's
   `validate_process` to fail if the mechanistic ODE cannot be built.

Option 2 or 3 is preferred — bp_format is already the source of truth.

## A.2 — Undocumented hooks (`build_learning_rate`, `estimate_all_scales`, `build_reaction_module(collection=...)`)

These hooks are load-bearing for the Kittler example and the harness, but
they are not mentioned in the spec sections 8.2 or 8.3. They have no unit
tests; if a user mis-spells one, the harness silently falls back to the
default behavior with no warning.

**Suggested fix**: document them in spec §8.2, add a unit test that
mis-spells each one and asserts the fallback happens with a `logger.info`
notice (today no notice is emitted).

## A.3 — Scaling stats are not persisted to `prepared.json`

Spec §13 explicitly allows persisting scale statistics in the prepared
artifact. Today they are computed at every training run by
`estimate_all_scales` from the in-memory collection. This is a missed
reproducibility opportunity:

- if a user changes `estimate_all_scales` between runs without changing the
  data, two runs that "use the same prepared artifact" will produce
  different trained models;
- there is no way to inspect what scales were used after the fact.

**Suggested fix**: optionally cache the four arrays into
`metadata["bp_train"]["scaling"]` during prepare (or first train run) and
prefer them at subsequent runs.

## A.4 — `observe(...)` is unused

`UserReactionModule.observe` defaults to identity, has a unit test for the
identity case, but is never actually called from the wrapper or the
postprocessing pipeline. Either:

- thread it through `_simulate_measurement_states_on_grid` so the loss is
  computed against `observe(states)` rather than raw `states[indices]`, or
- drop the abstraction from V1 and pull it back when there is a concrete
  user.

The half-implemented hook is the worst of both worlds.

## A.5 — `train_from_collection` is not in `__init__.py`

We added `train_from_collection` to `harness.py` (it is the canonical entry
point used by the CLI now), but `bp_train/__init__.py` only exports
`train_from_prepared_json` and `train_collection`. Library users who want to
re-use the CLI's "load-once-then-train" pattern have to reach into the
submodule. One-line fix.

## A.6 — `cli._handle_train` re-imports inside the function

[`bp_train/cli.py:250-252`](../bp_train/cli.py#L250) does

```python
from bp_format.serialization import load_process_collection_json
from .training_data import TrainingDataStore
```

inside `_handle_train`. These imports were moved into the function in the
"load once" change, but they could just live at module top with the other
imports. Cosmetic.

## A.7 — `min_real_volume = 1e-8` is a magic number

`HybridOdeWrapper.from_process` uses `min_real_volume=1e-8` as the floor for
`V_real = max(V_cont - V_sample_acc, min_real_volume)`. For datasets where
volumes are in litres this is fine; for datasets where volumes are in
millilitres (1e-8 mL = 1e-11 L → underflow) or larger units it could be
wrong. Worth either:

- expressing it relative to `process.volume.initial_volume`, or
- exposing it via `TrainHarnessConfig`.

## A.8 — `single_process_train_step` does not use Optax

[`bp_train/trainer.py:265-270`](../bp_train/trainer.py#L265) does manual
gradient descent (`-lr * grad`) instead of routing through `optax`. The
batched harness uses Optax via `_build_optimizer`. The single-process
primitive is older and predates the harness. It's still tested and probably
still useful as a debugging entry point, but the inconsistency invites bugs:
a user who wants Adam on a single process has to drop into the harness with
`batch_size=1`. Either:

- delete `single_process_train_step` (the harness covers everything it does), or
- migrate it to use `_build_optimizer` so the two paths share an optimizer.

## A.9 — Bench-time `_BatchIndexedControls` re-creates a wrapper per sample

`harness._batched_measurement_loss_from_batch._sample_loss` calls
`eqx.tree_at(... (controls, cin, cin_modeled))` inside the vmapped callable.
This is inside the JIT, so it's fine for performance, but it does mean every
training step re-substitutes Cin per process even though Cin is constant
across the run. If a future optimization wants to lift the substitution out
of vmap, this is the place. Not a bug today.

## A.10 — Soft naming inconsistency: `solver_max_steps` vs `max_solver_steps`

The harness config uses `solver_max_steps`. The trainer functions use
`max_solver_steps` as the kwarg name. Both internally hand it to diffrax as
`max_steps`. Three names for one thing. Worth picking one (`solver_max_steps`
matches the config and CLI).

## A.11 — `controls.py:163` mentions Kittler in a comment

The "1500 forced steps for the Kittler dataset" comment in
`bp_train/controls.py` is harmless context, but a general package should not
namedrop a specific case study in module docstrings/comments. Cosmetic.

---

# Appendix B — Test gaps and concrete additions

There are 93 tests today (counted with `pytest --collect-only`), grouped by
file roughly as follows:

| File | Tests | Module covered |
|---|---|---|
| `test_cli.py` | 3 | `bp_train.cli` (dispatch only, mocked harness) |
| `test_controls_store.py` | 9 | `ControlsStore`, `PerProcessControls`, `BatchControls` |
| `test_harness.py` | 15 | `train_collection`, batching, validation, signature stability, log cadence |
| `test_logging.py` | 11 | `RunLogger`, `_ConsoleTableFormatter`, `StepRecord`, CSV/JSONL sinks |
| `test_model_api.py` | 6 | `partition_trainable`, default vs custom, observe identity |
| `test_prepare.py` | 23 | `prepare_artifact`, `select_control_sources`, `build_bolus_sources`, semantic validation |
| `test_trainer.py` | 6 | `single_process_train_step`, `_measurement_loss_from_arrays`, signature summary |
| `test_training_data.py` | 12 | `TrainingDataStore`, target source resolution, batch gather |
| `test_wrapper.py` | 8 | `HybridOdeWrapper`, multi-feed, validation, volume regression |

Modules without **any** direct tests: `bp_train/postprocessing.py`,
`bp_train/utils.py`, `bp_train/validation.py`, `bp_train/defaults.py`. The
first three are partially exercised end-to-end through prepare/train tests,
but specific behaviors slip through.

Concrete tests worth adding:

## B.1 — Dense-grid interpolation error is below `max_rel_error`

Spec §19.1 calls this out explicitly. Today nothing checks it.

```python
def test_build_dense_payload_meets_max_rel_error():
    # Build a process with a known cumulative-feed signal.
    # Run controls.build_dense_payload with max_rel_error = 1e-4.
    # On a denser reference grid, assert
    #   max_rel_error_per_source = max(|payload(t) - source(t)|) / spread <= 1e-4.
```

Easy to write — `controls.build_dense_payload` is pure numpy, no JAX needed.

## B.2 — Cumulative-to-rate `transform_process_collection` hook

Spec §10.3 / §19.1: "cumulative-to-rate transformation hooks work as
expected." Today no test exercises a user hook that converts cumulative to
rate. Add one in `tests/test_prepare.py` that:

1. Builds a synthetic collection with a cumulative trace,
2. Provides a `transform_process_collection` hook that turns it into a
   rate signal,
3. Calls `prepare_artifact`,
4. Reads the prepared metadata and asserts the converted control is present
   and has the expected derivative.

## B.3 — `build_learning_rate` and `estimate_all_scales` hook coverage

A test that defines a `custom.py` module exposing both hooks, runs a
2-step `train_from_collection`, and asserts:

- the LR returned by the hook ends up in the optimizer,
- the scale arrays returned by the hook end up on the trained wrapper,
- a missing/mis-spelled hook falls back to defaults (and ideally logs a
  notice).

## B.4 — End-to-end bolus and sampling integration through the wrapper

Two tests in `tests/test_wrapper.py`:

```python
def test_wrapper_integrates_a_bolus_to_the_intended_amount():
    # Build a process with one bolus FeedVolumeChange (is_continuous=False)
    # of known total volume V_bolus at time t_event.
    # Build the wrapper, integrate from t0 to t_end through diffrax with
    # jump_ts = controls.active_step_ts.
    # Assert: V_cont(t_end) - V_cont(t0) ≈ V_bolus + (sum of any other inflows).

def test_wrapper_sampling_event_drops_v_real_by_intended_amount():
    # Similar setup with a SampleVolumeChange of known volume.
    # Integrate. Assert V_real(t_end) ≈ V_cont(t_end) - V_sample_acc(t_end)
    # and V_sample_acc(t_end) - V_sample_acc(t0) ≈ -ΔV_sample.
```

These directly cover spec §11.3 ("tests must verify integrated bolus
additions and integrated sampled-volume removals match their intended
amounts") which is currently only covered indirectly.

## B.5 — Solver step-count regression

Add a test that runs the wrapper across one process with several bolus
events and asserts that `diffrax`'s `solution.stats["num_accepted_steps"]`
is below a generous bound (say `10 * (n_events + n_meas)`). This would catch
a regression like the historical "step_ts populated with the full input
grid" bug. Spec §19.2.

## B.6 — Multi-species feed with one species absent from the feed medium

Spec §19.3: "feed-medium composition is applied correctly, including zero
contribution for absent species." Add a wrapper test where a 2-species
process has a feed whose `feed_medium` only contains species A, and assert
that the species-B time derivative coming from the wrapper has zero feed
contribution while species-A's matches `(F * Cin_A) / V`.

## B.7 — `validation.py` direct unit tests

Currently `validation.py` is only exercised via `prepare.prepare_artifact`.
A direct test file (`tests/test_validation.py`) would catch the biomass-name
generality issue in A.1 and would let the "biomass component required" rule
be replaced cleanly.

## B.8 — `postprocessing.save_model` round-trip

`tests/test_postprocessing.py` (new): build a small wrapper, call
`save_model` to a tmp path, reload via `eqx.tree_deserialise_leaves`, assert
the deserialized wrapper produces the same `__call__` output on a sample
state.

## B.9 — `cli` end-to-end with real prepared JSON

`tests/test_cli.py` only mocks `train_from_collection`. A small end-to-end
test that runs `bp-train prepare` then `bp-train train --steps 2 --no-plot`
on a synthetic prepared collection (no Kittler dependency) would catch
import/wiring regressions like the recent CLI/harness signature changes.
This is the biggest "we have no end-to-end test" gap.

## B.10 — Three-or-more processes

All harness tests today use 1 or 2 processes. Add at least one test with
≥3 processes to catch any off-by-one in batch index generation, padding, or
the per-process loss vector size. (Spec §19.4 implicitly requires this:
"batch-size/repeat behavior is correct".)

## B.11 — `n_targets != n_species`

`y_meas` columns are `[species..., B_modeled_cum...]`. The default
`target_state_indices` skips `V_cont`. A test where `n_targets > n_species`
(modeled feeds present) and a test where the user passes a custom
`target_state_indices` would pin the gather logic in
`_measurement_loss_from_arrays`.

## B.12 — `utils.resolve_config` precedence

A unit test that exercises the `module.CONFIG` vs `module.get_config()` vs
JSON-config merge order. One of these is silently overridden today and a
test would lock in the precedence.

---

## Summary

**Spec compliance**: ~95%. The package implements every numbered spec
requirement *as a feature*. The remaining 5% is:

- one generality bug (hard-coded biomass name, A.1),
- one missed reproducibility opportunity (scaling stats not persisted, A.3),
- one half-implemented abstraction (`observe`, A.4),
- a handful of cosmetic/inconsistency issues (A.5–A.11),
- and a meaningful set of missing tests (B.1–B.12) — most of which the spec
  itself flags as required (§19) but which the test suite has not yet
  caught up to.

**Tests**: 93 passing tests, organized by module. Coverage is strong on
prepare/controls/training-data/wrapper/harness; weak on validation,
postprocessing, utils, and end-to-end CLI. The biggest single gap is **no
end-to-end test that runs the real CLI from prepare → train**, which
matters because the recent CLI/harness signature changes have broken that
path twice and only manual smoke testing has caught it.

**Recommended next steps** (in the spirit of the prior "what's missing
before LOO-CV" doc, but updated):

1. Fix the biomass-name generality bug (A.1) — required before the package
   can credibly claim to be "general bioprocess".
2. Add tests B.1, B.4, B.7, B.9, B.10 — they cover spec requirements that
   currently slip through the suite.
3. Persist scaling stats into `prepared.json` (A.3) — moves the package
   closer to the "rerun is reproducible" promise of §13.
4. Decide what to do with `observe` (A.4) and `single_process_train_step`
   (A.8): keep and finish, or remove.
5. Then resume the LOO-CV roadmap from the prior status doc, which is still
   the right next milestone — train/eval boundary, fold orchestration,
   per-fold model re-init, result aggregation.
