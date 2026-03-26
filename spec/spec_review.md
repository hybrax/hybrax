# BP-Train V1 Spec Review

This document reviews the V1 detailed spec against the design Q&A threads
(`answers.md`, `answers-2.md`, `answers-3.md`), the `bpbench-api-notes.md`,
and the live `bpbench` 0.1.0 API.

---

## 1. Overall Assessment

The spec is coherent and the three Q&A rounds have sharpened it significantly.
The core architecture — raw load → prep → training — is sound and the hybrid
`custom.py` / JSON config split is well-motivated.

Three areas need tightening before implementation starts:

1. Several bpbench API realities differ from what the spec assumes.
2. The controls-payload pipeline description has internal ambiguities introduced
   by the late segment-collapse decision (answers-3.md Q7).
3. The runtime wrapper contract does not yet say how it maps onto the existing
   `bpbench.mechanistic` modules.

The sections below go through these and other findings in order of impact.

---

## 2. bpbench API Gaps and Mismatches

### 2.1 `interpolator` naming

The target API is `interpolator`. The `Interpolator` class remains the correct
type.

Concretely:

```python
FeedVolumeChange(..., spline: Optional[Interpolator] = None)
SampleVolumeChange(..., spline: Optional[Interpolator] = None)
ProcessVariable(..., spline: Optional[Interpolator] = None)
```

The spec should therefore treat `interpolator` as canonical.

**Action required:** Use `interpolator` directly in `bp-train` code and docs.

**Answer:** The field was renamed to `interpolator` in bpbench. `bp-train`
should therefore use `interpolator` directly.

### 2.2 `BenchmarkDataset` dataclass/serializer mismatch — may be resolved

The `bpbench-api-notes` flag a mismatch: serializer expects `case_studies` but
the dataclass only declares `metadata`. The live `help(bpbench)` output shows
`BenchmarkDataset(metadata: Dict[str, str], case_studies: Dict[str, CaseStudy])`,
so both fields are present in 0.1.0.

**Action:** Verify against actual serialization round-trip test and close the
item in `bpbench-api-notes` if confirmed fixed.

**Answer**: `BenchmarkDataset` has an attribute `metadata` and one `case_studies`. However, `bp-train` should work with `BioProcessCollection` for now. This makes our lives easier. Please change everywhere in the spec and API docs.

### 2.3 `bpbench.mechanistic` already exists and overlaps with the spec

The spec designs a new runtime wrapper and controls pipeline without
acknowledging the existing `bpbench.mechanistic` module, which already
provides:

- `ControlSplines` (`eqx.Module`) — evaluates all controlled signals at `t`,
  exposes `control_names`, `flow_indices`, `ctrl_indices`, and returns **flow
  rates** (derivative of cumulative-volume splines) for continuous feeds.
- `RhsOde` (`eqx.Module`) — computes `dc/dt` including `dV/dt`, accepts
  `f_modeled` for uncontrolled feed streams.
- `get_control_splines(process)`, `get_rhs_ode(process)`.

These overlap significantly with what the spec proposes for `controls.py` and
`wrapper.py`. The spec must explicitly choose one of:

a. **Reuse** — bp-train drives `bpbench.mechanistic` modules, adding only the
   dense-grid linearization on top and wrapping the user reaction module.
b. **Replace** — bp-train builds its own controls and wrapper, treating
   `bpbench.mechanistic` as prior art but not a dependency.

The dense-grid + linear-interpolation runtime strategy (answers.md Q6,
answers-3.md Q7) is intentionally different from `ControlSplines`' PPoly
approach, which is a real reason to build independently. However, the feed
classification logic (`flow_indices`, `ctrl_indices`) in `ControlSplines` is
valuable and should be replicated or borrowed.

**Action:** Add a paragraph to the spec stating the choice and the reason.

**Answer:** `bp-train` will need code that is optimized for run- and compile time whereas the integration methods in `bpbench.mechanistic` only care about correctness, mostly for testing purposes. Therefore we should reimplement these things in `bp-train`. However, we should take inspiration from patterns like the feed classification etc.

### 2.4 `DiscreteEvents` field is not used by the spec

`BioProcess.discrete_events` is `Optional[DiscreteEvents]` with fields `times`,
`labels`, `metadata`. The spec mentions bolus events but never says where their
source times come from. The `input.json` already contains `detected_jumps`
metadata for volume traces (under the `hybrax` metadata block).

For the bolus approximation (spec §11.3), prep code needs to know bolus times.
The spec does not say whether these come from `DiscreteEvents.times` filtered by
label, from `detected_jumps` metadata in the input JSON, or from user code in
`custom.py`.

**Action:** Specify the bolus source. Likely: `DiscreteEvents` with a label
convention (e.g. `"bolus"`) or a fallback to user-supplied times in
`transform_controls`.

**Answer**: For v1 we only consider discrete events that are detected by hybrax prep and thus in the input JSON. Jumps with positive `delta_V` are bolus feed, jumps with negative `delta_V` are sampling events.

### 2.5 `SampleVolumeChange` values are already present in bpbench

The spec's default `V_sample_acc` construction (§10.6) detects "negative
discontinuities in the volume trace". The bpbench `Volume` dataclass already
structures volume changes into typed subtypes:

- `FeedVolumeChange` — positive, carries `feed_medium` composition.
- `SampleVolumeChange` — negative (values ≤ 0), no medium.

Prep code should read `SampleVolumeChange` objects from `process.volume.volume_changes`
rather than detecting negative discontinuities heuristically. The heuristic is
only needed when no `SampleVolumeChange` entries exist in the artifact.

**Action:** Update §10.6 to say: use `SampleVolumeChange` entries when present;
fall back to discontinuity detection only when `volume.volume_changes` contains
no `SampleVolumeChange` instances.

**Answer:** Important: v1 should not include any functionality for detecting volume jumps etc.! This should have already happened during pre-processing with hybrax-prep. We only use the detected jumps already present in the JSON (or alternatively users can detect jumps in user code themselves).

---

## 3. Controls Pipeline — Spec Ambiguities

### 3.1 Segment collapse changes the build strategy but §11 is not fully updated

Answers-3.md Q7 introduced a major architecture change: no segmented public
API. The strategy is now:

1. For each original spline segment, evaluate values and first derivatives on a
   dense fixed-point grid.
2. Combine grids into one monotone time grid per experiment.
3. Build a single linear interpolation payload over the full time span.
4. Record segment-boundary times as `step_ts` for the solver's step size
   controller.

Spec §11 still uses the language "split conceptually at known segment
boundaries" and "evaluate each control … on dense per-segment grids", which is
consistent with this, but §11.2 also mentions `eval_segment(...)` API methods
that are explicitly **rejected** in answers-3.md Q7.

**Action:** Remove `eval_segment` and `eval_segment_batch` from §11. Replace
with a clear statement: the controls payload is a single dense linear
interpolation array; segment boundaries survive only as `step_ts`.

**Answer:** The reviewer is right. We should be consistent here and fully follow the new strategy of bolus ramps and no segments. We can actually refine it even more: Let's create ramps and `step_ts` for `V_sample_acc` as well (instead of worrying about instantaneous jumps and using `jump_ts`). Then we're using the same pattern for both samples and bolus.

### 3.2 The "increase grid density until error is below threshold" rule needs a concrete algorithm

The spec says "increase grid density until the linear interpolation error is
below a chosen threshold". The answers say "use a simple fixed number of points
per segment and increase it if error is too high". These two descriptions are
compatible but under-specified:

- What error metric? Max absolute error? Relative error?
- What threshold? Fixed in code, or configurable?
- What increase strategy? Double points? Add a fixed increment?
- Evaluated against what reference? The original higher-order spline?

Without a concrete algorithm the test in §19.1 ("dense-grid linear
interpolation error is below threshold") has no pass criterion.

**Action:** Add a concrete algorithm description and expose the threshold as a
configurable parameter with a sensible default (e.g. `max_rel_error = 1e-4`).

**Answer:** Let's have a configurable `max_rel_error` (default 1e-4) here which is relative to the spread of values in the trace in questions across all experiments (so that we don't have issues when a trace is constant in an experiment).

### 3.3 Elevation of linear/quadratic splines — `Interpolator` kinds

The spec says "elevate linear or quadratic splines to cubic exactly" (§11.1).
The `Interpolator.kind` field supports `interpax_cubic`, `interpax_linear`, and
`interpax_ppoly`. The elevation step must handle all three kinds. The spec does
not specify what to do when `kind = "interpax_ppoly"` (breakpoint/coefficient
form), which is a different representation.

**Action:** Specify the elevation rule for each `kind`, or restrict V1 input to
`interpax_linear` and `interpax_cubic` only and fail fast otherwise.

**Answer:** Since we're linearly interpolating all controls in v1 we actually don't need to elevate to cubic for now (this would only be necessary if we want to create linear combinations of splines or similar). We should drop mention of spline elevation from the spec entirely.

### 3.4 Global padding — padded shape written to `prepared.json` or derived at load time?

Spec §11.4 says the prepared artifact must record max grid length, number of
controls, and padding info. But §6.2 says "the prepared artifact does not have
to store the final padded runtime arrays verbatim — V1 allows those arrays to be
materialized at training-load time."

These two statements are not contradictory, but the boundary is unclear: does
`prepared.json` store the actual dense interpolation grid (padded), or only the
metadata needed to reconstruct it? If the latter, the training-load step must
re-run dense grid construction, which means the preparation is not truly a
one-time build.

**Action:** Make a concrete decision: store the full padded dense grid in
`prepared.json`, or re-materialize at load time. Document the chosen approach
and the reasoning.

**Action:** Let's make our lives easier and just store the padded arrays in `prepared.json`; no ambiguity and less faff.

---

## 4. Volume and Wrapper Contract — Gaps

### 4.1 `V_cont` initial condition

The spec says `V_cont` is an integrated state initialized from volume data. It
does not say what the initial value is. `Volume.initial_volume` in bpbench is
the natural source, but the spec should say this explicitly since initial
conditions have their own section (§15.2) that only addresses concentration
state variables.

**Action:** Add a bullet to §15.2: `V_cont(0) = process.volume.initial_volume`.

**Answer:** We should get the initial volume from the volume trace in the input JSON.

### 4.2 State vector ordering — where does `V_cont` sit?

The spec defines the state vector as containing measured concentration states
plus `V_cont`, but never specifies where `V_cont` is placed (first, last, or
configurable). This affects `y0`, `y_meas` column ordering, and model input
contracts.

**Action:** Specify `V_cont` position in the state vector (e.g. always last, or
config-defined). Also clarify that `y_meas` does **not** include `V_cont` unless
volume is a measured target.

**Answer:** `V_cont` should always be last.

### 4.3 Modeled feeds and `ReactionOutputs.modeled_feed_rates`

The spec says `modeled_feed_rates` is "a vector aligned with config-declared
modeled feed streams" and if there are none it is "an empty vector". It is
silent on:

- What the config declaration looks like. Is it a list of names? Indices?
- How bp-train validates that the returned vector has the right length.
- Whether the wrapper reads `f_modeled` by index or by name lookup.

This matters because `bpbench.mechanistic.RhsOde` already accepts `f_modeled`
by position — the same convention could be adopted.

**Action:** Add a concrete config declaration format for modeled feeds (e.g.
`modeled_feed_names: List[str]` in `custom.py`) and specify index alignment
rules.

**Answer:** We can use the same convention as in `bpbench.mechanistic.RhsOde`.

### 4.4 Feed medium coverage validation

`bpbench.validate_volume_change_states` already checks that every positive
volume change's feed medium covers all dynamic reactor state variables. The spec
(§9.3) lists validation rules but does not say whether this existing validator
is called or whether bp-train re-implements the check.

**Action:** Add to §9.3: call `bpbench.validate_process` and
`bpbench.validate_case_study` as a first-pass validation before any custom
transforms run.

---

## 5. Model Abstraction — Open Points

### 5.1 `partition_trainable()` return type

The spec says the user implements `partition_trainable()` for
trainable/static partitioning. It does not specify what the return type should
be. Equinox uses a filter function or a pytree-matched boolean mask (via
`eqx.partition`). The trainer needs to know the exact calling convention.

**Action:** Specify the contract, for example:
`partition_trainable() -> Tuple[eqx.Module, eqx.Module]` returning
`(trainable_leaves, static_leaves)` in the style of `eqx.partition`.

**Answer:** Yes, it should return two PyTrees of the same eqx.Module (similar to how it's done in the hybrax-train reference).

### 5.2 Default `partition_trainable()` behavior

The answers say the default should "take all neural network params if not
implemented". But detecting "neural network params" generically requires
inspecting the module tree for `eqx.nn` leaf types. This is fragile.

A more robust default: treat the entire module as trainable (i.e. partition
returns `(model, eqx.nn.StateIndex())` or equivalent). Document the default
clearly so researchers know to override when they have static parameters (e.g.
fixed kinetic constants).

**Answer:** Similar to hybrax-train the user RHS eqx.Module should have a `.model` attribute (the neural net). The params of this should be returned as trainable per default.

### 5.3 `observe()` and its post-hoc limitation

The spec (§12.4) correctly notes that `observe()` as a post-hoc call is
incompatible with stateful models. This note should be more prominent: it is not
just a future concern but a constraint that affects how researchers design their
models even in V1. If a researcher writes an `observe()` that depends on
internal model state at each timestep, the post-hoc call will silently give
wrong results.

**Action:** Add an explicit warning in the spec: "In V1 `observe()` is called
once on the full integrated trajectory. Models that require per-step observation
(e.g. growth-rate reconstruction from latent LSTM state) are not supported and
will produce incorrect results without an error."

**Answer:** Let's change this to instead of post-hoc rather call observe during integration for saving output (similar to hybrax-train; have another look at how it's done there).

---

## 6. `prepared.json` Metadata — Gaps

### 6.1 Namespace collision risk

The spec says to use `metadata["bp_train"]` as a namespace (§17). The
`input.json` already has a `metadata["hybrax"]` block. Confirm that
`save_process_collection_json` preserves existing top-level metadata keys when
writing the prepared artifact.

**Action:** Test round-trip preservation of existing metadata keys and document
the behavior.

### 6.2 Hash of what exactly?

The spec says "hash of the input JSON" (§17). The `input.json` contains float
arrays serialized as lists. JSON serialization of floats is not bit-stable
across Python versions and array shapes. A better strategy is to hash the raw
bytes of the input file.

**Action:** Specify "SHA-256 hash of the raw bytes of the input JSON file" and
confirm the hash is stored as a hex string.

### 6.3 Config hash / path

The spec says "config path or config hash". Since `custom.py` is code, hashing
it is more useful than a path (paths break across machines). The spec should
clarify what gets hashed: the `custom.py` file bytes, a subset of config
objects, or both.

**Answer:** Let's hash the `custom.py` file bytes as well as the JSON bytes.

---

## 7. Testing — Gaps and Additions Needed

### 7.1 Missing test: `V_cont` / `V_real` integration

§19.2 tests bolus and reconstructed `V_real`. There is no explicit test
verifying that `V_real = V_cont - V_sample_acc` matches the measured volume
trace at measurement times (within tolerance). This is the primary numerical
correctness check for the sampling model.

### 7.2 Missing test: feed medium coverage in wrapper

§19.3 does not include a test that verifies the wrapper correctly uses
`FeedMedium.components` for inlet concentrations. This should be tested with
a multi-component feed medium with at least one species absent (verifying the
correct zero-feed term).

### 7.3 Missing test: `partition_trainable()` gradient isolation

§19.4 says "only parameters returned by `partition_trainable()` receive
gradients". The test should also verify that static parameters receive
zero gradient (or are truly frozen, not just having small gradients).

### 7.4 Control prep tests: error threshold test needs a reference implementation

§19.1 says "dense-grid linear interpolation error is below threshold". The test
needs a concrete fixture: a known spline, a dense grid, a linear interpolant
evaluated on that grid, and a measured max error. Without a fixture this test
cannot be written.

---

## 8. Implementation Order — One Suggested Adjustment

The spec's implementation order (§20) places the wrapper at step 6 and the
trainer at step 7. Given that the wrapper depends on the volume/feed semantics
resolved in step 3 (control prep), this order is correct.

One addition worth considering: make `custom.py` for the `input.json` dataset
a first-class deliverable, produced in parallel with step 2. Having a concrete
case study wired up early catches API design problems before they solidify.

---

## 9. Non-Goals — Verification

The following V1 non-goals are confirmed consistent across the spec and all
three Q&A threads:

- Pseudo-batch integration: excluded. (answers.md Q2, answers-3.md Q4 dynamic-only)
- Stateful models: excluded, with explicit note required. (answers-2.md Q9, answers-3.md Q6)
- Data augmentation: excluded. `augment_training_data` hook dropped. (answers-3.md Q1)
- Checkpointing: excluded.
- LOO-CV: excluded.
- Fixed/externally supplied volume: excluded. (answers-3.md Q4)
- Segmented public controls API: excluded in favor of single dense payload. (answers-3.md Q7)

---

## 10. Minor Issues

| Location | Issue |
|---|---|
| §5.1 | Says "loader uses `bpbench.serialization.load_process_collection(...)`". Confirm the exact function name: `load_process_collection_json` (JSON path variant) vs `load_process_collection` (base path variant). Both exist. |
| §8.2 | Hook signature uses `process` but the loaded object is a `BioProcessCollection`. Clarify: hooks are called per-process (iterating the collection) or on the full collection? |
| §10.2 | "deterministic fallback" ordering example says "continuous controlled volume changes, controlled process variables" — confirm this matches `ControlSplines.control_names` ordering convention from `bpbench.mechanistic` for consistency. |
| §12.2 | `modeled_feed_rates` described as "empty vector" when no modeled feeds. Clarify shape: `jnp.zeros(0)` or `jnp.zeros((0,))`? JAX shape semantics matter for JIT. |
| §15.1 | "`y_meas` uses config-defined column order" — specify whether the config lists names (strings) or indices, and whether order must be a permutation of all measured states or may be a subset. |
| §18 | File layout lists `model_api.py` and `defaults.py` without description. Add one-line purpose for each, or remove them to avoid confusion. |

---

## 11. Summary of Actions

| Priority | Item |
|---|---|
| Closed | Use `interpolator` directly as the bp-train API. |
| Blocker | Decide whether bpbench.mechanistic is reused or replaced; document the choice in the spec. |
| High | Specify bolus event source (DiscreteEvents vs custom.py vs detected_jumps metadata). |
| High | Update §10.6 V_sample_acc construction to prefer SampleVolumeChange entries over heuristic detection. |
| High | Remove eval_segment API from §11 and fully replace with single dense interpolation description. |
| High | Specify `partition_trainable()` return type and calling convention. |
| High | Clarify whether full padded dense grid is stored in prepared.json or rebuilt at load time. |
| Medium | Specify dense-grid error metric, threshold, and increase strategy concretely. |
| Medium | Add `V_cont(0) = initial_volume` to §15.2 initial conditions. |
| Medium | Specify state vector ordering for `V_cont`. |
| Medium | Add `observe()` V1 limitation warning for post-hoc-only calling. |
| Medium | Add missing tests: V_real integration check, feed medium coverage, static parameter gradient isolation. |
| Low | Clarify hash strategy (raw file bytes, hex SHA-256). |
| Low | Clarify hook calling convention (per-process vs per-collection). |
| Low | Verify BenchmarkDataset dataclass/serializer mismatch status and close bpbench-api-notes item if fixed. |
