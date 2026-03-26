# BP-Train V1 Detailed Spec

## 1. Purpose

`bp-train` is the hybrid-model training package of the BP ecosystem. It consumes
`bpbench`-compatible serialized bioprocess data, prepares a training-ready
artifact, exposes a small researcher-facing model API, and trains hybrid models
with standardized handling of controls, feeds, volume, and dilution.

V1 is intentionally narrow. The goal is to get a reliable train step working on
real measurements with a clear architecture that can later absorb data
augmentation, checkpointing, LOO-CV, and stateful models.

## 2. V1 Goals

- Load `bpbench` `BioProcessCollection` JSON artifacts using `bpbench`
  deserialization.
- Standardize controls and measured states into a training-ready prepared
  artifact.
- Support a hybrid config approach:
  - persisted JSON artifacts for data,
  - code-level configuration and customization in `custom.py`,
  - optional lightweight JSON for run settings.
- Build a controls representation that avoids recompilation by using globally
  padded shapes.
- Convert controls to a dense linear-interpolation payload so the solver only
  performs a single interpolation lookup per experiment at runtime.
- Provide a library-owned RHS wrapper that handles all dilution and feed
  transport in V1.
- Support only stateless user models in V1.
- Train only against real measurement timestamps in V1.
- Provide strong validation and fail-fast behavior.
- Keep a concrete running note of `bpbench` API changes or adapters required by
  this package.

## 3. Explicit Non-Goals for V1

- Pseudo-batch integration.
- Stateful models such as RNNs or LSTMs.
- Checkpointing and resumption.
- LOO-CV orchestration.
- Data augmentation.
- A fully general segmented runtime controls API.
- Wrapper modes where the user partially handles dilution and the library
  handles the rest.
- Support for fixed/externally supplied volume. V1 supports dynamic volume only.

## 4. Design Principles

- `bpbench` remains the semantic source of truth for process data structures.
- The prepared artifact is explicit and persisted; training should not re-run
  bespoke preprocessing code on every invocation.
- Configuration is split by responsibility:
  - data artifacts stay declarative,
  - structural model setup and custom transformations live in code,
  - simple run settings may live in JSON.
- V1 must be strict. Missing metadata, invalid shapes, or ambiguous control
  semantics should fail at prep time rather than training time.
- Library code should own generic bioprocess mechanics such as feed transport,
  dilution, and volume integration.
- User code should own case-study-specific semantics, model definition, and ML
  representation choices such as scaling.

## 5. High-Level Architecture

V1 is split into three phases.

### 5.1 Phase A: Raw Data Load

- Input artifact is a `bpbench` JSON file, usually emitted by `hybrax-prep` or a
  `bpbench` example workflow.
- The loader uses `bpbench.serialization.load_process_collection_json(...)`
  when given a JSON file path and otherwise follows the
  process-collection-first `bpbench` loading path.
- The loaded object is treated as a `BioProcessCollection` plus metadata.

### 5.2 Phase B: Preparation

- `bp-train` applies config-driven and code-driven transforms to build a new
  prepared `bpbench` JSON artifact.
- This phase is the only place where custom case-study preprocessing is
  expected.
- Output is `prepared.json`, which contains all original fields plus updated
  control/state metadata and prep provenance.

### 5.3 Phase C: Training

- Training code reads `prepared.json`.
- It builds a training dataset with padded measurement arrays and control
  interpolation payloads.
- The trainer calls a library-owned wrapper around a user-defined model.

## 6. Artifact Model

### 6.1 Raw Artifact

The raw artifact is a `bpbench` JSON file such as `input.json`. It is assumed to
be valid `bpbench` data, but it is not assumed to already be training-ready.

Important implications:

- `is_controlled` flags may be incomplete or not aligned with training needs.
- A trace may not indicate whether it is cumulative or already a rate.
- Reactor/feed semantics may still need case-study-specific enrichment.
- Variable roles may need to be defined outside the raw JSON.

### 6.2 Prepared Artifact

The prepared artifact is a full `bpbench` JSON file derived from the raw one.
It must preserve all fields from the input and may modify or add:

- `is_controlled` flags to align with the training config,
- `Interpolator` payloads for transformed controls or states,
- derived control traces required by the V1 runtime contract,
- prep metadata,
- training metadata,
- model-scaling metadata,
- control ordering metadata,
- explicit feed semantics metadata.

The prepared artifact is the canonical input to V1 training.

The prepared artifact stores the final padded runtime arrays needed by V1
training. V1 does not rebuild those arrays at training-load time.

### 6.3 Optional Run Config

An optional lightweight JSON config may store:

- paths,
- optimizer hyperparameters,
- solver tolerances,
- batch size,
- random seed,
- logging options.

This JSON should not be responsible for rich structural semantics that are
better expressed in code.

## 7. Hybrid Configuration Approach

V1 uses a hybrid configuration model rather than a JSON-only system.

### 7.1 What Lives in Persisted JSON

- raw `bpbench` process data,
- prepared `bpbench` process data,
- prep provenance,
- training metadata derived during preparation,
- scaling statistics derived during preparation when needed,
- simple run settings if desired.

### 7.2 What Lives in Code

Code-level configuration lives in `custom.py` and is the place for:

- user model definition,
- model input and output scaling logic,
- trainable partitioning,
- control transformation hooks,
- state transformation hooks,
- reactor/feed semantic enrichment when the raw artifact is missing training-
  required medium definitions,
- case-study-specific mappings that are too structural or semantic for plain
  JSON.

This is intentionally similar in spirit to `hybrax-train`, but with a clearer
division between persisted artifacts and runtime customization.

Code-level configuration may directly construct, mutate, or enrich `bpbench`
dataclass objects before serialization. This code-first workflow is explicitly
supported in V1 and mirrors the style already used in `bpbench` examples and
notebooks.

### 7.3 Why V1 Uses This Split

- `bpbench` dataclasses already carry the true semantics of the domain.
- feed and volume semantics are too structural to encode comfortably in ad hoc
  JSON fields alone,
- model partitioning and scaling are better expressed as code,
- persisted artifacts still give reproducibility and inspectability.

## 8. `custom.py` Responsibilities

V1 uses a `custom.py` module for all non-default researcher code.

### 8.1 Required or Expected Contents

- user model definition,
- control preprocessing hook,
- state preprocessing hook,
- optional observation model,
- optional explicit config objects defined as Python code,
- optional code that constructs or enriches `bpbench` dataclass objects before
  writing `prepared.json`.

### 8.2 Default Hook Signatures

V1 standardizes the following default hook signatures:

```python
def transform_controls(process, config):
    return process


def transform_states(process, config):
    return process
```

Data augmentation hooks are intentionally deferred.

These hooks are called per process while iterating over the loaded
`BioProcessCollection`.

### 8.3 Allowed Hook Responsibilities

Hooks may:

- convert cumulative traces to rates,
- combine traces into derived controls,
- derive `V_sample_acc` from precomputed jump metadata or user-supplied
  sampling schedules,
- rename variables,
- update `is_controlled`,
- attach or replace `Interpolator` payloads,
- populate or repair `reactor_medium.components`,
- populate or repair feed-medium component metadata needed for downstream
  transport and dilution,
- declare biomass and other state/species semantics needed by the training
  runtime,
- compute scaling statistics,
- enrich process metadata needed for training.

Hooks should not:

- perform training,
- mutate runtime training state,
- leave ambiguous semantics unresolved.

## 9. Preparation Pipeline

The preparation step converts raw `bpbench` data into `prepared.json`.

### 9.1 Inputs

- raw `bpbench` JSON,
- code-level config from `custom.py`,
- optional lightweight run config.

### 9.2 Processing Steps

1. Load the raw `BioProcessCollection`.
2. Run first-pass `bpbench` validation on the raw collection, at minimum by
   calling `bpbench.validate_process(...)` for each process before custom
   transforms.
3. Resolve case-study-specific config in code.
4. Apply `transform_controls(process, config)` to each process.
5. Apply `transform_states(process, config)` to each process.
6. Enrich reactor/feed component semantics in code when the raw artifact does
   not yet contain the medium definitions required for training.
7. Build default derived controls required by the V1 runtime contract when they
   were not already provided by user code.
8. Validate control roles, state roles, feed semantics, reactor/feed component
   completeness, and required interpolators.
9. Run strict post-transform `bpbench` validation on the prepared processes
   before writing `prepared.json`.
10. Generate control interpolation payloads, padded runtime arrays, and
    training metadata.
11. Compute any required model scaling statistics.
12. Update or add prep metadata.
13. Serialize the full transformed collection as `prepared.json`.

### 9.3 Validation Rules

Prep must fail fast if:

- a config-declared control is missing,
- a required `Interpolator` is missing,
- feed stream metadata is underspecified,
- feed-media coverage is invalid for a positive feed stream,
- `reactor_medium.components` or feed-medium component metadata is missing for a
  dataset that is intended for training,
- biomass or other required dynamic species semantics are missing after prep,
- a required initial condition cannot be constructed,
- control ordering is inconsistent,
- shapes or units are irreconcilable,
- a variable is required both as a measured state and a control without a
  clearly declared role.

This validation layer should reuse existing `bpbench` checks where possible
rather than reimplementing them from scratch.

The intended V1 contract is that strict post-transform `bpbench` validation runs
before `prepared.json` is written. If the raw input is incomplete, the
expectation is that `custom.py` enriches the required medium/species semantics
during prep rather than downstream code working around them later.

## 10. Control Semantics

### 10.1 Generic Principle

The controls representation should be generic and not tied to the old
`[D, Cf_norm, T]` pseudo-batch interface.

### 10.2 Control Ordering

Control ordering is:

1. config-defined if explicitly provided,
2. otherwise deterministic fallback:
   - continuous controlled volume changes,
   - controlled process variables.

This matches the ordering convention already used by
`bpbench.mechanistic.ControlSplines` and must be written into the prepared
artifact metadata.

### 10.3 Rates vs Cumulative Traces

The stored control payload must represent actual control values consumed by the
solver. For continuous feeds and similar volume changes, this means rate
signals, not cumulative quantities.

If the raw artifact does not distinguish cumulative from rate traces, the user
must resolve this in `transform_controls(...)`.

### 10.4 Feed Streams

Feed streams are represented explicitly as separate controls. V1 does not
collapse them into a single scalar dilution term.

Each feed stream must have explicit metadata sufficient for the wrapper to
compute transport and dilution:

- name,
- source kind: `control` or `modeled`,
- source index in the aligned control vector or modeled-feed vector,
- inlet composition,
- explicit component metadata aligned to the prepared species/state semantics,
- any required mapping to state/species order.

### 10.5 Dynamic Volume Only

V1 supports dynamic volume only, but the integrated volume state is not the
fully realized reactor volume.

If the raw dataset contains volume but not explicit feed rates, the user may
derive feed-rate controls from preprocessing outputs in `transform_controls(...)`.

The V1 runtime contract is:

- integrate a continuous volume-like state `V_cont`,
- represent cumulative sampled volume as a separate control variable
  `V_sample_acc`,
- reconstruct the physical reactor volume inside the wrapper as
  `V_real = V_cont - V_sample_acc`.

This allows the solver to avoid representing sampling as direct state jumps
while still using the correct realized volume for dilution calculations.

`V_sample_acc` is a persisted derived control in `prepared.json`, not only a
runtime-only helper.

### 10.6 Default Sampling-Control Construction

V1 does not detect volume jumps on its own. It consumes jump information that
was already detected during upstream preprocessing, typically by `hybrax-prep`,
or information provided explicitly by user code.

Default behavior:

- read jump annotations already present in the input or prepared JSON,
- interpret jumps with negative `delta_V` as sampling events,
- accumulate their magnitudes into a monotone cumulative control
  `V_sample_acc(t)`,
- approximate each increment as a short ramp rather than an instantaneous jump,
- write the resulting derived control and its metadata into `prepared.json`.

This default must be overrideable in `custom.py`.

Reasons for override include:

- no usable upstream jump metadata is present,
- the user wants to provide explicit sampling times and amounts,
- the case study requires a different reconstruction rule.

## 11. Control Representation for V1 Runtime

V1 no longer exposes a segmented public controls API. Instead it uses a dense,
single-interpolator representation built during preparation and persisted in
`prepared.json`.

### 11.1 Build Strategy

For each experiment:

- start from `bpbench` `Interpolator` objects,
- reimplement the runtime controls path inside `bp-train` rather than reusing
  `bpbench.mechanistic`, because V1 optimizes for runtime and compile-time
  behavior rather than reference correctness,
- reuse `bpbench.mechanistic` conventions where useful, especially deterministic
  control ordering and feed classification,
- convert upstream detected positive `delta_V` jumps into short bolus-feed
  ramps,
- convert upstream detected negative `delta_V` jumps into short
  `V_sample_acc` ramps,
- evaluate each control and its first derivative on dense grids,
- start from a fixed initial grid density and then double the number of points
  until the control-wise interpolation error is below the configured threshold
  or a maximum refinement level is reached,
- measure interpolation error against the original source control on a denser
  reference grid,
- use `max_rel_error` as the acceptance criterion, with default `1e-4`,
  normalized by the cross-experiment value spread of the corresponding trace so
  that constant-in-one-experiment traces remain well-defined,
- combine those grids into one monotone time grid per experiment,
- build a single linear interpolation payload over the full experiment.

### 11.2 Segment Boundaries and Events

V1 does not expose segment iteration to downstream users.

The controls payload keeps:

- a dense time grid,
- control values,
- control derivatives,
- step boundary times for the step size controller.

Step boundary times (`step_ts`) must include the start and end times of
approximated bolus ramps, approximated `V_sample_acc` ramps, and any other
control boundaries introduced during preparation.

### 11.3 Bolus Approximation in V1

Bolus and sampling use the same V1 approximation pattern.

- upstream preprocessing provides jump annotations in the JSON,
- jumps with positive `delta_V` are interpreted as bolus-feed additions,
- jumps with negative `delta_V` are interpreted as sampling removals,
- each such jump is converted during prep into a short ramp whose duration is
  the shortest time difference in the original online data,
- bolus ramps contribute to feed-rate controls,
- sampling ramps contribute to the cumulative dummy control `V_sample_acc(t)`,
- `V_sample_acc` is stored in `prepared.json` as a derived control trace with an
  `Interpolator` and provenance metadata,
- `step_ts` includes the boundaries of these ramps,
- the wrapper reconstructs `V_real = V_cont - V_sample_acc` at each RHS call.

Tests must verify that integrated bolus additions and integrated sampled-volume
removals match their intended amounts.

### 11.4 Global Padding

The control store uses one global padded shape across all experiments in the
prepared dataset. This is required to avoid JIT recompilation.

The prepared artifact must record:

- number of experiments,
- maximum grid length,
- number of controls,
- any padding lengths or masks needed downstream,
- the full padded control arrays needed by the V1 runtime.

## 12. Model Abstraction

V1 supports stateless user models only.

### 12.1 User Reaction Module Contract

V1 separates:

- a library-owned wrapper that handles controls, feeds, volume, and dilution,
- a user-owned reaction module that predicts reaction-space quantities.

The user reaction module is an `eqx.Module` owned by `custom.py`.

It must implement:

- `__call__(...)` for reaction prediction,
- `partition_trainable()` for trainable/static partitioning.

It may implement:

- `observe(...)`, default identity.

### 12.2 Concrete Return Contract

To avoid ambiguity, V1 freezes a structured return contract for
`__call__(...)`.

The call must return a `ReactionOutputs`-like structure with:

- `reaction_terms`: reaction or source terms in concentration space, excluding
  dilution and feed transport,
- `modeled_feed_rates`: a vector aligned with config-declared modeled feed
  streams. If there are no modeled feeds, this is `jnp.zeros((0,))`.

The exact container type may be a small dataclass, namedtuple, or equivalent,
but the two fields above are semantically required.

### 12.3 Trainable Partitioning Contract

`partition_trainable()` returns two pytrees with the same module structure:

- a trainable pytree,
- a static or frozen pytree.

This follows the Equinox-style partitioning pattern used in `hybrax-train`.

Default behavior:

- the reaction module should expose a `.model` attribute for the neural network
  submodule,
- if `partition_trainable()` is not overridden, the default implementation
  treats the parameters of `.model` as trainable and the remaining leaves as
  static,
- if `.model` is absent and no custom `partition_trainable()` is provided, prep
  or model construction must fail fast.

### 12.4 Reaction-Only Contract

The user reaction module does not implement dilution in V1.

It is responsible only for reaction or source terms in concentration space.
The wrapper handles:

- continuous-volume-state mechanics,
- reconstruction of physical volume from sampling history,
- feed transport,
- dilution terms,
- combining reaction and transport contributions into the final state
  derivative.

### 12.5 Observations

`observe(...)` remains in the abstraction but defaults to identity.

In V1 it should be called during integration or trajectory saving rather than
as a purely post-hoc transformation of the final trajectory.

V1 still supports stateless models only. Models whose observation path depends
on internal recurrent state are out of scope and should fail design review
rather than being accepted silently.

## 13. Scaling

Scaling is owned by the model layer, not by the generic RHS wrapper.

### 13.1 Why

- scaling is an ML representation concern, not a bioprocess mechanics concern,
- the wrapper should stay generic and unit-aware,
- different case studies may need very different transforms.

### 13.2 How Scaling Is Configured

Scaling may be configured in `custom.py` and parameterized by statistics
computed during the prep phase.

The preparation step may compute and persist metadata such as:

- state means and standard deviations,
- control means and standard deviations,
- output scaling statistics,
- log-transform flags,
- positivity constraints,
- min/max ranges.

### 13.3 Runtime Ownership

The wrapper assembles physical inputs.

The reaction module then:

1. scales or transforms inputs,
2. predicts `ReactionOutputs`,
3. maps those outputs back into physical units or constrained domains.

This can be implemented either as helper submodules or as methods inside the
reaction module. V1 should keep this within the model abstraction rather than
creating a separate global scaling system.

## 14. Runtime Wrapper Contract

The library-owned wrapper is responsible for all dilution in V1.

### 14.1 Inputs

At runtime the wrapper receives:

- current state,
- current continuous volume state `V_cont`,
- control values at time `t`,
- reaction-module outputs,
- feed metadata.

### 14.2 Responsibilities

The wrapper must:

- read all feed-rate controls,
- maintain `V_cont` as part of the integrated state,
- read `V_sample_acc(t)` from the controls object,
- reconstruct `V_real = V_cont - V_sample_acc`,
- compute dilution and feed transport contributions,
- combine those contributions with model reaction terms,
- produce the final derivative for the ODE solver.

### 14.3 Modeled Feeds

V1 should support multiple feed streams explicitly, but it should keep the
runtime contract strict:

- every feed stream must be declared explicitly,
- the wrapper must know its source kind, aligned index, and inlet composition,
- controlled feeds come from the prepared control vector,
- modeled feeds come from `ReactionOutputs.modeled_feed_rates`,
- `ReactionOutputs.modeled_feed_rates` must follow the same positional
  convention used by `bpbench.mechanistic.RhsOde`, aligned to the ordered list
  of explicit modeled feed streams declared in the process metadata.

There is no partially automatic mode in which some dilution terms come from the
wrapper and others are manually implemented in user code.

### 14.4 Relationship to `bpbench.mechanistic`

V1 does not reuse `bpbench.mechanistic` as its runtime implementation.

Instead `bp-train` reimplements the controls and wrapper path with stronger
focus on JIT stability, padded shapes, and compile-time behavior. Existing
`bpbench.mechanistic` patterns remain valuable references for:

- deterministic control ordering,
- feed classification,
- positional modeled-feed conventions,
- wrapper math used as a correctness baseline in tests.

## 15. Training Data Object

The minimum V1 training data object per experiment is:

- `process_name`,
- `t_meas`,
- `y_meas`,
- `y0`,
- `controls`,
- optional `state_interpolators`.

### 15.1 `y_meas`

- contains only measured target variables,
- uses config-defined column order by variable name,
- may be a subset of all dynamic states,
- does not include `V_cont`,
- is padded across experiments for batching.

### 15.2 `y0`

- uses first measured values for observed states,
- uses config/code overrides for modeled but unobserved states,
- appends `V_cont` as the final state entry,
- initializes `V_cont(0)` from the first point of the prepared volume trace in
  the input JSON lineage,
- must be fully resolvable during prep.

### 15.3 Training Grid

V1 trains only against real measurement timestamps.

Deterministic resampling is allowed in the data layer for future use, but it is
not part of the V1 loss contract.

## 16. Solver Behavior

### 16.1 Integration Mode

- real-space integration only,
- no pseudo-batch dynamics,
- dynamic volume represented by an integrated continuous state `V_cont`,
- `V_cont` is always the last entry in the state vector,
- realized reactor volume reconstructed in the wrapper as
  `V_real = V_cont - V_sample_acc`.

### 16.2 Step Boundaries

The step size controller should receive boundary times derived during control
preparation. This reduces solver pathologies around steep ramp regions without
exposing a segmented public API.

These boundary times must include the start and end times of both bolus ramps
and `V_sample_acc` ramps.

### 16.3 Expected Runtime Objects

The runtime path should be JIT-friendly and rely on globally padded arrays so
that all experiments in the prepared artifact can share one compiled shape.

## 17. Metadata in `prepared.json`

At minimum, the prepared artifact metadata should include:

- source input path or dataset id,
- SHA-256 hash of the raw bytes of the input JSON file, stored as a hex string,
- prep timestamp,
- SHA-256 hash of the raw bytes of `custom.py`, stored as a hex string,
- names of applied transform hooks,
- control ordering,
- whether volume is dynamic,
- any updates applied to `is_controlled`,
- provenance for derived controls such as `V_sample_acc`,
- scaling metadata or a reference to where it is stored,
- shape metadata needed to interpret padded arrays.

V1 should store this under an explicit package-specific namespace such as
`metadata["bp_train"]` to avoid collisions with upstream `bpbench` metadata.

Long-term provenance can become more sophisticated, but V1 should still be
explicit enough to support debugging and reproducibility.

## 18. Suggested File Layout for V1

One possible package layout is:

```text
bp_train/
  prepare.py
  controls.py
  training_data.py
  wrapper.py
  trainer.py
  model_api.py
  validation.py
  defaults.py
custom.py
prepared.json
train-config.json
```

`model_api.py` is the home of the reaction-module and `ReactionOutputs`
contracts. `defaults.py` contains default prep and training settings such as
the dense-grid refinement parameters.

This is illustrative only; exact filenames may change.

## 19. Testing Requirements

V1 should prioritize tests that de-risk the chosen simplifications.

### 19.1 Control Prep Tests

- padded runtime arrays are written to `prepared.json` and load back without
  shape ambiguity,
- dense-grid linear interpolation error is below threshold,
- global padding shapes are stable,
- config-defined control ordering is preserved,
- missing controls fail fast,
- cumulative-to-rate transformation hooks work as expected,
- default `V_sample_acc` construction from upstream negative `delta_V` jump
  annotations works as expected,
- user override of default `V_sample_acc` construction works as expected.

### 19.2 Event Approximation Tests

- bolus approximation adds exactly the intended amount,
- boundary times are passed through correctly,
- reconstructed `V_real` matches the measured volume trace at measurement times
  within tolerance,
- reconstructed `V_real` matches the intended sampled-volume history,
- solver does not exhibit excessive rejected steps around approximated events.

### 19.3 Wrapper Tests

- dilution and feed transport are applied correctly,
- multiple feed streams sum correctly,
- feed-medium composition is applied correctly, including zero contribution for
  absent species,
- dynamic volume state updates correctly,
- user reaction terms are combined correctly with wrapper-managed transport.

### 19.4 Trainer Tests

- a single train step produces gradients,
- only parameters returned by `partition_trainable()` receive gradients,
- frozen parameters from `partition_trainable()` remain gradient-free,
- measurement-time loss uses the expected padded arrays and masks.

## 20. Implementation Order

Recommended implementation order:

1. preparation pipeline that loads raw `bpbench` JSON and writes `prepared.json`,
2. validation layer and prep metadata,
3. a concrete `custom.py` for the first real dataset, developed early enough to
   stress the API before it hardens,
4. control preparation path with dense linear interpolation and global padding,
5. training data object with padded measurement arrays,
6. user model abstraction and `partition_trainable()` flow,
7. library wrapper with full dilution handling,
8. minimal trainer with one train step and tests,
9. then evaluate whether the architecture is good enough before adding more
   features.

## 21. Deferred Items

These are explicitly deferred beyond V1:

- data augmentation,
- checkpointing,
- LOO-CV,
- stateful models,
- pseudo-batch dynamics,
- alternative runtime contracts where the user manually handles dilution,
- more sophisticated treatment of bolus events,
- adaptive knot placement beyond the simple dense-grid refinement strategy,
- automatic jump detection inside `bp-train`.
