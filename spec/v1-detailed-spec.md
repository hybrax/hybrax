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

- Load `bpbench` JSON artifacts using `bpbench` deserialization.
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
- The loader uses `bpbench.serialization.load_process_collection(...)`.
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

The prepared artifact does not have to store the final padded runtime arrays
verbatim. V1 allows those arrays to be materialized at training-load time from
the transformed `bpbench` data plus prep metadata.

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

### 8.3 Allowed Hook Responsibilities

Hooks may:

- convert cumulative traces to rates,
- combine traces into derived controls,
- derive `V_sample_acc` from a measured volume trace,
- rename variables,
- update `is_controlled`,
- attach or replace `Interpolator` payloads,
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
2. Resolve case-study-specific config in code.
3. Apply `transform_controls(process, config)` to each process.
4. Apply `transform_states(process, config)` to each process.
5. Build default derived controls required by the V1 runtime contract when they
   were not already provided by user code.
6. Validate control roles, state roles, feed semantics, and required
   interpolators.
7. Generate control interpolation payloads and training metadata.
8. Compute any required model scaling statistics.
9. Update or add prep metadata.
10. Serialize the full transformed collection as `prepared.json`.

### 9.3 Validation Rules

Prep must fail fast if:

- a config-declared control is missing,
- a required `Interpolator` is missing,
- feed stream metadata is underspecified,
- a required initial condition cannot be constructed,
- control ordering is inconsistent,
- shapes or units are irreconcilable,
- a variable is required both as a measured state and a control without a
  clearly declared role.

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

This order must be written into the prepared artifact metadata.

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
- any required mapping to state/species order.

### 10.5 Dynamic Volume Only

V1 supports dynamic volume only, but the integrated volume state is not the
fully realized reactor volume.

If the raw dataset contains volume but not explicit feed rates, the user may
derive feed-rate controls from volume in `transform_controls(...)`.

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

V1 provides a default rule for constructing `V_sample_acc` when user code has
not already done so.

Default behavior:

- inspect the available volume trace,
- detect negative discontinuities in that trace,
- interpret those discontinuities as sampling events,
- accumulate their magnitudes into a monotone cumulative control
  `V_sample_acc(t)`,
- write the resulting derived control and its metadata into `prepared.json`.

This default must be overrideable in `custom.py`.

Reasons for override include:

- no usable volume trace is present,
- the user wants to provide explicit sampling times and amounts,
- negative volume discontinuities should not be interpreted as sampling,
- the case study requires a different reconstruction rule.

## 11. Control Representation for V1 Runtime

V1 no longer exposes a segmented public controls API. Instead it uses a dense,
single-interpolator representation built during preparation.

### 11.1 Build Strategy

For each experiment:

- start from `bpbench` `Interpolator` objects,
- elevate linear or quadratic splines to cubic exactly before downstream
  handling,
- split conceptually at known segment boundaries if needed,
- evaluate each control and its first derivative on dense per-segment grids,
- increase grid density until the linear interpolation error is below a chosen
  threshold,
- combine those grids into one monotone time grid per experiment,
- build a single linear interpolation payload over the full experiment.

### 11.2 Segment Boundaries and Events

V1 does not expose segment iteration to downstream users, but it does preserve
boundary information for solver stepping.

The controls payload keeps:

- a dense time grid,
- control values,
- control derivatives,
- step boundary times for the step size controller.

In addition, the controls payload must contain a cumulative sampling control
`V_sample_acc(t)` with discontinuities at sampling times. Those sampling times
must also be present in the boundary/jump-time information passed to the solver.

### 11.3 Bolus Approximation in V1

Bolus and sampling are treated differently in V1.

Bolus events are not modeled as true instantaneous runtime events in the
controls layer. Instead:

- bolus feed events are approximated during prep as short periods of high feed
  rate,
- the duration is the shortest time difference in the original online data,
- tests must verify that the integrated added amount matches the intended bolus
  amount.

Sampling events are represented by a dummy control variable
`V_sample_acc(t)`, the total sampled volume removed up to time `t`.

- `V_sample_acc` is piecewise-constant with discontinuities at sampling times,
- `V_sample_acc` is stored in `prepared.json` as a derived control trace with an
  `Interpolator` and provenance metadata,
- sampling times are forwarded as `jump_ts` or equivalent solver-guidance times,
- the wrapper reconstructs `V_real = V_cont - V_sample_acc` at each RHS call,
- tests must verify that reconstructed physical volume matches the intended
  sampled-volume history.

This is a deliberate V1 simplification and should be documented as such.

### 11.4 Global Padding

The control store uses one global padded shape across all experiments in the
prepared dataset. This is required to avoid JIT recompilation.

The prepared artifact or derived training object must record:

- number of experiments,
- maximum grid length,
- number of controls,
- any padding lengths or masks needed downstream.

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
  streams. If there are no modeled feeds, this is an empty vector.

The exact container type may be a small dataclass, namedtuple, or equivalent,
but the two fields above are semantically required.

### 12.3 Reaction-Only Contract

The user reaction module does not implement dilution in V1.

It is responsible only for reaction or source terms in concentration space.
The wrapper handles:

- continuous-volume-state mechanics,
- reconstruction of physical volume from sampling history,
- feed transport,
- dilution terms,
- combining reaction and transport contributions into the final state
  derivative.

### 12.4 Observations

`observe(...)` remains in the abstraction but defaults to identity.

In V1 the trainer may call it post-integration.

The spec should explicitly note that this boundary may need revision once
stateful models are introduced, because post-hoc observation reconstruction is
not equivalent to online observation generation for models with internal state.

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
- `ReactionOutputs.modeled_feed_rates` must be aligned with the configured list
  of modeled feed streams.

There is no partially automatic mode in which some dilution terms come from the
wrapper and others are manually implemented in user code.

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
- uses config-defined column order,
- is padded across experiments for batching.

### 15.2 `y0`

- uses first measured values for observed states,
- uses config/code overrides for modeled but unobserved states,
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
- realized reactor volume reconstructed in the wrapper as
  `V_real = V_cont - V_sample_acc`.

### 16.2 Step Boundaries

The step size controller should receive boundary times derived during control
preparation. This reduces solver pathologies around discontinuity-like regions
without exposing a segmented public API.

These boundary times must include sampling-event times because `V_sample_acc`
has discontinuities there.

### 16.3 Expected Runtime Objects

The runtime path should be JIT-friendly and rely on globally padded arrays so
that all experiments in the prepared artifact can share one compiled shape.

## 17. Metadata in `prepared.json`

At minimum, the prepared artifact metadata should include:

- source input path or dataset id,
- hash of the input JSON,
- prep timestamp,
- config path or config hash,
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

This is illustrative only; exact filenames may change.

## 19. Testing Requirements

V1 should prioritize tests that de-risk the chosen simplifications.

### 19.1 Control Prep Tests

- exact elevation of linear/quadratic source splines,
- dense-grid linear interpolation error is below threshold,
- global padding shapes are stable,
- config-defined control ordering is preserved,
- missing controls fail fast,
- cumulative-to-rate transformation hooks work as expected,
- default `V_sample_acc` construction from negative volume discontinuities works
  as expected,
- user override of default `V_sample_acc` construction works as expected.

### 19.2 Event Approximation Tests

- bolus approximation adds exactly the intended amount,
- boundary times are passed through correctly,
- reconstructed `V_real` matches the intended sampled-volume history,
- solver does not exhibit excessive rejected steps around approximated events.

### 19.3 Wrapper Tests

- dilution and feed transport are applied correctly,
- multiple feed streams sum correctly,
- dynamic volume state updates correctly,
- user reaction terms are combined correctly with wrapper-managed transport.

### 19.4 Trainer Tests

- a single train step produces gradients,
- only parameters returned by `partition_trainable()` receive gradients,
- measurement-time loss uses the expected padded arrays and masks.

## 20. Implementation Order

Recommended implementation order:

1. preparation pipeline that loads raw `bpbench` JSON and writes `prepared.json`,
2. validation layer and prep metadata,
3. control preparation path with dense linear interpolation and global padding,
4. training data object with padded measurement arrays,
5. user model abstraction and `partition_trainable()` flow,
6. library wrapper with full dilution handling,
7. minimal trainer with one train step and tests,
8. then evaluate whether the architecture is good enough before adding more
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
- adaptive knot placement beyond the simple dense-grid refinement strategy.
