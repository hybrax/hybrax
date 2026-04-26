# BP-Train V1 Detailed Spec

## 1. Purpose

`bp-train` is the hybrid-model training package of the BP ecosystem. It consumes
`bp_format`-compatible serialized bioprocess data, prepares a training-ready
artifact, exposes a small researcher-facing model API, and trains hybrid models
with standardized handling of controls, feeds, volume, and dilution.

V1 is intentionally narrow. The goal is to get a reliable train step working on
real measurements with a clear architecture that can later absorb data
augmentation, checkpointing, LOO-CV, and stateful models.

## 2. V1 Goals

- Load `bp_format` `BioProcessCollection` JSON artifacts using `bp_format`
  deserialization.
- Standardize controls and measured states into a training-ready prepared
  artifact.
- Support a hybrid config approach:
  - persisted JSON artifacts for data,
  - code-level configuration and customization in `custom.py`,
  - optional lightweight JSON for run settings.
- Build a controls representation that avoids recompilation by using globally
  padded shapes.
- Materialize runtime control payloads as JAX arrays with stable collection-wide
  shapes rather than ragged Python containers.
- Convert controls to a dense linear-interpolation payload so the solver only
  performs a single interpolation lookup per experiment at runtime.
- Provide a library-owned RHS wrapper that handles all dilution and feed
  transport in V1.
- Support only stateless user models in V1.
- Train only against real measurement timestamps in V1.
- Provide strong validation and fail-fast behavior.
- Keep a concrete running note of `bp_format` API changes or adapters required by
  this package.

## 3. Explicit Non-Goals for V1

- Pseudo-batch integration.
- Stateful models such as RNNs or LSTMs.
- Full resumption (optimizer state + RNG) and warmstart from a prior run. Periodic parameter snapshots during training are supported: every `--log-every` steps, `bp-train train` writes `<output-dir>/checkpoints/step_NNNNN/{trained_wrapper.eqx, trained_wrapper.meta.json, loss_curve.png, predictions.csv}` plus a `latest` symlink. Checkpoint directories do not include per-process prediction plot PNGs. Loading a checkpoint as a warmstart init, and restoring optimizer state, are deferred.
- Data augmentation. (Note: a structural placeholder for augmented bioprocesses lives in `bp_format.AugmentedBioProcess`; LOO-CV groups augmented children with their parent so train/holdout splits cannot leak. See `spec/loo.md`.)
- A fully general segmented runtime controls API.
- Wrapper modes where the user partially handles dilution and the library
  handles the rest.
- Support for fixed/externally supplied volume. V1 supports dynamic volume only.

## 4. Design Principles

- `bp_format` remains the semantic source of truth for process data structures.
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

- Input artifact is a `bp_format` JSON file, usually emitted by `hybrax-prep` or a
  `bp_format` example workflow.
- The loader uses `bp_format.serialization.load_process_collection_json(...)`
  when given a JSON file path and otherwise follows the
  process-collection-first `bp_format` loading path.
- The loaded object is treated as a `BioProcessCollection` plus metadata.

### 5.2 Phase B: Preparation

- `bp-train` applies config-driven and code-driven transforms to build a new
  prepared `bp_format` JSON artifact.
- This phase is the only place where custom case-study preprocessing is
  expected.
- Output is `prepared.json`, which contains all original fields plus updated
  control/state metadata and prep provenance.

### 5.3 Phase C: Training

- Training code reads `prepared.json`.
- It builds a training dataset with padded measurement arrays and control
  interpolation payloads.
- The trainer calls a library-owned wrapper around a user-defined model.
- V1 uses a batched training harness:
  - one optimizer update per batch,
  - one JIT-compiled batched train-step function per run (unless explicitly
    rebuilt),
  - per-sample losses computed with `jax.vmap`,
  - parameter updates applied through Optax optimizers.

## 6. Artifact Model

### 6.1 Raw Artifact

The raw artifact is a `bp_format` JSON file such as `input.json`. It is assumed to
be valid `bp_format` data, but it is not assumed to already be training-ready.

Important implications:

- `is_controlled` flags may be incomplete or not aligned with training needs.
- A trace may not indicate whether it is cumulative or already a rate.
- Reactor/feed semantics may still need case-study-specific enrichment.
- Variable roles may need to be defined outside the raw JSON.

### 6.2 Prepared Artifact

The prepared artifact is a full `bp_format` JSON file derived from the raw one.
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

`TimeSeries` data in prepared artifacts should use canonical `bp_format`
`times`/`values` fields. Beyond carrying sampled traces, `TimeSeries` also
supports higher-level operations that V1 code may rely on where useful,
including:

- arithmetic combinations between compatible series,
- differentiation via `deriv(...)`,
- spline-based evaluation/integration for represented series.

The prepared artifact stores structural control metadata and preparation
provenance. Dense interpolation payloads and padded runtime arrays are built by
`ControlsStore` at load time from the prepared process collection.

### 6.3 Optional Run Config

An optional lightweight JSON config may store:

- paths,
- optimizer settings (`optimizer_name`, `learning_rate`),
- solver tolerances,
- batch size,
- random seed,
- logging options.

This JSON should not be responsible for rich structural semantics that are
better expressed in code.

## 7. Hybrid Configuration Approach

V1 uses a hybrid configuration model rather than a JSON-only system.

### 7.1 What Lives in Persisted JSON

- raw `bp_format` process data,
- prepared `bp_format` process data,
- prep provenance,
- training metadata derived during preparation,
- scaling statistics derived during preparation when needed,
- simple run settings if desired.

### 7.2 What Lives in Code

Code-level configuration lives in `custom.py` and is the place for:

- user model definition,
- model input and output scaling logic,
- trainable partitioning,
- process-collection transformation hooks,
- reactor/feed semantic enrichment when the raw artifact is missing training-
  required medium definitions,
- case-study-specific mappings that are too structural or semantic for plain
  JSON.

This is intentionally similar in spirit to `hybrax-train`, but with a clearer
division between persisted artifacts and runtime customization.

Code-level configuration may directly construct, mutate, or enrich `bp_format`
dataclass objects before serialization. This code-first workflow is explicitly
supported in V1 and mirrors the style already used in `bp_format` examples and
notebooks.

### 7.3 Why V1 Uses This Split

- `bp_format` dataclasses already carry the true semantics of the domain.
- feed and volume semantics are too structural to encode comfortably in ad hoc
  JSON fields alone,
- model partitioning and scaling are better expressed as code,
- persisted artifacts still give reproducibility and inspectability.

## 8. `custom.py` Responsibilities

V1 uses a `custom.py` module for all non-default researcher code.

### 8.1 Required or Expected Contents

- user model definition,
- process-collection preprocessing hook,
- optional observation model,
- optional explicit config objects defined as Python code,
- optional code that constructs or enriches `bp_format` dataclass objects before
  writing `prepared.json`.

### 8.2 Default Hook Signatures

V1 standardizes the following default hook signatures:

```python
def transform_process_collection(collection, config):
    return collection


def build_reaction_module(
    *, target_names, process_names, config, seed, collection
):
    ...


def build_learning_rate(config, train_cfg):
    ...


def estimate_all_scales(collection, target_names, config):
    ...


def build_sample_loss_fn(
    *, default_sample_loss_fn, store, collection, train_cfg, config
):
    return default_sample_loss_fn


def build_batched_loss_fn(*, default_loss_fn, store, collection, train_cfg, config):
    return default_loss_fn
```

Data augmentation hooks are intentionally deferred.

`transform_process_collection(...)` is called once and receives the raw
`BioProcessCollection`; it returns the updated collection used for the rest of
prep.

`build_sample_loss_fn(...)` is the preferred optional loss hook. It returns a
per-sample loss callable (same signature/contract as the default sample loss),
and the harness applies batching internally.

For custom per-sample losses that need default measurement loss plus solver-time
observables in one solve, the trainer exposes
`evaluate_sample_from_arrays(...) -> SingleSampleResult`. This returns the default
measurement loss together with measurement-grid `save_outputs(...)`, including
stacked `ReactionOutputs.auxiliary` values.

`build_batched_loss_fn(...)` remains an advanced optional hook for directly returning a
batched loss callable during training.

Hooks are mutually exclusive: defining both loss hooks is an error.
Forward per-process loss evaluation supports default loss and
`build_sample_loss_fn(...)`; `build_batched_loss_fn(...)` is rejected there because
arbitrary batched objectives are not guaranteed to decompose per process.

### 8.3 Allowed Hook Responsibilities

Hooks may:

- convert cumulative traces to rates,
- combine traces into derived controls,
- derive `V_sample_acc` from sample-volume traces or user-supplied sampling
  schedules,
- rename variables,
- update `is_controlled`,
- attach or replace `Interpolator` payloads,
- populate or repair `reactor_medium.components`,
- populate or repair feed-medium component metadata needed for downstream
  transport and dilution,
- declare biomass and other state/species semantics needed by the training
  runtime,
- compute scaling statistics,
- enrich process metadata needed for training,
- rename process keys during prep when upstream artifacts use inconvenient
  names.

Hooks should not:

- perform training,
- mutate runtime training state,
- leave ambiguous semantics unresolved.

Helper utilities for common transform tasks (for example process renaming) may
be added later, but V1 does not require them.

## 9. Preparation Pipeline

The preparation step converts raw `bp_format` data into `prepared.json`.

### 9.1 Inputs

- raw `bp_format` JSON,
- code-level config from `custom.py`,
- optional lightweight run config.

### 9.2 Processing Steps

1. Load the raw `BioProcessCollection`.
2. Run first-pass `bp_format` validation on the raw collection, at minimum by
   calling `bp_format.validate_process(...)` for each process before custom
   transforms.
3. Resolve case-study-specific config in code.
4. Apply `transform_process_collection(collection, config)` once for
   collection-level changes such as process-key normalization.
5. Enrich reactor/feed component semantics in code when the raw artifact does
   not yet contain the medium definitions required for training.
6. Build default derived controls required by the V1 runtime contract when they
   were not already provided by user code.
7. Validate control roles, state roles, feed semantics, reactor/feed component
   completeness, and required interpolators.
8. Run strict post-transform `bp_format` validation on the prepared processes
   before writing `prepared.json`.
9. Persist structural runtime-control metadata needed to rebuild controls at
   training-load time.
10. Compute any required model scaling statistics.
11. Update or add prep metadata.
12. Serialize the full transformed collection as `prepared.json`.

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

This validation layer should reuse existing `bp_format` checks where possible
rather than reimplementing them from scratch.

The intended V1 contract is that strict post-transform `bp_format` validation runs
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
`bp_format.mechanistic.ControlSplines` and must be written into the prepared
artifact metadata.

### 10.3 Rates vs Cumulative Traces

The stored control payload must represent actual control values consumed by the
solver. For continuous feeds and similar volume changes, this means rate
signals, not cumulative quantities.

If the raw artifact does not distinguish cumulative from rate traces, the user
must resolve this in `transform_process_collection(...)`.

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
derive feed-rate controls from preprocessing outputs in
`transform_process_collection(...)`.

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

V1 default sampling control construction reads `SampleVolumeChange` traces from
the process data directly. User code may override this behavior.

Default behavior:

- read `SampleVolumeChange` events,
- accumulate their magnitudes into a monotone cumulative control
  `V_sample_acc(t)`,
- approximate each increment with a finite-duration interpolation rather than an
  instantaneous jump,
- use the same effective `min_dt` basis as bolus controls:
  `min(run_min_dt, (t_end - t_start) / 1000)`,
- represent each sampling increment over one `min_dt` interval
  `[t0, t0 + min_dt]` (sampling width = `1 * min_dt`),
- write the resulting derived control and its metadata into `prepared.json`.

Sampling still differs from bolus in geometry:
- sampling uses one `min_dt` ramp segment per event,
- bolus uses a triangular rate segment with total width `1 * min_dt`.

This default must be overrideable in `custom.py`.

Reasons for override include:

- no usable sample-volume trace is present,
- the user wants to provide explicit sampling times and amounts,
- the case study requires a different reconstruction rule.

## 11. Control Representation for V1 Runtime

V1 no longer exposes a segmented public controls API. Instead it uses a dense,
single-interpolator representation built by `ControlsStore` at runtime from the
prepared process collection.

The runtime implementation is JAX-first. `bp_format` and `bp-train` both live in a
JAX-based stack, so the controls store should load the prepared payload into a
small number of padded JAX arrays, not a Python list of variable-shape arrays.
The canonical runtime representation is therefore one collection-level array per
payload kind, for example:

- `dense_grid`: `[n_processes, max_grid_length]`
- `control_values`: `[n_processes, max_grid_length, max_controls]`
- `control_derivatives`: `[n_processes, max_grid_length, max_controls]`
- `step_ts`: `[n_processes, max_step_ts_length]`
- `grid_lengths`: `[n_processes]`
- `step_ts_lengths`: `[n_processes]`

Per-process runtime views may exist as thin wrappers over those arrays, but they
should not become the canonical storage format.

The runtime controls store enforces one shared control ordering across
processes. If a prepared artifact contains differing control names/order across
processes, store construction must fail fast.

Process-name normalization belongs in preparation, not in the runtime controls
store. If upstream artifacts use awkward keys such as `process=...`, V1 should
offer an explicit prep-time renaming option and persist the chosen names into
the prepared artifact.

### 11.1 Build Strategy

For each experiment:

- start from `bp_format` `Interpolator` objects,
- build the runtime controls path inside `bp-train` (`ControlsStore`) from
  those interpolators using globally padded arrays,
- reuse `bp_format.mechanistic` conventions where useful, especially deterministic
  control ordering and feed classification,
- convert non-continuous controlled feed additions into bolus-feed triangles as
  specified in Section 11.3,
- convert sample-volume removals into `V_sample_acc` controls using the sampling
  rule from Section 10.6,
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
sampling interpolation segments, the start/peak/end times of bolus-feed
triangles, and any other control boundaries introduced during preparation.
For bolus boundaries, `step_ts` must be stored as one strictly increasing
sequence of unique timestamps after sorting and de-duplication.

### 11.3 Bolus Approximation in V1

This section is normative for non-continuous controlled feed additions
(boluses). Sampling semantics remain separate (Section 10.6).

- non-continuous controlled feed additions are interpreted as bolus events,
- for each process, define its online-data timestamp basis as the sorted unique
  finite timestamp vector available to control/event preparation for that
  process; define process-local candidate `min_dt` as the minimum strictly
  positive difference between consecutive timestamps in that process-local
  basis,
- define run-level `run_min_dt` as the minimum of all available process-local
  candidate `min_dt` values,
- if no process has a strictly positive process-local candidate `min_dt`, fall
  back to the minimum positive process duration cap
  `(t_end - t_start) / BOLUS_MIN_DT_DURATION_DENOMINATOR` across processes,
- if neither a strictly positive process-local candidate `min_dt` nor a
  positive process duration cap exists and at least one real event
  approximation segment must be constructed (bolus triangle and/or default
  sampling ramp), preparation must fail fast with an explicit error,
- define per-process bolus `min_dt` as
  `min(run_min_dt, (t_end - t_start) / BOLUS_MIN_DT_DURATION_DENOMINATOR)`,
  with `BOLUS_MIN_DT_DURATION_DENOMINATOR = 1000` in V1,
- for each bolus event, let `delta_v` be the bolus delta volume; `delta_v` must
  be strictly positive and non-positive values must fail fast,
- for each bolus event on feed `f` at event time `t0`, construct one triangular
  feed-rate contribution with support points at
  `(t0, t0 + 0.5 * min_dt, t0 + min_dt)`,
- triangle geometry is piecewise linear with zero rate at `t0` and
  `t0 + min_dt`, and one peak at `t0 + 0.5 * min_dt`,
- triangle area must equal `delta_v` for that event, with
  `peak_rate = 2 * delta_v / min_dt`,
- if `t0 + min_dt > t_end` for the process time horizon, preparation must
  fail fast with an explicit error,
- per-feed bolus control is the piecewise-linear sum of all event triangles for
  that feed, evaluated over the sorted union of all triangle breakpoints,
- sampling-ramp and bolus-triangle supports may overlap in time; in overlap
  regions, both controls remain active and are combined through the runtime
  equations without any special collision rule,
- `step_ts` must include all three points for each bolus event (start, peak,
  end) as a sorted de-duplicated timestamp list,
- process-level `step_ts` passed to the solver must be the sorted
  de-duplicated union of all source boundaries (including sampling ramp start
  and end boundaries, and bolus start/peak/end boundaries),
- loss/target extraction remains defined at measurement timestamps only; a
  measurement with `t_sample < t_bolus` must not include effects from that
  later bolus event,
- `V_sample_acc` remains a derived sampling control stored in `prepared.json`
  with `Interpolator` and provenance metadata, and the wrapper reconstructs
  `V_real = V_cont - V_sample_acc` at each RHS call.

Tests must verify that integrated bolus additions and integrated sampled-volume
removals match their intended amounts.

### 11.4 Global Padding

The control store uses one global padded shape across all experiments in the
prepared dataset. This is required to avoid JIT recompilation.

This padding guarantee applies to the runtime `ControlsStore`. V1 should not
strip those arrays into variable-length Python lists at load time, because that
would reintroduce shape instability and Python dispatch overhead.

The prepared artifact must record:

- stable process order,
- structural control metadata needed to recover control sources (including
  custom `V_sample_acc` sources),
- runtime-control build settings (grid refinement/error config),
- validation and semantic provenance.

When code needs an active per-process prefix for plotting or debugging, it may
derive that view from stored lengths or masks, but the underlying runtime store
remains padded and globally aligned.

Because V1 assumes one collection-wide control ordering, `eval(...)` and
`eval_derivative(...)` should return values in that shared order. V1 should not
add a separate runtime `eval_local(...)` path unless a concrete downstream use
case appears later.

### 11.5 Batched Controls Evaluator

For batch training, V1 should also expose a minimal all-process controls
evaluator that works directly on padded stacked arrays and avoids Python
metadata lookups in the hot path.

Target contract:

- construction from `ControlsStore`,
- `eval(process_idx: int, t: jax.Array) -> jax.Array`,
- no process-name fields or per-process dict lookups in the runtime-critical
  path.

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

- `specific_rates`: specific rates `q_i` aligned with the species state order.
  Mechanistic biomass scaling and transport/dilution are applied by the
  mechanistic ODE runtime,
- `modeled_feed_rates`: a vector aligned with the modeled-flow ordering from
  `RhsOde.modeled_flow_names`. If there are no modeled feeds, this is
  `jnp.zeros((0,))`,
- optional `auxiliary`: model-defined observables that may be saved alongside
  the solver trajectory. Conservative V1 contract: `None` or
  `dict[str, array]` with stable keys and scalar or 1D-array leaves.

The exact container type may be a small dataclass, namedtuple, or equivalent.
The first two fields above are semantically required; `auxiliary` is optional.

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

It is responsible for specific-rate prediction and modeled-feed-rate
prediction (`jnp.zeros((0,))` when there are no modeled feeds).
The wrapper handles:

- state clamping and continuous-volume-state mechanics,
- reconstruction of physical volume from sampling history,
- augmented-controls assembly (base controls plus flattened `Cin` inputs),
- delegation to `bp_format.mechanistic.RhsOde` for biomass scaling,
  feed transport, and dilution,
- returning the final state derivative.

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
- a mechanistic RHS object (`bp_format.mechanistic.RhsOde`) built from process
  metadata.

### 14.2 Responsibilities

The wrapper must:

- read all feed-rate controls,
  continuous controlled feeds are recovered as derivatives of cumulative
  controls, while bolus triangles are consumed directly as feed-rate controls,
- maintain `V_cont` as part of the integrated state,
- read `V_sample_acc(t)` from the controls object,
- reconstruct `V_real = V_cont - V_sample_acc`,
- include transport and volume contributions from controlled non-continuous
  feed additions (bolus triangles) using their feed-rate contributions and inlet
  feed composition metadata,
- build an augmented controls vector by appending flattened
  `RhsOde.Cin`/`RhsOde.Cin_modeled` to base controls,
- request `specific_rates` and `modeled_feed_rates` from the user module,
- delegate derivative assembly to `RhsOde` (`q * X_active`, transport, and
  dilution),
- produce the final derivative for the ODE solver.

### 14.3 Modeled Feeds

V1 should support multiple feed streams explicitly, but it should keep the
runtime contract strict:

- every feed stream must be declared explicitly,
- species ordering is taken from `RhsOde.species_names`,
- continuous controlled feeds are ordered by `RhsOde.flow_names`,
- modeled feeds are ordered by `RhsOde.modeled_flow_names`,
- controlled non-continuous feed additions (bolus triangles) come from the
  prepared control vector and process volume-change metadata and are applied by
  the wrapper as extra transport/volume terms,
- modeled feeds come from `ReactionOutputs.modeled_feed_rates`,
- `ReactionOutputs.modeled_feed_rates` must follow the same positional
  convention used by `bp_format.mechanistic.RhsOde`, aligned to the ordered list
  of explicit modeled feed streams declared in the process metadata.

There is no partially automatic mode in which some dilution terms come from the
wrapper and others are manually implemented in user code.

### 14.4 Relationship to `bp_format.mechanistic`

V1 uses `bp_format.mechanistic` directly for mechanistic RHS evaluation.

Current runtime split:

- `bp-train` owns controls preparation/evaluation (`ControlsStore`), batching,
  loss evaluation, and training-loop behavior.
- `bp_format.mechanistic.RhsOde` owns mechanistic derivative assembly
  for species kinetics plus continuous/modeled feed transport (`q * X_active`,
  transport, dilution, and `dV/dt` for those streams).
- `bp-train`'s wrapper remains a thin adapter that maps controls/model outputs
  to the `RhsOde` call signature.
- `bp-train`'s wrapper additionally applies transport and `dV/dt` contributions
  for controlled non-continuous feed bolus triangles so total integrated
  `V_cont` includes both continuous and bolus-feed additions.

This keeps mechanistic math aligned with bp_format while preserving padded,
JIT-stable training infrastructure in `bp-train`.

## 15. Training Data Object

The minimum V1 training data object per experiment is:

- `process_name`,
- `t_meas`,
- `y_meas`,
- `y0`,
- `controls`.

In V1, state interpolators are preserved in the transformed `bp_format`
collection itself via the canonical `interpolator` field on state-bearing
objects. They do not need to be duplicated into a separate `bp_train` metadata
store unless a later augmentation pipeline requires a runtime-optimized state
interpolator payload.

### 15.1 `y_meas`

- contains only measured target variables,
- uses explicit target order with precedence:
  CLI `--target` > `custom.py` `CONFIG["target_variable_order"]` >
  default auto-selected measured reactor components,
- if neither CLI nor config specifies targets, training must emit a loud warning
  and log resolved default target list at startup,
- may be a subset of all dynamic states,
- does not include `V_cont`,
- is padded across experiments for batching.

### 15.2 `y0`

- uses first measured values for observed target states,
- appends `V_cont` as the final state entry,
- currently initializes `V_cont(0)` from `process.volume.initial_volume`,
- is constructed at training-data build time from the prepared collection.

### 15.3 Training Grid

V1 trains only against real measurement timestamps.

Deterministic resampling is allowed in the data layer for future use, but it is
not part of the V1 loss contract.

During batched training, per-process measurement timestamps are selected by
index and the active measurement prefix is used for solver integration.
Loss calculation still uses padded arrays plus masks.

Future benchmark note (non-normative): profile dynamic-slice active-prefix
integration against padded timestamp strategies (for example padding by repeated
last timestamp) and keep whichever is faster/stabler.

## 16. Solver Behavior

### 16.1 Integration Mode

- real-space integration only,
- no pseudo-batch dynamics,
- dynamic volume represented by an integrated continuous state `V_cont`,
- `V_cont` is always the last entry in the state vector,
- integrated `V_cont` must include contributions from all controlled feeds
  (continuous and non-continuous bolus triangles) and modeled feeds,
- realized reactor volume reconstructed in the wrapper as
  `V_real = V_cont - V_sample_acc`.

### 16.2 Step Boundaries

The step size controller should receive boundary times derived during control
preparation. This reduces solver pathologies around steep ramp regions without
exposing a segmented public API.

These boundary times must include bolus-event start/peak/end times and
`V_sample_acc` interpolation segment boundaries.

### 16.3 Expected Runtime Objects

The runtime path should be JIT-friendly and rely on globally padded arrays so
that all experiments in the prepared artifact can share one compiled shape.

For V1, prefer one stacked JAX array per payload kind plus lightweight
per-process index metadata over Python containers of per-process arrays. This
is the better default for both compile-time stability and runtime batching.

### 16.4 Batched Train-Step Contract

V1 training-harness batching semantics are:

- `steps` means number of optimizer updates,
- each update consumes exactly one full batch,
- total sampled process indices per run is always `steps * batch_size`,
- `batch_size=None` resolves at runtime to `len(selected_processes)`,
- no `drop_last_batch` behavior in V1.
- if `process_names=None`, selected processes are exactly `store.process_order`,
- if `process_names` is provided, names must be unique and known in the store;
  otherwise fail fast.

Batch index-stream behavior:

- base mode is deterministic round-robin across selected processes,
- `shuffle_batches=True` shuffles each round-robin cycle,
- `batch_seed` controls all batch-index randomness.

Determinism contract:

- if `batch_seed is None`, batching randomness falls back to `seed`,
- with same prepared artifact, selected process set/order, and effective config,
  the index stream and update order must be identical.

Canonical index-stream algorithm:

1. Build canonical selected index vector in selected process order.
2. Repeatedly append one cycle until stream length >= `steps * batch_size`:
   - if `shuffle_batches=False`, cycle = selected index vector as-is,
   - if `shuffle_batches=True`, cycle = deterministic permutation of selected
     index vector using the active RNG stream.
3. Truncate stream to exactly `steps * batch_size`.
4. Reshape to `[steps, batch_size]`.

Optimization and loss behavior:

- optimizer backend is Optax,
- V1 exposes `optimizer_name in {"adam", "sgd"}` and `learning_rate`,
- default optimizer is `adam`,
- `learning_rate` must be strictly positive,
- batch loss is mean of per-sample process losses in the current batch.
- users may override training loss via
  `custom.py::build_sample_loss_fn(...)` (preferred) or
  `custom.py::build_batched_loss_fn(...)` (advanced),
- if no custom loss hook is provided, runtime falls back to default measurement
  loss,
- forward loss reporting uses default/sample-loss objective contract and rejects
  advanced batched hooks that are not guaranteed to decompose per process.

Compile/shape stability behavior:

- one batched train-step JIT boundary should be built per run under stable
  config and shapes,
- runtime should record the JIT input-signature summary in logs/results,
- runtime should record how often the train-step function was rebuilt in Python
  as a practical proxy for recompile risk.

Logging behavior:

- `log_every` controls step-based logging cadence,
- per-process losses logged at each logging step are for sampled batch members
  only.

### 16.5 Batching Config And Validation Contract

Canonical batching config for V1 training harness:

- `steps: int` (optimizer updates),
- `batch_size: int | None = None`,
- `shuffle_batches: bool = True`,
- `batch_seed: int | None = None`,
- `optimizer_name: str = "adam"` with allowed values `{"adam", "sgd"}`,
- `learning_rate: float`.

Runtime resolution rules:

- effective `batch_size = len(selected_processes)` if config value is `None`,
- effective RNG seed for batching = `batch_seed` when provided, else `seed`.

Fail-fast runtime validation:

- `steps <= 0` is invalid,
- effective `batch_size <= 0` is invalid,
- `learning_rate <= 0` is invalid,
- unknown process names in `process_names` are invalid,
- duplicate entries in explicit `process_names` are invalid,
- empty selected process set is invalid,
- unsupported `optimizer_name` is invalid.

Validation errors should be explicit `ValueError` messages naming the offending
field/condition.

### 16.6 Batch Telemetry Contract

V1 batch-oriented training results/logs should include at minimum:

- batch mean loss by step,
- sampled per-process losses at logging steps,
- batch composition by step (process names or indices),
- first compile/warmup time for the train-step JIT boundary,
- per-step runtime timings,
- JIT input-signature summary,
- train-step rebuild count in Python (practical proxy for recompile risk).

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
- runtime control-build settings and per-process control/source metadata needed
  to rebuild padded runtime controls in `ControlsStore`.

V1 should store this under an explicit package-specific namespace such as
`metadata["bp_train"]` to avoid collisions with upstream `bp_format` metadata.

Long-term provenance can become more sophisticated, but V1 should still be
explicit enough to support debugging and reproducibility.

## 18. Suggested File Layout for V1

One possible package layout is:

```text
bp_train/
  cli.py
  prepare.py
  controls.py                # prep-time: signal sources, adaptive grid, dense payload
  controls_store.py           # runtime: padded JAX tensors, eval(t) under JIT
  training_data.py
  wrapper.py
  trainer.py
  model_api.py
  validation.py
  utils.py
  defaults.py
custom.py
prepared.json
train-config.json
examples/
  01_kittler_2022/
    custom.py
```

`controls.py` owns prep-time signal processing: building `SignalSource` objects
from raw process data (volume changes, process variables), adaptive dense-grid
refinement, and computing the dense payload (grid, values, derivatives).
`controls_store.py` owns the runtime side: it loads the dense payload produced
by `controls.py`, pads and stacks tensors across processes into globally-shaped
JAX arrays, and exposes `PerProcessControls.eval(t)` for use inside the JIT'd
ODE solver. The prep/runtime split keeps numpy-only adaptive algorithms out of
the JAX-traced training path.

`model_api.py` is the home of the reaction-module and `ReactionOutputs`
contracts. `utils.py` contains internal helpers such as user-hook loading and
config resolution. `defaults.py` contains default prep and training settings such as
the dense-grid refinement parameters.

V1 should also expose a package CLI entrypoint:

```text
bp-train prepare --input input.json --output prepared.json --custom examples/01_kittler_2022/custom.py
```

V1 exposes `prepare`, `train`, and `forward` subcommands under the same
entrypoint shape.

Current `train` output contract (implemented behavior) is:

- always write `trained_wrapper.eqx` and `trained_wrapper.meta.json`,
- always write `losses.csv` (forward-evaluation per-process/per-target losses),
- always write `predictions.csv` (dense simulated trajectories),
- `predictions.csv` always includes state and standard wrapper-derived columns;
  it may additionally include `aux_*` columns when the reaction module returns
  solver-time auxiliary observables,
- use the same solver jump-ts setting for both loss and prediction exports
  (`solver_use_jump_ts` consistency),
- write plot PNGs only when plotting is enabled (`--plot`; default on).

This is illustrative only; exact filenames may change.

## 19. Testing Requirements

V1 should prioritize tests that de-risk the chosen simplifications.

### 19.1 Control Prep Tests

- prepared metadata is structural and does not persist padded dense arrays,
- dense-grid linear interpolation error is below threshold,
- global padding shapes are stable,
- config-defined control ordering is preserved,
- missing controls fail fast,
- cumulative-to-rate transformation hooks work as expected,
- default `V_sample_acc` construction from `SampleVolumeChange` traces works as
  expected,
- user override of default `V_sample_acc` construction works as expected.

### 19.2 Event Approximation Tests

- bolus approximation adds exactly the intended amount,
- run-level `run_min_dt` is computed as the minimum of process-local positive
  timestamp deltas (with duration-cap fallback when no process-local positive
  delta exists), and process-local bolus `min_dt` applies the configured
  duration cap,
- bolus triangle construction uses points
  `(t0, t0 + 0.5 * min_dt, t0 + min_dt)` and satisfies the event area/peak-rate
  constraint,
- prep fails fast when a bolus event would violate `t0 + min_dt <= t_end`,
- overlapping same-feed bolus events superpose additively in the feed-rate
  control,
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
- measurement-time loss uses the expected padded arrays and masks,
- batch-size/repeat behavior is correct (including single-process with
  `batch_size > 1`),
- batch index generation is deterministic with fixed seed/config,
- training loss decreases on toy data,
- invalid batching config fails fast,
- train-step input signatures are stable across updates in stable-shape runs,
- no explicit train-step rebuild path is triggered in stable-shape runs.

## 20. Implementation Order

Recommended implementation order (historical planning baseline):

1. preparation pipeline that loads raw `bp_format` JSON and writes `prepared.json`,
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

Status as of March 28, 2026:

- Steps 1-8 above are implemented in code,
- wrapper-correctness tests from Spec 19.3 are implemented,
- next work should focus on the remaining Phase C roadmap items beyond the
  minimal single-process path.

## 21. Deferred Items

These are explicitly deferred beyond V1:

- data augmentation (the `bp_format.AugmentedBioProcess` placeholder is in
  place but no augmentation hook in `bp-train prepare` produces these
  records yet — see `spec/loo.md`),
- stateful models,
- pseudo-batch dynamics,
- alternative runtime contracts where the user manually handles dilution,
- more sophisticated treatment of bolus events,
- adaptive knot placement beyond the simple dense-grid refinement strategy,
- automatic jump detection inside `bp-train`.

LOO-CV orchestration shipped as `bp-train loo` / `bp_train.loo`; the
contract and CLI/artifact details live in `spec/loo.md`.

## 22. Known Limitations

Current limitations:

- `Cin` is currently constant at runtime; processes requiring time-varying feed
  composition are not supported in this implementation.
- `bp-train` currently delegates mechanistic RHS evaluation to
  `bp_format.mechanistic.RhsOde`, so training/batching-specific performance
  optimizations in that mechanistic inner loop must be implemented in bp_format.
- The current reaction-module contract is `specific_rates q` plus
  `modeled_feed_rates`, and the mechanistic RHS applies `q * X_active` before
  adding transport/dilution terms. This means arbitrary non-transport,
  non-dilution process terms that should bypass `X_active` scaling (for example
  crystallization kinetics independent of biomass) are not cleanly representable
  in the current API.
