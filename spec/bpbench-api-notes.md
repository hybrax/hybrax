# BPbench API Notes for BP-Train

This file tracks `bpbench` API seams, required adapters, and suggested upstream
changes discovered while designing `bp-train`.

It is intentionally concrete. Some items are blockers, others are cleanup or
ergonomic improvements.

## 1. Naming Standard: `Interpolator` / `interpolator`

### Current State

- `bpbench` should expose `Interpolator` as the field type and
  `interpolator` as the serialized field name.
- `bp-train` should target the renamed `interpolator` API.
- deserialization fallback for older serialized artifacts remains useful.

### Impact on `bp-train`

`bp-train` should standardize on:

- dataclass field type: `Interpolator`,
- serialized field name: `interpolator`,
- metadata key: `interpolator_metadata`.

### Requested Direction

- use `interpolator` as the target downstream API,
- avoid introducing fresh direct `.spline` access in downstream code.

## 2. Process-Collection-First Workflow

### Current State

`bp-train` intentionally works with `BioProcessCollection` artifacts rather
than full `BenchmarkDataset` wrappers in V1. This keeps the data path aligned
with `hybrax-prep` outputs and reduces avoidable complexity.

### Requested Direction

- keep `load_process_collection(...)` and `save_process_collection(...)` as
  first-class APIs,
- document them clearly in `bpbench`,
- ensure example workflows demonstrate this path.

## 3. Relationship to `bpbench.mechanistic`

### Current State

`bpbench.mechanistic` already provides a correct reference implementation for
control evaluation and ODE wrapper logic.

### Impact on `bp-train`

V1 now depends on it directly for runtime mechanistic RHS evaluation:

- `bp_train.wrapper.HybridOdeWrapper` holds and calls `RhsOde`,
- `bp_train.harness` uses `get_rhs_ode(process)` and validates structural
  compatibility across selected processes in the training run,
- per-process `Cin` / `Cin_modeled` are injected per sample during batched
  loss evaluation.

`bp-train` still owns the controls-store representation, batching, and training
infrastructure.

### Requested Direction

- keep `RhsOde` and `get_rhs_ode` stable and documented as supported runtime
  dependencies for training packages,
- preserve deterministic ordering contracts (`species_names`, `flow_names`,
  `modeled_flow_names`) and shape contracts for `Cin`/`Cin_modeled`,
- continue exposing conventions such as modeled-feed positional alignment.

## 4. Stronger Metadata for Control Semantics

### Current State

The raw exported data may not say whether a trace is:

- a cumulative quantity,
- an instantaneous rate,
- a derived control,
- a measured state later re-purposed as a control.

### Impact on `bp-train`

This forces `bp-train` to rely on code-level preprocessing hooks in V1.

### Requested Direction

Long-term, `bpbench` should support richer metadata for control semantics, for
example:

- semantic role,
- cumulative-vs-rate meaning,
- derivation provenance,
- feed-stream classification.

V1 of `bp-train` will use custom prep hooks as an adapter until such metadata is
available upstream.

## 5. Provenance Metadata Namespace

### Current State

`bp-train` needs to write prep provenance and training metadata into a full
transformed JSON artifact.

### Requested Direction

Define or at least document a stable metadata namespace in `bpbench` for
downstream tooling to record:

- source dataset id or path,
- source hash,
- transform timestamp,
- transform names,
- control ordering,
- training-specific annotations.

V1 of `bp-train` can use a package-specific metadata block until this is
standardized.

## 6. Mutability / Copy Helpers for Process Transformations

### Current State

`bp-train` prep will need to:

- update `is_controlled`,
- replace or add `Interpolator` objects,
- attach metadata,
- possibly derive new process variables or volume-change traces.

### Requested Direction

`bpbench` would benefit from lightweight helpers for:

- copying processes,
- replacing process variables,
- replacing volume changes,
- updating metadata without manual nested mutation.

This is not a hard blocker, but it would simplify downstream prep code.

## 7. Reactor and Feed Semantics Need Better Examples

### Current State

The sample `input.json` used here has empty `reactor_medium.components`, while
`bpbench.mechanistic` expects richer reactor/feed semantics.

### Impact on `bp-train`

Case-study-specific code may need to enrich reactor and feed information before
training can use the wrapper-managed dilution path.

### Requested Direction

- add `bpbench` examples that explicitly populate reactor/feed semantics for
  mechanistic or hybrid training use,
- document the minimal fields required for downstream ODE usage.

## 8. Code-First Dataset Construction Should Be Treated as Supported

### Current State

The added `bpbench` example notebooks suggest a useful code-first workflow:
constructing or modifying `bpbench` dataclass objects directly in Python.

### Requested Direction

Treat this as an officially supported complement to declarative JSON artifacts.
That aligns well with the hybrid configuration approach `bp-train` wants:

- artifacts remain serialized JSON,
- structural setup and rich semantics can be expressed in code.

## 9. Backward Compatibility Policy

### Requested Direction

For the near term, `bpbench` should preserve read compatibility for older
serialized artifacts where possible, especially around:

- `spline` vs `interpolator`,
- older metadata keys,
- process collection vs dataset wrappers.

This will make `bp-train` easier to adopt while the surrounding APIs are still
settling.
