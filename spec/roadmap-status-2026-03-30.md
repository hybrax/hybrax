# BP-Train Roadmap Status (as of 2026-03-30)

## Did we update the spec during the TimeSeries migration?

No. The TimeSeries migration work updated runtime code/tests/data artifacts, but did not
change files under `spec/`.

## Where we are vs the current V1 roadmap

Reference roadmap: `spec/v1-detailed-spec.md` (Implementation Order section).

Implemented (roadmap steps 1-8):

1. Raw-data load + prep pipeline (`prepare_artifact`, CLI `prepare` command)
2. Validation + prep metadata
3. Custom hook pattern (`custom.py`)
4. Runtime controls preparation + dense interpolation + padding
5. TrainingData object with padded measurement arrays
6. User model abstraction + `partition_trainable()`
7. Library wrapper with dilution/feed transport handling
8. Minimal trainer with single-process train step + tests

Current practical scope:

- The codebase is at a "minimal single-process training primitive" level.
- There is no end-to-end training runner/orchestrator yet.
- There is no CV orchestration yet (LOO is still deferred in the spec).

## What is still missing before LOO-CV

Minimum required items before implementing LOO-CV cleanly:

1. **Training run orchestrator (collection-level)**
- Build a runner that loops optimization steps/epochs over one or more processes.
- Today we only have `single_process_train_step(...)` as a primitive.

2. **Train/eval API boundary**
- Define a stable API for:
  - fitting model params on a train set,
  - evaluating loss/metrics on a held-out set,
  - returning per-process predictions and metrics.
- Right now only single-process measurement loss is formalized.

3. **LOO split manager**
- Deterministic fold generation from `process_order`.
- For each fold: train set = all-but-one process, test set = held-out process.
- Persist fold membership in run metadata.

4. **Per-fold model re-initialization + RNG policy**
- Each fold must start from a fresh model/optimizer state.
- Seed policy must be explicit and reproducible.

5. **Optimizer/run configuration contract**
- Formalize run config schema for training (epochs/steps, LR, tolerances, etc.).
- CLI currently exposes only `prepare`; no train/cv subcommand yet.

6. **Result persistence + aggregation**
- Save per-fold metrics and aggregate summaries (mean/std, etc.).
- Define artifact format for CV outputs.

7. **Tests for multi-process/fold behavior**
- Unit + integration tests for fold splits, no leakage, reproducibility, and metric
  aggregation.

## Important constraints that may affect LOO design

Current runtime assumptions enforce shared structure across processes:

- `ControlsStore` requires identical control names/order across processes.
- `TrainingDataStore` requires identical measured target names/order across processes.

This is acceptable for many LOO scenarios, but if a dataset has heterogeneous process
schemas, LOO orchestration will fail unless we either:

- enforce/prep into a shared schema, or
- extend runtime to support per-process schema variation.

## Suggested next implementation order toward LOO-CV

1. Add a minimal **train runner** (multi-process training without CV).
2. Add a **train CLI/config** for reproducible runs.
3. Add an **evaluation/reporting layer**.
4. Add **LOO fold orchestration** on top of the runner.
5. Add **LOO result artifacts + tests**.
