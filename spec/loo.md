# Leave-One-Process-Out Cross-Validation for `bp-train`

## Context

`bp-train` currently supports training a hybrid bioprocess model on one or
more processes from a `prepared.json` collection (`bp-train train`) and
running a forward pass with per-process train/holdout loss labelling
(`bp-train forward`). LOO-CV is listed as deferred in
[v1-detailed-spec.md](v1-detailed-spec.md)
(§3, §21) but the
[2026-03-30 roadmap](roadmap-status-2026-03-30.md)
identifies it as the next milestone now that the multi-process train runner
(item 1), train CLI (item 2), and evaluation/reporting layer (item 3) are
all in place.

The infrastructure already supports the hard parts:

- [harness.py:452 `forward_from_collection`](../bp_train/harness.py#L452-L620)
  accepts `training_process_names` and labels each evaluated process as
  `train` vs. `holdout`.
- [cli.py:586 `_format_loss_table`](../bp_train/cli.py#L586-L674)
  aggregates `train (mean)` and `holdout (mean)` rows from that label.
- The trained-wrapper sidecar already records the trained subset
  ([cli.py:541-548](../bp_train/cli.py#L541-L548)).
- `TrainHarnessConfig.process_names` already filters which processes the
  trainer sees ([harness.py:55](../bp_train/harness.py#L55)).
- The shared post-train artifact writer
  [`_write_train_results`](../bp_train/cli.py#L395-L461) already
  encapsulates the per-run forward + losses.csv + predictions.csv +
  plots flow, so LOO can call it once per fold without further
  refactoring.

What is missing is the orchestration layer: build N folds (one per process
group), train each on N-1 groups with a fresh model, evaluate forward on the
full collection so train/holdout losses are recorded, and aggregate fold
results. **Plus** a structural prerequisite in `bp-format`: a placeholder
class for augmented bioprocesses so LOO can group augmented children with
their parent and avoid data leakage when augmentation lands in the prepare
step.

## Phase 0 (prerequisite) — `bp-format` `AugmentedBioProcess` placeholder

Files: `bp-format/bp_format/dataclasses.py`,
`bp-format/bp_format/validate.py`,
`bp-format/bp_format/__init__.py`,
`bp-format/tests/test_dataclasses.py`,
`bp-format/tests/test_validate.py`.

This phase ships only a placeholder; no augmentation logic is implemented
in this round. The goal is to lock down the data shape so downstream
packages (`bp-train`'s LOO orchestrator, future `bp-train prepare`
augmentation hooks) can rely on it.

### 1. New dataclass `AugmentedBioProcess`

Add after the `BioProcess` definition at
`bp-format/bp_format/dataclasses.py:223`:

```python
@dataclass(kw_only=True)
class AugmentedBioProcess(BioProcess):
    """A synthetic variant of an existing BioProcess in the same collection.

    Same fields as :class:`BioProcess` plus a mandatory ``parent_process``
    string referencing the parent's key in the enclosing
    :class:`BioProcessCollection` / :class:`CaseStudy`. Augmented children
    inherit the parent's structural identity (control/state schema,
    medium semantics) and must be grouped with the parent for any
    train/eval split so synthetic siblings cannot leak into a fold whose
    parent is held out.
    """

    parent_process: str
```

Implementation note: the parent class has fields without defaults
(`metadata`, `time_axis`, `volume`, `reactor_medium`); using
`kw_only=True` on the subclass avoids the "non-default after default" trap
and matches Python 3.10+ dataclass behavior already required by the
project.

`AugmentedBioProcess` is a `BioProcess` subclass, so existing
`Dict[str, BioProcess]` containers in
`BioProcessCollection.processes` (`bp-format/bp_format/dataclasses.py:252`)
and `CaseStudy.processes` (`bp-format/bp_format/dataclasses.py:262`)
accept it without typing changes.

### 2. Validation

Add a new helper `validate_augmented_parent_refs(collection_or_case_study)`
in `bp-format/bp_format/validate.py`. Contract:

- Iterate `processes.items()`.
- For every `AugmentedBioProcess` value, assert
  `value.parent_process` is a key of the same dict and the value at that
  key is **not** an `AugmentedBioProcess` (no augmented-of-augmented in
  v1 — defer that explicitly).
- Return `(all_valid, messages)` matching the existing
  `validate_*` shape used by `validate_process`.

Wire into `validate_case_study` (`bp-format/bp_format/validate.py:502`):
add a third pass after per-process and cross-process consistency checks.
Augmented children must satisfy structural consistency the same way
parents do (already covered by the existing signature checks since
`AugmentedBioProcess` shares `BioProcess` shape).

Cross-process consistency: when comparing signatures, do not require the
augmented child to share signature with the very first process in the
dict — instead require it to share signature with **its parent**. Today's
implementation just compares everything to the first process, which still
works for augmented children of that first process; for augmented
children of a later parent we add an explicit check that the child's
signature equals the parent's signature and skip the "vs first process"
diff message for them to keep error output clean.

Export `AugmentedBioProcess` and `validate_augmented_parent_refs` from
`bp-format/bp_format/__init__.py`.

### 3. Tests in `bp-format/tests/`

Add to `test_dataclasses.py`:

- `test_augmented_bioprocess_inherits_bioprocess_fields` — instantiate
  with valid args, check `isinstance(child, BioProcess)` and that
  `child.parent_process == "P0"`.
- `test_augmented_bioprocess_parent_process_required` — instantiating
  without `parent_process=` raises `TypeError`.
- `test_augmented_bioprocess_serialization_roundtrip` — if
  `bp_format.serialization` round-trips `BioProcessCollection`, ensure
  the type tag for `AugmentedBioProcess` survives a JSON
  encode/decode. (Implementation: may require adding the new class to
  the serializer's class registry; flag this in the plan even if it is
  a one-liner.)

Add to `test_validate.py`:

- `test_validate_augmented_parent_refs_ok` — two-process case study with
  one augmented child whose `parent_process` is the other process →
  `(True, [...])`.
- `test_validate_augmented_parent_refs_unknown_parent` — augmented child
  with `parent_process="ghost"` → `(False, [..."unknown parent"...])`.
- `test_validate_augmented_parent_refs_rejects_augmented_of_augmented` —
  child A points to parent B which is itself augmented → `(False, ...)`.
- `test_validate_case_study_runs_augmented_parent_refs` — full
  `validate_case_study` integration: a case study containing one
  augmented child with a bad parent fails with the expected message
  under `report["__consistency__"]`.

### 4. Serialization

`bp-format/bp_format/serialization.py` needs to round-trip
`AugmentedBioProcess` alongside `BioProcess`. If the serializer uses an
explicit class registry / `__type__` tag, add the new class there; if
it auto-discovers via `dataclass_fields`, the subclass will work
automatically but the type identity must be preserved. The implementer
should grep the serializer for `BioProcess` and add parallel handling
for `AugmentedBioProcess`.

### Phase 0 non-goals

- No changes to `bp-train prepare` to actually produce augmented
  children — that lands in a follow-up.
- No changes to the mechanistic RHS path — augmented children share
  parent semantics and pass through `get_rhs_ode` exactly like a
  parent `BioProcess`.

## Phase 1 — LOO design in `bp-train`

### Fold definition

Folds are built from **process groups**, not raw process keys, so an
augmented child stays bound to its parent in any split.

```python
def _build_fold_groups(
    collection: BioProcessCollection,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return (parent_name, group_member_names) tuples in canonical order.

    Each non-augmented BioProcess is a group; an AugmentedBioProcess is
    appended to its parent's group. Augmented processes never form their
    own fold.
    """
```

For each group `(p, members)`:

- Holdout group = `members` (the parent plus all its augmented children).
- Train set = every process **not** in `members`.
- The fold is identified by `p` (the parent process name) and `fold_idx`
  is its position in the canonical group order.

Each fold runs:
- A fresh reaction module via `_build_reaction_module`.
- A fresh optimizer state.
- Per-fold seed `cfg.seed + fold_idx`.

Forward eval uses **all** processes in the collection (parents +
augmented children) so the loss table records each as either `train` or
`holdout` via the existing `training_process_names` mechanism. Augmented
children of a held-out parent are correctly labelled `holdout` because
they are not in `train_processes`.

If the collection contains zero `AugmentedBioProcess` entries, this
mechanism reduces exactly to plain LOO over `process_order`.

### Public Python API — new module `bp_train/loo.py`

```python
@dataclass(frozen=True)
class LOOConfig:
    base_train_config: TrainHarnessConfig
    output_dir: Path
    selected_holdouts: tuple[str, ...] | None = None  # parent names; None = all
    render_plots: bool = True
    write_per_fold_predictions: bool = True

@dataclass(frozen=True)
class FoldResult:
    holdout_parent: str
    holdout_group: tuple[str, ...]    # parent + augmented children
    fold_idx: int
    train_processes: tuple[str, ...]
    train_result: TrainHarnessResult
    forward_result: ForwardResult
    fold_dir: Path

@dataclass(frozen=True)
class LOOResult:
    folds: tuple[FoldResult, ...]
    summary_csv_path: Path
    aggregate: dict[str, Any]   # mean/std/median of holdout total + per-target

def run_loo_cv(
    collection: BioProcessCollection,
    *,
    config: LOOConfig,
    custom_py: str | Path | None = None,
    runtime_config: dict[str, Any] | None = None,
) -> LOOResult: ...

def run_loo_fold(
    collection: BioProcessCollection,
    *,
    holdout_parent: str,
    config: LOOConfig,
    custom_py: str | Path | None = None,
    runtime_config: dict[str, Any] | None = None,
) -> FoldResult: ...
```

Internally `run_loo_fold` is a thin wrapper that:

1. Resolves the holdout group via `_build_fold_groups`.
2. Builds a per-fold `TrainHarnessConfig` via `dataclasses.replace`:
   - `process_names = tuple(p for p in store.process_order if p not in holdout_group)`
   - `seed = base.seed + fold_idx`
   - `checkpoint_dir = fold_dir / "checkpoints"` (unless explicitly
     disabled).
3. Calls `train_from_collection(collection, config=fold_cfg, custom_py=...)`.
4. Saves model + sidecar (mirrors
   [cli.py:533-564](../bp_train/cli.py#L533-L564)).
5. Calls `forward_from_collection(...)` over the full collection with
   `training_process_names = train_processes`.
6. Writes `losses.csv`, `predictions.csv`, optional plots — re-using
   the existing
   [`_write_train_results`](../bp_train/cli.py#L395-L461) helper
   (already factored out in commit `1af27eb` and used by
   `_handle_train`); LOO calls it once per fold with the per-fold
   `output_dir` and `training_process_names` set to the N-1
   train-side processes.

`run_loo_cv` loops folds sequentially, then aggregates:

- `loo_summary.csv` columns: `fold_idx`, `holdout_parent`,
  `holdout_group` (semicolon-joined), `holdout_total`,
  `holdout_<target_i>` for each target, `train_mean_total`,
  `train_mean_<target_i>` for each target, `final_train_loss`. Uses the
  existing `_format_loss_table`-style aggregation extended to one row
  per fold.
- `loo_aggregate.json`: `holdout_total_mean`, `holdout_total_std`,
  `holdout_total_median`, plus per-target equivalents across folds.

### CLI — extend `bp_train/cli.py`

New subparser `loo`, modeled on `train`:

```
bp-train loo
  --input prepared.json                   # required
  --output-dir output/loo                  # default: ./output/loo
  --custom path/to/custom.py
  --config runtime.json
  --holdouts NAME[,NAME]...                # optional: subset of parents
                                           # absent → all parents
                                           # one entry → cluster-friendly
                                           # single-fold mode
  --target ...                             # same flags as `train`
  --target-source ...
  --steps / --batch-size / --batch-seed / --optimizer / --learning-rate
  --grad-clip-norm / --seed / --log-every / --solver-* / --no-jump-ts
  --plot / --no-plot
  --log-process-losses / --log-decimals / --log-header-every
  --metrics-csv / --metrics-jsonl          # per-fold (templated path,
                                           #   `{fold}` token or `<base>.<fold>.csv`)
  --log-level
```

(`--holdouts` covers both the "subset" and the "single-fold cluster" use
cases. No separate `--fold` flag.)

Behavior:

- No `--holdouts` → run all folds (one per parent process).
- `--holdouts X` → run only fold X. Used for cluster parallelism.
- `--holdouts A,B` → run only folds for A and B.
- Augmented child names in `--holdouts` are rejected with an explicit
  error: only parent process names are valid fold identifiers.
- Per-fold seed = `--seed + fold_idx`. Top-level summary records both
  the base seed and the resolved per-fold seed.
- `--checkpoint-dir`: same default behavior, but located at
  `<fold_dir>/checkpoints` per fold. Empty string disables.
- Single-`--holdouts` invocations skip `loo_summary.csv` and
  `loo_aggregate.json` so parallel runs don't race; an
  `--aggregate-only` mode (deferred — see "Open polish") can re-scan
  finished `folds/` later.

### Output layout

```
output/loo/
  loo_summary.csv                # one row per fold + aggregate row
  loo_aggregate.json             # mean/std/median of holdout losses
  folds/
    <holdout_parent_0>/
      trained_wrapper.eqx
      trained_wrapper.meta.json  # sidecar — training_processes set to N-1
                                 # group (excludes parent + augmented children)
      losses.csv                 # train/holdout split labelling
      predictions.csv
      checkpoints/               # if cfg.checkpoint_dir is set
      plots/                     # if --plot
    <holdout_parent_1>/
      ...
```

### Files touched

| File | Change | Why |
|---|---|---|
| **`bp-format/bp_format/dataclasses.py`** | edit | Add `AugmentedBioProcess(BioProcess)` with `parent_process: str` (kw_only). |
| **`bp-format/bp_format/validate.py`** | edit | Add `validate_augmented_parent_refs`; wire into `validate_case_study`. |
| **`bp-format/bp_format/serialization.py`** | edit | Round-trip `AugmentedBioProcess` (likely class-registry one-liner). |
| **`bp-format/bp_format/__init__.py`** | edit | Export `AugmentedBioProcess`, `validate_augmented_parent_refs`. |
| **`bp-format/tests/test_dataclasses.py`** | edit | 3 tests (instantiation, required field, serialization roundtrip). |
| **`bp-format/tests/test_validate.py`** | edit | 4 tests (parent refs ok, unknown parent, augmented-of-augmented, integration with `validate_case_study`). |
| **`bp-train/bp_train/loo.py`** | **new** | `LOOConfig`, `FoldResult`, `LOOResult`, `run_loo_cv`, `run_loo_fold`, `_build_fold_groups`, summary aggregation. |
| **`bp-train/bp_train/cli.py`** | edit | Add `loo` subparser + `_handle_loo`. Per-fold artifacts are produced by the existing [`_write_train_results`](../bp_train/cli.py#L395-L461) helper (already shared with `_handle_train`); the post-train block at [cli.py:533-581](../bp_train/cli.py#L533-L581) is the structural template for `_handle_loo`. |
| **`bp-train/bp_train/__init__.py`** | edit | Export `LOOConfig`, `FoldResult`, `LOOResult`, `run_loo_cv`, `run_loo_fold`. Also export the existing `train_from_collection` (currently missing — flagged in [roadmap-status-2026-04-09.md Appendix A.5](roadmap-status-2026-04-09.md#L694-L700)). |
| **`bp-train/spec/v1-detailed-spec.md`** | edit | Move LOO out of §3/§21 deferred lists; add a short LOO section describing the CLI + artifact contract and the augmented-group rule. |
| **`bp-train/spec/roadmap-status-2026-04-09.md`** | edit | Strike LOO from "remaining items". |
| **`bp-train/tests/test_loo.py`** | **new** | See "Tests" below. |
| **`bp-train/examples/01_kittler_2022/run_loo.sh`** | **new** | Demo script wrapping `bp-train loo`. |

### Reused functions (no duplication)

- `harness._build_reaction_module` — fresh module per fold.
- `harness._build_template_wrapper`, `harness.train_from_collection` —
  full fold training; nothing to re-implement.
- `harness.forward_from_collection` — fold evaluation against full
  collection.
- `harness._resolve_batched_loss_fn` — already routes
  `build_sample_loss_fn` / `build_batched_loss_fn` custom hooks; per-fold
  custom losses work transparently.
- `cli._write_train_results`, `cli._format_loss_table`,
  `cli._write_loss_csv` — per-fold loss table and forward artifacts.
- `serialization.save_model`; `postprocessing.save_model_metadata`,
  `plot_training_results`, `plot_process_simulations`,
  `export_predictions_csv` — per-fold artifacts.
- `checkpointing.CheckpointConfig` / `CheckpointWriter` — per-fold
  `<fold_dir>/checkpoints/` is wired by setting `checkpoint_dir` on the
  per-fold `TrainHarnessConfig`; the harness already owns checkpoint
  writing (commit `c276aa1`).

### Validation / fail-fast rules

- Empty collection → fail.
- Collection with <2 parent groups → fail with
  `"LOO-CV requires at least 2 parent processes; got <n>"`.
- `--holdouts X` where X is unknown → fail.
- `--holdouts X` where X is an `AugmentedBioProcess` name → fail with
  `"--holdouts must reference parent processes; '<X>' is augmented "
  "(parent='<P>')"`.
- Duplicate names in `--holdouts` → fail.
- Same `_validate_batching_config` checks per fold (already happens via
  `train_collection`).

### Open polish (deferred, not blocking)

- `--aggregate-only` mode for cluster runs that finished all folds via
  separate `--holdouts X` invocations.
- Parallel fold execution within one invocation (currently sequential —
  matches GPU-bound workloads where folds compete for the same device).
- Persisting the LOO config alongside `loo_summary.csv` for full
  reproducibility.
- Actually wiring an augmentation hook into `bp-train prepare` (Phase 0
  only ships the placeholder).

## Tests

### `bp-train/tests/test_loo.py` (new)

Mirror `test_harness.py` style; reuse the synthetic 2-3 process fixture
from `tests/test_harness.py::test_train_collection_multi_process_tracks_per_process_histories`.

1. **`test_run_loo_cv_produces_one_fold_per_parent`** — 3-process
   collection (no augmentation), run `run_loo_cv(steps=2)`, assert
   `len(result.folds) == 3`, each fold's `train_processes` excludes its
   own holdout, no overlap.
2. **`test_run_loo_cv_groups_augmented_children_with_parent`** —
   collection with parents `P0`, `P1` and augmented child `P0_aug` whose
   `parent_process="P0"`. Run `run_loo_cv`, assert exactly **two** folds
   (P0, P1), and that the P0 fold's `holdout_group == ("P0", "P0_aug")`
   and `train_processes == ("P1",)`. The P0_aug process must be labelled
   `holdout` in P0's `losses.csv`.
3. **`test_run_loo_cv_writes_expected_artifact_layout`** — assert
   `output_dir/folds/<parent>/trained_wrapper.eqx` and `losses.csv`
   exist for each parent; assert `loo_summary.csv` and
   `loo_aggregate.json` exist at top level.
4. **`test_run_loo_fold_uses_seed_plus_fold_idx`** — train two folds
   with the same base seed but different fold_idx; assert resulting
   `trainable_params` differ (fresh init per fold), and re-running with
   the same fold_idx yields identical params (determinism).
5. **`test_loo_single_holdout_mode_runs_only_that_fold`** — call CLI with
   `--holdouts P1`; assert only `folds/P1/` is produced and no top-level
   summary appears.
6. **`test_loo_unknown_holdout_fails_fast`** — `--holdouts not_a_process`
   → `ValueError`/`SystemExit` with explicit message.
7. **`test_loo_augmented_holdout_fails_fast`** —
   `--holdouts P0_aug` (an augmented child name) → `ValueError` with
   message "must reference parent processes".
8. **`test_loo_single_parent_collection_fails_fast`** — 1-parent input
   raises `ValueError("LOO-CV requires at least 2 parent processes")`.
9. **`test_loo_summary_csv_aggregates_holdout_losses`** — assert one row
   per fold, columns include `fold_idx`, `holdout_parent`,
   `holdout_total`, `holdout_<target>`, plus a final mean row with
   `mean_holdout_total`.
10. **`test_loo_cli_dispatch_calls_run_loo_cv`** — mock `run_loo_cv` and
    assert `bp-train loo --input ... --output-dir ...` parses args
    correctly (mirrors the existing `tests/test_cli.py` mock-dispatch
    pattern).

Total: ~10 tests, ~200-260 lines.

### `bp-format/tests/` additions

See Phase 0 §3 above (3 dataclass tests + 4 validate tests).

## Verification

End-to-end smoke tests using the existing Kittler dataset:

```bash
cd /home/mgotsmy/code/bpbench

# bp-format placeholder lands cleanly first
pytest bp-format/tests/test_dataclasses.py bp-format/tests/test_validate.py -v

# baseline: existing bp-train single-train still works
bash bp-train/examples/01_kittler_2022/run.sh

# new: LOO over the same prepared artifact, low step count
cd bp-train
bp-train loo \
  --input examples/01_kittler_2022/prepared.json \
  --custom examples/01_kittler_2022/custom.py \
  --output-dir examples/01_kittler_2022/output_loo \
  --steps 5 --no-plot

# inspect artifact layout
ls examples/01_kittler_2022/output_loo/
ls examples/01_kittler_2022/output_loo/folds/

# single-fold (cluster-friendly) mode
bp-train loo \
  --input examples/01_kittler_2022/prepared.json \
  --custom examples/01_kittler_2022/custom.py \
  --output-dir examples/01_kittler_2022/output_loo_single \
  --holdouts DoE1_R1 \
  --steps 5 --no-plot

# unit tests
pytest tests/test_loo.py -v

# regression: ensure existing tests still pass
pytest tests/ -v
```

Acceptance criteria:

1. `output_loo/folds/<parent>/trained_wrapper.eqx` exists for every
   parent process in `prepared.json`.
2. Each fold's `losses.csv` shows the held-out parent (and any augmented
   children of that parent) labelled `holdout`; everything else
   `train`.
3. `output_loo/loo_summary.csv` has one row per fold + an aggregate
   row.
4. `output_loo/loo_aggregate.json` contains numeric
   `holdout_total_mean` and `holdout_total_std` fields.
5. All existing tests in `bp-train/tests/` and
   `bp-format/tests/` continue to pass.
6. `bp-format` augmented-process tests pass; `validate_case_study`
   reports unknown-parent errors clearly.
