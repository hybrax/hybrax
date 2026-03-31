# Batch-Training Grill (Single Living File)

This is the single canonical file for the batching design interview.
Please add your responses under each `Your answer:` section.
I will append follow-up questions inline as we converge.

## Context

Current behavior (from `bp_train/harness.py`):
- one compiled step function per process,
- warmup compile for each process,
- per-process updates applied sequentially inside each global step,
- no batch construction.

Target direction:
- one batched training path,
- batches built from repeated selected processes (optionally shuffled),
- batch loss via `jax.vmap`,
- one param update per batch.

Stale-spec note:
- `spec/v1-detailed-spec.md` training-loop wording is stale relative to this new
  direction and should be updated after decisions are finalized.

## How to answer

- If you agree with recommendation: write `Agree`.
- If not: replace with your preferred answer.
- We proceed in order because questions are dependency-linked.

---

## 1. Scope And Interfaces

### Q1. Keep one training API or split old/new?

Question:
Should we replace current `train_collection(...)` internals with batched training,
or add a second function (for example `train_collection_batched(...)`) and keep the
old path temporarily?

Recommended answer:
Replace internals of `train_collection(...)` now and remove per-process compile loop.
Given early-stage project + no backward-compat requirement, this avoids dual-path
maintenance and hidden divergence.

Your answer: Agree

### Q2. Batch unit definition

Question:
Should one optimizer update correspond to one full batch (single gradient on mean
batch loss), or should we still do micro-updates inside a batch?

Recommended answer:
One optimizer update per batch only. No inner micro-updates.

Your answer: Agree

### Q3. Process selection semantics

Question:
Input includes `prepared_json` + selected `process_names`. If `process_names=None`,
should we include all processes exactly as in `store.process_order`?

Recommended answer:
Yes. Keep existing selection semantics: default is full `process_order`.

Your answer: Agree

### Q4. Terminology and loop counters

Question:
Current config uses `steps`. In batched training, do we redefine this as:
- number of optimizer updates (`num_updates`), or
- number of epochs over an index pool?

Recommended answer:
Treat `steps` as number of optimizer updates (keep name for now to minimize surface
change). We can add epoch-based wrappers later.

Your answer: Agree

### Q5. New config surface

Question:
Should we extend `TrainHarnessConfig` with batching controls now?

Recommended answer:
Yes, add at least:
- `batch_size: int = 8` (or similar default),
- `shuffle_batches: bool = True`,
- `drop_last_batch: bool = True` (recommended for compile stability),
- `batch_seed: int | None = None` (fallback to `seed` if `None`).

Your answer: Yes, we should use the number of processes by default though

### Q6. Drop-last policy

Question:
If the final batch is smaller than `batch_size`, should we drop it or keep it?
Keeping it can cause recompilation due to shape changes.

Recommended answer:
Drop last by default (`drop_last_batch=True`) to enforce fixed batch shape and avoid
extra compilations.

Your answer: Drop it, but only if shape is different

### Q7. Optimization rule

Question:
Do we stay with simple SGD-style `param -= lr * grad` for now, or switch to Optax
optimizer state now?

Recommended answer:
Stay with current no-state SGD update for this refactor. Keep scope focused on
batching/compilation behavior first.

Your answer: No let's switch to Optax. This is what we will use going forward anyway, might as well switch now already

### Q8. Loss weighting across processes

Question:
Per-process loss currently normalizes by active measurements and targets, then mean
across processes per step. In batched mode, should batch loss be plain mean of per-
process losses in batch?

Recommended answer:
Yes: batch loss = mean(per_process_loss). This preserves comparable behavior while
moving to batched updates.

Your answer: Agree

---

## 2. Batch Construction

### Q9. Dataset of indices per training run

Question:
Given selected processes, how should we construct the index pool used to form
batches over `steps` updates?

Recommended answer:
Construct a 1D process-index stream of length `steps * batch_size` (or longer before
drop/trim logic), sampled from selected process indices with replacement.
This directly models "repeats of processes" and decouples from epoch semantics.

Your answer: Agree

### Q10. Sampling strategy

Question:
Should repeats be:
- deterministic round-robin, or
- random sampling with replacement, or
- random permutation without replacement per epoch-like cycle?

Recommended answer:
Random sampling with replacement for now.
Reason: simple, fixed-size stream, no last-batch corner cases from uneven process
counts, and aligns with "repeats" requirement.

Your answer: I might misunderstand, but we're asking the user whether to shuffle, right? In that case why don't we have round-robin and shuffling will give us permutations anyway? Please correct me if I'm misunderstanding.

### Q11. Shuffle switch meaning

Question:
If `shuffle_batches=False`, what exact behavior should we use?

Recommended answer:
Use deterministic round-robin over selected process indices to build the stream,
then chunk into fixed-size batches. This gives fully reproducible non-shuffled runs.

Your answer: Agree

### Q12. Determinism contract

Question:
What should seed control exactly?

Recommended answer:
`batch_seed` (or `seed` fallback) controls only batch-index generation. Same seed +
same inputs + same config => same index stream and same update order.

Your answer: Agree (I guess it should control the shuffling as well if possible)

### Q13. Minimum-data edge case

Question:
If `len(selected_processes) == 1`, should batching still run through same pipeline?

Recommended answer:
Yes. A single process should still support `batch_size > 1` via repeated indices.
No special branch.

Your answer: Agree

### Q14. Validation for impossible settings

Question:
How should we handle invalid combinations like `batch_size <= 0`, `steps <= 0`, or
`drop_last_batch=True` with too-short index stream?

Recommended answer:
Hard fail early with explicit `ValueError` messages; do not silently adjust.

Your answer: Agree

### Q15. Exposing batch composition in results

Question:
Do we want to persist which process indices were used per step for debugging and
reproducibility?

Recommended answer:
Yes. Add optional compact telemetry in result:
- `batch_process_names_by_step: tuple[tuple[str, ...], ...]` or
- `batch_process_indices_by_step: tuple[tuple[int, ...], ...]`.

Your answer: Agree, in general we should write detailed training logs (keeping track of the loss etc.)

---

## 3. JAX Step Shape

### Q16. Single compiled step function

Question:
Do we commit to exactly one compiled train-step function per run (per static config),
rather than one per process?

Recommended answer:
Yes. Build one `eqx.filter_jit` step that takes `batch_process_indices` and computes
all sample losses through `jax.vmap`.

Your answer: Yes, in hybrax-train compile times got quite long so we definitely want to try and compile only as many times as is strictly necessary.

### Q17. Where to put process-specific data access

Question:
Should process-specific tensors (`y0`, `t_meas`, `y_meas`, `meas_mask`, controls)
be selected via JAX indexing inside the vmapped loss (using process index), instead
of constructing per-process wrapper objects in Python?

Recommended answer:
Yes. Move to process-indexed array access inside JAX-traced code.
That is the key to a single compile path.

Your answer: Yes

### Q18. Wrapper strategy for batching

Question:
Current `LibraryRhsWrapper` embeds one `PerProcessControls` object. For batched
training, do we:
- create a new batched wrapper abstraction, or
- keep wrapper mostly as-is and add a process-indexed control eval function?

Recommended answer:
Keep current wrapper for now but add a process-index-aware path used by trainer,
e.g. helper functions that evaluate controls for `(process_idx, t)` from stacked
`ControlsStore` arrays.

Your answer: Add another class with controls for all processes that can be created with a method of `ControlsStore` and that is minimal. It has an `eval(process_idx, t)` method and wraps the necessary breaks and values arrays but no other bells and whistles (no fancy init, no pytree leaves like process names, etc.). The sole point of this class is efficient controls interpolator evaluation.

### Q19. Vmap granularity

Question:
Should we vmap over process samples only (batch axis), with each sample still doing
its own diffrax solve on that process’s measurement times?

Recommended answer:
Yes. `jax.vmap(single_sample_loss, in_axes=(0,))` over `batch_process_indices`.
Each sample keeps its process-specific ODE solve and masked measurement loss.

Your answer: Agree

### Q20. Handling variable `n_meas` in one compiled path

Question:
Measurement lengths differ by process (`n_meas`). Do we support this by always
solving on padded `t_meas` and masking, or by dynamic slicing active prefix?

Recommended answer:
Use dynamic slicing to active prefix for solver `ts` and keep mask-based loss on
padded arrays. This avoids integrating useless tail timestamps.

Your answer: Agree, but we should add to the spec that this is something we might want to benchmark later (dynamic slice vs padding `ts` with `ts[-1]` or `NaN` if those work with diffrax.)

### Q21. Compile stability and batch shape

Question:
Do we enforce fixed batch shape at runtime to avoid recompiles from shape drift?

Recommended answer:
Yes. Keep `batch_process_indices.shape == (batch_size,)` always.
If needed, drop last or prebuild exact-size index stream.

Your answer: This should be controlled by the drop-last-batch flag as discussed above.

### Q22. Initial compile telemetry

Question:
How should compile telemetry change once there is one compiled step?

Recommended answer:
Replace per-process compile metrics with run-level metrics:
- `compile_time_seconds` (single value),
- `compile_count` (expected 1 unless signature changes).

Your answer: Let's only get the first compile time and then the run times of the later batches for now. We can look into proper profiling / compiler traces later.

### Q23. Recompile detection strategy

Question:
Current code tracks partition signatures and recompiles per process when signature
changes. Keep this logic?

Recommended answer:
Keep a simplified single-signature check around the one step function; if signature
changes unexpectedly, recompile once and increment run-level compile count.

Your answer: Let's simply include the signature of the JIT boundary function (that would be train step I guess) in the logs (it should include the shapes of each input array).

---

## 4. Telemetry, Tests, And Rollout

### Q24. Result object compatibility

Question:
`TrainHarnessResult` currently includes per-process step times and compile data.
Should we keep these fields, deprecate them, or replace now?

Recommended answer:
Replace now with batch-oriented fields.
Given early-stage project and no compatibility requirement, old per-process compile
fields are misleading after this refactor.

Your answer: Agree, replace with batch-oriented telemetry. However, like discussed above, the telemetry can be lighter for now; we can always crank it up later if needed.

### Q25. Per-process loss history under batching

Question:
Do we still need `loss_by_process` histories every global step?
This is ambiguous once a step touches only sampled batch members.

Recommended answer:
Switch to:
- `batch_mean_loss_by_step`,
- optional `batch_loss_array_by_step` (per-sample losses in the batch),
- optional aggregate running means per process for diagnostics.

Your answer: similar to hybrax-train keep track of the per-process loss in each step and add to the logs. We want to plot this information after training.

### Q26. Logging cadence

Question:
Current logging is step-based (`log_every`). Keep same semantics?

Recommended answer:
Yes, keep step-based logging; report batch size, sampled process names, mean batch
loss, and step wall-time.

Your answer: Agree (and include function signature array shapes etc. as discussed above (and anything else that we might have mentioned above that I forgot))

### Q27. Test strategy (minimum)

Question:
What minimum test set do we require before merging?

Recommended answer:
Add/adjust tests to cover:
- batch-size/repeat behavior (including single-process + batch_size>1),
- deterministic batch indices with fixed seed,
- loss decreases on toy data (at least sanity trend),
- invalid batching config errors.

Your answer:
Add/adjust tests to cover:
- batch-size/repeat behavior (including single-process + batch_size>1),
- deterministic batch indices with fixed seed,
- loss decreases on toy data (at least sanity trend),
- invalid batching config errors,
- stable train-step input signature across all updates,
- no explicit re-jit/rebuild path triggered in our code.

### Q28. Should we use PyTorch `DataLoader` now?

Question:
You said dataloader is optional. Do we introduce torch dependency into runtime path
now, or keep pure JAX/Numpy index batching?

Recommended answer:
Do not use PyTorch dataloader now. Keep index batching in pure JAX/Numpy for lower
complexity and fewer dependencies.

Your answer: Agree

### Q29. Delivery split

Question:
Should implementation land in one PR/task, or two phases?

Recommended answer:
Two phases:
1. functional batching + single compile + tests,
2. richer telemetry/reporting cleanup and spec sync.

Your answer: Don't add this to the spec. We can worry about how many phases to implement it in later. The spec should only describe how the code should work.

### Q30. Spec update scope

Question:
Which spec sections must be updated once decisions are finalized?

Recommended answer:
At minimum in `spec/v1-detailed-spec.md`:
- training-loop description,
- compile behavior expectations,
- batching config contract,
- result telemetry contract.

Your answer: Agree

---

## Follow-up Questions

I will append targeted follow-ups here once you answer initial questions.

### FU1. Default `batch_size`

Why this follow-up:
You chose default `batch_size = number of selected processes`, which conflicts with
fixed static defaults when process subsets vary between runs.

Question:
Do we set `TrainHarnessConfig.batch_size: int | None = None` and resolve default at
runtime as `len(selected_processes)` when `None`?

Recommended answer:
Yes. Use `None` as config default, resolve to `len(selected_processes)` at runtime,
and validate `>= 1`.

Your answer: Agree

### FU2. `drop_last_batch` behavior and compile stability

Why this follow-up:
You said "drop it, but only if shape is different". Shape differs exactly when
batch is incomplete, which means runtime branch behavior must be explicit.

Question:
If `drop_last_batch=False`, do we accept potential recompiles on the final smaller
batch, or do we pad the final batch to full size (for example by reusing early
indices) to keep shape fixed?

Recommended answer:
Pad to full size when `drop_last_batch=False`.
This keeps single-shape JIT behavior and avoids recompilation while still using all
sampled indices.

Your answer: If padding to full size with reused earlier indices (again randomly
sampled) is an option, I guess we can actually drop the `drop_last_batch` flag.
I hadn't though of this but we should just increase the total number of samples
to `ceil(N_total / N_per_batch) * N_per_batch`. This way we'll never have an odd
batch at the end.

### FU3. Sampling mode model

Why this follow-up:
Q10/Q11 currently mixes "sampling with replacement" and "round-robin + shuffle".
We need one coherent algorithm.

Question:
Which explicit index-stream mode do you want?
- `round_robin` (deterministic cycle through selected indices),
- `round_robin_shuffled` (shuffle each cycle),
- `iid_with_replacement`.

Recommended answer:
Implement `round_robin` + optional per-cycle shuffle (`round_robin_shuffled`) first.
This matches your expectation that shuffle permutes deterministic coverage.

Your answer: Agree

### FU4. Seed ownership

Why this follow-up:
You want seed to control shuffling too.

Question:
Should `batch_seed` control all index-stream randomness (cycle shuffling and any
padding randomness if used), with deterministic behavior when `shuffle_batches=False`
(seed ignored)?

Recommended answer:
Yes.

Your answer: Agree, one seed for all the randomness.

### FU5. Optax migration scope

Why this follow-up:
Switching to Optax affects config and result payloads.

Question:
Which optimizer config should we expose now?

Recommended answer:
Keep v1 minimal but explicit:
- `optimizer_name: Literal["adam", "sgd"] = "adam"`
- `learning_rate: float`
No momentum/betas in config yet; use Optax defaults per optimizer.

Your answer: Agree

### FU6. Per-step per-process loss logging definition

Why this follow-up:
You want per-process loss for plotting each step. Full all-process evaluation each
step is expensive and may dominate runtime.

Question:
What should "per-process loss in each step" mean operationally?
- only sampled processes in current batch (cheap), or
- all selected processes every step (expensive), or
- periodic full sweep every `eval_every` steps.

Recommended answer:
Log sampled-process losses each step, plus optional full-sweep evaluation every
`eval_every` steps (default disabled).

Your answer: Above you mentioned a `log_every` parameter. Every `log_every`-th step we should write out the per-process loss (including it with the other logged stuff). Please let me know if I'm wrong, but I don't see how it can be expensive? we're getting the per-process loss anyway before we're calculating the mean. Look at hybrax-train to see how this is handled there. 

### FU7. Compile telemetry payload

Why this follow-up:
You requested function-signature shape logging.

Question:
Where should signature metadata live?

Recommended answer:
Store in result as compact structured field:
- `jit_signature: tuple[tuple[str, tuple[int, ...], str], ...]`
and also log a one-line summary.

Your answer: Agree

### FU8. `BatchControls` class contract

Why this follow-up:
You requested a new minimal all-process controls evaluator class.

Question:
Should this class live in `controls_store.py` and expose exactly:
- `eval(process_idx: int, t: jax.Array) -> jax.Array`
- no process names / metadata fields,
- no Python dict lookups in the hot path?

Recommended answer:
Yes.

Your answer: Agree

### FU9. Test coverage correction

Why this follow-up:
Your Q27 answer omitted explicit single-compile assertion (which is the primary goal
of this refactor).

Question:
Do we explicitly assert single-compile behavior in tests (for stable batch shape)?

Recommended answer:
Yes, assert compile count remains 1 for a representative multi-process run.

Your answer: I omitted this because I think it's hard to actually measure "compile count". For a start I think it's enough if we keep track of the shapes of arrays of the train step function and include this in the results object and logs.

### FU10. Spec-level note on dynamic-slice vs padded-ts benchmark

Why this follow-up:
You requested a benchmark note in the spec.

Question:
Should we add a short "future benchmark" note under training-runtime section stating
that dynamic-slice vs padded-ts strategies are implementation-equivalent candidates
for later profiling?

Recommended answer:
Yes, add as non-normative note.

Your answer: Agree

### FU11. `steps` semantics vs "pad total samples"

Why this follow-up:
Your FU2 answer suggests computing total samples via
`ceil(N_total / N_per_batch) * N_per_batch`, which is an epoch-style formula.
But we already agreed `steps = number of optimizer updates`.

Question:
Can we lock this as:
- `steps` is authoritative optimizer-update count,
- total sampled indices is always exactly `steps * batch_size`,
- therefore no incomplete final batch can occur,
- and we remove `drop_last_batch` entirely?

Recommended answer:
Yes, lock exactly this and delete `drop_last_batch` from config.

Your answer: Agree

### FU12. What "per-process loss" means in logs

Why this follow-up:
Your FU6 note is correct for sampled processes: those losses are already computed.
The expensive case is "all selected processes every log step".

Question:
Please choose one explicit logging contract:
- A) log losses for sampled batch members only each `log_every` step,
- B) additionally run full-eval over all selected processes each `log_every` step.

Recommended answer:
A only (sampled members only) for the training loop. Full-eval can be a separate
post-training or optional eval mode later.

Your answer: Agree, let's do A only (in reality batch size will usually be the same as or larger than num_processes. No need to be too worried about run time here).)

### FU13. Recompile observability test

Why this follow-up:
You are right that true compiler-internal compile count is hard. We still need one
practical regression signal in tests.

Question:
Should we test both:
1) train-step input signature stays constant across all updates,
2) no explicit re-jit/rebuild path is triggered in our code?

Recommended answer:
Yes. Use signature-stability + no-rebuild flag/counter as proxy for single-compile
behavior.

Your answer: Agree

### FU14. Q27 still missing answer

Why this follow-up:
Q27 `Your answer:` is still empty.

Question:
Should Q27 include the new proxy test from FU13 explicitly?

Recommended answer:
Yes, include it as an explicit required test item.

Your answer: Agree

### FU15. Minor consistency cleanup

Why this follow-up:
Current answers imply these final config fields for batching. Please confirm final
shape so implementation can start without ambiguity.

Proposed final batching config:
- `steps: int` (optimizer updates)
- `batch_size: int | None = None` (resolved to `len(selected_processes)`)
- `shuffle_batches: bool = True`
- `batch_seed: int | None = None`
- no `drop_last_batch`

Question:
Confirm this final config set?

Recommended answer:
Yes.

Your answer: Agree
