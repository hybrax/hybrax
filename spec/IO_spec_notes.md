# IO spec notes

Purpose: discussion notes for iterating on `spec/IO_spec.md` before talking to
other devs.

## Notes from our discussion

### Pydantic config base

Use a shared parent for typed config sections instead of repeating
`model_config` on every `*Config` class.

```python
_FROZEN = ConfigDict(extra="forbid", frozen=True)

class ConfigBase(BaseModel):
    model_config = _FROZEN
```

All typed sections inherit from `ConfigBase`. Keep the free-form user hook dict
as `dict[str, Any]`, while unknown sibling keys in typed sections stay forbidden.

Note: `frozen=True` prevents attribute reassignment, but nested plain dicts can
still be mutated unless converted to immutable mappings.

### User hook config section name

Rename the user/hook section from `model` to `custom`.

Reasons:
- avoids confusion with pydantic's `model_*` protected namespace
- matches `custom.py` terminology already used by the CLI/codebase
- makes `custom.config` clearly mean free-form user hook defaults

Preferred shape:

```json
{
  "custom": {
    "module": "custom.py",
    "config": {}
  }
}
```

Unknown sibling keys under `custom` stay forbidden; only `custom.config` is
free-form.

### CLI flag surface / config purity

Be stricter about CLI flags. Avoid cherry-picking persisted experiment knobs as
flags. `--learning-rate` should likely be dropped: if one hyperparameter is
config-only, all training hyperparameters should be config-only.

Config-only optimizer fields:

```text
optim.learning_rate
optim.optimizer
optim.grad_clip_norm
optim.batch_size
optim.shuffle
optim.batch_seed
```

Open decision: maybe drop `--steps` too. Argument: it is also an optimization
hyperparameter. Counterargument: it is often invocation/run-length control,
especially for resume/extension workflows.

Stronger principle to consider:

> CLI should expose only invocation controls that are not persisted in the replay
> config/bundle. Anything persisted as experiment config should be edited in the
> config file, not also supplied via a flag.

This implies a split:

- Experiment config: affects trained model / reproduced forward pass. Persist as
  resolved replay config.
- Housekeeping/invocation controls: affect where/how the command writes or
  reports. Keep as CLI and/or run-record provenance, but not replay config.

Likely housekeeping, not experiment config:

```text
--output-dir
--log-level
--no-plot
--resume
--overwrite
```

`--output-dir` should not be inside replay config; moving/copying a run would
make the config self-referential/stale. Store actual run dir as provenance if
useful.

`--log-level` is command verbosity only.

`--no-plot` controls optional derived artifacts. If plots are regeneratable from
metrics/predictions, it need not be part of model replay config. Record as
provenance only if exact artifact reproduction matters.

Possible minimal train CLI:

```text
bp-train train --config config.json --output-dir RUN_DIR \
  [--resume RUN_DIR] [--overwrite] [--no-plot] [--log-level LEVEL]
```

Open decision: whether `--input`, `--custom`, `--process`, and `--steps` are
experiment config only (preferred) or convenience overrides. This also argues
against `checkpoint.resume` living in replay config.

### Separate input config from run record

Avoid making persisted `config.json` mean two things at once. Prefer:

- user input config: experiment/replay settings only
- run record: status, argv, timestamps, paths, hashes, environment, plus embedded
  resolved experiment config

This keeps command housekeeping out of replay config while still recording what
happened.

If we do this, consider naming the run record `run.json` or `run_record.json`
instead of overloading `config.json`.

### Prepare config keys

Likely typed `prepare` keys from current code:

```text
strict_bp_format_validation
required_control_names
require_consistent_controls
bolus_run_min_dt
initial_grid_points
max_rel_error
max_refinement_rounds
metadata_namespace
process_rename_map
```

`process_rename_map` is debatable: current default hook reads it, but it is still
a library-provided behavior. Decide whether default-hook knobs live in typed
`prepare` or in free-form `custom.config`.

### Metadata namespace

Spec currently suggests `metadata_namespace = "hybrax"`; current code defaults to
`"bp_train"` and downstream code expects that namespace. Decide explicitly before
implementation. Defaulting to `"bp_train"` seems least surprising for this package.

## Notes from original critical review / handover

### Strong points in current spec

- Single typed `RunConfig` fixes duplicated `TrainHarnessConfig` / `ForwardConfig`
  / `LOOConfig` ownership.
- `extra="forbid"` on typed sections fixes silent typo/no-op behavior.
- Splitting typed library knobs from free-form user hook config is the right line.
- CLI override semantics require passed/not-passed tracking, e.g.
  `argparse.SUPPRESS`, if any persisted config fields remain CLI-overridable.
- Bundling `custom.py` in the run dir is needed for reload stability.
- Saving optimizer state is required for real resume.
- Retention `best+latest` fixes checkpoint scatter.
- Making `metrics.csv` core supports plotting and resume continuity.
- Moving serialization/loading into one module and exposing `load_run` /
  `load_params` is well aligned with current duplicated reconstruction code.

### Spec tensions / risks to keep visible

- `load_run(run_dir)` is not truly self-contained if prepared data is referenced,
  not bundled. More precise: run dir + resolvable prepared data with matching
  content hash. Future bundle mode can make it fully portable.
- `checkpoint.resume` in config conflicts with treating resume as invocation
  intent. Prefer CLI-only resume unless there is a strong reason to persist it.
- Keep the config shallow: one typed section level, plus explicitly free-form
  `custom.config`; avoid sub-subsection creep.
- Unknown sibling keys under `custom` should be rejected; only `custom.config` is
  free-form.
- Generated defaults must handle required paths cleanly. Options: placeholders,
  nullable fields with command-specific validation, or `init` writing concrete
  starter paths. Avoid a defaults file that is invalid in confusing ways.
- Stable `content_hash` over prepared collections needs a canonical serializer
  that excludes provenance; raw byte hashing is too unstable.
- Resume bit-identical claims should be modest: need optimizer state, absolute
  batch indexing, same config/custom/data, same versions, and still may not be
  deterministic across machines.
- Background plotting should stay CSV/render-only in worker; JAX simulation stays
  in main process.
- If checkpointing is disabled (`checkpoint.every = 0`), still define exact final
  model/optimizer-state behavior for load/resume.
- Forward using recorded solver settings is reproducible but a UX tradeoff; only
  `--solver-max-steps` override is intentionally allowed.

### Prepare / data provenance details

- Current prepare provenance already exists and should be extended, not ignored.
- Prepared data should store stable content hash and config/provenance, with
  timestamps excluded from the hash.
- Train should verify the prepared data hash before reconstruct/load/resume.

## Additional agent notes added after discussion

### `custom.config` read tracking limits

Read tracking can warn on unused free-form keys, but it is heuristic. It may miss
or over-warn if hooks copy/iterate/convert the dict. Treat warnings as typo help,
not as a strict validation guarantee.

### Required paths and generated defaults

This overlaps with the review concern above, but is worth deciding explicitly for
`bp-train config --defaults`: placeholders vs nullable fields vs `init`-only
concrete paths.
