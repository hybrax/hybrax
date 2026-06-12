# bp-train config input spec

Scope: input configuration only. This spec deliberately does **not** specify the
future FAIR run directory layout, checkpoint retention, optimizer-state
persistence, bundling, or public `load_run` implementation details.

This document is the durable contract. Temporary implementation notes and
current-code pointers live in `tmp/config-input-implementation/HANDOVER.md`.
If a detail affects user-visible behavior, keep it here; if it only helps the
next implementation session find/edit current code, keep it in the handover.

## Goal

Replace the current mix of CLI flags, `custom.py CONFIG`, and ad-hoc `--config`
JSON with one structured input config that answers:

- which experiment knobs exist
- which knobs are library-owned and validated
- which knobs are case-study-specific and passed to `custom.py`
- which command-line options remain outside the experiment definition

Core rule:

> Anything needed to reproduce the scientific/training result lives in the config
> file. CLI flags are only for execution controls: how this command invocation is
> run, displayed, resumed, or written.

Use **execution controls** instead of “housekeeping”: these flags control command
execution, not experiment identity.

## Configuration format

- Canonical format: JSON.
- Validation: pydantic v2.
- Typed library-owned sections reject unknown keys.
- The top-level `custom` section is intentionally free-form. bp-train only
  checks that it is a JSON object or `null` before passing it to `custom.py`.
- YAML/comments are out of scope.

## User mental model

Two keyspaces exist:

1. **Typed experiment config**: library-owned, validated, reproducibility-relevant.
2. **`custom` config**: user/case-study-owned, free-form, passed to hooks.

A key belongs in `custom` only if user hook code reads it. Otherwise it belongs in
a typed section.

Examples:

- `target_variable_order` -> `data.targets`
- `target_source` -> `data.target_source`
- `strict_bp_format_validation` -> `prepare.strict_bp_format_validation`
- `bolus_run_min_dt` -> `prepare.bolus_run_min_dt`
- `bounds_weight`, `target_loss_weights` -> `custom`, if read by custom hooks

## Config shape

Use shallow, responsibility-based sections. One typed section level only; avoid
sub-subsection creep.

```json
{
  "data": {
    "prepared": "prepared.json",
    "processes": null,
    "targets": null,
    "target_source": "auto"
  },
  "custom_py": null,
  "train": {
    "steps": 50,
    "seed": 0,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "grad_clip_norm": 1000.0,
    "batch_size": null,
    "shuffle": true,
    "batch_seed": null
  },
  "solver": {
    "max_steps": 2048,
    "rtol": 1e-5,
    "atol": 1e-7,
    "jump_ts": true
  },
  "prepare": null,
  "custom": {}
}
```

Rationale for `custom_py` + `custom` split:

- `custom_py` is the path to `custom.py`. It is typed because the custom
  code defines the model/hook behavior and is needed to reproduce training. Its
  default is `null`.
- Later run bundling/checkpoint specs may copy this module into the run
  directory. That does not make the input path a CLI-only concern: the training
  invocation still needs a reproducible source reference before bundling happens.
- On config resolution, compute and retain the custom module file hash alongside
  the resolved config/run record. If `custom_py` is `null`, do not load or hash a
  module and record `custom_py_sha256: null`. For the first milestone, train
  writes this train-time hash/null to the existing `trained_wrapper.meta.json`
  sidecar. Do not compare it against the prepare-time hash or fail on mismatches
  yet; later bundling/checkpoint specs can use the recorded hashes for
  provenance.
- `custom` is the user/case-study config blob. It is intentionally not validated
  by bp-train at initial parsing.

Path rule: relative paths in bp-train-owned typed `Path` fields resolve relative
to the config file's parent directory, not the process CWD. CLI
execution-control paths resolve relative to the invocation CWD unless documented
otherwise. Free-form `custom` values are not interpreted by bp-train; if a custom
field contains a path, `custom.py` owns its validation/resolution. For example,
custom code may resolve paths relative to `custom.py` itself via `__file__`, or
use any other case-study-specific convention.

Validation rule: parse and validate only the command-relevant config view. For
example, `bp-train prepare` validates `prepare`, `custom_py`, and `custom`;
`bp-train train` validates `data`, `train`, `solver`, `custom_py`, and
`custom`. Do not require train-only fields for a prepare-only use case, or
prepare-only fields for a train-only use case. Shared config files may contain
known top-level sections for other commands; unknown top-level keys are still
rejected. Typos inside command-irrelevant sections are caught when running the
command that uses that section.

Dynamic default rule: some fields may use `null` to mean "derive an effective
value from the loaded data at command runtime" instead of using a fixed pydantic
default. These fields remain explicit typed fields in the input config, but
resolution happens after the command-relevant data has been loaded. Hooks that
need command runtime semantics receive the effective `RunConfig`, not the raw
input object, and metadata records the effective value when it affects prepared
or trained artifacts. Current dynamic default field: `prepare.bolus_run_min_dt`.
If omitted/null, prepare detects a run-level bolus/sample ramp width from the
collection when needed, passes that effective value to prepare hooks, and records
it in prepared metadata. Future dynamic defaults should be resolved in the same
command-level effective-config step rather than by ad-hoc per-field helpers.

## Pydantic model sketch

```python
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_FROZEN = ConfigDict(extra="forbid", frozen=True)


class ConfigBase(BaseModel):
    model_config = _FROZEN


class DataConfig(ConfigBase):
    prepared: Path
    processes: tuple[str, ...] | None = None
    targets: tuple[str, ...] | None = None
    target_source: Literal[
        "process_variables",
        "reactor_components",
        "auto",
    ] = "auto"


class TrainConfig(ConfigBase):
    steps: int = Field(50, gt=0)
    seed: int = 0
    optimizer: Literal["adam", "sgd"] = "adam"
    learning_rate: float = Field(1e-3, gt=0)
    grad_clip_norm: float = Field(1000.0, ge=0)
    batch_size: int | None = Field(None, gt=0)
    shuffle: bool = True
    batch_seed: int | None = None


class SolverConfig(ConfigBase):
    max_steps: int = Field(2048, gt=0)
    rtol: float = Field(1e-5, gt=0)
    atol: float = Field(1e-7, gt=0)
    jump_ts: bool = True


class PrepareConfig(ConfigBase):
    raw_input: Path
    case_study: str | None = None
    strict_bp_format_validation: bool = False
    required_control_names: tuple[str, ...] | dict[str, tuple[str, ...]] = ()
    require_consistent_controls: bool = True
    bolus_run_min_dt: float | None = Field(None, gt=0)
    initial_grid_points: int = Field(16, gt=0)
    max_rel_error: float = Field(1e-4, gt=0)
    max_refinement_rounds: int = Field(8, ge=0)
    process_rename_map: dict[str, str] = Field(default_factory=dict)


class RunConfig(ConfigBase):
    data: DataConfig | None = None
    custom_py: Path | None = None
    train: TrainConfig = TrainConfig()
    solver: SolverConfig = SolverConfig()
    prepare: PrepareConfig | None = None
    custom: Any | None = None
```

Notes:

- The shared `ConfigBase` avoids repeated `model_config` declarations.
- `frozen=True` prevents attribute reassignment, but nested plain dicts can still
  be mutated unless converted to immutable mappings.
- `custom` is the only unchecked/free-form input section. Unknown keys anywhere
  else fail at parse time.
- `data` and `prepare` are optional at top level because required sections are
  command-specific. `train` requires `data`; `prepare` requires `prepare`.
- A field named `custom` avoids pydantic `model_*` namespace confusion.

## Command/section matrix

| Command | Reads typed experiment sections | Reads `custom` | Main execution controls |
|---|---|---:|---|
| `prepare` | `prepare`, `custom_py` | yes | `--config`, `--output`, `--log-level` |
| `train` | `data`, `custom_py`, `train`, `solver` | yes | `--config`, `--output-dir`, `--no-plot`, `--log-level` plus existing metrics/checkpoint execution flags |
| `forward` | recorded run config; optionally output-filtered processes | no new training custom config | `--run`, `--output-dir`, `--process`, `--solver-max-steps`, `--no-plot`, `--log-level` |

Only `prepare` and `train` are first-milestone implementation targets. The
`forward` row records the intended direction so `train` metadata decisions do
not conflict with later work.

`--resume` is also an execution control, but it is reserved for the later
resume/run-layout spec and is outside the first config-input implementation. Do
not put resume intent in the experiment config.

Prepare output is an execution-control path and is supplied via `bp-train prepare
--output PATH`. `prepare` does not read or validate `data.prepared`. A later
`train` reads `data.prepared` from config.

## Custom config resolution

There are two keyspaces.

### Library-owned typed config

Lowest to highest:

```text
pydantic defaults < JSON config
```

There is intentionally no CLI override layer for typed experiment fields. If a
value is part of reproduction, edit the config file.

### User-owned custom config

`custom.py` has one optional point of contact for custom config:

```python
def get_custom_config(raw_custom: dict[str, Any] | None, config: RunConfig) -> Any:
    ...
```

bp-train does not inspect `CONFIG`, `CustomConfig`, or other module globals. The
user decides what this hook does: return defaults, merge defaults with
`raw_custom`, validate with pydantic, validate with plain Python, or return any
object hooks should later read.

Resolution is two-stage because custom config handling may live in `custom.py`:

1. Read the raw JSON and separate the raw `custom` value from the rest of the
   config. Omitted and `null` both mean `None`; `{}` is distinct and is passed as
   an empty dict.
2. Parse typed non-custom config with `custom=None`, resolving `custom_py` and
   config-relative paths. The raw `custom` value only gets a simple shape check:
   it must be a JSON object or `null`.
3. If `custom_py` is set, load `custom.py` and compute the custom module file
   hash. If `custom_py` is `null`, skip module loading and use hash `None`.
4. If `custom.py` defines `get_custom_config`, call it with the raw JSON `custom`
   value and the typed config parsed so far, where `config.custom is None`.
   Store its return value as `config.custom` by creating a new frozen config via
   `model_copy(update={"custom": resolved_custom})`.
5. If the hook is undefined but JSON `custom` is present, wrap that dict in
   bp-train's default permissive custom model and store it as `config.custom`.
6. If the hook is undefined and JSON `custom` is absent/null, set
   `config.custom = None`.

Sketch:

```python
class DefaultCustomConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
```

## What `custom.py` receives

Hooks receive the resolved bp-train config object, not a loose merged dict. That
object includes typed library config and resolved custom config:

```python
config.data.targets
config.train.seed
config.solver.rtol
config.custom.bounds_weight      # if get_custom_config returned a typed object
```

This keeps one obvious object for hook authors while preserving the separation:
library-owned fields are typed by bp-train, and user-owned fields are produced by
`custom.py` when it opts in. Because the same root config type is used for
multiple commands, hooks should only assume the command-relevant sections are
present: prepare hooks can assume `config.prepare is not None`, while train hooks
can assume `config.data is not None`.

Hook contract:

- Hook signatures do not need a full redesign before implementation. For each
  hook, keep arguments that are currently needed and not obviously available from
  `RunConfig`; replace only the old free-form `config` dict argument with the
  resolved `RunConfig`.
- Prepare hooks (`transform_process_collection`, `build_sample_acc_series`) get
  the resolved config object.
- Train hooks (`estimate_all_scales`, `build_reaction_module`,
  `build_loss_module`, custom optimizer/lr schedule hooks) get the resolved
  config object.
- No transition adapter/backcompat is needed.
- Library/default behavior reads typed config fields, not ad-hoc dict keys.

`process_rename_map` is therefore typed as `prepare.process_rename_map`; it is
not an unchecked custom key.

## Minimal config examples

Minimal prepare config:

```json
{
  "prepare": {
    "raw_input": "raw_collection.json"
  },
  "custom_py": null
}
```

Prepare output is supplied separately:

```text
bp-train prepare --config config.json --output prepared.json
```

Minimal train config:

```json
{
  "data": {
    "prepared": "prepared.json"
  },
  "custom_py": null
}
```

`train` and `solver` can be omitted when defaults are desired. A shared pipeline
config may contain both `prepare` and `data` sections, but each command validates
only the sections it uses. `bp-train train` errors if `data` is absent;
`bp-train prepare` errors if `prepare` is absent.

## CLI surface

CLI flags are execution controls only. They must not define the experiment. For
this milestone, `--config` is required for both `prepare` and `train`; legacy
experiment-field CLI invocations are rejected.

Suggested train CLI:

```text
bp-train train --config config.json --output-dir RUN_DIR \
  [--plot|--no-plot] [--log-level LEVEL]
```

Suggested prepare CLI:

```text
bp-train prepare --config config.json --output prepared.json \
  [--log-level LEVEL]
```

Suggested forward CLI:

```text
bp-train forward --run RUN_DIR \
  [--output-dir DIR] [--process NAME ...] [--solver-max-steps N] \
  [--no-plot] [--log-level LEVEL]
```

Forward notes:

- `--run` identifies the trained run to consume. Prefer this over exposing a model
  file path as the primary UX.
- `--process` is an output filter only. It must not change targets or training
  identity.
- `--solver-max-steps` is a safety cap override only. Accuracy settings
  (`rtol`, `atol`, `jump_ts`) come from the recorded training config.

Prepare flags that should **not** exist after this milestone:

```text
--input
--custom
--case-study
```

These are experiment fields and belong in JSON (`prepare.raw_input`,
`custom_py`, and `prepare.case_study`).

Train flags that should **not** exist after this milestone:

```text
--input
--custom
--process
--target
--target-source
--steps
--seed
--batch-seed
--shuffle-batches
--learning-rate
--optimizer
--batch-size
--grad-clip-norm
--solver-max-steps
--solver-rtol
--solver-atol
--no-jump-ts
```

These are experiment fields and belong in JSON.

Telemetry/output flags are execution controls for this milestone:

- `--log-level`, existing plot flags (`--plot` / `--no-plot`), output locations
- existing metrics/checkpoint paths/cadence flags

Do not add new execution flags in this milestone just because this spec mentions
them. For example, if `--overwrite` does not exist today, do not add it now. Do
not redesign checkpoint or resume behavior now; keep only existing checkpoint
execution-control flags needed by the current training path.

## Prepare-specific config

Current prepare-time library keys that should become typed fields:

```text
raw_input
case_study
strict_bp_format_validation
required_control_names
require_consistent_controls
bolus_run_min_dt
initial_grid_points
max_rel_error
max_refinement_rounds
process_rename_map
```

Metadata namespace is not configurable. Use the fixed metadata key `"bp-train"`
consistently in prepare and downstream training code.

Prepared provenance stores `metadata["bp-train"]["source_input_path"]` relative
_to the prepared artifact file_ when the input path is absolute after config
resolution. This keeps regenerated prepared artifacts portable: the value is
interpreted relative to the directory containing the prepared JSON file, not
relative to the config file or `custom.py`.

## LOO behavior

Deferred from the first milestone. Current LOO orchestrates independent folds
sequentially in one Python process, with one fresh training run per holdout fold.
Its config semantics should be specified after the `prepare`/`train` config path
is working.

## Generated config commands

Backburner / not first milestone:

```text
bp-train config --defaults
bp-train config --schema
```

These are useful discovery/on-ramp commands, but the first implementation
milestone is just: one JSON config file works for `prepare` and `train`.

Future requirements:

- Defaults/schema are generated from pydantic models, not committed as static
  files.
- Every typed field should have a useful description in the schema.
- `config --schema` prints `model_json_schema()` for the relevant model.
- `config --defaults` prints a template JSON showing required fields plus all
  optional fields with defaults. Required path fields can use clear placeholders
  such as `"<path to prepared.json>"`.
- Support command-specific views, e.g. `bp-train config --defaults train` and
  `bp-train config --defaults prepare`, so users see the minimal required fields
  for the command they want to run.

## First-milestone acceptance checklist

The first implementation milestone is complete when:

- `bp-train prepare --config config.json --output prepared.json` reads
  `prepare.raw_input`, optional `prepare.case_study`, typed prepare controls,
  `custom_py`, and `custom` from the JSON config.
- `bp-train train --config config.json --output-dir RUN_DIR` reads
  `data.prepared`, data selection, train settings, solver settings,
  `custom_py`, and `custom` from the JSON config.
- Experiment-defining prepare/train CLI flags listed above are gone or rejected.
- Existing execution-control flags needed by the current prepare/train paths
  still work.
- Unknown keys in command-relevant typed sections fail validation with useful
  error locations; arbitrary keys under `custom` are accepted.
- Relative paths in typed bp-train fields resolve relative to the config file;
  execution-control paths remain invocation-CWD relative.
- `custom.py get_custom_config(raw_custom, config)` is honored, and bp-train does
  not inspect `CONFIG`, `CustomConfig`, `get_config`, or other globals.
- All prepare/train hooks receive the resolved config object instead of the old
  free-form config dict.
- Prepared metadata and downstream readers use fixed metadata namespace
  `"bp-train"`.
- Train writes the train-time `custom_py_sha256` to the existing
  `trained_wrapper.meta.json` sidecar.
- Examples/docs no longer rely on framework-level `custom.py CONFIG` for
  bp-train-owned fields.

Deferred from first milestone:

- `bp-train config --defaults` / `--schema`
- forward/run-dir consumer changes beyond not adding new config ambiguity
- LOO config migration
- checkpoint/resume/run-layout changes

Tests to add:

- unknown typed keys fail with useful paths
- arbitrary JSON `custom` keys are accepted by the default custom config wrapper
- optional `custom.py get_custom_config` produces `config.custom`
- `config.custom` is `None` when no hook exists and JSON `custom` is absent/null
- custom module file hash is computed during config resolution and written to
  the existing train sidecar
- relative config paths resolve relative to the config file
- removed experiment CLI flags are rejected
- prepare consumes typed fields (`case_study`, controls/grid knobs, rename map)
- train target/source/seed/optimizer/solver come from config, not `custom.py`
- generated defaults/schema stay in sync with models (when config command is implemented)

## Decisions still needed before first implementation milestone

No known first-milestone design questions remain. Proceed with implementation
planning for `prepare` and `train` config input.

Backburner decisions:

- Exact `bp-train config --defaults` output shape.
- Exact `bp-train config --schema` command UX.

## Out of scope for this spec

- checkpoint layout and retention
- optimizer-state save/resume mechanics
- run directory FAIR provenance format
- bundling `custom.py` / prepared data
- public `load_run` / `load_params` APIs
- LOO config migration
