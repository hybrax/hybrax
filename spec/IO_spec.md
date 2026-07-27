# bp-train: structured config, FAIR run directory, checkpointing & resume

## Context

`bp-train` trains well, but **both ends of a run — its input configuration and its on-disk
output — are weak on reproducibility (FAIR)**. A run is only reproducible if the inputs are
captured *and* the outputs are self-contained. This plan fixes the input side first (a typed
structured config), then the output side (a self-contained run directory that serialises
that same config), then the loader that closes the loop.

### Input today — three planes, blurry ownership

| Plane | Lives in | Owns | Typed? | Problem |
|---|---|---|---|---|
| CLI flags | argparse `train` ([cli.py:70-258](bp-train/bp_train/cli.py#L70)) | data sel, optim, solver, ckpt, output, telemetry | per-flag | ~25 flat flags, no grouping cue |
| `CONFIG` dict | `custom.py` | model/case-study knobs (`bounds_weight`, `target_loss_weights`, `bolus_run_min_dt`, …) | **no** | free-form; typos silent |
| `--config` JSON | runtime file | overlay on `CONFIG` | **no** | same keyspace as `CONFIG`, merged implicitly |

A new user who just downloaded the package faces: *which knob lives where?*
(`target_variable_order` is settable in all three with silent precedence,
[cli.py:674-681](bp-train/bp_train/cli.py#L674)); *what is even exposable?* (no single
list); *why did my `CONFIG["bounds_wieght"]` typo do nothing?* (silent default); and
*how do I train my very first model?* (every example ships a 600-line `custom.py`, so it
looks mandatory even though `DefaultReactionModule` exists).

### Output today — file scatter, not reloadable, no resume

- **File scatter / "which file do I load?"** Every `--log-every` step writes a full
  `checkpoints/step_NNNNN/` dir with `trained_wrapper.eqx` + `.meta.json` +
  `loss_curve.png` + `grad_norm_curve.png` + `predictions.csv` (~570 KB/ckpt), and the run
  root duplicates the final set. A 1500-step / log-every-50 run = ~30 dirs, 150+ files,
  ~19 MB. Re-running into the same `--output-dir` silently overwrites some files and
  accumulates others — hard to tell which run produced what.
- **Not shareable / not reload-stable.** The model is reloaded by *reconstructing its
  architecture* — `forward` re-runs the `custom.py` hooks against `prepared.json`
  ([cli.py:976](bp-train/bp_train/cli.py#L976), [harness.py:479](bp-train/bp_train/harness.py#L479)).
  But `custom.py` is only *referenced by relative path* in the sidecar
  ([cli.py:746](bp-train/bp_train/cli.py#L746)); if it moves or changes, the saved model
  can no longer be loaded. Nothing is bundled.
- **No optimizer state → no resume.** `optimizer_state` is created once
  ([harness.py:768](bp-train/bp_train/harness.py#L768)) and never persisted. Resume is an
  explicit V1 non-goal ([v1-detailed-spec.md:42](bp-train/spec/v1-detailed-spec.md#L42)).
- **CLI flags only partially recorded.** The sidecar stores a curated subset; `optimizer`,
  `learning_rate`, `grad_clip_norm`, `shuffle_batches`, `batch_seed`, `log_every`, the raw
  argv, and package versions are lost — the run isn't reproducible.

### Outcome

One **typed, structured, round-trippable config** that is the single source of truth for
"what's exposable"; one **self-contained, FAIR run directory** that serialises exactly that
config plus provenance, bundles `custom.py`, saves optimizer state (so
`bp-train train --resume <run_dir>` works), and renders per-checkpoint plots off the
training critical path; and one **`bp_train.load_run(dir)`** call that reconstructs a
trained model from the directory alone. The unifying insight: **the `config.json` written as
output IS a valid input** — so replay/resume/forward/load all share one schema.

Per the repo's **delete > deprecate** rule, the old config planes (free-form `--config`
overlay, library knobs in `CONFIG`) and the old layout (`trained_wrapper.eqx`,
`trained_wrapper.meta.json`, per-step plot/CSV duplication) are **replaced wholesale**.

### Decisions locked with the user

**Configuration**
- **Structured, shallow** config — one level of nesting, sections by responsibility; CLI
  stays flat and maps onto sections.
- **Pydantic v2 models** (already installed: `pydantic 2.13.3`; declare it in `pyproject`).
  Typed validation, `extra="forbid"` fail-fast on unknown keys, native JSON round-trip,
  bounds/choices via `Field`/`Literal`, path-pointed errors, auto knob reference via
  `model_json_schema()`.
- **JSON canonical** (pydantic-native, no new dep). Inputs accept whole-line `//`
  comments after optional indentation; inline/block comments and other JSON5-only
  syntax remain invalid. The stdlib parser and generated files retain its
  `NaN`/`Infinity` extensions for non-finite floats. PyYAML is **not** installed.
- **CLI = a few common flags only; `config.json` carries the full surface.** A knob without
  a flag is set in the config file — **no** generic `--set`/dotted-path mechanism
  (considered and rejected: hard to read). Full list discoverable via `bp-train config
  --defaults` / `--schema` and `bp-train init`. Kept flags: `--input`, `--config`,
  `--custom`, `--process`, `--steps`, `--learning-rate`, `--output-dir`, `--resume`,
  `--overwrite`, `--no-plot`, `--log-level` (~11). Everything else is config-only.

**Run directory**
- **prepared.json:** referenced by path + stable `content_hash` in `config.json` (not
  copied; the recipient needs `prepared.json` too — see "Portability"). A future
  `bp-train bundle` can copy it in for a portable archive — out of scope here.
- **Retention:** keep **best + latest only** (the `checkpoint.keep` default) — prune every
  other checkpoint dir after each write. Also bounds plots to 2 dirs (best + most-recent).
- **Per-checkpoint plots:** keep `loss_curve.png`, `grad_norm_curve.png`, and **add**
  per-process prediction plots, rendered in a **background worker process** so training
  never blocks on matplotlib.
- **Resume:** `bp-train train --resume <run_dir>` continues in place, appending to
  `metrics.csv`; `--steps` may extend the original target.

Both halves stay **planning/spec** for now — the plan carries detailed implementation, no
code lands yet.

---

# Part 1 — Input configuration (`RunConfig`)

## The config object

Split configuration by **who owns it**, not by where it's typed:

- **`RunConfig`** — library-owned, **typed pydantic tree**, the SSOT of exposable knobs.
  Validated, defaulted, serialised to/from `config.json`.
- **`model` section** — user/case-study blob (`extra="allow"`), passed verbatim to the
  `custom.py` hooks. Free-form because keys like `target_loss_weights={"glucose": .1}` are
  inherently per-study. Its *code-side defaults* live in `custom.py` `CONFIG`; the config
  file's `model.config` overlays them.

```python
# bp_train/run_config.py  — pydantic v2, one level of nesting, shallow on purpose
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path

_FROZEN = {"frozen": True, "extra": "forbid"}          # immutable + reject unknown keys

class DataConfig(BaseModel):
    model_config = _FROZEN
    input: Path
    processes: tuple[str, ...] | None = None
    targets: tuple[str, ...] | None = None
    target_source: Literal["process_variables", "reactor_components", "auto"] = "auto"

class ModelConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    custom: Path | None = None
    seed: int = 0
    config: dict = Field(default_factory=dict)         # free-form model dict (extra allowed here)

class OptimConfig(BaseModel):
    model_config = _FROZEN
    steps: int = Field(50, gt=0)
    optimizer: Literal["adam", "sgd"] = "adam"
    learning_rate: float = Field(1e-3, gt=0)
    grad_clip_norm: float = Field(1000.0, ge=0)        # 0 disables clipping
    batch_size: int | None = Field(None, gt=0)
    shuffle: bool = True
    batch_seed: int | None = None

class SolverConfig(BaseModel):
    model_config = _FROZEN
    max_steps: int = Field(2048, gt=0)
    rtol: float = Field(1e-5, gt=0)
    atol: float = Field(1e-7, gt=0)
    jump_ts: bool = True

class CheckpointConfig(BaseModel):
    model_config = _FROZEN
    every: int = Field(10, ge=0)                        # 0 disables checkpointing
    keep: Literal["best+latest", "all"] = "best+latest"
    resume: Path | None = None

class OutputConfig(BaseModel):
    model_config = _FROZEN
    dir: Path = Path("output")
    plots: bool = True

class LoggingConfig(BaseModel):
    model_config = _FROZEN
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    every: int = Field(10, gt=0)            # console log-row cadence — distinct from checkpoint.every
    decimals: int = Field(4, ge=0)          # console table formatting
    header_every: int = Field(30, ge=0)
    # NB: the streaming metrics.csv is ALWAYS written (it is the SSOT the curves render
    # from) — it is not a toggle. The old optional --metrics-csv/--metrics-jsonl are gone.

class RunConfig(BaseModel):
    model_config = _FROZEN
    data: DataConfig
    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()
    solver: SolverConfig = SolverConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    output: OutputConfig = OutputConfig()
    logging: LoggingConfig = LoggingConfig()
```

`TrainHarnessConfig` ([harness.py:60](bp-train/bp_train/harness.py#L60)) becomes a thin
adapter built from `RunConfig` (one `from_run_config` mapping), or is replaced outright —
no parallel field lists.

> **Naming note:** pydantic reserves the attribute name `model_config` (its settings dict)
> and guards the `model_` prefix via `protected_namespaces`. Our domain `model` section /
> `ModelConfig` don't collide with the `model_` prefix, but to be safe set
> `protected_namespaces=()` on `RunConfig` (or rename) so pydantic emits no warning.

## The resolver

One function does all layering; pydantic enforces types/unknown-keys at each `model_validate`:

```python
def resolve_run_config(cli_args, *, config_file: Path | None, custom_module) -> RunConfig:
    # 1. file layer (full structured config, all sections optional → defaults fill in)
    file_dict = load_json(config_file) if config_file else {}
    # 2. model-dict layer: custom.py CONFIG < file["model"]["config"]
    file_dict.setdefault("model", {}).setdefault("config", {})
    file_dict["model"]["config"] = {**resolve_config(custom_module, None),
                                    **file_dict["model"]["config"]}
    # 3. CLI override layer: only flags the user actually passed (argparse SUPPRESS
    #    defaults so unset flags don't clobber the file), mapped flat→sections
    overrides = _cli_overrides_to_sections(cli_args)        # {"optim": {"learning_rate": ...}, ...}
    merged = _deep_merge(file_dict, overrides)
    return RunConfig.model_validate(merged)                 # ← validates, fails fast, fills defaults
```

To make the CLI-wins layer correct, register `train` flags with `default=argparse.SUPPRESS`
so an *unset* flag contributes nothing (today's defaults would otherwise always override the
file). CLI stays **flat** (`--learning-rate`, not `--optim.learning-rate`); the resolver
maps flat CLI → structured sections. "Flat to type, structured to read."

## Precedence — order of importance (explicit)

There are **two independent keyspaces**, resolved separately.

**1. Library knobs** — the typed `RunConfig` sections (data, optim, solver, checkpoint,
output, logging). Lowest → highest:

```
pydantic field defaults   <   config.json sections   <   CLI flags
```

→ **CLI flags win over config.json.** A flag passed on the command line overrides the same
key in the file; a key present only in the file overrides the built-in default; anything
unset falls back to the pydantic default. (This is why `train` flags register with
`default=argparse.SUPPRESS`.)

**2. The model dict** — `model.config`, free-form, handed verbatim to the `custom.py`
hooks. Lowest → highest:

```
custom.py CONFIG   <   config.json "model.config" section
```

→ The config file's `model.config` overrides the code-side `CONFIG` defaults. There is **no
CLI layer** here at all (these keys are free-form and can't be typed flags; the generic
`--set` mechanism was rejected). Model-dict knobs are set in `custom.py CONFIG` or the
config file's `model.config` — nowhere else.

### What happens to `custom.py`'s `CONFIG`?

It **stays, but narrows sharply** — it becomes the *code-side default layer for genuine
train-time model-hook knobs only*. The litmus test: **a `model.config` key must be read by a
`custom.py` hook; otherwise it does not belong there.**

- **Genuine model knobs stay in `CONFIG`** — `bounds_weight`, `target_loss_weights`. These
  are read by the user's own loss/reaction hooks; they are the lowest layer of keyspace #2
  and the config file's `model.config` can override them.
- **Library knobs that currently sneak into `CONFIG` move OUT** to typed sections:
  - `target_variable_order` → `data.targets`, target source → `data.target_source`. Deletes
    today's triple-overlap (CLI `--target` ⊕ `--config` JSON ⊕ `CONFIG`,
    [cli.py:674-681](bp-train/bp_train/cli.py#L674)); hooks already get `target_names` /
    `process_names` as **explicit arguments** ([harness.py:267-283](bp-train/bp_train/harness.py#L267)).
  - **prepare-phase knobs** — `strict_bp_format_validation` ([prepare.py:252](bp-train/bp_train/prepare.py#L252),
    prepare-only), `bolus_run_min_dt` ([controls.py:574](bp-train/bp_train/controls.py#L574),
    control-build + baked into `prepared.json`), plus `required_control_names`,
    `require_consistent_controls`, `initial_grid_points`, `max_rel_error`,
    `max_refinement_rounds`, `metadata_namespace`, `process_rename_map` — go to a typed
    `prepare` section (see below). These were the worst "keys from nowhere": library-owned,
    consumed deep in `controls.py`/`prepare.py`, never by a user hook, untyped, undocumented.

Net: after the change, **`CONFIG`/`model.config` holds only keys a `custom.py` hook reads** —
one concept, one home. A user learns the rule once: *typed knob → a `RunConfig` section (file
or flag); a knob my own hook reads → `model.config`.*

### Phase-scoped config & "no keys from nowhere"

Two refinements driven by the audit above (the `bolus_run_min_dt` /
`strict_bp_format_validation` investigation):

1. **Config is phase-scoped — one file describes the whole pipeline.** The two commands —
   `prepare` (Phase B) and `train` (Phase C) — read different knobs but share one config
   file and one free-form `model.config` (the prepare hooks `transform_process_collection`
   / `build_sample_acc_series` and the train hooks all read the same user dict). `RunConfig`
   gains an optional, typed **`prepare` section** (`extra="forbid"`) consumed by
   `bp-train prepare`; the `data/optim/solver/checkpoint/output/logging/model` sections are
   consumed by `bp-train train`. `prepare.output` is the prepared artifact; `data.input`
   defaults to it — so `prepare --config p.json` then `train --config p.json` chain off one
   file (see "Part 2 §0" for the command wiring + `prepared.json` provenance).

   ```python
   class PrepareConfig(BaseModel):
       model_config = {"frozen": True, "extra": "forbid"}
       input: Path                                   # raw bp_format collection JSON
       output: Path                                  # prepared.json to emit
       strict_bp_format_validation: bool = False
       bolus_run_min_dt: float | None = Field(None, gt=0, description="triangle-ramp width (h) for bolus events")
       required_control_names: tuple[str, ...] = ()
       require_consistent_controls: bool = True
       initial_grid_points: int = Field(…, gt=0)
       max_rel_error: float = Field(…, gt=0)
       max_refinement_rounds: int = Field(…, ge=0)
       metadata_namespace: str = "hybrax"
       process_rename_map: dict[str, str] = Field(default_factory=dict)
       # free-form prepare-hook knobs live in the SHARED RunConfig.model.config (read-tracked),
       # not here — PrepareConfig holds only the typed library knobs.

   # in RunConfig:  prepare: PrepareConfig | None = None     # optional: train-only runs omit it
   # DataConfig.input becomes `Path | None = None`; a validator requires it OR falls back to
   # prepare.output (train errors only if both are unset).
   ```

2. **Free-form `model.config` is read-tracked → unused keys warn loudly.** Since per-study
   keys can't be typed, wrap the free-form dict in a small **read-tracking mapping** that
   records every key the hooks access; after module construction, emit a `WARNING` for any
   key no hook read:

   ```python
   class TrackedDict(dict):           # records reads, then surfaces the unused
       def __init__(self, *a, **k): super().__init__(*a, **k); self._read = set()
       def __getitem__(self, k): self._read.add(k); return super().__getitem__(k)
       def get(self, k, d=None):  self._read.add(k); return super().get(k, d)
       def unused(self): return set(self) - self._read
   # after hooks run: if leftovers := cfg_model.unused():
   #     logger.warning("model.config keys never read by any hook: %s — typo or stale?", sorted(leftovers))
   ```

   This is the "check that params are actually used" you asked for — applied to exactly the
   keyspace that can't be statically validated. Typed sections already fail fast on unknown
   keys via `extra="forbid"`; this gives the free-form layer a loud equivalent.

## One config, four commands

All four subcommands are **views over one `RunConfig`** — which is why a single typed spine
covers the whole CLI, and lets us **delete the three overlapping config dataclasses**
(`TrainHarnessConfig`, `ForwardConfig`, `LOOConfig`) — the simplification spec's "compose,
don't copy" ([2026-05-08 plan, Part 5](bp-train/spec/2026-05-08_simplification_plan.md)).

| Command | Reads sections | Writes | New section? |
|---|---|---|---|
| `prepare` | `prepare`, `model` | `prepared.json` (+ provenance) | `prepare` |
| `train` | `data, model, optim, solver, checkpoint, output, logging` | run dir | — |
| `loo` | all train sections **+** `loo` | per-fold run dirs + aggregate | `loo` (tiny) |
| `forward` | `data`, `solver` (from the model's `config.json`), + `--model` | `forward/` outputs | **none** — pure consumer; only `--solver-max-steps` overridable |

- **`loo` = `train` + a holdout selector.** Confirmed by the code: `LOOConfig.base_train_config`
  *is* a `TrainHarnessConfig` ([loo.py:49](bp-train/bp_train/loo.py#L49)); the only
  loo-specific knob is `selected_holdouts` (`--holdouts`). So the `loo` section is tiny:

  ```python
  class LOOConfig(BaseModel):
      model_config = _FROZEN
      holdouts: tuple[str, ...] | None = None      # parent-process folds to run; None = all
      metrics: tuple[str, ...] = ("rmse", "mae", "r2")
  # in RunConfig:  loo: LOOConfig | None = None
  ```

- **`forward` needs no section.** `ForwardConfig`'s fields (process selection, targets,
  target_source, solver) are already `data` + `solver` ([harness.py:452](bp-train/bp_train/harness.py#L452));
  forward only adds "which model" — a run dir. It becomes a pure consumer via `load_run`.

## Exposable-knob inventory ("list what we want to expose")

Every knob is **always settable in `config.json`**. The right-hand column shows the *only*
knobs that also get a CLI flag (the run-to-run ones). Everything else is config-only —
discoverable via `bp-train config --defaults` / `--schema`.

| Section | Knobs (full surface, all in `config.json`) | Also a CLI flag |
|---|---|---|
| data | input, processes, targets, target_source | `--input`, `--process` |
| model | custom, seed, config (free-form) | `--custom` |
| optim | steps, learning_rate, optimizer, grad_clip_norm, batch_size, shuffle, batch_seed | `--steps`, `--learning-rate` |
| solver | max_steps, rtol, atol, jump_ts | — |
| checkpoint | every, keep, resume | `--resume` (+ `--overwrite`) |
| output | dir, plots | `--output-dir`, `--no-plot` |
| logging | level, every, decimals, header_every (metrics.csv always written) | `--log-level` |

This realises the [2026-05-08 simplification](bp-train/spec/2026-05-08_simplification_plan.md)'s
"move rarely-changed knobs to the config file" — now with a concrete, typed home for the
full surface and a deliberately small CLI.

## New-user "first model" on-ramp

1. **Zero-`custom.py` default run works and is documented.** `bp-train train --input
   prepared.json` runs `DefaultReactionModule` + `DefaultLossModule` + identity scales,
   emitting one clear line: *"no --custom: fitting a default MLP surrogate; supply --custom
   for a mechanistic model."* (Mostly wiring + docs; the defaults already exist.)
2. **A defaults-filled config — GENERATED, never committed.** The user wants to *see* every
   available option. We render it from the pydantic models on demand so it can never drift
   from the code defaults (SSOT — rule 2). A static committed `default_config.json` is
   explicitly **rejected**: it becomes a second source of truth and goes stale the moment a
   default changes. Three entry points, all derived from `RunConfig`:
   - `bp-train init` — scaffolds a starter dir: a minimal `custom.py`
     (`build_reaction_module` + `CONFIG`) **and** a `config.json` covering the **whole
     pipeline** — a `prepare` section (`input` = `"<path to raw bp_format JSON>"`,
     `output` = `"prepared.json"`) plus the train sections with defaults filled in. The
     two-step on-ramp is then `bp-train prepare --config config.json` →
     `bp-train train --config config.json`, no editing beyond the two input paths.
   - **`bp-train init --data <collection.json>` — schema-aware scaffold.** When data is
     supplied, init introspects it and writes the **RhsOde / ReactionInputs–Outputs schema**
     into the scaffolded `custom.py` as a comment block, and pre-fills `data.targets` /
     `CONFIG` from the modeled state names — so the user sees the exact `c = [acetate,
     biomass, glucose, …]`, `u = [carbon_feed, …]`, and required outputs (`q_<name>`, …)
     they must produce. **Feasibility: easy** — `format_reaction_schema(rhs_ode, controls)`
     ([inspect.py:392](bp-train/bp_train/inspect.py#L392)) already returns this exact layout
     as a string and is **model-independent** (needs only `rhs_ode` + `controls`, both built
     from the collection; `print_reaction_schema` is a thin wrapper over it). init just
     builds `rhs_ode`/`controls` from the collection and templates the string in. *Caveat:*
     the schema reflects the collection's current layout, and `prepare` can rename/add
     series — so `init --data prepared.json` gives the final layout; on raw data it shows the
     pre-transform layout as a starting point.
   - `bp-train config --defaults` — prints that same fully-defaulted config to stdout.
   - `bp-train config --schema` — prints `RunConfig.model_json_schema()`: every knob's type,
     bounds, allowed values **and `Field(description=...)`** — the "what does this mean" view
     a bare JSON of defaults can't carry. (Motivates a `description` on every knob.)
3. **Round-trip:** every finished run leaves a fully-resolved `config.json`; copying it and
   editing two fields is the fastest path to the next experiment. Same object as the
   resume/FAIR-output config — no second schema.

---

# Part 2 — FAIR pipeline outputs: `prepare`, run directory, checkpointing & resume

Both commands consume the `RunConfig` from Part 1 and emit provenance-stamped, FAIR
artifacts: `prepare` → `prepared.json` (with a provenance block, §0); `train` → the
self-contained run directory below. The two chain by stable `content_hash` (`train`'s
`config.json` records `prepared_input.content_hash`), so a model traces back through its
data prep to the raw collection.

## New run directory layout

```
<output.dir>/
├── config.json              # RunConfig.model_dump() + run-status provenance — see below
├── custom.py                # bundled copy of the user model module (FAIR + resume-stable)
├── metrics.csv              # streaming, one row per step (append) — single source for curves
├── observations.csv         # measured points per process, written once (for plot overlays)
├── model/
│   ├── params.eqx           # canonical "load this" — final/best wrapper leaves
│   └── opt_state.eqx        # final optimizer state
├── checkpoints/
│   ├── latest -> step_01500
│   ├── best   -> step_00300
│   └── step_NNNNN/          # only `best` and `latest` survive (checkpoint.keep="best+latest")
│       ├── params.eqx
│       ├── opt_state.eqx
│       ├── train_state.json       # {step, mean_loss, best_loss, timestamp}
│       ├── loss_curve.png        # bg-rendered from metrics.csv
│       ├── grad_norm_curve.png   # bg-rendered from metrics.csv
│       ├── predictions.csv       # forward sim (main process, JAX)
│       └── <process>.png         # bg-rendered from predictions.csv + observations.csv
```

`config.json` schema (= the validated input + read-only provenance):

```json
{
  "status": "running | complete | failed",
  "error": null,
  "started_at": "2026-06-03T14:57:13+0200",
  "finished_at": "2026-06-03T15:40:02+0200",
  "steps_completed": 1500,
  "best": {"step": 300, "mean_loss": 0.71},
  "final_mean_loss": 0.74,
  "cli_argv": ["bp_train", "train", "--config", "config.json"],
  "config": {},
  "inputs": {
    "prepared_input": {"path": "prepared.json", "content_hash": "sha256:ab12…"},
    "custom_py": {"bundled": "custom.py", "file_hash": "sha256:cd34…"}
  },
  "environment": {"python": "3.13.x", "bp_train": "…", "bp_format": "…",
                  "jax": "…", "optax": "…", "equinox": "…", "diffrax": "…"}
}
```

The `error` object contains `type`, `message`, and `step` when status is
`"failed"`. The `config` object contains the entire resolved `RunConfig`, with
all defaults filled. Its data/model sections own targets, seeds, and processes;
these are not duplicated at the top level. `content_hash` is the stable data
hash, while `file_hash` covers the exact `custom.py` bytes.

`load_run`/`forward`/resume parse the `config` block back into a `RunConfig`; the rest
(`status`, `error`, timestamps, `best`, `cli_argv`, `inputs`, `environment`) is read-only
provenance.

**`status="failed"` is real, not decorative.** `_handle_train` wraps the loop in
try/except: success → `complete`; exception → `failed` with `error={type, message, step}`,
then re-raise. Without this a crashed run stays `"running"` indefinitely — indistinguishable
from a live one. The re-run guard blocks only on `complete`, so `failed`/stale-`running`
dirs can be re-run or resumed.

**Portability — what a recipient needs to re-run.** The `config` block + bundled `custom.py`
fully specify *how*; the missing piece is the *data*, `prepared.json`, which is **referenced,
not bundled** (size). So shipping the run dir alone is **not** enough — the recipient also
needs `prepared.json`. `load_run` resolves it robustly: a bundled copy in the run dir if
present, else `inputs.prepared_input.path` relative to the run dir; on a `content_hash`
mismatch it errors clearly ("prepared.json data differs from the run's record"). The
deferred `bp-train bundle` copies `prepared.json` in for a fully self-contained archive.

**Stable `content_hash` (not raw-file sha).** Hashing `prepared.json`'s bytes is unstable:
the provenance block embeds a timestamp (different every prepare) and JSON key-order/float
formatting varies across machines, so bit-identical *science* yields different file hashes.
Instead `content_hash` = sha256 over the **deserialised collection re-serialised
canonically** (sorted keys, normalised float repr, **provenance block excluded**) — stable
across re-prepares and machines. `custom_py.file_hash` stays an exact-bytes sha (any code
change *should* invalidate). The integrity guard compares `content_hash`, so a re-prepared
but semantically-identical `prepared.json` is accepted.

## Implementation

### 0. `bp-train prepare` — config-driven, typed, provenance-stamped

`prepare` becomes a first-class citizen of the same config spine, not a side door. In
`_handle_prepare` ([cli.py:558](bp-train/bp_train/cli.py#L558)):

- **Resolve once:** `cfg = resolve_run_config(args, config_file=args.config, custom_module=...)`;
  use `cfg.prepare` (typed `PrepareConfig`) + the shared `cfg.model.config` (read-tracked).
  `prepare.input`/`prepare.output` come from the config (CLI `--input`/`--output` override).
- **Replace `resolved_config.get("…")` reads** in `prepare.py`/`controls.py`/
  `controls_store.py`/`defaults.py` with typed `PrepareConfig` field access — unknown
  prepare knobs now fail fast at validation instead of silently no-op'ing.
- **Stamp provenance into `prepared.json`.** Alongside the existing prep metadata, write a
  block mirroring the run-dir `config.json`: the resolved `PrepareConfig` (`model_dump`),
  the `custom.py` `file_hash`, the raw-input `content_hash`, this prepared collection's own
  `content_hash` (canonical, self-excluding — see below), package versions, and a timestamp.
  A prepared artifact then carries *how it was built*; `train`'s `config.json` records
  `prepared_input.content_hash` matching it, chaining provenance raw → prepared → model.
  **The timestamp lives only in this provenance block, which is excluded from `content_hash`**,
  so re-preparing identical data leaves the hash unchanged.
- **Re-run guard + `--overwrite`**, same as train, so re-preparing into an existing
  `prepared.json` is explicit.

Kept `prepare` CLI flags: `--input`, `--output`, `--custom`, `--config`, `--overwrite`,
`--log-level` (everything else via the `prepare` config section). The `prepare` hooks
(`transform_process_collection`, `build_sample_acc_series`) keep their signatures but
receive the read-tracked `model.config`.

### 1. `serialization.py` (new) — all (de)serialisation in one module

Matching `bp_format/serialization.py`, create **`bp_train/serialization.py`** as the single
home for everything that reads/writes model + run state. **Move** `save_model` /
`load_trained_wrapper` here from `postprocessing.py` (currently
[postprocessing.py:401-413](bp-train/bp_train/postprocessing.py#L401); ~8 importers across
`cli`, `harness`, `loo`, `checkpointing`, tests — update their imports). **Delete**
`save_model_metadata` / `load_model_metadata` ([postprocessing.py:416-429](bp-train/bp_train/postprocessing.py#L416)):
the sidecar scheme is replaced by `config.json` read/write helpers here.

`serialization.py` owns: `save_model`/`load_trained_wrapper`, the opt-state twins below,
`config.json` read/write (`RunConfig` ⊕ run-status provenance), and the
`reconstruct_run`/`load_run`/`LoadedRun` loader (see §7). The Plan agent confirmed optax
`chain(zero_nans, clip_by_global_norm, adam)` state serialises cleanly leaf-by-leaf, and
that the `like=` template must come from `optimizer.init(reconstructed_trainable_params)`:

```python
def save_opt_state(opt_state: Any, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, opt_state)

def load_opt_state(path: str | Path, *, template: Any) -> Any:
    return eqx.tree_deserialise_leaves(Path(path), like=template)
```

### 2. `postprocessing.py` — split sim from render

**Split sim from render** so a JAX-free worker can draw prediction plots.
`plot_process_simulations` currently both simulates and renders. Factor the render half into
a pure function that reads CSVs only:

```python
def render_process_plots_from_csv(
    predictions_csv: Path, observations_csv: Path, output_dir: Path,
    *, process_names, target_names, training_process_names,
) -> None:
    """numpy/pandas/matplotlib only — NO jax/bp_train heavy imports. Picklable,
    safe to call inside a 'spawn' worker."""
```

The main process keeps producing `predictions.csv` via the existing `export_predictions_csv`
([postprocessing.py:300-398](bp-train/bp_train/postprocessing.py#L300)). Add a one-time
`export_observations_csv(collection, store, path, ...)` for the measured overlays.

### 3. `plotting_worker.py` (new) — off-critical-path rendering

matplotlib import + render is the GIL-blocking cost the user wants gone. Workers do **pure
numpy/pandas/matplotlib, never JAX**, so `spawn` is safe alongside the parent's initialised
accelerator. The forward simulation stays in the main process (offloading it would require
JAX-in-subprocess = fork hazards + per-job recompile).

```python
class BackgroundPlotter:
    """Submit picklable (fn, *args) plot jobs to a single spawned worker.
    Never blocks training: if the worker is backed up, drop the job —
    the next checkpoint re-renders the cumulative curve anyway."""
    def __init__(self, max_pending: int = 2):
        ctx = mp.get_context("spawn")
        self._pool = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
        self._pending: deque[Future] = deque()
        self._max_pending = max_pending
    def submit(self, fn, *args) -> None:
        self._pending = deque(f for f in self._pending if not f.done())
        if len(self._pending) >= self._max_pending:
            return
        self._pending.append(self._pool.submit(fn, *args))
    def close(self) -> None:                 # drain in-flight before exit
        for f in self._pending: f.result()
        self._pool.shutdown()
```

Submitted jobs: `plot_loss_curve` and `plot_grad_norm_curve` (already pure, take
arrays+path) and `render_process_plots_from_csv`.

### 4. `checkpointing.py` — rewrite `CheckpointWriter`

Replace the current writer ([checkpointing.py](bp-train/bp_train/checkpointing.py)). It is
**driven entirely by `RunConfig.checkpoint`** (the pydantic `CheckpointConfig` from Part 1:
`every`, `keep`) plus the run-dir path — no second config type, no hardcoded
cadence/retention. Responsibilities: write the **lightweight resumable state synchronously**
(`params.eqx`, `opt_state.eqx`, `train_state.json`), submit plot jobs to the `BackgroundPlotter`,
and prune per `checkpoint.keep`.

```python
class CheckpointWriter:
    # cfg is RunConfig.checkpoint (every, keep); checkpoints_dir = <output.dir>/checkpoints
    def __init__(self, checkpoints_dir: Path, cfg: CheckpointConfig): ...

    def maybe_write(self, *, step, wrapper, opt_state, mean_loss, best_loss,
                    metrics_csv, observations_csv, plotter, render_predictions_fn,
                    process_names, target_names, training_process_names) -> Path | None:
        if self._cfg.every == 0 or step <= 0 or step % self._cfg.every: return None
        d = self._checkpoints_dir / f"step_{step:05d}"; d.mkdir(parents=True, exist_ok=True)
        save_model(wrapper, d / "params.eqx")            # sync, fast (serialization.py)
        save_opt_state(opt_state, d / "opt_state.eqx")   # sync, fast (serialization.py)
        (d / "train_state.json").write_text(json.dumps(
            {"step": step, "mean_loss": mean_loss, "best_loss": best_loss,
             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2))
        render_predictions_fn(d / "predictions.csv")     # JAX forward sim, main process
        plotter.submit(plot_loss_curve_from_csv, metrics_csv, d / "loss_curve.png", ...)
        plotter.submit(plot_grad_norm_curve_from_csv, metrics_csv, d / "grad_norm_curve.png")
        plotter.submit(render_process_plots_from_csv, d / "predictions.csv",
                       observations_csv, d, process_names=process_names, ...)
        self._update_symlink("latest", d)
        if best_loss >= mean_loss: self._update_symlink("best", d)
        if self._cfg.keep == "best+latest":              # "all" → never prune
            self._prune_except({"latest", "best"})
        return d
```

`_prune_except` resolves the `latest`/`best` symlink targets and `shutil.rmtree`s every
other `step_*` dir (only when `keep == "best+latest"`). (No code consumes `step_*/*.png` or
`step_*/predictions.csv` — the Plan agent confirmed `loo_metrics` reads only top-level fold
CSVs — so pruning is safe.) Plot submission is skipped entirely when `output.plots` is
false. **Cadence note:** `checkpoint.every` (snapshot) is distinct from `logging.every`
(console row) — the two were fused as `--log-every` before.

### 5. `harness.py` — thread resume + opt_state through the loop

Inject resume at **`train_collection`** (it already holds the reconstruction chain). New
params:

```python
def train_collection(..., start_step: int = 0,
                     initial_trainable_params=None, initial_optimizer_state=None):
```

- After `partition_trainable` ([harness.py:756](bp-train/bp_train/harness.py#L756)) and
  `optimizer.init` ([harness.py:768](bp-train/bp_train/harness.py#L768)), override with
  `initial_*` when provided.
- Loop range ([harness.py:939](bp-train/bp_train/harness.py#L939)) `range(cfg.steps)` →
  `range(start_step, cfg.steps)`. **Keep the full-length batch stream and index it
  absolutely** — it's deterministic from `(seed, batch_seed, steps, batch_size,
  shuffle_batches, process set)` ([harness.py:227](bp-train/bp_train/harness.py#L227)), so
  slicing reproduces ordering exactly; do *not* regenerate a shortened stream.
- On resume, **pre-seed** `loss_so_far` / `grad_norm_so_far` / `per_target_loss_so_far` from
  the existing `metrics.csv` so the curves stay continuous (Plan agent risk #5).
- Pass `opt_state` and `best_loss` into `checkpoint_writer.maybe_write`; track
  `best_loss = min(best_loss, mean_loss)` each step.
- Create/own the `BackgroundPlotter`; `close()` it in a `finally` after `run_log.finalize()`.

Add a thin resume helper alongside `train_from_collection`
([harness.py:1063](bp-train/bp_train/harness.py#L1063)) that **delegates reconstruction to
the single `serialization.load_run` path** (which itself uses `reconstruct_run` — see §7)
rather than a third hand-rolled copy:

```python
def resume_run(run_dir: Path, *, steps_override: int | None = None) -> TrainHarnessResult:
    loaded = load_run(run_dir, checkpoint="latest", load_opt_state=True)  # parses RunConfig,
    #   verifies content_hash(prepared)/file_hash(custom.py), rebuilds template, restores params+opt_state
    cfg = loaded.config
    if steps_override is not None:                       # --steps may extend the target
        cfg = cfg.model_copy(update={"optim": cfg.optim.model_copy(update={"steps": steps_override})})
    start = load_json(run_dir / "checkpoints" / "latest" / "train_state.json")["step"]
    return train_collection(..., config=from_run_config(cfg), start_step=start,
                            initial_trainable_params=partition_trainable(loaded.wrapper)[0],
                            initial_optimizer_state=loaded.opt_state)
```

**Integrity guard** lives in `reconstruct_run`: it verifies the prepared collection's
`content_hash` (canonical, provenance-excluded) and `custom.py`'s `file_hash` against
`config.json` and hard-errors on mismatch (template would silently misalign — Plan agent
risks #1/#2). The content hash means a re-prepared but bit-different-yet-equivalent
`prepared.json` still passes. Because everything flows from the parsed
`RunConfig` (recorded `data.targets`/`target_source`/`processes`, `model.seed`,
`optim.optimizer`/`grad_clip_norm`) and the **bundled** `custom.py` — never live CLI
defaults — the `trainable_params` pytree (hence the `opt_state` template) matches
byte-for-structure.

### 6. `cli.py` — run-dir assembly, `--resume`, re-run guard

In `_handle_train` ([cli.py:663](bp-train/bp_train/cli.py#L663)):

- **Resolve config first:** `cfg = resolve_run_config(args, config_file=args.config,
  custom_module=...)` (Part 1). The run dir is `cfg.output.dir`; checkpoint cadence/retention
  are `cfg.checkpoint.every`/`.keep`; nothing reads loose CLI flags past this point.
- **Run-dir init (before training):** create `cfg.output.dir/`, `model/`, `checkpoints/`;
  copy `cfg.model.custom` → `<dir>/custom.py`; write `config.json` =
  `{status:"running", cli_argv: sys.argv, config: cfg.model_dump(), inputs: {content_hash of
  prepared + file_hash of custom.py}, environment: {pkg versions}}`; write `observations.csv`.
- **Re-run guard:** if `config.json` exists with `status="complete"` and neither `--resume`
  nor `--overwrite` is set → exit with a clear message (kills the "re-run scatter / which
  run is this?" problem).
- **After training:** copy `checkpoints/best/{params,opt_state}.eqx` → `model/`; update
  `config.json` to `status="complete"` with `finished_at`, `steps_completed`, `best`,
  `final_mean_loss`.
- **Delete** the `trained_wrapper.meta.json` block ([cli.py:744-763](bp-train/bp_train/cli.py#L744)),
  the `--checkpoint-dir` flag (checkpoints always live at `<output.dir>/checkpoints`), and
  the per-checkpoint plotting/predictions wiring now living in the loop.

New flags: `--resume <run_dir>` (→ `harness.resume_run`), `--overwrite`. Retention is the
`checkpoint.keep` config knob (default `best+latest`); no dedicated flag.

`_handle_forward` ([cli.py:883](bp-train/bp_train/cli.py#L883)): read `config.json` instead
of `trained_wrapper.meta.json`; default `--model` to `<run_dir>/model/params.eqx`; resolve
`prepared_input` / bundled `custom.py` / solver settings from `config.json` (via the same
`load_run` path).

### 7. `serialization.py` — loading a trained model (single-call API)

The reconstruction chain (`estimate_all_scales` → `build_reaction_module` →
`_build_template_wrapper` → `load_trained_wrapper`) is duplicated **three times** today — in
`forward` ([harness.py:479-567](bp-train/bp_train/harness.py#L479)), in resume (§5), and by
hand in user notebooks. The run dir makes all three collapse to one call, because
`config.json` records exactly the inputs the chain needs (seed, targets, target_source,
process set) plus the **bundled `custom.py`** and the **`prepared.json` reference**. Factor
the chain into one shared helper and expose a public loader (both in `serialization.py`).

```python
# bp_train/serialization.py
@dataclass(frozen=True)
class LoadedRun:
    wrapper: HybridOdeWrapper
    collection: BioProcessCollection
    store: TrainingDataStore
    config: RunConfig
    run_dir: Path
    opt_state: Any | None = None          # only when load_opt_state=True (for resume)
    def reload(self, checkpoint: str = "latest") -> HybridOdeWrapper:
        """Refresh just the weights from another checkpoint, reusing this run's
        already-built wrapper as the template — no dataset/custom.py reload."""
        return load_params(self.run_dir, into=self.wrapper, checkpoint=checkpoint)

def reconstruct_run(run_dir: Path, cfg: RunConfig) -> tuple[reaction_module, loss_module, store, collection]:
    """THE single reconstruction path — forward, resume, and load_run all call this.
    Verifies prepared content_hash + custom.py file_hash against config.json first."""
    prepared = _resolve_prepared(run_dir)          # bundled copy if present, else recorded path
    collection = load_process_collection_json(prepared)
    _check_content_hash(collection, run_dir)       # canonical hash, provenance-excluded
    custom = load_custom_module(run_dir / "custom.py")                     # bundled copy
    store = TrainingDataStore.from_collection(collection,
                target_variable_order=cfg.data.targets, target_source=cfg.data.target_source)
    scale_kwargs = _resolve_estimated_scales(custom, collection, store, cfg.model.config)
    reaction = _build_reaction_module(store, cfg, custom, collection, scale_kwargs)
    loss = _build_loss_module(store, cfg, custom, collection)
    return reaction, loss, store, collection

def load_run(run_dir, *, checkpoint: str = "best", load_opt_state: bool = False) -> LoadedRun:
    """Reconstruct a trained model from a run directory ALONE.

    `checkpoint`: "best" | "latest" | "step_00300" → resolves under checkpoints/;
    the run-root model/ is used when checkpoint="final" (the default copy of best).
    """
    run_dir = Path(run_dir)
    cfg, _ = read_run_config_json(run_dir / "config.json")
    reaction, loss, store, collection = reconstruct_run(run_dir, cfg)
    template, template_opt = _build_template_wrapper(store, reaction_module=reaction,
                                 loss_module=loss, collection=collection,
                                 selected_processes=tuple(store.process_order))
    params_path = (run_dir / "model" / "params.eqx" if checkpoint == "final"
                   else run_dir / "checkpoints" / checkpoint / "params.eqx")
    wrapper = load_trained_wrapper(params_path, template=template)
    opt_state = (load_opt_state(params_path.with_name("opt_state.eqx"), template=template_opt)
                 if load_opt_state else None)
    return LoadedRun(wrapper, collection, store, cfg, run_dir, opt_state)

# ---- lightweight: refresh weights into an EXISTING wrapper (no dataset/custom.py reload) ----

def checkpoint_params_path(run_dir, checkpoint: str = "latest") -> Path:
    run_dir = Path(run_dir)
    return (run_dir / "model" / "params.eqx" if checkpoint == "final"
            else run_dir / "checkpoints" / checkpoint / "params.eqx")

def load_params(run_dir, *, into: HybridOdeWrapper, checkpoint: str = "latest") -> HybridOdeWrapper:
    """Deserialise a checkpoint's leaves into an ALREADY-BUILT wrapper. Pure
    eqx.tree_deserialise_leaves(params.eqx, like=into) — no prepared.json, no custom.py,
    no reconstruct_run. `into` must be structurally identical to the trained wrapper
    (same architecture/process set); eqx errors on a pytree mismatch."""
    return load_trained_wrapper(checkpoint_params_path(run_dir, checkpoint), template=into)
```

**Two loaders, two costs:**
- `load_run(run_dir)` — *heavy, cold start*: rebuilds the template from `prepared.json` +
  `custom.py` (use when you have nothing in memory yet, or for resume).
- `load_params(run_dir, into=wrapper)` / `LoadedRun.reload()` — *lightweight*: you already
  hold a structurally-matching wrapper, so it only reads the small `params.eqx` and swaps
  leaves in. Ideal for "re-load the latest checkpoint into my existing model" in a notebook
  loop: `load_run` once, then `run.reload("latest")` (or `load_params`) as training
  progresses — no 17 MB dataset re-read.

Exposed as `bp_train.load_run` + `bp_train.load_params` (top-level `__init__` exports) for
notebook ergonomics. The user's 20-line snippet becomes:

```python
run = bp_train.load_run(EXAMPLE_DIR / "output_some")          # or checkpoint="latest"/"step_00300"
wrapper, collection, store = run.wrapper, run.collection, run.store
```

`forward` and resume are refactored to call `reconstruct_run` + `load_run` internally,
deleting their bespoke copies (delete > deprecate). **Note:** `load_run` targets the *new*
run-dir layout (`config.json` + bundled `custom.py` + `model/params.eqx`); existing
old-layout outputs (`trained_wrapper.eqx` + `.meta.json`) are re-generated by one fresh
training run — no back-compat shim.

### 8. `loo.py` + `forward` — same FAIR artifacts

**loo** orchestrates N `train` runs (one per holdout fold,
[loo.py:226](bp-train/bp_train/loo.py#L226)). Each fold is a **complete FAIR run dir**, so
all of Part 2 (checkpointing, plots, resume, `load_run`) applies per fold:

```
<output.dir>/
├── config.json          # RunConfig (incl. loo section) + provenance; status tracks fold completion
├── custom.py            # bundled once
├── folds/<holdout>/     # each a full run dir (params, opt_state, metrics.csv, plots, config.json)
└── loo_metrics.csv      # aggregated cross-fold metrics
```

- **Resume = skip complete folds.** `--resume <loo_dir>` reruns only folds whose
  `folds/<h>/config.json` status ≠ `"complete"` — natural fault tolerance for cluster runs
  (a single `--holdouts <h>` still selects one fold, as today).
- **Loading:** `load_run(<loo_dir>/folds/<h>)` loads one fold; add `load_loo(<loo_dir>)`
  returning all fold `LoadedRun`s + the aggregate. `loo_metrics.py` reads each fold's
  `config.json` ([loo_metrics.py:254-263](bp-train/bp_train/loo_metrics.py#L254)) instead of
  the deleted `*.meta.json`.

**forward** is a pure consumer: `bp-train forward --model <run_dir>` (or a fold dir) →
`load_run` + simulate + write a `forward/` subdir. `ForwardConfig` is deleted (its fields
are `data` + `solver`). **Crucially, forward uses the model's *recorded* solver config** —
`rtol`/`atol`/`jump_ts` come from the run dir's `config.json` and are **not** CLI-overridable,
so a forward pass reproduces the trajectory the model was fit under. The *only* permitted
solver override is `--solver-max-steps`: it's a safety cap (raising it can't change a solve
that already completes within budget — it just lets a longer/denser pass finish), not an
accuracy knob. To genuinely change `rtol`/`atol`/`jump_ts` you **edit `config.json` in the
`--model` dir** — that is redefining the model's evaluation, done explicitly in one place.
This deletes the simplification spec's footgun where `--solver-*` meant "setting" on
`train`/`loo` but "override" on `forward`. Forward CLI shrinks to `--model`, `--process`,
`--output-dir`, `--solver-max-steps`, `--no-plot`, `--log-level` (drop
`--solver-rtol`/`--solver-atol`/`--no-jump-ts`/`--target`/`--target-source`).

---

## Files touched

**Configuration**
- **new:** `bp_train/run_config.py` — pydantic models (`RunConfig` + sections +
  `PrepareConfig` + `LOOConfig`), `resolve_run_config`, and the `TrackedDict` for
  read-tracked free-form `model.config`. **Deletes** the overlapping `TrainHarnessConfig` /
  `ForwardConfig` / `LOOConfig` dataclasses (harness.py / loo.py), replaced by `RunConfig`
  sections + thin `from_run_config` adapters.
- **new:** `bp_train/cli.py` `init` subcommand (+ `--data` schema-aware mode reusing
  `inspect.format_reaction_schema` and a small collection→`(rhs_ode, controls)` builder
  factored out of the harness) + scaffold templates; `config` subcommand (`--defaults` dumps
  `RunConfig` defaults, `--schema` dumps `model_json_schema()`) — all generated from the
  models/data, nothing committed
- `bp_train/cli.py` — `_handle_train` builds `RunConfig` via the resolver; `_handle_prepare`
  builds `PrepareConfig` from the `prepare` section; `--config` now carries the full
  structured config (its `model`/`prepare` subsections replace the old free-form `--config`
  overlay); delete the scattered target/target_source precedence blocks
- `bp_train/prepare.py` / `bp_train/controls.py` / `bp_train/controls_store.py` /
  `bp_train/defaults.py` — read **typed `PrepareConfig` fields** (`strict_bp_format_validation`,
  `bolus_run_min_dt`, grid-refinement knobs, …) instead of `resolved_config.get("…")`;
  prepare records the resolved `PrepareConfig` into `prepared.json` provenance
- `bp_train/harness.py` — derive/replace `TrainHarnessConfig` from `RunConfig`; wrap
  `model.config` in `TrackedDict` and warn on unused keys after hooks build
- `bp_train/utils.py` — `resolve_config` folds into the resolver's model-dict layer
- examples `custom.py` — `CONFIG` shrinks to genuine model-hook knobs (`bounds_weight`,
  `target_loss_weights`); prepare/library knobs move to the `prepare` section of a committed
  `config.json` per example (delete > deprecate the `--steps … --lr …` shell args)

**Run directory / checkpointing**
- **new:** `bp_train/serialization.py` (matches `bp_format/serialization.py`) — owns
  `save_model`/`load_trained_wrapper` (moved from `postprocessing.py`),
  `save_opt_state`/`load_opt_state`, `config.json` read/write,
  `reconstruct_run`/`load_run`/`load_loo`/`LoadedRun` (+ `LoadedRun.reload`), the lightweight
  `load_params`/`checkpoint_params_path` (dataset-free weight refresh into an existing
  wrapper), a canonical `content_hash(collection)` (sorted keys, normalised floats,
  provenance-excluded) + `_resolve_prepared`/`_check_content_hash`, and a `file_hash(path)`;
  exported as `bp_train.load_run` + `bp_train.load_params`. The old
  `save_model_metadata`/`load_model_metadata` sidecar helpers are deleted.
- **new:** `bp_train/plotting_worker.py` — `BackgroundPlotter`
- `bp_train/checkpointing.py` — rewrite `CheckpointWriter` (lightweight sync state + bg
  plot jobs; retention driven by `RunConfig.checkpoint.keep`)
- `bp_train/postprocessing.py` — `export_observations_csv`, `render_process_plots_from_csv`
  + `*_from_csv` curve plotters; `save_model`/`load_trained_wrapper`/`*_model_metadata`
  **leave** for `serialization.py`
- `bp_train/harness.py` — `start_step`/`initial_*` params, absolute-indexed loop,
  metrics.csv pre-seed, opt_state + best tracking into checkpoints, `BackgroundPlotter`
  lifecycle, `resume_run`
- `bp_train/loo.py` / `loo_metrics.py` — consume `RunConfig.loo` (delete `LOOConfig`
  dataclass); per-fold FAIR run dirs under `folds/`, top-level `loo_metrics.csv`; resume
  skips complete folds; add `load_loo`; read each fold's `config.json`
- `bp_train/cli.py` / `bp_train/harness.py` — `forward` becomes a `load_run` consumer;
  delete `ForwardConfig`; `_handle_forward` reads the run dir's `config.json` for solver
  settings; drop `--solver-rtol`/`--solver-atol`/`--no-jump-ts`/`--target`/`--target-source`
  on forward (keep only `--solver-max-steps` as a safety-cap override)
- `spec/v1-detailed-spec.md` — move resume out of non-goals; document the run-dir layout
- `tests/test_checkpointing.py` — update asserts to `{params.eqx, opt_state.eqx,
  train_state.json, *.png, predictions.csv}` + best/latest pruning

**Dependency**
- `pyproject.toml` `dependencies`: add `pydantic>=2` (already present in `bench13` as
  2.13.3). No YAML dep (deferred).

## Verification

**Configuration**
1. **Round-trip:** `RunConfig.model_validate(rc.model_dump()) == rc`; a finished run's
   `config.json` re-run with `--config config.json` reproduces `metrics.csv` bit-identically.
2. **Precedence:** CLI `--learning-rate` beats `config.json` `optim.learning_rate` beats the
   pydantic default; and `model.config` section beats `custom.py CONFIG`.
3. **Fail-fast + use-tracking:** unknown key in a typed section (incl. `prepare`) raises
   with the offending path; a free-form `model.config` key that no hook reads emits the
   `TrackedDict` "never read — typo or stale?" warning (assert the warning fires).
4. **On-ramp + pipeline:** `bp-train init` then `prepare --config config.json` →
   `train --config config.json` runs end-to-end editing only the two input paths;
   `bp-train train --input <fixture>` (no custom) trains the default MLP with the warning.
5. **Prepare provenance + stable hash:** `prepared.json` carries a provenance block
   (resolved `PrepareConfig`, custom.py `file_hash`, raw-input + own `content_hash`,
   versions); train's `config.json` `prepared_input.content_hash` matches it. **Re-prepare
   the same data → identical `content_hash`** (timestamp-only byte differences don't change
   it), so the integrity guard still passes; assert raw → prepared → model chains; unknown
   `prepare` knob fails fast.

**Run directory / checkpointing**
5. **Round-trip serialise:** `optimizer.init(params)` → `save_opt_state` → reconstruct
   template → `load_opt_state` → assert `eqx.tree_equal`.
6. **Resume == continuous:** train N steps; resume to 2N; assert the 2N loss curve is
   bit-identical to a single 2N run *for the resumed segment*. Smoke-run on a
   `vibrio_slim/03_train_models/04_fba_hyb` config with small `--steps`.
7. **Retention:** after a multi-checkpoint run, assert exactly 2 `step_*` dirs survive
   (`best`, `latest`) when `keep="best+latest"`; `model/params.eqx` matches `best`.
8. **FAIR reload + `load_run`:** with `prepared.json` alongside, `bp_train.load_run(<dir>)`
   reproduces the forward predictions bit-identically; `checkpoint="best"` vs `"latest"`
   resolve the right dirs; a re-prepared (timestamp-different) `prepared.json` still loads
   (stable `content_hash`); a *content*-changed `prepared.json` or an edited bundled
   `custom.py` trips the guard; a missing `prepared.json` gives a clear error.
9b. **Lightweight reload:** `load_params(run_dir, into=wrapper)` (and `LoadedRun.reload()`)
    refresh weights from `latest` **without** reading `prepared.json`/`custom.py` (assert no
    collection load happens); a structurally-mismatched `into` raises a clear eqx error.
9. **Non-blocking plots:** per-step `dt` in `metrics.csv` unchanged vs a no-plot run;
   `BackgroundPlotter.close()` drains pending PNGs at exit.
10. **loo + forward:** a 2-fold loo run produces two FAIR `folds/<h>/` run dirs +
    `loo_metrics.csv`; killing it mid-second-fold and `--resume <loo_dir>` reruns only the
    incomplete fold; `load_loo` returns both folds + aggregate; `bp-train forward --model
    <fold_dir>` simulates via `load_run` using the model's recorded `rtol`/`atol`/`jump_ts`
    (assert there is no CLI path to override them; `--solver-max-steps` is accepted and does
    not change a within-budget trajectory).
11. **`init --data`:** `bp-train init --data <fixture>` writes a `custom.py` whose schema
    comment lists the collection's actual state/control/rate names (matches
    `format_reaction_schema`) and pre-fills `data.targets`; the scaffold trains as-is.
12. **Full suite:** `pytest -n auto` green (with updated `test_checkpointing.py`).

---

## Consistency checks (run-dir layer ↔ config)

The run-dir/checkpointing layer consumes `RunConfig` directly — these seams are closed:

1. **One `CheckpointConfig`.** `CheckpointWriter` is driven by the pydantic
   `RunConfig.checkpoint` (`every`, `keep`) + run-dir path — no duplicate writer-local
   config type.
2. **Retention is config-driven**, honoring `checkpoint.keep` (`best+latest` default, or
   `all`) — not a hardcoded prune.
3. **Cadence split.** `checkpoint.every` (snapshot) ≠ `logging.every` (console row) — the
   two were fused as `--log-every`.
4. **`--checkpoint-dir` deleted.** Checkpoints always live at `<output.dir>/checkpoints`.
5. **`config.json` = `RunConfig.model_dump()` + provenance**, written via the resolver — not
   `dataclasses.asdict(TrainHarnessConfig)` / `vars(args)`; no duplicated top-level
   `targets`/`target_source`/`processes` (they live under `config.data`/`config.model`).
6. **Single reconstruction path.** `resume_run` and `forward` delegate to
   `serialization.load_run` → `reconstruct_run`; `--steps` maps to `optim.steps` via
   `RunConfig.model_copy`.
7. **`metrics.csv` is core, not a toggle** (the curves' SSOT); the old
   `--metrics-csv`/`--metrics-jsonl` knobs are removed.
8. **Plot gating** via `output.plots`.
9. **Serialisation home.** opt-state + model save/load live in `serialization.py`, not
   `postprocessing.py`.
