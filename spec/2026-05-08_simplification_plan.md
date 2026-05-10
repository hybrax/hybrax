# Simplification Audit: `bp-format` (recap) + `bp-train` (deep dive)

## Context

The user is mid-sweep. `bp-format/bp_format/*` was just simplified (Apr 27 –
May 8 2026). `bp-train/` is **pre-rewrite** — many tests fail. Goal:

1. Capture the design principles applied to `bp-format` so the same pattern
   carries to `bp-train`.
2. Audit `bp-train` for: CLI surface, duplicate functions, core functions,
   harness mechanism, and CLI-argument bloat.
3. Form an opinion on harness-vs-ABC.

No code changes proposed in this plan — it is documentation + memory only.

---

## Part 1 — Design Principles from the `bp-format` Sweep

Eight principles, each visible in surviving code (citations preserved
from prior plan version).

1. **Delete > deprecate.** No back-compat shims. `RateDecl` wrapper gone;
   `integrate_process` + `build_q_func` + `build_rates_func` +
   `estimate_specific_rates` deleted; equivalence verified once off-repo
   (8 fixtures × 20 probes, max diff 1 ULP) then the old path is gone.
2. **Single source of truth for layout.**
   [`ProcessOrdering`](bp-format/bp_format/dataclasses.py) owns every
   canonical name layout; every factory takes the same ordering object.
3. **Naming: full words, role prefixes, distinct concepts get distinct
   names.** `mb` → `rhs_ode`. `derived` → `algebraic` (collided with
   `TimeSeries.derived`). `q_<rmc>` for RMC rates, `r_<pv>` for PV rates.
4. **Minimum attributes / arguments.** Drop single-field wrappers
   (`RateDecl` → `Bounds`). Drop derivable fields (`n_derived` → use
   `len(name_modeled_algebraic)`). Drop fields rendered dead by upstream
   changes (`biomass_idx`, `is_intracellular`, `intracellular_indices`,
   `control_names`, `flow_indices`, …).
5. **Auto-generate sensible defaults; users opt out by supplying their
   own.** `BioProcess.__post_init__` builds `biological_ode` from the
   reactor medium when not user-supplied; raises if `"biomass"` missing.
6. **Single-path API.** No auto/user dispatch branching. `get_rhs_ode`
   always returns one type; `rates_func` takes a single flat
   `jnp.ndarray` aligned with `rate_names` (was `(q, r)` tuple).
7. **Fail-fast over silent fallbacks.**
   [`_require_reactor_volume_above_threshold`](bp-format/bp_format/mechanistic.py#L82-L93)
   uses `eqx.error_if` instead of an epsilon floor; loaders reject
   already-transformed `c*` carriers; `inspect` helpers fail on malformed
   dynamic values.
8. **Vectorize at the JIT boundary.** Static tuples for names; one flat
   array per lambdified expression; `_batch_splines` stacks per-piece
   coefficients into one PPoly (17× win in commit `5a1e8f3`).

---

## Part 2 — `bp-train` CLI Concepts

`bp-train` has one entry point, four subcommands (single argparse parser,
[cli.py:40-43](bp-train/bp_train/cli.py#L40-L43)).

### Pipeline overview

```
raw collection JSON ──► prepare ──► prepared.json ──► train ──► trained_wrapper.eqx
                                                  │              + .meta.json (sidecar)
                                                  ├─► loo (= train per parent process group)
                                                  └─► forward (= load model + simulate, no training)
```

### Per-command summary

| Command | Purpose | Inputs | Outputs |
|---|---|---|---|
| `prepare` | Apply case-study transforms (renames, smoothing, sample-acc series) to raw collection | raw JSON, `--custom`, `--case-study` | prepared JSON |
| `train` | Fit a `HybridOdeWrapper` to selected targets via diffrax+optax | prepared JSON, `--custom` | `trained_wrapper.eqx`, `*.meta.json`, plots, loss tables |
| `forward` | Reload trained model, run forward ODE, regenerate plots | `--model` (+ optional `--input` from sidecar) | plots, losses CSV, optional dense `--timeseries-csv` |
| `loo` | Per-fold `train` over LOO process groups + aggregate | prepared JSON, `--custom` | per-fold dirs (model + plots), aggregated metrics |

`forward` already demonstrates the principle the rest can follow: **read
canonical context from the sidecar, accept `--<flag>` only as override**.

---

## Part 3 — CLI-Argument Audit

`cli.py` is **1,099 lines**; ~50 unique flags across the four
subcommands. Counts come from
[cli.py:_build_parser](bp-train/bp_train/cli.py#L40).

### Flags grouped by reduction strategy

| Strategy | Flags | Currently on | Recommendation |
|---|---|---|---|
| **Pull to global parent parser** | `--log-level`, `--config`, `--custom` | all four commands | `argparse` parent; zero behavior change |
| **Pull to global parent parser** | `--solver-max-steps`, `--solver-rtol`, `--solver-atol`, `--no-jump-ts` | `train`, `loo`, `forward` (as overrides) | parent parser; `forward` overrides remain semantically the same |
| **Move to config file (rarely used in examples)** | `--batch-size`, `--batch-seed`, `--shuffle-batches/--no-shuffle-batches` | `train`, `loo` | only `13_volume_integration` uses any of these |
| **Move to config file** | `--metrics-csv`, `--metrics-jsonl`, `--log-process-losses`, `--log-decimals`, `--log-header-every` | `train`, `loo` | telemetry knobs, **never appear in any example shell script** |
| **Drop one half of pair** | `--plot/--no-plot` | `train`, `loo`, `forward` | keep only `--no-plot` (default true → suffices) |
| **Drop entirely (rarely used)** | `--loss-csv` (forward), `--target` + `--target-source` overrides on `forward` | `forward` | sidecar already has them; overrides are defensive |

### Estimated reduction

~50 flags → ~25 flags (≈50%) without losing power-user capability, by:

- one parent parser for `--log-level`/`--config`/`--custom`/`--solver-*`,
- moving telemetry/batch knobs into the JSON `--config` file,
- dropping one half of every `--X / --no-X` pair (keep the non-default one).

### Footguns to flag

- `--solver-*` are **settings** on `train`/`loo` but **overrides** on
  `forward` — same flag, different semantics. Resolve via parent parser
  + explicit "override" doc on `forward`.
- `--target` / `--target-source` are merged from CLI ⊕ JSON config ⊕
  custom-module `CONFIG` ⊕ sidecar with implicit precedence
  ([cli.py:674-681](bp-train/bp_train/cli.py#L674-L681), and parallel
  blocks at 945, 1026). Document the precedence order in `--help`.

---

## Part 4 — Harness Inventory & Opinion

### Mechanism

- Loader: [`load_custom_module`](bp-train/bp_train/utils.py#L10-L25)
  uses `importlib.util.spec_from_file_location` to load `custom.py` as
  module name `bp_train_user_custom`.
- Hook lookup: [`get_hook(module, name, default)`](bp-train/bp_train/utils.py#L49-L52)
  is a one-line `getattr(module, name, default)`.
- Config merge: [`resolve_config`](bp-train/bp_train/utils.py#L28-L46)
  prefers `module.get_config()` over `module.CONFIG`, then overlays
  `--config` JSON.

### The complete hook list

Defaults live in [defaults.py](bp-train/bp_train/defaults.py).

| # | Hook name | Where called from | Default | Real-world override frequency |
|---|---|---|---|---|
| 1 | `CONFIG` (or `get_config()`) | [utils.py:28-46](bp-train/bp_train/utils.py#L28-L46) | `{}` | **Always** |
| 2 | `transform_process_collection(collection, config)` | [prepare.py:252-256](bp-train/bp_train/prepare.py#L252-L256) | `default_transform_process_collection` (rename only) | ~50% of examples |
| 3 | `build_sample_acc_series(process, name, meta, config)` | [prepare.py:257-261](bp-train/bp_train/prepare.py#L257-L261) | `default_build_sample_acc_series` | rare |
| 4 | `build_reaction_module(*, target_names, process_names, config, seed, collection)` | [harness.py:267-283](bp-train/bp_train/harness.py#L267-L283) | `DefaultReactionModule` (small MLP, ignores controls) | **Always** |
| 5 | `build_learning_rate(config, train_cfg)` | [harness.py:1187-1190](bp-train/bp_train/harness.py#L1187-L1190) | scalar from `--learning-rate` | when LR schedule wanted |
| 6 | `build_sample_loss_fn(default_fn, store, collection, train_cfg, config)` | [harness.py:323-343](bp-train/bp_train/harness.py#L323-L343) | `measurement_loss_from_arrays` | rare (e.g. `11_tub_2025` adds non-neg penalty) |
| 7 | `build_batched_loss_fn(default_fn, store, collection, train_cfg, config)` | [harness.py:324-362](bp-train/bp_train/harness.py#L324-L362) | default batched MSE; **mutually exclusive with #6** | rare |
| 8 | `estimate_all_scales(collection, target_names, config)` → `(state, controls, q, f)` | [harness.py:558-571, 1201-1214](bp-train/bp_train/harness.py#L558-L571) | **None → no scaling (silent identity)** | **Almost always** |

### Friction observed

- Hooks split across two files (`prepare.py` + `harness.py`) with no
  single registry — users have to grep to find what's overridable.
- Typos in hook names are silent: `getattr(module, "build_reaciton_module", default)`
  returns the default and runs untrained nonsense.
- Hook contracts inconsistent: some take keyword-only (`build_reaction_module`),
  some positional, some return `(callable, tuple[str])` tuples
  (`build_sample_loss_fn`) and some return bare callables.
- Mutual exclusivity between hooks #6 and #7 is enforced at runtime only.
- "No scaling" (hook #8 absent) silently degrades training — no warning.

### Harness vs ABC — my recommendation

**Neither pure-pattern is right. Use a frozen `Harness` dataclass loaded
through a strict factory.** Trade-off summary:

| Property | Status quo (`getattr` lookup) | Pure ABC (subclass-or-die) | **Recommended hybrid** |
|---|---|---|---|
| Discoverability (IDE / `--help-hooks`) | poor | good | **good** |
| Typo safety | none | excellent | **excellent** |
| Optional hooks | trivial | awkward (`@abstractmethod` → forces override) | **trivial** |
| Single source of truth for the hook contract | scattered | the ABC | **the dataclass fields** |
| Friction for the user writing `custom.py` | very low | high (boilerplate class) | **very low — still flat functions** |
| Multi-module composition / mix-in | impossible | natural | **possible via field override** |

**Sketch:**

```python
# bp_train/harness_spec.py
@dataclass(frozen=True)
class Harness:
    config: dict
    transform_process_collection: Callable = default_transform_process_collection
    build_sample_acc_series: Callable      = default_build_sample_acc_series
    build_reaction_module: Callable        = default_build_reaction_module
    build_learning_rate: Callable | None   = None
    build_sample_loss_fn: Callable | None  = None
    build_batched_loss_fn: Callable | None = None
    estimate_all_scales: Callable | None   = None

    @classmethod
    def from_module(cls, module, cli_config):
        if module is None:
            return cls(config=cli_config or {})
        known = {f.name for f in fields(cls)} | {"CONFIG", "get_config"}
        unknown = [n for n in vars(module) if not n.startswith("_") and callable(getattr(module, n)) and n not in known]
        if unknown:
            raise ValueError(f"custom.py defines unknown hook(s): {unknown}. "
                             f"Known hooks: {sorted(known - {'CONFIG','get_config'})}")
        # mutual-exclusivity check at construction, not at runtime
        if hasattr(module, "build_sample_loss_fn") and hasattr(module, "build_batched_loss_fn"):
            raise ValueError("Define either build_sample_loss_fn or build_batched_loss_fn, not both.")
        return cls(
            config=resolve_config(module, cli_config),
            **{f.name: getattr(module, f.name) for f in fields(cls) if hasattr(module, f.name) and f.name != "config"},
        )
```

What this buys (matched to `bp-format`'s eight principles):

- **Single source of truth** (principle 2): the dataclass *is* the hook
  contract.
- **Fail-fast over silent fallback** (principle 7): unknown hook names
  raise; mutual exclusivity checked at construction; absent
  `estimate_all_scales` could log a one-line warning ("training without
  feature scales — set `estimate_all_scales` for stability").
- **Single-path API** (principle 6): `harness.build_reaction_module(...)`
  is the only call site — no `getattr` scattered across `prepare.py` +
  `harness.py`.
- **Minimum attributes** (principle 4): no `Harness` class hierarchy,
  no abstract methods, no registration.
- The user's `custom.py` stays a flat module of functions. **No syntax
  burden moved onto users.**

A pure ABC would force more boilerplate on users for the smallest
benefit (typo safety) that the dataclass-factory already gives.

---

## Part 5 — Duplicate / Near-Duplicate Functions

| Group | Members | Action |
|---|---|---|
| **Metrics** split across modules | [`_mse_and_r2()` postprocessing.py:324](bp-train/bp_train/postprocessing.py#L324) vs [`_r2/_mae/_rmse` loo_metrics.py:51-66](bp-train/bp_train/loo_metrics.py#L51-L66) | Consolidate into one `metrics.py`; have postprocessing use the same registry as loo_metrics |
| **Diffrax solver setup** twice | [`_solve_trajectory()` trainer.py:66-87](bp-train/bp_train/trainer.py#L66-L87) and [trainer.py:113-133](bp-train/bp_train/trainer.py#L113-L133) | Extract one `_make_solver(saveat)` factory; the two callers pass different `saveat` |
| **Model save/load boilerplate** | `save_model` + `save_model_metadata` called in pairs at cli.py, loo.py, checkpointing.py | Wrap to single `save_trained_model(wrapper, meta, dir)` in postprocessing.py |
| **Trivial bp_format re-wrap** | [`_load_collection()` loo_metrics.py:934](bp-train/bp_train/loo_metrics.py#L934) | Inline `load_process_collection_json` directly; remove the helper |
| **Three config dataclasses with overlapping fields** | `TrainHarnessConfig`, `ForwardConfig`, `LOOConfig` | `ForwardConfig` doesn't reuse the solver fields from `TrainHarnessConfig`. Compose, don't copy |
| **Validation wrappers** | `validate_collection` / `validate_raw_collection` ([validation.py:17-48](bp-train/bp_train/validation.py#L17-L48)) both loop `bp_format.validate_process` | Already thin — keep, they add a batch interface |
| **`_serialize_concentration` duplicated** | [controls.py:162-172](bp-train/bp_train/controls.py#L162-L172) and [validation.py:51-61](bp-train/bp_train/validation.py#L51-L61) — semantically identical (controls.py wraps `np.asarray(x, dtype=float)` as `_as_numpy`, validation.py inlines it) | Keep the controls.py version; have validation.py import it. Or hoist into a shared `_serialization.py` if more such helpers appear. |
| **NaN-policy divergence in LOO stats** | [loo.py:570-584](bp-train/bp_train/loo.py#L570-L584) (`_mean/_std/_median` strip NaNs) vs [loo_metrics.py:514-525](bp-train/bp_train/loo_metrics.py#L514-L525) (`_safe_mean/_safe_std/_safe_median` don't strip) | Same statistics, opposite NaN handling. Decide one policy, lift into the consolidated `metrics.py` already proposed in the row above for `_mse_and_r2` / `_r2/_mae/_rmse`, and delete the loser. |

---

## Part 6 — Core Functions to Make More Robust

Functions called from many sites that carry the package's complexity.

### A. `train_collection()` — [harness.py:673-780](bp-train/bp_train/harness.py#L673-L780)

- Call sites: `train_from_collection` (~harness.py:1165), `run_loo_fold`
  ([loo.py:250](bp-train/bp_train/loo.py#L250)).
- Problem: 8 keyword-only args, including 4 optional scaling arrays
  (`state_scale`, `controls_scale`, `q_scale`, `f_scale`) each with
  None-checks at lines 700-720.
- Fix: **Make scales always-required.** Caller computes them once
  (or uses `IDENTITY_SCALES` constant); inside `train_collection`, no
  None-checks. This mirrors bp-format principle 7 (fail-fast).

### B. Loss-function composition — [harness.py:307-360](bp-train/bp_train/harness.py#L307-L360)

- The "which loss?" decision branches on: `target_source` string flag,
  presence of custom hook #6, presence of custom hook #7, mutual
  exclusivity, extra-loss-names tuple.
- Fix: extract a `loss_registry.py` with one entry point
  `resolve_loss(target_source, harness)` that returns
  `(batched_loss_fn, extra_loss_names)`. The harness check + mutual
  exclusivity moves to `Harness.from_module` (Part 4).

### C. `run_loo_fold()` — [loo.py:171-350](bp-train/bp_train/loo.py#L171-L350)

- 12 keyword-only parameters; every fold rebuilds the same context.
- Fix: a frozen `FoldContext` dataclass (`collection`, `fold_groups`,
  `selected_folds`, `base_train_cfg`) passed once.

### D. `HybridOdeWrapper.__call__` — [wrapper.py](bp-train/bp_train/wrapper.py)

- ~20 call sites. **Already clean.** Don't touch.

### E. `TrainingDataStore` — [training_data.py](bp-train/bp_train/training_data.py)

- One central data accessor; `from_collection` constructor, name-keyed
  access. **Already clean.** Don't touch.

---

## Part 7 — Named Return Types for Harness Hooks

### Motivation

Friction in Part 4 boils down to one root cause: **hook outputs are
shapeless**. Bare `Callable` and ad-hoc `(callable, tuple[str])` tuples
mean the user can't tell from the signature what to return, and mypy
can't catch wrong returns. Mutual exclusivity between
`build_sample_loss_fn` and `build_batched_loss_fn` exists only because
neither return type carries the distinction.

This part covers the **highest-impact** half of the four leverages
discussed in conversation: structured outputs. Per-hook input
dataclasses (`BuildReactionModuleInput`, …), `typing.Protocol`s, and
the `bp-train init-custom` stub generator are useful follow-ups but
out of scope here — name-the-output first, then layer the rest.

### Why this is not promoting hooks to ABCs

Named return types constrain the **value** a hook produces; ABCs
constrain the **shape of user code**. A `LossSpec` return type works
identically whether the hook is a flat function in `custom.py`, a
method on a class, or a lambda. The user's `custom.py` stays a flat
module of functions — no subclassing, no `self`, no
`@abstractmethod`. `typing.Protocol` (next iteration, not this part)
provides the type-checking half of what an ABC would give without
forcing class structure.

### New module: `bp_train/spec/types.py`

One file. ~150 lines. Contains every hook output type. Imported by
`bp_train.harness`, `bp_train.defaults`, and every example
`custom.py`.

### The new return types

```python
# bp_train/spec/types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import jax

# ---- Loss

SampleLossFn  = Callable[..., jax.Array]   # tighten to real signature when extracting
BatchedLossFn = Callable[..., jax.Array]

@dataclass(frozen=True)
class LossSpec:
    """Return value of `build_loss` (replaces the two old hooks).

    Set exactly one of `sample_fn` or `batched_fn`. `extra_loss_names`
    are appended as additional panels in CSV/plot output; their values
    must be returned as part of the loss dict from the chosen function.
    """
    sample_fn:  SampleLossFn  | None = None
    batched_fn: BatchedLossFn | None = None
    extra_loss_names: tuple[str, ...] = ()

    def __post_init__(self):
        if (self.sample_fn is None) == (self.batched_fn is None):
            raise ValueError(
                "LossSpec: set exactly one of sample_fn or batched_fn."
            )

# ---- Scales

@dataclass(frozen=True)
class Scales:
    """Return value of `estimate_scales`.

    Every field required — pass `Scales.identity(...)` for "no scaling".
    Eliminates the silent-identity fallback noted in Part 4.
    """
    state:    jax.Array
    controls: jax.Array
    q:        jax.Array
    f:        jax.Array

    @classmethod
    def identity(cls, *, n_state: int, n_controls: int,
                 n_q: int, n_f: int) -> "Scales":
        import jax.numpy as jnp
        return cls(state=jnp.ones(n_state),    controls=jnp.ones(n_controls),
                   q=jnp.ones(n_q),            f=jnp.ones(n_f))

# ---- Reaction module + learning rate

# `build_reaction_module` already returns the named type
# `UserReactionModule` (model_api.py) — no change.

# `build_learning_rate` returns `float | optax.Schedule` — already
# discriminated by union; no wrapper needed.
```

### Hook-by-hook impact

| Hook | Before | After | File / line of old call site |
|---|---|---|---|
| `build_sample_loss_fn` + `build_batched_loss_fn` | two hooks, return bare callable or `(callable, tuple[str])`; mutual exclusivity at runtime | **one hook** `build_loss` returns `LossSpec`; mutual exclusivity in `LossSpec.__post_init__` | [harness.py:307-360](bp-train/bp_train/harness.py#L307-L360) |
| `estimate_all_scales` | returns `(state, controls, q, f)` tuple OR is absent → silent identity | returns `Scales`; absence triggers a `Scales.identity(...)` default with one-line warning | [harness.py:558-571](bp-train/bp_train/harness.py#L558-L571), [harness.py:1201-1214](bp-train/bp_train/harness.py#L1201-L1214) |
| `build_reaction_module` | returns `UserReactionModule` (already named) | unchanged | [harness.py:267-283](bp-train/bp_train/harness.py#L267-L283) |
| `build_learning_rate` | returns `float \| optax.Schedule` (union, already discriminated) | unchanged | [harness.py:1187-1190](bp-train/bp_train/harness.py#L1187-L1190) |
| `transform_process_collection` | returns `BioProcessCollection` (already named) | unchanged | [prepare.py:252-256](bp-train/bp_train/prepare.py#L252-L256) |
| `build_sample_acc_series` | returns `ControlSource` (already named) | unchanged | [prepare.py:257-261](bp-train/bp_train/prepare.py#L257-L261) |

Net hook count: **8 → 7** (loss merge). Six of seven outputs are now
named types; the seventh (`build_learning_rate`) is a discriminated
union which is fine.

### Migration

1. **Add the module.** Write `bp_train/spec/types.py` with `LossSpec`
   and `Scales`. Export from `bp_train.__init__` for examples.
2. **Replace `build_*_loss_fn` plumbing in `harness.py`.** Single
   resolver: read `harness.build_loss`, get a `LossSpec`, branch on
   `spec.sample_fn is not None` exactly once. Delete the runtime
   mutual-exclusivity check at [harness.py:326](bp-train/bp_train/harness.py#L326).
3. **Replace `estimate_all_scales` plumbing.** Caller computes
   `Scales.identity(...)` once when the hook is absent, logs a
   one-line warning. Delete `None`-checks at the four scale fields
   inside `train_collection` ([harness.py:700-720](bp-train/bp_train/harness.py#L700-L720)).
4. **Update `defaults.py`.** No default `build_loss` (current code
   has none either; `measurement_loss_from_arrays` is wired
   directly). Add `default_estimate_scales` returning
   `Scales.identity(...)` so absence and explicit-identity become the
   same code path (fail-fast: principle 7).
5. **Migrate the 5 example `custom.py` files** that override hooks 6,
   7, or 8. Mechanical: wrap returns in the new types.
   - `examples/01_kittler_2022/vanilla/custom.py` — `estimate_all_scales` only
   - `examples/03_martens_2025_expanded/structured/custom.py` — likely scales
   - `examples/11_tub_2025/custom.py` — both loss + scales (the
     non-negativity penalty case from Part 4)
   - `examples/13_volume_integration/custom.py` — neither (no migration)
   - any other that exists under `examples/` overriding these hooks
6. **Delete the runtime mutual-exclusivity validation** and the
   `(callable, tuple[str])`-vs-bare-callable adapter inside
   `harness.py`'s loss resolver.

### Worked example — TUB custom.py before/after

Before (today):

```python
def build_sample_loss_fn(default_sample_loss_fn, store, collection,
                         train_cfg, config):
    def sample_loss(...): ...
    extra_loss_names = tuple(f"nonneg/{t}" for t in config["target_variable_order"])
    return sample_loss, extra_loss_names   # bare tuple
```

After:

```python
from bp_train.spec.types import LossSpec

def build_loss(default_sample_loss_fn, store, collection,
               train_cfg, config) -> LossSpec:
    def sample_loss(...): ...
    return LossSpec(
        sample_fn=sample_loss,
        extra_loss_names=tuple(f"nonneg/{t}" for t in config["target_variable_order"]),
    )
```

Same code, named return, IDE-discoverable shape, mypy can verify.

### Risks / open questions

- **`Scales.identity` ergonomics.** Caller must know `n_state`,
  `n_controls`, `n_q`, `n_f`. These are derivable from the harness
  context but currently scattered. May want a `Scales.identity_from(ctx)`
  convenience after Part 4's narrow-input dataclasses land.
- **Example breakage.** Migrating examples is a breaking change for
  any out-of-repo user `custom.py`. Per principle 1 (delete >
  deprecate): take the hit, update the README, no shim.
- **`SampleLossFn` / `BatchedLossFn` aliases.** Currently `Callable[..., jax.Array]`.
  Tighten to the real argument shape when extracting `loss_registry.py`
  (Part 6.B) — that change is gated on the loss-resolver refactor and
  doesn't block this part.
- **Should `LossSpec` carry the loss name itself** (i.e. the
  "primary" loss panel name, not just `extra_loss_names`)? Current
  code defaults to "loss"; revisit when consolidating CSV column
  conventions in Part 5.

### Verification

- `cd bp-train && /home/mgotsmy/anaconda3/envs/bench13/bin/python -m pytest -q tests/test_harness.py tests/test_loo.py`
  — same coverage, new return-type assertions added.
- Run each migrated `examples/*/run.sh`; diff produced loss-curve
  PNGs and `losses.csv` against pre-migration outputs. Numerical
  equivalence required (no behavioral change, only typing).
- `bp-train --help` and per-subcommand help unchanged — this part
  doesn't touch CLI surface.
