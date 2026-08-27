"""Typed, validated run configuration for ``prepare``/``train``/``loo``/``forward``.

``RunConfig`` (via :func:`load_run_config` and its per-command wrappers) is
the config for ``prepare``/``train``/``loo``; ``ForwardRunConfig`` (via
:func:`load_forward_config`) is the separate, simpler config for ``forward``.
Every section is its own pydantic model — see each class's own docstring for
its fields.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hybrax.format.json_io import load_json
from hybrax.train.utils import load_custom_module

_FROZEN = ConfigDict(extra="forbid", frozen=True)
_ALLOWED_TOP_LEVEL = {
    "data",
    "custom_py",
    "train",
    "solver",
    "checkpoint",
    "output",
    "logging",
    "prepare",
    "custom",
    "loo",
}
_TRAIN_SECTIONS = {
    "data",
    "train",
    "solver",
    "checkpoint",
    "output",
    "logging",
    "custom_py",
    "custom",
}
_COMMAND_SECTIONS = {
    "prepare": {"prepare", "custom_py", "custom", "output"},
    "train": _TRAIN_SECTIONS,
    # loo reuses every train section (each fold is a train run) plus its own
    # `loo` block (holdout sets + fold-level parallelism).
    "loo": _TRAIN_SECTIONS | {"loo"},
}
_Command = Literal["prepare", "train", "loo"]
InitialValueSource = Literal["measured", "spline", "augmented"]
PredictionScope = Literal["none", "parents", "all"]


class ConfigBase(BaseModel):
    """Shared pydantic base for every config section.

    ``model_config`` forbids unknown keys (``extra="forbid"``), so a typo like
    ``"epocs": 300`` is a hard validation error rather than being silently
    dropped — the single most useful piece of the config system's strictness.
    It also freezes every instance (``frozen=True``): a config object handed
    to a hook is never mutated out from under the caller, and changing a
    field means ``model_copy(update=...)``. Carries no fields of its own.
    """

    model_config = _FROZEN


class DataConfig(ConfigBase):
    """Where a training/loo run reads its prepared dataset from.

    ``prepared`` (required) points at the prepared-data artifact — either the
    directory ``hybrax prepare`` wrote, or a ``prepared.json``/
    ``prepared.json.gz`` file directly (see :func:`resolve_prepared_path`);
    resolved relative to the config file's directory like every other path
    here. ``processes`` (optional) restricts training/evaluation to this
    subset of process names; omitted means every process in the prepared
    dataset. ``targets`` (optional) restricts the training targets to this
    explicit list of measured-quantity names, overriding automatic derivation
    from ``target_source``. ``target_source`` decides *which* measurements
    the loss is computed against: ``"reactor_components"`` (medium components
    only), ``"process_variables"`` (process variables only), ``"combined"``
    (both), or ``"auto"`` (default; decide from what the dataset actually
    has). Set it explicitly once you have modeled process variables —
    ``"auto"`` is a convenience, not a decision you want made implicitly on a
    dataset you care about.
    """

    prepared: Path
    processes: tuple[str, ...] | None = None
    targets: tuple[str, ...] | None = None
    target_source: Literal[
        "process_variables",
        "reactor_components",
        "combined",
        "auto",
    ] = "auto"


class TrainConfig(ConfigBase):
    """Optimizer, batching, and device settings for one training run.

    ``epochs`` (default 5) is how many passes over the selected process set
    to run; ``--epochs`` on the CLI overrides this, since it is the one knob
    meant to change constantly while iterating. ``seed`` is the base random
    seed, handed to the ``build_reaction_module``/``build_loss_module`` hooks
    for parameter initialization and, unless ``batch_seed`` is set, to the
    batch-shuffling RNG too. ``optimizer`` selects the optax base transform
    (``"adam"`` or ``"sgd"``); ``learning_rate`` is its base rate (a custom
    ``build_learning_rate`` hook can turn it into a schedule instead of a
    constant). ``grad_clip_norm`` (default 1000, effectively off) clips the
    raw, pre-optimizer gradient by global norm before every update — check
    ``grad_norm_curve.png``, which plots the pre-clip norm, to pick a real
    value once scales are right. ``batch_size`` (default ``None``) is how
    many processes make up one batch; ``None`` means full-batch. ``shuffle``
    (default ``True``) reshuffles the process order every epoch;
    ``batch_seed`` (optional) seeds that shuffling independently of ``seed``.
    ``devices`` (default ``1``) is how many CPU devices to shard the batch
    across; ``"max"`` resolves to ``min(n_processes, n_cpus)`` — never every
    core, since idle surplus devices can deadlock the ``pmap`` all-reduce —
    and the ``HYBRAX_TRAIN_DEVICES`` environment variable always overrides
    it. ``allow_stateful_models`` (default ``False``) is the required opt-in
    for reaction modules with a nonzero latent state: training raises before
    it starts without it, because a latent state changes what the model
    *is*, not just its size.
    """

    epochs: int = Field(5, gt=0)
    seed: int = 0
    optimizer: Literal["adam", "sgd"] = "adam"
    learning_rate: float = Field(1e-3, gt=0)
    grad_clip_norm: float = Field(1000.0, ge=0)
    batch_size: int | None = Field(None, gt=0)
    shuffle: bool = True
    batch_seed: int | None = None
    devices: int | Literal["max"] = 1
    allow_stateful_models: bool = False


class SolverConfig(ConfigBase):
    """diffrax ODE solver settings for every forward and backward solve.

    ``max_steps`` (default 2048) is the step budget for the whole solve — the
    first knob to raise when solves start bailing out (a bail is not fatal;
    points after it are just masked out of the loss, but a run where most
    samples bail is fitting almost nothing). ``rtol``/``atol`` (defaults
    ``1e-5``/``1e-7``) are diffrax's adaptive-step relative/absolute
    tolerances. ``jump_ts`` (default ``True``) tells the step-size
    controller about the process's own known vector-field discontinuity
    times (from ``BioProcess.discrete_events``, e.g. discrete control
    steps), so it anticipates them instead of discovering them by trial and
    error; turning it off makes the controller behave like a plain
    ``PIDController(rtol, atol)``. This is independent of bolus/sample state
    jumps, which are always applied through a separate event mechanism
    regardless of this flag.
    """

    max_steps: int = Field(2048, gt=0)
    rtol: float = Field(1e-5, gt=0)
    atol: float = Field(1e-7, gt=0)
    jump_ts: bool = True


class CheckpointConfig(ConfigBase):
    """Checkpoint cadence for ``train`` and ``loo``.

    ``every`` (optional, in epochs) is how often a checkpoint is written.
    ``None`` (default) selects an automatic
    cadence: at least every 5 epochs, at most 20 checkpoints over the whole
    run. ``0`` disables periodic checkpointing, but the final checkpoint at
    the end of training is always written regardless. Must be finite.

    ``bundle_prepared`` (default ``True``) includes prepared data so every
    checkpoint is self-contained; disable it to reduce repeated disk use.

    Checkpointing re-exports predictions and re-writes the bundled data, so
    on a fast run it can dominate the wall clock — set it coarse enough
    that it is not the bottleneck.
    """

    every: float | None = Field(None, ge=0)
    bundle_prepared: bool = True

    @field_validator("every")
    @classmethod
    def _validate_every(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("checkpoint.every must be finite")
        return value


class OutputConfig(ConfigBase):
    """Where a ``train``/``loo`` run writes its output, and what it exports.

    ``dir`` (default ``"output"``) is the run directory everything lands in
    — config, logs, checkpoints, ``losses.csv``, and (if requested)
    predictions — resolved relative to the config file's directory unless
    overridden by ``--output-dir``. ``predictions`` (default ``"none"``)
    controls whether a dense ``predictions.csv`` is exported: ``"none"``
    writes nothing, ``"parents"`` writes every evaluated non-augmented
    process, and ``"all"`` also includes synthetic augmentation children.
    """

    dir: Path = Path("output")
    predictions: PredictionScope = "none"


class LoggingConfig(ConfigBase):
    """Console output formatting for ``train`` and ``loo``.

    ``decimals`` (default 4) is the number of decimal places used when
    formatting loss and gradient-norm values in the interactive console log.
    It does not affect ``metrics.csv``/``metrics.jsonl``, which always
    record full precision regardless of this setting.
    """

    decimals: int = Field(4, ge=0)


class AugmentationConfig(ConfigBase):
    """Synthetic sibling-process generation for ``prepare.augmentation``.

    Generates ``n_children_per_process`` synthetic children per real
    (parent) process. ``seed`` seeds the augmentation RNG independently of
    ``train.seed``. ``n_time_points`` (>= 2) is how many points each child
    is resampled onto, from the parent's fitted spline.
    ``min_spacing_fraction`` (default 0.1, in ``(0, 1]``) sets a floor on
    the spacing between consecutive points in that resampled grid, as a
    fraction of the uniform spacing; the remaining duration is distributed
    as random jitter between the interior points, and ``1.0`` forces
    exactly uniform spacing. ``noise_std`` (required, non-empty) maps a
    modeled state name to the standard deviation of the Gaussian noise
    added to its resampled trajectory; only states named here are
    perturbed, every other state is copied through unchanged, and every
    name must be a modeled (not controlled) state with a fitted spline.
    ``initial_value_source`` (default ``"measured"``) decides how each
    noised state's t=0 value is set: ``"measured"`` pins it to the parent's
    actual t=0 measurement (raising if there is none), ``"spline"`` pins it
    to the parent's fitted spline value at t=0, and ``"augmented"`` leaves
    it noised like every other point; it can be one value applied to every
    state in ``noise_std``, or a per-state dict whose keys must exactly
    match ``noise_std``'s.
    """

    seed: int = 0
    n_children_per_process: int = Field(gt=0)
    n_time_points: int = Field(ge=2)
    min_spacing_fraction: float = Field(0.1, gt=0.0, le=1.0, strict=True)
    noise_std: dict[str, float] = Field(min_length=1)
    initial_value_source: InitialValueSource | dict[str, InitialValueSource] = (
        "measured"
    )

    @field_validator("noise_std")
    @classmethod
    def _validate_noise_std(cls, value: dict[str, float]) -> dict[str, float]:
        invalid = [
            name
            for name, noise_std in value.items()
            if not math.isfinite(noise_std) or noise_std < 0.0
        ]
        if invalid:
            raise ValueError(
                "noise_std values must be finite and nonnegative: " + ", ".join(invalid)
            )
        return value

    @model_validator(mode="after")
    def _validate_initial_value_source(self) -> AugmentationConfig:
        if not isinstance(self.initial_value_source, dict):
            return self
        configured = set(self.noise_std)
        specified = set(self.initial_value_source)
        if configured != specified:
            raise ValueError(
                "initial_value_source keys must match noise_std; "
                f"missing={sorted(configured - specified)}, "
                f"unexpected={sorted(specified - configured)}"
            )
        return self


class PrepareConfig(ConfigBase):
    """Dataset-preparation settings: the ``prepare`` command's own config section.

    ``raw_input`` (required) is a ``BioProcessCollection``, as a file or
    directory. ``augmentation`` (optional) generates synthetic sibling
    processes; see :class:`AugmentationConfig`. ``strict_format_validation``
    (default ``False``) decides whether ``hybrax.format`` validation
    failures stop the run or are reported and tolerated — set it ``True``
    for a dataset you intend to publish. ``required_control_names``
    (default empty) fails prepare early if a named control is missing: a
    flat tuple applies the same requirement to every process, a dict maps
    process name to its own required list. ``require_consistent_controls``
    (default ``True``) additionally requires every process to classify its
    controls into the same inflow/outflow/process-variable layout; disable
    it to allow processes with heterogeneous control layouts in the same
    prepare run. ``process_rename_map`` (default empty) renames processes
    by key, applied by the default ``transform_process_collection`` hook —
    a custom hook of that name in ``custom.py`` overrides this default
    entirely. ``diagnostics`` (default ``True``) emits per-process control
    diagnostic plots (raw data vs. the stored control spline) into
    ``<output-dir>/prepare_diagnostics/`` at the end of prepare.
    """

    raw_input: Path
    augmentation: AugmentationConfig | None = None
    strict_format_validation: bool = False
    required_control_names: tuple[str, ...] | dict[str, tuple[str, ...]] = ()
    require_consistent_controls: bool = True
    process_rename_map: dict[str, str] = Field(default_factory=dict)
    diagnostics: bool = True


class HoldoutSet(ConfigBase):
    """One LOO fold.

    ``test`` is the held-out process set; ``train`` (optional) pins the exact
    processes to train on. ``train=None`` means "every process not in ``test``"
    (augmentation-corrected: holding out any member of an augmentation group
    excludes the whole group — parent + all children — from train; see
    :func:`hybrax.train.loo.resolve_folds`). ``name`` (optional) labels the fold's
    output directory and summary row; without it the directory is derived from
    the test process names.
    """

    name: str | None = None
    test: tuple[str, ...] = Field(min_length=1)
    train: tuple[str, ...] | None = None


class LooConfig(ConfigBase):
    """Leave-one/some-process-out cross-validation settings.

    ``per_fold_holdout_sets=None`` runs classic leave-one-out: one fold per
    parent process group. ``parallel_folds`` is how many folds train at once
    (each fold is its own subprocess, because the JAX CPU device count is fixed
    per process). ``devices_per_fold`` can set each fold's JAX CPU device
    count; otherwise the available JAX CPU device budget is split across
    concurrent folds so that ``parallel_folds * devices_per_fold <= n_cpu``.
    Set these from what your RAM can hold — there is deliberately no automatic
    RAM sizing.
    """

    per_fold_holdout_sets: tuple[HoldoutSet, ...] | None = None
    parallel_folds: int = Field(1, gt=0)
    devices_per_fold: int | None = Field(None, gt=0)

    @field_validator("parallel_folds", "devices_per_fold", mode="before")
    @classmethod
    def _reject_bool(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("loo parallel/device counts must be int >= 1")
        return value


class RunConfig(ConfigBase):
    """Top-level config object: assembles every ``prepare``/``train``/``loo``
    section (``data``, ``custom_py``, ``train``, ``solver``, ``checkpoint``,
    ``output``, ``logging``, ``prepare``, ``custom``, ``loo``) — see each
    section's own docstring for its fields. A given command only reads and
    validates the sections it actually uses.
    """

    data: DataConfig | None = None
    custom_py: Path | None = None
    train: TrainConfig = Field(default_factory=TrainConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    prepare: PrepareConfig | None = None
    custom: Any | None = None
    loo: LooConfig | None = None


class DefaultCustomConfig(BaseModel):
    """Permissive fallback wrapper for the top-level ``custom`` config block.

    Used automatically when ``custom.py`` does not define
    ``get_custom_config``: every key under ``custom`` in the config JSON is
    accepted as-is (``extra="allow"``) rather than validated against a
    schema, so hook code reads it via attribute access
    (``config.custom.hidden_width``) with no fields declared here. Instances
    are frozen. ``ser_json_inf_nan="constants"`` lets a custom field holding
    ``inf``/``-inf``/``nan`` round-trip through ``config.json`` instead of
    silently becoming ``null``.
    """

    model_config = ConfigDict(extra="allow", frozen=True, ser_json_inf_nan="constants")


# --- forward config: a list of self-contained model dirs + optional data override ---
class ForwardDataConfig(ConfigBase):
    """Where a ``forward`` run reads its dataset from, and how it differs
    from training's ``DataConfig``.

    ``prepared`` (optional, unlike training's required
    ``DataConfig.prepared``) overrides which prepared dataset to evaluate
    against; with a single model, the prepared data bundled inside that
    model's own run/checkpoint directory is used automatically, so this is
    required only when averaging more than one model or predicting on data
    different from what the model was trained on. ``processes`` (optional)
    restricts evaluation to this subset of process names.
    """

    prepared: Path | None = None
    processes: tuple[str, ...] | None = None


class ModelRef(ConfigBase):
    """One ``models`` entry: a self-contained run/checkpoint dir. ``name`` defaults
    to the basename of the bundle's ``config.output.dir`` (resolved later)."""

    name: str | None = None
    path: Path


class ForwardOutputConfig(ConfigBase):
    """Output settings for ``forward``.

    ``dir`` (optional) is the directory forward writes directly into —
    ``losses.csv``, ``models/``, and optionally ``predictions.csv`` /
    ``plots/`` all land there, with no intermediate subdirectory; defaults to
    ``<first model>/forward``. ``predictions`` (default ``"none"``) mirrors
    ``OutputConfig.predictions`` and controls whether ``predictions.csv`` is
    written. ``plots`` (default ``False``) additionally renders one figure
    per process into ``plots/<process>.png``, best-effort (a rendering
    failure is logged, not raised); it requires ``predictions`` to be
    ``"parents"`` or ``"all"`` — a validator rejects ``plots=True`` with
    ``predictions="none"``.
    """

    dir: Path | None = None
    predictions: PredictionScope = "none"
    plots: bool = False

    @model_validator(mode="after")
    def _plots_require_predictions(self):
        if self.plots and self.predictions == "none":
            raise ValueError(
                "output.plots requires output.predictions to be parents or all"
            )
        return self


class ForwardRunConfig(ConfigBase):
    """Top-level config for ``forward``: the models to evaluate, an optional
    data override, and output settings.

    ``models`` (required, non-empty) is the list of self-contained run or
    checkpoint directories to evaluate — more than one turns the run into
    an ensemble (mean predictions plus a standard-deviation export); see
    :class:`ModelRef`. A bare path string in the list is coerced into
    ``{"path": ...}`` before validation, so the minimal
    ``{"models": ["run"]}`` works directly. ``data`` (optional); see
    :class:`ForwardDataConfig`. ``output`` defaults to every field at its
    own default; see :class:`ForwardOutputConfig`.
    """

    models: tuple[ModelRef, ...] = Field(min_length=1)
    data: ForwardDataConfig | None = None
    output: ForwardOutputConfig = Field(default_factory=ForwardOutputConfig)

    @field_validator("models", mode="before")
    @classmethod
    def _coerce_models(cls, value: Any) -> Any:
        # accept bare path strings or {name, path} objects, interchangeably
        if not isinstance(value, (list, tuple)):
            return value
        return [{"path": m} if isinstance(m, (str, Path)) else m for m in value]


def load_forward_config(config_path: str | Path) -> ForwardRunConfig:
    """Load + validate a forward_config.json, resolving model paths and the
    optional ``data.prepared`` relative to the config file's directory."""
    path = Path(config_path)
    raw = _read_raw_config(path)
    base_dir = path.parent.resolve()
    cfg = ForwardRunConfig.model_validate(raw)
    updates: dict[str, Any] = {
        "models": tuple(
            m.model_copy(update={"path": _resolve_path(m.path, base_dir=base_dir)})
            for m in cfg.models
        )
    }
    if cfg.data is not None and cfg.data.prepared is not None:
        updates["data"] = cfg.data.model_copy(
            update={
                "prepared": resolve_prepared_path(
                    _resolve_path(cfg.data.prepared, base_dir=base_dir)
                )
            }
        )
    output_updates: dict[str, Any] = {}
    if cfg.output.dir is not None:
        output_updates["dir"] = _resolve_path(cfg.output.dir, base_dir=base_dir)
    if output_updates:
        updates["output"] = cfg.output.model_copy(update=output_updates)
    return cfg.model_copy(update=updates)


@dataclass(frozen=True)
class LoadedRunConfig:
    """A validated :class:`RunConfig` plus its loaded ``custom.py`` (if any)."""

    config: RunConfig
    custom_module: ModuleType | None
    custom_py_sha256: str | None


def load_prepare_config(config_path: str | Path) -> LoadedRunConfig:
    """Load and validate a config for the ``prepare`` command; see
    :func:`load_run_config`.
    """
    return load_run_config(config_path, command="prepare")


def load_train_config(config_path: str | Path) -> LoadedRunConfig:
    """Load and validate a config for the ``train`` command; see
    :func:`load_run_config`.
    """
    return load_run_config(config_path, command="train")


def load_loo_config(config_path: str | Path) -> LoadedRunConfig:
    """Load and validate a config for the ``loo`` command; see
    :func:`load_run_config`.
    """
    return load_run_config(config_path, command="loo")


def load_run_config(config_path: str | Path, *, command: _Command) -> LoadedRunConfig:
    """Load, validate, and resolve a run config JSON for one command.

    Restricts the raw JSON to the sections ``command`` actually uses, validates
    it against :class:`RunConfig`, resolves every path (``custom_py``,
    ``data.prepared``, ``prepare.raw_input``, ``output.dir``) relative to the
    config file's directory, loads ``custom_py`` if set, and resolves the
    ``custom`` section through its hook (or :class:`DefaultCustomConfig`).

    Args:
        config_path: Path to the run config JSON file.
        command: Which command is loading the config; determines which
            top-level sections are read and which are required.

    Returns:
        The resolved config, loaded custom module (if any), and its SHA-256.

    Raises:
        ValueError: If the JSON has an unknown top-level key, or a section
            required by ``command`` is missing.
    """
    path = Path(config_path)
    raw = _read_raw_config(path)
    raw_custom = raw.get("custom")
    _validate_top_level(raw)
    _validate_raw_custom(raw_custom)

    base_dir = path.parent.resolve()
    view = _command_view(raw, command=command)
    config = RunConfig.model_validate(view)
    _validate_required_sections(config, command=command)
    config = _resolve_config_paths(
        config,
        base_dir=base_dir,
        resolve_output_symlinks=command != "loo",
    )

    custom_module, custom_py_sha256 = _load_custom_module_and_hash(config.custom_py)
    resolved_custom = _resolve_custom(raw_custom, config, custom_module)
    config = config.model_copy(update={"custom": resolved_custom})
    return LoadedRunConfig(
        config=config,
        custom_module=custom_module,
        custom_py_sha256=custom_py_sha256,
    )


def _read_raw_config(path: Path) -> dict[str, Any]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise TypeError("run config must be a JSON object")
    return raw


def _validate_top_level(raw: dict[str, Any]) -> None:
    unknown = sorted(set(raw) - _ALLOWED_TOP_LEVEL)
    if unknown:
        keys = ", ".join(unknown)
        raise ValueError(f"unknown top-level config key(s): {keys}")


def _validate_raw_custom(raw_custom: Any) -> None:
    if raw_custom is not None and not isinstance(raw_custom, dict):
        raise TypeError("config custom section must be a JSON object or null")


def _command_view(
    raw: dict[str, Any],
    *,
    command: _Command,
) -> dict[str, Any]:
    sections = _COMMAND_SECTIONS[command]
    view = {key: value for key, value in raw.items() if key in sections}
    view["custom"] = None
    return view


def _resolve_config_paths(
    config: RunConfig,
    *,
    base_dir: Path,
    resolve_output_symlinks: bool,
) -> RunConfig:
    updates: dict[str, Any] = {}
    if config.custom_py is not None:
        updates["custom_py"] = _resolve_path(config.custom_py, base_dir=base_dir)
    if config.data is not None:
        updates["data"] = config.data.model_copy(
            update={
                "prepared": resolve_prepared_path(
                    _resolve_path(config.data.prepared, base_dir=base_dir)
                )
            }
        )
    if config.prepare is not None:
        updates["prepare"] = config.prepare.model_copy(
            update={
                "raw_input": _resolve_path(
                    config.prepare.raw_input,
                    base_dir=base_dir,
                )
            }
        )
    if resolve_output_symlinks:
        output_dir = _resolve_path(config.output.dir, base_dir=base_dir)
    else:
        output_dir = config.output.dir
        if not output_dir.is_absolute():
            output_dir = base_dir / output_dir
        output_dir = Path(os.path.abspath(output_dir))
    updates["output"] = config.output.model_copy(update={"dir": output_dir})
    if not updates:
        return config
    return config.model_copy(update=updates)


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def resolve_prepared_path(path: Path) -> Path:
    """Resolve a prepared-data reference to the prepared.json file.

    ``hybrax prepare`` writes its output into a directory
    (``<dir>/prepared.json``). Accept either that directory (resolve the bundled
    ``prepared.json[.gz]`` inside) or a plain prepared.json file, so ``train`` /
    ``forward`` / ``loo`` can point at the prepare output-dir directly.
    """
    path = Path(path)
    if path.is_dir():
        gz = path / "prepared.json.gz"
        return gz if gz.is_file() else path / "prepared.json"
    return path


def _validate_required_sections(config: RunConfig, *, command: _Command) -> None:
    if command == "prepare" and config.prepare is None:
        raise ValueError("prepare command requires a prepare config section")
    if command in ("train", "loo") and config.data is None:
        raise ValueError(f"{command} command requires a data config section")


def _load_custom_module_and_hash(
    custom_py: Path | None,
) -> tuple[ModuleType | None, str | None]:
    if custom_py is None:
        return None, None
    custom_bytes = custom_py.read_bytes()
    custom_module = load_custom_module(custom_py)
    return custom_module, hashlib.sha256(custom_bytes).hexdigest()


def _resolve_custom(
    raw_custom: dict[str, Any] | None,
    config: RunConfig,
    custom_module: ModuleType | None,
) -> Any | None:
    if custom_module is not None and hasattr(custom_module, "get_custom_config"):
        return custom_module.get_custom_config(raw_custom, config)
    if raw_custom is None:
        return None
    return DefaultCustomConfig.model_validate(raw_custom)


def reresolve_custom(config: RunConfig, custom_module: ModuleType | None) -> RunConfig:
    """Re-resolve ``config.custom`` (a raw dict loaded from a run's config.json)
    into the typed object the custom hooks expect.

    A fresh run wraps the raw ``custom`` section via ``get_custom_config`` (or
    ``DefaultCustomConfig``) so hooks can use attribute access
    (``config.custom.target_loss_weights``). When a run is reconstructed from
    config.json (resume / model_load / forward), ``custom`` comes back as a plain
    dict; this restores the same typed wrapper so reconstruction matches a fresh
    run. No-op if ``custom`` is already resolved (not a dict) or absent.
    """
    if not isinstance(config.custom, dict):
        return config
    resolved = _resolve_custom(config.custom, config, custom_module)
    return config.model_copy(update={"custom": resolved})
