from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from bp_train.utils import load_custom_module

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
    "prepare": {"prepare", "custom_py", "custom"},
    "train": _TRAIN_SECTIONS,
    # loo reuses every train section (each fold is a train run) plus its own
    # `loo` block (holdout sets + fold-level parallelism).
    "loo": _TRAIN_SECTIONS | {"loo"},
}
_Command = Literal["prepare", "train", "loo"]


class ConfigBase(BaseModel):
    model_config = _FROZEN


class DataConfig(ConfigBase):
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
    steps: int = Field(50, gt=0)
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
    max_steps: int = Field(2048, gt=0)
    rtol: float = Field(1e-5, gt=0)
    atol: float = Field(1e-7, gt=0)
    jump_ts: bool = True


class CheckpointConfig(ConfigBase):
    every: int = Field(100, ge=0)  # 0 disables; DISTINCT from logging.every
    keep: Literal["best+latest", "all"] = "all"


class OutputConfig(ConfigBase):
    dir: Path = Path("output")
    plots: bool = True


class LoggingConfig(ConfigBase):
    every: int = Field(100, gt=0)
    decimals: int = Field(4, ge=0)
    # Re-emit the console table header every N rows so the column labels (loss +
    # per-target names) stay visible on a long scroll; 0 disables.
    header_every: int = Field(10, ge=0)


class PrepareConfig(ConfigBase):
    raw_input: Path
    strict_bp_format_validation: bool = False
    required_control_names: tuple[str, ...] | dict[str, tuple[str, ...]] = ()
    require_consistent_controls: bool = True
    initial_grid_points: int = Field(16, gt=0)
    max_rel_error: float = Field(1e-4, gt=0)
    max_refinement_rounds: int = Field(8, ge=0)
    process_rename_map: dict[str, str] = Field(default_factory=dict)
    # Emit per-process control diagnostic plots (raw data vs stored control spline) into
    # ``<output-dir>/prepare_diagnostics/`` at the end of prepare.
    diagnostics: bool = True


class HoldoutSet(ConfigBase):
    """One LOO fold.

    ``test`` is the held-out process set; ``train`` (optional) pins the exact
    processes to train on. ``train=None`` means "every process not in ``test``"
    (augmentation-corrected: holding out any member of an augmentation group
    excludes the whole group — parent + all children — from train; see
    :func:`bp_train.loo.resolve_folds`). ``name`` (optional) labels the fold's
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
    # Cadence (in steps) for evaluating each fold's holdout loss. None -> the
    # logging cadence (`logging.every`); set to 1 to evaluate it every step (one
    # extra forward solve over the holdout per step — slower but a dense curve).
    monitor_every: int | None = Field(None, gt=0)

    @field_validator("parallel_folds", "devices_per_fold", mode="before")
    @classmethod
    def _reject_bool(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("loo parallel/device counts must be int >= 1")
        return value


class RunConfig(ConfigBase):
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
    model_config = ConfigDict(extra="allow", frozen=True)


# --- forward config: a list of self-contained model dirs + optional data override ---
class ForwardDataConfig(ConfigBase):
    prepared: Path | None = None
    processes: tuple[str, ...] | None = None


class ModelRef(ConfigBase):
    """One `models` entry: a self-contained run/checkpoint dir. ``name`` defaults
    to the basename of the bundle's ``config.output.dir`` (resolved later)."""

    name: str | None = None
    path: Path


class ForwardOutputConfig(ConfigBase):
    dir: Path | None = None  # None -> <first model>/forward
    plots: bool = True


class ForwardRunConfig(ConfigBase):
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
    config: RunConfig
    custom_module: ModuleType | None
    custom_py_sha256: str | None


def load_prepare_config(config_path: str | Path) -> LoadedRunConfig:
    return load_run_config(config_path, command="prepare")


def load_train_config(config_path: str | Path) -> LoadedRunConfig:
    return load_run_config(config_path, command="train")


def load_loo_config(config_path: str | Path) -> LoadedRunConfig:
    return load_run_config(config_path, command="loo")


def load_run_config(config_path: str | Path, *, command: _Command) -> LoadedRunConfig:
    path = Path(config_path)
    raw = _read_raw_config(path)
    raw_custom = raw.get("custom")
    _validate_top_level(raw)
    _validate_raw_custom(raw_custom)

    base_dir = path.parent.resolve()
    view = _command_view(raw, command=command)
    config = RunConfig.model_validate(view)
    _validate_required_sections(config, command=command)
    config = _resolve_config_paths(config, base_dir=base_dir)

    custom_module, custom_py_sha256 = _load_custom_module_and_hash(config.custom_py)
    resolved_custom = _resolve_custom(raw_custom, config, custom_module)
    config = config.model_copy(update={"custom": resolved_custom})
    return LoadedRunConfig(
        config=config,
        custom_module=custom_module,
        custom_py_sha256=custom_py_sha256,
    )


def _read_raw_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
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


def _resolve_config_paths(config: RunConfig, *, base_dir: Path) -> RunConfig:
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
    updates["output"] = config.output.model_copy(
        update={"dir": _resolve_path(config.output.dir, base_dir=base_dir)}
    )
    if not updates:
        return config
    return config.model_copy(update=updates)


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def resolve_prepared_path(path: Path) -> Path:
    """Resolve a prepared-data reference to the prepared.json file.

    ``bp-train prepare`` writes its output into a directory
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
    config.json (resume / load_run / forward), ``custom`` comes back as a plain
    dict; this restores the same typed wrapper so reconstruction matches a fresh
    run. No-op if ``custom`` is already resolved (not a dict) or absent.
    """
    if not isinstance(config.custom, dict):
        return config
    resolved = _resolve_custom(config.custom, config, custom_module)
    return config.model_copy(update={"custom": resolved})
