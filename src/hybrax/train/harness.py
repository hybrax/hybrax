from __future__ import annotations

import logging
import time
import warnings
from collections import Counter
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from bp_format.dataclasses import BioProcessCollection
from bp_format.mechanistic import get_rhs_ode
from bp_format.serialization import load_process_collection_json

from .checkpointing import CheckpointConfig, CheckpointWriter
from .defaults import default_build_reaction_module
from .model_api import UserReactionModule, partition_trainable
from .trainer import (
    build_batched_loss_fn_from_sample_loss,
    clamp_padded_time_rows,
    measurement_loss_from_arrays,
    BatchedLossFn,
)
from .logging import RunLogger, StepRecord
from .training_data import (
    TARGET_SOURCE_AUTO,
    TrainingDataStore,
)
from .utils import get_hook, load_custom_module, resolve_config
from .wrapper import HybridOdeWrapper, validate_rhs_ode_compatibility
from .postprocessing import export_predictions_csv

_DEFAULT_BATCHED_MEASUREMENT_LOSS = build_batched_loss_fn_from_sample_loss(
    measurement_loss_from_arrays
)

logger = logging.getLogger(__name__)

# Floor below which `np.var` is treated as "all measurements identical" when
# computing per-target variance for loss normalization.
VARIANCE_EPS = 1e-12


@dataclass(frozen=True)
class TrainHarnessConfig:
    """Configuration for collection-level training harness runs."""

    process_names: tuple[str, ...] | None = None
    target_variable_order: tuple[str, ...] | None = None
    target_source: str = TARGET_SOURCE_AUTO
    steps: int = 50
    batch_size: int | None = None
    shuffle_batches: bool = True
    batch_seed: int | None = None
    optimizer_name: str = "adam"
    learning_rate: Any = 1e-3
    grad_clip_norm: float = 1000.0
    seed: int = 0
    log_every: int = 10
    solver_max_steps: int = 2048
    solver_rtol: float = 1e-5
    solver_atol: float = 1e-7
    solver_use_jump_ts: bool = True
    # Logging / telemetry options (additive; all optional).
    log_process_losses: bool = False
    metrics_csv: str | None = None
    metrics_jsonl: str | None = None
    log_decimals: int = 4
    log_header_every: int = 30
    # Checkpointing (periodic wrapper snapshot + loss curve).
    # When set, cadence ties to log_every; pass None to disable.
    checkpoint_dir: Path | None = None
    # Optional monitor / validation set: a tuple of process names whose loss
    # is evaluated every `log_every` steps with the current wrapper. Diagnostic
    # only — never drives optimizer updates. None disables the monitor.
    monitor_processes: tuple[str, ...] | None = None
    monitor_label: str = "validation"


@dataclass(frozen=True)
class TrainHarnessResult:
    """Summary object returned by the training harness."""

    trained_wrapper: Any
    mean_loss_by_step: tuple[float, ...]
    sampled_loss_by_process_at_log_steps: dict[int, tuple[tuple[str, float], ...]]
    batch_process_names_by_step: tuple[tuple[str, ...], ...]
    per_process_loss_by_step: tuple[tuple[float, ...], ...]
    compile_warmup_seconds: float
    step_time_seconds: tuple[float, ...]
    train_step_input_signature: tuple[object, ...]
    train_step_rebuild_count: int
    # Optional monitor (validation) loss series, populated only when
    # `TrainHarnessConfig.monitor_processes` is set. Maps step -> loss.
    monitor_loss_by_log_step: dict[int, float] = dataclasses.field(default_factory=dict)
    monitor_label: str | None = None


def _ensure_process_names(
    store: TrainingDataStore,
    requested: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if requested is None:
        selected = tuple(store.process_order)
    else:
        selected = tuple(requested)

    if len(selected) == 0:
        raise ValueError("selected process_names must be non-empty")

    counts = Counter(selected)
    duplicates = sorted([name for name, count in counts.items() if count > 1])
    if duplicates:
        raise ValueError(
            f"duplicate entries in process_names are not allowed: {duplicates}"
        )

    missing = [name for name in selected if name not in store.process_order]
    if missing:
        raise ValueError(f"unknown process names in process_names: {missing}")

    return selected


def summarize_train_step_input_signature(*values: object) -> tuple[object, ...]:
    """Return a stable pytree leaf-shape/type summary for train-step inputs."""
    leaves = jtu.tree_leaves(values, is_leaf=lambda value: value is None)
    signature: list[object] = []
    for leaf in leaves:
        if leaf is None:
            signature.append(("none",))
            continue
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            signature.append(("array", tuple(leaf.shape), str(leaf.dtype)))
            continue
        try:
            hash(leaf)
        except TypeError:
            signature.append(("object", type(leaf).__name__))
            continue
        signature.append(("scalar", type(leaf).__name__, repr(leaf)))
        continue
    return tuple(signature)


def _build_optimizer(
    optimizer_name: str,
    learning_rate,
    *,
    grad_clip_norm: float = 1000.0,
) -> optax.GradientTransformation:
    # learning_rate can be a float or an optax Schedule
    if isinstance(learning_rate, (int, float)):
        if float(learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
    if float(grad_clip_norm) < 0.0:
        raise ValueError("grad_clip_norm must be non-negative")
    name = str(optimizer_name)
    if name == "adam":
        base = optax.adam(learning_rate)
    elif name == "sgd":
        base = optax.sgd(learning_rate)
    else:
        raise ValueError("optimizer_name must be one of {'adam', 'sgd'}")
    # zero_nans handles ODE-solver failures (rare); optional
    # clip_by_global_norm is the safety net against blowups in early neural-ODE
    # training.
    transforms = [optax.zero_nans()]
    if float(grad_clip_norm) > 0.0:
        transforms.append(optax.clip_by_global_norm(float(grad_clip_norm)))
    transforms.append(base)
    return optax.chain(*transforms)


def _resolve_effective_batch_size(
    batch_size: int | None, *, selected_process_count: int
) -> int:
    if batch_size is None:
        return int(selected_process_count)
    return int(batch_size)


def _validate_batching_config(
    config: TrainHarnessConfig,
    *,
    selected_process_count: int,
) -> int:
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if isinstance(config.learning_rate, (int, float)) and config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.grad_clip_norm < 0.0:
        raise ValueError("grad_clip_norm must be non-negative")
    if config.log_every <= 0:
        raise ValueError("log_every must be positive")
    if str(config.optimizer_name) not in {"adam", "sgd"}:
        raise ValueError("optimizer_name must be one of {'adam', 'sgd'}")
    effective_batch_size = _resolve_effective_batch_size(
        config.batch_size,
        selected_process_count=selected_process_count,
    )
    if effective_batch_size <= 0:
        raise ValueError("effective batch_size must be positive")
    return effective_batch_size


def _build_batch_index_stream(
    *,
    selected_process_indices: jax.Array | np.ndarray,
    steps: int,
    batch_size: int,
    shuffle_batches: bool,
    batch_seed: int | None,
    seed: int,
) -> jax.Array:
    selected_indices = np.asarray(selected_process_indices, dtype=np.int32)
    if selected_indices.ndim != 1:
        raise ValueError("selected_process_indices must be a 1D array")
    if selected_indices.size == 0:
        raise ValueError("selected_process_indices must be non-empty")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    target_length = int(steps) * int(batch_size)
    if target_length <= 0:
        raise ValueError("steps * batch_size must be positive")

    effective_seed = int(seed) if batch_seed is None else int(batch_seed)
    rng = np.random.default_rng(effective_seed)

    stream: list[int] = []
    while len(stream) < target_length:
        cycle = np.array(selected_indices, copy=True)
        if shuffle_batches:
            cycle = rng.permutation(cycle)
        stream.extend(int(v) for v in cycle.tolist())

    flattened = np.asarray(stream[:target_length], dtype=np.int32)
    return jnp.asarray(flattened.reshape((int(steps), int(batch_size))))


def _build_reaction_module(
    *,
    store: TrainingDataStore,
    config: TrainHarnessConfig,
    custom_module,
    custom_config: dict[str, Any],
    collection: BioProcessCollection,
) -> UserReactionModule:
    hook = get_hook(
        custom_module,
        "build_reaction_module",
        default_build_reaction_module,
    )
    module = hook(
        target_names=list(store.target_names),
        process_names=list(store.process_order),
        config=custom_config,
        seed=int(config.seed),
        collection=collection,
    )
    if not isinstance(module, UserReactionModule):
        raise TypeError(
            "build_reaction_module(...) must return a UserReactionModule instance"
        )
    return module


def _resolve_batched_loss_fn(
    *,
    custom_module,
    custom_cfg: dict[str, Any],
    store: TrainingDataStore,
    collection: BioProcessCollection,
    train_cfg: TrainHarnessConfig,
    allow_batched_loss_hook: bool = True,
) -> BatchedLossFn:
    sample_loss_hook = get_hook(custom_module, "build_sample_loss_fn", None)
    batched_loss_hook = get_hook(custom_module, "build_batched_loss_fn", None)

    if sample_loss_hook is not None and batched_loss_hook is not None:
        raise ValueError(
            "Define either build_sample_loss_fn(...) or "
            "build_batched_loss_fn(...), not both"
        )

    if sample_loss_hook is not None:
        sample_loss_fn = sample_loss_hook(
            default_sample_loss_fn=measurement_loss_from_arrays,
            store=store,
            collection=collection,
            train_cfg=train_cfg,
            config=custom_cfg,
        )
        if not callable(sample_loss_fn):
            raise TypeError("build_sample_loss_fn(...) must return a callable")
        return build_batched_loss_fn_from_sample_loss(sample_loss_fn)

    if batched_loss_hook is not None:
        if not allow_batched_loss_hook:
            raise ValueError(
                "forward loss evaluation supports default loss or "
                "build_sample_loss_fn(...); "
                "build_batched_loss_fn(...) is not supported"
            )
        batched_loss_fn = batched_loss_hook(
            default_loss_fn=_DEFAULT_BATCHED_MEASUREMENT_LOSS,
            store=store,
            collection=collection,
            train_cfg=train_cfg,
            config=custom_cfg,
        )
        if not callable(batched_loss_fn):
            raise TypeError("build_batched_loss_fn(...) must return a callable")
        return batched_loss_fn

    return _DEFAULT_BATCHED_MEASUREMENT_LOSS


def _validate_batched_loss_outputs(
    total_loss,
    per_target_loss,
    per_sample_loss,
    *,
    n_targets: int,
    batch_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    total_arr = jnp.asarray(total_loss)
    per_target_arr = jnp.asarray(per_target_loss)
    per_sample_arr = jnp.asarray(per_sample_loss)

    if total_arr.ndim != 0:
        raise ValueError(
            "batched loss must return scalar total_loss; "
            f"got shape {tuple(total_arr.shape)}"
        )
    if per_target_arr.ndim != 1 or per_target_arr.shape[0] != n_targets:
        raise ValueError(
            "batched loss must return per_target_loss with shape "
            f"({n_targets},); got {tuple(per_target_arr.shape)}"
        )
    if per_sample_arr.ndim != 1 or per_sample_arr.shape[0] != batch_size:
        raise ValueError(
            "batched loss must return per_sample_loss with shape "
            f"({batch_size},); got {tuple(per_sample_arr.shape)}"
        )

    return total_arr, per_target_arr, per_sample_arr


def _build_template_wrapper(
    store: TrainingDataStore,
    *,
    reaction_module: UserReactionModule,
    collection: BioProcessCollection,
    selected_processes: tuple[str, ...],
    state_scale: jax.Array | None = None,
    controls_scale: jax.Array | None = None,
    q_scale: jax.Array | None = None,
    f_scale: jax.Array | None = None,
) -> tuple[HybridOdeWrapper, dict[str, Any]]:
    """Build a HybridOdeWrapper with the same structure train_collection produces.

    Returns the wrapper plus a dict with the per-process RhsOde map under
    ``per_process_rhs`` so callers can reuse it for evaluation.
    """
    if len(selected_processes) == 0:
        raise ValueError("selected_processes must be non-empty")

    per_process_rhs: dict[str, Any] = {}
    reference_rhs = None
    reference_process_name = selected_processes[0]
    for process_name in store.process_order:
        process = collection.processes[process_name]
        rhs = get_rhs_ode(process)
        per_process_rhs[process_name] = rhs
        if process_name == reference_process_name:
            reference_rhs = rhs
    assert reference_rhs is not None
    for process_name in selected_processes[1:]:
        validate_rhs_ode_compatibility(
            reference_process_name,
            reference_rhs,
            process_name,
            per_process_rhs[process_name],
        )

    n_y_cols = int(store.y_meas.shape[2])
    per_col_values: list[list[float]] = [[] for _ in range(n_y_cols)]
    for pname in selected_processes:
        pd = store.get_process(pname)
        y_active = np.asarray(pd.active_y_meas)
        for col in range(n_y_cols):
            per_col_values[col].extend(y_active[:, col].tolist())
    target_variance = jnp.asarray(
        [
            float(np.var(vals)) if vals and float(np.var(vals)) > VARIANCE_EPS else 1.0
            for vals in per_col_values
        ],
        dtype=jnp.float32,
    )

    n_species = len(store.target_names)
    n_modeled_feeds = len(store.modeled_flow_names)
    target_state_indices = jnp.asarray(
        list(range(n_species))
        + list(range(n_species + 1, n_species + 1 + n_modeled_feeds)),
        dtype=jnp.int32,
    )

    wrapper = HybridOdeWrapper.from_process(
        reaction_module=reaction_module,
        process=collection.processes[reference_process_name],
        controls=store.get_process(reference_process_name).controls,
        state_scale=state_scale,
        controls_scale=controls_scale,
        q_scale=q_scale,
        f_scale=f_scale,
        target_variance=target_variance,
        target_state_indices=target_state_indices,
    )
    return wrapper, {"per_process_rhs": per_process_rhs}


@dataclass(frozen=True)
class ForwardConfig:
    """Configuration for a forward evaluation run (no optimizer)."""

    process_names: tuple[str, ...] | None = None
    target_variable_order: tuple[str, ...] | None = None
    target_source: str = TARGET_SOURCE_AUTO
    solver_max_steps: int = 4096
    solver_rtol: float = 1e-5
    solver_atol: float = 1e-7
    solver_use_jump_ts: bool = True


@dataclass
class ForwardResult:
    """Outputs of :func:`forward_from_collection`."""

    trained_wrapper: HybridOdeWrapper
    store: TrainingDataStore
    process_names: tuple[str, ...]
    target_names: tuple[str, ...]
    modeled_flow_names: tuple[str, ...]
    training_process_names: tuple[str, ...]
    per_process_total_loss: dict[str, float]
    per_process_per_target_loss: dict[str, tuple[float, ...]]


def forward_from_collection(
    collection: BioProcessCollection,
    *,
    model_path: str | Path,
    config: ForwardConfig | None = None,
    custom_py: str | Path | None = None,
    runtime_config: dict[str, Any] | None = None,
    training_process_names: tuple[str, ...] | None = None,
) -> ForwardResult:
    """Load a trained wrapper and run one forward pass per selected process.

    Mirrors the setup portion of :func:`train_from_collection` — builds the
    TrainingDataStore, reaction module, and scaling exactly as training did —
    so that ``eqx.tree_deserialise_leaves`` has a structurally identical
    template to deserialise into.
    """
    from .postprocessing import load_trained_wrapper

    cfg = config or ForwardConfig()
    custom_module = load_custom_module(custom_py)
    custom_cfg = resolve_config(custom_module, runtime_config)
    config_targets = custom_cfg.get("target_variable_order")
    if cfg.target_variable_order is not None:
        effective_target_order = cfg.target_variable_order
    elif config_targets:
        effective_target_order = tuple(config_targets)
    else:
        effective_target_order = None

    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=effective_target_order,
        target_source=cfg.target_source,
    )

    # Use ALL processes in the store for template construction so the pytree
    # matches training even if the caller selects a holdout subset below.
    template_processes = tuple(store.process_order)

    # Build a throwaway TrainHarnessConfig for hook-reuse only (build_reaction_module
    # needs a config object with .seed).
    hook_process_names = (
        tuple(training_process_names)
        if training_process_names is not None
        else template_processes
    )
    train_like_cfg = TrainHarnessConfig(
        process_names=hook_process_names,
        target_variable_order=effective_target_order,
        target_source=cfg.target_source,
    )
    reaction_module = _build_reaction_module(
        store=store,
        config=train_like_cfg,
        custom_module=custom_module,
        custom_config=custom_cfg,
        collection=collection,
    )

    estimate_scales_hook = get_hook(custom_module, "estimate_all_scales", None)
    scale_kwargs: dict[str, Any] = {}
    if estimate_scales_hook is not None:
        state_scale, controls_scale, q_scale, f_scale = estimate_scales_hook(
            collection,
            list(store.target_names),
            custom_cfg,
        )
        scale_kwargs = {
            "state_scale": state_scale,
            "controls_scale": controls_scale,
            "q_scale": q_scale,
            "f_scale": f_scale,
        }

    template_wrapper, extras = _build_template_wrapper(
        store,
        reaction_module=reaction_module,
        collection=collection,
        selected_processes=template_processes,
        **scale_kwargs,
    )
    per_process_rhs = extras["per_process_rhs"]
    batched_loss_fn = _resolve_batched_loss_fn(
        custom_module=custom_module,
        custom_cfg=custom_cfg,
        store=store,
        collection=collection,
        train_cfg=train_like_cfg,
        allow_batched_loss_hook=False,
    )

    trained_wrapper = load_trained_wrapper(model_path, template=template_wrapper)

    # Resolve which processes to evaluate
    if cfg.process_names is not None:
        missing = [n for n in cfg.process_names if n not in store.process_order]
        if missing:
            raise ValueError(
                f"forward: unknown process names {missing}; "
                f"available={tuple(store.process_order)}"
            )
        eval_processes = tuple(cfg.process_names)
    else:
        eval_processes = tuple(store.process_order)

    if cfg.solver_max_steps <= 0:
        raise ValueError("solver_max_steps must be positive")
    if cfg.solver_rtol <= 0.0:
        raise ValueError("solver_rtol must be positive")
    if cfg.solver_atol <= 0.0:
        raise ValueError("solver_atol must be positive")

    batch_controls = store.controls_store.as_batch_controls()
    all_Cin = []
    all_Cin_modeled = []
    for process_name in store.process_order:
        rhs = per_process_rhs[process_name]
        all_Cin.append(rhs.Cin)
        all_Cin_modeled.append(rhs.Cin_modeled)
    batched_Cin = jnp.stack(all_Cin)
    batched_Cin_modeled = jnp.stack(all_Cin_modeled)

    per_process_total: dict[str, float] = {}
    per_process_per_target: dict[str, tuple[float, ...]] = {}
    for process_name in eval_processes:
        process_idx = store.process_order.index(process_name)
        batch = store.gather_batch(jnp.asarray([process_idx], dtype=jnp.int32))
        jump_ts_rows = None
        if cfg.solver_use_jump_ts:
            jump_ts_rows = clamp_padded_time_rows(
                store.controls_store.step_ts[batch.process_indices],
                store.controls_store.step_ts_lengths[batch.process_indices],
            )
        total, per_target, _per_sample = batched_loss_fn(
            trained_wrapper,
            batch,
            batch_controls,
            batched_Cin,
            batched_Cin_modeled,
            jump_ts_rows,
            max_solver_steps=int(cfg.solver_max_steps),
            solver_rtol=float(cfg.solver_rtol),
            solver_atol=float(cfg.solver_atol),
        )
        total, per_target, _per_sample = _validate_batched_loss_outputs(
            total,
            per_target,
            _per_sample,
            n_targets=int(batch.y_meas.shape[2]),
            batch_size=int(batch.process_indices.shape[0]),
        )
        per_process_total[process_name] = float(total)
        per_process_per_target[process_name] = tuple(float(v) for v in per_target)

    target_column_labels = tuple(store.target_names) + tuple(
        f"B_{name}_cum" for name in store.modeled_flow_names
    )

    return ForwardResult(
        trained_wrapper=trained_wrapper,
        store=store,
        process_names=eval_processes,
        target_names=target_column_labels,
        modeled_flow_names=tuple(store.modeled_flow_names),
        training_process_names=tuple(training_process_names)
        if training_process_names is not None
        else (),
        per_process_total_loss=per_process_total,
        per_process_per_target_loss=per_process_per_target,
    )


def train_collection(
    store: TrainingDataStore,
    *,
    reaction_module: UserReactionModule,
    collection: BioProcessCollection,
    config: TrainHarnessConfig | None = None,
    state_scale: jax.Array | None = None,
    controls_scale: jax.Array | None = None,
    q_scale: jax.Array | None = None,
    f_scale: jax.Array | None = None,
    batched_loss_fn: BatchedLossFn | None = None,
) -> TrainHarnessResult:
    """Train one reaction module over one or many processes from one store."""
    cfg = config or TrainHarnessConfig()
    effective_batched_loss_fn = (
        _DEFAULT_BATCHED_MEASUREMENT_LOSS
        if batched_loss_fn is None
        else batched_loss_fn
    )
    selected_processes = _ensure_process_names(store, cfg.process_names)

    effective_batch_size = _validate_batching_config(
        cfg,
        selected_process_count=len(selected_processes),
    )
    selected_process_indices = jnp.asarray(
        [store.process_order.index(name) for name in selected_processes],
        dtype=jnp.int32,
    )
    batch_index_stream = _build_batch_index_stream(
        selected_process_indices=selected_process_indices,
        steps=int(cfg.steps),
        batch_size=effective_batch_size,
        shuffle_batches=bool(cfg.shuffle_batches),
        batch_seed=cfg.batch_seed,
        seed=cfg.seed,
    )
    if cfg.solver_max_steps <= 0:
        raise ValueError("solver_max_steps must be positive")
    if cfg.solver_rtol <= 0.0:
        raise ValueError("solver_rtol must be positive")
    if cfg.solver_atol <= 0.0:
        raise ValueError("solver_atol must be positive")

    batch_controls = store.controls_store.as_batch_controls()
    warmup_batch = store.gather_batch(batch_index_stream[0])

    # Build per-process RhsOde and validate structural compatibility
    per_process_rhs = {}
    reference_rhs = None
    reference_process_name = selected_processes[0]
    for process_name in store.process_order:
        process = collection.processes[process_name]
        rhs = get_rhs_ode(process)
        per_process_rhs[process_name] = rhs
        if process_name == reference_process_name:
            reference_rhs = rhs

    assert reference_rhs is not None
    for process_name in selected_processes[1:]:
        validate_rhs_ode_compatibility(
            reference_process_name,
            reference_rhs,
            process_name,
            per_process_rhs[process_name],
        )

    # Compute per-target variance for loss normalization.
    # y_meas columns are [species..., B_modeled_cum_per_modeled_feed...].
    # Rule:
    #   - if any nonzero variance is observed → use that variance
    #   - if all measurements are identical (variance == 0) → fall back to 1.0
    # The previous version used max(var, 1.0) which clamped small but nonzero
    # variances (e.g. cumulative base feed in kg² ~ 5e-3) to 1.0, making the
    # loss insensitive by ~200x.
    n_y_cols = int(store.y_meas.shape[2])  # n_species + n_modeled_feeds
    _per_col_values: list[list[float]] = [[] for _ in range(n_y_cols)]
    for pname in selected_processes:
        pd = store.get_process(pname)
        y_active = np.asarray(pd.active_y_meas)  # [n_meas, n_y_cols]
        for col in range(n_y_cols):
            _per_col_values[col].extend(y_active[:, col].tolist())
    target_variance = jnp.asarray(
        [
            float(np.var(vals)) if vals and float(np.var(vals)) > VARIANCE_EPS else 1.0
            for vals in _per_col_values
        ],
        dtype=jnp.float32,
    )
    _target_labels = list(store.target_names) + [
        f"B_{name}_cum" for name in store.modeled_flow_names
    ]
    logger.info(
        "target_variance: %s",
        {name: f"{v:.2f}" for name, v in zip(_target_labels, target_variance.tolist())},
    )

    # Build target_state_indices: species columns + cumulative-modeled-feed
    # columns. State layout is [species..., V_cont, B_modeled_cum_0, ...] so
    # V_cont (at index n_species) is in the state but NOT a loss target.
    n_species = len(store.target_names)
    n_modeled_feeds = len(store.modeled_flow_names)
    target_state_indices = jnp.asarray(
        list(range(n_species))
        + list(range(n_species + 1, n_species + 1 + n_modeled_feeds)),
        dtype=jnp.int32,
    )

    # Build wrapper from reference process
    wrapper = HybridOdeWrapper.from_process(
        reaction_module=reaction_module,
        process=collection.processes[reference_process_name],
        controls=store.get_process(reference_process_name).controls,
        state_scale=state_scale,
        controls_scale=controls_scale,
        q_scale=q_scale,
        f_scale=f_scale,
        target_variance=target_variance,
        target_state_indices=target_state_indices,
    )

    # Stack per-process Cin arrays: [n_store_processes, n_feeds, n_species]
    all_Cin = []
    all_Cin_modeled = []
    for process_name in store.process_order:
        rhs = per_process_rhs[process_name]
        all_Cin.append(rhs.Cin)
        all_Cin_modeled.append(rhs.Cin_modeled)
    batched_Cin = jnp.stack(all_Cin)
    batched_Cin_modeled = jnp.stack(all_Cin_modeled)

    trainable_params, trainable_static = partition_trainable(reaction_module)
    optimizer = _build_optimizer(
        cfg.optimizer_name,
        cfg.learning_rate,
        grad_clip_norm=float(cfg.grad_clip_norm),
    )
    optimizer_state = optimizer.init(trainable_params)
    train_step_input_signature = summarize_train_step_input_signature(
        wrapper,
        trainable_params,
        optimizer_state,
        warmup_batch,
    )

    def _make_batched_step():
        def _step_fn(
            current_wrapper: HybridOdeWrapper,
            current_trainable_params: Any,
            current_optimizer_state: Any,
            current_batch,
        ):
            jump_ts_rows = None
            if cfg.solver_use_jump_ts:
                jump_ts_rows = clamp_padded_time_rows(
                    store.controls_store.step_ts[current_batch.process_indices],
                    store.controls_store.step_ts_lengths[current_batch.process_indices],
                )

            def _loss_fn(trainable_params: Any) -> jax.Array:
                reaction_module_updated = eqx.combine(
                    trainable_params,
                    trainable_static,
                )
                candidate_wrapper = eqx.tree_at(
                    lambda current: current.reaction_module,
                    current_wrapper,
                    reaction_module_updated,
                )
                total_loss, per_target, per_sample = effective_batched_loss_fn(
                    candidate_wrapper,
                    current_batch,
                    batch_controls,
                    batched_Cin,
                    batched_Cin_modeled,
                    jump_ts_rows,
                    max_solver_steps=int(cfg.solver_max_steps),
                    solver_rtol=float(cfg.solver_rtol),
                    solver_atol=float(cfg.solver_atol),
                )
                total_loss, per_target, per_sample = _validate_batched_loss_outputs(
                    total_loss,
                    per_target,
                    per_sample,
                    n_targets=int(current_batch.y_meas.shape[2]),
                    batch_size=int(current_batch.process_indices.shape[0]),
                )
                return total_loss, (per_target, per_sample)

            (loss, (per_target_loss, per_sample_loss)), grads = (
                eqx.filter_value_and_grad(
                    _loss_fn,
                    has_aux=True,
                )(current_trainable_params)
            )
            updates, next_optimizer_state = optimizer.update(
                grads,
                current_optimizer_state,
                params=current_trainable_params,
            )
            trainable_updated = eqx.apply_updates(current_trainable_params, updates)
            reaction_module_updated = eqx.combine(
                trainable_updated,
                trainable_static,
            )
            wrapper_updated = eqx.tree_at(
                lambda current: current.reaction_module,
                current_wrapper,
                reaction_module_updated,
            )
            return (
                wrapper_updated,
                trainable_updated,
                loss,
                per_target_loss,
                per_sample_loss,
                next_optimizer_state,
            )

        return eqx.filter_jit(_step_fn)

    step_fn = _make_batched_step()
    rebuild_count = 0

    logger.info(
        "train harness setup: processes=%s, targets=%s source=%s steps=%d "
        "batch_size=%d optimizer=%s lr=%s grad_clip=%s",
        list(selected_processes),
        list(store.target_names),
        store.target_source,
        cfg.steps,
        effective_batch_size,
        cfg.optimizer_name,
        cfg.learning_rate,
        cfg.grad_clip_norm,
    )

    logger.info("JIT compiling train step (warmup)...")
    warmup_t0 = time.perf_counter()
    (
        _warmup_wrapper,
        _warmup_trainable,
        warmup_loss,
        _warmup_pt,
        _warmup_ps,
        _warmup_opt_state,
    ) = step_fn(
        wrapper,
        trainable_params,
        optimizer_state,
        warmup_batch,
    )
    jax.block_until_ready(warmup_loss)
    warmup_compile_seconds = time.perf_counter() - warmup_t0
    logger.info(
        "JIT compilation done in %.1fs, warmup loss=%.6g",
        warmup_compile_seconds,
        float(warmup_loss),
    )

    checkpoint_writer = CheckpointWriter(
        CheckpointConfig(
            output_dir=Path(cfg.checkpoint_dir)
            if cfg.checkpoint_dir is not None
            else Path("."),
            every=int(cfg.log_every) if cfg.checkpoint_dir is not None else 0,
        )
    )
    loss_so_far: list[float] = []

    # Optional monitor (validation) batch — diagnostic only, recomputed at
    # log-step cadence with the current wrapper. JIT compiles once on first
    # use because the batch shape is stable across log steps.
    monitor_batch = None
    monitor_jump_ts_rows = None
    if cfg.monitor_processes:
        monitor_unknown = [
            n for n in cfg.monitor_processes if n not in store.process_order
        ]
        if monitor_unknown:
            raise ValueError(
                f"monitor_processes contains unknown names: {monitor_unknown}; "
                f"available={tuple(store.process_order)}"
            )
        monitor_indices = jnp.asarray(
            [store.process_order.index(name) for name in cfg.monitor_processes],
            dtype=jnp.int32,
        )
        monitor_batch = store.gather_batch(monitor_indices)
        if cfg.solver_use_jump_ts:
            monitor_jump_ts_rows = clamp_padded_time_rows(
                store.controls_store.step_ts[monitor_batch.process_indices],
                store.controls_store.step_ts_lengths[monitor_batch.process_indices],
            )

    with RunLogger(
        log_every=int(cfg.log_every),
        log_process_losses=bool(cfg.log_process_losses),
        metrics_csv=cfg.metrics_csv,
        metrics_jsonl=cfg.metrics_jsonl,
        log_decimals=int(cfg.log_decimals),
        log_header_every=int(cfg.log_header_every),
    ) as run_log:
        run_log.start(
            target_names=_target_labels,
            process_names=selected_processes,
            total_steps=int(cfg.steps),
            compile_warmup_seconds=float(warmup_compile_seconds),
        )

        for step_index in range(cfg.steps):
            batch_indices = batch_index_stream[step_index]
            batch = store.gather_batch(batch_indices)
            current_signature = summarize_train_step_input_signature(
                wrapper,
                trainable_params,
                optimizer_state,
                batch,
            )
            if current_signature != train_step_input_signature:
                step_fn = _make_batched_step()
                rebuild_count += 1
                run_log.record_rebuild(step_index + 1)

            t0 = time.perf_counter()
            (
                wrapper,
                trainable_params,
                loss,
                per_target_loss,
                per_sample_loss,
                optimizer_state,
            ) = step_fn(
                wrapper,
                trainable_params,
                optimizer_state,
                batch,
            )
            jax.block_until_ready(loss)
            step_dt = time.perf_counter() - t0

            batch_names = tuple(
                store.process_order[int(i)]
                for i in np.asarray(batch.process_indices).tolist()
            )

            # Monitor / validation loss at log-step cadence (cheap: one
            # forward pass per `log_every` training steps).
            monitor_loss_value: float | None = None
            if monitor_batch is not None and (step_index + 1) % int(cfg.log_every) == 0:
                m_total, _m_per_target, _m_per_sample = effective_batched_loss_fn(
                    wrapper,
                    monitor_batch,
                    batch_controls,
                    batched_Cin,
                    batched_Cin_modeled,
                    monitor_jump_ts_rows,
                    max_solver_steps=int(cfg.solver_max_steps),
                    solver_rtol=float(cfg.solver_rtol),
                    solver_atol=float(cfg.solver_atol),
                )
                jax.block_until_ready(m_total)
                monitor_loss_value = float(m_total)

            run_log.record_step(
                StepRecord(
                    step=step_index + 1,
                    total_steps=int(cfg.steps),
                    mean_loss=float(loss),
                    per_target_loss=tuple(
                        float(v) for v in np.asarray(per_target_loss).tolist()
                    ),
                    per_process_loss=tuple(
                        float(v) for v in np.asarray(per_sample_loss).tolist()
                    ),
                    target_names=tuple(_target_labels),
                    process_names=batch_names,
                    step_dt=float(step_dt),
                    rebuild_count=int(rebuild_count),
                    monitor_loss=monitor_loss_value,
                    monitor_label=cfg.monitor_label if monitor_loss_value is not None else None,
                )
            )

            loss_so_far.append(float(loss))
            checkpoint_step_dir = checkpoint_writer.maybe_write(
                step=step_index + 1,
                wrapper=wrapper,
                mean_loss_by_step=loss_so_far,
            )
            if checkpoint_step_dir is not None:
                export_predictions_csv(
                    wrapper,
                    collection,
                    store,
                    checkpoint_step_dir / "predictions.csv",
                    process_names=selected_processes,
                    solver_max_steps=int(cfg.solver_max_steps),
                    solver_rtol=float(cfg.solver_rtol),
                    solver_atol=float(cfg.solver_atol),
                    solver_use_jump_ts=bool(cfg.solver_use_jump_ts),
                )
                checkpoint_writer.publish_latest(checkpoint_step_dir)

        history = run_log.finalize()

    return TrainHarnessResult(
        trained_wrapper=wrapper,
        compile_warmup_seconds=float(warmup_compile_seconds),
        train_step_input_signature=train_step_input_signature,
        **history,
    )


def train_from_collection(
    collection: BioProcessCollection,
    *,
    config: TrainHarnessConfig | None = None,
    custom_py: str | Path | None = None,
    runtime_config: dict[str, Any] | None = None,
) -> TrainHarnessResult:
    """Train from an already-loaded process collection with optional custom hooks.

    Use this entry point when the caller has already deserialized the
    prepared JSON (e.g. the CLI, which also needs the collection for
    post-training plotting). For path-based callers,
    :func:`train_from_prepared_json` is a thin wrapper that loads the
    collection and delegates here.
    """
    cfg = config or TrainHarnessConfig()
    custom_module = load_custom_module(custom_py)
    custom_cfg = resolve_config(custom_module, runtime_config)
    config_targets = custom_cfg.get("target_variable_order")
    if cfg.target_variable_order is not None:
        effective_target_order = cfg.target_variable_order
    elif config_targets:
        effective_target_order = tuple(config_targets)
    else:
        effective_target_order = None
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=effective_target_order,
        target_source=cfg.target_source,
    )
    if effective_target_order is None:
        warnings.warn(
            "No target_variable_order specified in custom.py CONFIG or --target "
            f"flag. Defaulting to target_source={store.target_source!r} measured "
            f"targets: {tuple(store.target_names)}. Specify "
            "CONFIG['target_variable_order'] in custom.py to silence this warning.",
            stacklevel=2,
        )
    logger.info("Training targets: %s", tuple(store.target_names))

    selected_processes = _ensure_process_names(store, cfg.process_names)
    train_cfg = dataclasses.replace(
        cfg,
        process_names=selected_processes,
        target_variable_order=effective_target_order,
    )
    reaction_module = _build_reaction_module(
        store=store,
        config=train_cfg,
        custom_module=custom_module,
        custom_config=custom_cfg,
        collection=collection,
    )
    # Call optional build_learning_rate hook (overrides CLI --learning-rate)
    lr_hook = get_hook(custom_module, "build_learning_rate", None)
    if lr_hook is not None:
        lr = lr_hook(custom_cfg, train_cfg)
        train_cfg = dataclasses.replace(train_cfg, learning_rate=lr)

    batched_loss_fn = _resolve_batched_loss_fn(
        custom_module=custom_module,
        custom_cfg=custom_cfg,
        store=store,
        collection=collection,
        train_cfg=train_cfg,
    )

    # Call optional estimate_all_scales hook for state/controls/q/f scaling
    estimate_scales_hook = get_hook(custom_module, "estimate_all_scales", None)
    scale_kwargs: dict[str, Any] = {}
    if estimate_scales_hook is not None:
        state_scale, controls_scale, q_scale, f_scale = estimate_scales_hook(
            collection,
            list(store.target_names),
            custom_cfg,
        )
        scale_kwargs = {
            "state_scale": state_scale,
            "controls_scale": controls_scale,
            "q_scale": q_scale,
            "f_scale": f_scale,
        }

    return train_collection(
        store,
        reaction_module=reaction_module,
        collection=collection,
        config=train_cfg,
        batched_loss_fn=batched_loss_fn,
        **scale_kwargs,
    )


def train_from_prepared_json(
    prepared_json: str | Path,
    *,
    config: TrainHarnessConfig | None = None,
    custom_py: str | Path | None = None,
    runtime_config: dict[str, Any] | None = None,
) -> TrainHarnessResult:
    """Load a prepared JSON and train from it.

    Thin wrapper around :func:`train_from_collection` for callers that
    only have a path. Callers that already hold a deserialized collection
    (e.g. the CLI, which reuses it for post-training plotting) should
    invoke ``train_from_collection`` directly to avoid loading the
    prepared JSON twice.
    """
    collection = load_process_collection_json(Path(prepared_json))
    return train_from_collection(
        collection,
        config=config,
        custom_py=custom_py,
        runtime_config=runtime_config,
    )
