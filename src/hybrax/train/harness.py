from __future__ import annotations

import logging
import math
import sys
import os
import time
import warnings
from collections import Counter
import dataclasses
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from bp_format.dataclasses import BioProcessCollection
from bp_format.inspect import print_rhs_ode
from bp_format.mechanistic import build_rhs_ode
from bp_format.serialization import load_process_collection

from .checkpointing import CheckpointWriter
from .plotting_worker import BackgroundPlotter
from .defaults import (
    DefaultLossModule,
    default_build_loss_module,
    default_build_reaction_module,
)
from .inspect import print_reaction_schema, print_trainable_structure
from .model_api import (
    EstimatedScales,
    UserLossModule,
    UserReactionModule,
    partition_trainable,
)
from .trainer import (
    build_batched_loss_fn,
    clamp_padded_time_rows,
    evaluate_one_sample_loss,
)
from .logging import RunLogger, StepRecord
from .training_data import (
    TARGET_SOURCE_AUTO,
    TrainingDataStore,
)
from .run_config import RunConfig
from .utils import get_hook, load_custom_module, resolve_config
from .wrapper import HybridOdeWrapper, validate_rhs_ode_compatibility
from .postprocessing import (
    DenseProcessExport,
    build_process_plot_data,
    dense_exports_from_save_outputs,
    export_predictions_csv,
    plot_grad_norm_curve,
    plot_loss_curve,
    plot_process_simulations,
)

# Single batched loss fn: module-agnostic, reads wrapper.loss_module at call time.
_BATCHED_LOSS_FN = build_batched_loss_fn()
# JIT'd once at import; reused by every dense-export call (forward + each training
# checkpoint) so they share one compile per batch shape.
_BATCHED_LOSS_FN_JIT = eqx.filter_jit(_BATCHED_LOSS_FN)

logger = logging.getLogger(__name__)


def compute_dense_exports(
    trained_wrapper: HybridOdeWrapper,
    store: TrainingDataStore,
    collection: BioProcessCollection,
    process_names: tuple[str, ...],
    *,
    solver_max_steps: int,
    solver_rtol: float,
    solver_atol: float,
    solver_use_jump_ts: bool,
    prediction_grid_n: int = 200,
) -> tuple[np.ndarray, np.ndarray, dict[str, DenseProcessExport]]:
    """One batched, jitted solve over ``process_names`` → per-process dense exports.

    THE single source of dense prediction trajectories: forward evaluation and the
    training-checkpoint ``predictions.csv`` writer both call this instead of
    solving per process. The prediction grid is harvested from the same loss solve
    (``BatchControls`` + ``discrete_events`` ``jump_ts``), so predictions match
    training and there is no second simulation. Returns ``(per_sample_total,
    per_sample_per_target, dense_exports)`` (loss arrays are ``np``, aligned with
    ``process_names``)."""
    rhs_by_name = {
        name: build_rhs_ode(collection.processes[name]) for name in store.process_order
    }
    batched_Cin = jnp.stack(
        [rhs_by_name[name].Cin_controlled_FVCs for name in store.process_order]
    )
    batched_Cin_modeled = jnp.stack(
        [rhs_by_name[name].Cin_modeled_FVCs for name in store.process_order]
    )
    batch_controls = store.controls_store.as_batch_controls()
    eval_indices = jnp.asarray(
        [store.process_order.index(name) for name in process_names], dtype=jnp.int32
    )
    batch = store.gather_batch(eval_indices)
    jump_ts_rows = None
    if solver_use_jump_ts:
        jump_ts_rows = clamp_padded_time_rows(
            store.controls_store.jump_ts[batch.process_indices],
            store.controls_store.jump_ts_lengths[batch.process_indices],
        )
    (
        _mean_total,
        per_sample_per_target,
        per_sample_total,
        prediction_t,
        prediction_save_outputs,
        *_,  # trailing per-sample fail_time (unused on the forward/export path)
    ) = _BATCHED_LOSS_FN_JIT(
        trained_wrapper,
        batch,
        batch_controls,
        batched_Cin,
        batched_Cin_modeled,
        jump_ts_rows,
        max_solver_steps=int(solver_max_steps),
        solver_rtol=float(solver_rtol),
        solver_atol=float(solver_atol),
        prediction_grid_n=int(prediction_grid_n),
    )
    dense_exports = dense_exports_from_save_outputs(
        prediction_t, prediction_save_outputs, trained_wrapper, process_names
    )
    return (
        np.asarray(per_sample_total),
        np.asarray(per_sample_per_target),
        dense_exports,
    )


# Floor below which `np.var` is treated as "all measurements identical" when
# computing per-target variance for loss normalization.


@dataclass(frozen=True)
class TrainHarnessConfig:
    """Configuration for collection-level training harness runs."""

    process_names: tuple[str, ...] | None = None
    target_variable_order: tuple[str, ...] | None = None
    target_source: str = TARGET_SOURCE_AUTO
    epochs: int = 5
    batch_size: int | None = None
    shuffle_batches: bool = True
    batch_seed: int | None = None
    optimizer_name: str = "adam"
    learning_rate: Any = 1e-3
    grad_clip_norm: float = 1000.0
    seed: int = 0
    solver_max_steps: int = 2048
    solver_rtol: float = 1e-5
    solver_atol: float = 1e-7
    solver_use_jump_ts: bool = True
    allow_stateful_models: bool = False
    # Logging / telemetry options (additive; all optional).
    log_process_losses: bool = False
    metrics_csv: str | None = None
    metrics_jsonl: str | None = None
    log_decimals: int = 4
    # Checkpointing. ``checkpoint_every`` is measured in epochs; zero disables
    # periodic writes, while a final checkpoint remains mandatory when configured.
    # ``plots`` gates background plot rendering; ``prepared_path`` is the resolved
    # prepared.json(.gz) bundled into every checkpoint (so each is self-contained)
    # and the source of measured-point overlays for per-checkpoint process plots.
    checkpoint_dir: Path | None = None
    checkpoint_every: float = 1.0
    plots: bool = True
    prepared_path: Path | None = None
    # Optional LOO holdout set, evaluated whenever a checkpoint is written.
    holdout_processes: tuple[str, ...] | None = None
    holdout_label: str = "holdout"


@dataclass(frozen=True)
class TrainHarnessResult:
    """Summary object returned by the training harness."""

    trained_wrapper: Any
    mean_loss_by_step: tuple[float, ...]
    batch_process_names_by_step: tuple[tuple[str, ...], ...]
    per_process_loss_by_step: tuple[tuple[float, ...], ...]
    compile_warmup_seconds: float
    step_time_seconds: tuple[float, ...]
    train_step_input_signature: tuple[object, ...]
    train_step_rebuild_count: int
    # Per-target training-loss breakdown: tuple of length n_targets per step.
    per_target_loss_by_step: tuple[tuple[float, ...], ...] = ()
    target_names: tuple[str, ...] = ()
    holdout_loss_by_step: dict[int, float] = dataclasses.field(default_factory=dict)
    holdout_label: str | None = None
    grad_norm_by_step: tuple[float, ...] = ()
    updates_completed: int = 0


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


def build_optimizer_for_run(
    *,
    custom_module,
    custom_cfg: Any,
    train_cfg: TrainHarnessConfig,
    total_updates: int,
) -> tuple[optax.GradientTransformation, TrainHarnessConfig]:
    """Resolve the optimizer exactly as :func:`train_from_collection` does.

    Applies the optional ``build_learning_rate`` + ``build_optimizer`` hooks,
    falling back to the default chain. Returns ``(optimizer, train_cfg)`` where
    ``train_cfg`` carries any hook-overridden learning rate. Shared with
    ``serialization.load_run`` so optimizer-state loading uses an identical
    template.
    """
    lr_hook = get_hook(custom_module, "build_learning_rate", None)
    if lr_hook is not None:
        train_cfg = dataclasses.replace(
            train_cfg,
            learning_rate=lr_hook(custom_cfg, train_cfg, total_updates),
        )
    optimizer_hook = get_hook(custom_module, "build_optimizer", None)
    if optimizer_hook is not None:
        optimizer = optimizer_hook(custom_cfg, train_cfg)
    else:
        optimizer = _build_optimizer(
            train_cfg.optimizer_name,
            train_cfg.learning_rate,
            grad_clip_norm=train_cfg.grad_clip_norm,
        )
    return optimizer, train_cfg


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
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if isinstance(config.learning_rate, (int, float)) and config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.grad_clip_norm < 0.0:
        raise ValueError("grad_clip_norm must be non-negative")
    if str(config.optimizer_name) not in {"adam", "sgd"}:
        raise ValueError("optimizer_name must be one of {'adam', 'sgd'}")
    effective_batch_size = _resolve_effective_batch_size(
        config.batch_size,
        selected_process_count=selected_process_count,
    )
    if effective_batch_size <= 0:
        raise ValueError("effective batch_size must be positive")
    if effective_batch_size > selected_process_count:
        raise ValueError("batch_size cannot exceed the selected process count")
    return effective_batch_size


def derive_update_budget(
    config: TrainHarnessConfig, *, selected_process_count: int
) -> tuple[int, int, int]:
    batch_size = _validate_batching_config(
        config, selected_process_count=selected_process_count
    )
    batches_per_epoch = selected_process_count // batch_size
    return batch_size, batches_per_epoch, config.epochs * batches_per_epoch


def _build_batch_index_stream(
    *,
    selected_process_indices: jax.Array | np.ndarray,
    epochs: int,
    batch_size: int,
    shuffle_batches: bool,
    batch_seed: int | None,
    seed: int,
) -> jax.Array:
    selected_indices = np.asarray(selected_process_indices, dtype=np.int32)
    if selected_indices.ndim != 1 or selected_indices.size == 0:
        raise ValueError("selected_process_indices must be a non-empty 1D array")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if batch_size > selected_indices.size:
        raise ValueError("batch_size cannot exceed the selected process count")
    batches_per_epoch = selected_indices.size // batch_size
    used = batches_per_epoch * batch_size
    rng = np.random.default_rng(int(seed) if batch_seed is None else int(batch_seed))
    epochs_indices = []
    for _ in range(epochs):
        indices = (
            rng.permutation(selected_indices)
            if shuffle_batches
            else np.array(selected_indices, copy=True)
        )
        epochs_indices.append(indices[:used].reshape(batches_per_epoch, batch_size))
    return jnp.asarray(np.stack(epochs_indices).reshape(-1, batch_size))


def _checkpoint_update_boundaries(
    every: float, *, batches_per_epoch: int, total_updates: int
) -> frozenset[int]:
    boundaries = {total_updates}
    if not math.isfinite(every) or every < 0:
        raise ValueError("checkpoint_every must be finite and nonnegative")
    cadence = Fraction(str(every))
    if cadence == 0:
        return frozenset(boundaries)
    interval = cadence * batches_per_epoch
    k = 1
    while True:
        threshold = interval * k
        update = (
            threshold.numerator + threshold.denominator - 1
        ) // threshold.denominator
        boundaries.add(min(update, total_updates))
        if update >= total_updates:
            return frozenset(boundaries)
        # Skip periodic ordinals that round to the same update boundary.
        quotient = Fraction(update, 1) / interval
        k = max(k + 1, quotient.numerator // quotient.denominator + 1)


def _require_stateful_opt_in(reaction_module, allow_stateful_models: bool) -> None:
    """Reject a stateful (``n_latent > 0``) module unless opt-in is set."""
    if reaction_module.n_latent > 0 and not allow_stateful_models:
        raise ValueError(
            "Stateful reaction modules (n_latent > 0) require explicit opt-in: "
            "set train.allow_stateful_models=true."
        )


def _build_reaction_module(
    *,
    store: TrainingDataStore,
    config: TrainHarnessConfig,
    custom_module,
    custom_config: Any,
    collection: BioProcessCollection,
    scale_kwargs: dict[str, Any],
) -> UserReactionModule:
    hook = get_hook(
        custom_module,
        "build_reaction_module",
        default_build_reaction_module,
    )
    module = hook(
        target_names=list(store.name_measured),
        process_names=list(store.process_order),
        config=custom_config,
        seed=int(config.seed),
        collection=collection,
        **scale_kwargs,
    )
    if not isinstance(module, UserReactionModule):
        raise TypeError(
            "build_reaction_module(...) must return a UserReactionModule instance"
        )
    _require_stateful_opt_in(module, config.allow_stateful_models)
    return module


def _loss_target_labels(store: TrainingDataStore) -> list[str]:
    """Canonical loss target-column labels: measured species + cumulative feeds.

    These name the columns of ``SCL_target_pred`` (``target_state_indices``),
    so ``DefaultLossModule`` emits exactly one term per label.
    """
    return list(store.name_measured) + [
        f"B_{name}_cum" for name in (store.name_modeled_FVCs + store.name_modeled_SVCs)
    ]


def _build_loss_module(
    *,
    store: TrainingDataStore,
    config: TrainHarnessConfig,
    custom_module,
    custom_config: Any,
    collection: BioProcessCollection,
) -> UserLossModule:
    hook = get_hook(custom_module, "build_loss_module", default_build_loss_module)
    module = hook(
        target_names=_loss_target_labels(store),
        process_names=list(store.process_order),
        config=custom_config,
        seed=int(config.seed),
        collection=collection,
    )
    if not isinstance(module, UserLossModule):
        raise TypeError("build_loss_module(...) must return a UserLossModule instance")
    return module


def _resolve_estimated_scales(
    *,
    custom_module,
    collection: BioProcessCollection,
    store: TrainingDataStore,
    custom_cfg: Any,
) -> dict[str, Any]:
    """Call the optional ``estimate_all_scales`` hook and unpack into kwargs.

    The hook returns an ``EstimatedScales`` dataclass (or a falsy result if no
    hook is configured). The output flattens into ``SCALE_*`` kwargs that feed
    ``build_reaction_module``.
    """
    hook = get_hook(custom_module, "estimate_all_scales", None)
    if hook is None:
        return {}
    estimated = hook(collection, list(store.name_measured), custom_cfg)
    if not isinstance(estimated, EstimatedScales):
        raise TypeError(
            "estimate_all_scales(...) must return an EstimatedScales dataclass; "
            f"got {type(estimated).__name__}"
        )
    return {
        "SCALE_modeled_RMCs": estimated.SCALE_modeled_RMCs,
        "SCALE_modeled_PVs": estimated.SCALE_modeled_PVs,
        "SCALE_V_in_cumulative": estimated.SCALE_V_in_cumulative,
        "SCALE_modeled_FVCs_cumulative": estimated.SCALE_modeled_FVCs_cumulative,
        "SCALE_controlled_FVCs_cumulative": estimated.SCALE_controlled_FVCs_cumulative,
        "SCALE_controlled_FVCs_rates": estimated.SCALE_controlled_FVCs_rates,
        "SCALE_controlled_FVCs_Cin": estimated.SCALE_controlled_FVCs_Cin,
        "SCALE_controlled_PVs": estimated.SCALE_controlled_PVs,
        "SCALE_modeled_FVCs_Cin": estimated.SCALE_modeled_FVCs_Cin,
        "SCALE_modeled_BiologicalOde_rates": (
            estimated.SCALE_modeled_BiologicalOde_rates
        ),
        "SCALE_modeled_FVCs_rates": estimated.SCALE_modeled_FVCs_rates,
    }


def _target_state_indices(store: TrainingDataStore, rhs_ode: Any) -> jax.Array:
    """Map measured target columns to physical state columns."""
    n_RMCs = len(rhs_ode.name_modeled_RMCs)
    n_PVs = len(rhs_ode.name_modeled_PVs)
    indices: list[int] = []
    for name in store.name_measured_RMCs:
        indices.append(rhs_ode.name_modeled_RMCs.index(name))
    for name in store.name_measured_PVs:
        indices.append(n_RMCs + rhs_ode.name_modeled_PVs.index(name))

    n_modeled_feeds = len(store.name_modeled_FVCs) + len(store.name_modeled_SVCs)
    feed_start = n_RMCs + n_PVs + 1
    indices.extend(range(feed_start, feed_start + n_modeled_feeds))
    return jnp.asarray(indices, dtype=jnp.int32)


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
    loss_module: UserLossModule | None = None,
) -> tuple[HybridOdeWrapper, dict[str, Any]]:
    """Build a HybridOdeWrapper with the same structure train_collection produces.

    Returns the wrapper plus a dict with the per-process RhsOde map under
    ``per_process_rhs_ode`` so callers can reuse it for evaluation.

    Scales now live on ``reaction_module``; the wrapper validates the shapes
    in its constructor.
    """
    if len(selected_processes) == 0:
        raise ValueError("selected_processes must be non-empty")

    per_process_rhs_ode: dict[str, Any] = {}
    reference_rhs_ode = None
    reference_process_name = selected_processes[0]
    for process_name in store.process_order:
        process = collection.processes[process_name]
        rhs_ode = build_rhs_ode(process)
        per_process_rhs_ode[process_name] = rhs_ode
        if process_name == reference_process_name:
            reference_rhs_ode = rhs_ode
    assert reference_rhs_ode is not None
    for process_name in selected_processes[1:]:
        validate_rhs_ode_compatibility(
            reference_process_name,
            reference_rhs_ode,
            process_name,
            per_process_rhs_ode[process_name],
        )

    target_state_indices = _target_state_indices(store, reference_rhs_ode)

    wrapper = HybridOdeWrapper.from_process(
        reaction_module=reaction_module,
        process=collection.processes[reference_process_name],
        controls=store.get_process(reference_process_name).controls,
        target_state_indices=target_state_indices,
        loss_module=loss_module,
    )
    return wrapper, {"per_process_rhs_ode": per_process_rhs_ode}


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
    name_modeled_FVCs: tuple[str, ...]
    name_modeled_SVCs: tuple[str, ...]
    training_process_names: tuple[str, ...]
    per_process_total_loss: dict[str, float]
    per_process_per_target_loss: dict[str, tuple[float, ...]]
    # Dense prediction trajectories harvested from the batched loss solve
    # (``compute_dense_exports``); the source for predictions.csv + plots.
    dense_exports: dict[str, DenseProcessExport] | None = None


def forward_from_collection(
    collection: BioProcessCollection,
    *,
    model_path: str | Path,
    config: ForwardConfig | None = None,
    custom_py: str | Path | None = None,
    runtime_config: dict[str, Any] | None = None,
    training_process_names: tuple[str, ...] | None = None,
    run_config: RunConfig | None = None,
    custom_module: Any | None = None,
    prediction_grid_n: int = 200,
) -> ForwardResult:
    """Load a trained wrapper and run one forward pass per selected process.

    Mirrors the setup portion of :func:`train_from_collection` — builds the
    TrainingDataStore, reaction module, and scaling exactly as training did —
    so that ``eqx.tree_deserialise_leaves`` has a structurally identical
    template to deserialise into.
    """
    from .serialization import load_trained_wrapper

    cfg = config or ForwardConfig()
    if custom_module is None:
        custom_module = load_custom_module(custom_py)
    if run_config is not None:
        # When the RunConfig was reconstructed from config.json, custom is a raw
        # dict; re-wrap it so hooks get the typed object (config.custom.X).
        from .run_config import reresolve_custom

        run_config = reresolve_custom(run_config, custom_module)
    custom_cfg = (
        run_config
        if run_config is not None
        else resolve_config(custom_module, runtime_config)
    )
    config_targets = None
    if run_config is not None and run_config.data is not None:
        config_targets = run_config.data.targets
    elif isinstance(custom_cfg, dict):
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
        seed=run_config.train.seed if run_config is not None else 0,
        allow_stateful_models=bool(
            run_config is not None and run_config.train.allow_stateful_models
        ),
    )
    # `estimate_all_scales` runs FIRST: its output is plumbed into the
    # reaction-module constructor as SCALE_* kwargs (the module is the
    # single source of truth for scales).
    scale_kwargs = _resolve_estimated_scales(
        custom_module=custom_module,
        collection=collection,
        store=store,
        custom_cfg=custom_cfg,
    )
    reaction_module = _build_reaction_module(
        store=store,
        config=train_like_cfg,
        custom_module=custom_module,
        custom_config=custom_cfg,
        collection=collection,
        scale_kwargs=scale_kwargs,
    )

    loss_module = _build_loss_module(
        store=store,
        config=train_like_cfg,
        custom_module=custom_module,
        custom_config=custom_cfg,
        collection=collection,
    )

    template_wrapper, _extras = _build_template_wrapper(
        store,
        reaction_module=reaction_module,
        collection=collection,
        selected_processes=template_processes,
        loss_module=loss_module,
    )
    loss_names = tuple(loss_module.loss_names)

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

    per_sample_total, per_sample_per_target, dense_exports = compute_dense_exports(
        trained_wrapper,
        store,
        collection,
        eval_processes,
        solver_max_steps=int(cfg.solver_max_steps),
        solver_rtol=float(cfg.solver_rtol),
        solver_atol=float(cfg.solver_atol),
        solver_use_jump_ts=cfg.solver_use_jump_ts,
        prediction_grid_n=int(prediction_grid_n),
    )
    per_process_total = {
        name: float(per_sample_total[i]) for i, name in enumerate(eval_processes)
    }
    per_process_per_target = {
        name: tuple(float(v) for v in per_sample_per_target[i])
        for i, name in enumerate(eval_processes)
    }

    target_column_labels = loss_names

    return ForwardResult(
        trained_wrapper=trained_wrapper,
        store=store,
        process_names=eval_processes,
        target_names=target_column_labels,
        name_modeled_FVCs=tuple(store.name_modeled_FVCs),
        name_modeled_SVCs=tuple(store.name_modeled_SVCs),
        training_process_names=tuple(training_process_names)
        if training_process_names is not None
        else (),
        per_process_total_loss=per_process_total,
        per_process_per_target_loss=per_process_per_target,
        dense_exports=dense_exports,
    )


def forward_plot_losses(
    result: ForwardResult,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Build per-process ``(named_losses, total_loss)`` dicts for plot annotations.

    ``named_losses[p]`` maps each loss name → value; ``total_loss[p]`` is the SUM
    of the named terms (matching the historical per-process plot annotation, which
    differs from the mean reported in ``losses.csv``).
    """
    named = {
        name: dict(zip(result.target_names, result.per_process_per_target_loss[name]))
        for name in result.process_names
    }
    total = {
        name: float(sum(result.per_process_per_target_loss[name]))
        for name in result.process_names
    }
    return named, total


def train_collection(
    store: TrainingDataStore,
    *,
    reaction_module: UserReactionModule,
    loss_module: UserLossModule | None = None,
    collection: BioProcessCollection,
    config: TrainHarnessConfig | None = None,
    optimizer: optax.GradientTransformation | None = None,
) -> TrainHarnessResult:
    """Train one reaction module over one or many processes from one store.

    Scales live on ``reaction_module``; loss terms are produced by
    ``loss_module`` (both attached to the wrapper, partitioned together). When
    ``loss_module`` is None, the default per-target MSE module is used.
    ``optimizer``, when provided (via the ``build_optimizer`` hook), fully owns
    optimizer construction; otherwise the default ``_build_optimizer`` chain is
    used.
    """
    cfg = config or TrainHarnessConfig()
    _require_stateful_opt_in(reaction_module, cfg.allow_stateful_models)
    if loss_module is None:
        loss_module = DefaultLossModule(target_names=_loss_target_labels(store))
    effective_batched_loss_fn = _BATCHED_LOSS_FN
    selected_processes = _ensure_process_names(store, cfg.process_names)

    effective_batch_size, batches_per_epoch, total_updates = derive_update_budget(
        cfg, selected_process_count=len(selected_processes)
    )
    selected_process_indices = jnp.asarray(
        [store.process_order.index(name) for name in selected_processes],
        dtype=jnp.int32,
    )
    batch_index_stream = _build_batch_index_stream(
        selected_process_indices=selected_process_indices,
        epochs=int(cfg.epochs),
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
    per_process_rhs_ode = {}
    reference_rhs_ode = None
    reference_process_name = selected_processes[0]
    for process_name in store.process_order:
        process = collection.processes[process_name]
        rhs_ode = build_rhs_ode(process)
        per_process_rhs_ode[process_name] = rhs_ode
        if process_name == reference_process_name:
            reference_rhs_ode = rhs_ode

    assert reference_rhs_ode is not None
    for process_name in selected_processes[1:]:
        validate_rhs_ode_compatibility(
            reference_process_name,
            reference_rhs_ode,
            process_name,
            per_process_rhs_ode[process_name],
        )

    # Loss target columns map into the save/loss physical state
    # [modeled_RMCs | modeled_PVs | V | modeled_FVCs_cumulative].
    # V is in the state but not a loss target.
    target_state_indices = _target_state_indices(store, reference_rhs_ode)

    # Per-target labels come straight from the loss module — one panel/column
    # per named loss term, in declared order.
    _target_labels = list(loss_module.loss_names)

    # Build wrapper from reference process. Scales now live on
    # ``reaction_module``; the wrapper validates SCALE_* shapes in its
    # constructor. The loss module rides along (partitioned with the wrapper).
    wrapper = HybridOdeWrapper.from_process(
        reaction_module=reaction_module,
        process=collection.processes[reference_process_name],
        controls=store.get_process(reference_process_name).controls,
        target_state_indices=target_state_indices,
        loss_module=loss_module,
    )

    # Stack per-process Cin arrays: [n_store_processes, n_feeds, n_species]
    all_Cin = []
    all_Cin_modeled = []
    for process_name in store.process_order:
        rhs_ode = per_process_rhs_ode[process_name]
        all_Cin.append(rhs_ode.Cin_controlled_FVCs)
        all_Cin_modeled.append(rhs_ode.Cin_modeled_FVCs)
    batched_Cin = jnp.stack(all_Cin)
    batched_Cin_modeled = jnp.stack(all_Cin_modeled)

    # Partition the WHOLE wrapper so the loss module's trainable_field() leaves
    # are optimized alongside the reaction module's. Untagged leaves (controls,
    # rhs_ode Cin, indices) stay frozen exactly as before.
    trainable_params, trainable_static = partition_trainable(wrapper)
    print_rhs_ode(collection)
    sys.stdout.flush()
    print_trainable_structure(reaction_module)
    print_trainable_structure(loss_module, title="UserLossModule")
    print_reaction_schema(wrapper)
    if optimizer is None:
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
        jnp.asarray(0, dtype=jnp.int32),
    )

    def _make_batched_step():
        def _step_fn(
            current_wrapper: HybridOdeWrapper,
            current_trainable_params: Any,
            current_optimizer_state: Any,
            current_batch,
            step: jax.Array,
        ):
            jump_ts_rows = None
            if cfg.solver_use_jump_ts:
                jump_ts_rows = clamp_padded_time_rows(
                    store.controls_store.jump_ts[current_batch.process_indices],
                    store.controls_store.jump_ts_lengths[current_batch.process_indices],
                )

            del current_wrapper  # whole-wrapper partition: combine reconstructs it

            def _loss_fn(trainable_params: Any) -> jax.Array:
                candidate_wrapper = eqx.combine(trainable_params, trainable_static)
                # `*_pred` swallows the prediction outputs (unused in training);
                # the trailing element is the per-sample fail_time (see
                # build_batched_loss_fn).
                (
                    total_loss,
                    per_sample_per_target,
                    per_sample,
                    *_pred,
                    per_sample_fail,
                ) = (
                    effective_batched_loss_fn(
                        candidate_wrapper,
                        current_batch,
                        batch_controls,
                        batched_Cin,
                        batched_Cin_modeled,
                        jump_ts_rows,
                        max_solver_steps=int(cfg.solver_max_steps),
                        solver_rtol=float(cfg.solver_rtol),
                        solver_atol=float(cfg.solver_atol),
                        step=step,
                    )
                )
                per_target = jnp.mean(per_sample_per_target, axis=0)
                total_loss, per_target, per_sample = _validate_batched_loss_outputs(
                    total_loss,
                    per_target,
                    per_sample,
                    n_targets=len(loss_module.loss_names),
                    batch_size=int(current_batch.process_indices.shape[0]),
                )
                return total_loss, (per_target, per_sample, per_sample_fail)

            (loss, (per_target_loss, per_sample_loss, per_sample_fail)), grads = (
                eqx.filter_value_and_grad(
                    _loss_fn,
                    has_aux=True,
                )(current_trainable_params)
            )
            grad_norm = optax.tree.norm(grads)
            updates, next_optimizer_state = optimizer.update(
                grads,
                current_optimizer_state,
                params=current_trainable_params,
            )
            trainable_updated = eqx.apply_updates(current_trainable_params, updates)
            wrapper_updated = eqx.combine(trainable_updated, trainable_static)
            return (
                wrapper_updated,
                trainable_updated,
                loss,
                per_target_loss,
                per_sample_loss,
                next_optimizer_state,
                grad_norm,
                per_sample_fail,
            )

        return eqx.filter_jit(_step_fn)

    def _make_sharded_step():
        """Data-parallel step: shard the batch across local devices via pmap,
        psum/all_gather to the batch mean, then apply the optimiser update.
        Same math as the vmap step up to float32 cross-device reduction order.
        Pseudo-space variant: y0 stays RAW (wrapper builds the pseudobatch state).
        Uses min(devices, batch) devices, so a device count exceeding the process
        count shards across the processes instead of collapsing to one device."""
        bs = int(effective_batch_size)
        n_dev = min(jax.local_device_count(), bs)
        devices = jax.local_devices()[:n_dev]
        pad_n = (-bs) % n_dev
        padded = bs + pad_n
        per_dev = padded // n_dev
        weight_full = jnp.concatenate(
            [jnp.ones(bs, dtype=jnp.float64), jnp.zeros(pad_n, dtype=jnp.float64)]
        )

        def _pad(x):
            return (
                x
                if pad_n == 0
                else jnp.concatenate([x, jnp.repeat(x[-1:], pad_n, axis=0)], axis=0)
            )

        def _shard(x):
            return x.reshape((n_dev, per_dev) + x.shape[1:])

        weight_sharded = _shard(weight_full)
        use_jump = bool(cfg.solver_use_jump_ts)
        max_steps = int(cfg.solver_max_steps)
        rtol = float(cfg.solver_rtol)
        atol = float(cfg.solver_atol)

        @partial(
            jax.pmap,
            axis_name="bp_dev",
            devices=devices,
            in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, (0 if use_jump else None), None),
        )
        def _pgrad(params, pi, tm, ym, mk, nm, y0, cin, cinm, wt, jt, step):
            def _local(p):
                wrapper = eqx.combine(p, trainable_static)
                module = wrapper.reaction_module
                scale_targets = module.SCALE_state[wrapper.target_state_indices]
                SCL_ym = ym / scale_targets[None, None, :]
                tmc = clamp_padded_time_rows(tm, nm)

                def _one(pidx, t_row, scl_ym, mask, n_meas, raw_y0, ci, cim, jts):
                    return evaluate_one_sample_loss(
                        wrapper,
                        batch_controls,
                        pidx,
                        t_row,
                        scl_ym,
                        mask,
                        n_meas,
                        raw_y0,
                        ci,
                        cim,
                        jts,
                        max_solver_steps=max_steps,
                        solver_rtol=rtol,
                        solver_atol=atol,
                        step=step,
                    )

                totl, pert, faild = jax.vmap(
                    _one, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, (0 if use_jump else None))
                )(pi, tmc, SCL_ym, mk, nm, y0, cin, cinm, jt)
                return jnp.sum(totl * wt), (pert, totl, faild)

            (_l, (pert, totl, faild)), grads = eqx.filter_value_and_grad(
                _local, has_aux=True
            )(params)
            loss = jax.lax.psum(_l, "bp_dev") / bs
            grads = jax.tree_util.tree_map(
                lambda g: jax.lax.psum(g, "bp_dev") / bs, grads
            )
            per_target = (
                jax.lax.psum(jnp.sum(pert * wt[:, None], axis=0), "bp_dev") / bs
            )
            per_sample = jax.lax.all_gather(totl, "bp_dev")
            per_sample_fail = jax.lax.all_gather(faild, "bp_dev")
            return loss, grads, per_target, per_sample, per_sample_fail

        def _step_fn(
            current_wrapper,
            current_trainable_params,
            current_optimizer_state,
            current_batch,
            step,
        ):
            del current_wrapper
            jump_ts_rows = None
            if use_jump:
                jump_ts_rows = clamp_padded_time_rows(
                    store.controls_store.jump_ts[current_batch.process_indices],
                    store.controls_store.jump_ts_lengths[current_batch.process_indices],
                )
            cin = batched_Cin[current_batch.process_indices]
            cinm = batched_Cin_modeled[current_batch.process_indices]
            ins = [
                _shard(_pad(a))
                for a in (
                    current_batch.process_indices,
                    current_batch.t_measured,
                    current_batch.y_measured,
                    current_batch.mask_measured,
                    current_batch.n_measured,
                    current_batch.y0_measured,
                    cin,
                    cinm,
                )
            ]
            jt = _shard(_pad(jump_ts_rows)) if jump_ts_rows is not None else None
            loss, grads, per_target, per_sample, per_sample_fail = _pgrad(
                current_trainable_params,
                ins[0],
                ins[1],
                ins[2],
                ins[3],
                ins[4],
                ins[5],
                ins[6],
                ins[7],
                weight_sharded,
                jt,
                step,
            )
            loss = loss[0]
            grads = jax.tree_util.tree_map(lambda g: g[0], grads)
            per_target_loss = per_target[0]
            per_sample_loss = per_sample[0].reshape(-1)[:bs]
            per_sample_fail = per_sample_fail[0].reshape(-1)[:bs]
            grad_norm = optax.tree.norm(grads)
            updates, next_optimizer_state = optimizer.update(
                grads, current_optimizer_state, params=current_trainable_params
            )
            trainable_updated = eqx.apply_updates(current_trainable_params, updates)
            wrapper_updated = eqx.combine(trainable_updated, trainable_static)
            return (
                wrapper_updated,
                trainable_updated,
                loss,
                per_target_loss,
                per_sample_loss,
                next_optimizer_state,
                grad_norm,
                per_sample_fail,
            )

        return _step_fn

    def _make_gspmd_step():
        """Data-parallel step via ``jax.jit`` + GSPMD auto-sharding — the modern,
        future-proof replacement for the deprecated ``jax.pmap``. The batch axis is
        sharded across an Auto-axis ``Mesh`` (``device_put`` with ``P('bp_dev')``);
        params/optimiser-state are replicated. The single full vmap-over-batch grad +
        optimiser update runs under one ``jit``, and XLA's SPMD partitioner splits the
        per-sample solves across devices and inserts the all-reduce for the
        batch-mean loss/grad automatically — no manual ``psum``/``all_gather``.

        Why GSPMD and not ``jax.shard_map``: on jax>=0.10 ``shard_map`` runs
        the body in *manual* sharding mode, which trips
        ``assert not hlo_sharding.is_manual()`` inside diffrax's nested
        ``eqx.filter_eval_shape`` solver loop; explicit-sharding ``jit``
        likewise breaks on diffrax's internal ``select``s. Auto-axis GSPMD
        keeps shardings out of the per-op type system, so the diffrax adjoint
        composes cleanly — and it needs **no** equinox/diffrax monkeypatch
        (unlike the pmap path)."""
        from jax.sharding import PartitionSpec as P, NamedSharding, AxisType

        bs = int(effective_batch_size)
        n_dev = min(jax.local_device_count(), bs)
        devices = jax.local_devices()[:n_dev]
        mesh = jax.make_mesh(
            (n_dev,), ("bp_dev",), axis_types=(AxisType.Auto,), devices=devices
        )
        S_batch = NamedSharding(mesh, P("bp_dev"))  # shard the leading batch axis
        S_repl = NamedSharding(mesh, P())  # replicated
        pad_n = (-bs) % n_dev
        weight_full = jnp.concatenate(
            [jnp.ones(bs, dtype=jnp.float64), jnp.zeros(pad_n, dtype=jnp.float64)]
        )

        def _pad(x):
            return (
                x
                if pad_n == 0
                else jnp.concatenate([x, jnp.repeat(x[-1:], pad_n, axis=0)], axis=0)
            )

        use_jump = bool(cfg.solver_use_jump_ts)
        max_steps = int(cfg.solver_max_steps)
        rtol = float(cfg.solver_rtol)
        atol = float(cfg.solver_atol)

        @eqx.filter_jit
        def _full_step(
            params, opt_state, pi, tm, ym, mk, nm, y0, cin, cinm, wt, jt, step
        ):
            # Inside-jit sharding constraints (per the equinox autoparallelism
            # tutorial): emits jax.lax.with_sharding_constraint so XLA keeps
            # params/opt replicated and the batch axis sharded *through* the
            # computation, rather than replicating it.
            params, opt_state = eqx.filter_shard((params, opt_state), S_repl)
            pi, tm, ym, mk, nm, y0, cin, cinm, wt, jt = eqx.filter_shard(
                (pi, tm, ym, mk, nm, y0, cin, cinm, wt, jt), S_batch
            )

            def _local(p):
                wrapper = eqx.combine(p, trainable_static)
                module = wrapper.reaction_module
                scale_targets = module.SCALE_state[wrapper.target_state_indices]
                SCL_ym = ym / scale_targets[None, None, :]
                tmc = clamp_padded_time_rows(tm, nm)

                def _one(pidx, t_row, scl_ym, mask, n_meas, raw_y0, ci, cim, jts):
                    return evaluate_one_sample_loss(
                        wrapper,
                        batch_controls,
                        pidx,
                        t_row,
                        scl_ym,
                        mask,
                        n_meas,
                        raw_y0,
                        ci,
                        cim,
                        jts,
                        max_solver_steps=max_steps,
                        solver_rtol=rtol,
                        solver_atol=atol,
                        step=step,
                    )

                totl, pert, faild = jax.vmap(
                    _one, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, (0 if use_jump else None))
                )(pi, tmc, SCL_ym, mk, nm, y0, cin, cinm, jt)
                # Constrain the per-sample (batch-axis) results to stay sharded. NOTE:
                # measured to NOT help — even with this hint XLA still replicates the
                # data-dependent diffrax while_loops rather than partitioning the
                # vmap, so
                # the GSPMD path stays ~60x slower than pmap. Kept as correct practice.
                totl, pert, faild = eqx.filter_shard((totl, pert, faild), S_batch)
                # weighted mean: padding rows carry weight 0; XLA all-reduces it
                # (and grad).
                return jnp.sum(totl * wt) / bs, (pert, totl, faild)

            (loss, (pert, totl, faild)), grads = eqx.filter_value_and_grad(
                _local, has_aux=True
            )(params)
            per_target = jnp.sum(pert * wt[:, None], axis=0) / bs
            updates, next_opt_state = optimizer.update(grads, opt_state, params=params)
            params_next = eqx.apply_updates(params, updates)
            grad_norm = optax.tree.norm(grads)
            params_next, next_opt_state = eqx.filter_shard(
                (params_next, next_opt_state), S_repl
            )
            return (
                params_next,
                next_opt_state,
                loss,
                per_target,
                totl,
                grad_norm,
                faild,
            )

        def _step_fn(
            current_wrapper,
            current_trainable_params,
            current_optimizer_state,
            current_batch,
            step,
        ):
            del current_wrapper
            jump_ts_rows = None
            if use_jump:
                jump_ts_rows = clamp_padded_time_rows(
                    store.controls_store.jump_ts[current_batch.process_indices],
                    store.controls_store.jump_ts_lengths[current_batch.process_indices],
                )
            cin = batched_Cin[current_batch.process_indices]
            cinm = batched_Cin_modeled[current_batch.process_indices]
            with jax.set_mesh(
                mesh
            ):  # jax>=0.10 needs an active mesh for the sharded jit
                sharded = [
                    jax.device_put(_pad(a), S_batch)
                    for a in (
                        current_batch.process_indices,
                        current_batch.t_measured,
                        current_batch.y_measured,
                        current_batch.mask_measured,
                        current_batch.n_measured,
                        current_batch.y0_measured,
                        cin,
                        cinm,
                    )
                ]
                wt = jax.device_put(weight_full, S_batch)
                jt = (
                    jax.device_put(_pad(jump_ts_rows), S_batch)
                    if jump_ts_rows is not None
                    else None
                )
                params_r = jax.device_put(current_trainable_params, S_repl)
                opt_r = jax.device_put(current_optimizer_state, S_repl)
                step_r = jax.device_put(jnp.asarray(step), S_repl)
                (
                    trainable_updated,
                    next_optimizer_state,
                    loss,
                    per_target_loss,
                    per_sample,
                    grad_norm,
                    per_sample_fail,
                ) = _full_step(
                    params_r,
                    opt_r,
                    sharded[0],
                    sharded[1],
                    sharded[2],
                    sharded[3],
                    sharded[4],
                    sharded[5],
                    sharded[6],
                    sharded[7],
                    wt,
                    jt,
                    step_r,
                )
            per_sample_loss = per_sample.reshape(-1)[:bs]
            per_sample_fail = per_sample_fail.reshape(-1)[:bs]
            wrapper_updated = eqx.combine(trainable_updated, trainable_static)
            return (
                wrapper_updated,
                trainable_updated,
                loss,
                per_target_loss,
                per_sample_loss,
                next_optimizer_state,
                grad_norm,
                per_sample_fail,
            )

        return _step_fn

    _n_local_devices = jax.local_device_count()
    _n_shard = min(_n_local_devices, int(effective_batch_size))
    _use_sharded = _n_shard > 1
    # pmap stays the DEFAULT sharded path: it is the only fast option on jax>=0.10.
    # shard_map (manual mode) crashes inside diffrax's nested filter_eval_shape
    # (`assert not hlo_sharding.is_manual()`), and GSPMD auto-sharding (BP_GSPMD=1) is
    # correct + patch-free but ~60x slower because XLA cannot partition the
    # data-dependent ODE solve (it replicates the per-device work). pmap needs the
    # equinox closure-convert patch on jax>=0.10; GSPMD needs no patch.
    _use_gspmd = os.environ.get("BP_GSPMD", "") not in ("", "0", "false", "False")
    if _use_sharded and _use_gspmd:
        _make_step = _make_gspmd_step
    elif _use_sharded:
        _make_step = _make_sharded_step
    else:
        _make_step = _make_batched_step
    if _use_sharded:
        logger.info(
            "training sharded across %d local devices (batch=%d) via %s",
            _n_shard,
            int(effective_batch_size),
            "gspmd" if _use_gspmd else "pmap",
        )
    step_fn = _make_step()
    rebuild_count = 0

    logger.info(
        "train harness setup: processes=%s, targets=%s source=%s epochs=%d "
        "batch_size=%d optimizer=%s lr=%s grad_clip=%s",
        list(selected_processes),
        list(store.name_measured),
        (
            "combined"
            if store.name_measured_RMCs and store.name_measured_PVs
            else "reactor_components"
            if store.name_measured_RMCs
            else "process_variables"
        ),
        cfg.epochs,
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
        _warmup_grad_norm,
        _warmup_fail,
    ) = step_fn(
        wrapper,
        trainable_params,
        optimizer_state,
        warmup_batch,
        jnp.asarray(0, dtype=jnp.int32),
    )
    jax.block_until_ready(warmup_loss)
    warmup_compile_seconds = time.perf_counter() - warmup_t0
    logger.info(
        "JIT compilation done in %.1fs, warmup loss=%.6g",
        warmup_compile_seconds,
        float(warmup_loss),
    )

    checkpoint_enabled = cfg.checkpoint_dir is not None
    checkpoint_boundaries = (
        _checkpoint_update_boundaries(
            cfg.checkpoint_every,
            batches_per_epoch=batches_per_epoch,
            total_updates=total_updates,
        )
        if checkpoint_enabled
        else frozenset()
    )
    plotter = BackgroundPlotter() if (checkpoint_enabled and cfg.plots) else None
    checkpoint_writer = (
        CheckpointWriter(
            Path(cfg.checkpoint_dir),
            plotter=plotter,
            plots_enabled=cfg.plots,
            prepared_src=cfg.prepared_path,
        )
        if cfg.checkpoint_dir is not None
        else None
    )
    loss_so_far: list[float] = []
    per_target_loss_so_far: list[tuple[float, ...]] = []
    grad_norm_so_far: list[float] = []
    holdout_loss_so_far: dict[int, float] = {}
    holdout_per_target_so_far: dict[int, tuple[float, ...]] = {}

    holdout_batch = None
    holdout_jump_ts_rows = None
    if cfg.holdout_processes:
        unknown = [n for n in cfg.holdout_processes if n not in store.process_order]
        if unknown:
            raise ValueError(
                f"holdout_processes contains unknown names: {unknown}; "
                f"available={tuple(store.process_order)}"
            )
        holdout_indices = jnp.asarray(
            [store.process_order.index(name) for name in cfg.holdout_processes],
            dtype=jnp.int32,
        )
        holdout_batch = store.gather_batch(holdout_indices)
        if cfg.solver_use_jump_ts:
            holdout_jump_ts_rows = clamp_padded_time_rows(
                store.controls_store.jump_ts[holdout_batch.process_indices],
                store.controls_store.jump_ts_lengths[holdout_batch.process_indices],
            )

    def _evaluate_holdout(step: int):
        if holdout_batch is None:
            return None, None
        total, per_sample_per_target, _per_sample, *_ = effective_batched_loss_fn(
            wrapper,
            holdout_batch,
            batch_controls,
            batched_Cin,
            batched_Cin_modeled,
            holdout_jump_ts_rows,
            max_solver_steps=int(cfg.solver_max_steps),
            solver_rtol=float(cfg.solver_rtol),
            solver_atol=float(cfg.solver_atol),
            step=jnp.asarray(step, dtype=jnp.int32),
        )
        jax.block_until_ready(total)
        per_target = tuple(
            float(value)
            for value in np.asarray(jnp.mean(per_sample_per_target, axis=0)).tolist()
        )
        return float(total), per_target

    checkpoint_per_target_losses = None
    checkpoint_dense_exports = None

    def _render_predictions(path: Path):
        nonlocal checkpoint_per_target_losses, checkpoint_dense_exports
        _, checkpoint_per_target_losses, checkpoint_dense_exports = (
            compute_dense_exports(
                wrapper,
                store,
                collection,
                selected_processes,
                solver_max_steps=int(cfg.solver_max_steps),
                solver_rtol=float(cfg.solver_rtol),
                solver_atol=float(cfg.solver_atol),
                solver_use_jump_ts=bool(cfg.solver_use_jump_ts),
            )
        )
        export_predictions_csv(
            wrapper,
            checkpoint_dense_exports,
            path,
            process_names=selected_processes,
        )
        if not cfg.plots:
            return None
        return build_process_plot_data(
            wrapper,
            collection,
            store,
            checkpoint_dense_exports,
            selected_processes,
            training_process_names=selected_processes,
        )

    try:
        with RunLogger(
            log_process_losses=cfg.log_process_losses,
            metrics_csv=cfg.metrics_csv,
            metrics_jsonl=cfg.metrics_jsonl,
            log_decimals=cfg.log_decimals,
        ) as run_log:
            run_log.start(
                target_names=_target_labels,
                process_names=selected_processes,
                total_updates=total_updates,
                compile_warmup_seconds=warmup_compile_seconds,
            )
            epoch_losses: list[float] = []
            epoch_training_seconds = 0.0
            for step_index in range(total_updates):
                step = step_index + 1
                epoch = step_index // batches_per_epoch + 1
                batch_in_epoch = step_index % batches_per_epoch + 1
                batch = store.gather_batch(batch_index_stream[step_index])
                current_signature = summarize_train_step_input_signature(
                    wrapper,
                    trainable_params,
                    optimizer_state,
                    batch,
                    jnp.asarray(step, dtype=jnp.int32),
                )
                if current_signature != train_step_input_signature:
                    step_fn = _make_step()
                    rebuild_count += 1
                    run_log.record_rebuild(step)

                t0 = time.perf_counter()
                (
                    wrapper,
                    trainable_params,
                    loss,
                    per_target_loss,
                    per_sample_loss,
                    optimizer_state,
                    grad_norm,
                    per_sample_fail,
                ) = step_fn(
                    wrapper,
                    trainable_params,
                    optimizer_state,
                    batch,
                    jnp.asarray(step, dtype=jnp.int32),
                )
                jax.block_until_ready(loss)
                step_dt = time.perf_counter() - t0
                epoch_training_seconds += step_dt
                epoch_losses.append(float(loss))

                per_target_values = tuple(
                    float(value) for value in np.asarray(per_target_loss).tolist()
                )
                per_process_values = tuple(
                    float(value) for value in np.asarray(per_sample_loss).tolist()
                )
                batch_names = tuple(
                    store.process_order[int(index)]
                    for index in np.asarray(batch.process_indices).tolist()
                )
                # A finite fail_time flags a sample whose ODE solve bailed mid-
                # trajectory (post-failure points were dropped from the loss). Name
                # the affected processes so the logger can report how often it happens.
                failed_process_names = tuple(
                    name
                    for name, ft in zip(
                        batch_names, np.asarray(per_sample_fail).reshape(-1).tolist()
                    )
                    if np.isfinite(ft)
                )
                loss_so_far.append(float(loss))
                per_target_loss_so_far.append(per_target_values)
                grad_norm_so_far.append(float(grad_norm))

                holdout_loss, holdout_per_target = (None, None)
                if step in checkpoint_boundaries:
                    holdout_loss, holdout_per_target = _evaluate_holdout(step)
                    if holdout_loss is not None:
                        holdout_loss_so_far[step] = holdout_loss
                        holdout_per_target_so_far[step] = holdout_per_target

                epoch_end = batch_in_epoch == batches_per_epoch
                epoch_mean_loss = (
                    sum(epoch_losses) / len(epoch_losses) if epoch_end else None
                )
                epoch_time_seconds = epoch_training_seconds if epoch_end else None
                run_log.record_step(
                    StepRecord(
                        step=step,
                        total_updates=total_updates,
                        epoch=epoch,
                        batch_in_epoch=batch_in_epoch,
                        samples_seen=step * effective_batch_size,
                        mean_loss=float(loss),
                        per_target_loss=per_target_values,
                        per_process_loss=per_process_values,
                        target_names=tuple(_target_labels),
                        process_names=batch_names,
                        step_dt=step_dt,
                        rebuild_count=rebuild_count,
                        holdout_loss=holdout_loss,
                        holdout_label=(
                            cfg.holdout_label if holdout_loss is not None else None
                        ),
                        epoch_mean_loss=epoch_mean_loss,
                        epoch_time_seconds=epoch_time_seconds,
                        grad_norm=float(grad_norm),
                        failed_process_names=failed_process_names,
                    )
                )
                if epoch_end:
                    epoch_losses = []
                    epoch_training_seconds = 0.0

                if checkpoint_writer is not None and step in checkpoint_boundaries:
                    checkpoint_writer.write(
                        step=step,
                        samples_seen=step * effective_batch_size,
                        wrapper=wrapper,
                        opt_state=optimizer_state,
                        mean_loss=float(loss),
                        holdout_loss=holdout_loss,
                        render_predictions_fn=_render_predictions,
                        loss_by_step=loss_so_far,
                        grad_norm_by_step=grad_norm_so_far,
                        per_target_loss_by_step=per_target_loss_so_far,
                        target_names=tuple(_target_labels),
                        holdout_loss_by_step=holdout_loss_so_far or None,
                        holdout_per_target_by_step=holdout_per_target_so_far or None,
                        holdout_label=(
                            cfg.holdout_label if holdout_loss_so_far else None
                        ),
                    )

            history = run_log.finalize()

            # Run-root final-model outputs: predictions.csv + the complete final plot
            # set (per-process simulations, loss curve, grad-norm curve) written
            # directly to the run dir — synchronously, so they are reliable and not
            # subject to the per-checkpoint background renderer.
            if cfg.checkpoint_dir is not None:
                run_dir = Path(cfg.checkpoint_dir).parent
                try:
                    if (
                        checkpoint_per_target_losses is None
                        or checkpoint_dense_exports is None
                    ):
                        raise RuntimeError("final checkpoint exports are unavailable")
                    _final_per_target = checkpoint_per_target_losses
                    _final_dense = checkpoint_dense_exports
                    export_predictions_csv(
                        wrapper,
                        _final_dense,
                        run_dir / "predictions.csv",
                        process_names=selected_processes,
                    )
                    if bool(cfg.plots):
                        named = {
                            p: dict(zip(_target_labels, _final_per_target[i]))
                            for i, p in enumerate(selected_processes)
                        }
                        total = {
                            p: float(sum(_final_per_target[i]))
                            for i, p in enumerate(selected_processes)
                        }
                        plot_process_simulations(
                            wrapper,
                            collection,
                            store,
                            run_dir,
                            _final_dense,
                            process_names=selected_processes,
                            training_process_names=selected_processes,
                            per_process_named_losses=named,
                            per_process_total_loss=total,
                        )
                        if loss_so_far:
                            plot_loss_curve(
                                list(loss_so_far),
                                run_dir / "loss_curve.png",
                                title=f"Training loss (through step {total_updates})",
                                per_target_loss_by_step=list(per_target_loss_so_far),
                                target_names=tuple(_target_labels),
                                monitor_loss_by_step=(
                                    dict(holdout_loss_so_far)
                                    if holdout_loss_so_far
                                    else None
                                ),
                                monitor_per_target_by_step=(
                                    dict(holdout_per_target_so_far)
                                    if holdout_per_target_so_far
                                    else None
                                ),
                                monitor_label=(
                                    cfg.holdout_label if holdout_loss_so_far else None
                                ),
                            )
                        if grad_norm_so_far:
                            plot_grad_norm_curve(
                                list(grad_norm_so_far),
                                run_dir / "grad_norm_curve.png",
                                title=f"Gradient norm (through step {total_updates})",
                            )
                except Exception:  # noqa: BLE001 - the trained model is already saved
                    logger.exception("failed to write run-root final outputs")

    finally:
        if plotter is not None:
            plotter.close()

    return TrainHarnessResult(
        trained_wrapper=wrapper,
        compile_warmup_seconds=float(warmup_compile_seconds),
        train_step_input_signature=train_step_input_signature,
        updates_completed=total_updates,
        **history,
    )


def train_from_collection(
    collection: BioProcessCollection,
    *,
    config: TrainHarnessConfig | None = None,
    custom_py: str | Path | None = None,
    runtime_config: dict[str, Any] | None = None,
    run_config: RunConfig | None = None,
    custom_module: Any | None = None,
) -> TrainHarnessResult:
    """Train from an already-loaded process collection with optional custom hooks.

    Use this entry point when the caller has already deserialized the
    prepared JSON (e.g. the CLI, which also needs the collection for
    post-training plotting). For path-based callers,
    :func:`train_from_prepared_json` is a thin wrapper that loads the
    collection and delegates here.
    """
    cfg = config or TrainHarnessConfig()
    if custom_module is None:
        custom_module = load_custom_module(custom_py)
    custom_cfg = (
        run_config
        if run_config is not None
        else resolve_config(custom_module, runtime_config)
    )
    config_targets = None
    if run_config is not None and run_config.data is not None:
        config_targets = run_config.data.targets
    elif isinstance(custom_cfg, dict):
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
        _resolved_source = (
            "combined"
            if store.name_measured_RMCs and store.name_measured_PVs
            else "reactor_components"
            if store.name_measured_RMCs
            else "process_variables"
        )
        warnings.warn(
            "No training targets specified in run config data.targets. "
            f"Defaulting to target_source={_resolved_source!r} measured "
            f"targets: {tuple(store.name_measured)}.",
            stacklevel=2,
        )
    logger.info("Training targets: %s", tuple(store.name_measured))

    selected_processes = _ensure_process_names(store, cfg.process_names)
    train_cfg = dataclasses.replace(
        cfg,
        process_names=selected_processes,
        target_variable_order=effective_target_order,
    )
    # estimate_all_scales runs FIRST: its return flattens into SCALE_* kwargs
    # feeding build_reaction_module (the module is the single source of
    # truth for scales).
    scale_kwargs = _resolve_estimated_scales(
        custom_module=custom_module,
        collection=collection,
        store=store,
        custom_cfg=custom_cfg,
    )
    reaction_module = _build_reaction_module(
        store=store,
        config=train_cfg,
        custom_module=custom_module,
        custom_config=custom_cfg,
        collection=collection,
        scale_kwargs=scale_kwargs,
    )
    _, _, total_updates = derive_update_budget(
        train_cfg, selected_process_count=len(selected_processes)
    )
    optimizer, train_cfg = build_optimizer_for_run(
        custom_module=custom_module,
        custom_cfg=custom_cfg,
        train_cfg=train_cfg,
        total_updates=total_updates,
    )

    loss_module = _build_loss_module(
        store=store,
        config=train_cfg,
        custom_module=custom_module,
        custom_config=custom_cfg,
        collection=collection,
    )

    return train_collection(
        store,
        reaction_module=reaction_module,
        loss_module=loss_module,
        collection=collection,
        config=train_cfg,
        optimizer=optimizer,
    )


def train_harness_config_from_run_config(
    cfg: RunConfig,
    *,
    run_dir: Path,
) -> TrainHarnessConfig:
    """Map a typed :class:`RunConfig` + run-directory layout to the harness
    config, wiring the FAIR run-dir artifact paths (metrics.csv, checkpoints/,
    observations.csv) and the checkpoint/output/logging sections. Shared by the
    CLI train and LOO paths.
    """
    data = cfg.data
    return TrainHarnessConfig(
        process_names=data.processes if data is not None else None,
        target_variable_order=data.targets if data is not None else None,
        target_source=data.target_source if data is not None else TARGET_SOURCE_AUTO,
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        shuffle_batches=cfg.train.shuffle,
        batch_seed=cfg.train.batch_seed,
        optimizer_name=cfg.train.optimizer,
        learning_rate=cfg.train.learning_rate,
        grad_clip_norm=cfg.train.grad_clip_norm,
        seed=cfg.train.seed,
        solver_max_steps=cfg.solver.max_steps,
        solver_rtol=cfg.solver.rtol,
        solver_atol=cfg.solver.atol,
        solver_use_jump_ts=cfg.solver.jump_ts,
        log_decimals=cfg.logging.decimals,
        metrics_csv=str(Path(run_dir) / "metrics.csv"),
        checkpoint_dir=Path(run_dir) / "checkpoints",
        checkpoint_every=cfg.checkpoint.every,
        plots=cfg.output.plots,
        prepared_path=cfg.data.prepared if cfg.data is not None else None,
        allow_stateful_models=cfg.train.allow_stateful_models,
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
    collection = load_process_collection(Path(prepared_json))
    return train_from_collection(
        collection,
        config=config,
        custom_py=custom_py,
        runtime_config=runtime_config,
    )
