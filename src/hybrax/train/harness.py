"""The training and forward-evaluation harness: the layer above
``trainer.py``'s single batch step.

This module provides:

- ``train_collection`` / ``train_from_collection`` / ``train_from_prepared_json`` —
  train one reaction/loss module pair over a batch stream, at increasing
  levels of convenience (a prepared store, a raw collection, a prepared JSON
  file on disk).
- ``model_predict`` / ``forward_from_collection`` / ``evaluate_trained_wrapper`` —
  run a trained model forward over new or held-out data, without touching the
  optimizer.
- ``prepare_training`` / ``prepare_training_from_runtime_artifact`` /
  ``train_harness_config_from_run_config`` — build every hook-visible object
  (reaction module, loss module, optimizer, harness config) from a
  :class:`~hybrax.train.run_config.RunConfig` and a runtime artifact, the
  path the CLI and LOO folds both go through.

Batching, the JIT-compiled train step, and per-sample loss evaluation live in
``trainer.py``; this module owns everything around that step — process
selection, checkpointing, logging, and the public entry points.
"""

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
from hybrax.format.dataclasses import BioProcessCollection
from hybrax.format.inspect import print_rhs_ode
from hybrax.format.mechanistic import RhsOde
from hybrax.format.serialization import load_process_collection

from .checkpointing import CheckpointWriter
from .defaults import (
    DefaultLossModule,
    _default_scale_kwargs,
    default_build_loss_module,
    default_build_reaction_module,
)
from .inspect import print_reaction_schema, print_trainable_structure
from .model_api import (
    EstimatedScales,
    UserLossModule,
    UserReactionModule,
    _as_scaler,
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
    replace_rhs_ode_process_matrices,
)
from .run_config import RunConfig
from .runtime_artifact import RuntimeArtifact
from .runtime_context import (
    ProducerCollectionData,
    RuntimeDataContext,
    canonical_training_parents,
    original_parent_processes,
)
from .utils import (
    get_hook,
    load_custom_module,
    resolve_config,
    split_hooks_by_customization,
)
from .wrapper import HybridOdeWrapper, validate_rhs_ode_compatibility
from .postprocessing import (
    DenseProcessExport,
    dense_exports_from_save_outputs,
    plot_grad_norm_curve,
    plot_loss_curve,
)

# Single batched loss fn: module-agnostic, reads wrapper.loss_module at call time.
_BATCHED_LOSS_FN = build_batched_loss_fn()
# JIT'd once at import; reused by holdout evaluation and dense exports. Each
# distinct fixed batch shape compiles once, then parameter updates reuse it.
_BATCHED_LOSS_FN_JIT = eqx.filter_jit(_BATCHED_LOSS_FN)

logger = logging.getLogger(__name__)

_EVALUATION_BATCH_SIZE = 32


def _iter_batched_loss_outputs(
    trained_wrapper: HybridOdeWrapper,
    store: TrainingDataStore,
    process_names: tuple[str, ...],
    *,
    solver_max_steps: int,
    solver_rtol: float,
    solver_atol: float,
    solver_use_jump_ts: bool,
    step: jax.Array | None = None,
    prediction_grid_n: int | None = None,
):
    """Yield fixed-size JIT evaluation batches and their valid process names."""
    if not process_names:
        raise ValueError("process_names must not be empty")
    unknown = [name for name in process_names if name not in store.process_order]
    if unknown:
        raise ValueError(
            f"unknown process names: {unknown}; available={tuple(store.process_order)}"
        )
    duplicates = [name for name, count in Counter(process_names).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate process names: {duplicates}")
    store.validate_control_support(process_names)

    batch_size = _EVALUATION_BATCH_SIZE
    for start in range(0, len(process_names), batch_size):
        valid_names = process_names[start : start + batch_size]
        indices = [store.process_order.index(name) for name in valid_names]
        indices.extend([indices[-1]] * (batch_size - len(indices)))
        batch = store.gather_batch(jnp.asarray(indices, dtype=jnp.int32))
        jump_ts_rows = None
        if solver_use_jump_ts:
            jump_ts_rows = clamp_padded_time_rows(
                store.controls_store.jump_ts[batch.process_indices],
                store.controls_store.jump_ts_lengths[batch.process_indices],
            )
        outputs = _BATCHED_LOSS_FN_JIT(
            trained_wrapper,
            batch,
            store.Cin_controlled_Inflows,
            store.Cin_modeled_Inflows,
            jump_ts_rows,
            max_solver_steps=int(solver_max_steps),
            solver_rtol=float(solver_rtol),
            solver_atol=float(solver_atol),
            step=step,
            prediction_grid_n=prediction_grid_n,
        )
        jax.block_until_ready(outputs[0])
        yield valid_names, outputs


def compute_dense_exports(
    trained_wrapper: HybridOdeWrapper,
    store: TrainingDataStore,
    process_names: tuple[str, ...],
    *,
    solver_max_steps: int,
    solver_rtol: float,
    solver_atol: float,
    solver_use_jump_ts: bool,
    prediction_grid_n: int = 200,
) -> tuple[np.ndarray, np.ndarray, dict[str, DenseProcessExport]]:
    """Evaluate ``process_names`` in padded JIT batches and build dense exports.

    The single source of dense prediction trajectories for forward and final
    training exports. The prediction grid is harvested from the same loss solve
    (``BatchControls`` + ``discrete_events`` ``jump_ts``), so predictions match
    training and there is no second simulation. Returns ``(per_sample_total,
    per_sample_per_target, dense_exports)`` (loss arrays are ``np``, aligned with
    ``process_names``)."""
    per_sample_total_parts = []
    per_sample_per_target_parts = []
    dense_exports = {}
    for valid_names, outputs in _iter_batched_loss_outputs(
        trained_wrapper,
        store,
        process_names,
        solver_max_steps=solver_max_steps,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
        solver_use_jump_ts=solver_use_jump_ts,
        prediction_grid_n=int(prediction_grid_n),
    ):
        (
            _mean_total,
            per_sample_per_target,
            per_sample_total,
            prediction_t,
            prediction_save_outputs,
            prediction_valid,
            *_,
        ) = outputs
        valid = len(valid_names)
        per_sample_total_parts.append(np.asarray(per_sample_total[:valid]))
        per_sample_per_target_parts.append(np.asarray(per_sample_per_target[:valid]))
        dense_exports.update(
            dense_exports_from_save_outputs(
                prediction_t[:valid],
                jtu.tree_map(lambda leaf: leaf[:valid], prediction_save_outputs),
                trained_wrapper,
                valid_names,
                prediction_valid=prediction_valid[:valid],
            )
        )
    return (
        np.concatenate(per_sample_total_parts),
        np.concatenate(per_sample_per_target_parts),
        dense_exports,
    )


def model_predict(
    trained_wrapper: HybridOdeWrapper,
    config: RunConfig,
    collection: BioProcessCollection,
    *,
    process_names: tuple[str, ...] | None = None,
    grid_n: int = 200,
) -> dict[str, DenseProcessExport]:
    """Forward-solve a trained model over ``collection`` in one batched solve.

    The companion to :func:`~hybrax.train.model_load`: pass the pair it returned plus
    the collection you want predictions for. Solver settings come from
    ``config.solver`` — the values the model was actually fitted under — so there
    is nothing to re-decide here.

    ``collection`` may hold processes the model never trained on. Every process in
    a collection shares one ``RhsOde`` layout; only controls and events differ, and
    a mismatch fails fast via :func:`~hybrax.train.validate_rhs_ode_compatibility`.

    To predict a subset, pass ``process_names`` — do **not** slice
    ``collection.processes``. The hybrax.train metadata block carries its own
    ``process_order``, so a hand-sliced collection fails when the controls store
    is rebuilt.

    ``process_names`` defaults to every process in ``collection``. ``grid_n`` is
    the size of the evenly-spaced output grid; each process's own measurement
    times are spliced into it, so the returned ``t`` is that union (sorted, and
    therefore usually a little longer than ``grid_n``) — a node lands exactly on
    every measurement instead of interpolating across bolus/feed discontinuities.

    Two requirements on ``collection``, both structural rather than incidental:

    - Every process needs a measurement at its first time for **every** target —
      that is where the ODE initial condition comes from.
    - The target set must match what the model was trained on
      (``config.data.targets``).

    Returns ``{process_name: DenseProcessExport}``. Losses are deliberately not
    returned: they are meaningless for a process whose measurements are only a
    ``t0`` seed. Use :func:`evaluate_trained_wrapper` when you want them.

    .. note::
       This reuses ``trained_wrapper``'s ``SCALE_*`` as-is and never rebuilds the
       reaction module. :func:`forward_from_collection` rebuilds them, but always
       from the model's *own* recorded training input, so it does not re-scale the
       model against the collection it is asked to evaluate either.
    """
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=(
            tuple(config.data.targets)
            if config.data is not None and config.data.targets
            else None
        ),
        target_source=(
            config.data.target_source if config.data is not None else TARGET_SOURCE_AUTO
        ),
    )

    selected = (
        tuple(process_names)
        if process_names is not None
        else tuple(store.process_order)
    )
    if not selected:
        raise ValueError("model_predict: no processes to predict")
    unknown = [name for name in selected if name not in store.process_order]
    if unknown:
        raise ValueError(
            f"model_predict: unknown process names {unknown}; "
            f"available={tuple(store.process_order)}"
        )

    # Fail fast on a layout mismatch: without this the solve either dies deep in
    # diffrax on a shape error or, worse, integrates the wrong axes silently.
    validate_rhs_ode_compatibility(
        "trained model",
        trained_wrapper.rhs_ode,
        f"collection process {selected[0]!r}",
        store.rhs_ode,
    )
    _per_sample_total, _per_sample_per_target, dense_exports = compute_dense_exports(
        trained_wrapper,
        store,
        selected,
        solver_max_steps=int(config.solver.max_steps),
        solver_rtol=float(config.solver.rtol),
        solver_atol=float(config.solver.atol),
        solver_use_jump_ts=bool(config.solver.jump_ts),
        prediction_grid_n=int(grid_n),
    )
    return dense_exports


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
    learning_rate: float | optax.Schedule = 1e-3
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
    # Checkpointing. ``checkpoint_every`` is measured in epochs; None selects an
    # automatic cadence of at least five epochs and at most 20 checkpoints. Zero
    # disables periodic writes, while a final checkpoint remains mandatory when
    # configured. ``prepared_path`` is the resolved prepared.json(.gz) bundled
    # into every checkpoint so each is self-contained.
    checkpoint_dir: Path | None = None
    checkpoint_every: float | None = None
    prepared_path: Path | None = None
    # Optional LOO holdout set, evaluated whenever a checkpoint is written.
    holdout_processes: tuple[str, ...] | None = None
    holdout_label: str = "holdout"


@dataclass(frozen=True)
class TrainHarnessResult:
    """Summary object returned by the training harness."""

    trained_wrapper: HybridOdeWrapper
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
    holdout_per_target_by_step: dict[int, tuple[float, ...]] = dataclasses.field(
        default_factory=dict
    )
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
    if callable(learning_rate):
        # optax passes an int32 count to learning-rate callables, making some schedules
        # compute in float32 even with JAX x64 enabled (this depends on the actual
        # schedule arithmetic, exponential decay, for example, uses f32 and cosine uses
        # f64). We thus promote the count before schedule arithmetic to make sure f64 is
        # used in any case (which fixes some minor issues we saw in training
        # reproducibility).
        schedule = learning_rate

        def learning_rate(count):
            return schedule(jnp.asarray(count, dtype=jnp.int64))

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
    ``train_cfg`` carries any hook-overridden learning rate.
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
    """Validate batching config and derive the run's total optimizer-step count.

    Args:
        config: Harness config; ``batch_size=None`` means full-batch.
        selected_process_count: Number of processes selected for training.

    Returns:
        ``(batch_size, batches_per_epoch, total_updates)``, where
        ``total_updates = config.epochs * batches_per_epoch``.

    Raises:
        ValueError: If ``config``'s batching settings are invalid; see
            ``_validate_batching_config``.
    """
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


def _resolve_checkpoint_every(every: float | None, *, epochs: int) -> float:
    return max(5, (epochs + 19) // 20) if every is None else every


def _checkpoint_update_boundaries(
    every: float | None, *, batches_per_epoch: int, total_updates: int
) -> frozenset[int]:
    boundaries = {total_updates}
    every = _resolve_checkpoint_every(every, epochs=total_updates // batches_per_epoch)
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


def _validate_training_parent_collection(
    collection: BioProcessCollection,
    expected_parent_names: tuple[str, ...],
) -> None:
    actual_parent_names = tuple(collection.processes)
    if actual_parent_names != expected_parent_names:
        raise ValueError(
            "training parent collection keys differ from represented parents: "
            f"expected {expected_parent_names!r}, got {actual_parent_names!r}"
        )


def _build_reaction_module(
    *,
    config: TrainHarnessConfig,
    custom_module,
    custom_config: Any,
    store: TrainingDataStore,
    scales: EstimatedScales,
    training_parent_collection: BioProcessCollection,
) -> UserReactionModule:
    hook = get_hook(
        custom_module,
        "build_reaction_module",
        default_build_reaction_module,
    )
    module = hook(
        target_names=list(store.name_measured),
        process_names=list(_ensure_process_names(store, config.process_names)),
        config=custom_config,
        seed=int(config.seed),
        training_parent_collection=training_parent_collection,
        **{
            field.name: getattr(scales, field.name)
            for field in dataclasses.fields(EstimatedScales)
        },
    )
    if not isinstance(module, UserReactionModule):
        raise TypeError(
            "build_reaction_module(...) must return a UserReactionModule instance"
        )
    _require_stateful_opt_in(module, config.allow_stateful_models)
    return module


def _loss_target_labels(store: TrainingDataStore) -> list[str]:
    """Canonical labels for measured species and modeled cumulative flows.

    These name the columns of ``SCL_target_pred`` (``target_state_indices``),
    so ``DefaultLossModule`` emits exactly one term per label.
    """
    return list(store.name_measured) + [
        f"B_{name}_cum"
        for name in (store.name_modeled_Inflows + store.name_modeled_Outflows)
    ]


def _build_loss_module(
    *,
    config: TrainHarnessConfig,
    custom_module,
    custom_config: Any,
    store: TrainingDataStore,
    training_parent_collection: BioProcessCollection,
) -> UserLossModule:
    hook = get_hook(custom_module, "build_loss_module", default_build_loss_module)
    module = hook(
        target_names=_loss_target_labels(store),
        process_names=list(_ensure_process_names(store, config.process_names)),
        config=custom_config,
        seed=int(config.seed),
        training_parent_collection=training_parent_collection,
    )
    if not isinstance(module, UserLossModule):
        raise TypeError("build_loss_module(...) must return a UserLossModule instance")
    return module


def _resolve_estimated_scales(
    *,
    custom_module,
    runtime_data: RuntimeDataContext,
    custom_cfg: Any,
) -> EstimatedScales:
    """Resolve every semantic-axis scale into normalized scalers."""
    store = runtime_data.training_data
    hook = get_hook(custom_module, "estimate_all_scales", None)
    if hook is None:
        defaults = _default_scale_kwargs(
            n_RMCs=len(store.rhs_ode.name_modeled_RMCs),
            n_rates=len(store.rhs_ode.name_modeled_rates),
            n_modeled_Inflows=len(store.rhs_ode.name_modeled_Inflows),
            n_controlled_Inflows=len(store.rhs_ode.name_controlled_Inflows),
            n_modeled_Outflows=len(store.rhs_ode.name_modeled_Outflows),
            n_controlled_Outflows=len(store.rhs_ode.name_controlled_Outflows),
            rhs_ode=store.rhs_ode,
        )
        defaults.pop("SCALE_latent")
        estimated = EstimatedScales(**defaults)
    else:
        estimated = hook(runtime_data, list(store.name_measured), custom_cfg)
    if not isinstance(estimated, EstimatedScales):
        raise TypeError(
            "estimate_all_scales(...) must return an EstimatedScales dataclass; "
            f"got {type(estimated).__name__}"
        )
    resolved = {
        field.name: _as_scaler(getattr(estimated, field.name))
        for field in dataclasses.fields(EstimatedScales)
    }
    return EstimatedScales(**resolved)


def _build_modules_from_selected_parents(
    *,
    store: TrainingDataStore,
    scales: EstimatedScales,
    training_parent_collection: BioProcessCollection,
    expected_parents: tuple[str, ...],
    config: TrainHarnessConfig,
    custom_module: Any,
    custom_config: Any,
    build_loss: bool,
) -> tuple[UserReactionModule, UserLossModule | None]:
    """Build reaction and loss modules behind the same parent-key guards."""
    _validate_training_parent_collection(training_parent_collection, expected_parents)
    reaction_module = _build_reaction_module(
        config=config,
        custom_module=custom_module,
        custom_config=custom_config,
        store=store,
        scales=scales,
        training_parent_collection=training_parent_collection,
    )
    if not build_loss:
        return reaction_module, None
    _validate_training_parent_collection(training_parent_collection, expected_parents)
    loss_module = _build_loss_module(
        config=config,
        custom_module=custom_module,
        custom_config=custom_config,
        store=store,
        training_parent_collection=training_parent_collection,
    )
    return reaction_module, loss_module


def _build_runtime_modules(
    *,
    store: TrainingDataStore,
    collection: BioProcessCollection,
    config: TrainHarnessConfig,
    custom_module,
    custom_config: Any,
    build_loss: bool = True,
) -> tuple[UserReactionModule, UserLossModule | None]:
    """Build runtime hook modules once from parent-selected scale evidence."""
    scale_data = ProducerCollectionData.from_collection(
        store, collection
    ).select_training_parents(
        collection, _ensure_process_names(store, config.process_names)
    )
    scales = _resolve_estimated_scales(
        custom_module=custom_module,
        runtime_data=scale_data,
        custom_cfg=custom_config,
    )
    return _build_modules_from_selected_parents(
        store=store,
        scales=scales,
        training_parent_collection=scale_data.training_parent_collection,
        expected_parents=tuple(scale_data.process_order),
        config=config,
        custom_module=custom_module,
        custom_config=custom_config,
        build_loss=build_loss,
    )


def _target_state_indices(store: TrainingDataStore, rhs_ode: RhsOde) -> jax.Array:
    """Map measured target columns to physical state columns."""
    n_RMCs = len(rhs_ode.name_modeled_RMCs)
    n_PVs = len(rhs_ode.name_modeled_PVs)
    indices: list[int] = []
    for name in store.name_measured_RMCs:
        indices.append(rhs_ode.name_modeled_RMCs.index(name))
    for name in store.name_measured_PVs:
        indices.append(n_RMCs + rhs_ode.name_modeled_PVs.index(name))

    n_modeled_flows = len(store.name_modeled_Inflows) + len(store.name_modeled_Outflows)
    flow_start = n_RMCs + n_PVs + 1
    indices.extend(range(flow_start, flow_start + n_modeled_flows))
    return jnp.asarray(indices, dtype=jnp.int32)


_EVALUATION_COMPATIBILITY_FIELDS = (
    "name_measured",
    "name_measured_RMCs",
    "name_measured_PVs",
)


def _require_evaluation_compatibility(
    training_store: TrainingDataStore, evaluation_store: TrainingDataStore
) -> None:
    """Reject evaluation data the trained wrapper cannot score.

    A template wrapper's ``target_state_indices`` and every ``SCALE_*`` axis are
    fixed by the *training* store's measured/modeled variable names **and their
    order**. An evaluation collection whose variables differ would be scored
    against the wrong columns — or blow up deep inside a JIT trace — so name the
    difference here instead.
    """
    differences: list[str] = []
    for field in _EVALUATION_COMPATIBILITY_FIELDS:
        trained = tuple(getattr(training_store, field))
        evaluated = tuple(getattr(evaluation_store, field))
        if trained != evaluated:
            differences.append(
                f"{field}: model trained on {list(trained)}, "
                f"evaluation data has {list(evaluated)}"
            )
    if differences:
        raise ValueError(
            "forward: the evaluation collection is incompatible with the data this "
            "model was trained on, so the trained wrapper cannot evaluate it:\n  - "
            + "\n  - ".join(differences)
        )
    try:
        validate_rhs_ode_compatibility(
            "training data",
            training_store.rhs_ode,
            "evaluation data",
            evaluation_store.rhs_ode,
        )
    except ValueError as exc:
        raise ValueError(
            "forward: the evaluation collection is incompatible with the data this "
            f"model was trained on: {exc}"
        ) from exc


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
    selected_processes: tuple[str, ...],
    loss_module: UserLossModule | None = None,
) -> HybridOdeWrapper:
    """Build the wrapper structure shared by training and deserialization."""
    if len(selected_processes) == 0:
        raise ValueError("selected_processes must be non-empty")

    reference_index = store.process_order.index(selected_processes[0])
    reference_rhs_ode = replace_rhs_ode_process_matrices(
        store.rhs_ode,
        store.Cin_controlled_Inflows[reference_index],
        store.Cin_modeled_Inflows[reference_index],
        store.retention_controlled_Outflows[reference_index],
        store.retention_modeled_Outflows[reference_index],
    )
    target_state_indices = _target_state_indices(store, reference_rhs_ode)
    return HybridOdeWrapper.from_rhs_ode(
        reaction_module=reaction_module,
        rhs_ode=reference_rhs_ode,
        controls=store.get_process(selected_processes[0]).controls,
        target_state_indices=target_state_indices,
        loss_module=loss_module,
    )


@dataclass(frozen=True)
class ForwardConfig:
    """Configuration for a forward evaluation run (no optimizer)."""

    process_names: tuple[str, ...] | None = None
    target_variable_order: tuple[str, ...] | None = None
    target_source: str | None = None
    solver_max_steps: int = 4096
    solver_rtol: float = 1e-5
    solver_atol: float = 1e-7
    solver_use_jump_ts: bool = True


@dataclass
class ForwardResult:
    """Outputs of a trained-wrapper forward evaluation."""

    trained_wrapper: HybridOdeWrapper
    store: TrainingDataStore
    process_names: tuple[str, ...]
    target_names: tuple[str, ...]
    name_modeled_Inflows: tuple[str, ...]
    name_modeled_Outflows: tuple[str, ...]
    training_process_names: tuple[str, ...]
    per_process_total_loss: dict[str, float]
    per_process_per_target_loss: dict[str, tuple[float, ...]]
    # Dense prediction trajectories harvested from the batched loss solve
    # (``compute_dense_exports``); the source for predictions.csv.
    dense_exports: dict[str, DenseProcessExport] = dataclasses.field(
        default_factory=dict
    )


def forward_from_collection(
    collection: BioProcessCollection,
    *,
    model_path: str | Path,
    config: ForwardConfig | None = None,
    custom_py: str | Path | None = None,
    training_process_names: tuple[str, ...] | None = None,
    run_config: RunConfig | None = None,
    custom_module: Any | None = None,
    prediction_process_names: tuple[str, ...] | None = None,
    prediction_grid_n: int = 200,
) -> ForwardResult:
    """Run one forward pass per selected process of ``collection``.

    ``collection`` is **evaluation data only**. The model itself is rebuilt from
    the prepared collection *it* trained on: that input is resolved from
    ``model_path``'s run directory and its recorded
    ``inputs.prepared_input.content_hash`` is verified before any hook runs (see
    :func:`~hybrax.train.serialization.reconstruct_training`). So the reaction module,
    the loss module, every ``SCALE_*`` and the deserialisation template are exactly
    training's, and evaluation data never reaches a constructor hook.

    A separate store is built from ``collection`` for the solves. It must be
    compatible with the training data — same measured/modeled variables in the
    same order — or the call fails with an explicit error.

    Args:
        collection: Evaluation-only process collection to forward-solve.
        model_path: Path to the trained model's run/checkpoint directory.
        config: Forward-run settings (prediction scope, plotting), or
            ``None`` for defaults.
        custom_py: Override for the recorded ``custom.py`` path, or ``None``
            to use the one the run recorded.
        training_process_names: Restrict reconstruction to this subset of the
            model's original training processes, or ``None`` for all of them.
        run_config: Pre-loaded run config, or ``None`` to read it from
            ``model_path``'s ``config.json``.
        custom_module: Pre-loaded custom module, or ``None`` to load it from
            ``custom_py``/the recorded path.
        prediction_process_names: Restrict prediction to this subset of
            ``collection``'s processes, or ``None`` for all of them.
        prediction_grid_n: Size of the evenly-spaced prediction grid; see
            :func:`model_predict`'s ``grid_n``.

    Returns:
        The forward result: per-process dense exports plus the rebuilt
        wrapper and config.

    Raises:
        ValueError: If ``collection``'s measured/modeled variables are
            incompatible with the training data.
    """
    from .serialization import (
        content_hash,
        load_trained_wrapper,
        read_run_config_json,
        reconstruct_training,
        resolve_forward_model_path,
    )

    cfg = config or ForwardConfig()
    # Resolve (and existence-check) the weights before the reconstruction: it
    # costs seconds on a real input, and failing afterwards reports a path the
    # caller never wrote.
    run_dir, model_path = resolve_forward_model_path(model_path)
    document: dict[str, Any] | None = None
    if run_config is None:
        run_config, document = read_run_config_json(run_dir / "config.json")
    rebuilt = reconstruct_training(
        run_dir,
        run_config,
        document,
        custom_module=custom_module,
        custom_py=custom_py,
        training_process_names=training_process_names,
    )

    recorded_targets = (
        tuple(rebuilt.config.data.targets)
        if rebuilt.config.data is not None and rebuilt.config.data.targets
        else None
    )
    recorded_source = (
        rebuilt.config.data.target_source
        if rebuilt.config.data is not None
        else TARGET_SOURCE_AUTO
    )
    effective_target_order = (
        cfg.target_variable_order
        if cfg.target_variable_order is not None
        else recorded_targets
    )
    effective_target_source = (
        cfg.target_source if cfg.target_source is not None else recorded_source
    )

    # The evaluation store is a separate object built from the caller's data. The
    # one exception is a byte-for-byte identity: when the evaluation collection IS
    # the model's verified training input under the same target layout, building a
    # second identical store would only cost time.
    same_input = (
        effective_target_order == recorded_targets
        and effective_target_source == recorded_source
        and content_hash(collection) == rebuilt.prepared_content_hash
    )
    if same_input:
        evaluation_store = rebuilt.store
    else:
        evaluation_store = TrainingDataStore.from_collection(
            collection,
            target_variable_order=effective_target_order,
            target_source=effective_target_source,
        )
        _require_evaluation_compatibility(rebuilt.store, evaluation_store)

    trained_wrapper = load_trained_wrapper(
        model_path, template=rebuilt.template_wrapper
    )
    return evaluate_trained_wrapper(
        trained_wrapper,
        evaluation_store,
        config=cfg,
        target_names=tuple(rebuilt.loss_module.loss_names),
        training_process_names=rebuilt.training_process_names,
        prediction_process_names=prediction_process_names,
        prediction_grid_n=prediction_grid_n,
    )


def evaluate_trained_wrapper(
    trained_wrapper: HybridOdeWrapper,
    store: TrainingDataStore,
    *,
    config: ForwardConfig,
    target_names: tuple[str, ...],
    training_process_names: tuple[str, ...] = (),
    prediction_process_names: tuple[str, ...] | None = None,
    prediction_grid_n: int = 200,
) -> ForwardResult:
    """Evaluate an existing store and trained wrapper without reconstruction."""
    if config.process_names is not None:
        missing = [n for n in config.process_names if n not in store.process_order]
        if missing:
            raise ValueError(
                f"forward: unknown process names {missing}; "
                f"available={tuple(store.process_order)}"
            )
        eval_processes = tuple(config.process_names)
    else:
        eval_processes = tuple(store.process_order)

    if config.solver_max_steps <= 0:
        raise ValueError("solver_max_steps must be positive")
    if config.solver_rtol <= 0.0:
        raise ValueError("solver_rtol must be positive")
    if config.solver_atol <= 0.0:
        raise ValueError("solver_atol must be positive")

    prediction_processes = (
        eval_processes
        if prediction_process_names is None
        else tuple(prediction_process_names)
    )
    duplicates = [
        name for name, count in Counter(prediction_processes).items() if count > 1
    ]
    if duplicates:
        raise ValueError(f"duplicate prediction process names: {duplicates}")
    unknown = [name for name in prediction_processes if name not in eval_processes]
    if unknown:
        raise ValueError(
            f"prediction process names are not evaluated: {unknown}; "
            f"evaluated={eval_processes}"
        )

    per_process_total: dict[str, float] = {}
    per_process_per_target: dict[str, tuple[float, ...]] = {}
    dense_exports: dict[str, DenseProcessExport] = {}

    def _record_losses(
        names: tuple[str, ...], total: np.ndarray, per_target: np.ndarray
    ) -> None:
        for i, name in enumerate(names):
            per_process_total[name] = float(total[i])
            per_process_per_target[name] = tuple(float(v) for v in per_target[i])

    # A mixed scope intentionally uses two padded batches and JIT variants. This
    # costs compute, but avoids the dense-grid peak memory for unexported processes.
    if prediction_processes:
        total, per_target, dense_exports = compute_dense_exports(
            trained_wrapper,
            store,
            prediction_processes,
            solver_max_steps=int(config.solver_max_steps),
            solver_rtol=float(config.solver_rtol),
            solver_atol=float(config.solver_atol),
            solver_use_jump_ts=config.solver_use_jump_ts,
            prediction_grid_n=int(prediction_grid_n),
        )
        _record_losses(prediction_processes, total, per_target)

    prediction_set = set(prediction_processes)
    loss_only_processes = tuple(
        name for name in eval_processes if name not in prediction_set
    )
    if loss_only_processes:
        total_parts = []
        per_target_parts = []
        for valid_names, outputs in _iter_batched_loss_outputs(
            trained_wrapper,
            store,
            loss_only_processes,
            solver_max_steps=int(config.solver_max_steps),
            solver_rtol=float(config.solver_rtol),
            solver_atol=float(config.solver_atol),
            solver_use_jump_ts=config.solver_use_jump_ts,
        ):
            _, per_target, total, *_ = outputs
            valid = len(valid_names)
            total_parts.append(np.asarray(total[:valid]))
            per_target_parts.append(np.asarray(per_target[:valid]))
        _record_losses(
            loss_only_processes,
            np.concatenate(total_parts),
            np.concatenate(per_target_parts),
        )

    per_process_total = {name: per_process_total[name] for name in eval_processes}
    per_process_per_target = {
        name: per_process_per_target[name] for name in eval_processes
    }

    return ForwardResult(
        trained_wrapper=trained_wrapper,
        store=store,
        process_names=eval_processes,
        target_names=target_names,
        name_modeled_Inflows=tuple(store.name_modeled_Inflows),
        name_modeled_Outflows=tuple(store.name_modeled_Outflows),
        training_process_names=tuple(training_process_names),
        per_process_total_loss=per_process_total,
        per_process_per_target_loss=per_process_per_target,
        dense_exports=dense_exports,
    )


def train_collection(
    store: TrainingDataStore,
    *,
    reaction_module: UserReactionModule,
    loss_module: UserLossModule | None = None,
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

    Args:
        store: Prepared training data for every candidate process.
        reaction_module: Reaction module to train.
        loss_module: Loss module to train alongside it, or ``None`` for the
            default per-target MSE module.
        config: Harness settings (batching, optimizer, solver, checkpointing,
            logging), or ``None`` for defaults.
        optimizer: Pre-built optimizer, or ``None`` to use the default chain.

    Returns:
        The trained wrapper plus the full per-step training history.

    Raises:
        ValueError: If ``reaction_module`` has a nonzero latent state without
            ``config.allow_stateful_models``, or ``config``'s solver settings
            are non-positive.
    """
    cfg = config or TrainHarnessConfig()
    _require_stateful_opt_in(reaction_module, cfg.allow_stateful_models)
    if loss_module is None:
        loss_module = DefaultLossModule(target_names=_loss_target_labels(store))
    effective_batched_loss_fn = _BATCHED_LOSS_FN
    selected_processes = _ensure_process_names(store, cfg.process_names)
    store.validate_control_support(selected_processes)

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

    warmup_batch = store.gather_batch(batch_index_stream[0])

    # Per-target labels come straight from the loss module — one panel/column
    # per named loss term, in declared order.
    _target_labels = list(loss_module.loss_names)

    wrapper = _build_template_wrapper(
        store,
        reaction_module=reaction_module,
        selected_processes=selected_processes,
        loss_module=loss_module,
    )
    batched_Cin = store.Cin_controlled_Inflows
    batched_Cin_modeled = store.Cin_modeled_Inflows

    # Partition the WHOLE wrapper so the loss module's trainable_field() leaves
    # are optimized alongside the reaction module's. Untagged leaves (controls,
    # rhs_ode Cin, indices) stay frozen exactly as before.
    trainable_params, trainable_static = partition_trainable(wrapper)
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
            current_trainable_params: eqx.Module,
            current_optimizer_state: optax.OptState,
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

            def _loss_fn(trainable_params: eqx.Module) -> jax.Array:
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
                ) = effective_batched_loss_fn(
                    candidate_wrapper,
                    current_batch,
                    batched_Cin,
                    batched_Cin_modeled,
                    jump_ts_rows,
                    max_solver_steps=int(cfg.solver_max_steps),
                    solver_rtol=float(cfg.solver_rtol),
                    solver_atol=float(cfg.solver_atol),
                    step=step,
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
            axis_name="hybrax_dev",
            devices=devices,
            in_axes=(
                None,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                (0 if use_jump else None),
                None,
            ),
        )
        def _pgrad(
            params, controls, tm, ym, mk, nm, y0, cin, cinm, ret, retm, wt, jt, step
        ):
            def _local(p):
                wrapper = eqx.combine(p, trainable_static)
                module = wrapper.reaction_module
                # Use the solver's same target scaler so affine offsets cancel
                # in residuals; divergence silently shifts them by b/s.
                scale_targets = module.SCALE_state[wrapper.target_state_indices]
                SCL_ym = ym / scale_targets[None, None, :]
                tmc = clamp_padded_time_rows(tm, nm)

                def _one(
                    pidx,
                    t_row,
                    scl_ym,
                    mask,
                    n_meas,
                    raw_y0,
                    ci,
                    cim,
                    retention,
                    retention_modeled,
                    jts,
                ):
                    return evaluate_one_sample_loss(
                        wrapper,
                        controls,
                        pidx,
                        t_row,
                        scl_ym,
                        mask,
                        n_meas,
                        raw_y0,
                        ci,
                        cim,
                        retention,
                        retention_modeled,
                        jts,
                        max_solver_steps=max_steps,
                        solver_rtol=rtol,
                        solver_atol=atol,
                        step=step,
                    )

                totl, pert, faild = jax.vmap(
                    _one,
                    in_axes=(
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        (0 if use_jump else None),
                    ),
                )(
                    jnp.arange(tm.shape[0], dtype=jnp.int32),
                    tmc,
                    SCL_ym,
                    mk,
                    nm,
                    y0,
                    cin,
                    cinm,
                    ret,
                    retm,
                    jt,
                )
                return jnp.sum(totl * wt), (pert, totl, faild)

            (_l, (pert, totl, faild)), grads = eqx.filter_value_and_grad(
                _local, has_aux=True
            )(params)
            loss = jax.lax.psum(_l, "hybrax_dev") / bs
            grads = jax.tree_util.tree_map(
                lambda g: jax.lax.psum(g, "hybrax_dev") / bs, grads
            )
            per_target = (
                jax.lax.psum(jnp.sum(pert * wt[:, None], axis=0), "hybrax_dev") / bs
            )
            per_sample = jax.lax.all_gather(totl, "hybrax_dev")
            per_sample_fail = jax.lax.all_gather(faild, "hybrax_dev")
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
                    current_batch.t_measured,
                    current_batch.y_measured,
                    current_batch.mask_measured,
                    current_batch.n_measured,
                    current_batch.y0_measured,
                    cin,
                    cinm,
                    current_batch.retention_controlled_Outflows,
                    current_batch.retention_modeled_Outflows,
                )
            ]
            controls = jtu.tree_map(_shard, jtu.tree_map(_pad, current_batch.controls))
            jt = _shard(_pad(jump_ts_rows)) if jump_ts_rows is not None else None
            loss, grads, per_target, per_sample, per_sample_fail = _pgrad(
                current_trainable_params,
                controls,
                ins[0],
                ins[1],
                ins[2],
                ins[3],
                ins[4],
                ins[5],
                ins[6],
                ins[7],
                ins[8],
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
        sharded across an Auto-axis ``Mesh`` (``device_put`` with ``P('hybrax_dev')``);
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
            (n_dev,), ("hybrax_dev",), axis_types=(AxisType.Auto,), devices=devices
        )
        S_batch = NamedSharding(mesh, P("hybrax_dev"))  # shard the leading batch axis
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
            params,
            opt_state,
            controls,
            tm,
            ym,
            mk,
            nm,
            y0,
            cin,
            cinm,
            ret,
            retm,
            wt,
            jt,
            step,
        ):
            # Inside-jit sharding constraints (per the equinox autoparallelism
            # tutorial): emits jax.lax.with_sharding_constraint so XLA keeps
            # params/opt replicated and the batch axis sharded *through* the
            # computation, rather than replicating it.
            params, opt_state = eqx.filter_shard((params, opt_state), S_repl)
            controls, tm, ym, mk, nm, y0, cin, cinm, ret, retm, wt, jt = (
                eqx.filter_shard(
                    (controls, tm, ym, mk, nm, y0, cin, cinm, ret, retm, wt, jt),
                    S_batch,
                )
            )

            def _local(p):
                wrapper = eqx.combine(p, trainable_static)
                module = wrapper.reaction_module
                # Use the solver's same target scaler so affine offsets cancel
                # in residuals; divergence silently shifts them by b/s.
                scale_targets = module.SCALE_state[wrapper.target_state_indices]
                SCL_ym = ym / scale_targets[None, None, :]
                tmc = clamp_padded_time_rows(tm, nm)

                def _one(
                    pidx,
                    t_row,
                    scl_ym,
                    mask,
                    n_meas,
                    raw_y0,
                    ci,
                    cim,
                    retention,
                    retention_modeled,
                    jts,
                ):
                    return evaluate_one_sample_loss(
                        wrapper,
                        controls,
                        pidx,
                        t_row,
                        scl_ym,
                        mask,
                        n_meas,
                        raw_y0,
                        ci,
                        cim,
                        retention,
                        retention_modeled,
                        jts,
                        max_solver_steps=max_steps,
                        solver_rtol=rtol,
                        solver_atol=atol,
                        step=step,
                    )

                totl, pert, faild = jax.vmap(
                    _one,
                    in_axes=(
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        (0 if use_jump else None),
                    ),
                )(
                    jnp.arange(tm.shape[0], dtype=jnp.int32),
                    tmc,
                    SCL_ym,
                    mk,
                    nm,
                    y0,
                    cin,
                    cinm,
                    ret,
                    retm,
                    jt,
                )
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
                        current_batch.t_measured,
                        current_batch.y_measured,
                        current_batch.mask_measured,
                        current_batch.n_measured,
                        current_batch.y0_measured,
                        cin,
                        cinm,
                        current_batch.retention_controlled_Outflows,
                        current_batch.retention_modeled_Outflows,
                    )
                ]
                controls = jtu.tree_map(
                    lambda x: jax.device_put(_pad(x), S_batch), current_batch.controls
                )
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
                    controls,
                    sharded[0],
                    sharded[1],
                    sharded[2],
                    sharded[3],
                    sharded[4],
                    sharded[5],
                    sharded[6],
                    sharded[7],
                    sharded[8],
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
    # (`assert not hlo_sharding.is_manual()`), and GSPMD auto-sharding
    # (HYBRAX_GSPMD=1) is correct + patch-free but ~60x slower because XLA
    # cannot partition the data-dependent ODE solve (it replicates the
    # per-device work). pmap needs the
    # equinox closure-convert patch on jax>=0.10; GSPMD needs no patch.
    _use_gspmd = os.environ.get("HYBRAX_GSPMD", "") not in ("", "0", "false", "False")
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
    resolved_checkpoint_every = _resolve_checkpoint_every(
        cfg.checkpoint_every, epochs=cfg.epochs
    )
    checkpoint_boundaries = (
        _checkpoint_update_boundaries(
            resolved_checkpoint_every,
            batches_per_epoch=batches_per_epoch,
            total_updates=total_updates,
        )
        if checkpoint_enabled
        else frozenset()
    )
    if checkpoint_enabled and cfg.checkpoint_every is None:
        logger.info(
            "checkpoint_every is null; using sensible automatic default "
            "every=%d epochs (%d checkpoints including final)",
            resolved_checkpoint_every,
            len(checkpoint_boundaries),
        )
    checkpoint_writer = (
        CheckpointWriter(
            Path(cfg.checkpoint_dir),
            prepared_src=cfg.prepared_path,
        )
        if cfg.checkpoint_dir is not None
        else None
    )
    # The train/holdout loss series the final curve needs are already accumulated
    # by ``RunLogger`` (see ``finalize()`` below); only the per-target holdout
    # breakdown is tracked here, because it is a result field rather than history.
    holdout_per_target_so_far: dict[int, tuple[float, ...]] = {}

    if cfg.holdout_processes:
        unknown = [n for n in cfg.holdout_processes if n not in store.process_order]
        if unknown:
            raise ValueError(
                f"holdout_processes contains unknown names: {unknown}; "
                f"available={tuple(store.process_order)}"
            )

    def _evaluate_holdout(step: int):
        if not cfg.holdout_processes:
            return None, None, None
        per_sample_total_parts = []
        per_sample_per_target_parts = []
        dense_exports: dict[str, DenseProcessExport] = {}
        for valid_names, outputs in _iter_batched_loss_outputs(
            wrapper,
            store,
            cfg.holdout_processes,
            solver_max_steps=int(cfg.solver_max_steps),
            solver_rtol=float(cfg.solver_rtol),
            solver_atol=float(cfg.solver_atol),
            solver_use_jump_ts=bool(cfg.solver_use_jump_ts),
            step=jnp.asarray(step, dtype=jnp.int32),
        ):
            (
                _,
                per_sample_per_target,
                per_sample_total,
                _,
                _,
                _,
                measurement_t,
                measurement_save_outputs,
                measurement_prediction_valid,
                *_,
            ) = outputs
            valid = len(valid_names)
            per_sample_total_parts.append(np.asarray(per_sample_total[:valid]))
            per_sample_per_target_parts.append(
                np.asarray(per_sample_per_target[:valid])
            )
            dense_exports.update(
                dense_exports_from_save_outputs(
                    measurement_t[:valid],
                    jtu.tree_map(
                        lambda leaf, n_valid=valid: leaf[:n_valid],
                        measurement_save_outputs,
                    ),
                    wrapper,
                    valid_names,
                    prediction_valid=measurement_prediction_valid[:valid],
                )
            )
        per_sample_total = np.concatenate(per_sample_total_parts)
        per_sample_per_target = np.concatenate(per_sample_per_target_parts)
        return (
            float(np.mean(per_sample_total)),
            tuple(float(value) for value in np.mean(per_sample_per_target, axis=0)),
            dense_exports,
        )

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

            holdout_loss, holdout_per_target, holdout_predictions = (None, None, None)
            if step in checkpoint_boundaries:
                (
                    holdout_loss,
                    holdout_per_target,
                    holdout_predictions,
                ) = _evaluate_holdout(step)
                if holdout_loss is not None:
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
                    holdout_predictions=holdout_predictions,
                )
                _write_training_plots(
                    history=run_log.snapshot(),
                    holdout_per_target_by_step=holdout_per_target_so_far,
                    output_dir=Path(cfg.checkpoint_dir).parent,
                    step=step,
                )

        history = run_log.finalize()

    return TrainHarnessResult(
        trained_wrapper=wrapper,
        compile_warmup_seconds=float(warmup_compile_seconds),
        train_step_input_signature=train_step_input_signature,
        updates_completed=total_updates,
        holdout_per_target_by_step=holdout_per_target_so_far,
        **history,
    )


def _write_training_plots(
    *,
    history: dict[str, Any],
    holdout_per_target_by_step: dict[int, tuple[float, ...]],
    output_dir: Path,
    step: int,
) -> None:
    """Refresh the run-level training plots after a checkpoint write."""
    try:
        plot_loss_curve(
            history["mean_loss_by_step"],
            output_dir / "loss_curve.png",
            title=f"Training loss (through step {step})",
            per_target_loss_by_step=history["per_target_loss_by_step"],
            target_names=history["target_names"],
            monitor_loss_by_step=history["holdout_loss_by_step"] or None,
            monitor_per_target_by_step=holdout_per_target_by_step or None,
            monitor_label=history["holdout_label"],
        )
    except Exception:
        # A diagnostic PNG must not fail an otherwise valid training run.
        logger.exception("failed to write loss curve at checkpoint")

    try:
        plot_grad_norm_curve(
            history["grad_norm_by_step"],
            output_dir / "grad_norm_curve.png",
            title=f"Gradient norm (through step {step})",
        )
    except Exception:
        logger.exception("failed to write gradient norm curve at checkpoint")


@dataclass(frozen=True)
class PreparedTraining:
    """Collection-free inputs for :func:`train_collection`."""

    store: TrainingDataStore
    reaction_module: UserReactionModule
    loss_module: UserLossModule
    config: TrainHarnessConfig
    optimizer: optax.GradientTransformation
    prediction_parent_process_names: tuple[str, ...] = ()


_TRAIN_HOOK_NAMES = (
    "estimate_all_scales",
    "build_reaction_module",
    "build_learning_rate",
    "build_optimizer",
    "build_loss_module",
)


def _log_train_hooks(custom_module: Any) -> None:
    customized_hooks, default_hooks = split_hooks_by_customization(
        custom_module, _TRAIN_HOOK_NAMES
    )
    logger.info("train hooks detected: %s", ", ".join(customized_hooks) or "none")
    logger.info("train hooks default: %s", ", ".join(default_hooks) or "none")


def _prepare_training_from_selected_parents(
    *,
    store: TrainingDataStore,
    scales: EstimatedScales,
    augmentation_parents: tuple[str | None, ...],
    training_parent_collection: BioProcessCollection,
    config: TrainHarnessConfig,
    custom_module: Any,
    custom_cfg: Any,
) -> PreparedTraining:
    """Build every hook-visible object from one selected training-parent set."""
    _log_train_hooks(custom_module)
    process_order = tuple(store.process_order)
    selected_processes = _ensure_process_names(store, config.process_names)
    train_cfg = dataclasses.replace(config, process_names=selected_processes)
    expected_parents = canonical_training_parents(
        process_order, augmentation_parents, selected_processes
    )
    reaction_module, loss_module = _build_modules_from_selected_parents(
        store=store,
        scales=scales,
        training_parent_collection=training_parent_collection,
        expected_parents=expected_parents,
        config=train_cfg,
        custom_module=custom_module,
        custom_config=custom_cfg,
        build_loss=True,
    )
    assert loss_module is not None
    _, _, total_updates = derive_update_budget(
        train_cfg, selected_process_count=len(selected_processes)
    )
    optimizer, train_cfg = build_optimizer_for_run(
        custom_module=custom_module,
        custom_cfg=custom_cfg,
        train_cfg=train_cfg,
        total_updates=total_updates,
    )
    return PreparedTraining(
        store=store,
        reaction_module=reaction_module,
        loss_module=loss_module,
        config=train_cfg,
        optimizer=optimizer,
        prediction_parent_process_names=original_parent_processes(
            process_order, augmentation_parents
        ),
    )


def prepare_training_from_runtime_artifact(
    artifact: RuntimeArtifact,
    *,
    config: TrainHarnessConfig,
    custom_module: Any,
    custom_cfg: Any,
) -> PreparedTraining:
    """Prepare training solely from an already-loaded runtime artifact."""
    if not isinstance(artifact, RuntimeArtifact):
        raise TypeError("artifact must be a RuntimeArtifact")
    return _prepare_training_from_selected_parents(
        store=artifact.training_data,
        scales=artifact.scales,
        augmentation_parents=artifact.augmentation_parents,
        training_parent_collection=artifact.training_parent_collection,
        config=config,
        custom_module=custom_module,
        custom_cfg=custom_cfg,
    )


def prepare_training(
    collection: BioProcessCollection,
    *,
    config: TrainHarnessConfig | None = None,
    custom_py: str | Path | None = None,
    runtime_config: dict[str, Any] | None = None,
    run_config: RunConfig | None = None,
    custom_module: Any | None = None,
) -> PreparedTraining:
    """Build runtime inputs while borrowing, but never retaining, collection."""
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
    print_rhs_ode(collection)
    sys.stdout.flush()

    selected_processes = _ensure_process_names(store, cfg.process_names)
    train_cfg = dataclasses.replace(
        cfg,
        process_names=selected_processes,
        target_variable_order=effective_target_order,
    )
    producer_data = ProducerCollectionData.from_collection(store, collection)
    scale_data = producer_data.select_training_parents(collection, selected_processes)
    return _prepare_training_from_selected_parents(
        store=store,
        scales=_resolve_estimated_scales(
            custom_module=custom_module,
            runtime_data=scale_data,
            custom_cfg=custom_cfg,
        ),
        augmentation_parents=producer_data.augmentation_parents,
        training_parent_collection=scale_data.training_parent_collection,
        config=train_cfg,
        custom_module=custom_module,
        custom_cfg=custom_cfg,
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
    """Prepare and train; external collection references remain caller-owned."""
    prepared = prepare_training(
        collection,
        config=config,
        custom_py=custom_py,
        runtime_config=runtime_config,
        run_config=run_config,
        custom_module=custom_module,
    )
    del collection
    return train_collection(
        prepared.store,
        reaction_module=prepared.reaction_module,
        loss_module=prepared.loss_module,
        config=prepared.config,
        optimizer=prepared.optimizer,
    )


def train_harness_config_from_run_config(
    cfg: RunConfig,
    *,
    run_dir: Path,
) -> TrainHarnessConfig:
    """Map a typed :class:`RunConfig` + run-directory layout to the harness
    config, wiring the run-dir artifact paths (``metrics.csv``, ``checkpoints/``)
    and the checkpoint/output/logging sections. Shared by the CLI train and LOO
    paths.
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
    should invoke ``train_from_collection`` directly to avoid loading the
    prepared JSON twice.
    """
    collection = load_process_collection(Path(prepared_json))
    return train_from_collection(
        collection,
        config=config,
        custom_py=custom_py,
        runtime_config=runtime_config,
    )
