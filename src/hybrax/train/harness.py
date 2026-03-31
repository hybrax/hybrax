from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from bpbench.serialization import load_process_collection_json

from .defaults import (
    DefaultReactionModule,
    default_build_modeled_feeds,
    default_build_reaction_module,
)
from .model_api import UserReactionModule, partition_trainable
from .trainer import (
    _build_optimizer,
    _BatchIndexedControls,
    _clamp_padded_time_rows,
    _measurement_loss_from_arrays,
    single_process_measurement_loss,
    summarize_train_step_input_signature,
)
from .controls_store import BatchControls
from .training_data import (
    BatchTrainingData,
    TARGET_SOURCE_AUTO,
    TrainingDataStore,
)
from .utils import get_hook, load_custom_module, resolve_config
from .wrapper import LibraryRhsWrapper, ModeledFeedSpec


logger = logging.getLogger(__name__)


def _wrapper_feed_signature(
    wrapper: LibraryRhsWrapper,
) -> dict[str, object]:
    return {
        "controlled_feed_names": tuple(wrapper.controlled_feed_names),
        "controlled_feed_control_indices": np.asarray(
            wrapper.controlled_feed_control_indices, dtype=np.int32
        ),
        "controlled_feed_cin_xp": np.asarray(
            wrapper.controlled_feed_cin_xp, dtype=np.float32
        ),
        "controlled_feed_cin_fp": np.asarray(
            wrapper.controlled_feed_cin_fp, dtype=np.float32
        ),
        "modeled_feed_names": tuple(wrapper.modeled_feed_names),
        "modeled_feed_cin_xp": np.asarray(
            wrapper.modeled_feed_cin_xp, dtype=np.float32
        ),
        "modeled_feed_cin_fp": np.asarray(
            wrapper.modeled_feed_cin_fp, dtype=np.float32
        ),
        "sample_acc_control_index": int(wrapper.sample_acc_control_index),
    }


def _validate_wrapper_feed_compatibility(
    *,
    reference_name: str,
    reference_wrapper: LibraryRhsWrapper,
    candidate_name: str,
    candidate_wrapper: LibraryRhsWrapper,
) -> None:
    reference_signature = _wrapper_feed_signature(reference_wrapper)
    candidate_signature = _wrapper_feed_signature(candidate_wrapper)

    for field_name, reference_value in reference_signature.items():
        candidate_value = candidate_signature[field_name]
        if isinstance(reference_value, tuple):
            if reference_value != candidate_value:
                raise ValueError(
                    "selected processes have incompatible wrapper feed semantics: "
                    f"{field_name} differs between {reference_name!r} and "
                    f"{candidate_name!r}"
                )
            continue

        if not np.array_equal(reference_value, candidate_value):
            raise ValueError(
                "selected processes have incompatible wrapper feed semantics: "
                f"{field_name} differs between {reference_name!r} and "
                f"{candidate_name!r}"
            )


def _batched_measurement_loss_from_batch(
    wrapper: LibraryRhsWrapper,
    batch: BatchTrainingData,
    batch_controls: BatchControls,
    jump_ts_rows: jax.Array | None,
    *,
    max_solver_steps: int,
    solver_rtol: float,
    solver_atol: float,
) -> jax.Array:
    batch_t_meas = _clamp_padded_time_rows(batch.t_meas, batch.n_meas)

    def _sample_loss(
        process_idx: jax.Array,
        t_meas: jax.Array,
        y_meas: jax.Array,
        meas_mask: jax.Array,
        n_meas: jax.Array,
        y0: jax.Array,
        jump_ts: jax.Array | None,
    ) -> jax.Array:
        controls = _BatchIndexedControls(
            batch_controls=batch_controls,
            process_idx=process_idx,
        )
        sample_wrapper = eqx.tree_at(
            lambda current: current.controls, wrapper, controls
        )
        return _measurement_loss_from_arrays(
            sample_wrapper,
            t_meas=t_meas,
            y_meas=y_meas,
            meas_mask=meas_mask,
            n_meas=n_meas,
            y0=y0,
            jump_ts=jump_ts,
            max_solver_steps=max_solver_steps,
            solver_rtol=solver_rtol,
            solver_atol=solver_atol,
        )

    if jump_ts_rows is None:
        per_sample = jax.vmap(
            lambda process_idx, t_meas, y_meas, meas_mask, n_meas, y0: _sample_loss(
                process_idx,
                t_meas,
                y_meas,
                meas_mask,
                n_meas,
                y0,
                None,
            )
        )(
            batch.process_indices,
            batch_t_meas,
            batch.y_meas,
            batch.meas_mask,
            batch.n_meas,
            batch.y0,
        )
    else:
        per_sample = jax.vmap(_sample_loss)(
            batch.process_indices,
            batch_t_meas,
            batch.y_meas,
            batch.meas_mask,
            batch.n_meas,
            batch.y0,
            jump_ts_rows,
        )
    return jnp.mean(per_sample)


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
    learning_rate: float = 1e-3
    seed: int = 0
    log_every: int = 10
    solver_max_steps: int = 100_000
    solver_rtol: float = 1e-5
    solver_atol: float = 1e-7
    solver_use_jump_ts: bool = True


@dataclass(frozen=True)
class TrainHarnessResult:
    """Summary object returned by the training harness."""

    mean_loss_by_step: tuple[float, ...]
    sampled_loss_by_process_at_log_steps: dict[int, tuple[tuple[str, float], ...]]
    batch_process_names_by_step: tuple[tuple[str, ...], ...]
    compile_warmup_seconds: float
    step_time_seconds: tuple[float, ...]
    train_step_input_signature: tuple[object, ...]
    train_step_rebuild_count: int


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
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
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


def _as_modeled_feed_specs(payload: Any) -> tuple[ModeledFeedSpec, ...]:
    if payload is None:
        return ()
    specs: list[ModeledFeedSpec] = []
    for entry in payload:
        if isinstance(entry, ModeledFeedSpec):
            specs.append(entry)
            continue
        if not isinstance(entry, dict):
            raise TypeError("modeled feed entries must be ModeledFeedSpec or dict")
        specs.append(
            ModeledFeedSpec(
                name=str(entry["name"]),
                component_concentrations={
                    str(key): float(value)
                    for key, value in dict(entry["component_concentrations"]).items()
                },
            )
        )
    return tuple(specs)


def _build_reaction_module(
    *,
    store: TrainingDataStore,
    config: TrainHarnessConfig,
    custom_module,
    custom_config: dict[str, Any],
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
    )
    if not isinstance(module, UserReactionModule):
        raise TypeError(
            "build_reaction_module(...) must return a UserReactionModule instance"
        )
    return module


def _build_modeled_feeds(
    *,
    custom_module,
    custom_config: dict[str, Any],
    target_names: tuple[str, ...],
) -> tuple[ModeledFeedSpec, ...]:
    hook = get_hook(custom_module, "build_modeled_feeds", default_build_modeled_feeds)
    payload = hook(target_names=list(target_names), config=custom_config)
    return _as_modeled_feed_specs(payload)


def train_collection(
    store: TrainingDataStore,
    *,
    reaction_module: UserReactionModule,
    config: TrainHarnessConfig | None = None,
    modeled_feeds: tuple[ModeledFeedSpec, ...] | list[ModeledFeedSpec] | None = None,
) -> TrainHarnessResult:
    """Train one reaction module over one or many processes from one store."""
    cfg = config or TrainHarnessConfig()
    selected_processes = _ensure_process_names(store, cfg.process_names)
    modeled_feed_specs = tuple(modeled_feeds or ())

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
    wrapper: LibraryRhsWrapper | None = None
    reference_process_name = selected_processes[0]
    for process_name in selected_processes:
        process_wrapper = LibraryRhsWrapper.from_process_controls(
            reaction_module=reaction_module,
            controls=store.get_process(process_name).controls,
            species_names=store.target_names,
            modeled_feeds=list(modeled_feed_specs),
        )
        if wrapper is None:
            wrapper = process_wrapper
            continue
        _validate_wrapper_feed_compatibility(
            reference_name=reference_process_name,
            reference_wrapper=wrapper,
            candidate_name=process_name,
            candidate_wrapper=process_wrapper,
        )
    assert wrapper is not None

    trainable_params, trainable_static = partition_trainable(reaction_module)
    optimizer = _build_optimizer(cfg.optimizer_name, float(cfg.learning_rate))
    optimizer_state = optimizer.init(trainable_params)
    train_step_input_signature = summarize_train_step_input_signature(
        wrapper,
        trainable_params,
        optimizer_state,
        warmup_batch,
    )

    def _make_batched_step():
        def _step_fn(
            current_wrapper: LibraryRhsWrapper,
            current_trainable_params: Any,
            current_optimizer_state: Any,
            current_batch,
        ):
            jump_ts_rows = None
            if cfg.solver_use_jump_ts:
                jump_ts_rows = _clamp_padded_time_rows(
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
                return _batched_measurement_loss_from_batch(
                    candidate_wrapper,
                    current_batch,
                    batch_controls,
                    jump_ts_rows,
                    max_solver_steps=int(cfg.solver_max_steps),
                    solver_rtol=float(cfg.solver_rtol),
                    solver_atol=float(cfg.solver_atol),
                )

            loss, grads = eqx.filter_value_and_grad(_loss_fn)(current_trainable_params)
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
            return wrapper_updated, trainable_updated, loss, next_optimizer_state

        return eqx.filter_jit(_step_fn)

    step_fn = _make_batched_step()
    rebuild_count = 0

    warmup_t0 = time.perf_counter()
    # Warmup intentionally executes once with step-0 batch shape/signature so
    # the timed training loop excludes first-call compilation latency.
    _warmup_wrapper, _warmup_trainable, warmup_loss, _warmup_opt_state = step_fn(
        wrapper,
        trainable_params,
        optimizer_state,
        warmup_batch,
    )
    jax.block_until_ready(warmup_loss)
    warmup_compile_seconds = time.perf_counter() - warmup_t0

    mean_loss_by_step: list[float] = []
    step_time_seconds: list[float] = []
    batch_process_names_by_step: list[tuple[str, ...]] = []
    sampled_loss_by_process_at_log_steps: dict[int, tuple[tuple[str, float], ...]] = {}

    logger.info(
        "train harness setup: processes=%s, targets=%s source=%s steps=%d "
        "batch_size=%d optimizer=%s lr=%g",
        list(selected_processes),
        list(store.target_names),
        store.target_source,
        cfg.steps,
        effective_batch_size,
        cfg.optimizer_name,
        cfg.learning_rate,
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
            logger.warning(
                "rebuilt train-step at step=%d due signature drift",
                step_index + 1,
            )

        t0 = time.perf_counter()
        wrapper, trainable_params, loss, optimizer_state = step_fn(
            wrapper,
            trainable_params,
            optimizer_state,
            batch,
        )
        jax.block_until_ready(loss)
        step_dt = time.perf_counter() - t0

        mean_loss = float(loss)
        mean_loss_by_step.append(mean_loss)
        step_time_seconds.append(step_dt)

        batch_indices_tuple = tuple(
            int(v) for v in np.asarray(batch.process_indices).tolist()
        )
        batch_names = tuple(store.process_order[idx] for idx in batch_indices_tuple)
        batch_process_names_by_step.append(batch_names)

        should_log = (step_index + 1) % cfg.log_every == 0
        if should_log:
            # Telemetry path: sampled per-process losses for current batch members.
            # This intentionally uses the single-process helper at log cadence.
            sampled_losses: list[tuple[str, float]] = []
            for process_idx in batch_indices_tuple:
                process_data = store.get_process(process_idx)
                process_wrapper = eqx.tree_at(
                    lambda current: current.controls,
                    wrapper,
                    process_data.controls,
                )
                sample_loss = single_process_measurement_loss(
                    process_wrapper,
                    process_data,
                    max_solver_steps=int(cfg.solver_max_steps),
                    solver_rtol=float(cfg.solver_rtol),
                    solver_atol=float(cfg.solver_atol),
                    solver_use_jump_ts=bool(cfg.solver_use_jump_ts),
                )
                sampled_losses.append((process_data.process_name, float(sample_loss)))

            sampled_loss_by_process_at_log_steps[step_index + 1] = tuple(sampled_losses)
            logger.info(
                "step %d/%d sampled=%s",
                step_index + 1,
                cfg.steps,
                sampled_loss_by_process_at_log_steps[step_index + 1],
            )

    return TrainHarnessResult(
        mean_loss_by_step=tuple(mean_loss_by_step),
        sampled_loss_by_process_at_log_steps=sampled_loss_by_process_at_log_steps,
        batch_process_names_by_step=tuple(batch_process_names_by_step),
        compile_warmup_seconds=float(warmup_compile_seconds),
        step_time_seconds=tuple(step_time_seconds),
        train_step_input_signature=train_step_input_signature,
        train_step_rebuild_count=int(rebuild_count),
    )


def train_from_prepared_json(
    prepared_json: str | Path,
    *,
    config: TrainHarnessConfig | None = None,
    custom_py: str | Path | None = None,
    runtime_config: dict[str, Any] | None = None,
) -> TrainHarnessResult:
    """Train from prepared JSON with optional custom model hooks."""
    cfg = config or TrainHarnessConfig()
    custom_module = load_custom_module(custom_py)
    custom_cfg = resolve_config(custom_module, runtime_config)
    collection = load_process_collection_json(Path(prepared_json))
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=cfg.target_variable_order,
        target_source=cfg.target_source,
    )

    selected_processes = _ensure_process_names(store, cfg.process_names)
    effective_cfg = TrainHarnessConfig(
        process_names=selected_processes,
        target_variable_order=cfg.target_variable_order,
        target_source=cfg.target_source,
        steps=cfg.steps,
        batch_size=cfg.batch_size,
        shuffle_batches=cfg.shuffle_batches,
        batch_seed=cfg.batch_seed,
        optimizer_name=cfg.optimizer_name,
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,
        log_every=cfg.log_every,
        solver_max_steps=cfg.solver_max_steps,
        solver_rtol=cfg.solver_rtol,
        solver_atol=cfg.solver_atol,
        solver_use_jump_ts=cfg.solver_use_jump_ts,
    )
    reaction_module = _build_reaction_module(
        store=store,
        config=effective_cfg,
        custom_module=custom_module,
        custom_config=custom_cfg,
    )
    modeled_feeds = _build_modeled_feeds(
        custom_module=custom_module,
        custom_config=custom_cfg,
        target_names=tuple(store.target_names),
    )
    return train_collection(
        store,
        reaction_module=reaction_module,
        config=effective_cfg,
        modeled_feeds=modeled_feeds,
    )
