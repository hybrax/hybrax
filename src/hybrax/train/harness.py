from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
from bpbench.serialization import load_process_collection_json

from .defaults import (
    DefaultReactionModule,
    default_build_modeled_feeds,
    default_build_reaction_module,
)
from .model_api import UserReactionModule, partition_trainable
from .trainer import single_process_measurement_loss
from .training_data import (
    PerProcessTrainingData,
    TARGET_SOURCE_AUTO,
    TrainingDataStore,
)
from .utils import get_hook, load_custom_module, resolve_config
from .wrapper import LibraryRhsWrapper, ModeledFeedSpec


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainHarnessConfig:
    """Configuration for collection-level training harness runs."""

    process_names: tuple[str, ...] | None = None
    target_variable_order: tuple[str, ...] | None = None
    target_source: str = TARGET_SOURCE_AUTO
    steps: int = 50
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

    final_reaction_module: UserReactionModule
    process_names: tuple[str, ...]
    mean_loss_by_step: tuple[float, ...]
    loss_by_process: dict[str, tuple[float, ...]]
    compile_time_seconds_by_process: dict[str, float]
    step_time_seconds_by_process: dict[str, tuple[float, ...]]
    compile_count_by_process: dict[str, int]
    total_compile_seconds: float
    total_compile_count: int
    total_step_seconds: float
    suspicious_step_spikes_by_process: dict[str, int]


def _ensure_process_names(
    store: TrainingDataStore,
    requested: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if requested is None:
        return tuple(store.process_order)
    missing = [name for name in requested if name not in store.process_order]
    if missing:
        raise ValueError(f"requested process names not found in dataset: {missing}")
    return requested


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


def _make_compiled_step(
    *,
    process_data: PerProcessTrainingData,
    modeled_feeds: tuple[ModeledFeedSpec, ...],
    learning_rate: float,
    solver_max_steps: int,
    solver_rtol: float,
    solver_atol: float,
    solver_use_jump_ts: bool,
):
    def _step_fn(trainable: eqx.Module, static: eqx.Module):
        base_module = eqx.combine(trainable, static)
        wrapper = LibraryRhsWrapper.from_process_controls(
            reaction_module=base_module,
            controls=process_data.controls,
            species_names=process_data.target_names,
            modeled_feeds=list(modeled_feeds),
        )

        def _loss_fn(trainable_params: eqx.Module):
            reaction_module = eqx.combine(trainable_params, static)
            candidate_wrapper = eqx.tree_at(
                lambda current: current.reaction_module,
                wrapper,
                reaction_module,
            )
            return single_process_measurement_loss(
                candidate_wrapper,
                process_data,
                max_solver_steps=solver_max_steps,
                solver_rtol=solver_rtol,
                solver_atol=solver_atol,
                solver_use_jump_ts=solver_use_jump_ts,
            )

        loss, grads = eqx.filter_value_and_grad(_loss_fn)(trainable)
        updates = jax.tree_util.tree_map(
            lambda grad: None if grad is None else -float(learning_rate) * grad,
            grads,
        )
        trainable_updated = eqx.apply_updates(trainable, updates)
        return trainable_updated, loss

    return eqx.filter_jit(_step_fn)


def _partition_signature(
    trainable: eqx.Module, static: eqx.Module
) -> tuple[object, ...]:
    leaves = jax.tree_util.tree_leaves(
        (trainable, static),
        is_leaf=lambda value: value is None,
    )
    signature: list[object] = []
    for leaf in leaves:
        if leaf is None:
            signature.append(("none",))
            continue
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            signature.append(("array", tuple(leaf.shape), str(leaf.dtype)))
            continue
        signature.append(("object", type(leaf).__name__))
    return tuple(signature)


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

    if cfg.steps <= 0:
        raise ValueError("steps must be positive")
    if cfg.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if cfg.solver_max_steps <= 0:
        raise ValueError("solver_max_steps must be positive")
    if cfg.solver_rtol <= 0.0:
        raise ValueError("solver_rtol must be positive")
    if cfg.solver_atol <= 0.0:
        raise ValueError("solver_atol must be positive")

    compiled_steps: dict[str, Any] = {}
    compiled_signatures: dict[str, tuple[object, ...]] = {}
    process_data_by_name = {
        process_name: store.get_process(process_name)
        for process_name in selected_processes
    }

    compile_time_seconds_by_process: dict[str, float] = {}
    compile_count_by_process: dict[str, int] = {}
    step_time_seconds_by_process: dict[str, list[float]] = {
        process_name: [] for process_name in selected_processes
    }
    loss_by_process: dict[str, list[float]] = {
        process_name: [] for process_name in selected_processes
    }
    mean_loss_by_step: list[float] = []

    trainable_params, static_params = partition_trainable(reaction_module)
    logger.info(
        "train harness setup: processes=%s, targets=%s source=%s steps=%d lr=%g",
        list(selected_processes),
        list(store.target_names),
        store.target_source,
        cfg.steps,
        cfg.learning_rate,
    )

    for process_name in selected_processes:
        process_data = process_data_by_name[process_name]
        step_fn = _make_compiled_step(
            process_data=process_data,
            modeled_feeds=modeled_feed_specs,
            learning_rate=float(cfg.learning_rate),
            solver_max_steps=int(cfg.solver_max_steps),
            solver_rtol=float(cfg.solver_rtol),
            solver_atol=float(cfg.solver_atol),
            solver_use_jump_ts=bool(cfg.solver_use_jump_ts),
        )
        t0 = time.perf_counter()
        _warmup_params, warmup_loss = step_fn(trainable_params, static_params)
        jax.block_until_ready(warmup_loss)
        compile_dt = time.perf_counter() - t0
        compiled_steps[process_name] = step_fn
        compiled_signatures[process_name] = _partition_signature(
            trainable_params,
            static_params,
        )
        compile_time_seconds_by_process[process_name] = compile_dt
        compile_count_by_process[process_name] = 1
        logger.info(
            "compiled train-step for process=%s in %.4fs (warmup loss=%.6g)",
            process_name,
            compile_dt,
            float(warmup_loss),
        )

    for step_index in range(cfg.steps):
        step_losses: list[float] = []
        for process_name in selected_processes:
            current_signature = _partition_signature(trainable_params, static_params)
            if current_signature != compiled_signatures[process_name]:
                process_data = process_data_by_name[process_name]
                step_fn = _make_compiled_step(
                    process_data=process_data,
                    modeled_feeds=modeled_feed_specs,
                    learning_rate=float(cfg.learning_rate),
                    solver_max_steps=int(cfg.solver_max_steps),
                    solver_rtol=float(cfg.solver_rtol),
                    solver_atol=float(cfg.solver_atol),
                    solver_use_jump_ts=bool(cfg.solver_use_jump_ts),
                )
                t_compile = time.perf_counter()
                _warmup_params, warmup_loss = step_fn(trainable_params, static_params)
                jax.block_until_ready(warmup_loss)
                compile_dt = time.perf_counter() - t_compile
                compiled_steps[process_name] = step_fn
                compiled_signatures[process_name] = current_signature
                compile_time_seconds_by_process[process_name] += compile_dt
                compile_count_by_process[process_name] += 1
                logger.warning(
                    "recompiled train-step for process=%s in %.4fs at step=%d",
                    process_name,
                    compile_dt,
                    step_index + 1,
                )

            step_fn = compiled_steps[process_name]
            t0 = time.perf_counter()
            trainable_params, loss = step_fn(trainable_params, static_params)
            jax.block_until_ready(loss)
            step_dt = time.perf_counter() - t0
            loss_scalar = float(loss)
            step_losses.append(loss_scalar)
            loss_by_process[process_name].append(loss_scalar)
            step_time_seconds_by_process[process_name].append(step_dt)

        mean_loss = sum(step_losses) / len(step_losses)
        mean_loss_by_step.append(mean_loss)
        if step_index == 0 or (step_index + 1) % max(cfg.log_every, 1) == 0:
            logger.info(
                "step %d/%d mean_loss=%.6g",
                step_index + 1,
                cfg.steps,
                mean_loss,
            )

    suspicious_step_spikes_by_process: dict[str, int] = {}
    for process_name in selected_processes:
        values = step_time_seconds_by_process[process_name]
        if not values:
            suspicious_step_spikes_by_process[process_name] = 0
            continue
        baseline = max(statistics.median(values), 1e-6)
        threshold = max(5.0 * baseline, 5e-2)
        n_spikes = sum(1 for value in values if value > threshold)
        suspicious_step_spikes_by_process[process_name] = int(n_spikes)
        if n_spikes > 0:
            logger.warning(
                "process=%s observed %d suspiciously slow steps (threshold=%.4fs); "
                "possible recompile or runtime stall",
                process_name,
                n_spikes,
                threshold,
            )

    total_compile_seconds = float(sum(compile_time_seconds_by_process.values()))
    total_compile_count = int(sum(compile_count_by_process.values()))
    total_step_seconds = float(
        sum(sum(values) for values in step_time_seconds_by_process.values())
    )
    logger.info(
        "timing summary: total_compile_count=%d total_compile_seconds=%.4fs "
        "total_step_seconds=%.4fs",
        total_compile_count,
        total_compile_seconds,
        total_step_seconds,
    )

    final_module = eqx.combine(trainable_params, static_params)
    return TrainHarnessResult(
        final_reaction_module=final_module,
        process_names=selected_processes,
        mean_loss_by_step=tuple(mean_loss_by_step),
        loss_by_process={
            process_name: tuple(losses)
            for process_name, losses in loss_by_process.items()
        },
        compile_time_seconds_by_process=compile_time_seconds_by_process,
        step_time_seconds_by_process={
            process_name: tuple(values)
            for process_name, values in step_time_seconds_by_process.items()
        },
        compile_count_by_process=compile_count_by_process,
        total_compile_seconds=total_compile_seconds,
        total_compile_count=total_compile_count,
        total_step_seconds=total_step_seconds,
        suspicious_step_spikes_by_process=suspicious_step_spikes_by_process,
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
