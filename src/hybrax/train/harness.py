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
from bpbench.dataclasses import BioProcessCollection
from bpbench.mechanistic import get_rhs_ode
from bpbench.serialization import load_process_collection_json

from .defaults import default_build_reaction_module
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
from .logging import RunLogger, StepRecord
from .training_data import (
    BatchTrainingData,
    TARGET_SOURCE_AUTO,
    TrainingDataStore,
)
from .utils import get_hook, load_custom_module, resolve_config
from .wrapper import HybridOdeWrapper, validate_rhs_ode_compatibility


logger = logging.getLogger(__name__)


def _batched_measurement_loss_from_batch(
    wrapper: HybridOdeWrapper,
    batch: BatchTrainingData,
    batch_controls: BatchControls,
    batched_Cin: jax.Array,
    batched_Cin_modeled: jax.Array,
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
        cin: jax.Array,
        cin_modeled: jax.Array,
        jump_ts: jax.Array | None,
    ) -> jax.Array:
        controls = _BatchIndexedControls(
            batch_controls=batch_controls,
            process_idx=process_idx,
        )
        sample_wrapper = eqx.tree_at(
            lambda w: (w.controls, w.rhs_ode.Cin, w.rhs_ode.Cin_modeled),
            wrapper,
            (controls, cin, cin_modeled),
        )
        total_loss, per_target = _measurement_loss_from_arrays(
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
        return total_loss, per_target

    if jump_ts_rows is None:
        per_sample_total, per_sample_per_target = jax.vmap(
            lambda pi, tm, ym, mm, nm, y0, ci, cm: _sample_loss(
                pi,
                tm,
                ym,
                mm,
                nm,
                y0,
                ci,
                cm,
                None,
            )
        )(
            batch.process_indices,
            batch_t_meas,
            batch.y_meas,
            batch.meas_mask,
            batch.n_meas,
            batch.y0,
            batched_Cin[batch.process_indices],
            batched_Cin_modeled[batch.process_indices],
        )
    else:
        per_sample_total, per_sample_per_target = jax.vmap(_sample_loss)(
            batch.process_indices,
            batch_t_meas,
            batch.y_meas,
            batch.meas_mask,
            batch.n_meas,
            batch.y0,
            batched_Cin[batch.process_indices],
            batched_Cin_modeled[batch.process_indices],
            jump_ts_rows,
        )
    # per_sample_per_target: [batch_size, n_targets]
    # per_sample_total: [batch_size]
    mean_per_target = jnp.mean(per_sample_per_target, axis=0)
    mean_total = jnp.mean(per_sample_total)
    # Return the unreduced per-sample vector too — the harness wants it for
    # per-process logging, and it costs nothing extra inside the JIT graph.
    return mean_total, mean_per_target, per_sample_total


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
    if isinstance(config.learning_rate, (int, float)) and config.learning_rate <= 0.0:
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
) -> TrainHarnessResult:
    """Train one reaction module over one or many processes from one store."""
    cfg = config or TrainHarnessConfig()
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
    _variance_eps = 1e-12
    target_variance = jnp.asarray(
        [
            float(np.var(vals)) if vals and float(np.var(vals)) > _variance_eps else 1.0
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
    optimizer = _build_optimizer(cfg.optimizer_name, cfg.learning_rate)
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
                total_loss, per_target, per_sample = (
                    _batched_measurement_loss_from_batch(
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
        "batch_size=%d optimizer=%s lr=%s",
        list(selected_processes),
        list(store.target_names),
        store.target_source,
        cfg.steps,
        effective_batch_size,
        cfg.optimizer_name,
        cfg.learning_rate,
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
                )
            )

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
        collection=collection,
    )
    # Call optional build_learning_rate hook (overrides CLI --learning-rate)
    lr_hook = get_hook(custom_module, "build_learning_rate", None)
    if lr_hook is not None:
        lr = lr_hook(custom_cfg)
        effective_cfg = TrainHarnessConfig(
            process_names=effective_cfg.process_names,
            target_variable_order=effective_cfg.target_variable_order,
            target_source=effective_cfg.target_source,
            steps=effective_cfg.steps,
            batch_size=effective_cfg.batch_size,
            shuffle_batches=effective_cfg.shuffle_batches,
            batch_seed=effective_cfg.batch_seed,
            optimizer_name=effective_cfg.optimizer_name,
            learning_rate=lr,
            seed=effective_cfg.seed,
            log_every=effective_cfg.log_every,
            solver_max_steps=effective_cfg.solver_max_steps,
            solver_rtol=effective_cfg.solver_rtol,
            solver_atol=effective_cfg.solver_atol,
            solver_use_jump_ts=effective_cfg.solver_use_jump_ts,
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
        config=effective_cfg,
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
