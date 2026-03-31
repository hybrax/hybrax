from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

import diffrax

from .controls_store import BatchControls
from .model_api import partition_trainable
from .training_data import PerProcessTrainingData, TrainingDataStore
from .wrapper import LibraryRhsWrapper


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


def _clamp_padded_time_rows(times: jax.Array, lengths: jax.Array) -> jax.Array:
    """Right-clamp padded rows to avoid NaNs in padded tails."""
    max_length = times.shape[1]
    last_index = jnp.clip(lengths - 1, 0, max_length - 1)
    last_values = times[jnp.arange(times.shape[0], dtype=jnp.int32), last_index]
    tail_mask = jnp.arange(max_length, dtype=jnp.int32)[None, :] >= lengths[:, None]
    return jnp.where(tail_mask, last_values[:, None], times)


def _simulate_measurement_states_on_grid(
    wrapper: LibraryRhsWrapper,
    *,
    t_eval: jax.Array,
    n_meas: int | jax.Array,
    y0: jax.Array,
    max_steps: int,
    rtol: float,
    atol: float,
    jump_ts: jax.Array | None,
) -> jax.Array:
    """Simulate state trajectories at possibly padded measurement timestamps."""
    n_meas_arr = jnp.asarray(n_meas, dtype=jnp.int32)
    t1 = t_eval[jnp.clip(n_meas_arr - 1, 0, t_eval.shape[0] - 1)]

    def _solve_trajectory(_) -> jax.Array:
        term = diffrax.ODETerm(lambda t, y, args: wrapper(t, y))
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(
            rtol=float(rtol),
            atol=float(atol),
            jump_ts=jump_ts,
        )
        solution = diffrax.diffeqsolve(
            term,
            solver=solver,
            t0=t_eval[0],
            t1=t1,
            dt0=None,
            y0=y0,
            saveat=diffrax.SaveAt(ts=t_eval),
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )
        return solution.ys

    def _single_point(_) -> jax.Array:
        return jnp.repeat(y0[None, :], repeats=t_eval.shape[0], axis=0)

    return jax.lax.cond(n_meas_arr > 1, _solve_trajectory, _single_point, operand=None)


class _BatchIndexedControls(eqx.Module):
    """Lightweight per-sample controls adapter for batched loss evaluation."""

    batch_controls: BatchControls
    process_idx: jax.Array

    def eval(self, ts: float | jax.Array) -> jax.Array:
        return self.batch_controls.eval(self.process_idx, ts)


def simulate_measurement_states(
    wrapper: LibraryRhsWrapper,
    process_data: PerProcessTrainingData,
    *,
    max_steps: int = 100_000,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    use_jump_ts: bool = True,
) -> jax.Array:
    """Simulate full state trajectories at active measurement timestamps."""
    ts = process_data.active_t_meas
    if ts.size == 0:
        raise ValueError("process has no active measurement timestamps")
    jump_ts = process_data.controls.active_step_ts if use_jump_ts else None
    return _simulate_measurement_states_on_grid(
        wrapper,
        t_eval=ts,
        n_meas=process_data.n_meas,
        y0=process_data.y0,
        max_steps=max_steps,
        rtol=rtol,
        atol=atol,
        jump_ts=jump_ts,
    )


def _measurement_loss_from_arrays(
    wrapper: LibraryRhsWrapper,
    *,
    t_meas: jax.Array,
    y_meas: jax.Array,
    meas_mask: jax.Array,
    n_meas: int | jax.Array,
    y0: jax.Array,
    jump_ts: jax.Array | None,
    max_solver_steps: int,
    solver_rtol: float,
    solver_atol: float,
) -> jax.Array:
    states = _simulate_measurement_states_on_grid(
        wrapper,
        t_eval=t_meas,
        n_meas=n_meas,
        y0=y0,
        max_steps=max_solver_steps,
        rtol=solver_rtol,
        atol=solver_atol,
        jump_ts=jump_ts,
    )
    state_species = states[:, :-1]
    y_pred = jnp.asarray(
        wrapper.reaction_module.observe(state_species),
        dtype=state_species.dtype,
    )
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    if y_pred.ndim != 2:
        raise ValueError("observe(...) output must be rank-2 `[n_meas, n_targets]`")
    if y_pred.shape[0] != state_species.shape[0]:
        raise ValueError("observe(...) output must preserve measurement-time axis")
    if y_pred.shape[1] != y_meas.shape[1]:
        raise ValueError(
            "observe(...) output target dimension must match process y_meas columns"
        )

    sq_err = jnp.square(y_pred - y_meas)
    masked_sq_err = jnp.where(meas_mask[:, None], sq_err, 0.0)
    denom = jnp.maximum(jnp.sum(meas_mask) * y_meas.shape[1], 1)
    return jnp.sum(masked_sq_err) / denom


def single_process_measurement_loss(
    wrapper: LibraryRhsWrapper,
    process_data: PerProcessTrainingData,
    *,
    max_solver_steps: int = 100_000,
    solver_rtol: float = 1e-5,
    solver_atol: float = 1e-7,
    solver_use_jump_ts: bool = True,
) -> jax.Array:
    """Compute masked MSE over padded measurement arrays for one process."""
    n_meas = int(process_data.n_meas)
    y_pred_padded = jnp.full_like(process_data.y_meas, jnp.nan)
    states_active = simulate_measurement_states(
        wrapper,
        process_data,
        max_steps=max_solver_steps,
        rtol=solver_rtol,
        atol=solver_atol,
        use_jump_ts=solver_use_jump_ts,
    )
    state_species_active = states_active[:, :-1]
    y_pred_active = jnp.asarray(
        wrapper.reaction_module.observe(state_species_active),
        dtype=state_species_active.dtype,
    )
    if y_pred_active.ndim == 1:
        y_pred_active = y_pred_active[:, None]
    if y_pred_active.ndim != 2:
        raise ValueError("observe(...) output must be rank-2 `[n_meas, n_targets]`")
    if y_pred_active.shape[0] != state_species_active.shape[0]:
        raise ValueError("observe(...) output must preserve measurement-time axis")
    if y_pred_active.shape[1] != process_data.y_meas.shape[1]:
        raise ValueError(
            "observe(...) output target dimension must match process y_meas columns"
        )

    y_pred_padded = y_pred_padded.at[:n_meas, :].set(y_pred_active)

    sq_err = jnp.square(y_pred_padded - process_data.y_meas)
    mask = process_data.meas_mask[:, None]
    masked_sq_err = jnp.where(mask, sq_err, 0.0)

    n_targets = process_data.y_meas.shape[1]
    denom = jnp.maximum(jnp.sum(process_data.meas_mask) * n_targets, 1)
    return jnp.sum(masked_sq_err) / denom


def batched_measurement_loss(
    wrapper: LibraryRhsWrapper,
    store: TrainingDataStore,
    process_indices: jax.Array,
    *,
    max_solver_steps: int = 100_000,
    solver_rtol: float = 1e-5,
    solver_atol: float = 1e-7,
    solver_use_jump_ts: bool = True,
) -> jax.Array:
    """Compute mean process loss over a process-index batch."""
    batch = store.gather_batch(process_indices)
    batch_controls = store.controls_store.as_batch_controls()
    batch_t_meas = _clamp_padded_time_rows(batch.t_meas, batch.n_meas)

    jump_ts_rows = None
    if solver_use_jump_ts:
        jump_ts_rows = _clamp_padded_time_rows(
            store.controls_store.step_ts[batch.process_indices],
            store.controls_store.step_ts_lengths[batch.process_indices],
        )

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


def _build_optimizer(
    optimizer_name: str, learning_rate: float
) -> optax.GradientTransformation:
    if float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    name = str(optimizer_name)
    if name == "adam":
        return optax.adam(learning_rate)
    if name == "sgd":
        return optax.sgd(learning_rate)
    raise ValueError("optimizer_name must be one of {'adam', 'sgd'}")


def batched_train_step(
    wrapper: LibraryRhsWrapper,
    store: TrainingDataStore,
    process_indices: jax.Array,
    *,
    optimizer_name: str = "adam",
    learning_rate: float = 1e-3,
    optimizer_state: optax.OptState | None = None,
    max_solver_steps: int = 100_000,
    solver_rtol: float = 1e-5,
    solver_atol: float = 1e-7,
    solver_use_jump_ts: bool = True,
) -> tuple[LibraryRhsWrapper, jax.Array, eqx.Module, optax.OptState]:
    """Run one Optax-backed batch update and return updated wrapper/loss/grads/state."""
    optimizer = _build_optimizer(optimizer_name, learning_rate)
    trainable, static = partition_trainable(wrapper.reaction_module)
    opt_state = (
        optimizer_state if optimizer_state is not None else optimizer.init(trainable)
    )

    def _loss_fn(trainable_params: eqx.Module) -> jax.Array:
        reaction_module = eqx.combine(trainable_params, static)
        candidate_wrapper = eqx.tree_at(
            lambda current: current.reaction_module,
            wrapper,
            reaction_module,
        )
        return batched_measurement_loss(
            candidate_wrapper,
            store,
            process_indices,
            max_solver_steps=max_solver_steps,
            solver_rtol=solver_rtol,
            solver_atol=solver_atol,
            solver_use_jump_ts=solver_use_jump_ts,
        )

    loss, grads = eqx.filter_value_and_grad(_loss_fn)(trainable)
    updates, next_opt_state = optimizer.update(grads, opt_state, params=trainable)
    trainable_updated = eqx.apply_updates(trainable, updates)
    reaction_module_updated = eqx.combine(trainable_updated, static)
    wrapper_updated = eqx.tree_at(
        lambda current: current.reaction_module,
        wrapper,
        reaction_module_updated,
    )
    return wrapper_updated, loss, grads, next_opt_state


def single_process_train_step(
    wrapper: LibraryRhsWrapper,
    process_data: PerProcessTrainingData,
    *,
    learning_rate: float = 1e-3,
    max_solver_steps: int = 100_000,
    solver_rtol: float = 1e-5,
    solver_atol: float = 1e-7,
    solver_use_jump_ts: bool = True,
) -> tuple[LibraryRhsWrapper, jax.Array, eqx.Module]:
    """Run one train step and return `(updated_wrapper, loss, trainable_grads)`."""
    trainable, static = partition_trainable(wrapper.reaction_module)

    def _loss_fn(trainable_params):
        reaction_module = eqx.combine(trainable_params, static)
        candidate_wrapper = eqx.tree_at(
            lambda current: current.reaction_module,
            wrapper,
            reaction_module,
        )
        return single_process_measurement_loss(
            candidate_wrapper,
            process_data,
            max_solver_steps=max_solver_steps,
            solver_rtol=solver_rtol,
            solver_atol=solver_atol,
            solver_use_jump_ts=solver_use_jump_ts,
        )

    loss, grads = eqx.filter_value_and_grad(_loss_fn)(trainable)
    updates = jtu.tree_map(
        lambda grad: None if grad is None else -float(learning_rate) * grad,
        grads,
    )
    trainable_updated = eqx.apply_updates(trainable, updates)
    reaction_module_updated = eqx.combine(trainable_updated, static)
    wrapper_updated = eqx.tree_at(
        lambda current: current.reaction_module,
        wrapper,
        reaction_module_updated,
    )
    return wrapper_updated, loss, grads
