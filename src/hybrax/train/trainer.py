from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

import diffrax

from .controls_store import BatchControls
from .model_api import partition_trainable
from .training_data import PerProcessTrainingData
from .wrapper import HybridOdeWrapper


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
    wrapper: HybridOdeWrapper,
    *,
    t_eval: jax.Array,
    n_meas: int | jax.Array,
    y0: jax.Array,
    max_steps: int,
    rtol: float,
    atol: float,
    jump_ts: jax.Array | None,
) -> jax.Array:
    """Simulate state trajectories at possibly padded measurement timestamps.

    ``y0`` and the returned states are in **physical** space.  The wrapper
    integrates internally in scaled space and the results are un-scaled before
    returning.
    """
    n_meas_arr = jnp.asarray(n_meas, dtype=jnp.int32)
    t1 = t_eval[jnp.clip(n_meas_arr - 1, 0, t_eval.shape[0] - 1)]

    # Scale the initial state for the solver
    y0_scaled = wrapper.scale_state(y0)

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
            y0=y0_scaled,
            saveat=diffrax.SaveAt(ts=t_eval),
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
            throw=False,
        )
        # Un-scale back to physical space
        return jax.vmap(wrapper.unscale_state)(solution.ys)

    def _single_point(_) -> jax.Array:
        return jnp.repeat(y0[None, :], repeats=t_eval.shape[0], axis=0)

    return jax.lax.cond(n_meas_arr > 1, _solve_trajectory, _single_point, operand=None)


class _BatchIndexedControls(eqx.Module):
    """Lightweight per-sample controls adapter for batched loss evaluation."""

    batch_controls: BatchControls
    process_idx: jax.Array

    def eval(self, ts: float | jax.Array) -> jax.Array:
        return self.batch_controls.eval(self.process_idx, ts)

    def eval_derivative(self, ts: float | jax.Array) -> jax.Array:
        return self.batch_controls.eval_derivative(self.process_idx, ts)


def simulate_measurement_states(
    wrapper: HybridOdeWrapper,
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
    wrapper: HybridOdeWrapper,
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
    # Gather predicted target columns from the integrated state.
    # State layout: [c_species..., V_cont, B_modeled_cum_0, ...]
    # target_state_indices selects the species + cumulative-modeled-feed columns
    # (V_cont is in the state but not a loss target).
    y_pred = states[:, wrapper.target_state_indices]

    # Normalize per-target MSE by variance (all-zero targets get variance=1)
    sq_err = jnp.square(y_pred - y_meas) / wrapper.target_variance[None, :]
    masked_sq_err = jnp.where(meas_mask[:, None], sq_err, 0.0)
    n_active = jnp.maximum(jnp.sum(meas_mask), 1)
    # Per-target mean: [n_targets+1]
    per_target_loss = jnp.sum(masked_sq_err, axis=0) / n_active
    # Total: mean over all targets
    total_loss = jnp.mean(per_target_loss)
    return total_loss, per_target_loss


def single_process_measurement_loss(
    wrapper: HybridOdeWrapper,
    process_data: PerProcessTrainingData,
    *,
    max_solver_steps: int = 100_000,
    solver_rtol: float = 1e-5,
    solver_atol: float = 1e-7,
    solver_use_jump_ts: bool = True,
) -> jax.Array:
    """Compute masked MSE over padded measurement arrays for one process."""
    n_meas = int(process_data.n_meas)
    y_pred_padded = jnp.zeros_like(process_data.y_meas)
    states_active = simulate_measurement_states(
        wrapper,
        process_data,
        max_steps=max_solver_steps,
        rtol=solver_rtol,
        atol=solver_atol,
        use_jump_ts=solver_use_jump_ts,
    )
    # Gather predicted target columns from the integrated state.
    y_pred_active = states_active[:, wrapper.target_state_indices]
    y_pred_padded = y_pred_padded.at[:n_meas, :].set(y_pred_active)

    sq_err = (
        jnp.square(y_pred_padded - process_data.y_meas)
        / wrapper.target_variance[None, :]
    )
    mask = process_data.meas_mask[:, None]
    masked_sq_err = jnp.where(mask, sq_err, 0.0)

    n_active = jnp.maximum(jnp.sum(process_data.meas_mask), 1)
    per_target_loss = jnp.sum(masked_sq_err, axis=0) / n_active
    return jnp.mean(per_target_loss)


def _build_optimizer(
    optimizer_name: str, learning_rate
) -> optax.GradientTransformation:
    # learning_rate can be a float or an optax Schedule
    if isinstance(learning_rate, (int, float)):
        if float(learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
    name = str(optimizer_name)
    if name == "adam":
        base = optax.adam(learning_rate)
    elif name == "sgd":
        base = optax.sgd(learning_rate)
    else:
        raise ValueError("optimizer_name must be one of {'adam', 'sgd'}")
    # zero_nans handles ODE-solver failures (rare); clip_by_global_norm is the
    # safety net against blowups in the early epochs of neural-ODE training.
    # Bound 1000 is generous enough to leave normal updates untouched but
    # bounds catastrophic gradient explosions.
    return optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(1000.0),
        base,
    )


def single_process_train_step(
    wrapper: HybridOdeWrapper,
    process_data: PerProcessTrainingData,
    *,
    learning_rate: float = 1e-3,
    max_solver_steps: int = 100_000,
    solver_rtol: float = 1e-5,
    solver_atol: float = 1e-7,
    solver_use_jump_ts: bool = True,
) -> tuple[HybridOdeWrapper, jax.Array, eqx.Module]:
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
