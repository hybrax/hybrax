from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import diffrax

from .model_api import partition_trainable
from .training_data import PerProcessTrainingData
from .wrapper import LibraryRhsWrapper


def simulate_measurement_states(
    wrapper: LibraryRhsWrapper,
    process_data: PerProcessTrainingData,
    *,
    max_steps: int = 100_000,
) -> jax.Array:
    """Simulate full state trajectories at active measurement timestamps."""
    ts = process_data.active_t_meas
    if ts.size == 0:
        raise ValueError("process has no active measurement timestamps")
    if ts.size == 1:
        return process_data.y0[None, :]

    term = diffrax.ODETerm(lambda t, y, args: wrapper(t, y))
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(
        rtol=1e-5,
        atol=1e-7,
        jump_ts=process_data.controls.active_step_ts,
    )
    solution = diffrax.diffeqsolve(
        term,
        solver=solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=None,
        y0=process_data.y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )
    return solution.ys


def single_process_measurement_loss(
    wrapper: LibraryRhsWrapper,
    process_data: PerProcessTrainingData,
) -> jax.Array:
    """Compute masked MSE over padded measurement arrays for one process."""
    states_active = simulate_measurement_states(wrapper, process_data)
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

    n_meas = int(process_data.n_meas)
    y_pred_padded = jnp.full_like(process_data.y_meas, jnp.nan)
    y_pred_padded = y_pred_padded.at[:n_meas, :].set(y_pred_active)

    sq_err = jnp.square(y_pred_padded - process_data.y_meas)
    mask = process_data.meas_mask[:, None]
    masked_sq_err = jnp.where(mask, sq_err, 0.0)

    n_targets = process_data.y_meas.shape[1]
    denom = jnp.maximum(jnp.sum(process_data.meas_mask) * n_targets, 1)
    return jnp.sum(masked_sq_err) / denom


def single_process_train_step(
    wrapper: LibraryRhsWrapper,
    process_data: PerProcessTrainingData,
    *,
    learning_rate: float = 1e-3,
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
        return single_process_measurement_loss(candidate_wrapper, process_data)

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
