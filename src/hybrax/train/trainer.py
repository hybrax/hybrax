from __future__ import annotations
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import diffrax

from .controls_store import BatchControls
from .training_data import BatchTrainingData, PerProcessTrainingData
from .wrapper import HybridOdeWrapper, SaveOutputs


SampleLossFn = Callable[..., tuple[jax.Array, jax.Array]]
BatchedLossFn = Callable[..., tuple[jax.Array, jax.Array, jax.Array]]


class SingleSampleResult(eqx.Module):
    """Single-sample default-loss evaluation plus solver-time observables."""

    total_loss: jax.Array
    per_target_loss: jax.Array
    states: jax.Array
    save_outputs: SaveOutputs


def clamp_padded_time_rows(times: jax.Array, lengths: jax.Array) -> jax.Array:
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
    t_eval = clamp_padded_time_rows(t_eval[None, :], n_meas_arr[None])[0]
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


def _solve_measurement_save_outputs_on_grid(
    wrapper: HybridOdeWrapper,
    *,
    t_eval: jax.Array,
    n_meas: int | jax.Array,
    y0: jax.Array,
    max_steps: int,
    rtol: float,
    atol: float,
    jump_ts: jax.Array | None,
) -> SaveOutputs:
    """Solve measurement grid once and return stacked save-time observables."""
    n_meas_arr = jnp.asarray(n_meas, dtype=jnp.int32)
    t_eval = clamp_padded_time_rows(t_eval[None, :], n_meas_arr[None])[0]
    t1 = t_eval[jnp.clip(n_meas_arr - 1, 0, t_eval.shape[0] - 1)]

    y0_scaled = wrapper.scale_state(y0)

    def _solve_trajectory(_) -> SaveOutputs:
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
            saveat=diffrax.SaveAt(ts=t_eval, fn=wrapper.save_outputs),
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
            throw=False,
        )
        return solution.ys

    def _single_point(_) -> SaveOutputs:
        single = wrapper.save_outputs(t_eval[0], y0_scaled)
        return jtu.tree_map(
            lambda leaf: jnp.repeat(leaf[None, ...], repeats=t_eval.shape[0], axis=0),
            single,
        )

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


def evaluate_sample_from_arrays(
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
) -> SingleSampleResult:
    save_outputs = _solve_measurement_save_outputs_on_grid(
        wrapper,
        t_eval=t_meas,
        n_meas=n_meas,
        y0=y0,
        max_steps=max_solver_steps,
        rtol=solver_rtol,
        atol=solver_atol,
        jump_ts=jump_ts,
    )
    states = save_outputs.states_physical
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
    return SingleSampleResult(
        total_loss=total_loss,
        per_target_loss=per_target_loss,
        states=states,
        save_outputs=save_outputs,
    )


def measurement_loss_from_arrays(
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
) -> tuple[jax.Array, jax.Array]:
    result = evaluate_sample_from_arrays(
        wrapper,
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
    return result.total_loss, result.per_target_loss


def build_batched_loss_fn_from_sample_loss(
    sample_loss_fn: SampleLossFn,
) -> BatchedLossFn:
    """Lift a per-sample loss fn to batched harness contract."""

    def _batched_loss_fn(
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
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        batch_t_meas = clamp_padded_time_rows(batch.t_meas, batch.n_meas)

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
        ) -> tuple[jax.Array, jax.Array]:
            controls = _BatchIndexedControls(
                batch_controls=batch_controls,
                process_idx=process_idx,
            )
            sample_wrapper = eqx.tree_at(
                lambda w: (w.controls, w.rhs_ode.Cin, w.rhs_ode.Cin_modeled),
                wrapper,
                (controls, cin, cin_modeled),
            )
            total_loss, per_target = sample_loss_fn(
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

        per_sample_total, per_sample_per_target = jax.vmap(
            _sample_loss,
            # when `jump_ts_rows` is `None` we also have to pass `None` to vmap so that
            # it's not iterated over
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None if jump_ts_rows is None else 0),
        )(
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
        mean_per_target = jnp.mean(per_sample_per_target, axis=0)
        mean_total = jnp.mean(per_sample_total)
        return mean_total, mean_per_target, per_sample_total

    return _batched_loss_fn
