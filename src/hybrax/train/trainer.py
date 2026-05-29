from __future__ import annotations
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import diffrax

from .controls_store import BatchControls
from .model_api import LossInputs
from .training_data import BatchTrainingData, PerProcessTrainingData
from .wrapper import HybridOdeWrapper, SaveOutputs


BatchedLossFn = Callable[..., tuple[jax.Array, jax.Array, jax.Array]]


class SingleSampleResult(eqx.Module):
    """Single-sample default-loss evaluation plus solver-time observables."""

    total_loss: jax.Array
    per_target_loss: jax.Array
    states: jax.Array
    save_outputs: SaveOutputs
    # -1 means no training step was supplied, e.g. forward evaluation.
    step: jax.Array = eqx.field(
        default_factory=lambda: jnp.asarray(-1, dtype=jnp.int32)
    )


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
    n_measured: int | jax.Array,
    RAW_y0: jax.Array,
    max_steps: int,
    rtol: float,
    atol: float,
    jump_ts: jax.Array | None,
) -> jax.Array:
    """Simulate state trajectories at possibly padded measurement timestamps.

    ``RAW_y0`` and the returned states are in **physical** (RAW) space. The
    wrapper integrates internally in SCL space; this helper scales the initial
    state on the way in and un-scales the saved trajectory on the way out.
    """
    n_meas_arr = jnp.asarray(n_measured, dtype=jnp.int32)
    t_eval = clamp_padded_time_rows(t_eval[None, :], n_meas_arr[None])[0]
    t1 = t_eval[jnp.clip(n_meas_arr - 1, 0, t_eval.shape[0] - 1)]

    module = wrapper.reaction_module
    SCL_y0 = module.scale_state(RAW_y0)

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
            y0=SCL_y0,
            saveat=diffrax.SaveAt(ts=t_eval),
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
            throw=False,
        )
        # Un-scale saved SCL states back to RAW for the export-side caller.
        return jax.vmap(module.unscale_state)(solution.ys)

    def _single_point(_) -> jax.Array:
        return jnp.repeat(RAW_y0[None, :], repeats=t_eval.shape[0], axis=0)

    return jax.lax.cond(n_meas_arr > 1, _solve_trajectory, _single_point, operand=None)


def _solve_measurement_save_outputs_on_grid(
    wrapper: HybridOdeWrapper,
    *,
    t_eval: jax.Array,
    n_measured: int | jax.Array,
    SCL_y0: jax.Array,
    max_steps: int,
    rtol: float,
    atol: float,
    jump_ts: jax.Array | None,
) -> SaveOutputs:
    """Solve measurement grid once and return stacked save-time observables.

    ``SCL_y0`` is already in SCL space (the trainer pre-scales measurements
    via the module's helpers). ``SaveOutputs.SCL_states`` is the SCL trajectory;
    rate fields are RAW.
    """
    n_meas_arr = jnp.asarray(n_measured, dtype=jnp.int32)
    t_eval = clamp_padded_time_rows(t_eval[None, :], n_meas_arr[None])[0]
    t1 = t_eval[jnp.clip(n_meas_arr - 1, 0, t_eval.shape[0] - 1)]

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
            y0=SCL_y0,
            saveat=diffrax.SaveAt(ts=t_eval, fn=wrapper.save_outputs),
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
            throw=False,
        )
        return solution.ys

    def _single_point(_) -> SaveOutputs:
        single = wrapper.save_outputs(t_eval[0], SCL_y0)
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

    def eval_u(self, ts: float | jax.Array) -> jax.Array:
        return self.batch_controls.eval_u(self.process_idx, ts)


def simulate_measurement_states(
    wrapper: HybridOdeWrapper,
    process_data: PerProcessTrainingData,
    *,
    max_steps: int = 100_000,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    use_jump_ts: bool = True,
) -> jax.Array:
    """Simulate full state trajectories at active measurement timestamps.

    Returns RAW (physical) state for export / plotting. The stored
    ``y0_measured`` on ``process_data`` is RAW physical.
    """
    ts = process_data.active_t_measured
    if ts.size == 0:
        raise ValueError("process has no active measurement timestamps")
    jump_ts = process_data.controls.active_step_ts if use_jump_ts else None
    return _simulate_measurement_states_on_grid(
        wrapper,
        t_eval=ts,
        n_measured=process_data.n_measured,
        RAW_y0=process_data.y0_measured,
        max_steps=max_steps,
        rtol=rtol,
        atol=atol,
        jump_ts=jump_ts,
    )


def evaluate_sample_with_loss_module(
    wrapper: HybridOdeWrapper,
    *,
    t_measured: jax.Array,
    SCL_target_measured: jax.Array,
    mask_measured: jax.Array,
    n_measured: int | jax.Array,
    SCL_target0_measured: jax.Array,
    jump_ts: jax.Array | None,
    max_solver_steps: int,
    solver_rtol: float,
    solver_atol: float,
    step: int | jax.Array | None = None,
) -> SingleSampleResult:
    """Solve once, build ``LossInputs``, and evaluate ``wrapper.loss_module``.

    The single shared ODE solve produces ``save_outputs`` (SCL states, RAW
    rates / volume). We derive the SCL/RAW trajectory pairs by elementwise
    broadcast against the reaction module's ``SCALE_*`` (no vmap), build a
    :class:`LossInputs`, and call the loss module. ``per_target_loss`` is the
    stacked ``named_losses`` in ``loss_module.loss_names`` order; ``total_loss``
    is their sum.

    ``SCL_target_measured`` is the measurement matrix already divided by
    ``module.SCALE_state[target_state_indices]`` (pre-scaled by the batched
    wrapper). ``SCL_target0_measured`` is the FULL SCL initial state matching
    the integration state layout.
    """
    module = wrapper.reaction_module
    save_outputs = _solve_measurement_save_outputs_on_grid(
        wrapper,
        t_eval=t_measured,
        n_measured=n_measured,
        SCL_y0=SCL_target0_measured,
        max_steps=max_solver_steps,
        rtol=solver_rtol,
        atol=solver_atol,
        jump_ts=jump_ts,
    )
    SCL_states = save_outputs.SCL_states
    # State layout: [modeled_RMCs | V_in_cumulative | modeled_FVCs_cumulative].
    # target_state_indices selects modeled_RMCs + modeled_FVCs_cumulative.
    SCL_target_pred = SCL_states[:, wrapper.target_state_indices]

    # Derive SCL/RAW trajectory pairs. Scaling is elementwise → broadcasts over
    # the leading time axis: (n_meas, k) op (k,), (n_meas,) op scalar.
    RAW_states = module.unscale_state(SCL_states)
    RAW_rates = save_outputs.RAW_modeled_BiologicalOde_rates
    SCL_rates = module.scale_modeled_BiologicalOde_rates(RAW_rates)
    RAW_fvc_rates = save_outputs.RAW_modeled_FVCs_rates
    SCL_fvc_rates = module.scale_modeled_FVCs_rates(RAW_fvc_rates)
    RAW_V = save_outputs.RAW_V
    SCL_V = module.scale_modeled_V(RAW_V)

    mask_any = jnp.any(mask_measured, axis=1).astype(SCL_states.dtype)
    auxiliary = save_outputs.auxiliary or {}
    step_arr = jnp.asarray(step if step is not None else -1, dtype=jnp.int32)

    inputs = LossInputs(
        SCL_states=SCL_states,
        RAW_states=RAW_states,
        SCL_modeled_BiologicalOde_rates=SCL_rates,
        RAW_modeled_BiologicalOde_rates=RAW_rates,
        SCL_modeled_FVCs_rates=SCL_fvc_rates,
        RAW_modeled_FVCs_rates=RAW_fvc_rates,
        SCL_V=SCL_V,
        RAW_V=RAW_V,
        auxiliary=auxiliary,
        SCL_target_pred=SCL_target_pred,
        SCL_target_measured=SCL_target_measured,
        mask_measured=mask_measured,
        mask_measured_any=mask_any,
        t_measured=t_measured,
        n_measured=jnp.asarray(n_measured, dtype=jnp.int32),
        reaction_module=module,
        step=step_arr,
    )

    loss_module = wrapper.loss_module
    outputs = loss_module(inputs)
    named = outputs.named_losses
    per_target_loss = jnp.stack([named[name] for name in loss_module.loss_names])
    total_loss = jnp.mean(per_target_loss)
    return SingleSampleResult(
        total_loss=total_loss,
        per_target_loss=per_target_loss,
        states=SCL_states,
        save_outputs=save_outputs,
        step=step_arr,
    )


def build_batched_loss_fn() -> BatchedLossFn:
    """Build the batched loss fn that evaluates ``wrapper.loss_module``.

    The returned fn receives RAW measurements (``batch.y_measured`` /
    ``batch.y0_measured``); it pre-scales them via the module's ``SCALE_state``,
    then vmaps :func:`evaluate_sample_with_loss_module` over the batch. The loss
    module is read off the wrapper, so swapping it (or its trainable params)
    flows through without rebuilding this closure.
    """

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
        step: int | jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        batch_t_meas = clamp_padded_time_rows(batch.t_measured, batch.n_measured)

        # Pre-scale measurements and initial states to SCL once per batch.
        SCALE_state = wrapper.reaction_module.SCALE_state
        SCALE_state_for_targets = SCALE_state[wrapper.target_state_indices]
        SCL_y_measured = batch.y_measured / SCALE_state_for_targets[None, None, :]
        SCL_y0_measured = batch.y0_measured / SCALE_state[None, :]

        def _sample_loss(
            process_idx: jax.Array,
            t_measured: jax.Array,
            SCL_target_measured: jax.Array,
            mask_measured: jax.Array,
            n_measured: jax.Array,
            SCL_target0_measured: jax.Array,
            cin: jax.Array,
            cin_modeled: jax.Array,
            jump_ts: jax.Array | None,
        ) -> tuple[jax.Array, jax.Array]:
            controls = _BatchIndexedControls(
                batch_controls=batch_controls,
                process_idx=process_idx,
            )
            sample_wrapper = eqx.tree_at(
                lambda w: (w.controls, w.rhs_ode.Cin_controlled_FVCs, w.rhs_ode.Cin_modeled_FVCs),
                wrapper,
                (controls, cin, cin_modeled),
            )
            result = evaluate_sample_with_loss_module(
                sample_wrapper,
                t_measured=t_measured,
                SCL_target_measured=SCL_target_measured,
                mask_measured=mask_measured,
                n_measured=n_measured,
                SCL_target0_measured=SCL_target0_measured,
                jump_ts=jump_ts,
                max_solver_steps=max_solver_steps,
                solver_rtol=solver_rtol,
                solver_atol=solver_atol,
                step=step,
            )
            return result.total_loss, result.per_target_loss

        per_sample_total, per_sample_per_target = jax.vmap(
            _sample_loss,
            # when `jump_ts_rows` is `None` we also have to pass `None` to vmap so that
            # it's not iterated over
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None if jump_ts_rows is None else 0),
        )(
            batch.process_indices,
            batch_t_meas,
            SCL_y_measured,
            batch.mask_measured,
            batch.n_measured,
            SCL_y0_measured,
            batched_Cin[batch.process_indices],
            batched_Cin_modeled[batch.process_indices],
            jump_ts_rows,
        )
        mean_per_target = jnp.mean(per_sample_per_target, axis=0)
        mean_total = jnp.mean(per_sample_total)
        return mean_total, mean_per_target, per_sample_total

    return _batched_loss_fn
