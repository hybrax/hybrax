from __future__ import annotations
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from .controls_store import BatchControls
from .model_api import LossInputs
from .training_data import BatchTrainingData
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
    # Independent prediction-grid trajectory (forward-only output grid), distinct
    # from the loss module's own dense view. Populated only when
    # ``prediction_grid_n`` is requested; ``None`` otherwise (e.g. training).
    prediction_t: jax.Array | None = None
    prediction_save_outputs: SaveOutputs | None = None


def clamp_padded_time_rows(times: jax.Array, lengths: jax.Array) -> jax.Array:
    """Right-clamp padded rows to avoid NaNs in padded tails."""
    max_length = times.shape[1]
    last_index = jnp.clip(lengths - 1, 0, max_length - 1)
    last_values = times[jnp.arange(times.shape[0], dtype=jnp.int32), last_index]
    tail_mask = jnp.arange(max_length, dtype=jnp.int32)[None, :] >= lengths[:, None]
    return jnp.where(tail_mask, last_values[:, None], times)


def _solve_measurement_save_outputs_on_grid(
    wrapper: HybridOdeWrapper,
    *,
    t_eval: jax.Array,
    n_measured: int | jax.Array,
    RAW_y0: jax.Array,
    max_steps: int,
    rtol: float,
    atol: float,
    jump_ts: jax.Array | None,
) -> SaveOutputs:
    """Solve measurement grid once and return stacked save-time observables.

    ``RAW_y0`` is the physical-layout initial state. The wrapper builds the
    internal pseudobatch solver state. ``SaveOutputs.SCL_states`` stays in
    physical public layout; rate fields are RAW.
    """
    # Bounded physical-state solve (manual jumps at events) — well-conditioned
    # gradient. Replaces the pseudobatch single-solve whose unbounded accumulator
    # corrupted the adjoint (see spec/pseudo_diagnosis.md).
    from .physical_solve import solve_physical_states

    n_meas_arr = jnp.asarray(n_measured, dtype=jnp.int32)
    t_eval = clamp_padded_time_rows(t_eval[None, :], n_meas_arr[None])[0]
    states = solve_physical_states(
        wrapper,
        t_eval=t_eval,
        n_measured=n_meas_arr,
        RAW_y0=RAW_y0,
        max_steps=max_steps,
        rtol=float(rtol),
        atol=float(atol),
    )
    return jax.vmap(wrapper.physical_save_outputs)(t_eval, states)


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

    @property
    def sample_event_times(self) -> jax.Array:
        return self.batch_controls.sample_event_times[self.process_idx]

    @property
    def sample_event_volumes(self) -> jax.Array:
        return self.batch_controls.sample_event_volumes[self.process_idx]

    @property
    def sample_event_mask(self) -> jax.Array:
        return self.batch_controls.sample_event_mask[self.process_idx]

    @property
    def bolus_event_times(self) -> jax.Array:
        return self.batch_controls.bolus_event_times[self.process_idx]

    @property
    def bolus_event_volumes(self) -> jax.Array:
        return self.batch_controls.bolus_event_volumes[self.process_idx]

    @property
    def bolus_event_Cin(self) -> jax.Array:
        return self.batch_controls.bolus_event_Cin[self.process_idx]

    @property
    def bolus_event_mask(self) -> jax.Array:
        return self.batch_controls.bolus_event_mask[self.process_idx]


def evaluate_sample_with_loss_module(
    wrapper: HybridOdeWrapper,
    *,
    t_measured: jax.Array,
    SCL_target_measured: jax.Array,
    mask_measured: jax.Array,
    n_measured: int | jax.Array,
    RAW_y0_measured: jax.Array,
    jump_ts: jax.Array | None,
    max_solver_steps: int,
    solver_rtol: float,
    solver_atol: float,
    step: int | jax.Array | None = None,
    prediction_grid_n: int | None = None,
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
    wrapper). ``RAW_y0_measured`` is the full physical initial state.
    """
    module = wrapper.reaction_module
    loss_module = wrapper.loss_module
    dense_grid_n = loss_module.dense_grid_n

    def _views_from_save_outputs(save_outputs):
        """Build SCL/RAW pairs from a SaveOutputs whose leading dim is whatever
        time grid we solved on. Returns a dict of arrays keyed by LossInputs
        field name (without the SCL_target_pred slice -- that's handled by the
        caller using ``target_state_indices``)."""
        SCL_states = save_outputs.SCL_states
        RAW_rates = save_outputs.RAW_modeled_BiologicalOde_rates
        RAW_fvc_rates = save_outputs.RAW_modeled_FVCs_rates
        RAW_V = save_outputs.RAW_V
        return {
            "SCL_states": SCL_states,
            "RAW_states": module.unscale_state(SCL_states),
            "SCL_modeled_BiologicalOde_rates": module.scale_modeled_BiologicalOde_rates(
                RAW_rates
            ),
            "RAW_modeled_BiologicalOde_rates": RAW_rates,
            "SCL_modeled_FVCs_rates": module.scale_modeled_FVCs_rates(RAW_fvc_rates),
            "RAW_modeled_FVCs_rates": RAW_fvc_rates,
            "SCL_V": module.scale_modeled_V(RAW_V),
            "RAW_V": RAW_V,
            "auxiliary": save_outputs.auxiliary or {},
        }

    if dense_grid_n is None and prediction_grid_n is None:
        # Measurement-grid-only path (unchanged behavior).
        save_outputs = _solve_measurement_save_outputs_on_grid(
            wrapper,
            t_eval=t_measured,
            n_measured=n_measured,
            RAW_y0=RAW_y0_measured,
            max_steps=max_solver_steps,
            rtol=solver_rtol,
            atol=solver_atol,
            jump_ts=jump_ts,
        )
        meas_views = _views_from_save_outputs(save_outputs)
        dense_t = None
        dense_views = {key: None for key in meas_views}
        # auxiliary is already dict-typed in meas_views; mirror as None on dense.
        dense_views["auxiliary"] = None
        prediction_t = None
        prediction_save_outputs = None
    else:
        # Dense/prediction path: solve ONCE on the union of the measurement grid,
        # the loss module's dense grid, and the forward prediction grid, then
        # index-split. Loss reads the dense block; forward reads the prediction
        # block. SaveAt just records at more points, so the loss value is
        # unchanged.
        from .dense import build_union_time_grid

        (
            t_eval,
            sample_idx,
            dense_t,
            dense_idx,
            prediction_t,
            prediction_idx,
        ) = build_union_time_grid(
            t_measured,
            n_measured,
            n_dense=None if dense_grid_n is None else int(dense_grid_n),
            n_prediction=(
                None if prediction_grid_n is None else int(prediction_grid_n)
            ),
        )
        save_outputs = _solve_measurement_save_outputs_on_grid(
            wrapper,
            t_eval=t_eval,
            n_measured=t_eval.shape[0],
            RAW_y0=RAW_y0_measured,
            max_steps=max_solver_steps,
            rtol=solver_rtol,
            atol=solver_atol,
            jump_ts=jump_ts,
        )
        sample_save_outputs = jtu.tree_map(lambda leaf: leaf[sample_idx], save_outputs)
        meas_views = _views_from_save_outputs(sample_save_outputs)
        if dense_grid_n is None:
            dense_views = {key: None for key in meas_views}
            dense_views["auxiliary"] = None
        else:
            dense_save_outputs = jtu.tree_map(
                lambda leaf: leaf[dense_idx], save_outputs
            )
            dense_views = _views_from_save_outputs(dense_save_outputs)
        if prediction_grid_n is None:
            prediction_save_outputs = None
        else:
            prediction_save_outputs = jtu.tree_map(
                lambda leaf: leaf[prediction_idx], save_outputs
            )

    SCL_states = meas_views["SCL_states"]
    # State layout: [modeled_RMCs | V_in_cumulative | modeled_FVCs_cumulative].
    # target_state_indices selects modeled_RMCs + modeled_FVCs_cumulative.
    SCL_target_pred = SCL_states[:, wrapper.target_state_indices]

    mask_any = jnp.any(mask_measured, axis=1).astype(SCL_states.dtype)
    step_arr = jnp.asarray(step if step is not None else -1, dtype=jnp.int32)

    inputs = LossInputs(
        SCL_states=SCL_states,
        RAW_states=meas_views["RAW_states"],
        SCL_modeled_BiologicalOde_rates=meas_views["SCL_modeled_BiologicalOde_rates"],
        RAW_modeled_BiologicalOde_rates=meas_views["RAW_modeled_BiologicalOde_rates"],
        SCL_modeled_FVCs_rates=meas_views["SCL_modeled_FVCs_rates"],
        RAW_modeled_FVCs_rates=meas_views["RAW_modeled_FVCs_rates"],
        SCL_V=meas_views["SCL_V"],
        RAW_V=meas_views["RAW_V"],
        auxiliary=meas_views["auxiliary"],
        SCL_target_pred=SCL_target_pred,
        SCL_target_measured=SCL_target_measured,
        mask_measured=mask_measured,
        mask_measured_any=mask_any,
        t_measured=t_measured,
        n_measured=jnp.asarray(n_measured, dtype=jnp.int32),
        reaction_module=module,
        step=step_arr,
        jump_ts=jump_ts,
        dense_t=dense_t,
        dense_SCL_states=dense_views["SCL_states"],
        dense_RAW_states=dense_views["RAW_states"],
        dense_SCL_modeled_BiologicalOde_rates=dense_views[
            "SCL_modeled_BiologicalOde_rates"
        ],
        dense_RAW_modeled_BiologicalOde_rates=dense_views[
            "RAW_modeled_BiologicalOde_rates"
        ],
        dense_SCL_modeled_FVCs_rates=dense_views["SCL_modeled_FVCs_rates"],
        dense_RAW_modeled_FVCs_rates=dense_views["RAW_modeled_FVCs_rates"],
        dense_SCL_V=dense_views["SCL_V"],
        dense_RAW_V=dense_views["RAW_V"],
        dense_auxiliary=dense_views["auxiliary"],
    )

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
        prediction_t=prediction_t,
        prediction_save_outputs=prediction_save_outputs,
    )


def evaluate_one_sample_loss(
    wrapper: HybridOdeWrapper,
    batch_controls: BatchControls,
    process_idx: jax.Array,
    t_measured: jax.Array,
    SCL_target_measured: jax.Array,
    mask_measured: jax.Array,
    n_measured: jax.Array,
    RAW_y0_measured: jax.Array,
    cin: jax.Array,
    cin_modeled: jax.Array,
    jump_ts: jax.Array | None,
    *,
    max_solver_steps: int,
    solver_rtol: float,
    solver_atol: float,
    step: int | jax.Array | None = None,
):
    """One process -> ``(total_loss, per_target_loss)``. Shared by the vmap
    batched loss and the device-sharded (pmap) step. ``y0`` stays RAW physical;
    the callbacks solve (``physical_solve.solve_physical_states``) integrates it."""
    controls = _BatchIndexedControls(batch_controls=batch_controls, process_idx=process_idx)
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
        RAW_y0_measured=RAW_y0_measured,
        jump_ts=jump_ts,
        max_solver_steps=max_solver_steps,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
        step=step,
    )
    return result.total_loss, result.per_target_loss


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
        prediction_grid_n: int | None = None,
    ) -> tuple:
        batch_t_meas = clamp_padded_time_rows(batch.t_measured, batch.n_measured)

        # Pre-scale measurements to SCL once per batch. Initial states stay RAW.
        SCALE_state = wrapper.reaction_module.SCALE_state
        SCALE_state_for_targets = SCALE_state[wrapper.target_state_indices]
        SCL_y_measured = batch.y_measured / SCALE_state_for_targets[None, None, :]

        def _sample_loss(
            process_idx: jax.Array,
            t_measured: jax.Array,
            SCL_target_measured: jax.Array,
            mask_measured: jax.Array,
            n_measured: jax.Array,
            RAW_y0_measured: jax.Array,
            cin: jax.Array,
            cin_modeled: jax.Array,
            jump_ts: jax.Array | None,
        ) -> tuple:
            controls = _BatchIndexedControls(
                batch_controls=batch_controls,
                process_idx=process_idx,
            )
            sample_wrapper = eqx.tree_at(
                lambda w: (
                    w.controls,
                    w.rhs_ode.Cin_controlled_FVCs,
                    w.rhs_ode.Cin_modeled_FVCs,
                ),
                wrapper,
                (controls, cin, cin_modeled),
            )
            result = evaluate_sample_with_loss_module(
                sample_wrapper,
                t_measured=t_measured,
                SCL_target_measured=SCL_target_measured,
                mask_measured=mask_measured,
                n_measured=n_measured,
                RAW_y0_measured=RAW_y0_measured,
                jump_ts=jump_ts,
                max_solver_steps=max_solver_steps,
                solver_rtol=solver_rtol,
                solver_atol=solver_atol,
                step=step,
                prediction_grid_n=prediction_grid_n,
            )
            return (
                result.total_loss,
                result.per_target_loss,
                result.prediction_t,
                result.prediction_save_outputs,
            )

        (
            per_sample_total,
            per_sample_per_target,
            prediction_t,
            prediction_save_outputs,
        ) = jax.vmap(
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
            batch.y0_measured,
            batched_Cin[batch.process_indices],
            batched_Cin_modeled[batch.process_indices],
            jump_ts_rows,
        )
        mean_total = jnp.mean(per_sample_total)
        # 2nd element is PER-SAMPLE per-target (one row per process), matching the
        # forward harvest path (compute_dense_exports); the vmap training step
        # means it. Last two are non-None only when forward requested
        # `prediction_grid_n`; training ignores them.
        return (
            mean_total,
            per_sample_per_target,
            per_sample_total,
            prediction_t,
            prediction_save_outputs,
        )

    return _batched_loss_fn
