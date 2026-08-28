"""The JIT-compiled per-sample solve and per-batch loss/gradient step.

``build_batched_loss_fn`` builds the function ``harness.py`` JIT-compiles for
every train step: it vmaps a single sample's ODE solve plus loss module
evaluation (``evaluate_sample_with_loss_module``) across a batch, and
``evaluate_one_sample_loss`` provides the equivalent single-sample path used
outside training (holdout evaluation, dense exports).
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from .controls_store import BatchControls
from .model_api import LossInputs
from .physical_solve import within_fail_time
from .training_data import (
    BatchTrainingData,
    PerProcessTrainingData,
    replace_rhs_ode_process_matrices,
)
from .wrapper import HybridOdeWrapper, SaveOutputs


class SingleSampleResult(eqx.Module):
    """Single-sample default-loss evaluation plus solver-time observables."""

    total_loss: jax.Array
    per_target_loss: jax.Array
    states: jax.Array
    save_outputs: SaveOutputs
    # Measurement-grid view for export only. Loss inputs keep using ``save_outputs``
    # and the finite placeholders produced by the physical solve.
    measurement_save_outputs: SaveOutputs
    # Row-level solve validity, before observation availability is applied.
    measurement_prediction_valid: jax.Array
    # Time of this sample's first failed ODE segment (``+inf`` if the solve
    # never bailed). A finite value flags a partially-failed lane; the harness
    # counts finite entries per step to report how often segments fail. Always
    # produced by the solve, so it is required (no silent clean-solve default).
    fail_time: jax.Array
    # -1 means no training step was supplied, e.g. forward evaluation.
    step: jax.Array = eqx.field(
        default_factory=lambda: jnp.asarray(-1, dtype=jnp.int32)
    )
    # Independent prediction-grid trajectory (forward-only output grid), distinct
    # from the loss module's own dense view. Populated only when
    # ``prediction_grid_n`` is requested; ``None`` otherwise (e.g. training).
    prediction_t: jax.Array | None = None
    prediction_save_outputs: SaveOutputs | None = None
    # Validity on ``prediction_t`` from this evaluation's own solve.
    prediction_valid: jax.Array | None = None


def clamp_padded_time_rows(times: jax.Array, lengths: jax.Array) -> jax.Array:
    """Right-clamp padded rows to avoid NaNs in padded tails."""
    max_length = times.shape[1]
    if max_length == 0:
        # No jump times anywhere (e.g. no discrete events) — nothing to clamp;
        # the empty row reaches the solver as ``jump_ts=None`` (a plain solve).
        return times
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
    n_linspace: int | None = None,
) -> tuple[SaveOutputs, jax.Array]:
    """Solve measurement grid once; return save-time observables and ``fail_time``.

    ``RAW_y0`` is the physical-layout initial state. The wrapper builds the
    internal pseudobatch solver state. ``SaveOutputs.SCL_states`` stays in
    physical public layout; rate fields are RAW. ``fail_time`` is the scalar time of
    the first segment failure (``inf`` if the solve never bailed); the caller masks
    measurements at ``t > fail_time`` out of the loss. Those rows carry a finite ``y0``
    placeholder here (the loss-facing solve is sanitized), not a real prediction.

    ``n_linspace`` counts the evenly-spaced points ``build_union_time_grid``
    spliced into ``t_eval``; it only sizes the solver's per-segment output window.
    ``None`` (a bare
    measurement grid) leaves the solver to bound it from the grid length.
    """
    # Bounded physical-state solve (manual jumps at events) — well-conditioned
    # gradient. Replaces the pseudobatch single-solve whose unbounded accumulator
    # corrupted the adjoint (see specs/pseudo_diagnosis.md).
    from .physical_solve import solve_physical_states

    n_meas_arr = jnp.asarray(n_measured, dtype=jnp.int32)
    t_eval = clamp_padded_time_rows(t_eval[None, :], n_meas_arr[None])[0]
    states, fail_time = solve_physical_states(
        wrapper,
        t_eval=t_eval,
        n_measured=n_meas_arr,
        RAW_y0=RAW_y0,
        max_steps=max_steps,
        rtol=float(rtol),
        atol=float(atol),
        jump_ts=jump_ts,
        n_linspace=n_linspace,
        return_fail_time=True,
    )
    return jax.vmap(wrapper.physical_save_outputs)(t_eval, states), fail_time


def simulate_measurement_states(
    wrapper: HybridOdeWrapper,
    process_data: PerProcessTrainingData,
    *,
    max_steps: int = 100_000,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> jax.Array:
    """RAW physical state trajectories at the active measurement timestamps.

    Convenience entry point for forward/export/plotting and tests: it points the
    wrapper at ``process_data``'s per-process controls and integrates from
    ``process_data.y0_measured`` with the discrete-jump callbacks solve
    (:func:`physical_solve.solve_physical_states`). Returns the RAW physical state
    ``[modeled_RMCs | V | modeled cumulative flows]`` at each padded
    measurement time. (Replaces the pre-callbacks single-solve of the same name;
    no ``jump_ts`` argument — the callbacks solve lands segment ends on the events.)

    For a stateful model the returned width is the full integrated ``n_state``
    (physical columns followed by the trailing latent block); slice physical
    columns by index rather than assuming the physical-only width.
    """
    from .physical_solve import solve_physical_states

    ts = process_data.active_t_measured
    if ts.size == 0:
        raise ValueError("process has no active measurement timestamps")
    process_data.controls.validate_support(float(ts[0]), float(ts[-1]))
    # The wrapper's baked flow matrices belong to its template process, so all
    # four process-specific matrices are substituted alongside controls.
    rhs_ode = replace_rhs_ode_process_matrices(
        wrapper.rhs_ode,
        process_data.Cin_controlled_Inflows,
        process_data.Cin_modeled_Inflows,
        process_data.retention_controlled_Outflows,
        process_data.retention_modeled_Outflows,
    )
    sample_wrapper = eqx.tree_at(
        lambda w: (w.controls, w.rhs_ode),
        wrapper,
        (process_data.controls, rhs_ode),
    )
    return solve_physical_states(
        sample_wrapper,
        t_eval=ts,
        n_measured=process_data.n_measured,
        RAW_y0=process_data.y0_measured,
        max_steps=max_steps,
        rtol=rtol,
        atol=atol,
        # Bare measurement grid — no linspace spliced in — so the output window needs
        # only the per-gap measurement count. Saying so beats letting the solver bound
        # it from the grid length, and per-segment save cost scales with the window.
        n_linspace=0,
    )


class _BatchIndexedControls(eqx.Module):
    """Lightweight per-sample controls adapter for batched loss evaluation."""

    batch_controls: BatchControls
    row_idx: jax.Array

    def eval_controlled_Inflows_cumulative(self, t_arr, states) -> jax.Array:
        return self.batch_controls.eval_controlled_Inflows_cumulative(
            self.row_idx, t_arr, states
        )

    def eval_controlled_Inflows_rates(self, t_arr, states) -> jax.Array:
        return self.batch_controls.eval_controlled_Inflows_rates(
            self.row_idx, t_arr, states
        )

    def eval_controlled_Outflows_cumulative(self, t_arr, states) -> jax.Array:
        return self.batch_controls.eval_controlled_Outflows_cumulative(
            self.row_idx, t_arr, states
        )

    def eval_controlled_Outflows_rates(self, t_arr, states) -> jax.Array:
        return self.batch_controls.eval_controlled_Outflows_rates(
            self.row_idx, t_arr, states
        )

    def eval_controlled_PVs(self, t_arr, states) -> jax.Array:
        return self.batch_controls.eval_controlled_PVs(self.row_idx, t_arr, states)

    @property
    def min_V(self) -> jax.Array:
        return self.batch_controls.min_V[self.row_idx]

    @property
    def sample_event_times(self) -> jax.Array:
        return self.batch_controls.sample_event_times[self.row_idx]

    @property
    def sample_event_volumes(self) -> jax.Array:
        return self.batch_controls.sample_event_volumes[self.row_idx]

    @property
    def sample_event_mask(self) -> jax.Array:
        return self.batch_controls.sample_event_mask[self.row_idx]

    @property
    def bolus_event_times(self) -> jax.Array:
        return self.batch_controls.bolus_event_times[self.row_idx]

    @property
    def bolus_event_volumes(self) -> jax.Array:
        return self.batch_controls.bolus_event_volumes[self.row_idx]

    @property
    def bolus_event_Cin(self) -> jax.Array:
        return self.batch_controls.bolus_event_Cin[self.row_idx]

    @property
    def bolus_event_mask(self) -> jax.Array:
        return self.batch_controls.bolus_event_mask[self.row_idx]

    @property
    def max_event_gap_fraction(self) -> float:
        return self.batch_controls.max_event_gap_fraction

    @property
    def max_measurements_per_event_gap(self) -> int:
        return self.batch_controls.max_measurements_per_event_gap


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
    is their mean.

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
        RAW_rates = save_outputs.RAW_modeled_ReactionOde_rates
        RAW_Inflow_rates = save_outputs.RAW_modeled_Inflows_rates
        RAW_Outflow_rates = save_outputs.RAW_modeled_Outflows_rates
        RAW_V = save_outputs.RAW_V
        return {
            "SCL_states": SCL_states,
            "RAW_states": module.unscale_state(SCL_states),
            "SCL_modeled_ReactionOde_rates": module.scale_modeled_ReactionOde_rates(
                RAW_rates
            ),
            "RAW_modeled_ReactionOde_rates": RAW_rates,
            "SCL_modeled_Inflows_rates": module.scale_modeled_Inflows_rates(
                RAW_Inflow_rates
            ),
            "RAW_modeled_Inflows_rates": RAW_Inflow_rates,
            "SCL_modeled_Outflows_rates": module.scale_modeled_Outflows_rates(
                RAW_Outflow_rates
            ),
            "RAW_modeled_Outflows_rates": RAW_Outflow_rates,
            "SCL_V": module.scale_modeled_V(RAW_V),
            "RAW_V": RAW_V,
            "RAW_V_unclamped": save_outputs.RAW_V_export,
            "auxiliary": save_outputs.auxiliary or {},
        }

    if dense_grid_n is None and prediction_grid_n is None:
        # Measurement-grid-only path (unchanged behavior).
        save_outputs, fail_time = _solve_measurement_save_outputs_on_grid(
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
        prediction_valid = None
    else:
        # Dense/prediction path: solve ONCE on the union of the measurement grid,
        # the loss module's dense grid, and the forward prediction grid, then
        # index-split. Loss reads the dense block; forward reads the prediction
        # block. The union grid only decides where the solve is SAVED, not where it is
        # split — segments come from bolus/sample events alone — so a finer grid leaves
        # the loss value, ``fail_time`` and the post-failure mask alone.
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
        save_outputs, fail_time = _solve_measurement_save_outputs_on_grid(
            wrapper,
            t_eval=t_eval,
            n_measured=t_eval.shape[0],
            RAW_y0=RAW_y0_measured,
            max_steps=max_solver_steps,
            rtol=solver_rtol,
            atol=solver_atol,
            jump_ts=jump_ts,
            # Exactly the two linspace blocks spliced in just above, so the solver can
            # size its per-segment output window tightly instead of bounding the whole
            # grid. Both are static Python ints here.
            n_linspace=(dense_grid_n or 0) + (prediction_grid_n or 0),
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
            prediction_valid = None
        else:
            # Splice the exact measurement grid into the forward export grid so
            # predictions.csv carries a node at every measurement time (padded
            # repeats + shared t0/t1 endpoints are collapsed by
            # dense_exports_from_save_outputs). Scoring then reads exact values at
            # the measurements instead of interpolating a straight ramp across
            # bolus/feed discontinuities.
            export_idx = jnp.concatenate([sample_idx, prediction_idx])
            prediction_t = t_eval[export_idx]
            prediction_save_outputs = jtu.tree_map(
                lambda leaf: leaf[export_idx], save_outputs
            )
            prediction_valid = within_fail_time(prediction_t, fail_time)

    SCL_states = meas_views["SCL_states"]
    # target_state_indices selects measured states and modeled cumulative flows.
    SCL_target_pred = SCL_states[:, wrapper.target_state_indices]

    # Drop measurement points past a segment failure: the loss-facing solve replaced
    # those rows with a finite ``y0`` placeholder (not a real prediction), so without
    # this mask they would enter the loss as bogus targets. Earlier good points are kept
    # (their signal is valuable, esp. early in training). No-op on a healthy solve
    # (``fail_time == inf``). ``within_fail_time`` owns the boundary tolerance.
    valid_time = within_fail_time(t_measured, fail_time)
    mask_measured = mask_measured & valid_time[:, None]

    # Same failure cutoff for the dense grid, which has no other failure mask. ``None``
    # when the dense grid is disabled; all-True on a healthy solve (fail_time == inf).
    dense_valid_time = (
        None
        if dense_t is None or dense_views["SCL_states"] is None
        else within_fail_time(dense_t, fail_time)
    )

    mask_any = jnp.any(mask_measured, axis=1).astype(SCL_states.dtype)
    step_arr = jnp.asarray(step if step is not None else -1, dtype=jnp.int32)

    inputs = LossInputs(
        SCL_states=SCL_states,
        RAW_states=meas_views["RAW_states"],
        SCL_modeled_ReactionOde_rates=meas_views["SCL_modeled_ReactionOde_rates"],
        RAW_modeled_ReactionOde_rates=meas_views["RAW_modeled_ReactionOde_rates"],
        SCL_modeled_Inflows_rates=meas_views["SCL_modeled_Inflows_rates"],
        RAW_modeled_Inflows_rates=meas_views["RAW_modeled_Inflows_rates"],
        SCL_modeled_Outflows_rates=meas_views["SCL_modeled_Outflows_rates"],
        RAW_modeled_Outflows_rates=meas_views["RAW_modeled_Outflows_rates"],
        SCL_V=meas_views["SCL_V"],
        RAW_V=meas_views["RAW_V"],
        RAW_V_unclamped=meas_views["RAW_V_unclamped"],
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
        dense_SCL_modeled_ReactionOde_rates=dense_views[
            "SCL_modeled_ReactionOde_rates"
        ],
        dense_RAW_modeled_ReactionOde_rates=dense_views[
            "RAW_modeled_ReactionOde_rates"
        ],
        dense_SCL_modeled_Inflows_rates=dense_views["SCL_modeled_Inflows_rates"],
        dense_RAW_modeled_Inflows_rates=dense_views["RAW_modeled_Inflows_rates"],
        dense_SCL_modeled_Outflows_rates=dense_views["SCL_modeled_Outflows_rates"],
        dense_RAW_modeled_Outflows_rates=dense_views["RAW_modeled_Outflows_rates"],
        dense_SCL_V=dense_views["SCL_V"],
        dense_RAW_V=dense_views["RAW_V"],
        dense_RAW_V_unclamped=dense_views["RAW_V_unclamped"],
        dense_auxiliary=dense_views["auxiliary"],
        dense_valid_time=dense_valid_time,
    )

    outputs = loss_module(inputs)
    named = outputs.named_losses
    per_target_loss = jnp.stack([named[name] for name in loss_module.loss_names])
    total_loss = jnp.mean(per_target_loss)
    measurement_save_outputs = (
        save_outputs
        if dense_grid_n is None and prediction_grid_n is None
        else sample_save_outputs
    )
    return SingleSampleResult(
        total_loss=total_loss,
        per_target_loss=per_target_loss,
        states=SCL_states,
        save_outputs=save_outputs,
        measurement_save_outputs=measurement_save_outputs,
        measurement_prediction_valid=valid_time,
        step=step_arr,
        prediction_t=prediction_t,
        prediction_save_outputs=prediction_save_outputs,
        prediction_valid=prediction_valid,
        fail_time=fail_time,
    )


def evaluate_one_sample_loss(
    wrapper: HybridOdeWrapper,
    batch_controls: BatchControls,
    row_idx: jax.Array,
    t_measured: jax.Array,
    SCL_target_measured: jax.Array,
    mask_measured: jax.Array,
    n_measured: jax.Array,
    RAW_y0_measured: jax.Array,
    cin: jax.Array,
    cin_modeled: jax.Array,
    retention: jax.Array,
    retention_modeled: jax.Array,
    jump_ts: jax.Array | None,
    *,
    max_solver_steps: int,
    solver_rtol: float,
    solver_atol: float,
    step: int | jax.Array | None = None,
):
    """One process -> ``(total_loss, per_target_loss, fail_time)``. Shared by the
    vmap batched loss and the device-sharded (pmap) step. ``fail_time`` is ``+inf``
    for a clean solve, finite when a segment bailed. ``y0`` stays RAW physical;
    the callbacks solve (``physical_solve.solve_physical_states``) integrates it."""
    controls = _BatchIndexedControls(
        batch_controls=batch_controls,
        row_idx=row_idx,
    )
    rhs_ode = replace_rhs_ode_process_matrices(
        wrapper.rhs_ode,
        cin,
        cin_modeled,
        retention,
        retention_modeled,
    )
    sample_wrapper = eqx.tree_at(
        lambda w: (w.controls, w.rhs_ode),
        wrapper,
        (controls, rhs_ode),
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
    return result.total_loss, result.per_target_loss, result.fail_time


def build_batched_loss_fn() -> Callable[..., tuple]:
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
        # This MUST be the same target scaler the solver applies to predictions:
        # only then does an affine offset cancel in the residual. Diverging
        # scalers silently shift every residual by b/s.
        SCALE_state = wrapper.reaction_module.SCALE_state
        SCALE_state_for_targets = SCALE_state[wrapper.target_state_indices]
        SCL_y_measured = batch.y_measured / SCALE_state_for_targets[None, None, :]

        def _sample_loss(
            row_idx: jax.Array,
            t_measured: jax.Array,
            SCL_target_measured: jax.Array,
            mask_measured: jax.Array,
            n_measured: jax.Array,
            RAW_y0_measured: jax.Array,
            cin: jax.Array,
            cin_modeled: jax.Array,
            retention: jax.Array,
            retention_modeled: jax.Array,
            jump_ts: jax.Array | None,
        ) -> tuple:
            controls = _BatchIndexedControls(
                batch_controls=batch.controls,
                row_idx=row_idx,
            )
            rhs_ode = replace_rhs_ode_process_matrices(
                wrapper.rhs_ode,
                cin,
                cin_modeled,
                retention,
                retention_modeled,
            )
            sample_wrapper = eqx.tree_at(
                lambda w: (w.controls, w.rhs_ode),
                wrapper,
                (controls, rhs_ode),
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
                result.prediction_valid,
                result.measurement_save_outputs,
                result.measurement_prediction_valid,
                result.fail_time,
            )

        (
            per_sample_total,
            per_sample_per_target,
            prediction_t,
            prediction_save_outputs,
            prediction_valid,
            measurement_save_outputs,
            measurement_prediction_valid,
            per_sample_fail_time,
        ) = jax.vmap(
            _sample_loss,
            # when `jump_ts_rows` is `None` we also have to pass `None` to vmap so that
            # it's not iterated over
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
                None if jump_ts_rows is None else 0,
            ),
        )(
            jnp.arange(batch.process_indices.shape[0], dtype=jnp.int32),
            batch_t_meas,
            SCL_y_measured,
            batch.mask_measured,
            batch.n_measured,
            batch.y0_measured,
            batched_Cin[batch.process_indices],
            batched_Cin_modeled[batch.process_indices],
            batch.retention_controlled_Outflows,
            batch.retention_modeled_Outflows,
            jump_ts_rows,
        )
        mean_total = jnp.mean(per_sample_total)
        # 2nd element is PER-SAMPLE per-target (one row per process), matching the
        # forward harvest path (compute_dense_exports); the vmap training step
        # means it. `prediction_*` are non-None only when forward requested
        # `prediction_grid_n`; training ignores them. The measurement grid and its
        # outputs follow for checkpoint exports. `per_sample_fail_time` remains LAST
        # so positional callers can continue to collect it with `*_`.
        return (
            mean_total,
            per_sample_per_target,
            per_sample_total,
            prediction_t,
            prediction_save_outputs,
            prediction_valid,
            batch_t_meas,
            measurement_save_outputs,
            measurement_prediction_valid,
            per_sample_fail_time,
        )

    return _batched_loss_fn
