"""
Core solver: diffeqsolve_with_callbacks.

Features:
  - ContinuousCallbacks: zero-crossing detection with root-finding + repeat_nudge
  - DiscreteCallbacks: evaluated at every segment boundary
  - PresetTimeCallbacks: events at known times
  - PeriodicCallbacks: events every Δt
  - ManifoldProjection: constraint enforcement
  - evaluate_trajectory: reconstruct full trajectory at arbitrary time points
  - Full differentiability through all event types
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import diffrax

from ._callbacks import (
    ContinuousCallback,
    DiscreteCallback,
    PresetTimeCallback,
    PeriodicCallback,
    ManifoldProjection,
    CallbackSet,
)
from ._solution import CallbackSolution


def _wrap_single_callback(callback):
    if isinstance(callback, CallbackSet):
        return callback
    return CallbackSet(callback)


def _build_diffrax_event(callback_set: CallbackSet):
    """Build a diffrax.Event from all ContinuousCallbacks."""
    if callback_set.n_continuous == 0:
        return None

    ccs = callback_set.continuous_callbacks

    if len(ccs) == 1:
        cc = ccs[0]

        def cond_fn(t, y, args, **kwargs):
            return cc.condition_fn(y, t, args)

        return diffrax.Event(
            cond_fn=cond_fn,
            root_finder=cc.root_finder,
            direction=cc._diffrax_direction,
        )
    else:
        cond_fns = []
        directions = []
        root_finder = ccs[0].root_finder
        for cc in ccs:

            def _make_cond(cc_captured):
                def cond_fn(t, y, args, **kwargs):
                    return cc_captured.condition_fn(y, t, args)

                return cond_fn

            cond_fns.append(_make_cond(cc))
            directions.append(cc._diffrax_direction)
        return diffrax.Event(
            cond_fn=cond_fns,
            root_finder=root_finder,
            direction=directions,
        )


# Event/segment time comparisons are magnitude-relative *and* dtype-aware: the tolerance
# is ``factor * eps(dtype) * (1 + |t|)`` rather than a fixed constant. A fixed constant
# (e.g. 1e-10) sits below float32 ULP at t~15 (~1e-6), so in float32
# events silently fail to fire and the solve freezes; in float64 it would be
# needlessly loose. Deriving from
# ``jnp.finfo(dtype).eps`` makes both precisions correct automatically:
#   float32 eps ~1.2e-7 -> EVENT ~1.9e-4, STEP ~3.8e-6  at t~15
#   float64 eps ~2.2e-16 -> EVENT ~3.5e-13, STEP ~7e-15  at t~15
# "did the solver land on the preset?" — loose (solver stop error)
_EVENT_TOL_FACTOR = 1e2
# "strictly future / positive-length segment" — tight (above ULP)
_STEP_TOL_FACTOR = 2.0


def _dtype_tol(t, factor):
    eps = jnp.finfo(jnp.asarray(t).dtype).eps
    return factor * eps * (1.0 + jnp.abs(t))


def _find_next_preset_time(preset_times, t_current, t_end):
    future_mask = preset_times > t_current + _dtype_tol(t_current, _STEP_TOL_FACTOR)
    masked_times = jnp.where(future_mask, preset_times, t_end + 1.0)
    idx = jnp.argmin(masked_times)
    next_time = masked_times[idx]
    is_valid = next_time <= t_end
    return jnp.where(is_valid, next_time, t_end), jnp.where(is_valid, idx, -1)


# ================================================================
# Main solver
# ================================================================


def diffeqsolve_with_callbacks(
    terms: diffrax.AbstractTerm,
    solver: diffrax.AbstractSolver,
    t0: float,
    t1: float,
    dt0: float,
    y0: jnp.ndarray,
    args=None,
    *,
    callbacks: ContinuousCallback
    | DiscreteCallback
    | PresetTimeCallback
    | PeriodicCallback
    | ManifoldProjection
    | CallbackSet,
    max_events: int = 20,
    stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
        rtol=1e-6, atol=1e-8
    ),
    max_steps_per_segment: int = 4096,
    adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint(),
) -> CallbackSolution:
    """Solve an ODE with Julia-style callbacks.

    Wraps diffrax.diffeqsolve in a scan-based event loop:
      1. Solves until the next event (continuous, preset, or t1)
      2. Applies the corresponding affect function
      3. Runs DiscreteCallbacks / ManifoldProjection at segment boundary
      4. Restarts from the modified state
      5. Repeats up to max_events times

    Args:
        terms: Diffrax term (e.g., ODETerm).
        solver: Diffrax solver (e.g., Tsit5, Kvaerno5).
        t0, t1: Time interval.
        dt0: Initial step size.
        y0: Initial state array.
        args: ODE arguments (passed to vector field and callbacks).
        callbacks: Single callback or CallbackSet.
        max_events: Maximum events (fixed for JIT). Unused slots are padded.
        stepsize_controller: Diffrax step size controller.
        max_steps_per_segment: Max solver steps between events.
        adjoint: Diffrax adjoint method for backpropagation.

    Returns:
        CallbackSolution with final state and event log.
        Use evaluate_trajectory() to reconstruct the full trajectory.
    """
    callback_set = _wrap_single_callback(callbacks)
    diffrax_event = _build_diffrax_event(callback_set)

    # Keep solver time in the state dtype. Diffrax stage buffers are allocated from
    # y0; letting float64 times promote the vector field output breaks float32 states.
    time_dtype = jnp.asarray(y0).dtype
    t0 = jnp.asarray(t0, dtype=time_dtype)
    t1 = jnp.asarray(t1, dtype=time_dtype)
    dt0 = jnp.asarray(dt0, dtype=time_dtype)

    n_continuous = callback_set.n_continuous
    n_preset = callback_set.n_preset
    n_discrete = callback_set.n_discrete
    has_presets = n_preset > 0
    has_continuous = n_continuous > 0
    has_discrete = n_discrete > 0
    repeat_nudge = callback_set.get_max_repeat_nudge()

    # Merge preset times
    if has_presets:
        all_preset_times = callback_set.get_all_preset_times().astype(time_dtype)
        preset_affect_indices = callback_set.get_preset_affect_indices()
    else:
        all_preset_times = jnp.asarray([t1 + jnp.asarray(1.0, dtype=time_dtype)])
        preset_affect_indices = jnp.array([0], dtype=jnp.int32)

    # ---- Affect dispatchers (built at trace time) ----

    def _dispatch_continuous_affect(y, t, args, event_mask):
        if n_continuous == 1:
            return callback_set.continuous_callbacks[0].affect_fn(y, t, args)
        else:
            mask_array = jnp.array(jax.tree.leaves(event_mask))
            idx = jnp.argmax(mask_array)
            branches = [cc.affect_fn for cc in callback_set.continuous_callbacks]
            return jax.lax.switch(idx, branches, y, t, args)

    def _dispatch_preset_affect(y, t, args, preset_cb_idx):
        if n_preset == 1:
            return callback_set.preset_callbacks[0].affect_fn(y, t, args)
        else:
            branches = [cb.affect_fn for cb in callback_set.preset_callbacks]
            return jax.lax.switch(preset_cb_idx, branches, y, t, args)

    def _apply_discrete_callbacks(y, t, args):
        """Apply all discrete callbacks in order."""
        for dc in callback_set.discrete_callbacks:
            cond = dc.condition_fn(y, t, args)
            y_new = dc.affect_fn(y, t, args)
            y = jnp.where(cond, y_new, y)
        return y

    # ---- Scan body ----

    def scan_fn(carry, _):
        y, t_current, done, terminated = carry

        # Determine segment end time
        if has_presets:
            next_preset_time, next_preset_idx = _find_next_preset_time(
                all_preset_times, t_current, t1
            )
            segment_t1 = jnp.minimum(next_preset_time, t1)
        else:
            segment_t1 = t1
            next_preset_time = t1 + jnp.asarray(1.0, dtype=time_dtype)
            next_preset_idx = jnp.int32(-1)

        # Ensure segment_t1 >= t_current (can be violated by repeat_nudge or when done)
        segment_t1 = jnp.maximum(
            segment_t1,
            t_current + _dtype_tol(t_current, _STEP_TOL_FACTOR),
        )

        # Solve the segment
        dt = jnp.minimum(dt0, segment_t1 - t_current)

        saveat = diffrax.SaveAt(t1=True)

        sol = diffrax.diffeqsolve(
            terms,
            solver,
            t0=t_current,
            t1=segment_t1,
            dt0=dt,
            y0=y,
            args=args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            event=diffrax_event,
            max_steps=max_steps_per_segment,
            throw=False,
            adjoint=adjoint,
        )

        y_at_stop = sol.ys[-1]
        t_at_stop = sol.ts[-1]

        # ---- Determine what happened ----
        if has_continuous:
            if n_continuous == 1:
                continuous_triggered = sol.event_mask & (~done) & (~terminated)
            else:
                continuous_triggered = (
                    jax.tree.reduce(lambda a, b: a | b, sol.event_mask)
                    & (~done)
                    & (~terminated)
                )
        else:
            continuous_triggered = jnp.bool_(False)

        if has_presets:
            preset_triggered = (
                (next_preset_idx >= 0)
                & (~continuous_triggered)
                & (~done)
                & (~terminated)
                & (
                    jnp.abs(t_at_stop - next_preset_time)
                    < _dtype_tol(next_preset_time, _EVENT_TOL_FACTOR)
                )
            )
        else:
            preset_triggered = jnp.bool_(False)

        any_event = continuous_triggered | preset_triggered
        y_before_event = y_at_stop

        # ---- Apply continuous or preset affect ----
        if has_continuous and has_presets:
            y_cont = _dispatch_continuous_affect(
                y_at_stop, t_at_stop, args, sol.event_mask
            )
            y_pres = _dispatch_preset_affect(
                y_at_stop,
                t_at_stop,
                args,
                preset_affect_indices[jnp.clip(next_preset_idx, 0)],
            )
            y_after = jnp.where(
                continuous_triggered,
                y_cont,
                jnp.where(preset_triggered, y_pres, y_at_stop),
            )
        elif has_continuous:
            y_cont = _dispatch_continuous_affect(
                y_at_stop, t_at_stop, args, sol.event_mask
            )
            y_after = jnp.where(continuous_triggered, y_cont, y_at_stop)
        elif has_presets:
            y_pres = _dispatch_preset_affect(
                y_at_stop,
                t_at_stop,
                args,
                preset_affect_indices[jnp.clip(next_preset_idx, 0)],
            )
            y_after = jnp.where(preset_triggered, y_pres, y_at_stop)
        else:
            y_after = y_at_stop

        # ---- Apply discrete callbacks at segment boundary ----
        if has_discrete:
            y_after = jnp.where(
                ~terminated,
                _apply_discrete_callbacks(y_after, t_at_stop, args),
                y_after,
            )

        # ---- Check for termination signal ----
        # Termination is signaled by the affect returning NaN in the
        # last state element and Inf in the second-to-last.
        # (This is a workaround since we can't return tuples from
        # lax.switch branches with different structures.)
        # For now, termination is handled via the `terminated` flag
        # in the carry, set when no event fires (simulation complete).

        # ---- Event type for logging ----
        if has_continuous and n_continuous == 1:
            continuous_idx = jnp.int32(0)
        elif has_continuous:
            mask_leaves = jnp.array(jax.tree.leaves(sol.event_mask))
            continuous_idx = jnp.argmax(mask_leaves).astype(jnp.int32)
        else:
            continuous_idx = jnp.int32(0)

        if has_presets:
            preset_type_idx = (
                n_continuous + preset_affect_indices[jnp.clip(next_preset_idx, 0)]
            )
        else:
            preset_type_idx = jnp.int32(-1)

        event_type = jnp.where(
            continuous_triggered,
            continuous_idx,
            jnp.where(preset_triggered, preset_type_idx, jnp.int32(-1)),
        )

        # ---- repeat_nudge: advance time slightly after continuous events ----
        # Clamp to t1 so we don't overshoot the end time
        t_next = jnp.where(
            continuous_triggered,
            jnp.minimum(t_at_stop + repeat_nudge, t1),
            t_at_stop,
        )

        # ---- Update done/terminated ----
        new_done = done | (~any_event)
        new_terminated = terminated

        event_time = jnp.where(any_event & (~done) & (~terminated), t_at_stop, t1)

        output = (
            event_time,
            jnp.where(done | terminated, jnp.int32(-1), event_type),
            y_before_event,
            y_after,
        )

        return (y_after, t_next, new_done, new_terminated), output

    # ---- Run the scan ----
    init_carry = (
        y0,
        t0,
        jnp.bool_(False),
        jnp.bool_(False),
    )
    final_carry, outputs = jax.lax.scan(scan_fn, init_carry, None, length=max_events)
    y_final, t_final, _, _ = final_carry
    event_times, event_types, states_before, states_after = outputs

    event_count = jnp.sum((event_types >= 0).astype(jnp.int32))

    return CallbackSolution(
        y_final=y_final,
        t_final=t_final,
        event_times=event_times,
        event_types=event_types,
        event_states_before=states_before,
        event_states_after=states_after,
        event_count=event_count,
    )


def evaluate_trajectory(
    sol: CallbackSolution,
    terms: diffrax.AbstractTerm,
    solver: diffrax.AbstractSolver,
    t0: float,
    t1: float,
    dt0: float,
    y0: jnp.ndarray,
    args=None,
    *,
    ts: jnp.ndarray,
    stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
        rtol=1e-6, atol=1e-8
    ),
    max_steps_per_segment: int = 4096,
):
    """Reconstruct the full trajectory at arbitrary time points.

    Re-solves the ODE segment-by-segment using the event log from a
    CallbackSolution, saving at the requested time points. This correctly
    handles state jumps at event boundaries.

    Not JIT-compatible (Python loop over segments), but fast since each
    segment is short. Intended for plotting and analysis.

    Args:
        sol: Solution from diffeqsolve_with_callbacks.
        terms: Same terms used in the original solve.
        solver: Same solver.
        t0, t1: Same time interval.
        dt0: Initial step size.
        y0: Initial state.
        args: Same args.
        ts: Time points at which to evaluate the trajectory.
        stepsize_controller: Step size controller.
        max_steps_per_segment: Max steps per segment.

    Returns:
        (ts_out, ys_out): Arrays of times and states at the requested points.
    """
    n_events = int(sol.event_count)

    # Build segment boundaries: [t0, event_1, event_2, ..., t1]
    boundaries = [float(t0)]
    for i in range(n_events):
        boundaries.append(float(sol.event_times[i]))
    boundaries.append(float(t1))

    # Initial states for each segment: y0, then post-event states
    segment_y0s = [y0]
    for i in range(n_events):
        segment_y0s.append(sol.event_states_after[i])

    all_ts = []
    all_ys = []

    for seg_idx in range(len(boundaries) - 1):
        seg_t0 = boundaries[seg_idx]
        seg_t1 = boundaries[seg_idx + 1]
        seg_y0 = segment_y0s[seg_idx]

        if seg_t1 <= seg_t0 + 1e-12:
            continue

        # Find requested time points within this segment
        mask = (ts >= seg_t0 - 1e-10) & (ts <= seg_t1 + 1e-10)
        # Exclude points already saved in previous segment (avoid duplicates)
        if seg_idx > 0:
            mask = mask & (ts > seg_t0 + 1e-10)
        seg_ts = ts[mask]

        if len(seg_ts) == 0:
            continue

        seg_sol = diffrax.diffeqsolve(
            terms,
            solver,
            seg_t0,
            seg_t1,
            dt0=min(dt0, seg_t1 - seg_t0),
            y0=seg_y0,
            args=args,
            saveat=diffrax.SaveAt(ts=seg_ts),
            stepsize_controller=stepsize_controller,
            max_steps=max_steps_per_segment,
        )
        all_ts.append(seg_sol.ts)
        all_ys.append(seg_sol.ys)

    if all_ts:
        return jnp.concatenate(all_ts), jnp.concatenate(all_ys)
    else:
        return jnp.array([]), jnp.zeros((0, y0.shape[0]))
