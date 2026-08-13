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


def _cast_like(value, like):
    return jax.tree.map(
        lambda x, ref: jnp.asarray(x, dtype=jnp.asarray(ref).dtype), value, like
    )


def _wrap_ode_term_dtype(terms):
    if not isinstance(terms, diffrax.ODETerm):
        return terms

    def vector_field(t, y, args):
        return _cast_like(terms.vector_field(t, y, args), y)

    return diffrax.ODETerm(vector_field)


def _wrap_affect_dtype(affect_fn):
    def wrapped(y, t, args):
        return _cast_like(affect_fn(y, t, args), y)

    return wrapped


def _wrap_preset_affect_dtype(affect_fn):
    """Same as ``_wrap_affect_dtype`` for the 4-arg preset contract.

    Preset affects additionally receive ``preset_index`` (the slot within their own
    ``times``) so they can identify the firing preset exactly instead of comparing
    floats against ``t``. Speculative batched evaluation passes ``-1``; see
    ``PresetTimeCallback``.
    """

    def wrapped(y, t, args, preset_index):
        return _cast_like(affect_fn(y, t, args, preset_index), y)

    return wrapped


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
    output_times: jnp.ndarray | None = None,
    output_window: int | None = None,
    stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
        rtol=1e-6, atol=1e-8
    ),
    max_steps: int = 4096,
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
        output_times: Optional (n_output,) ASCENDING times inside ``[t0, t1]`` at which
            to record the trajectory, returned as ``CallbackSolution.output_states``.
            Each segment saves its own share with ``SaveAt(ts=...)``, which is pure
            interpolation and costs no extra solver steps -- so an output time does NOT
            have to be a preset, and asking for a finer trajectory no longer subdivides
            the integration. ``None`` keeps the segment solves on ``SaveAt(t1=True)``
            and leaves ``output_states`` as ``None``.
        output_window: Optional static bound on how many ``output_times`` entries a
            single segment may own. Each segment then only sees a ``dynamic_slice``
            window of that size instead of the whole grid. Without it the per-segment
            save work is ``O(max_events * n_output)``, because diffrax writes every
            slot in every segment (out-of-range times pin onto the segment endpoints)
            -- which makes an event-heavy process SLOWER than saving at boundaries.
            ``None`` uses the whole grid. If a segment ever owns more than
            ``output_window`` points the
            excess would be dropped, so ``output_overflow`` is raised instead.
        stepsize_controller: Diffrax step size controller.
        max_steps: Step budget for the WHOLE trajectory. Must be a static Python int --
            diffrax sizes its step loop from it. It bounds each segment's inner solve
            AND the running sum across segments, so once the total is exceeded the lane
            terminates exactly as if a segment had failed (state poisoned to inf,
            ``fail_time`` recorded).

            This replaces the earlier pair of a per-segment cap plus an optional total.
            The per-segment cap was a *latency* bound (stopping one stiff lane spinning
            while other devices block on a collective) and predates the total; once a
            trajectory budget exists it is redundant, because the total bounds a lane
            more tightly than ``cap * n_segments`` ever did. Keeping both was actively
            harmful: with few, long segments the smaller per-segment cap silently became
            the binding constraint, so a process needing more steps than the cap bailed
            even though its trajectory budget was ample.
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

    solve_terms = _wrap_ode_term_dtype(terms)

    # ---- Tier-B output grid (see ``output_times`` in the docstring) ----
    has_output = output_times is not None
    if has_output:
        output_times = jnp.asarray(output_times, dtype=time_dtype)
        if output_times.ndim != 1:
            raise ValueError(f"output_times must be 1-D, got {output_times.ndim}-D")
        if output_times.shape[0] == 0:
            # diffrax maps an empty ``ts`` to ``None`` and then reports the unrelated
            # "Empty saveat -- nothing will be saved"; fail here with the real reason.
            raise ValueError("output_times must be non-empty; pass None to disable")
        n_output = output_times.shape[0]
        window = (
            n_output if output_window is None else min(int(output_window), n_output)
        )
    else:
        n_output = 0
        window = 0

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
        preset_local_indices = callback_set.get_preset_local_indices()
    else:
        all_preset_times = jnp.zeros((0,), dtype=time_dtype)

    # A PresetTimeCallback may carry an EMPTY ``times`` array: a process with zero
    # bolus AND zero sample events is legitimate data, not user error (the padded
    # widths are collection-wide maxima and are legitimately 0). Such a callback makes
    # ``n_preset > 0`` but leaves nothing that can ever fire, and
    # ``_find_next_preset_time`` would ``argmin`` an empty array -- a hard trace-time
    # error. Decide ``has_presets`` from the MERGED array, not the callback count, and
    # fall through to the same parked entry the no-callback case uses. Static: array
    # shapes are compile-time constants, so every ``if has_presets`` branch below is
    # unaffected.
    has_presets = all_preset_times.shape[0] > 0
    if not has_presets:
        all_preset_times = jnp.asarray([t1 + jnp.asarray(1.0, dtype=time_dtype)])
        preset_affect_indices = jnp.array([0], dtype=jnp.int32)
        preset_local_indices = jnp.array([0], dtype=jnp.int32)

    # ---- Affect dispatchers (built at trace time) ----

    def _dispatch_continuous_affect(y, t, args, event_mask):
        if n_continuous == 1:
            return _wrap_affect_dtype(callback_set.continuous_callbacks[0].affect_fn)(
                y, t, args
            )
        else:
            mask_array = jnp.array(jax.tree.leaves(event_mask))
            idx = jnp.argmax(mask_array)
            branches = [
                _wrap_affect_dtype(cc.affect_fn)
                for cc in callback_set.continuous_callbacks
            ]
            return jax.lax.switch(idx, branches, y, t, args)

    def _dispatch_preset_affect(y, t, args, preset_cb_idx, preset_index):
        if n_preset == 1:
            return _wrap_preset_affect_dtype(
                callback_set.preset_callbacks[0].affect_fn
            )(y, t, args, preset_index)
        else:
            branches = [
                _wrap_preset_affect_dtype(cb.affect_fn)
                for cb in callback_set.preset_callbacks
            ]
            return jax.lax.switch(preset_cb_idx, branches, y, t, args, preset_index)

    def _apply_discrete_callbacks(y, t, args):
        """Apply all discrete callbacks in order."""
        for dc in callback_set.discrete_callbacks:
            cond = dc.condition_fn(y, t, args)
            y_new = _cast_like(dc.affect_fn(y, t, args), y)
            y = jnp.where(cond, y_new, y)
        return y

    # ---- Scan body ----

    def scan_fn(carry, _):
        (
            y,
            t_current,
            done,
            terminated,
            fail_time,
            steps_used,
            dt_prev,
            output_buffer,
            output_overflow,
        ) = carry

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

        # Once a lane has ``terminated`` (a prior segment bailed — e.g. hit
        # ``max_steps_per_segment`` on a stiff blow-up) OR is ``done`` (a healthy
        # lane that consumed every preset and reached ``t1``), collapse every later
        # segment to
        # zero length so the batched solve does ~0 steps for it. This is the vmap-safe
        # analogue of "skip the rest of the trajectory": under ``vmap`` the per-segment
        # ``diffeqsolve`` while-loop is paced by the slowest *live* lane, so a
        # zero-length lane satisfies ``t >= t1`` at entry, takes no steps and stops
        # driving the loop — whereas a ``lax.cond`` skip would vectorise to running both
        # branches.
        #
        # ``done`` must be in here, not just ``terminated``. The scan always runs
        # ``max_events`` iterations; without the collapse a finished lane still gets
        # ``segment_t1 = max(t1, t_current + tol)`` from the clamp just above — a
        # tolerance-LENGTH (not zero-length) segment, which diffrax runs at >= 1
        # accepted
        # step. Those steps are real: they cost a full 6-stage Tsit5 evaluation each AND
        # they accumulate into ``steps_used`` below, so they silently eat the
        # ``max_steps_total`` budget. Measured on a 2023_bayer-shaped process: 38 ODE
        # steps at ``max_events=25`` vs 27 at ``max_events=14``, for the same 10 real
        # segments.
        segment_t1 = jnp.where(terminated | done, t_current, segment_t1)

        # Solve the segment (dt == 0 for a collapsed lane; diffrax handles t0 == t1).
        #
        # ``dt_prev`` is the previous segment's average ACCEPTED step, floored at
        # ``dt0``
        # (see the carry update below). Every segment is a fresh ``diffeqsolve`` with no
        # controller history, so starting each one from the fixed ``dt0`` makes the
        # controller re-ramp from scratch; when the natural step is much larger than
        # ``dt0`` that ramp is most of the segment's cost. Seeding from what the last
        # segment actually sustained removes it (measured 10-40% fewer steps on bp-bench
        # training grids). Still clamped to the segment length so the first step cannot
        # overshoot the event.
        dt = jnp.minimum(dt_prev, segment_t1 - t_current)

        if has_output:
            live = (~done) & (~terminated)
            # Window bounds are DERIVED from the segment endpoints, never carried as a
            # running counter. A counter drifts as soon as the grid holds duplicate
            # times -- and it does: a union grid splices a linspace into the measurement
            # times, so t0/t1 appear twice and measurement times coincide with events.
            # Slots no segment owns are then never counted, the pointer falls behind and
            # the window slides off the owned range, silently leaving ``inf`` rows.
            # ``side="right"`` counts entries <= t, so ``hi - lo`` is exactly the
            # count in
            # the half-open ``(t_current, segment_t1]`` that ``owns`` selects below.
            lo = jnp.searchsorted(output_times, t_current, side="right")
            hi = jnp.searchsorted(output_times, segment_t1, side="right")
            start = jnp.clip(lo, 0, max(n_output - window, 0))
            ts_window = jax.lax.dynamic_slice(output_times, (start,), (window,))
            # Clipping is required (diffrax rejects ts outside [t0, t1]) and exact at
            # both ends; it is monotone, so an ascending grid stays ascending.
            saveat = diffrax.SaveAt(
                t1=True, ts=jnp.clip(ts_window, t_current, segment_t1)
            )
        else:
            saveat = diffrax.SaveAt(t1=True)

        sol = diffrax.diffeqsolve(
            solve_terms,
            solver,
            t0=t_current,
            t1=segment_t1,
            dt0=dt,
            y0=y,
            args=args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            event=diffrax_event,
            max_steps=max_steps,
            throw=False,
            adjoint=adjoint,
        )

        y_at_stop = sol.ys[-1]
        t_at_stop = sol.ts[-1]

        # A segment "fails" when diffrax bails before reaching segment_t1. Treat every
        # result EXCEPT the two legitimate stops as failure: ``successful`` (reached the
        # segment end) and ``event_occurred`` (a continuous-callback zero-crossing — the
        # inner solve's ``event`` is built only from continuous callbacks, so that is
        # the sole valid non-``successful`` code). This covers ``max_steps_reached``,
        # ``dt_min_reached``, nonlinear-solve failures, etc.; a naive ``!= successful``
        # would instead wrongly poison legitimate event stops.
        #
        # diffrax returns a *finite* last-reached state on such a bail (not inf), so we
        # force the lane's state to inf ourselves: a detectable non-finite SENTINEL on
        # the DIAGNOSTIC ``CallbackSolution`` outputs (``y_final``/``event_states_*``).
        # ``terminated`` (below) then collapses the lane's later segments, and
        # ``fail_time`` (below) records WHEN the lane first failed.
        #
        # NB: this inf is a forward-side diagnostic marker only, and makes NO promise
        # about the reverse-mode gradient of a failed lane (that is slot/loss-dependent
        # and unstable). The EXPLICIT, stable failure signal for callers is
        # ``fail_time``: ``bp_train`` uses it to drop post-failure measurement/dense
        # points from the loss AND to replace their predicted states with a finite
        # fallback, so no inf reaches a loss (see
        # ``physical_solve.solve_physical_states`` and
        # ``trainer.evaluate_sample_with_loss_module``). Callers must not rely on the
        # raw inf sentinel surviving into loss-facing state — only on ``fail_time``.
        R = diffrax.RESULTS
        seg_failed = (sol.result != R.successful) & (sol.result != R.event_occurred)

        # Trajectory-level budget. A per-segment bound alone would be grid-dependent:
        # subdividing the same horizon into more segments hands each one a fresh
        # allowance, so the effective ceiling would be ``max_steps * n_segments`` and a
        # caller asking for N steps could silently get orders of magnitude more.
        # Accumulating the (already computed) per-segment counts bounds the whole solve,
        # so ``fail_time`` does not depend on how the horizon was chopped. Treated as
        # exactly the same kind of failure as a segment bail, so all the existing
        # poisoning / collapse / fail_time machinery applies unchanged.
        new_steps_used = steps_used + jnp.asarray(sol.stats["num_steps"], jnp.int32)
        seg_failed = seg_failed | (
            new_steps_used > jnp.asarray(int(max_steps), jnp.int32)
        )

        if has_output:
            # With ``SaveAt(ts=...)`` diffrax fills every UNREACHED save slot -- the
            # ``t1`` one included -- with ``inf`` on a bail, where ``SaveAt(t1=True)``
            # alone hands back the last-reached (finite) state. An ``inf`` ``t_at_stop``
            # would NaN ``t_next``, ``dt`` and the collapse arithmetic on every later
            # iteration, so freeze time at the segment start. Nothing is lost:
            # ``fail_time`` is derived separately just below.
            t_at_stop = jnp.where(seg_failed, t_current, t_at_stop)

        y_at_stop = jnp.where(
            seg_failed, jnp.asarray(jnp.inf, y_at_stop.dtype), y_at_stop
        )

        if has_output:
            # OWNERSHIP: half-open ``(t_current, segment_t1]``. Exactly one live segment
            # owns each output time, and a time landing ON an event is owned by the
            # segment that ENDS there -- i.e. it reports the PRE-affect state,
            # matching ``event_states_before``. Assigning it to the starting segment
            # instead would return the post-jump state and silently shift volumes and
            # concentrations at
            # exactly the times every yield and rate is anchored to.
            owns = (ts_window > t_current) & (ts_window <= segment_t1) & live
            seg_saved = sol.ys[:-1]
            # A bailing segment's REACHED slots are genuine converged values, so
            # they are kept (poisoning them would put ``inf`` into rows
            # ``fail_time`` calls valid).
            # But today the only array read back from a failed segment is
            # ``event_states_before``, which is ``inf`` -- so a failed lane
            # contributes exactly ZERO gradient. ``stop_gradient`` preserves that
            # contract; without it
            # the change would silently start backpropagating through a blow-up.
            seg_saved = jnp.where(
                seg_failed, jax.lax.stop_gradient(seg_saved), seg_saved
            )
            slots = start + jnp.arange(window, dtype=start.dtype)
            new_output_buffer = output_buffer.at[slots].set(
                jnp.where(owns[:, None], seg_saved, output_buffer[slots])
            )
            new_output_overflow = output_overflow | (
                jnp.where(live, hi - lo, 0) > window
            )
        else:
            new_output_buffer = output_buffer
            new_output_overflow = output_overflow

        # Record the time of the FIRST failure on this lane. ``terminated`` is the
        # INCOMING carry value (a prior segment already bailed), so ``first_failure``
        # is true only on the segment that fails first. ``t_current`` is the start of
        # that segment == the last preset/output node the lane successfully reached (a
        # segment boundary, not an internal adaptive step), i.e. the correct
        # conservative cutoff: measurements at ``t > fail_time`` are past the failed
        # solve. Stays ``inf`` for a lane that never fails.
        first_failure = seg_failed & (~terminated)
        # ``t_current`` — the START of the failing segment — stays the cutoff even
        # though ``SaveAt(ts=...)`` could now report the last output time actually
        # reached. That finer cutoff was implemented and rejected: on a blow-up the
        # solver *does* reach output points past the last good node, but the values
        # there are garbage (1e11 on the ``_BlowUpReactionModule`` fixture), so a
        # later cutoff stops masking them
        # and presents them as real predictions. The conservative node-level cutoff is
        # the whole point of ``fail_time``.
        new_fail_time = jnp.where(first_failure, t_current, fail_time)

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
            preset_index = jnp.where(
                preset_triggered,
                preset_local_indices[jnp.clip(next_preset_idx, 0)],
                -1,
            )
            y_pres = _dispatch_preset_affect(
                y_at_stop,
                t_at_stop,
                args,
                preset_affect_indices[jnp.clip(next_preset_idx, 0)],
                preset_index,
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
            preset_index = jnp.where(
                preset_triggered,
                preset_local_indices[jnp.clip(next_preset_idx, 0)],
                -1,
            )
            y_pres = _dispatch_preset_affect(
                y_at_stop,
                t_at_stop,
                args,
                preset_affect_indices[jnp.clip(next_preset_idx, 0)],
                preset_index,
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

        # Re-assert the inf sentinel AFTER the affect/discrete-callback block. On the
        # segment that first bails, ``terminated`` is still False, so a sanitising
        # callback here (e.g. an upper-clamping DiscreteCallback, or a
        # ManifoldProjection) would otherwise turn the inf back into a finite value and
        # hide the failure from downstream detection. Later (already ``terminated``)
        # segments are collapsed and skip the callbacks, so ``seg_failed`` is enough.
        y_after = jnp.where(seg_failed, jnp.asarray(jnp.inf, y_after.dtype), y_after)

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
        new_terminated = terminated | seg_failed

        event_time = jnp.where(any_event & (~done) & (~terminated), t_at_stop, t1)

        output = (
            event_time,
            jnp.where(done | terminated, jnp.int32(-1), event_type),
            y_before_event,
            y_after,
            jnp.asarray(sol.stats["num_steps"], dtype=jnp.int32),
        )

        # Seed the next segment from what this one actually sustained: its average
        # ACCEPTED step. Floored at ``dt0`` so the policy is MONOTONE -- it can
        # only ever
        # start larger than the fixed default, never smaller. Without that floor a very
        # short segment (two events close together) would hand a tiny step to the next,
        # possibly much longer, segment and force it to ramp back up -- a regression the
        # fixed ``dt0`` cannot have. A collapsed/zero-length segment carries ``dt_prev``
        # through unchanged.
        seg_len = jnp.maximum(t_at_stop - t_current, jnp.asarray(0.0, time_dtype))
        n_accepted = jnp.maximum(
            jnp.asarray(sol.stats["num_accepted_steps"], time_dtype), 1.0
        )
        new_dt = jnp.where(seg_len > 0, seg_len / n_accepted, dt_prev)
        new_dt = jnp.maximum(new_dt, jnp.asarray(dt0, time_dtype))

        return (
            y_after,
            t_next,
            new_done,
            new_terminated,
            new_fail_time,
            new_steps_used,
            new_dt,
            new_output_buffer,
            new_output_overflow,
        ), output

    # ---- Run the scan ----
    y0_arr = jnp.asarray(y0)
    if has_output:
        # No segment precedes ``t0``, and ownership is half-open, so a ``t0`` slot is
        # never written by the loop -- seed it with ``y0`` directly. Every other slot
        # starts at ``inf`` so an unreached row is detectable rather than stale.
        output_init = jnp.where(
            (output_times <= t0)[:, None],
            y0_arr[None, :],
            jnp.asarray(jnp.inf, y0_arr.dtype),
        )
    else:
        output_init = jnp.zeros((0,) + y0_arr.shape, dtype=y0_arr.dtype)
    init_carry = (
        y0,
        t0,
        jnp.bool_(False),
        jnp.bool_(False),
        jnp.asarray(jnp.inf, dtype=time_dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(dt0, dtype=time_dtype),
        output_init,
        jnp.bool_(False),
    )
    final_carry, outputs = jax.lax.scan(scan_fn, init_carry, None, length=max_events)
    (
        y_final,
        t_final,
        _,
        _,
        fail_time,
        _,
        _,
        output_states,
        output_overflow,
    ) = final_carry
    event_times, event_types, states_before, states_after, segment_num_steps = outputs

    event_count = jnp.sum((event_types >= 0).astype(jnp.int32))

    return CallbackSolution(
        y_final=y_final,
        t_final=t_final,
        fail_time=fail_time,
        event_times=event_times,
        event_types=event_types,
        event_states_before=states_before,
        event_states_after=states_after,
        event_count=event_count,
        segment_num_steps=segment_num_steps,
        output_states=output_states if has_output else None,
        output_overflow=output_overflow if has_output else None,
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
    state_dtype = jnp.asarray(y0).dtype
    t0 = jnp.asarray(t0, dtype=state_dtype)
    t1 = jnp.asarray(t1, dtype=state_dtype)
    dt0 = jnp.asarray(dt0, dtype=state_dtype)
    ts = jnp.asarray(ts, dtype=state_dtype)
    solve_terms = _wrap_ode_term_dtype(terms)

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
        seg_y0 = jnp.asarray(segment_y0s[seg_idx], dtype=state_dtype)

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
            solve_terms,
            solver,
            jnp.asarray(seg_t0, dtype=state_dtype),
            jnp.asarray(seg_t1, dtype=state_dtype),
            dt0=jnp.minimum(dt0, jnp.asarray(seg_t1 - seg_t0, dtype=state_dtype)),
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
        return jnp.array([], dtype=state_dtype), jnp.zeros(
            (0, y0.shape[0]), dtype=state_dtype
        )
