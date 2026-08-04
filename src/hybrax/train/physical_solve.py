"""Bounded physical-state ODE solve with discrete bolus/sample jumps.

Integrates ``[RAW_RMCs | RAW_PVs | RAW_V | RAW_modeled_cum | RAW_latent]`` in
**scaled** space (each value transformed by its composed
``SCALE_integrated_state`` scaler: ``(RAW - offset) / scale``), so one
rtol/atol controls every species uniformly and the adjoint stays O(1) — better
float32 conditioning across a wide dynamic range) and applies boluses/samples as
discrete state jumps at their (known) event times, using
``diffrax_callbacks`` (a differentiable discrete-event layer for diffrax). The jumps
are applied between separate segment solves, so the adjoint is standard and correct
(verified: gradient through a preset jump matches the analytic value to 9 digits) —
unlike a custom solver that modifies state inside ``step`` (which silently breaks the
diffrax adjoint).

RAM: the segments run in a sequential ``lax.scan`` (one segment's solve live at a time)
and ``RecursiveCheckpointAdjoint`` keeps only ~``O(log max_steps_per_segment)``
checkpoints, so RAM is **flat in the number of events** — *not*
``max_steps × #segments`` (measured ~2.34 GB for a full 37-way step). The
per-segment cap is therefore a cheap *safety* ceiling, not a budget that must be
rationed across segments.
"""

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp

from diffrax_callbacks import PresetTimeCallback, diffeqsolve_with_callbacks

# Boundary headroom for the failure cutoff. ``fail_time`` is the START of the first
# segment that bailed, i.e. the last node the solver actually reached — so a point
# sitting exactly there WAS reached and must not be dropped by float noise in the
# ``<=``. Only ever relevant at that single boundary point of a failed solve (on a
# healthy solve ``fail_time`` is ``inf`` and the comparison is all-True). Wants to be
# as small as possible while still capturing the boundary node; the next node is
# typically orders of magnitude further away than this.
_FAIL_TIME_EPS = 1e-4

# Initial step for a DEGENERATE solve window (``t1 == t0``: one active measurement, or
# every measurement at the same timestamp), where ``span * 1e-3`` is exactly zero.
# diffrax requires a strictly positive ``dt0``; a zero one makes it bail on the
# tol-floored micro-segment inside the callback loop, and that bail is then flagged as
# a solve failure and poisoned to ``inf`` — so a healthy single-point process would
# look like a crashed solve. Any positive value works; the adaptive controller corrects
# on the first step.
_DEGENERATE_DT0 = 1e-4

# Per-segment step ceiling — a LATENCY bound, deliberately independent of the step
# budget. Under pmap a high per-segment cap lets one stiff process (e.g. under a
# too-hot lr) spin for many seconds inside its solve while the other devices finish and
# block on the ``all_gather``; once one device lags >~20 s the XLA collective rendezvous
# times out and the whole run dead-locks. A bounded cap makes a genuinely stiff segment
# *bail fast* instead of hanging the barrier. Segments are short and exit early anyway,
# so the clamp costs nothing in the stable case (512 vs 8096 → identical speed,
# measured). Override per call for tests; the trajectory budget is ``max_steps``.
_MAX_STEPS_PER_SEGMENT = 512

# NB there is deliberately NO event-matching tolerance. ``affect_fn`` identifies the
# firing bolus/sample by the preset INDEX the solver hands it, then compares times
# exactly against its own ``preset_times`` array. A tolerance here would make any
# output node within it re-trigger the event — see ``affect_fn``.


def within_fail_time(t, fail_time):
    """True where time ``t`` is at/before a solve's failure cutoff (all-True when
    ``fail_time`` is ``inf``, i.e. no failure). Post-failure points are the complement,
    ``~within_fail_time``. Centralizes the ``fail_time + _FAIL_TIME_EPS`` boundary
    tolerance so the measurement mask, the dense mask, the loss-facing finiteness
    fallback and the export sentinel all use one identical cutoff."""
    return t <= fail_time + jnp.asarray(_FAIL_TIME_EPS, jnp.asarray(t).dtype)


def solve_physical_states(
    wrapper,
    *,
    t_eval: jax.Array,
    n_measured: jax.Array,
    RAW_y0: jax.Array,
    max_steps: int,
    rtol: float,
    atol: float,
    jump_ts: jax.Array | None = None,
    max_steps_per_segment: int | None = None,
    return_fail_time: bool = False,
):
    """RAW physical states at each (padded) measurement time, ``[max_n_meas, n_state]``.

    ``fail_time`` is the scalar time of the first segment failure on this solve (``inf``
    if none). Post-failure rows (``t > fail_time``) are overwritten from ``fail_time``
    either way — the raw gather is NOT trusted there, since on a mid-trajectory bail the
    argmin falls back to the last reached node and yields a stale FINITE value:
    - ``return_fail_time=True`` (loss path) returns ``(states, fail_time)`` with those
      rows set to a finite ``y0`` placeholder, so the ``penalty(state) * mask`` idiom
      stays safe; the caller must still mask them out via ``fail_time``.
    - ``return_fail_time=False`` (forward/export) returns ``states`` with those rows set
      to ``inf``, so a failed forward is detectable and never reads back a stale value.

    Two independent bounds apply, and exceeding either is the same failure as any
    segment bail (state poisoned to ``inf``, ``fail_time`` recorded, later segments
    collapsed):

    - ``max_steps`` (from ``--solver-max-steps``) is the **budget for the whole solve**,
      passed down as ``max_steps_total``. This is what makes the knob mean what it says.
      Without it the only bound was per-segment, so the effective ceiling was
      ``cap * n_segments`` — e.g. ``max_steps=8096`` on a 174-segment process really
      allowed ~89,000 steps — and ``fail_time`` depended on how finely the horizon
      happened to be chopped (a dense loss/export grid subdivides it, so the same model
      could bail at a different time purely because more output points were requested).
    - ``max_steps_per_segment`` is a **latency** bound against the pmap deadlock; see
      :data:`_MAX_STEPS_PER_SEGMENT`. It defaults to that constant and is settable per
      call; it is deliberately NOT derived from ``max_steps``, which measures a
      different thing.
    """
    if max_steps_per_segment is None:
        max_steps_per_segment = _MAX_STEPS_PER_SEGMENT
    n_RMCs = len(wrapper.modeled_RMC_names)
    n_PVs = len(wrapper.modeled_PV_names)
    n_FVCs = len(wrapper.modeled_FVC_names)
    n_latent = wrapper.reaction_module.n_latent
    n_state = n_RMCs + n_PVs + 1 + n_FVCs + n_latent
    dtype = RAW_y0.dtype
    controls = wrapper.controls
    min_V = jnp.asarray(wrapper.min_V, dtype=dtype)
    # Per-state characteristic scale (the user-definable ``SCALE_*`` hook). Integrate
    # ``SCL = (RAW - b) / s`` so a single rtol/atol is uniformly meaningful and the
    # adjoint stays O(1). Pure reparametrisation applied only at the solve
    # boundary: the RHS, jumps and gather stay physical; states are unscaled back
    # before returning. NB the ODE RHS derivative is scaled via
    # ``scale_derivative`` (offset-free) — NOT the value ``/`` path — so an affine
    # state offset is not spuriously subtracted from the derivative.
    SCALE = wrapper.reaction_module.SCALE_integrated_state.astype(dtype)
    assert SCALE.shape == (n_state,), (SCALE.shape, n_state)

    n_meas_arr = jnp.asarray(n_measured, dtype=jnp.int32)
    t_eval = jnp.asarray(t_eval, dtype=dtype)
    t0 = t_eval[0]
    t1 = t_eval[jnp.clip(n_meas_arr - 1, 0, t_eval.shape[0] - 1)]
    M = t_eval.shape[0]
    meas_active = jnp.arange(M) < n_meas_arr

    bt = controls.bolus_event_times.astype(dtype)
    bv = controls.bolus_event_volumes.astype(dtype)
    bC = controls.bolus_event_Cin.astype(dtype)
    bmask = controls.bolus_event_mask
    st = controls.sample_event_times.astype(dtype)
    sv = controls.sample_event_volumes.astype(dtype)
    smask = controls.sample_event_mask

    # Preset times = bolus ∪ sample ∪ measurement times. Inactive/padded slots are
    # parked past t1 so they never trigger. Measurement-only times get an identity
    # affect (so the event log records the state there for the loss).
    BIG = t1 + jnp.asarray(1.0e6, dtype=dtype)
    bolus_times = jnp.where(bmask, bt, BIG)
    sample_times = jnp.where(smask, st, BIG)
    meas_times = jnp.where(meas_active, t_eval, BIG)
    preset_times = jnp.concatenate([bolus_times, sample_times, meas_times])

    def affect_fn(y_scl, t, args, preset_index):
        # ``preset_index`` is the slot in ``preset_times`` the solver stopped at, so the
        # firing node's time is looked up EXACTLY -- no tolerance, and no dependence on
        # ``t`` (the solver's realised stop time) or on any dtype round-trip inside
        # diffrax_callbacks. Matching ``|t - bt| < eps`` instead used to let an output
        # node merely NEAR a feed re-apply it, double-counting the bolus and drifting
        # volume/concentrations for the rest of the trajectory; that needed a separate
        # guard parking such nodes. Exact lookup removes the failure mode outright.
        #
        # Co-timed events still group into ONE node: every bolus/sample whose stored
        # time equals ``t_node`` fires together (e.g. a sample and a feed both at 24 h),
        # which is required because the solver only accepts strictly-future nodes and
        # would otherwise skip the duplicate slots.
        t_node = preset_times[preset_index]
        y = y_scl * SCALE  # scaled -> physical (the jump is a physical mass balance)
        C = y[:n_RMCs]
        # Modeled PVs are intensive (ratios/observables), so volume jumps
        # don't touch them.
        PVs = y[n_RMCs : n_RMCs + n_PVs]
        V = y[n_RMCs + n_PVs]
        cum = y[n_RMCs + n_PVs + 1 : n_RMCs + n_PVs + 1 + n_FVCs]
        h = y[n_RMCs + n_PVs + 1 + n_FVCs :]
        s_on = (st == t_node) & smask
        sample_dv = jnp.sum(jnp.where(s_on, sv, 0.0))
        b_on = (bt == t_node) & bmask
        bolus_dv = jnp.sum(jnp.where(b_on, bv, 0.0))
        bolus_mass = jnp.sum(jnp.where(b_on[:, None], bC * bv[:, None], 0.0), axis=0)
        # Physical order at a coincident timestamp: sample FIRST (well-mixed removal —
        # concentrations unchanged, volume drops), THEN feed/bolus (dilute from the
        # post-sample volume and add fed mass). Bolus-before-sample dilutes fed species
        # from the larger pre-sample volume and systematically under-dilutes them.
        V_after_sample = V - sample_dv
        V_after = V_after_sample + bolus_dv
        C2 = (C * V_after_sample + bolus_mass) / jnp.maximum(V_after, min_V)
        # physical -> scaled; PVs and latent pass through unchanged.
        return jnp.concatenate([C2, PVs, V_after[None], cum, h]) / SCALE

    # ``jump_ts`` = genuine vector-field discontinuity times (from
    # ``BioProcess.discrete_events``); ``None``/empty ⇒ the controller behaves
    # exactly as a plain ``PIDController(rtol, atol)``. Bolus/sample STATE jumps
    # are handled by ``affect_fn`` below, NOT here.
    jump_ts_arg = jump_ts if jump_ts is not None and jump_ts.shape[0] > 0 else None

    cb = PresetTimeCallback(times=preset_times, affect_fn=affect_fn)
    y0 = wrapper.initial_physical_state_from_raw(RAW_y0)
    # RHS evaluated on the unscaled state (value unscale, ``yy * SCALE``); the
    # RAW derivative is rescaled via ``scale_derivative`` (offset-free: under an
    # affine scaler ``d((RAW-b)/s)/dt = (dRAW/dt)/s``, so the offset must NOT be
    # subtracted here — the value ``/`` path would subtract it).
    term = diffrax.ODETerm(
        lambda t, yy, a: SCALE.scale_derivative(wrapper.physical_rhs(t, yy * SCALE))
    )

    # ``dt0`` must be strictly positive. For a degenerate window (``t1 == t0``: a single
    # active measurement, or all measurements at t0) ``(t1 - t0) * 1e-3`` is exactly 0,
    # and a zero initial step makes diffrax bail on the tol-floored micro-segment inside
    # the callback loop; that bail is then flagged as a failure and poisoned to inf.
    # Only the exact zero-span case needs a fallback (any positive value works there, as
    # the adaptive controller adjusts immediately), so a genuinely short run keeps its
    # usual ``span * 1e-3`` untouched.
    span = t1 - t0
    dt0 = jnp.where(span > 0, span * 1e-3, jnp.asarray(_DEGENERATE_DT0, dtype))

    sol = diffeqsolve_with_callbacks(
        term,
        diffrax.Tsit5(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0 / SCALE,
        callbacks=cb,
        max_events=preset_times.shape[0],
        stepsize_controller=diffrax.PIDController(
            rtol=rtol, atol=atol, jump_ts=jump_ts_arg
        ),
        max_steps_per_segment=max_steps_per_segment,
        max_steps_total=int(max_steps),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
    )

    # The event log records the pre-affect state at every node (bolus/sample/meas).
    # Gather the state at each measurement time (closest *real* event in the log;
    # unused/padded slots are parked past t1 so they never win the argmin).
    ev_t = jnp.where(sol.event_types >= 0, sol.event_times, BIG)  # [max_events]
    ev_y = sol.event_states_before  # [max_events, n_state] (scaled)

    def _gather(tm):
        i = jnp.argmin(jnp.abs(ev_t - tm))
        y = ev_y[i] * SCALE  # scaled -> physical (states are returned unscaled)
        # Report the POST-sample state at a sample-coincident time: a well-mixed
        # sample leaves concentrations unchanged and only removes volume, so subtract
        # any sample volume scheduled at ``tm`` from V. Boluses stay pre-bolus (the
        # offline measurement is taken before feeding). This makes the reported V
        # honour ``v0 + Σfeeds − Σsamples`` at the final sample, matching pmap.
        # Exact, for the same reason ``affect_fn`` matches exactly: with a tolerance an
        # output point merely NEAR a sample would also get the correction, even though
        # the solve has already applied that sample to the state it is reading.
        s_here = (st == tm) & smask
        sample_dv_here = jnp.sum(jnp.where(s_here, sv, 0.0))
        return y.at[n_RMCs + n_PVs].add(-sample_dv_here)

    states = jax.vmap(_gather)(t_eval)  # [M, n_state]
    # Every grid point at t0 is the initial state (no event precedes t0). The dense /
    # prediction export solves on a union grid that can carry t0 at *several* indices
    # (measurement-t0, dense-t0, prediction-t0), not only index 0 — so patch all of
    # them. Patching only states[0] left the other t0 rows on the _gather boundary
    # value (first feed interval already integrated in), which surfaced as an inflated
    # V0 / diluted concentrations in the predictions.csv first row for continuous-feed
    # processes.
    # Exact: ``t0`` IS ``t_eval[0]``, and every extra t0 row comes from a
    # ``linspace(t0, ...)`` whose first element is bitwise ``t0``.
    at_t0 = (t_eval == t0) & meas_active
    states = jnp.where(at_t0[:, None], y0[None, :], states)

    # Overwrite every post-failure row (``t > fail_time``) from ``fail_time`` — do NOT
    # trust the raw gather to carry a marker there. On a MID-trajectory bail the
    # post-failure nodes are parked (``event_type == -1``), so the argmin lands on the
    # last successfully-reached node and returns a stale FINITE value (the inf sentinel
    # only survives for a first-segment bail, where the argmin defaults to the poisoned
    # index 0). Healthy solve: ``fail_time == inf`` -> no-op.
    fail_time = sol.fail_time
    post_fail = ~within_fail_time(t_eval, fail_time)
    if return_fail_time:
        # LOSS-FACING path: replace post-failure rows with a finite ``y0`` placeholder
        # so the ``penalty(state) * mask`` idiom stays safe (an inf left in makes
        # ``0 * inf = nan`` poison the loss/gradient even for a masked row). The caller
        # still EXCLUDES these rows via ``fail_time`` (measurement mask +
        # ``dense_valid_time``); the placeholder only guarantees finiteness.
        states = jnp.where(post_fail[:, None], y0[None, :], states)
        return states, fail_time
    # FORWARD/EXPORT path (``simulate_measurement_states``, plotting): write an explicit
    # non-finite sentinel so a failed forward is detectable and never reads back as a
    # stale finite value silently presented as a real prediction.
    states = jnp.where(post_fail[:, None], jnp.asarray(jnp.inf, states.dtype), states)
    return states
