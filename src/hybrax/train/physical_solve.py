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
and ``RecursiveCheckpointAdjoint`` keeps only ~``O(log max_steps)`` checkpoints, so RAM
is **flat in the number of events** — *not* ``max_steps × #segments`` (measured ~2.34 GB
for a full 37-way step).

Output times (measurements, and any dense/prediction grid spliced in by
``dense.build_union_time_grid``) are NOT segment boundaries. Segments are decided purely
by the physics — bolus and sample events, the only things that jump the state — and the
trajectory is read out with ``SaveAt(ts=...)`` inside each segment, which is pure
interpolation and costs no solver steps. Making every output time a boundary used to
subdivide the integration: a 2023_bayer-shaped process went from 10 segments / 38 ODE
steps on the measurement grid to 208 / 426 once a 200-point export grid was requested.
"""

from __future__ import annotations

import math

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from diffrax_callbacks import PresetTimeCallback, diffeqsolve_with_callbacks
from .wrapper import HybridOdeWrapper

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


def _output_window(controls, n_linspace: int) -> int:
    """Static per-segment output-window size, given how many linspace points are spliced
    into the output grid.

    Without a window each segment is handed the whole grid and diffrax writes every slot
    in every segment, so the save work is ``O(n_segments * n_output)`` — measured 10x
    SLOWER than a boundary save on a 160-event process under vmap. With it the work is
    ``O(n_segments * window)``, and the per-segment cost scales with the window
    (10.6 us/segment for a plain ``SaveAt(t1=True)``, 14 at window 1, 68 at 25, 209
    at 200), so the window wants to be as tight as it can provably be.

    The bound is exact, not an estimate. An output grid is the measurement block plus up
    to two ``linspace(t0, t1, N)`` blocks:

    - measurement block: at most ``max_measurements_per_event_gap`` per gap, counted
      exactly (on the PADDED grid) at prepare time.
    - a linspace block of ``N`` points has spacing ``h = (t1 - t0) / (N - 1)``, so
      a gap of length ``f * (t1 - t0)`` holds at most
      ``floor(f * (N - 1)) + 1`` of them. Two blocks therefore contribute at most
      ``f * n_linspace + 2 <= ceil(f * n_linspace) + 2``.

    ``n_linspace`` is a static Python int: the caller building the union grid knows
    ``dense_grid_n`` and ``prediction_grid_n`` exactly. A caller that just has a
    grid and
    cannot say passes the whole grid length instead, which is a valid (looser) bound
    because the measurement block it double-counts is itself bounded by ``f``.
    """
    return (
        math.ceil(controls.max_event_gap_fraction * n_linspace)
        + controls.max_measurements_per_event_gap
        + 2
    )


def solve_physical_states(
    wrapper: HybridOdeWrapper,
    *,
    t_eval: jax.Array,
    n_measured: jax.Array,
    RAW_y0: jax.Array,
    max_steps: int,
    rtol: float,
    atol: float,
    jump_ts: jax.Array | None = None,
    n_linspace: int | None = None,
    return_fail_time: bool = False,
):
    """RAW physical states at each (padded) measurement time, ``[max_n_meas, n_state]``.

    ``n_linspace`` is how many of ``t_eval``'s points come from evenly-spaced blocks
    spliced in by :func:`dense.build_union_time_grid` (``dense_grid_n`` +
    ``prediction_grid_n``), as opposed to the measurement block. It only sizes the
    per-segment output window (:func:`_output_window`), and passing it exactly is what
    makes the window tight. ``None`` falls back to the whole grid length: still a valid
    bound, just looser, which is the right default for a caller that simply has a grid
    and no way to say how it was composed.

    ``fail_time`` is the scalar time of the first segment failure on this solve (``inf``
    if none), i.e. the start of the failing segment == the last node it reached. Rows at
    ``t > fail_time`` are overwritten either way:
    - ``return_fail_time=True`` (loss path) returns ``(states, fail_time)`` with those
      rows set to a finite ``y0`` placeholder, so the ``penalty(state) * mask`` idiom
      stays safe; the caller must still mask them out via ``fail_time``.
    - ``return_fail_time=False`` (forward/export) returns ``states`` with those rows set
      to ``inf``, so a failed forward is detectable and never reads back a stale value.

    ``max_steps`` (from ``--solver-max-steps``) is the **budget for the whole
    solve**: it
    bounds each segment's inner solve and the running sum across segments, so exceeding
    it is the same failure as any segment bail (state poisoned to ``inf``, ``fail_time``
    recorded, later segments collapsed). It is grid-independent — chopping the same
    horizon into more segments does not multiply the ceiling, so ``fail_time`` no longer
    moves just because a denser output grid was requested.
    """
    n_RMCs = len(wrapper.modeled_RMC_names)
    n_PVs = len(wrapper.modeled_PV_names)
    n_FVCs = len(wrapper.modeled_FVC_names)
    n_latent = wrapper.reaction_module.n_latent
    n_state = n_RMCs + n_PVs + 1 + n_FVCs + n_latent
    dtype = RAW_y0.dtype
    controls = wrapper.controls
    min_V = jnp.asarray(controls.min_V, dtype=dtype)
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

    # TIER A — segment boundaries. Only bolus and sample events, because only they jump
    # the state (see ``affect_fn``). Output times are NOT here: they are read out with
    # ``SaveAt(ts=...)`` inside a segment, which is interpolation and costs no solver
    # steps. Vector-field kinks (control-spline knots, ``discrete_events``) are handled
    # by ``PIDController(jump_ts=...)`` below and were never boundaries either.
    # Inactive/padded slots are parked past t1 so they never trigger.
    BIG = t1 + jnp.asarray(1.0e6, dtype=dtype)
    bolus_times = jnp.where(bmask, bt, BIG)
    sample_times = jnp.where(smask, st, BIG)
    preset_times = jnp.concatenate([bolus_times, sample_times])

    # Tier-B output grid. Inactive/padded slots are parked at ``t1`` (NOT at ``BIG``:
    # ``SaveAt(ts=...)`` requires every entry inside ``[t0, t1]`` and ascending). The
    # active prefix is already ascending and ``t1`` is its maximum, so the parked tail
    # keeps the array sorted. This is the same value ``clamp_padded_time_rows`` writes,
    # so the production path (which clamps before calling) is unaffected and a direct
    # caller passing a raw zero-padded row is handled identically.
    output_times = jnp.where(meas_active, t_eval, t1)

    def apply_events(y_scl, t_node):
        """Apply every sample and bolus stored at one exact event time."""
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
        V_after_sample = eqx.error_if(
            V_after_sample,
            jnp.any(s_on) & (V_after_sample <= min_V),
            "sample reached minimum reactor volume.",
        )
        V_after = V_after_sample + bolus_dv
        C2 = (C * V_after_sample + bolus_mass) / V_after
        # physical -> scaled; PVs and latent pass through unchanged.
        return jnp.concatenate([C2, PVs, V_after[None], cum, h]) / SCALE

    def affect_fn(y_scl, t, args, preset_index):
        # ``preset_index == -1`` means the dispatcher is being evaluated speculatively
        # for a lane where no preset fired. This occurs when ``vmap`` batches lanes with
        # different event progress; mask the event time rather than relying on
        # ``lax.cond``, whose batched form evaluates both branches.
        t_node = jnp.where(preset_index >= 0, preset_times[preset_index], jnp.inf)
        # Otherwise, ``preset_index`` is the slot in ``preset_times`` where the
        # solver stopped, so the firing node's time is looked up EXACTLY -- no
        # tolerance, and no dependence on
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
        return apply_events(y_scl, t_node)

    # ``jump_ts`` = genuine vector-field discontinuity times (from
    # ``BioProcess.discrete_events``); ``None``/empty ⇒ the controller behaves
    # exactly as a plain ``PIDController(rtol, atol)``. Bolus/sample STATE jumps
    # are handled by ``affect_fn`` below, NOT here.
    jump_ts_arg = jump_ts if jump_ts is not None and jump_ts.shape[0] > 0 else None

    cb = PresetTimeCallback(times=preset_times, affect_fn=affect_fn)
    y0 = wrapper.initial_physical_state_from_raw(RAW_y0)
    V_index = n_RMCs + n_PVs
    y0 = eqx.error_if(
        y0,
        y0[V_index] <= min_V,
        "initial state reached minimum reactor volume.",
    )
    sample_at_t0 = jnp.sum(jnp.where((st == t0) & smask, sv, 0.0))
    y0_report = y0.at[V_index].add(-sample_at_t0)
    y0 = apply_events(y0 / SCALE, t0) * SCALE
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
        # N jump nodes need N+1 segments: one stop per node plus the final leg to t1.
        # (With measurement times in the preset set t1 was itself a preset, so the old
        # ``shape[0]`` already covered the last leg.)
        max_events=preset_times.shape[0] + 1,
        output_times=output_times,
        output_window=_output_window(
            wrapper.controls,
            output_times.shape[0] if n_linspace is None else n_linspace,
        ),
        stepsize_controller=diffrax.PIDController(
            rtol=rtol, atol=atol, jump_ts=jump_ts_arg
        ),
        max_steps=int(max_steps),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
    )

    # Assertion, not a safety net: the window bound is exact (see
    # ``_output_window``), so
    # this can only fire if that derivation is wrong. Left in because the alternative
    # failure mode is silent -- the excess rows would simply stay ``inf``.
    states = eqx.error_if(
        sol.output_states,
        sol.output_overflow,
        "output window too small: a segment owned more output times than it can hold",
    )
    states = states * SCALE  # scaled -> physical (states are returned unscaled)
    # Report the POST-sample state at a sample-coincident time: a well-mixed sample
    # leaves concentrations unchanged and only removes volume, so subtract any sample
    # volume scheduled at that time from V. Boluses stay pre-bolus (the offline
    # measurement is taken before feeding). This makes the reported V honour
    # ``v0 + Σfeeds − Σsamples`` at the final sample, matching pmap. Exact, for the same
    # reason ``affect_fn`` matches exactly: with a tolerance an output point merely NEAR
    # a sample would also get the correction, even though the solve has already applied
    # that sample to the state it is reading. Keyed on the caller's ``t_eval`` (not the
    # parked ``output_times``) so padded rows behave exactly as before.
    s_here = (t_eval[:, None] == st[None, :]) & smask[None, :]
    sample_dv_here = jnp.sum(jnp.where(s_here, sv[None, :], 0.0), axis=1)
    states = states.at[:, n_RMCs + n_PVs].add(-sample_dv_here)
    # Every grid point at t0 reports the post-sample, pre-bolus state. The dense /
    # prediction export solves on a union grid that can carry t0 at *several* indices
    # (measurement-t0, dense-t0, prediction-t0), not only index 0 — so patch all of
    # them. The solve already seeds its t0 slots with ``y0``, but only in SCALED space,
    # so the ``/ SCALE`` → ``* SCALE`` round trip can differ from ``y0`` by an ULP;
    # this writes the exact physical value.
    # Exact: ``t0`` IS ``t_eval[0]``, and every extra t0 row comes from a
    # ``linspace(t0, ...)`` whose first element is bitwise ``t0``.
    at_t0 = (t_eval == t0) & meas_active
    states = jnp.where(at_t0[:, None], y0_report[None, :], states)

    # Overwrite every post-failure row (``t > fail_time``) from ``fail_time``. Healthy
    # solve: ``fail_time == inf`` -> no-op.
    fail_time = sol.fail_time
    post_fail = ~within_fail_time(t_eval, fail_time)
    if return_fail_time:
        # LOSS-FACING path: replace post-failure rows with a finite ``y0`` placeholder
        # so the ``penalty(state) * mask`` idiom stays safe (an inf left in makes
        # ``0 * inf = nan`` poison the loss/gradient even for a masked row). The caller
        # still EXCLUDES these rows via ``fail_time`` (measurement mask +
        # ``dense_valid_time``); the placeholder only guarantees finiteness.
        #
        # The ``~isfinite`` arm is load-bearing, not defensive. ``within_fail_time``
        # classifies rows up to ``fail_time + _FAIL_TIME_EPS`` as valid, while the solve
        # writes ``inf`` for every slot it did not reach — so an output time inside that
        # 1e-4 window is called valid AND is ``inf``. Time alone is therefore not a
        # sufficient test for finiteness here.
        states = jnp.where(
            post_fail[:, None] | ~jnp.isfinite(states), y0[None, :], states
        )
        return states, fail_time
    # FORWARD/EXPORT path (``simulate_measurement_states``, plotting): write an explicit
    # non-finite sentinel so a failed forward is detectable and never reads back as a
    # stale finite value silently presented as a real prediction.
    states = jnp.where(post_fail[:, None], jnp.asarray(jnp.inf, states.dtype), states)
    return states
