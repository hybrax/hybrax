"""Bounded physical-state ODE solve with discrete bolus/sample jumps.

Integrates ``[RAW_RMCs | RAW_PVs | RAW_V | RAW_modeled_cum | RAW_latent]`` in
**scaled** space (each component divided by its characteristic
``SCALE_integrated_state``),
so one rtol/atol controls every species uniformly and the adjoint stays O(1) — better
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

_EVENT_EPS = 1e-4  # << ~0.08 h event spacing; > float32 ULP at t~15


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
):
    """RAW physical states at each (padded) measurement time, ``[max_n_meas, n_state]``.

    ``max_steps`` (from ``--solver-max-steps``) sets the per-segment step ceiling,
    but it is **clamped to a barrier-safe maximum** (512). This matters under pmap:
    a *high* per-segment cap lets one stiff process (e.g. under a too-hot lr) spin for
    many seconds inside its solve while the other devices finish and block on the
    ``all_gather``; once one device lags >~20 s the XLA collective rendezvous times out
    and the whole run dead-locks. A bounded cap instead makes a genuinely stiff segment
    *bail fast* (returns ``inf``, which ``zero_nans`` in the optimiser absorbs) rather
    than hang the barrier. Segments are short and exit early anyway, so the clamp costs
    nothing in the stable case (512 vs 8096 → identical speed, measured).
    ``max_steps_per_segment=None`` ⇒ ``min(512, max_steps)``; pass a value to pin a
    smaller cap (e.g. tests).
    """
    if max_steps_per_segment is None:
        max_steps_per_segment = min(512, int(max_steps))
    n_RMCs = len(wrapper.modeled_RMC_names)
    n_PVs = len(wrapper.modeled_PV_names)
    n_FVCs = len(wrapper.modeled_FVC_names)
    n_latent = wrapper.reaction_module.n_latent
    n_state = n_RMCs + n_PVs + 1 + n_FVCs + n_latent
    dtype = RAW_y0.dtype
    controls = wrapper.controls
    min_V = jnp.asarray(wrapper.min_V, dtype=dtype)
    # Per-state characteristic scale (the user-definable ``SCALE_*`` hook). Integrate
    # ``SCL = RAW / SCALE`` so a single rtol/atol is uniformly meaningful and the
    # adjoint stays O(1). Pure linear reparametrization applied only at the solve
    # boundary: the RHS, jumps and gather stay physical; states are unscaled back
    # before returning.
    SCALE = jnp.asarray(wrapper.reaction_module.SCALE_integrated_state, dtype=dtype)
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
    eps = jnp.asarray(_EVENT_EPS, dtype=dtype)
    bolus_times = jnp.where(bmask, bt, BIG)
    sample_times = jnp.where(smask, st, BIG)
    # Park any measurement/output time that coincides (within eps) with a bolus/sample
    # node. affect_fn applies *every* event within eps of the current node, so a
    # measurement node sitting within eps of a feed would apply that feed a SECOND time
    # — double-counting it and drifting the volume/concentrations by ~one bolus from
    # there on (visible in dense prediction grids, where a grid point can land on a
    # feed). The event log still has the feed/sample node, so the gather below recovers
    # the state at that measurement time. On the measurement grid (the loss) no point
    # coincides with a feed, so this is a no-op there.
    near_event = jnp.any(
        (jnp.abs(t_eval[:, None] - bt[None, :]) < eps) & bmask[None, :], axis=1
    ) | jnp.any((jnp.abs(t_eval[:, None] - st[None, :]) < eps) & smask[None, :], axis=1)
    meas_times = jnp.where(meas_active & ~near_event, t_eval, BIG)
    preset_times = jnp.concatenate([bolus_times, sample_times, meas_times])

    def affect_fn(y_scl, t, args):
        y = y_scl * SCALE  # scaled -> physical (the jump is a physical mass balance)
        C = y[:n_RMCs]
        # Modeled PVs are intensive (ratios/observables), so volume jumps
        # don't touch them.
        PVs = y[n_RMCs : n_RMCs + n_PVs]
        V = y[n_RMCs + n_PVs]
        cum = y[n_RMCs + n_PVs + 1 : n_RMCs + n_PVs + 1 + n_FVCs]
        h = y[n_RMCs + n_PVs + 1 + n_FVCs :]
        s_on = (jnp.abs(t - st) < eps) & smask
        sample_dv = jnp.sum(jnp.where(s_on, sv, 0.0))
        b_on = (jnp.abs(t - bt) < eps) & bmask
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
    # RHS evaluated on the unscaled state; the derivative is rescaled (scale_state is
    # linear, so d(SCL)/dt == d(RAW)/dt / SCALE).
    term = diffrax.ODETerm(lambda t, yy, a: wrapper.physical_rhs(t, yy * SCALE) / SCALE)

    sol = diffeqsolve_with_callbacks(
        term,
        diffrax.Tsit5(),
        t0=t0,
        t1=t1,
        dt0=(t1 - t0) * 1e-3,  # small initial step; adaptive controller adapts
        y0=y0 / SCALE,
        callbacks=cb,
        max_events=preset_times.shape[0],
        stepsize_controller=diffrax.PIDController(
            rtol=rtol, atol=atol, jump_ts=jump_ts_arg
        ),
        max_steps_per_segment=max_steps_per_segment,
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
        s_here = (jnp.abs(tm - st) < eps) & smask
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
    at_t0 = (jnp.abs(t_eval - t0) < eps) & meas_active
    states = jnp.where(at_t0[:, None], y0[None, :], states)
    return states
