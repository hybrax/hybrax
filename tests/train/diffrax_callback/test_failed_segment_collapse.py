"""Forward-side regression tests for failed-segment collapse + inf sentinel.

When a segment bails (max_steps / dt_min / etc.) diffeqsolve_with_callbacks marks that
vmap lane's state with an inf sentinel and collapses its remaining segments to zero
length. These tests pin only the STABLE forward contract:
  1. a failed lane's final state is the inf sentinel; healthy lanes are unchanged;
  2. a failure on an early segment stays poisoned through later slots;
  2b. the production-consumed ``event_states_before``/``after`` are poisoned;
  2c. every segment after a failure takes zero solver steps;
  3. a finite ``dt_min_reached`` bail is also detected and poisoned;
  4. a legitimate continuous ``event_occurred`` stop is NOT treated as a failure;
  5. a sanitising DiscreteCallback cannot hide the failure;
  6. a healthy solve reports ``fail_time == inf``;
  7. a mid-trajectory bail sets ``fail_time`` to the last successfully-reached node,
     so a ``t_meas <= fail_time`` mask keeps the good early points and drops the rest.

Deliberately NO assertion on reverse-mode gradients or optimiser behaviour: whether a
failed lane yields a finite or nan gradient is slot- and loss-dependent (an unstable
property), and the failure->optimiser policy lives in the caller/harness, not here.
"""

import jax
import jax.numpy as jnp
import diffrax

from diffrax_callbacks import (
    ContinuousCallback,
    DiscreteCallback,
    PresetTimeCallback,
    CallbackSet,
    diffeqsolve_with_callbacks,
)

jax.config.update("jax_enable_x64", True)

_EVENTS = jnp.linspace(0.4, 2.0, 5, dtype=jnp.float64)  # max_events > 1


def _solve_lane(freq, *, callbacks=None, controller=None, max_steps=64, max_events=5):
    """One trajectory. ``freq`` drives an oscillatory forcing; large freq -> stiff ->
    exhausts ``max_steps``. freq == 0 -> smooth healthy decay."""

    def rhs(t, y, args):
        return -0.3 * y + freq * jnp.sin(freq * t) * y

    cb = (
        callbacks
        if callbacks is not None
        else PresetTimeCallback(times=_EVENTS, affect_fn=lambda y, t, args, i: y)
    )
    sol = diffeqsolve_with_callbacks(
        diffrax.ODETerm(rhs),
        diffrax.Tsit5(),
        t0=0.0,
        t1=2.0,
        dt0=1e-3,
        y0=jnp.ones(2, dtype=jnp.float64),
        callbacks=cb,
        max_events=max_events,
        stepsize_controller=controller or diffrax.PIDController(rtol=1e-8, atol=1e-11),
        max_steps_per_segment=max_steps,
    )
    return sol.y_final


def test_failed_lane_poisoned_healthy_lanes_unchanged():
    """(1) float64 vmap: failed lane -> inf sentinel; healthy lanes bit-identical to an
    all-healthy batch."""
    freqs = jnp.array([0.0, 2.0e4, 0.0, 0.0], dtype=jnp.float64)
    healthy_freqs = jnp.zeros(4, dtype=jnp.float64)
    out = jax.jit(jax.vmap(_solve_lane))(freqs)
    ref = jax.jit(jax.vmap(_solve_lane))(healthy_freqs)

    assert bool(jnp.all(jnp.isinf(out[1]))), "failed lane must be the inf sentinel"
    healthy = jnp.array([0, 2, 3])
    assert bool(jnp.all(jnp.isfinite(out[healthy]))), "healthy lanes must stay finite"
    assert bool(jnp.array_equal(out[healthy], ref[healthy])), (
        "healthy lanes must be bit-identical to the all-healthy batch (isolation)"
    )


def test_early_failure_remains_poisoned():
    """(2) A lane that fails on an early segment is still inf at the final state, with
    several event slots after the failure.

    NB: this test alone pins only that the failure *remains* poisoned. The next test
    independently verifies the zero-length collapse through ``segment_num_steps``.
    """
    out = jax.jit(_solve_lane)(jnp.asarray(2.0e4, jnp.float64))
    assert bool(jnp.all(jnp.isinf(out))), "failure must remain poisoned to the end"


def test_post_failure_segments_take_zero_steps():
    """The performance fix: after an early bail, every remaining segment is a
    zero-length solve and must take no adaptive solver steps."""
    max_steps = 64

    def rhs(t, y, args):
        return 2.0e4 * jnp.sin(2.0e4 * t) * y

    sol = diffeqsolve_with_callbacks(
        diffrax.ODETerm(rhs),
        diffrax.Tsit5(),
        t0=0.0,
        t1=2.0,
        dt0=1e-3,
        y0=jnp.ones(2, dtype=jnp.float64),
        callbacks=PresetTimeCallback(times=_EVENTS, affect_fn=lambda y, t, args, i: y),
        max_events=5,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-11),
        max_steps_per_segment=max_steps,
    )

    steps = sol.segment_num_steps.tolist()
    assert float(sol.fail_time) == 0.0
    assert steps[0] > 0
    assert steps[1:] == [0, 0, 0, 0]


def test_event_states_before_and_after_are_poisoned():
    """(2b) The arrays production actually consumes are poisoned. physical_solve gathers
    ``sol.event_states_before`` (not ``y_final``), so the pre-affect-block poison must
    reach it. Use an all-parked early failure (fails on segment 1 before any node), so
    every logged pre-state belongs to the failed/collapsed lane.

    This pins BOTH poison placements: removing the pre-block ``y_at_stop`` poison leaves
    ``event_states_before`` finite (caught here) even though ``y_final`` stays inf;
    removing the post-block re-assert leaves ``event_states_after`` sanitisable.
    """

    def rhs(t, y, args):
        # blows max_steps immediately (from t0), so no measurement node is ever reached
        return 3.0e4 * jnp.sin(3.0e4 * t) * y

    sol = diffeqsolve_with_callbacks(
        diffrax.ODETerm(rhs),
        diffrax.Tsit5(),
        t0=0.0,
        t1=2.0,
        dt0=1e-3,
        y0=jnp.ones(2, dtype=jnp.float64),
        callbacks=PresetTimeCallback(times=_EVENTS, affect_fn=lambda y, t, args, i: y),
        max_events=5,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-11),
        max_steps_per_segment=64,
    )
    assert bool(jnp.all(jnp.isinf(sol.event_states_before))), (
        "event_states_before (gathered by physical_solve) must be poisoned"
    )
    assert bool(jnp.all(jnp.isinf(sol.event_states_after))), (
        "event_states_after must be poisoned"
    )


def test_dt_min_reached_is_detected():
    """(3) A finite ``dt_min_reached`` bail (not max_steps) is also poisoned — pins the
    broadened result predicate, not just == max_steps_reached."""

    # RHS goes non-finite past t=0.5; with force_dtmin=False the controller bails with
    # ``dt_min_reached`` and a finite last-reached state (i.e. NOT caught by a
    # max_steps-only or isfinite check).
    def rhs(t, y, args):
        return jnp.where(t < 0.5, y, jnp.nan * y)

    controller = diffrax.PIDController(
        rtol=1e-10, atol=1e-12, dtmin=1e-4, force_dtmin=False
    )
    sol = diffeqsolve_with_callbacks(
        diffrax.ODETerm(rhs),
        diffrax.Tsit5(),
        t0=0.0,
        t1=1.0,
        dt0=1e-3,
        y0=jnp.asarray([1.0], dtype=jnp.float64),
        callbacks=PresetTimeCallback(
            times=jnp.asarray([0.25, 0.75, 1.0], dtype=jnp.float64),
            affect_fn=lambda y, t, args, i: y,
        ),
        max_events=4,
        stepsize_controller=controller,
        max_steps_per_segment=10_000,
    )
    assert bool(jnp.all(jnp.isinf(sol.y_final))), "dt_min_reached bail must be poisoned"


def test_continuous_event_is_not_treated_as_failure():
    """(4) A legitimate continuous-event stop reports ``event_occurred`` and must NOT be
    poisoned."""

    def cross_zero(y, t, args):
        # trigger when the (decaying) first component drops below 0.6
        return y[0] - 0.6

    cc = ContinuousCallback(
        condition_fn=cross_zero,
        affect_fn=lambda y, t, args: y,
    )
    out = jax.jit(lambda: _solve_lane(jnp.asarray(0.0, jnp.float64), callbacks=cc))()
    assert bool(jnp.all(jnp.isfinite(out))), (
        "a continuous event stop must stay finite (event_occurred is not a failure)"
    )


def test_fail_time_is_inf_on_a_healthy_solve():
    """(6) A solve that never bails reports ``fail_time == inf`` — so the downstream
    ``t_meas <= fail_time`` mask keeps every point (no-op)."""
    sol = diffeqsolve_with_callbacks(
        diffrax.ODETerm(lambda t, y, args: -0.3 * y),
        diffrax.Tsit5(),
        t0=0.0,
        t1=2.0,
        dt0=1e-3,
        y0=jnp.ones(2, dtype=jnp.float64),
        callbacks=PresetTimeCallback(times=_EVENTS, affect_fn=lambda y, t, args, i: y),
        max_events=5,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-11),
        max_steps_per_segment=64,
    )
    assert bool(jnp.isinf(sol.fail_time)), "healthy solve must have fail_time == inf"


def test_fail_time_marks_the_last_good_node():
    """(7) A mid-trajectory bail sets ``fail_time`` to the start of the failing segment
    == the last successfully-reached node, so ``t_meas <= fail_time`` keeps the good
    early nodes and drops the post-failure ones."""

    # Smooth until t=0.5, then non-finite: the segment leaving the t=0.25 node crosses
    # 0.5 and bails (dt_min). The node at 0.25 was reached; 0.75 and 1.0 are post-fail.
    def rhs(t, y, args):
        return jnp.where(t < 0.5, y, jnp.nan * y)

    controller = diffrax.PIDController(
        rtol=1e-10, atol=1e-12, dtmin=1e-4, force_dtmin=False
    )
    sol = diffeqsolve_with_callbacks(
        diffrax.ODETerm(rhs),
        diffrax.Tsit5(),
        t0=0.0,
        t1=1.0,
        dt0=1e-3,
        y0=jnp.asarray([1.0], dtype=jnp.float64),
        callbacks=PresetTimeCallback(
            times=jnp.asarray([0.25, 0.75, 1.0], dtype=jnp.float64),
            affect_fn=lambda y, t, args, i: y,
        ),
        max_events=4,
        stepsize_controller=controller,
        max_steps_per_segment=10_000,
    )
    assert float(sol.fail_time) == 0.25, (
        "fail_time must be the last successfully-reached node (0.25), not later"
    )
    # The mask a caller applies: keep t <= fail_time, drop the rest.
    t_meas = jnp.asarray([0.25, 0.75, 1.0], dtype=jnp.float64)
    keep = t_meas <= sol.fail_time
    assert keep.tolist() == [True, False, False]


def test_clamping_callback_cannot_hide_failure():
    """(5) A sanitising DiscreteCallback on the failing boundary must not un-poison the
    lane (poison is re-asserted after the callback block)."""
    clamp = DiscreteCallback(
        condition_fn=lambda y, t, args: jnp.any(y > 1e6),
        affect_fn=lambda y, t, args: jnp.minimum(y, 10.0),
    )
    preset = PresetTimeCallback(times=_EVENTS, affect_fn=lambda y, t, args, i: y)
    cbset = CallbackSet(preset, clamp)
    freqs = jnp.array([0.0, 2.0e4, 0.0], dtype=jnp.float64)
    out = jax.jit(jax.vmap(lambda f: _solve_lane(f, callbacks=cbset)))(freqs)
    assert bool(jnp.all(jnp.isinf(out[1]))), "clamp must not sanitise the failed lane"
    assert bool(jnp.all(jnp.isfinite(out[jnp.array([0, 2])]))), (
        "healthy lanes stay finite"
    )
