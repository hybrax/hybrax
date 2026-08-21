"""Two cheap contracts of the segment scan that are easy to regress silently.

1. ``done`` collapse — the scan always runs ``max_events`` iterations. A lane that has
   consumed every preset and reached ``t1`` must collapse its remaining iterations to
   ZERO length. Without it each leftover iteration gets ``segment_t1 = t_current + tol``
   from the clamp, i.e. a tolerance-length (not zero-length) segment that diffrax runs at
   >= 1 accepted step. Those steps cost a full 6-stage Tsit5 evaluation each AND they
   accumulate into the ``max_steps`` budget, so a caller could lose ~10% of its trajectory
   budget to iterations that integrate nothing -- reintroducing the grid-dependence the
   budget exists to remove.

2. Empty preset times — a ``PresetTimeCallback`` may legitimately carry a zero-length
   ``times`` array (a collection with no bolus and no sample events; the padded widths are
   collection-wide maxima and are legitimately 0). ``has_presets`` counts CALLBACKS, so
   such a callback used to reach ``_find_next_preset_time`` and ``jnp.argmin`` an empty
   array -- a hard trace-time error.
"""

import diffrax
import jax
import jax.numpy as jnp

from hybrax.train.diffrax_callbacks import (
    PresetTimeCallback,
    diffeqsolve_with_callbacks,
)

jax.config.update("jax_enable_x64", True)

_TERM = diffrax.ODETerm(lambda t, y, args: -0.3 * y)
_Y0 = jnp.ones(2, dtype=jnp.float64)


def _solve(times, max_events):
    return diffeqsolve_with_callbacks(
        _TERM,
        diffrax.Tsit5(),
        t0=0.0,
        t1=10.0,
        dt0=1e-2,
        y0=_Y0,
        callbacks=PresetTimeCallback(times=times, affect_fn=lambda y, t, args, i: y),
        max_events=max_events,
        stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-12),
        max_steps=100_000,
    )


def test_done_lanes_take_zero_steps():
    events = jnp.asarray([2.0, 4.0, 6.0], dtype=jnp.float64)
    tight = _solve(events, 4)  # 3 events + the final leg == exactly enough
    slack = _solve(events, 40)

    n_real = int(tight.event_count) + 1
    assert int(jnp.sum(slack.segment_num_steps[n_real:])) == 0, (
        "iterations after the lane is done must integrate nothing"
    )
    assert int(jnp.sum(tight.segment_num_steps)) == int(
        jnp.sum(slack.segment_num_steps)
    ), "total step count must not depend on the padded scan length"
    assert bool(jnp.array_equal(tight.y_final, slack.y_final)), (
        "and the answer must be bit-identical"
    )


def test_generous_max_events_does_not_eat_the_step_budget():
    """The budget consequence, stated directly: padding the scan must not consume steps
    that a tight ``max_events`` would have had available."""
    events = jnp.asarray([2.0, 4.0, 6.0], dtype=jnp.float64)
    used = int(jnp.sum(_solve(events, 4).segment_num_steps))

    def budgeted(max_events, max_steps):
        return diffeqsolve_with_callbacks(
            _TERM,
            diffrax.Tsit5(),
            t0=0.0,
            t1=10.0,
            dt0=1e-2,
            y0=_Y0,
            callbacks=PresetTimeCallback(
                times=events, affect_fn=lambda y, t, args, i: y
            ),
            max_events=max_events,
            stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-12),
            max_steps=max_steps,
        )

    # A budget that exactly fits must still fit with 10x the scan padding.
    assert bool(jnp.all(jnp.isfinite(budgeted(4, used).y_final)))
    assert bool(jnp.all(jnp.isfinite(budgeted(40, used).y_final))), (
        "padded scan iterations must not spend the trajectory budget"
    )


def test_empty_preset_times_solve_as_if_there_were_no_callback():
    """Zero events is valid data, not user error: the solve runs as a single segment."""
    sol = _solve(jnp.zeros((0,), dtype=jnp.float64), 2)
    assert int(sol.event_count) == 0
    assert bool(jnp.isinf(sol.fail_time))
    expected = _Y0 * jnp.exp(jnp.asarray(-3.0, dtype=jnp.float64))  # -0.3 * 10
    assert jnp.allclose(sol.y_final, expected, rtol=1e-8)
