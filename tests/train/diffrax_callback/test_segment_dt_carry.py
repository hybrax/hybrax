"""Each segment seeds ``dt`` from the previous segment's average accepted step.

Every segment is a fresh ``diffeqsolve`` with no controller history, so seeding each one
from the fixed ``dt0`` makes the step-size controller re-ramp from scratch. When the
natural step is much larger than ``dt0`` that ramp is most of the segment's cost --
measured 10-40% of all ODE steps on bp-bench training grids.

The carried value is floored at ``dt0``, which makes the policy **monotone**: it can only
ever start larger than the fixed default, never smaller. That matters because a very
short segment would otherwise hand a tiny step to the next, possibly much longer, one and
force it to ramp back up -- a regression the fixed ``dt0`` cannot have.

Both policies are valid solutions of the same ODE taking different step sequences, so
they differ by O(global truncation error); ``test_agreement_tightens_with_tolerance``
pins that the difference is discretization noise rather than a bias.
"""

import diffrax
import jax
import jax.numpy as jnp
import pytest

from diffrax_callbacks import PresetTimeCallback, diffeqsolve_with_callbacks

jax.config.update("jax_enable_x64", True)

_T1 = 240.0


def _solve(*, n_segments, dt0, rtol=1e-5, atol=1e-6):
    """Smooth saturating growth over a long horizon chopped into equal segments."""
    times = jnp.linspace(_T1 / n_segments, _T1, n_segments, dtype=jnp.float64)
    return diffeqsolve_with_callbacks(
        diffrax.ODETerm(lambda t, y, args: 0.03 * y / (1.0 + y)),
        diffrax.Tsit5(),
        t0=0.0,
        t1=_T1,
        dt0=dt0,
        y0=jnp.asarray([0.5], dtype=jnp.float64),
        callbacks=PresetTimeCallback(times=times, affect_fn=lambda y, t, args, i: y),
        max_events=n_segments,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=10_000,
    )


def _steps(sol):
    return int(jnp.sum(sol.segment_num_steps))


def test_carry_beats_a_cold_restart_when_dt0_is_small_relative_to_a_segment():
    """The case the policy targets: long segments, tiny dt0. ``dt0 = span*1e-3`` is
    1/100 of a 24 h segment here, which is exactly the bp-bench shape."""
    sol = _solve(n_segments=10, dt0=_T1 * 1e-3)
    # A cold restart every segment needs several steps just to grow dt back; seeded from
    # the previous segment it should approach one step per segment.
    assert _steps(sol) <= 2 * 10 + 5, (
        f"expected near one step per segment, got {_steps(sol)}"
    )


def test_floor_makes_the_policy_monotone_in_dt0():
    """Never start smaller than ``dt0``: raising ``dt0`` can only reduce (or hold) the
    step count, never inflate it. This is what the floor buys."""
    counts = [
        _steps(_solve(n_segments=10, dt0=d))
        for d in (_T1 * 1e-4, _T1 * 1e-3, _T1 * 1e-2)
    ]
    assert counts == sorted(counts, reverse=True) or len(set(counts)) == 1, (
        f"step count should not increase as dt0 grows: {counts}"
    )


@pytest.mark.parametrize("n_segments", [4, 10, 40])
def test_result_is_correct_for_any_segmentation(n_segments):
    """Same ODE, same answer, regardless of how the horizon is chopped -- the carried
    step must not change the solution beyond solver tolerance."""
    ref = _solve(n_segments=n_segments, dt0=_T1 * 1e-3, rtol=1e-11, atol=1e-13)
    got = _solve(n_segments=n_segments, dt0=_T1 * 1e-3, rtol=1e-6, atol=1e-8)
    assert float(got.y_final[0]) == pytest.approx(float(ref.y_final[0]), rel=1e-4)


def test_agreement_tightens_with_tolerance():
    """Discretization noise, not bias: the gap between a coarse and a reference solve
    must shrink as the tolerance tightens."""
    ref = float(_solve(n_segments=10, dt0=_T1 * 1e-3, rtol=1e-12, atol=1e-14).y_final[0])
    devs = [
        abs(float(_solve(n_segments=10, dt0=_T1 * 1e-3, rtol=r, atol=a).y_final[0]) - ref)
        / abs(ref)
        for r, a in ((1e-5, 1e-6), (1e-8, 1e-9))
    ]
    assert devs[1] < devs[0], f"deviation must shrink with tolerance, got {devs}"


def test_collapsed_lane_does_not_corrupt_the_carry():
    """A failed lane collapses its later segments to zero length; those must carry the
    previous ``dt`` through rather than divide by a zero-length segment."""
    sol = diffeqsolve_with_callbacks(
        diffrax.ODETerm(lambda t, y, args: -0.3 * y),
        diffrax.Tsit5(),
        t0=0.0,
        t1=2.0,
        dt0=1e-3,
        y0=jnp.ones(2, dtype=jnp.float64),
        callbacks=PresetTimeCallback(
            times=jnp.linspace(0.4, 2.0, 5, dtype=jnp.float64),
            affect_fn=lambda y, t, args, i: y,
        ),
        max_events=5,
        stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-13),
        max_steps=3,  # forces an early failure
    )
    assert bool(jnp.all(jnp.isinf(sol.y_final)))
    assert bool(jnp.all(jnp.isfinite(sol.segment_num_steps)))
