"""``max_steps``: a trajectory-level step budget, independent of segmentation.

A per-segment bound alone would not be a budget: each segment gets a fresh allowance, so
the effective ceiling is ``cap * n_segments``. A caller asking for N steps could silently
get orders of magnitude more -- ``solver.max_steps = 8096`` on a 174-segment process
really allowed ~89,000 -- and ``fail_time`` moved when the horizon was chopped more
finely (which a dense loss or export grid used to do).

``max_steps`` sums the per-segment counts in the scan carry and terminates the lane
exactly like a segment bail, so the budget is a property of the trajectory rather than of
the output grid. It is also the bound handed to each inner solve, so there is exactly one
knob and it can never be undercut by a second, tighter one.
"""

import diffrax
import jax
import jax.numpy as jnp

from diffrax_callbacks import PresetTimeCallback, diffeqsolve_with_callbacks

jax.config.update("jax_enable_x64", True)


def _solve(*, n_segments, max_steps=100_000):
    """Smooth decay over [0, 2] chopped into ``n_segments`` identity-affect segments.

    The dynamics are identical regardless of ``n_segments``; only the segmentation of
    the same horizon changes, which is exactly the variable a trajectory budget must be
    invariant to.
    """
    times = jnp.linspace(2.0 / n_segments, 2.0, n_segments, dtype=jnp.float64)
    return diffeqsolve_with_callbacks(
        diffrax.ODETerm(lambda t, y, args: -0.3 * y),
        diffrax.Tsit5(),
        t0=0.0,
        t1=2.0,
        dt0=1e-3,
        y0=jnp.ones(2, dtype=jnp.float64),
        callbacks=PresetTimeCallback(times=times, affect_fn=lambda y, t, args, i: y),
        max_events=n_segments,
        stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-13),
        max_steps=max_steps,
    )


def test_generous_budget_does_not_trigger():
    sol = _solve(n_segments=8, max_steps=100_000)
    assert bool(jnp.all(jnp.isfinite(sol.y_final)))
    assert float(sol.fail_time) == float("inf")


def test_exhausted_budget_terminates_the_lane_like_a_segment_bail():
    """Blowing the budget poisons the state and records a finite ``fail_time`` -- the
    same contract a ``max_steps``/``dt_min`` bail already has, so every downstream
    consumer (loss masking, export sentinel) works unchanged."""
    generous = _solve(n_segments=8, max_steps=100_000)
    used = int(jnp.sum(generous.segment_num_steps))
    assert used > 4, "need a solve that takes real steps for this test to mean anything"

    sol = _solve(n_segments=8, max_steps=used // 2)
    assert bool(jnp.all(jnp.isinf(sol.y_final)))
    assert float(sol.fail_time) < float("inf")


def test_budget_is_independent_of_how_the_horizon_is_chopped():
    """THE point of the change.

    Same dynamics, same total budget, different segment counts. Under a per-segment cap
    alone the coarse grid would bail and the fine grid would sail through on
    ``cap * n_segments``; with a trajectory budget both must reach the same verdict.
    """
    budget = int(jnp.sum(_solve(n_segments=4, max_steps=100_000).segment_num_steps))

    coarse = _solve(n_segments=4, max_steps=budget // 2)
    fine = _solve(n_segments=32, max_steps=budget // 2)
    assert bool(jnp.all(jnp.isinf(coarse.y_final)))
    assert bool(jnp.all(jnp.isinf(fine.y_final))), (
        "a finer output grid must not buy extra step budget"
    )


def test_max_steps_also_bounds_the_inner_solve():
    """There is exactly ONE knob, and it bounds a single segment too. A budget below what
    one segment needs must bail rather than let that segment run unbounded -- this is what
    makes the trajectory bound safe to use as the pmap latency guard now that the separate
    per-segment cap is gone."""
    sol = _solve(n_segments=1, max_steps=4)
    assert bool(jnp.all(jnp.isinf(sol.y_final)))
    assert float(sol.fail_time) == 0.0, "the single segment bails from t0"
