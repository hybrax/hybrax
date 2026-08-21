from __future__ import annotations

import jax
import jax.numpy as jnp

from hybrax.train.defaults import DefaultStatefulReactionModule
from stateful_helpers import (
    build_stateful_wrapper,
    default_stateful_scale_kwargs,
    make_process,
    solve,
)


_SCALE_KWARGS = default_stateful_scale_kwargs()


def _solve(rtol, atol):
    module = DefaultStatefulReactionModule(
        key=jax.random.key(2), n_latent=1, **_SCALE_KWARGS
    )
    wrapper = build_stateful_wrapper(make_process(feed_rate=0.1), module)
    return solve(
        wrapper,
        jnp.linspace(0.0, 1.0, 8),
        jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        rtol=rtol,
        atol=atol,
    )


def test_stateful_latent_trajectory_converges_as_tolerances_tighten():
    loose = _solve(1e-2, 1e-4)
    mid = _solve(1e-4, 1e-6)
    tight = _solve(1e-6, 1e-8)

    loose_to_mid = jnp.linalg.norm(loose[:, -1] - mid[:, -1])
    mid_to_tight = jnp.linalg.norm(mid[:, -1] - tight[:, -1])
    loss_loose_to_mid = jnp.abs(jnp.sum(loose[:, :1]) - jnp.sum(mid[:, :1]))
    loss_mid_to_tight = jnp.abs(jnp.sum(mid[:, :1]) - jnp.sum(tight[:, :1]))

    # Every tolerance refinement leaves the trajectory (and its physical proxy)
    # essentially unchanged: the solve has converged to a tolerance-independent
    # limit. A per-step, non-integrated latent would instead track the step
    # count and these gaps would not shrink to the solver noise floor.
    assert loose_to_mid < 1e-4
    assert mid_to_tight < 1e-4
    assert loss_loose_to_mid < 1e-4
    assert loss_mid_to_tight < 1e-4
