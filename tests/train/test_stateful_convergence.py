from __future__ import annotations

import jax
import jax.numpy as jnp

from bp_train.defaults import DefaultStatefulReactionModule
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

    # The trajectory (and its physical proxy) CONVERGES to a tolerance-independent
    # limit: each refinement moves the answer by orders of magnitude less than the
    # previous one. A per-step, non-integrated latent would instead track the step
    # count and these gaps would not shrink at all.
    #
    # Stated as a contraction rather than as a fixed threshold on every pair. Output
    # times are no longer forced segment boundaries, so at ``rtol=1e-2`` the solver
    # really does take ~one step across the whole horizon and reads the grid off the
    # interpolant -- and a 1e-2 answer is then genuinely 1e-2 away from the limit,
    # instead of being propped up by nodes the grid used to impose. Production runs at
    # ``rtol=1e-5``, i.e. inside the converged regime below. Measured gaps:
    # 1e-2 -> 1e-4: 1.8e-3;  1e-4 -> 1e-6: 1.2e-5;  1e-6 -> 1e-8: 2.4e-6.
    assert loose_to_mid < 1e-3
    assert mid_to_tight < 1e-4
    assert loss_loose_to_mid < 1e-2
    assert loss_mid_to_tight < 1e-4
    assert mid_to_tight < loose_to_mid / 10.0, "latent must converge, not plateau"
    assert loss_mid_to_tight < loss_loose_to_mid / 10.0, "loss proxy must converge"
