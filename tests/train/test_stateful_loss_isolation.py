from __future__ import annotations

import jax.numpy as jnp

from stateful_helpers import (
    ZeroLatentDerivativeModule,
    build_stateful_wrapper,
    make_process,
    solve,
)


def test_target_state_indices_exclude_latent_and_decoupled_h0_leaves_loss_unchanged():
    process = make_process(feed_rate=0.1)
    y0 = jnp.asarray([1.0, 1.0])
    t_eval = jnp.asarray([0.0, 0.5])

    wrapper_a = build_stateful_wrapper(
        process,
        ZeroLatentDerivativeModule(jnp.asarray([0.0])),
    )
    wrapper_b = build_stateful_wrapper(
        process,
        ZeroLatentDerivativeModule(jnp.asarray([10.0])),
    )

    states_a = solve(wrapper_a, t_eval, y0)
    states_b = solve(wrapper_b, t_eval, y0)

    n_phys = 2
    assert jnp.all(wrapper_a.target_state_indices < n_phys)
    assert jnp.allclose(states_a[:, :n_phys], states_b[:, :n_phys], atol=1e-6)
    assert not jnp.allclose(states_a[:, n_phys:], states_b[:, n_phys:])
