from __future__ import annotations

import jax.numpy as jnp

from stateful_helpers import (
    ZeroLatentDerivativeModule,
    build_stateful_wrapper,
    make_process,
    solve,
)


def test_latent_state_survives_sample_and_bolus_jump_unchanged():
    process = make_process(jump=True)
    h0 = jnp.asarray([3.0, -2.0], dtype=jnp.float32)
    wrapper = build_stateful_wrapper(process, ZeroLatentDerivativeModule(h0))

    states = solve(
        wrapper,
        jnp.asarray([0.0, 1.0, 1.5], dtype=jnp.float32),
        jnp.asarray([1.0, 1.0], dtype=jnp.float32),
    )

    assert states.shape == (3, 4)
    assert jnp.allclose(states[:, -2:], h0[None, :], atol=1e-6)
