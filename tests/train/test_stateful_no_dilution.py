from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from bp_train.model_api import AffineScaler

from stateful_helpers import (
    TrainableH0Module,
    ZeroLatentDerivativeModule,
    build_stateful_wrapper,
    make_process,
    solve,
)


def test_continuous_feed_dilutes_physical_state_not_latent_state():
    process = make_process(feed_rate=0.2)
    h0 = jnp.asarray([3.0, -2.0])
    wrapper = build_stateful_wrapper(process, ZeroLatentDerivativeModule(h0))
    t_eval = jnp.linspace(0.0, 2.0, 5)

    y0 = jnp.asarray([1.0, 1.0])
    states = solve(wrapper, t_eval, y0)

    assert states.shape == (5, 4)
    assert states.dtype == y0.dtype
    assert states[-1, 0] < states[0, 0]
    assert states[-1, 1] > states[0, 1]
    assert jnp.allclose(states[:, -2:], h0[None, :], atol=1e-6)


def test_affine_latent_offset_keeps_zero_derivative_stationary():
    # Latent derivative conversion must use unscale_derivative (offset-free),
    # not value unscale (which would turn zero into the offset and cause drift).
    process = make_process()
    h0 = jnp.asarray([10.0])
    wrapper = build_stateful_wrapper(process, ZeroLatentDerivativeModule(h0))
    wrapper = eqx.tree_at(
        lambda w: w.reaction_module.SCALE_latent,
        wrapper,
        AffineScaler(
            jnp.asarray([2.0]),
            jnp.asarray([10.0]),
        ),
    )
    t_eval = jnp.linspace(0.0, 2.0, 5)
    y0 = jnp.asarray([1.0, 1.0])
    states = solve(wrapper, t_eval, y0)
    assert jnp.allclose(states[:, -1], h0[0], rtol=0.0, atol=1e-6)


def test_stateful_solve_runs_and_differentiates_through_latent_state():
    process = make_process(feed_rate=0.1)
    wrapper = build_stateful_wrapper(
        process,
        TrainableH0Module(jnp.asarray([0.5])),
    )
    t_eval = jnp.asarray([0.0, 0.5])
    y0 = jnp.asarray([1.0, 1.0])

    def final_latent_sum(module):
        local_wrapper = eqx.tree_at(lambda w: w.reaction_module, wrapper, module)
        states = solve(local_wrapper, t_eval, y0)
        return states[-1, -1]

    value, grad = eqx.filter_value_and_grad(final_latent_sum)(wrapper.reaction_module)

    assert jnp.isfinite(value)
    assert grad.h0 is not None
    assert jnp.all(jnp.isfinite(grad.h0))
    assert grad.h0.shape == (1,)
