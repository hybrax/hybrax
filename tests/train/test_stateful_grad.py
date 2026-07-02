from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from stateful_helpers import (
    TrainableH0DefaultStateful,
    build_stateful_wrapper,
    default_stateful_scale_kwargs,
    make_process,
    solve,
)


_FD_EPS = 1e-2
_SCALE_KWARGS = default_stateful_scale_kwargs()


def _loss(module):
    process = make_process(feed_rate=0.1)
    wrapper = build_stateful_wrapper(process, module)
    states = solve(
        wrapper,
        jnp.asarray([0.0, 0.1], dtype=jnp.float32),
        jnp.asarray([1.0, 1.0], dtype=jnp.float32),
    )
    return states[-1, -1]


def _central_diff(module, make_plus_minus):
    plus, minus = make_plus_minus(module)
    return (_loss(plus) - _loss(minus)) / (2 * _FD_EPS)


def _perturb_h0(module):
    return (
        eqx.tree_at(lambda m: m.h0, module, module.h0 + _FD_EPS),
        eqx.tree_at(lambda m: m.h0, module, module.h0 - _FD_EPS),
    )


def _perturb_gru_weight(module):
    weight = module.gru_cell.weight_ih
    return (
        eqx.tree_at(
            lambda m: m.gru_cell.weight_ih,
            module,
            weight.at[0, 0].add(_FD_EPS),
        ),
        eqx.tree_at(
            lambda m: m.gru_cell.weight_ih,
            module,
            weight.at[0, 0].add(-_FD_EPS),
        ),
    )


def test_stateful_gradients_match_finite_difference_for_h0_and_gru_weight():
    module = TrainableH0DefaultStateful(
        key=jax.random.key(0),
        h0=jnp.asarray([0.2], dtype=jnp.float32),
        **_SCALE_KWARGS,
    )

    _value, grad = eqx.filter_value_and_grad(_loss)(module)

    fd_h0 = _central_diff(module, _perturb_h0)
    fd_weight = _central_diff(module, _perturb_gru_weight)

    assert jnp.allclose(grad.h0[0], fd_h0, rtol=1e-4, atol=1e-5)
    assert jnp.allclose(grad.gru_cell.weight_ih[0, 0], fd_weight, rtol=1e-3, atol=1e-5)
