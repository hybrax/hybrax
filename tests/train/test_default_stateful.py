from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from bp_train.defaults import DefaultStatefulReactionModule
from bp_train.model_api import ReactionInputs
from stateful_helpers import TrainableH0DefaultStateful, default_stateful_scale_kwargs


_SCALE_KWARGS = default_stateful_scale_kwargs(n_controlled_fvcs=0)


def _inputs(h):
    return ReactionInputs(
        SCL_modeled_RMCs=jnp.asarray([1.0], dtype=h.dtype),
        SCL_modeled_V=jnp.asarray(1.0, dtype=h.dtype),
        SCL_modeled_FVCs_cumulative=jnp.zeros(0, dtype=h.dtype),
        SCL_controlled_FVCs_cumulative=jnp.zeros(0, dtype=h.dtype),
        SCL_controlled_FVCs_rates=jnp.zeros(0, dtype=h.dtype),
        SCL_controlled_FVCs_Cin=jnp.zeros((0, 1), dtype=h.dtype),
        SCL_controlled_PVs=jnp.zeros(0, dtype=h.dtype),
        SCL_modeled_FVCs_Cin=jnp.zeros((0, 1), dtype=h.dtype),
        SCL_latent=h,
    )


def test_default_stateful_module_uses_gru_cell_as_latent_derivative():
    module = DefaultStatefulReactionModule(
        key=jax.random.key(0), n_latent=2, **_SCALE_KWARGS
    )
    h = jnp.asarray([0.2, -0.3], dtype=jnp.float32)

    outputs = module(jnp.asarray(0.0, dtype=jnp.float32), _inputs(h))

    cell_input = jnp.concatenate([jnp.asarray([1.0, 1.0], dtype=jnp.float32), h])
    assert module.n_latent == 2
    assert jnp.array_equal(module.SCALE_latent, jnp.ones(2, dtype=jnp.float32))
    assert jnp.allclose(
        outputs.SCL_latent_derivative, module.gru_cell(cell_input, h) - h
    )
    assert outputs.SCL_modeled_BiologicalOde_rates.shape == (1,)
    assert outputs.SCL_modeled_FVCs_rates.shape == (0,)


def test_default_stateful_module_call_is_pure_for_identical_inputs():
    module = DefaultStatefulReactionModule(
        key=jax.random.key(2), n_latent=2, **_SCALE_KWARGS
    )
    h = jnp.asarray([0.2, -0.3], dtype=jnp.float32)
    inputs = _inputs(h)

    first = module(jnp.asarray(0.0, dtype=jnp.float32), inputs)
    second = module(jnp.asarray(0.0, dtype=jnp.float32), inputs)

    assert jnp.array_equal(first.SCL_latent_derivative, second.SCL_latent_derivative)
    assert jnp.array_equal(
        first.SCL_modeled_BiologicalOde_rates,
        second.SCL_modeled_BiologicalOde_rates,
    )
    assert jnp.array_equal(first.SCL_modeled_FVCs_rates, second.SCL_modeled_FVCs_rates)


def test_stateful_module_can_override_trainable_initial_latent():
    module = TrainableH0DefaultStateful(
        key=jax.random.key(1),
        h0=jnp.asarray([0.5, -0.25], dtype=jnp.float32),
        **_SCALE_KWARGS,
    )

    trainable, static = eqx.partition(module, eqx.is_array)

    assert jnp.array_equal(module.initial_latent(jnp.ones(2)), module.h0)
    assert trainable.h0 is not None
    assert static.h0 is None
