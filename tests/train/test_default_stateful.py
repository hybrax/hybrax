from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from hybrax.train.defaults import DefaultStatefulReactionModule
from hybrax.train.model_api import ReactionInputs, partition_trainable
from stateful_helpers import TrainableH0DefaultStateful, default_stateful_scale_kwargs


_SCALE_KWARGS = default_stateful_scale_kwargs(
    n_controlled_inflows=0, n_controlled_outflows=0
)


def _inputs(h, *, n_modeled_inflows=0, n_modeled_outflows=0):
    return ReactionInputs(
        SCL_modeled_RMCs=jnp.asarray([1.0], dtype=h.dtype),
        SCL_modeled_V=jnp.asarray(1.0, dtype=h.dtype),
        SCL_modeled_Inflows_cumulative=jnp.zeros(n_modeled_inflows, dtype=h.dtype),
        SCL_modeled_Outflows_cumulative=jnp.zeros(n_modeled_outflows, dtype=h.dtype),
        SCL_controlled_Inflows_cumulative=jnp.zeros(0, dtype=h.dtype),
        SCL_controlled_Inflows_rates=jnp.zeros(0, dtype=h.dtype),
        SCL_controlled_Inflows_Cin=jnp.zeros((0, 1), dtype=h.dtype),
        SCL_controlled_Outflows_cumulative=jnp.zeros(0, dtype=h.dtype),
        SCL_controlled_Outflows_rates=jnp.zeros(0, dtype=h.dtype),
        RAW_controlled_Outflows_retention=jnp.zeros((0, 1), dtype=h.dtype),
        SCL_controlled_PVs=jnp.zeros(0, dtype=h.dtype),
        SCL_modeled_Inflows_Cin=jnp.zeros((n_modeled_inflows, 1), dtype=h.dtype),
        RAW_modeled_Outflows_retention=jnp.zeros(
            (n_modeled_outflows, 1), dtype=h.dtype
        ),
        SCL_latent=h,
    )


def _initialization_keys(key):
    key_gru, key_rate, key_inflow, key_outflow = jax.random.split(key, 4)
    _, gru_init_key = jax.random.split(key_gru)
    _, rate_init_key = jax.random.split(key_rate)
    _, inflow_init_key = jax.random.split(key_inflow)
    _, outflow_init_key = jax.random.split(key_outflow)
    return (
        jax.random.split(gru_init_key, 6),
        rate_init_key,
        inflow_init_key,
        outflow_init_key,
    )


def _flow_scale_kwargs():
    return {
        **_SCALE_KWARGS,
        "SCALE_modeled_Inflows_cumulative": jnp.ones(1),
        "SCALE_modeled_Outflows_cumulative": jnp.ones(1),
        "SCALE_modeled_Inflows_rates": jnp.ones(1),
        "SCALE_modeled_Outflows_rates": jnp.ones(1),
        "SCALE_modeled_Inflows_Cin": jnp.ones((1, 1)),
    }


def test_default_stateful_module_uses_gru_cell_as_latent_derivative():
    module = DefaultStatefulReactionModule(
        key=jax.random.key(0), n_latent=2, **_SCALE_KWARGS
    )
    h = jnp.asarray([0.2, -0.3])

    outputs = module(jnp.asarray(0.0), _inputs(h))

    cell_input = jnp.asarray([1.0, 1.0])
    expected_input_size = (
        module.n_modeled_RMCs
        + module.n_modeled_PVs
        + 1
        + module.n_modeled_Inflows
        + module.n_modeled_Outflows
        + 2 * module.n_controlled_Inflows
        + 2 * module.n_controlled_Outflows
        + module.n_controlled_PVs
    )
    assert module.n_latent == 2
    assert module.gru_cell.weight_ih.shape == (3 * module.n_latent, expected_input_size)
    assert jnp.array_equal(module.SCALE_latent.scale, jnp.ones(2))
    assert jnp.allclose(
        outputs.SCL_latent_derivative, module.gru_cell(cell_input, h) - h
    )
    assert outputs.SCL_modeled_BiologicalOde_rates.shape == (1,)
    assert outputs.SCL_modeled_Inflows_rates.shape == (0,)
    assert outputs.SCL_modeled_Outflows_rates.shape == (0,)


def test_default_stateful_mixed_flow_axes_size_the_gru_input():
    module = DefaultStatefulReactionModule(
        key=jax.random.key(19),
        n_latent=2,
        **default_stateful_scale_kwargs(
            n_rmcs=2,
            n_modeled_inflows=1,
            n_modeled_outflows=2,
            n_controlled_inflows=3,
            n_controlled_outflows=4,
            n_controlled_pvs=5,
        ),
    )

    assert module.n_modeled_Inflows == 1
    assert module.n_modeled_Outflows == 2
    assert module.n_controlled_Inflows == 3
    assert module.n_controlled_Outflows == 4
    assert module.gru_cell.weight_ih.shape[1] == 25


def test_default_stateful_gru_initialization_is_per_gate_and_trainable():
    key = jax.random.key(17)
    module = DefaultStatefulReactionModule(key=key, n_latent=3, **_SCALE_KWARGS)
    gru_keys, rate_init_key, _, _ = _initialization_keys(key)
    glorot_init = jax.nn.initializers.glorot_uniform(in_axis=1, out_axis=0)
    orthogonal_init = jax.nn.initializers.orthogonal()

    for i, block in enumerate(jnp.split(module.gru_cell.weight_ih, 3)):
        expected = glorot_init(gru_keys[i], block.shape, block.dtype)
        assert block.dtype == module.gru_cell.weight_ih.dtype
        assert jnp.array_equal(block, expected)

    for i, block in enumerate(jnp.split(module.gru_cell.weight_hh, 3)):
        expected = orthogonal_init(gru_keys[i + 3], block.shape, block.dtype)
        assert jnp.array_equal(block, expected)
        assert jnp.allclose(block @ block.T, jnp.eye(3), atol=1e-6)
        assert jnp.allclose(block.T @ block, jnp.eye(3), atol=1e-6)

    expected_rate = 0.01 * glorot_init(
        rate_init_key,
        module.rate_head.weight.shape,
        module.rate_head.weight.dtype,
    )
    assert jnp.array_equal(module.rate_head.weight, expected_rate)
    assert jnp.array_equal(module.gru_cell.bias, jnp.zeros_like(module.gru_cell.bias))
    assert jnp.array_equal(
        module.gru_cell.bias_n, jnp.zeros_like(module.gru_cell.bias_n)
    )
    assert jnp.array_equal(module.rate_head.bias, jnp.zeros_like(module.rate_head.bias))

    trainable, static = partition_trainable(module)
    assert trainable.gru_cell.bias is not None
    assert trainable.gru_cell.bias_n is not None
    assert static.gru_cell.bias is None
    assert static.gru_cell.bias_n is None


def test_default_stateful_empty_rate_head_skips_glorot(monkeypatch):
    original_glorot = jax.nn.initializers.glorot_uniform

    def guarded_glorot(*args, **kwargs):
        initializer = original_glorot(*args, **kwargs)

        def guarded_initializer(key, shape, dtype):
            assert shape[0] != 0, "empty rate head must skip Glorot"
            return initializer(key, shape, dtype)

        return guarded_initializer

    monkeypatch.setattr(jax.nn.initializers, "glorot_uniform", guarded_glorot)
    module = DefaultStatefulReactionModule(
        key=jax.random.key(3),
        n_latent=3,
        **{
            **_SCALE_KWARGS,
            "SCALE_modeled_BiologicalOde_rates": jnp.zeros(0),
        },
    )

    assert module.rate_head.weight.shape == (0, 4)
    assert module.rate_head.bias.shape == (0,)
    assert module.rate_head.weight.size == 0
    assert module.rate_head.bias.size == 0


def test_default_stateful_flow_heads_are_calibrated_and_signed():
    key = jax.random.key(23)
    module = DefaultStatefulReactionModule(key=key, n_latent=3, **_flow_scale_kwargs())
    _, _, inflow_init_key, outflow_init_key = _initialization_keys(key)
    glorot_init = jax.nn.initializers.glorot_uniform(in_axis=1, out_axis=0)

    for head, init_key in (
        (module.inflow_head, inflow_init_key),
        (module.outflow_head, outflow_init_key),
    ):
        assert head is not None
        expected_weight = 0.01 * glorot_init(
            init_key, head.weight.shape, head.weight.dtype
        )
        expected_bias = jnp.zeros_like(head.bias) + jnp.log(
            jnp.expm1(jnp.asarray(0.01, dtype=head.bias.dtype))
        )
        assert jnp.array_equal(head.weight, expected_weight)
        assert jnp.array_equal(head.bias, expected_bias)

        zero_readout = jnp.zeros(head.in_features, dtype=head.weight.dtype)
        assert jnp.allclose(jax.nn.softplus(head(zero_readout)), 0.01, atol=1e-7)
        direct_sensitivity = jax.grad(lambda bias: jax.nn.softplus(bias)[0])(head.bias)[
            0
        ]
        expected_sensitivity = -jnp.expm1(jnp.asarray(-0.01, dtype=head.bias.dtype))
        assert jnp.allclose(direct_sensitivity, expected_sensitivity, atol=1e-7)

        nonzero_readout = jnp.ones(head.in_features, dtype=head.weight.dtype)
        gradient = eqx.filter_grad(
            lambda candidate: jax.nn.softplus(candidate(nonzero_readout))[0]
        )(head)
        assert jnp.all(jnp.isfinite(gradient.weight))
        assert jnp.all(jnp.isfinite(gradient.bias))
        assert jnp.all(gradient.weight != 0)
        assert jnp.all(gradient.bias != 0)

    inputs = _inputs(jnp.zeros(3), n_modeled_inflows=1, n_modeled_outflows=1)
    outputs = module(jnp.asarray(0.0), inputs)
    assert jnp.all(outputs.SCL_modeled_Inflows_rates > 0)
    assert jnp.all(outputs.SCL_modeled_Outflows_rates < 0)

    modified = eqx.tree_at(
        lambda candidate: candidate.outflow_head.bias,
        module,
        jnp.full_like(module.outflow_head.bias, 10.0),
    )
    modified_outputs = modified(jnp.asarray(0.0), inputs)
    assert jnp.array_equal(
        modified_outputs.SCL_modeled_Inflows_rates,
        outputs.SCL_modeled_Inflows_rates,
    )
    assert not jnp.array_equal(
        modified_outputs.SCL_modeled_Outflows_rates,
        outputs.SCL_modeled_Outflows_rates,
    )


def test_default_stateful_initialization_is_deterministic_per_key():
    first = DefaultStatefulReactionModule(
        key=jax.random.key(29), n_latent=3, **_SCALE_KWARGS
    )
    second = DefaultStatefulReactionModule(
        key=jax.random.key(29), n_latent=3, **_SCALE_KWARGS
    )
    different = DefaultStatefulReactionModule(
        key=jax.random.key(30), n_latent=3, **_SCALE_KWARGS
    )
    first_leaves = jax.tree_util.tree_leaves(first)
    second_leaves = jax.tree_util.tree_leaves(second)
    different_leaves = jax.tree_util.tree_leaves(different)

    for first_leaf, second_leaf in zip(first_leaves, second_leaves):
        if eqx.is_array(first_leaf):
            assert jnp.array_equal(first_leaf, second_leaf)
    assert any(
        not jnp.array_equal(first_leaf, different_leaf)
        for first_leaf, different_leaf in zip(first_leaves, different_leaves)
        if eqx.is_inexact_array(first_leaf)
    )


def test_default_stateful_module_call_is_pure_for_identical_inputs():
    module = DefaultStatefulReactionModule(
        key=jax.random.key(2), n_latent=2, **_SCALE_KWARGS
    )
    h = jnp.asarray([0.2, -0.3])
    inputs = _inputs(h)

    first = module(jnp.asarray(0.0), inputs)
    second = module(jnp.asarray(0.0), inputs)

    assert jnp.array_equal(first.SCL_latent_derivative, second.SCL_latent_derivative)
    assert jnp.array_equal(
        first.SCL_modeled_BiologicalOde_rates,
        second.SCL_modeled_BiologicalOde_rates,
    )
    assert jnp.array_equal(
        first.SCL_modeled_Inflows_rates, second.SCL_modeled_Inflows_rates
    )
    assert jnp.array_equal(
        first.SCL_modeled_Outflows_rates, second.SCL_modeled_Outflows_rates
    )


def test_stateful_module_can_override_trainable_initial_latent():
    module = TrainableH0DefaultStateful(
        key=jax.random.key(1),
        h0=jnp.asarray([0.5, -0.25]),
        **_SCALE_KWARGS,
    )

    trainable, static = eqx.partition(module, eqx.is_array)

    assert jnp.array_equal(module.initial_latent(jnp.ones(2)), module.h0)
    assert trainable.h0 is not None
    assert static.h0 is None
