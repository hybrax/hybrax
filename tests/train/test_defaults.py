"""Regression tests for :func:`hybrax.train.defaults.default_build_reaction_module`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hybrax.format.dataclasses import BioProcessCollection
from hybrax.format.mechanistic import build_rhs_ode
from hybrax.train.defaults import DefaultReactionModule, default_build_reaction_module
from hybrax.train.model_api import ReactionInputs
from test_harness import _make_collection, _make_multi_process_collection


def _reaction_module(
    *, key, depth=2, width_size=None, n_in=2, n_out=1, n_inflows=1, n_outflows=2
):
    return DefaultReactionModule(
        key=key,
        depth=depth,
        width_size=width_size,
        SCALE_modeled_RMCs=jnp.ones(n_in),
        SCALE_modeled_Inflows_cumulative=jnp.ones(n_inflows),
        SCALE_modeled_Outflows_cumulative=jnp.ones(n_outflows),
        SCALE_modeled_Inflows_rates=jnp.ones(n_inflows),
        SCALE_modeled_Outflows_rates=jnp.ones(n_outflows),
        SCALE_modeled_BiologicalOde_rates=jnp.ones(n_out),
    )


def test_shallow_default_reaction_module_uses_tanh_and_glorot():
    key = jax.random.key(3)
    module = _reaction_module(key=key)
    _, init_key = jax.random.split(key)
    layer_keys = jax.random.split(init_key, 3)
    glorot_init = jax.nn.initializers.glorot_uniform(in_axis=1, out_axis=0)

    assert module.model.depth == 2
    assert module.model.width_size == 8
    assert module.model.activation is jax.nn.tanh
    for layer, layer_key in zip(module.model.layers[:-1], layer_keys[:-1]):
        expected = glorot_init(layer_key, layer.weight.shape, layer.weight.dtype)
        assert jnp.array_equal(layer.weight, expected)
        assert jnp.count_nonzero(layer.bias) == 0

    output = module.model.layers[-1]
    expected_output = 0.01 * glorot_init(
        layer_keys[-1], output.weight.shape, output.weight.dtype
    )
    assert jnp.array_equal(output.weight, expected_output)
    assert jnp.count_nonzero(output.bias) == 0


def test_stateless_default_emits_zero_rates_on_distinct_flow_axes():
    module = _reaction_module(key=jax.random.key(31))
    inputs = ReactionInputs(
        SCL_modeled_RMCs=jnp.ones(2),
        SCL_modeled_V=jnp.asarray(1.0),
        SCL_modeled_Inflows_cumulative=jnp.ones(1),
        SCL_modeled_Outflows_cumulative=jnp.ones(2),
        SCL_controlled_Inflows_cumulative=jnp.zeros(0),
        SCL_controlled_Inflows_rates=jnp.zeros(0),
        SCL_controlled_Inflows_Cin=jnp.zeros((0, 2)),
        SCL_controlled_Outflows_cumulative=jnp.zeros(0),
        SCL_controlled_Outflows_rates=jnp.zeros(0),
        RAW_controlled_Outflows_retention=jnp.zeros((0, 2)),
        SCL_controlled_PVs=jnp.zeros(0),
        SCL_modeled_Inflows_Cin=jnp.ones((1, 2)),
        RAW_modeled_Outflows_retention=jnp.ones((2, 2)),
    )

    outputs = module(jnp.asarray(0.0), inputs)

    assert jnp.array_equal(outputs.SCL_modeled_Inflows_rates, jnp.zeros(1))
    assert jnp.array_equal(outputs.SCL_modeled_Outflows_rates, jnp.zeros(2))


def test_deep_default_reaction_module_uses_requested_width_silu_and_he():
    key = jax.random.key(4)
    module = _reaction_module(key=key, depth=5, width_size=13)
    _, init_key = jax.random.split(key)
    layer_keys = jax.random.split(init_key, 6)
    he_init = jax.nn.initializers.he_uniform(in_axis=1, out_axis=0)

    assert module.model.depth == 5
    assert module.model.width_size == 13
    assert module.model.activation is jax.nn.silu
    for layer, layer_key in zip(module.model.layers[:-1], layer_keys[:-1]):
        expected = he_init(layer_key, layer.weight.shape, layer.weight.dtype)
        assert jnp.array_equal(layer.weight, expected)
        assert jnp.count_nonzero(layer.bias) == 0

    output = module.model.layers[-1]
    glorot_init = jax.nn.initializers.glorot_uniform(in_axis=1, out_axis=0)
    expected_output = 0.01 * glorot_init(
        layer_keys[-1], output.weight.shape, output.weight.dtype
    )
    assert jnp.array_equal(output.weight, expected_output)
    assert jnp.count_nonzero(output.bias) == 0


def test_default_reaction_module_pins_depth_threshold_and_computed_width():
    shallow = _reaction_module(key=jax.random.key(5), depth=3, n_in=5, n_out=2)
    deep = _reaction_module(key=jax.random.key(6), depth=4, n_in=5, n_out=2)

    assert shallow.model.width_size == 10
    assert shallow.model.activation is jax.nn.tanh
    assert deep.model.width_size == 10
    assert deep.model.activation is jax.nn.silu


def test_default_reaction_module_allows_empty_rate_head():
    module = _reaction_module(key=jax.random.key(9), n_out=0)

    assert module.model.layers[-1].weight.shape == (0, 8)
    assert module.model.layers[-1].bias.shape == (0,)


def test_zero_depth_module_uses_small_glorot_linear_layer():
    key = jax.random.key(7)
    module = _reaction_module(key=key, depth=0, width_size=3)
    _, init_key = jax.random.split(key)
    (layer_key,) = jax.random.split(init_key, 1)
    layer = module.model.layers[0]
    expected = 0.01 * jax.nn.initializers.glorot_uniform(in_axis=1, out_axis=0)(
        layer_key, layer.weight.shape, layer.weight.dtype
    )

    assert module.model.depth == 0
    assert jnp.array_equal(layer.weight, expected)
    assert jnp.count_nonzero(layer.bias) == 0


@pytest.mark.parametrize(
    ("depth", "width_size", "message"),
    [(-1, None, "depth must be non-negative"), (2, 0, "width_size must be positive")],
)
def test_default_reaction_module_rejects_invalid_architecture(
    depth, width_size, message
):
    with pytest.raises(ValueError, match=message):
        _reaction_module(key=jax.random.key(8), depth=depth, width_size=width_size)


def test_default_reaction_module_scale_follows_modeled_state_not_targets():
    """``SCALE_modeled_RMCs`` is sized by the modeled RMC state slice, not by the
    targets.

    Under ``target_source="combined"`` / ``"process_variables"`` the measured-target
    set (reactor components *plus* modeled PVs) is a different length than the
    reactor-component state slice the module scales. Sizing the fallback RMC scale
    from ``len(target_names)`` then produces a wrong-length axis that the wrapper
    rejects. This guards that regression: the RMC scale must always match
    ``len(rhs_ode.name_modeled_RMCs)``.
    """
    collection = _make_collection()
    rhs_ode = build_rhs_ode(collection.processes["p1"])
    n_rmc = len(rhs_ode.name_modeled_RMCs)

    # A measured-target set LONGER than the RMC state slice, mimicking
    # target_source="combined" (reactor components + process variables).
    target_names = ["biomass", "viability", "extra_pv"]
    assert (
        len(target_names) != n_rmc
    )  # precondition: counts differ, else nothing to catch

    module = default_build_reaction_module(
        target_names=target_names,
        process_names=list(collection.processes),
        config=None,
        seed=0,
        training_parent_collection=collection,
    )

    assert module.SCALE_modeled_RMCs.shape[0] == n_rmc
    assert module.n_modeled_RMCs == n_rmc


def test_default_reaction_module_scale_independent_of_target_count():
    """The RMC scale length is invariant to how many targets are passed."""
    collection = _make_collection()
    n_rmc = len(build_rhs_ode(collection.processes["p1"]).name_modeled_RMCs)

    shapes = {
        default_build_reaction_module(
            target_names=targets,
            process_names=list(collection.processes),
            config=None,
            seed=0,
            training_parent_collection=collection,
        ).SCALE_modeled_RMCs.shape[0]
        for targets in (
            ["biomass"],
            ["biomass", "viability"],
            ["biomass", "viability", "product", "lactate"],
        )
    }

    assert shapes == {n_rmc}


def test_default_reaction_module_rejects_disagreeing_training_parents():
    """The default builder speaks for all parents via the first one's RhsOde.

    That is only sound if every parent declares an equivalent BiologicalOde, so
    disagreement must be rejected rather than silently resolved by ordering.
    """
    collection = _make_multi_process_collection(2)
    p2 = collection.processes["p2"]
    p2.biological_ode.rates = {"q_other": (None, None)}
    p2.biological_ode.derivatives["biomass"] = "q_other * biomass"

    with pytest.raises(ValueError, match="biological_ode_equivalence"):
        default_build_reaction_module(
            target_names=["biomass"],
            process_names=list(collection.processes),
            config=None,
            seed=0,
            training_parent_collection=collection,
        )


def test_default_reaction_module_requires_a_training_parent():
    with pytest.raises(ValueError, match="requires a training parent"):
        default_build_reaction_module(
            target_names=["biomass"],
            process_names=["p1"],
            config=None,
            seed=0,
            training_parent_collection=BioProcessCollection(processes={}),
        )
