from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from bp_train.model_api import (
    ReactionOutputs,
    UserReactionModule,
    partition_trainable,
)


class _DefaultPartitionModule(UserReactionModule):
    model: eqx.nn.Linear
    non_model_bias: jax.Array

    def __init__(self):
        self.model = eqx.nn.Linear(in_features=2, out_features=1, key=jax.random.key(0))
        self.non_model_bias = jnp.asarray([1.0], dtype=jnp.float32)

    def __call__(self, t, c_species, controls_vector) -> ReactionOutputs:
        del t, c_species, controls_vector
        return ReactionOutputs(
            specific_rates=jnp.asarray([0.0], dtype=jnp.float32),
            modeled_feed_rates=jnp.zeros((0,), dtype=jnp.float32),
        )


class _MissingModelModule(UserReactionModule):
    gain: jax.Array

    def __init__(self):
        self.gain = jnp.asarray([1.0], dtype=jnp.float32)

    def __call__(self, t, c_species, controls_vector) -> ReactionOutputs:
        del t, c_species, controls_vector
        return ReactionOutputs(
            specific_rates=jnp.asarray([0.0], dtype=jnp.float32),
            modeled_feed_rates=jnp.zeros((0,), dtype=jnp.float32),
        )


class _CustomPartitionModule(UserReactionModule):
    model: eqx.nn.Linear
    non_model_bias: jax.Array

    def __init__(self):
        self.model = eqx.nn.Linear(in_features=2, out_features=1, key=jax.random.key(1))
        self.non_model_bias = jnp.asarray([2.0], dtype=jnp.float32)

    def __call__(self, t, c_species, controls_vector) -> ReactionOutputs:
        del t, c_species, controls_vector
        return ReactionOutputs(
            specific_rates=jnp.asarray([0.0], dtype=jnp.float32),
            modeled_feed_rates=jnp.zeros((0,), dtype=jnp.float32),
        )

    def partition_trainable(self):
        return eqx.partition(self, eqx.is_inexact_array)


class _InvalidPartitionModule(_CustomPartitionModule):
    def partition_trainable(self):
        return 1, 2


class _OverlappingPartitionModule(_CustomPartitionModule):
    def partition_trainable(self):
        return self, self


def test_default_partition_trainable_uses_model_subtree_only():
    module = _DefaultPartitionModule()

    trainable, static = partition_trainable(module)

    assert trainable.model.weight is not None
    assert static.model.weight is None
    assert trainable.non_model_bias is None
    assert static.non_model_bias is not None


def test_default_partition_trainable_fails_without_model_attribute():
    module = _MissingModelModule()

    with pytest.raises(ValueError, match="requires a `.model` attribute"):
        partition_trainable(module)


def test_partition_trainable_uses_custom_override_when_present():
    module = _CustomPartitionModule()

    trainable, static = partition_trainable(module)

    assert trainable.model.weight is not None
    assert trainable.non_model_bias is not None
    assert static.model.weight is None
    assert static.non_model_bias is None


def test_partition_trainable_rejects_invalid_partition_structure():
    module = _InvalidPartitionModule()

    with pytest.raises(ValueError, match="must match module structure"):
        partition_trainable(module)


def test_partition_trainable_rejects_overlapping_partitions():
    module = _OverlappingPartitionModule()

    with pytest.raises(ValueError, match="exactly one partition"):
        partition_trainable(module)


def test_user_reaction_module_default_observe_is_identity():
    module = _DefaultPartitionModule()
    y = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

    assert jnp.array_equal(module.observe(y), y)
