from __future__ import annotations

import re

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from bp_train.inspect import format_trainable_structure
from bp_train.model_api import (
    TRAINABLE_METADATA_KEY,
    ReactionInputs,
    ReactionOutputs,
    UserReactionModule,
    frozen_field,
    partition_trainable,
    trainable_field,
)


# ---------------------------------------------------------------------------
# Metadata-only partition: trainability is declared solely via field tags.
# ---------------------------------------------------------------------------


class _TagPartitionModule(UserReactionModule):
    model: eqx.nn.Linear = trainable_field()
    non_model_bias: jax.Array = frozen_field()

    def __init__(self):
        super().__init__()
        self.model = eqx.nn.Linear(in_features=2, out_features=1, key=jax.random.key(1))
        self.non_model_bias = jnp.asarray([2.0], dtype=jnp.float32)

    def __call__(self, t, inputs) -> ReactionOutputs:
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray([0.0], dtype=jnp.float32),
            SCL_modeled_FVCs_rates=jnp.zeros((0,), dtype=jnp.float32),
        )


def test_partition_trainable_uses_field_tags():
    module = _TagPartitionModule()

    trainable, static = partition_trainable(module)

    # trainable_field() leaf is trained; frozen_field() leaf is not.
    assert trainable.model.weight is not None
    assert static.model.weight is None
    assert trainable.non_model_bias is None
    assert static.non_model_bias is not None


# ---------------------------------------------------------------------------
# Metadata-based partition: helpers
# ---------------------------------------------------------------------------


def test_trainable_field_sets_metadata():
    fld = trainable_field()
    assert fld.metadata[TRAINABLE_METADATA_KEY] is True


def test_frozen_field_sets_metadata():
    fld = frozen_field()
    assert fld.metadata[TRAINABLE_METADATA_KEY] is False


# ---------------------------------------------------------------------------
# Metadata-based partition: leaf cases
# ---------------------------------------------------------------------------


class _UntaggedArrayModule(UserReactionModule):
    raw: jax.Array

    def __init__(self):
        super().__init__()
        self.raw = jnp.asarray([1.0], dtype=jnp.float32)

    def __call__(self, t, inputs) -> ReactionOutputs:
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray([0.0], dtype=jnp.float32),
            SCL_modeled_VC_rates=jnp.zeros((0,), dtype=jnp.float32),
        )


class _MixedTagsModule(UserReactionModule):
    weights: jax.Array = trainable_field()
    bias_frozen: jax.Array = frozen_field()

    def __init__(self):
        super().__init__()
        self.weights = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
        self.bias_frozen = jnp.asarray([10.0], dtype=jnp.float32)

    def __call__(self, t, inputs) -> ReactionOutputs:
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray([0.0], dtype=jnp.float32),
            SCL_modeled_VC_rates=jnp.zeros((0,), dtype=jnp.float32),
        )


def test_untagged_array_defaults_to_frozen():
    module = _UntaggedArrayModule()

    trainable, static = partition_trainable(module)

    assert trainable.raw is None
    assert static.raw is not None


def test_trainable_field_marks_array_leaves_trainable():
    module = _MixedTagsModule()

    trainable, static = partition_trainable(module)

    assert trainable.weights is not None
    assert static.weights is None
    assert trainable.bias_frozen is None
    assert static.bias_frozen is not None


# ---------------------------------------------------------------------------
# Metadata-based partition: nested inheritance (symmetric override rule)
# ---------------------------------------------------------------------------


class _Inner(eqx.Module):
    array1: jax.Array = trainable_field()
    array2: jax.Array = frozen_field()

    def __init__(self):
        self.array1 = jnp.asarray([1.0], dtype=jnp.float32)
        self.array2 = jnp.asarray([2.0], dtype=jnp.float32)


class _OuterFrozenParent(UserReactionModule):
    sub: _Inner = frozen_field()

    def __init__(self):
        super().__init__()
        self.sub = _Inner()

    def __call__(self, t, inputs) -> ReactionOutputs:
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray([0.0], dtype=jnp.float32),
            SCL_modeled_VC_rates=jnp.zeros((0,), dtype=jnp.float32),
        )


class _OuterTrainableParent(UserReactionModule):
    sub: _Inner = trainable_field()

    def __init__(self):
        super().__init__()
        self.sub = _Inner()

    def __call__(self, t, inputs) -> ReactionOutputs:
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray([0.0], dtype=jnp.float32),
            SCL_modeled_VC_rates=jnp.zeros((0,), dtype=jnp.float32),
        )


class _OuterUntaggedParent(UserReactionModule):
    sub: _Inner

    def __init__(self):
        super().__init__()
        self.sub = _Inner()

    def __call__(self, t, inputs) -> ReactionOutputs:
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray([0.0], dtype=jnp.float32),
            SCL_modeled_VC_rates=jnp.zeros((0,), dtype=jnp.float32),
        )


def test_frozen_field_overrides_trainable_descendants():
    module = _OuterFrozenParent()

    trainable, static = partition_trainable(module)

    assert trainable.sub.array1 is None
    assert trainable.sub.array2 is None
    assert static.sub.array1 is not None
    assert static.sub.array2 is not None


def test_trainable_field_overrides_frozen_descendants():
    module = _OuterTrainableParent()

    trainable, static = partition_trainable(module)

    # Both inner arrays are trainable because the parent override dominates.
    assert trainable.sub.array1 is not None
    assert trainable.sub.array2 is not None
    assert static.sub.array1 is None
    assert static.sub.array2 is None


def test_untagged_parent_delegates_to_child_tags():
    module = _OuterUntaggedParent()

    trainable, static = partition_trainable(module)

    # Untagged parent → respect each child field's own tag.
    assert trainable.sub.array1 is not None  # child trainable_field
    assert static.sub.array1 is None
    assert trainable.sub.array2 is None  # child frozen_field
    assert static.sub.array2 is not None


# ---------------------------------------------------------------------------
# Optimizer correctness
# ---------------------------------------------------------------------------


def test_optimizer_only_updates_trainable_leaves():
    """The user-requested guard: a real optimizer step must update only
    `trainable_field()` leaves and leave `frozen_field()` leaves byte-identical."""
    module = _MixedTagsModule()
    weights_before = module.weights
    bias_before = module.bias_frozen

    trainable_params, static_params = partition_trainable(module)

    optimizer = optax.sgd(0.1)
    opt_state = optimizer.init(trainable_params)

    grads = jtu.tree_map(
        lambda x: jnp.ones_like(x) if eqx.is_inexact_array(x) else x,
        trainable_params,
    )
    updates, _ = optimizer.update(grads, opt_state, trainable_params)
    trainable_updated = eqx.apply_updates(trainable_params, updates)
    module_after = eqx.combine(trainable_updated, static_params)

    assert jnp.allclose(module_after.weights, weights_before - 0.1)
    assert jnp.array_equal(module_after.bias_frozen, bias_before)


# ---------------------------------------------------------------------------
# Structure print-out
# ---------------------------------------------------------------------------


def test_format_trainable_structure_renders_status():
    module = _MixedTagsModule()
    text = format_trainable_structure(module)

    assert "weights" in text
    assert "bias_frozen" in text
    assert "trainable" in text
    assert "frozen" in text
    # Two-state status only; no "(inherited)" variant.
    assert "inherited" not in text


def test_format_trainable_structure_auto_widths():
    class _LongName(UserReactionModule):
        a_very_long_field_name_for_testing: jax.Array = trainable_field()

        def __init__(self):
            super().__init__()
            self.a_very_long_field_name_for_testing = jnp.asarray(
                [1.0], dtype=jnp.float32
            )

        def __call__(self, t, c_species, controls_vector) -> ReactionOutputs:
            del t, c_species, controls_vector
            return ReactionOutputs(
                specific_rates=jnp.asarray([0.0], dtype=jnp.float32),
                modeled_feed_rates=jnp.zeros((0,), dtype=jnp.float32),
            )

    module = _LongName()
    text = format_trainable_structure(module)

    assert "a_very_long_field_name_for_testing" in text
    # Header lines should be at least as wide as the longest name plus borders.
    lines = text.splitlines()
    assert all(len(line) == len(lines[0]) for line in lines if line.startswith("|"))


def test_format_trainable_structure_color_off_by_default():
    module = _MixedTagsModule()
    text = format_trainable_structure(module)

    assert "\x1b" not in text


def test_format_trainable_structure_color_on_wraps_trainable_rows():
    module = _MixedTagsModule()
    text = format_trainable_structure(module, color=True)

    ansi_strip = re.compile(r"\x1b\[[0-9;]*m")
    rows = [
        line for line in text.splitlines() if ansi_strip.sub("", line).startswith("|")
    ]

    has_red_trainable = any(
        "\x1b[31m" in row and "trainable" in ansi_strip.sub("", row) for row in rows
    )
    no_red_on_frozen = all(
        "\x1b[31m" not in row
        for row in rows
        if "frozen" in ansi_strip.sub("", row)
        and "trainable" not in ansi_strip.sub("", row)
    )
    assert has_red_trainable
    assert no_red_on_frozen

    # Stripping ANSI from colored rows yields the uncolored render verbatim.
    plain_text = format_trainable_structure(module, color=False)
    plain_rows = [line for line in plain_text.splitlines() if line.startswith("|")]
    stripped = [ansi_strip.sub("", r) for r in rows]
    assert stripped == plain_rows


def test_user_reaction_module_default_observe_is_identity():
    module = _UntaggedArrayModule()
    y = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

    assert jnp.array_equal(module.observe(y), y)


# ---------------------------------------------------------------------------
# Latent-state contract
# ---------------------------------------------------------------------------


def test_reaction_inputs_default_to_zero_width_latent():
    inputs = ReactionInputs(
        SCL_modeled_RMCs=jnp.ones(2, dtype=jnp.float32),
        SCL_modeled_V=jnp.asarray(1.0, dtype=jnp.float32),
        SCL_modeled_FVCs_cumulative=jnp.zeros(1, dtype=jnp.float32),
        SCL_controlled_FVCs_cumulative=jnp.zeros(0, dtype=jnp.float32),
        SCL_controlled_FVCs_rates=jnp.zeros(0, dtype=jnp.float32),
        SCL_controlled_FVCs_Cin=jnp.zeros((0, 2), dtype=jnp.float32),
        SCL_controlled_PVs=jnp.zeros(0, dtype=jnp.float32),
        SCL_modeled_FVCs_Cin=jnp.zeros((1, 2), dtype=jnp.float32),
    )

    assert inputs.SCL_latent.shape == (0,)
    assert inputs.SCL_latent.dtype == jnp.float64


def test_reaction_outputs_default_to_zero_width_latent_derivative():
    outputs = ReactionOutputs(
        SCL_modeled_BiologicalOde_rates=jnp.zeros(2, dtype=jnp.float32),
        SCL_modeled_FVCs_rates=jnp.zeros(1, dtype=jnp.float32),
    )

    assert outputs.SCL_latent_derivative.shape == (0,)
    assert outputs.SCL_latent_derivative.dtype == jnp.float64


class _LatentScaleModule(UserReactionModule):
    def __init__(self):
        super().__init__()
        self.SCALE_modeled_RMCs = jnp.asarray([2.0, 4.0], dtype=jnp.float32)
        self.SCALE_V_in_cumulative = jnp.asarray(10.0, dtype=jnp.float32)
        self.SCALE_modeled_FVCs_cumulative = jnp.asarray([5.0], dtype=jnp.float32)
        self.SCALE_latent = jnp.asarray([3.0, 6.0], dtype=jnp.float32)

    def __call__(self, t, inputs) -> ReactionOutputs:
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.zeros(0, dtype=jnp.float32),
            SCL_modeled_FVCs_rates=jnp.zeros(0, dtype=jnp.float32),
        )


def test_user_reaction_module_latent_scales_are_separate_from_state_scales():
    module = _LatentScaleModule()

    assert module.n_latent == 2
    assert jnp.array_equal(module.SCALE_state, jnp.asarray([2.0, 4.0, 10.0, 5.0]))
    assert jnp.array_equal(
        module.SCALE_integrated_state,
        jnp.asarray([2.0, 4.0, 10.0, 5.0, 3.0, 6.0]),
    )


def test_user_reaction_module_latent_helpers_are_linear():
    module = _LatentScaleModule()
    raw = jnp.asarray([9.0, 24.0], dtype=jnp.float32)
    scl = jnp.asarray([3.0, 4.0], dtype=jnp.float32)

    assert jnp.array_equal(module.scale_latent(raw), jnp.asarray([3.0, 4.0]))
    assert jnp.array_equal(module.unscale_latent(scl), raw)


def test_user_reaction_module_default_initial_latent_matches_width_and_dtype():
    module = _LatentScaleModule()
    y0 = jnp.asarray([1.0, 2.0], dtype=jnp.float16)

    h0 = module.initial_latent(y0)

    assert h0.shape == (2,)
    assert h0.dtype == y0.dtype
    assert jnp.array_equal(h0, jnp.zeros(2, dtype=y0.dtype))


def test_user_reaction_module_defaults_are_stateless():
    module = _UntaggedArrayModule()
    y0 = jnp.asarray([1.0], dtype=jnp.float32)

    assert module.n_latent == 0
    assert module.SCALE_latent.shape == (0,)
    assert jnp.array_equal(module.SCALE_integrated_state, module.SCALE_state)
    assert module.initial_latent(y0).shape == (0,)


def test_scale_latent_is_frozen_by_default():
    module = _LatentScaleModule()

    trainable, static = partition_trainable(module)

    assert trainable.SCALE_latent is None
    assert static.SCALE_latent is not None
