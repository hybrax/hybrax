from __future__ import annotations

import numpy as np
import pytest

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

import hybrax.train.model_api as model_api
from hybrax.train.inspect import format_trainable_structure
from hybrax.train.model_api import (
    TRAINABLE_METADATA_KEY,
    AffineScaler,
    LinearScaler,
    ReactionInputs,
    ReactionOutputs,
    Scaler,
    UserReactionModule,
    _as_scaler,
    _compose_scalers,
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
        self.non_model_bias = jnp.asarray([2.0])

    def __call__(self, t, inputs) -> ReactionOutputs:
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray([0.0]),
            SCL_modeled_Inflows_rates=jnp.zeros((0,)),
            SCL_modeled_Outflows_rates=jnp.zeros((0,)),
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
        self.raw = jnp.asarray([1.0])

    def __call__(self, t, inputs) -> ReactionOutputs:
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray([0.0]),
            SCL_modeled_Inflows_rates=jnp.zeros((0,)),
            SCL_modeled_Outflows_rates=jnp.zeros((0,)),
        )


class _MixedTagsModule(UserReactionModule):
    weights: jax.Array = trainable_field()
    bias_frozen: jax.Array = frozen_field()

    def __init__(self):
        super().__init__()
        self.weights = jnp.asarray([1.0, 2.0])
        self.bias_frozen = jnp.asarray([10.0])

    def __call__(self, t, inputs) -> ReactionOutputs:
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray([0.0]),
            SCL_modeled_Inflows_rates=jnp.zeros((0,)),
            SCL_modeled_Outflows_rates=jnp.zeros((0,)),
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
        self.array1 = jnp.asarray([1.0])
        self.array2 = jnp.asarray([2.0])


class _OuterFrozenParent(UserReactionModule):
    sub: _Inner = frozen_field()

    def __init__(self):
        super().__init__()
        self.sub = _Inner()

    def __call__(self, t, inputs) -> ReactionOutputs:
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray([0.0]),
            SCL_modeled_Inflows_rates=jnp.zeros((0,)),
            SCL_modeled_Outflows_rates=jnp.zeros((0,)),
        )


class _OuterTrainableParent(UserReactionModule):
    sub: _Inner = trainable_field()

    def __init__(self):
        super().__init__()
        self.sub = _Inner()

    def __call__(self, t, inputs) -> ReactionOutputs:
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray([0.0]),
            SCL_modeled_Inflows_rates=jnp.zeros((0,)),
            SCL_modeled_Outflows_rates=jnp.zeros((0,)),
        )


class _OuterUntaggedParent(UserReactionModule):
    sub: _Inner

    def __init__(self):
        super().__init__()
        self.sub = _Inner()

    def __call__(self, t, inputs) -> ReactionOutputs:
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray([0.0]),
            SCL_modeled_Inflows_rates=jnp.zeros((0,)),
            SCL_modeled_Outflows_rates=jnp.zeros((0,)),
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
            self.a_very_long_field_name_for_testing = jnp.asarray([1.0])

        def __call__(self, t, c_species, controls_vector) -> ReactionOutputs:
            del t, c_species, controls_vector
            return ReactionOutputs(
                SCL_modeled_BiologicalOde_rates=jnp.asarray([0.0]),
                SCL_modeled_Inflows_rates=jnp.zeros((0,)),
                SCL_modeled_Outflows_rates=jnp.zeros((0,)),
            )

    module = _LongName()
    text = format_trainable_structure(module)

    assert "a_very_long_field_name_for_testing" in text
    # Header lines should be at least as wide as the longest name plus borders.
    lines = text.splitlines()
    assert all(len(line) == len(lines[0]) for line in lines if line.startswith("|"))


def test_user_reaction_module_default_observe_is_identity():
    module = _UntaggedArrayModule()
    y = jnp.asarray([1.0, 2.0])

    assert jnp.array_equal(module.observe(y), y)


# ---------------------------------------------------------------------------
# Latent-state contract
# ---------------------------------------------------------------------------


def test_reaction_inputs_default_to_zero_width_latent():
    inputs = ReactionInputs(
        SCL_modeled_RMCs=jnp.ones(2),
        SCL_modeled_V=jnp.asarray(1.0),
        SCL_modeled_Inflows_cumulative=jnp.zeros(1),
        SCL_modeled_Outflows_cumulative=jnp.zeros(2),
        SCL_controlled_Inflows_cumulative=jnp.zeros(3),
        SCL_controlled_Inflows_rates=jnp.zeros(3),
        SCL_controlled_Inflows_Cin=jnp.zeros((3, 2)),
        SCL_controlled_Outflows_cumulative=jnp.zeros(4),
        SCL_controlled_Outflows_rates=jnp.zeros(4),
        RAW_controlled_Outflows_retention=jnp.zeros((4, 2)),
        SCL_controlled_PVs=jnp.zeros(0),
        SCL_modeled_Inflows_Cin=jnp.zeros((1, 2)),
        RAW_modeled_Outflows_retention=jnp.zeros((2, 2)),
    )

    assert inputs.SCL_latent.shape == (0,)
    assert inputs.SCL_latent.dtype == jnp.float64
    assert inputs.SCL_controlled_Inflows_Cin.shape == (3, 2)
    assert inputs.RAW_controlled_Outflows_retention.shape == (4, 2)
    assert inputs.RAW_modeled_Outflows_retention.shape == (2, 2)


def test_reaction_outputs_default_to_zero_width_latent_derivative():
    outputs = ReactionOutputs(
        SCL_modeled_BiologicalOde_rates=jnp.zeros(2),
        SCL_modeled_Inflows_rates=jnp.zeros(1),
        SCL_modeled_Outflows_rates=jnp.zeros(2),
    )

    assert outputs.SCL_latent_derivative.shape == (0,)
    assert outputs.SCL_latent_derivative.dtype == jnp.float64


class _LatentScaleModule(UserReactionModule):
    def __init__(self):
        super().__init__(
            SCALE_modeled_RMCs=jnp.asarray([2.0, 4.0]),
            SCALE_V_in_cumulative=jnp.asarray(10.0),
            SCALE_modeled_Inflows_cumulative=jnp.asarray([5.0]),
            SCALE_modeled_Outflows_cumulative=jnp.asarray([7.0, 8.0]),
            SCALE_latent=jnp.asarray([3.0, 6.0]),
        )

    def __call__(self, t, inputs) -> ReactionOutputs:
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.zeros(0),
            SCL_modeled_Inflows_rates=jnp.zeros(0),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
        )


def test_user_reaction_module_latent_scales_are_separate_from_state_scales():
    module = _LatentScaleModule()

    assert module.n_latent == 2
    assert jnp.array_equal(
        module.SCALE_state.scale, jnp.asarray([2.0, 4.0, 10.0, 5.0, 7.0, 8.0])
    )
    assert jnp.array_equal(
        module.SCALE_integrated_state.scale,
        jnp.asarray([2.0, 4.0, 10.0, 5.0, 7.0, 8.0, 3.0, 6.0]),
    )


def test_user_reaction_module_latent_helpers_are_linear():
    module = _LatentScaleModule()
    raw = jnp.asarray([9.0, 24.0])
    scl = jnp.asarray([3.0, 4.0])

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
    y0 = jnp.asarray([1.0])

    assert module.n_latent == 0
    assert module.SCALE_latent.shape == (0,)
    assert jnp.array_equal(
        module.SCALE_integrated_state.scale, module.SCALE_state.scale
    )
    assert module.initial_latent(y0).shape == (0,)


def test_scale_latent_is_frozen_by_default():
    module = _LatentScaleModule()

    trainable, static = partition_trainable(module)

    assert trainable.SCALE_latent.scale is None
    assert static.SCALE_latent.scale is not None


# ---------------------------------------------------------------------------
# Scaler abstraction (OP12 Phase B). Pure-division default is bit-identical to
# the pre-scaler RAW/SCALE code; affine diverges value vs derivative (§0).
# ---------------------------------------------------------------------------


def test_linear_scaler_is_pure_division_bit_identical():
    # Test 1: LinearScaler is exactly x/s and x*s, bit-for-bit.
    s = jnp.asarray([2.0, 4.0, 8.0])
    raw = jnp.asarray([4.0, 8.0, 16.0])
    scaler = LinearScaler(s)
    assert jnp.array_equal(raw / scaler, raw / s)
    assert jnp.array_equal(raw * scaler, raw * s)
    # round-trip
    assert jnp.array_equal((raw * scaler) / scaler, raw)
    # derivative ops match value ops for pure division
    assert jnp.array_equal(scaler.scale_derivative(raw), raw / s)
    assert jnp.array_equal(scaler.unscale_derivative(raw), raw * s)


@pytest.mark.parametrize(
    "scaler",
    [
        LinearScaler(jnp.ones(3)),
        AffineScaler(jnp.ones(3), jnp.zeros(3)),
    ],
)
def test_scaler_not_silently_array_coercible(scaler):
    # Test 10: every array coercion raises rather than producing object arrays.
    with pytest.raises(TypeError):
        jnp.asarray(scaler)
    with pytest.raises(TypeError):
        jnp.concatenate([scaler, jnp.ones(3)])
    with pytest.raises(TypeError):
        np.ones(3) / scaler
    with pytest.raises(TypeError):
        np.asarray(scaler)


@pytest.mark.parametrize(
    "scaler",
    [
        # float32 on purpose: this test pins that a numpy input's dtype survives the
        # scale/unscale round trip, which a float64 scaler would silently promote away.
        LinearScaler(jnp.asarray([2.0, 4.0], dtype=jnp.float32)),
        AffineScaler(
            jnp.asarray([2.0, 4.0], dtype=jnp.float32),
            jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        ),
    ],
)
def test_value_helpers_accept_numpy_arrays(scaler):
    module = UserReactionModule(SCALE_modeled_RMCs=scaler)
    raw = np.asarray([4.0, 8.0], dtype=np.float32)

    scaled = module.scale_modeled_RMCs(raw)
    unscaled = module.unscale_modeled_RMCs(np.asarray(scaled))

    assert scaled.dtype == jnp.float32
    assert unscaled.dtype == jnp.float32
    assert np.array_equal(np.asarray(unscaled), raw)


def test_scaler_reversed_arithmetic_raises_loudly():
    # Test 9 (part): scaler * x and scaler / x have no dunder and raise,
    # so a reversed order cannot silently pick a value transform.
    scaler = LinearScaler(jnp.ones(3))
    x = jnp.ones(3)
    with pytest.raises(TypeError):
        scaler * x
    with pytest.raises(TypeError):
        scaler / x


def test_concat_scaler_preserves_zero_width_dtype_promotion():
    # Test 8: a zero-width float64 axis (the empty-PV default in 9/10 examples)
    # drives the composed state dtype to float64 — lock the promotion in
    # deliberately, do NOT filter empties.
    parts = [
        LinearScaler(jnp.asarray([2.0, 4.0], dtype=jnp.float32)),
        LinearScaler(jnp.zeros(0, dtype=jnp.float64)),  # zero-width f64
        LinearScaler(jnp.asarray(10.0, dtype=jnp.float32)),
        LinearScaler(jnp.asarray([5.0], dtype=jnp.float32)),
    ]
    composed = _compose_scalers(*parts)
    assert composed.shape == (4,)
    # The composed scale array is float64 because the empty f64 part promotes it.
    assert composed.scale.dtype == jnp.float64
    # Dropping the empty part would make it float32 — guard against that tidy-up.
    without_empty = _compose_scalers(parts[0], parts[2], parts[3])
    assert without_empty.scale.dtype == jnp.float32


def test_concat_scaler_getitem_and_value_ops():
    parts = [
        LinearScaler(jnp.asarray([2.0, 4.0])),
        LinearScaler(jnp.asarray([10.0])),
    ]
    composed = _compose_scalers(*parts)
    sub = composed[jnp.asarray([0, 1])]
    assert type(sub) is LinearScaler
    assert jnp.array_equal(sub.scale, jnp.asarray([2.0, 4.0]))
    raw = jnp.asarray([4.0, 8.0, 20.0])
    assert jnp.array_equal(raw / composed, jnp.asarray([2.0, 2.0, 2.0]))
    assert jnp.array_equal(raw * composed, jnp.asarray([8.0, 32.0, 200.0]))


def test_rate_helpers_use_derivative_semantics():
    class DivergentRateScaler(Scaler):
        scale: jax.Array

        def __init__(self):
            self.scale = jnp.ones(1)

        def __rtruediv__(self, raw):
            return raw / 10.0

        def __rmul__(self, scl):
            return scl * 10.0

        def scale_derivative(self, rate):
            return rate / 3.0

        def unscale_derivative(self, rate):
            return rate * 3.0

        @property
        def shape(self):
            return self.scale.shape

        def astype(self, dtype):
            return self

        def __getitem__(self, idx):
            return self

    module = UserReactionModule(
        SCALE_controlled_Inflows_rates=DivergentRateScaler(),
        SCALE_controlled_Outflows_rates=DivergentRateScaler(),
        SCALE_modeled_BiologicalOde_rates=DivergentRateScaler(),
        SCALE_modeled_Inflows_rates=DivergentRateScaler(),
        SCALE_modeled_Outflows_rates=DivergentRateScaler(),
    )
    raw = jnp.asarray([6.0])
    scl = jnp.asarray([2.0])
    helpers = (
        (
            module.scale_controlled_Inflows_rates,
            module.unscale_controlled_Inflows_rates,
        ),
        (
            module.scale_controlled_Outflows_rates,
            module.unscale_controlled_Outflows_rates,
        ),
        (
            module.scale_modeled_BiologicalOde_rates,
            module.unscale_modeled_BiologicalOde_rates,
        ),
        (module.scale_modeled_Inflows_rates, module.unscale_modeled_Inflows_rates),
        (
            module.scale_modeled_Outflows_rates,
            module.unscale_modeled_Outflows_rates,
        ),
    )

    for scale, unscale in helpers:
        assert jnp.array_equal(scale(raw), scl)
        assert jnp.array_equal(unscale(scl), raw)


def test_tree_at_affine_offset_zero_to_nonzero_updates_value_and_state():
    scaler = AffineScaler(
        jnp.asarray([2.0]),
        jnp.asarray([0.0]),
    ).astype(jnp.float64)
    scaler = eqx.tree_at(
        lambda x: x.offset,
        scaler,
        jnp.asarray([10.0], dtype=jnp.float64),
    )

    assert jnp.array_equal(
        scaler.unscale_value(jnp.asarray([1.0])), jnp.asarray([12.0])
    )

    module = UserReactionModule(SCALE_modeled_RMCs=scaler)
    state = jnp.asarray([1.0, 1.0])
    expected = jnp.asarray([12.0, 1.0])
    assert jnp.array_equal(module.unscale_state(state), expected)
    assert jnp.array_equal(
        jax.jit(lambda m, x: m.unscale_state(x))(module, state), expected
    )
    cast = jax.jit(
        lambda x: module.SCALE_integrated_state.astype(jnp.float64).unscale_value(x)
    )(state.astype(jnp.float64))
    assert jnp.array_equal(cast, expected.astype(jnp.float64))


def test_tree_at_affine_offset_nonzero_to_zero_preserves_negative_zero():
    scaler = AffineScaler(jnp.asarray([2.0]), jnp.asarray([10.0]))
    scaler = eqx.tree_at(lambda x: x.offset, scaler, jnp.asarray([0.0]))

    bare = scaler.unscale_value(jnp.asarray([-0.0]))
    assert bool(jnp.signbit(bare[0]))

    module = UserReactionModule(SCALE_modeled_RMCs=scaler)
    state = jnp.asarray([-0.0, -0.0])
    assert bool(jnp.signbit(module.unscale_state(state)[0]))
    jitted = jax.jit(lambda m, x: m.unscale_state(x))(module, state)
    cast = jax.jit(
        lambda x: module.SCALE_integrated_state.astype(jnp.float64).unscale_value(x)
    )(state.astype(jnp.float64))
    assert bool(jnp.signbit(jitted[0]))
    assert bool(jnp.signbit(cast[0]))


def test_dynamic_negative_zero_affine_offset_preserves_negative_zero_value():
    scaler = AffineScaler(
        jnp.asarray([2.0]),
        jnp.asarray([-0.0]),
    )
    raw = jnp.asarray([-0.0])

    scaled = jax.jit(lambda dynamic_scaler, x: dynamic_scaler.scale_value(x))(
        scaler, raw
    )

    assert bool(jnp.signbit(scaled[0]))


def test_closed_over_zero_offset_affine_gradient_preserves_negative_zero():
    module = UserReactionModule(
        SCALE_modeled_RMCs=AffineScaler(
            jnp.asarray([2.0]),
            jnp.asarray([0.0]),
        )
    )
    state = jnp.asarray([1.0, 1.0])
    negative_zero = jnp.asarray(-0.0)

    scale_grad = jax.jit(
        jax.grad(lambda raw: jnp.sum(module.scale_state(raw) * negative_zero))
    )(state)
    unscale_grad = jax.jit(
        jax.grad(lambda scl: jnp.sum(module.unscale_state(scl) * negative_zero))
    )(state)

    cast_state = state.astype(jnp.float64)
    cast_negative_zero = negative_zero.astype(jnp.float64)
    cast_unscale_grad = jax.jit(
        jax.grad(
            lambda scl: jnp.sum(
                module.SCALE_integrated_state.astype(scl.dtype).unscale_value(scl)
                * cast_negative_zero
            )
        )
    )(cast_state)

    assert bool(jnp.all(jnp.signbit(scale_grad)))
    assert bool(jnp.all(jnp.signbit(unscale_grad)))
    assert bool(jnp.all(jnp.signbit(cast_unscale_grad)))


def test_zero_offset_affine_state_scaler_preserves_negative_zero():
    module = UserReactionModule(
        SCALE_modeled_RMCs=AffineScaler(
            jnp.asarray([2.0]),
            jnp.asarray([0.0]),
        )
    )
    scl = jnp.asarray([-0.0, -0.0], dtype=jnp.float64)

    assert type(module.SCALE_integrated_state) is LinearScaler
    eager = module.unscale_state(scl)
    jitted = jax.jit(lambda m, x: m.unscale_state(x))(module, scl)
    cast_jitted = jax.jit(
        lambda m, x: m.SCALE_integrated_state.astype(jnp.float32).unscale_value(x)
    )(module, scl.astype(jnp.float32))

    assert bool(jnp.signbit(eager[0]))
    assert bool(jnp.signbit(jitted[0]))
    assert bool(jnp.signbit(cast_jitted[0]))


def test_concat_scaler_materializes_arrays_once(monkeypatch):
    calls = 0
    concatenate = model_api.jnp.concatenate

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return concatenate(*args, **kwargs)

    monkeypatch.setattr(model_api.jnp, "concatenate", counted)
    scaler = _compose_scalers(
        LinearScaler(jnp.asarray([2.0])),
        AffineScaler(
            jnp.asarray([4.0]),
            jnp.asarray([1.0]),
        ),
    )
    raw = jnp.asarray([2.0, 9.0])

    raw / scaler
    raw * scaler
    scaler.scale_derivative(raw)
    scaler.unscale_derivative(raw)

    assert calls == 2


def test_all_linear_concat_is_op_for_op_and_preserves_negative_zero():
    # Test 12: the default path must not re-express LinearScaler composition as
    # affine with b=0. In particular, SCL*s + 0 flips -0.0 to +0.0.
    scales = [
        jnp.asarray([2.0, 4.0]),
        jnp.zeros(0, dtype=jnp.float64),
        jnp.asarray(10.0),
        jnp.asarray([5.0]),
    ]
    composed = _compose_scalers(*(LinearScaler(scale) for scale in scales))
    concat_scale = jnp.concatenate([jnp.atleast_1d(scale) for scale in scales])
    raw = jnp.asarray([-0.0, 8.0, 10.0, 5.0], dtype=concat_scale.dtype)
    scl = jnp.asarray([-0.0, 2.0, 1.0, 1.0], dtype=concat_scale.dtype)
    expected_scale = raw / concat_scale
    expected_unscale = scl * concat_scale
    actual_scale = raw / composed
    actual_unscale = scl * composed
    assert jnp.array_equal(actual_scale, expected_scale)
    assert jnp.array_equal(actual_unscale, expected_unscale)
    assert bool(jnp.signbit(actual_unscale[0]))
    assert bool(jnp.signbit(expected_unscale[0]))


def test_as_scaler_promotes_array_passes_scaler():
    arr = jnp.ones(3)
    assert isinstance(_as_scaler(arr), LinearScaler)
    scaler = LinearScaler(arr)
    assert _as_scaler(scaler) is scaler


def test_user_reaction_module_promotes_bare_scale_arrays():
    # Modules constructed with bare arrays (tests, user modules) get
    # LinearScaler via __init__ promotion.
    module = UserReactionModule(
        SCALE_modeled_RMCs=jnp.asarray([2.0, 4.0]),
        SCALE_V_in_cumulative=jnp.asarray(10.0),
    )
    assert isinstance(module.SCALE_modeled_RMCs, LinearScaler)
    assert isinstance(module.SCALE_V_in_cumulative, LinearScaler)
    assert module.n_modeled_RMCs == 2
    # All-linear composition reuses the concrete linear implementation.
    assert type(module.SCALE_state) is LinearScaler
    assert module.SCALE_state.shape == (3,)


def test_scaler_constructors_reject_non_array_inputs_immediately():
    with pytest.raises(TypeError, match="LinearScaler.scale"):
        LinearScaler([1.0])
    with pytest.raises(TypeError, match="AffineScaler.scale"):
        AffineScaler(1.0, jnp.asarray(0.0))
    with pytest.raises(TypeError, match="AffineScaler.offset"):
        AffineScaler(jnp.asarray(1.0), 0.0)


def test_user_reaction_module_rejects_unknown_scale_keyword():
    with pytest.raises(TypeError, match="SCALE_V_in_cumulativ"):
        UserReactionModule(SCALE_V_in_cumulativ=jnp.asarray(7.0))


def test_user_reaction_module_initializes_defaulted_subclass_fields():
    class Child(UserReactionModule):
        w: jax.Array = trainable_field(default_factory=lambda: jnp.ones(1))

    supplied = jnp.asarray([2.0], dtype=jnp.float32)
    defaulted = Child()
    explicit = Child(w=supplied)

    assert jnp.array_equal(defaulted.w, jnp.ones(1))
    assert explicit.w is supplied
    assert explicit.w.dtype == jnp.float32


def test_affine_value_and_derivative_ops_diverge():
    # Test 2 core (§0): value scale subtracts b; derivative scale must NOT.
    scaler = AffineScaler(
        scale=jnp.asarray([2.0, 4.0]),
        offset=jnp.asarray([10.0, 100.0]),
    )
    raw = jnp.asarray([12.0, 104.0])
    assert jnp.array_equal(raw / scaler, jnp.asarray([1.0, 1.0]))
    assert jnp.array_equal(scaler.scale_derivative(raw), jnp.asarray([6.0, 26.0]))
    assert jnp.array_equal(
        scaler.unscale_derivative(jnp.asarray([1.0, 1.0])),
        jnp.asarray([2.0, 4.0]),
    )


def test_affine_dunders_apply_offset_and_reversed_order_raises():
    # Test 9: dunders route to VALUE semantics, not a bare scale array.
    scaler = AffineScaler(
        scale=jnp.asarray([2.0, 4.0]),
        offset=jnp.asarray([10.0, 100.0]),
    )
    raw = jnp.asarray([12.0, 104.0])
    scl = jnp.asarray([1.0, 1.0])
    assert jnp.array_equal(raw / scaler, scl)
    assert jnp.array_equal(scl * scaler, raw)
    with pytest.raises(TypeError):
        scaler * scl
    with pytest.raises(TypeError):
        scaler / raw


def test_affine_roundtrip_scalar_zero_width_and_2d():
    # Test 5: all legitimate axis shapes round-trip through value ops.
    cases = [
        AffineScaler(jnp.asarray(2.0), jnp.asarray(10.0)),
        AffineScaler(jnp.zeros(0), jnp.zeros(0)),
        AffineScaler(
            jnp.asarray([[2.0, 4.0], [5.0, 10.0]]),
            jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        ),
    ]
    for scaler in cases:
        raw = scaler.offset + 2.0 * scaler.scale
        assert jnp.array_equal((raw / scaler) * scaler, raw)
        assert scaler.shape == scaler.scale.shape


def test_concat_scaler_preserves_offset_layout_and_getitem():
    # Tests 4 + 11: distinguishable values and non-width-1 state axes catch a
    # semantic permutation; composed value op and [idx] both preserve offsets.
    rmcs = AffineScaler(
        jnp.asarray([2.0, 3.0]),
        jnp.asarray([10.0, 20.0]),
    )
    pvs = LinearScaler(jnp.zeros(0))
    volume = AffineScaler(jnp.asarray(4.0), jnp.asarray(30.0))
    cumulative = AffineScaler(
        jnp.asarray([5.0, 6.0]),
        jnp.asarray([40.0, 50.0]),
    )
    state = _compose_scalers(rmcs, pvs, volume, cumulative)
    assert type(state) is AffineScaler
    assert jnp.array_equal(state.scale, jnp.asarray([2.0, 3.0, 4.0, 5.0, 6.0]))
    assert jnp.array_equal(state.offset, jnp.asarray([10.0, 20.0, 30.0, 40.0, 50.0]))
    raw = state.offset + state.scale
    assert jnp.array_equal(raw / state, jnp.ones(5))
    sub = state[jnp.asarray([0, 3])]
    assert type(sub) is AffineScaler
    assert jnp.array_equal(sub.offset, jnp.asarray([10.0, 40.0]))
    assert jnp.array_equal(jnp.asarray([12.0, 45.0]) / sub, jnp.ones(2))


def test_integrated_state_preserves_discarded_zero_offset_dtype():
    module = UserReactionModule(
        SCALE_modeled_RMCs=AffineScaler(
            jnp.asarray([2.0], dtype=jnp.float32),
            jnp.asarray([0.0], dtype=jnp.float64),
        ),
        SCALE_modeled_PVs=LinearScaler(jnp.zeros(0, dtype=jnp.float32)),
        SCALE_V_in_cumulative=LinearScaler(jnp.asarray(3.0, dtype=jnp.float32)),
        SCALE_modeled_Inflows_cumulative=LinearScaler(jnp.zeros(0, dtype=jnp.float32)),
        SCALE_modeled_Outflows_cumulative=LinearScaler(jnp.zeros(0, dtype=jnp.float32)),
        SCALE_latent=AffineScaler(
            jnp.asarray([4.0], dtype=jnp.float32),
            jnp.asarray([5.0], dtype=jnp.float32),
        ),
    )

    integrated = module.SCALE_integrated_state
    assert type(module.SCALE_state) is LinearScaler
    assert type(integrated) is AffineScaler
    assert integrated.offset.dtype == jnp.float64
    assert integrated.unscale_value(jnp.ones(3, dtype=jnp.float32)).dtype == jnp.float64


def test_affine_scaler_leaves_are_frozen_by_partition():
    # Test 7: both affine leaves belong to the static partition.
    module = UserReactionModule(
        SCALE_modeled_RMCs=AffineScaler(
            jnp.asarray([2.0, 4.0]), jnp.asarray([10.0, 20.0])
        )
    )
    trainable, static = partition_trainable(module)
    assert trainable.SCALE_modeled_RMCs.scale is None
    assert trainable.SCALE_modeled_RMCs.offset is None
    assert jnp.array_equal(static.SCALE_modeled_RMCs.scale, jnp.asarray([2.0, 4.0]))
    assert jnp.array_equal(static.SCALE_modeled_RMCs.offset, jnp.asarray([10.0, 20.0]))


def test_concat_rejects_builtin_subclasses_with_custom_semantics():
    class LogLinearScaler(LinearScaler):
        def __rtruediv__(self, raw):
            return jnp.log(raw)

        def __rmul__(self, scl):
            return jnp.exp(scl)

    class LogAffineScaler(AffineScaler):
        def __rtruediv__(self, raw):
            return jnp.log(raw)

        def __rmul__(self, scl):
            return jnp.exp(scl)

    with pytest.raises(TypeError, match="exact LinearScaler"):
        _compose_scalers(LogLinearScaler(jnp.ones(1)))
    with pytest.raises(TypeError, match="exact LinearScaler"):
        _compose_scalers(LogAffineScaler(jnp.ones(1), jnp.zeros(1)))


def test_concat_rejects_custom_non_affine_state_scaler():
    # State-axis restriction: even a custom scaler exposing scale/offset metadata
    # must be rejected — composition would reinterpret its actual value ops as
    # affine and silently corrupt the solver coordinate system.
    class LogScaler(Scaler):
        scale: jax.Array
        offset: jax.Array

        def __init__(self):
            self.scale = jnp.ones(1)
            self.offset = jnp.zeros(1)

        def __rtruediv__(self, raw):
            return jnp.log(raw)

        def __rmul__(self, scl):
            return jnp.exp(scl)

        def scale_derivative(self, rate):
            return rate

        def unscale_derivative(self, rate):
            return rate

        @property
        def shape(self):
            return self.scale.shape

        def astype(self, dtype):
            return LogScaler()

        def __getitem__(self, idx):
            return self

    with pytest.raises(TypeError, match="exact LinearScaler"):
        _compose_scalers(LogScaler())
