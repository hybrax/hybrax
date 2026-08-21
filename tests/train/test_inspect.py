from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from bp_train.defaults import DefaultStatefulReactionModule
from bp_train.inspect import (
    format_reaction_schema,
    format_trainable_structure,
)
from bp_train.model_api import (
    ReactionOutputs,
    UserReactionModule,
    frozen_field,
    trainable_field,
)
from stateful_helpers import default_stateful_scale_kwargs


# ---------------------------------------------------------------------------
# format_reaction_schema
# ---------------------------------------------------------------------------


def _mock_rhs_ode_and_controls():
    rhs_ode = SimpleNamespace(
        name_modeled_RMCs=("biomass", "glucose", "acetate"),
        name_modeled_PVs=("ratio",),
        name_modeled_Inflows=("feed_glc",),
        name_controlled_Inflows=("base", "antifoam"),
        name_modeled_Outflows=("perfusion",),
        name_controlled_Outflows=("bleed",),
        name_controlled_PVs=("pH", "DO", "T"),
        name_modeled_rates=("q_biomass", "q_glucose", "q_acetate"),
    )
    controls = SimpleNamespace()
    return rhs_ode, controls


def test_format_reaction_schema_contains_all_axis_names():
    rhs_ode, controls = _mock_rhs_ode_and_controls()
    text = format_reaction_schema(rhs_ode, controls)

    for axis in (
        "SCL_modeled_RMCs",
        "SCL_modeled_PVs",
        "SCL_modeled_V",
        "SCL_modeled_Inflows_cumulative",
        "SCL_modeled_Outflows_cumulative",
        "SCL_controlled_Inflows_cumulative",
        "SCL_controlled_Inflows_rates",
        "SCL_controlled_Outflows_cumulative",
        "SCL_controlled_Outflows_rates",
        "SCL_controlled_Inflows_Cin",
        "SCL_controlled_PVs",
        "SCL_modeled_Inflows_Cin",
        "SCL_modeled_BiologicalOde_rates",
        "SCL_modeled_Inflows_rates",
        "SCL_modeled_Outflows_rates",
        "RAW_controlled_Outflows_retention",
        "RAW_modeled_Outflows_retention",
    ):
        assert axis in text, f"axis {axis} missing from rendered schema"


def test_format_reaction_schema_renders_bp_format_names():
    rhs_ode, controls = _mock_rhs_ode_and_controls()
    text = format_reaction_schema(rhs_ode, controls)

    for species in ("biomass", "glucose", "acetate"):
        assert species in text
    for flow in ("base", "antifoam", "feed_glc", "bleed", "perfusion"):
        assert flow in text
    for pv in ("pH", "DO", "T"):
        assert pv in text
    for rate in ("q_biomass", "q_glucose", "q_acetate"):
        assert rate in text
    # Bolus Inflows are applied as discrete state jumps, not reaction-module inputs,
    # so neither the bolus name nor the wrapper-internal V_sample_acc surfaces here.
    assert "inducer_bolus" not in text
    assert "V_sample_acc" not in text


def test_format_reaction_schema_matrix_followups_render_rows_cols():
    rhs_ode, controls = _mock_rhs_ode_and_controls()
    text = format_reaction_schema(rhs_ode, controls)

    # 2-D Cin matrices emit indented "rows:" / "cols:" follow-up lines.
    assert "rows: base, antifoam" in text
    assert "cols: biomass, glucose, acetate" in text
    assert "rows: feed_glc" in text


def test_format_reaction_schema_handles_empty_axes():
    rhs_ode = SimpleNamespace(
        name_modeled_RMCs=("biomass",),
        name_modeled_PVs=(),
        name_modeled_Inflows=(),
        name_controlled_Inflows=(),
        name_modeled_Outflows=(),
        name_controlled_Outflows=(),
        name_controlled_PVs=(),
        name_modeled_rates=("q_biomass",),
    )
    controls = SimpleNamespace()
    text = format_reaction_schema(rhs_ode, controls)

    # The (0,) shape signals an empty axis; the names cell stays blank
    # (no "(none)" placeholder).
    assert "(none)" not in text
    assert "biomass" in text
    assert "q_biomass" in text


# ---------------------------------------------------------------------------
# format_trainable_structure (smoke check that the move didn't break it)
# ---------------------------------------------------------------------------


class _MixedTagsFixture(UserReactionModule):
    weights: jax.Array = trainable_field()
    bias_frozen: jax.Array = frozen_field()

    def __init__(self):
        super().__init__()
        self.weights = jnp.zeros((2, 2))
        self.bias_frozen = jnp.zeros((2,))

    def __call__(self, t, inputs):
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.zeros((0,)),
            SCL_modeled_Inflows_rates=jnp.zeros((0,)),
            SCL_modeled_Outflows_rates=jnp.zeros((0,)),
        )


def test_format_trainable_structure_works_from_inspect_module():
    module = _MixedTagsFixture()
    text = format_trainable_structure(module)

    assert "weights" in text
    assert "bias_frozen" in text
    assert "trainable" in text
    assert "frozen" in text


class _DictFieldFixture(UserReactionModule):
    heads: dict[str, jax.Array] = trainable_field()

    def __init__(self):
        super().__init__()
        self.heads = {"a": jnp.zeros((2,)), "b": jnp.zeros((3,))}

    def __call__(self, t, inputs):
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.zeros((0,)),
            SCL_modeled_Inflows_rates=jnp.zeros((0,)),
            SCL_modeled_Outflows_rates=jnp.zeros((0,)),
        )


def test_format_trainable_structure_reveals_dict_valued_trainable_fields():
    # Container types the old hand-rolled walker didn't special-case (e.g.
    # dict) used to vanish silently; the jtu.tree_flatten_with_path-based
    # walk finds every array leaf regardless of container type.
    module = _DictFieldFixture()
    text = format_trainable_structure(module)

    assert "heads['a']" in text
    assert "heads['b']" in text
    assert "trainable" in text


def test_format_trainable_structure_omits_none_valued_optional_field():
    # A trainable_field() declared as `X | None` and currently unset (e.g.
    # inflow_head when n_modeled_Inflows == 0) has zero children under JAX's
    # pytree flattening, so it produces no row -- concise by design, not a
    # bug: nothing about a None field could ever be an array leaf.
    module = DefaultStatefulReactionModule(
        key=jax.random.key(0),
        n_latent=2,
        **default_stateful_scale_kwargs(n_controlled_inflows=0),
    )
    assert module.inflow_head is None
    text = format_trainable_structure(module)

    assert "inflow_head" not in text
    assert "gru_cell.weight_ih" in text
    assert "rate_head.weight" in text
