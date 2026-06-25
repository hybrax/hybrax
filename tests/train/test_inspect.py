from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp

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


# ---------------------------------------------------------------------------
# format_reaction_schema
# ---------------------------------------------------------------------------


def _mock_rhs_ode_and_controls():
    rhs_ode = SimpleNamespace(
        name_modeled_RMCs=("biomass", "glucose", "acetate"),
        name_modeled_PVs=("ratio",),
        name_modeled_FVCs=("feed_glc",),
        name_controlled_FVCs=("base", "antifoam"),
        name_controlled_PVs=("pH", "DO", "T"),
        name_modeled_rates=("q_biomass", "q_glucose", "q_acetate"),
    )
    controls = SimpleNamespace(
        name_extras=("inducer_bolus", "V_sample_acc"),
    )
    return rhs_ode, controls


def test_format_reaction_schema_contains_all_axis_names():
    rhs_ode, controls = _mock_rhs_ode_and_controls()
    text = format_reaction_schema(rhs_ode, controls)

    for axis in (
        "SCL_modeled_RMCs",
        "SCL_modeled_PVs",
        "SCL_modeled_V",
        "SCL_modeled_FVCs_cumulative",
        "SCL_controlled_FVCs_cumulative",
        "SCL_controlled_FVCs_rates",
        "SCL_controlled_FVCs_Cin",
        "SCL_controlled_PVs",
        "SCL_modeled_FVCs_Cin",
        "SCL_modeled_BiologicalOde_rates",
        "SCL_modeled_FVCs_rates",
    ):
        assert axis in text, f"axis {axis} missing from rendered schema"


def test_format_reaction_schema_renders_bp_format_names():
    rhs_ode, controls = _mock_rhs_ode_and_controls()
    text = format_reaction_schema(rhs_ode, controls)

    for species in ("biomass", "glucose", "acetate"):
        assert species in text
    for fvc in ("base", "antifoam", "feed_glc"):
        assert fvc in text
    for pv in ("pH", "DO", "T"):
        assert pv in text
    for rate in ("q_biomass", "q_glucose", "q_acetate"):
        assert rate in text
    # Bolus FVCs are applied as discrete state jumps, not reaction-module inputs,
    # so neither the bolus name nor the wrapper-internal V_sample_acc surfaces here.
    assert "inducer_bolus" not in text
    assert "V_sample_acc" not in text


def test_format_reaction_schema_cin_followups_render_rows_cols():
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
        name_modeled_FVCs=(),
        name_controlled_FVCs=(),
        name_controlled_PVs=(),
        name_modeled_rates=("q_biomass",),
    )
    controls = SimpleNamespace(name_extras=("V_sample_acc",))
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
        self.weights = jnp.zeros((2, 2), dtype=jnp.float32)
        self.bias_frozen = jnp.zeros((2,), dtype=jnp.float32)

    def __call__(self, t, inputs):
        del t, inputs
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.zeros((0,), dtype=jnp.float32),
            SCL_modeled_FVCs_rates=jnp.zeros((0,), dtype=jnp.float32),
        )


def test_format_trainable_structure_works_from_inspect_module():
    module = _MixedTagsFixture()
    text = format_trainable_structure(module)

    assert "weights" in text
    assert "bias_frozen" in text
    assert "trainable" in text
    assert "frozen" in text
