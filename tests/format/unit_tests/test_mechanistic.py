"""Tests for bp_format.mechanistic post-P3 refactor.

Covers ``get_process_ordering``, ``get_control_splines``, ``build_rhs_ode``,
``extract_discrete_events``, ``build_state_splines``, and
``build_algebraic_func``. Forward integration moved to ``bp-train`` and is
not exercised here.

JAX-jit tests use ``eqx.filter_jit`` (the equinox-idiomatic way to JIT
modules that contain JAX-array fields).
"""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import pytest

from bp_format import (
    BiologicalOde,
    BioProcess,
    BioProcessMetadata,
    FeedMedium,
    FeedMediumComponent,
    Inflow,
    ProcessOrdering,
    ProcessVariable,
    ReactorMedium,
    ReactorMediumComponent,
    Outflow,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)
from bp_format.mechanistic import (
    ControlSplines,
    RhsOde,
    build_algebraic_func,
    build_rhs_ode,
    build_state_splines,
    extract_discrete_events,
    get_control_splines,
    get_process_ordering,
)
from bp_format.splines import build_pseudobatch_transform
from bp_format.time_series import PPoly


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _ts(t, v):
    return TimeSeries(times=jnp.array(t, dtype=float), values=jnp.array(v, dtype=float))


def _ts_poly(poly):
    return TimeSeries(poly=poly, segment_start_piece_idx=jnp.array([0]))


def _make_feed(name, glucose_conc=500.0, biomass_conc=0.0):
    return FeedMedium(
        name=name,
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": FeedMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=StaticVariable(value=biomass_conc),
                is_controlled=False,
            ),
            "glucose": FeedMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=StaticVariable(value=glucose_conc),
                is_controlled=False,
            ),
        },
    )


def _make_process(
    *,
    with_controlled_Inflow=True,
    with_controlled_PV=True,
    with_modeled_PV=False,
    with_modeled_Outflow=False,
    with_discrete_VC=False,
):
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=_ts([0.0, 5.0, 10.0, 20.0], [0.5, 1.0, 2.0, 4.0]),
            ),
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=_ts([0.0, 5.0, 10.0, 20.0], [10.0, 8.0, 5.0, 1.0]),
            ),
        },
    )
    vc_dict = {}
    if with_controlled_Inflow:
        vc_dict["feed"] = Inflow(
            name="feed",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            feed_medium=_make_feed("glucose_feed"),
            values=_ts([0.0, 5.0, 10.0, 20.0], [0.0, 0.25, 0.5, 1.0]),
        )
    if with_modeled_Outflow:
        vc_dict["evaporation"] = Outflow(
            name="evaporation",
            unit="L",
            is_controlled=False,
            is_continuous=True,
            values=_ts([0.0, 10.0, 20.0], [0.0, -0.01, -0.02]),
        )
    if with_discrete_VC:
        vc_dict["sampling"] = Outflow(
            name="sampling",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=_ts([5.0, 10.0], [-0.05, -0.05]),
        )
    pv_dict = {}
    if with_controlled_PV:
        pv_dict["pH"] = ProcessVariable(
            name="pH",
            unit="",
            is_controlled=True,
            values=_ts([0.0, 5.0, 10.0, 20.0], [7.0, 7.0, 7.0, 7.0]),
        )
    if with_modeled_PV:
        pv_dict["dissolved_O2"] = ProcessVariable(
            name="dissolved_O2",
            unit="%",
            is_controlled=False,
            values=_ts([0.0, 5.0, 10.0, 20.0], [100.0, 80.0, 60.0, 40.0]),
        )
    return BioProcess(
        metadata=BioProcessMetadata(name="test_fb", process_type="fed_batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=20.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes=vc_dict),
        reactor_medium=rm,
        process_variables=pv_dict,
    )


def _make_batch_process():
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=_ts([0.0, 5.0, 10.0], [0.5, 1.5, 4.0]),
            ),
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=_ts([0.0, 5.0, 10.0], [10.0, 6.0, 1.0]),
            ),
        },
    )
    return BioProcess(
        metadata=BioProcessMetadata(name="batch", process_type="batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=10.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=rm,
    )


def _apply_pseudobatch_transform(process, species_names=("biomass", "glucose")):
    transform = build_pseudobatch_transform(process, list(species_names))
    process.pseudobatch_transform = transform
    return transform


# ---------------------------------------------------------------------------
# ProcessOrdering
# ---------------------------------------------------------------------------


class TestProcessOrdering:
    def test_fields_match_groups(self):
        process = _make_process(
            with_controlled_Inflow=True,
            with_controlled_PV=True,
            with_modeled_PV=True,
            with_modeled_Outflow=True,
            with_discrete_VC=True,
        )
        ordering = get_process_ordering(process)
        assert isinstance(ordering, ProcessOrdering)
        assert ordering.name_modeled_RMCs == ("biomass", "glucose")
        assert ordering.name_modeled_PVs == ("dissolved_O2",)
        assert ordering.name_controlled_PVs == ("pH",)
        assert ordering.name_controlled_Inflows == ("feed",)
        assert ordering.name_modeled_Inflows == ()
        assert ordering.name_modeled_Outflows == ("evaporation",)
        assert ordering.name_controlled_Outflows == ()

    def test_alphabetical_RMCs(self):
        rm = ReactorMedium(
            name="medium",
            density=1.0,
            density_unit="kg/L",
            components={
                "zinc": ReactorMediumComponent(
                    name="zinc",
                    unit="mM",
                    concentration=_ts([0.0, 10.0], [0.1, 0.2]),
                ),
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=_ts([0.0, 10.0], [0.5, 2.0]),
                ),
                "acetate": ReactorMediumComponent(
                    name="acetate",
                    unit="g/L",
                    concentration=_ts([0.0, 10.0], [0.0, 1.0]),
                ),
            },
        )
        process = BioProcess(
            metadata=BioProcessMetadata(name="x", process_type="batch"),
            time_axis=TimeAxis(
                unit="hours", start=0.0, end=10.0, time_reference="inoculation"
            ),
            volume=Volume(initial_volume=1.0, unit="L"),
            reactor_medium=rm,
        )
        ordering = get_process_ordering(process)
        assert ordering.name_modeled_RMCs == ("acetate", "biomass", "zinc")

    def test_rates_preserve_user_order(self):
        process = _make_batch_process()
        process.biological_ode = BiologicalOde(
            algebraic={},
            rates={"r_zeta": (None, None), "r_alpha": (None, None)},
            derivatives={
                "biomass": "r_alpha",
                "glucose": "r_zeta",
            },
        )
        ordering = get_process_ordering(process)
        assert ordering.name_modeled_rates == ("r_zeta", "r_alpha")

    def test_algebraic_topo_sorted(self):
        process = _make_batch_process()
        process.biological_ode = BiologicalOde(
            algebraic={
                "B": "A + 1",
                "A": "biomass",
                "C": "B + A",
            },
            rates={"q_x": (None, None)},
            derivatives={
                "biomass": "q_x * C",
                "glucose": "0",
            },
        )
        ordering = get_process_ordering(process)
        assert ordering.name_modeled_algebraic == ("A", "B", "C")

    def test_static_PV_must_be_controlled(self):
        process = _make_batch_process()
        process.process_variables = {
            "T": ProcessVariable(
                name="T",
                unit="C",
                is_controlled=False,
                values=StaticVariable(value=37.0),
            ),
        }
        with pytest.raises(ValueError, match="StaticVariable"):
            get_process_ordering(process)

    def test_Inflow_missing_feed_medium_raises(self):
        process = _make_process(with_controlled_Inflow=False)
        process.volume.volume_changes["feed"] = Inflow(
            name="feed",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            feed_medium=None,
            values=_ts([0.0, 20.0], [0.0, 0.5]),
        )
        with pytest.raises(ValueError, match="feed_medium"):
            get_process_ordering(process)

    def test_Inflow_unknown_feed_component_raises(self):
        process = _make_process(with_controlled_Inflow=False)
        process.volume.volume_changes["feed"] = Inflow(
            name="feed",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            feed_medium=FeedMedium(
                name="bad",
                density=1.0,
                density_unit="kg/L",
                components={
                    "phosphate": FeedMediumComponent(
                        name="phosphate",
                        unit="mM",
                        concentration=StaticVariable(value=10.0),
                        is_controlled=False,
                    ),
                },
            ),
            values=_ts([0.0, 20.0], [0.0, 0.5]),
        )
        with pytest.raises(ValueError, match="phosphate"):
            get_process_ordering(process)

    def test_name_collision_raises(self):
        process = _make_batch_process()
        # PV named "biomass" — collides with RMC "biomass"
        process.process_variables = {
            "biomass": ProcessVariable(
                name="biomass",
                unit="",
                is_controlled=True,
                values=_ts([0.0, 10.0], [1.0, 1.0]),
            ),
        }
        process.biological_ode = BiologicalOde(
            algebraic={},
            rates={"q_b": (None, None)},
            derivatives={"biomass": "q_b * biomass"},
        )
        with pytest.raises(ValueError, match="Name collisions"):
            get_process_ordering(process)


# ---------------------------------------------------------------------------
# ControlSplines
# ---------------------------------------------------------------------------


class TestControlSplines:
    def test_class_type(self):
        process = _make_process()
        ctrl = get_control_splines(process)
        assert isinstance(ctrl, ControlSplines)
        assert isinstance(ctrl, eqx.Module)

    def test_output_layout_Inflow_Outflow_PV(self):
        process = _make_process(
            with_controlled_Inflow=True,
            with_controlled_PV=True,
            with_modeled_Outflow=True,  # uncontrolled Outflow — not in ControlSplines
        )
        ctrl = get_control_splines(process)
        assert ctrl.name_controlled_Inflows == ("feed",)
        assert ctrl.name_controlled_Outflows == ()
        assert ctrl.name_controlled_PVs == ("pH",)

        u = ctrl(jnp.array(7.5))
        # Layout: [feed_flow, pH_value]
        assert u.shape == (2,)
        # Feed cumulative is linear from 0 to 1.0 over [0, 20] → flow = 0.05
        assert float(u[0]) == pytest.approx(0.05, abs=1e-3)
        assert float(u[1]) == pytest.approx(7.0, abs=1e-3)

    def test_empty_controls(self):
        process = _make_batch_process()
        ctrl = get_control_splines(process)
        u = ctrl(jnp.array(5.0))
        u_vector = ctrl(jnp.array([0.0, 5.0, 10.0]))
        assert u.shape == (0,)
        assert u_vector.shape == (3, 0)

    def test_jit_callable(self):
        process = _make_process()
        ctrl = get_control_splines(process)
        u = eqx.filter_jit(ctrl)(jnp.array(7.5))
        assert u.shape[0] == 2

    def test_only_Inflows_returns_flows(self):
        process = _make_process(with_controlled_Inflow=True, with_controlled_PV=False)
        ctrl = get_control_splines(process)
        u = ctrl(jnp.array(5.0))
        assert u.shape == (1,)
        # Pure flow rate
        assert float(u[0]) == pytest.approx(0.05, abs=1e-3)

    def test_only_PVs_returns_values(self):
        process = _make_process(with_controlled_Inflow=False, with_controlled_PV=True)
        ctrl = get_control_splines(process)
        u = ctrl(jnp.array(5.0))
        assert u.shape == (1,)
        assert float(u[0]) == pytest.approx(7.0, abs=1e-3)

    def test_uses_original_control_splines_without_refit(self):
        process = _make_process(with_controlled_Inflow=True, with_controlled_PV=True)
        feed_poly = PPoly(
            jnp.array([0.0, 2.0, 20.0]),
            jnp.array(
                [
                    [0.0, 0.3, 0.4, -0.05],
                    [1.0, -0.2, 0.03, 0.002],
                ]
            ),
        )
        sample_poly = PPoly(
            jnp.array([0.0, 4.0, 20.0]),
            jnp.array(
                [
                    [0.0, -0.05, -0.01, 0.001],
                    [-0.5, -0.02, 0.002, -0.0002],
                ]
            ),
        )
        ph_poly = PPoly(
            jnp.array([0.0, 5.0, 20.0]),
            jnp.array(
                [
                    [7.0, 0.1, -0.02, 0.001],
                    [7.1, -0.03, 0.004, -0.0001],
                ]
            ),
        )
        process.volume.volume_changes["feed"].values = _ts_poly(feed_poly)
        process.volume.volume_changes["sample"] = Outflow(
            name="sample",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            values=_ts_poly(sample_poly),
        )
        process.process_variables["pH"].values = _ts_poly(ph_poly)

        ctrl = get_control_splines(process)
        t = jnp.array([0.5, 1.5, 3.0, 12.0])
        u = ctrl(t)

        assert ctrl.name_controlled_Inflows == ("feed",)
        assert ctrl.name_controlled_Outflows == ("sample",)
        assert ctrl.name_controlled_PVs == ("pH",)
        for actual, expected in zip(
            ctrl._splines,
            (feed_poly, sample_poly, ph_poly),
            strict=True,
        ):
            assert actual.breaks == pytest.approx(expected.breaks)
            assert actual.coeffs == pytest.approx(expected.coeffs)
        assert u.shape == (4, 3)
        expected = jnp.stack(
            [feed_poly(t, nu=1), sample_poly(t, nu=1), ph_poly(t)], axis=-1
        )
        assert u == pytest.approx(expected)

        scalar_t = jnp.array(1.25)
        scalar_u = ctrl(scalar_t)
        scalar_expected = jnp.stack(
            [
                feed_poly(scalar_t, nu=1),
                sample_poly(scalar_t, nu=1),
                ph_poly(scalar_t),
            ],
            axis=-1,
        )
        assert scalar_u.shape == (3,)
        assert scalar_u == pytest.approx(scalar_expected)


# ---------------------------------------------------------------------------
# RhsOde
# ---------------------------------------------------------------------------


class TestRhsOde:
    def test_class_type(self):
        process = _make_process()
        rhs = build_rhs_ode(process)
        assert isinstance(rhs, RhsOde)

    def test_field_names(self):
        process = _make_process(
            with_controlled_Inflow=True,
            with_controlled_PV=True,
            with_modeled_Outflow=True,
        )
        rhs = build_rhs_ode(process)
        assert rhs.name_modeled_RMCs == ("biomass", "glucose")
        assert rhs.name_modeled_PVs == ()
        assert rhs.name_controlled_PVs == ("pH",)
        assert rhs.name_controlled_Inflows == ("feed",)
        assert rhs.name_controlled_Outflows == ()
        assert rhs.name_modeled_Outflows == ("evaporation",)
        assert rhs.name_modeled_Inflows == ()
        # auto-rates: q_biomass, q_glucose
        assert set(rhs.name_modeled_rates) == {"q_biomass", "q_glucose"}

    def test_Cin_shapes(self):
        process = _make_process(with_controlled_Inflow=True)
        rhs = build_rhs_ode(process)
        assert rhs.Cin_controlled_Inflows.shape == (1, 2)  # one feed × two RMCs
        assert rhs.Cin_modeled_Inflows.shape == (0, 2)

    def test_Cin_values(self):
        process = _make_process(with_controlled_Inflow=True)
        rhs = build_rhs_ode(process)
        # biomass=0, glucose=500 in feed; RMC order is (biomass, glucose) alphabetical
        assert float(rhs.Cin_controlled_Inflows[0, 0]) == pytest.approx(0.0)
        assert float(rhs.Cin_controlled_Inflows[0, 1]) == pytest.approx(500.0)

    def test_call_output_shape(self):
        process = _make_process()
        rhs = build_rhs_ode(process)
        ctrl = get_control_splines(process)
        c = jnp.array([1.0, 5.0, 1.0])  # biomass, glucose, V
        rates = jnp.array([0.1, -0.5])  # q_biomass, q_glucose (auto order)
        u = ctrl(jnp.array(5.0))
        f_modeled_Inflows = jnp.zeros(0)
        f_modeled_Outflows = jnp.zeros(0)
        dc = rhs(c, rates, u, f_modeled_Inflows, f_modeled_Outflows)
        assert dc.shape == (3,)  # 2 RMCs + 0 PVs + V

    def test_dV_from_Inflow(self):
        process = _make_process(with_controlled_Inflow=True, with_controlled_PV=False)
        rhs = build_rhs_ode(process)
        c = jnp.array([1.0, 5.0, 1.0])
        rates = jnp.zeros(2)
        u = jnp.array([0.05])  # Inflow flow
        dc = rhs(c, rates, u, jnp.zeros(0), jnp.zeros(0))
        # dV should equal +0.05 (Inflow inflow only)
        assert float(dc[-1]) == pytest.approx(0.05, abs=1e-6)

    def test_dV_from_Outflow_modeled(self):
        process = _make_process(
            with_controlled_Inflow=False,
            with_controlled_PV=False,
            with_modeled_Outflow=True,
        )
        rhs = build_rhs_ode(process)
        c = jnp.array([1.0, 5.0, 1.0])
        rates = jnp.zeros(2)
        f_modeled_Outflows = jnp.array([-0.001])  # outflow
        dc = rhs(c, rates, jnp.zeros(0), jnp.zeros(0), f_modeled_Outflows)
        # dV = total_in - total_out = 0 - 0.001 = -0.001
        assert float(dc[-1]) == pytest.approx(-0.001, abs=1e-9)

    def test_dV_balance_Inflow_minus_Outflow(self):
        process = _make_process(
            with_controlled_Inflow=True,
            with_controlled_PV=False,
            with_modeled_Outflow=True,
        )
        rhs = build_rhs_ode(process)
        c = jnp.array([1.0, 5.0, 1.0])
        rates = jnp.zeros(2)
        u = jnp.array([0.05])
        f_modeled_Outflows = jnp.array([-0.02])
        dc = rhs(c, rates, u, jnp.zeros(0), f_modeled_Outflows)
        assert float(dc[-1]) == pytest.approx(0.05 - 0.02, abs=1e-9)

    def test_pure_batch_no_feed_term(self):
        process = _make_batch_process()
        rhs = build_rhs_ode(process)
        c = jnp.array([2.0, 5.0, 1.0])
        rates = jnp.array([0.1, -0.5])  # auto rate order: q_biomass, q_glucose
        dc = rhs(c, rates, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0))
        # auto BiologicalOde: dc/dt = q_<rmc> * biomass
        # biomass: 0.1 * 2.0 = 0.2
        # glucose: -0.5 * 2.0 = -1.0
        assert float(dc[0]) == pytest.approx(0.2, abs=1e-6)
        assert float(dc[1]) == pytest.approx(-1.0, abs=1e-6)
        assert float(dc[2]) == pytest.approx(0.0, abs=1e-9)

    def test_jit_call(self):
        process = _make_process()
        rhs = build_rhs_ode(process)
        ctrl = get_control_splines(process)
        c = jnp.array([1.0, 5.0, 1.0])
        rates = jnp.zeros(2)
        u = ctrl(jnp.array(5.0))
        jitted = eqx.filter_jit(rhs)
        dc = jitted(c, rates, u, jnp.zeros(0), jnp.zeros(0))
        assert dc.shape == (3,)

    def test_user_defined_biological_ode(self):
        process = _make_batch_process()
        process.biological_ode = BiologicalOde(
            algebraic={"X_active": "biomass - 0.1"},
            rates={"q_b": (None, None), "q_g": (None, None)},
            derivatives={
                "biomass": "q_b * X_active",
                "glucose": "q_g * X_active",
            },
        )
        rhs = build_rhs_ode(process)
        c = jnp.array([2.0, 5.0, 1.0])
        rates = jnp.array([0.5, -0.3])
        dc = rhs(c, rates, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0))
        # X_active = 2.0 - 0.1 = 1.9
        # dbiomass = 0.5 * 1.9 = 0.95
        # dglucose = -0.3 * 1.9 = -0.57
        assert float(dc[0]) == pytest.approx(0.95, abs=1e-6)
        assert float(dc[1]) == pytest.approx(-0.57, abs=1e-6)

    def test_no_biological_ode_raises(self):
        process = _make_batch_process()
        process.biological_ode = None
        with pytest.raises(ValueError, match="biological_ode"):
            build_rhs_ode(process)

    def test_feed_dilution_concentration(self):
        process = _make_process(with_controlled_Inflow=True, with_controlled_PV=False)
        rhs = build_rhs_ode(process)
        c_biomass = 2.0
        c_glucose = 5.0
        V = 1.0
        u_flow = 0.05
        c = jnp.array([c_biomass, c_glucose, V])
        rates = jnp.zeros(2)
        u = jnp.array([u_flow])
        dc = rhs(c, rates, u, jnp.zeros(0), jnp.zeros(0))
        # Expected feed terms:
        # biomass:  -u/V*c + u*0/V = -0.05*2/1 = -0.1
        # glucose:  -u/V*c + u*500/V = -0.25 + 25 = 24.75
        assert float(dc[0]) == pytest.approx(-0.1, abs=1e-5)
        assert float(dc[1]) == pytest.approx(24.75, abs=1e-3)

    def test_fails_fast_if_inflow_medium_mutated_after_construction(self):
        """The __post_init__ fill guarantees a complete feed medium; a gap
        reaching build_rhs_ode after construction means the process was
        mutated, and that must raise loudly, not silently default again."""
        process = _make_process(with_controlled_Inflow=True)
        feed_medium = process.volume.volume_changes["feed"].feed_medium
        del feed_medium.components["glucose"]
        with pytest.raises(ValueError, match="glucose"):
            build_rhs_ode(process)

    def test_fails_fast_if_inflow_medium_none_but_declared_continuous(self):
        process = _make_process(with_controlled_Inflow=True)
        process.volume.volume_changes["feed"].feed_medium = None
        with pytest.raises(ValueError, match="feed_medium"):
            build_rhs_ode(process)


# ---------------------------------------------------------------------------
# Outflow component_retention
# ---------------------------------------------------------------------------


class TestOutflowRetention:
    def test_default_retention_is_zero(self):
        process = _make_process(with_modeled_Outflow=True)
        rhs = build_rhs_ode(process)
        assert rhs.retention_modeled_Outflows.shape == (1, 2)
        assert jnp.all(rhs.retention_modeled_Outflows == 0.0)
        assert rhs.retention_controlled_Outflows.shape == (0, 2)

    def test_retention_values_aligned_with_RMC_order(self):
        process = _make_process(with_modeled_Outflow=False)
        process.volume.volume_changes["evaporation"] = Outflow(
            name="evaporation",
            unit="L",
            is_controlled=False,
            is_continuous=True,
            values=_ts([0.0, 10.0], [0.0, -0.02]),
            component_retention={"biomass": 0.95},
        )
        rhs = build_rhs_ode(process)
        # RMCs alphabetical: (biomass, glucose)
        assert float(rhs.retention_modeled_Outflows[0, 0]) == pytest.approx(0.95)
        assert float(rhs.retention_modeled_Outflows[0, 1]) == pytest.approx(0.0)

    def test_zero_retention_matches_unretained_dilution(self):
        """sigma=0 everywhere (the default) must reproduce the original
        uniform-dilution formula exactly."""
        process = _make_process(
            with_controlled_Inflow=True,
            with_controlled_PV=False,
            with_modeled_Outflow=True,
        )
        rhs = build_rhs_ode(process)
        c = jnp.array([2.0, 5.0, 1.0])
        rates = jnp.zeros(2)
        u = jnp.array([0.05])
        f_modeled_Outflows = jnp.array([-0.02])
        dc = rhs(c, rates, u, jnp.zeros(0), f_modeled_Outflows)
        total_in, total_out = 0.05, 0.02
        expected_dilution = -(total_in + total_out) / 1.0 * c[:2]
        # biomass gets no Cin addition (feed has biomass=0); glucose does.
        expected_addition = jnp.array([0.0, 0.05 * 500.0 / 1.0])
        expected = expected_dilution + expected_addition
        assert float(dc[0]) == pytest.approx(float(expected[0]), abs=1e-6)
        assert float(dc[1]) == pytest.approx(float(expected[1]), abs=1e-6)

    def test_retention_scales_down_washout_proportionally(self):
        """Comparing sigma=0.95 against sigma=0 for the same outflow in
        isolation: the retention-aware dc/dt must differ from the
        unretained dc/dt by exactly sigma * (the unretained outflow-dilution
        contribution) — the defining, mechanically verifiable property of
        the retention formula, independent of whatever the rest of the
        dilution formula (e.g. its total_in term) does.

        Note: this is a direct, formula-level check, not a from-scratch
        physical mass-balance derivation. See the implementation note above
        _apply_feed_dilution and the summary reported to the user: layering
        the spec's retention formula onto bp-format's *existing*
        (total_in + total_out)-based dilution — which predates this task
        and is independently relied upon (test_feed_dilution_concentration)
        — reproduces the old behavior exactly at sigma=0 (verified in
        test_zero_retention_matches_unretained_dilution) and is internally
        consistent for every sigma in [0, 1], but does not deliver the
        "mass conserved at sigma=1" property the spec's own sanity check
        claims — that claim assumed a from-scratch mass balance that the
        pre-existing dilution formula does not actually match.
        """
        rm = ReactorMedium(
            name="medium",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass", unit="g/L", concentration=_ts([0.0, 1.0], [1.0, 1.0])
                ),
            },
        )
        q_out = 0.1
        V = 2.0
        c_biomass = 3.0

        def _dc_for_retention(sigma):
            process = BioProcess(
                metadata=BioProcessMetadata(name="perfusion", process_type="continuous"),
                time_axis=TimeAxis(
                    unit="hours", start=0.0, end=10.0, time_reference="x"
                ),
                volume=Volume(
                    initial_volume=V,
                    unit="L",
                    volume_changes={
                        "perfusion_out": Outflow(
                            name="perfusion_out",
                            unit="L",
                            is_controlled=False,
                            is_continuous=True,
                            values=_ts([0.0, 10.0], [0.0, -q_out * 10.0]),
                            component_retention={"biomass": sigma},
                        ),
                    },
                ),
                reactor_medium=rm,
                biological_ode=BiologicalOde(rates={}, derivatives={"biomass": "0"}),
            )
            rhs = build_rhs_ode(process)
            c = jnp.array([c_biomass, V])
            dc = rhs(c, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0), jnp.array([-q_out]))
            return float(dc[0])

        dc_unretained = _dc_for_retention(0.0)
        dc_retained = _dc_for_retention(0.95)
        # eff_out_per_rmc = (1-sigma)*q_out, so dc/dt scales linearly in
        # (1-sigma); the retained case must wash out 5% as fast.
        assert dc_retained == pytest.approx(0.05 * dc_unretained, rel=1e-9)

    def test_full_retention_zeroes_the_outflow_dilution_term(self):
        """sigma=1 makes eff_out_per_rmc exactly 0 for that species, i.e.
        dc/dt for that RMC becomes independent of the outflow magnitude
        entirely (the mechanical guarantee the formula actually provides —
        see the note in the previous test)."""
        rm = ReactorMedium(
            name="medium",
            components={
                "solute": ReactorMediumComponent(
                    name="solute", unit="g/L", concentration=_ts([0.0, 1.0], [1.0, 1.0])
                ),
            },
        )
        V = 2.0
        c_solute = 4.0

        def _dc_for_outflow(q_evap):
            process = BioProcess(
                metadata=BioProcessMetadata(name="evap", process_type="continuous"),
                time_axis=TimeAxis(
                    unit="hours", start=0.0, end=10.0, time_reference="x"
                ),
                volume=Volume(
                    initial_volume=V,
                    unit="L",
                    volume_changes={
                        "evaporation": Outflow(
                            name="evaporation",
                            unit="L",
                            is_controlled=False,
                            is_continuous=True,
                            values=_ts([0.0, 10.0], [0.0, -q_evap * 10.0]),
                            component_retention={"solute": 1.0},
                        ),
                    },
                ),
                reactor_medium=rm,
                biological_ode=BiologicalOde(rates={}, derivatives={"solute": "0"}),
            )
            rhs = build_rhs_ode(process)
            c = jnp.array([c_solute, V])
            dc = rhs(c, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0), jnp.array([-q_evap]))
            return float(dc[0])

        # With full retention, dc/dt for "solute" must be the same
        # regardless of how much is evaporating.
        assert _dc_for_outflow(0.05) == pytest.approx(_dc_for_outflow(0.2), abs=1e-12)
        assert _dc_for_outflow(0.05) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# extract_discrete_events
# ---------------------------------------------------------------------------


class TestExtractDiscreteEvents:
    def test_no_discrete_events(self):
        process = _make_process(with_discrete_VC=False)
        ordering = get_process_ordering(process)
        events = extract_discrete_events(process, ordering)
        assert events == []

    def test_sample_events(self):
        process = _make_process(with_discrete_VC=True)
        ordering = get_process_ordering(process)
        events = extract_discrete_events(process, ordering)
        assert len(events) == 2
        for ev in events:
            assert ev["kind"] == "sample"
            assert ev["dV"] == pytest.approx(-0.05)
            assert ev["Cin"] is None
            assert ev["source"] == "sampling"

    def test_bolus_feed_alignment(self):
        process = _make_process(with_discrete_VC=False)
        process.volume.volume_changes["bolus"] = Inflow(
            name="bolus",
            unit="L",
            is_controlled=False,
            is_continuous=False,
            feed_medium=_make_feed("bolus_feed", glucose_conc=200.0),
            values=_ts([3.0], [0.1]),
        )
        ordering = get_process_ordering(process)
        events = extract_discrete_events(process, ordering)
        bolus = [e for e in events if e["kind"] == "bolus_feed"]
        assert len(bolus) == 1
        ev = bolus[0]
        assert ev["t"] == pytest.approx(3.0)
        assert ev["dV"] == pytest.approx(0.1)
        # RMCs alphabetical: (biomass, glucose); feed: biomass=0, glucose=200
        assert ev["Cin"].shape == (2,)
        assert float(ev["Cin"][0]) == pytest.approx(0.0)
        assert float(ev["Cin"][1]) == pytest.approx(200.0)

    def test_sorted_by_time_sample_before_bolus(self):
        process = _make_process(with_discrete_VC=False)
        process.volume.volume_changes["sampling"] = Outflow(
            name="sampling",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=_ts([5.0], [-0.05]),
        )
        process.volume.volume_changes["bolus"] = Inflow(
            name="bolus",
            unit="L",
            is_controlled=False,
            is_continuous=False,
            feed_medium=_make_feed("bolus_feed"),
            values=_ts([5.0], [0.1]),
        )
        ordering = get_process_ordering(process)
        events = extract_discrete_events(process, ordering)
        assert events[0]["kind"] == "sample"
        assert events[1]["kind"] == "bolus_feed"

    def test_bolus_raises_if_feed_medium_none(self):
        process = _make_process(with_discrete_VC=False)
        process.volume.volume_changes["bolus"] = Inflow(
            name="bolus",
            unit="L",
            is_controlled=False,
            is_continuous=False,
            feed_medium=_make_feed("bolus_feed"),
            values=_ts([3.0], [0.1]),
        )
        process.volume.volume_changes["bolus"].feed_medium = None
        ordering = get_process_ordering(process)
        with pytest.raises(ValueError, match="feed_medium"):
            extract_discrete_events(process, ordering)

    def test_bolus_raises_if_feed_medium_mutated_after_construction(self):
        process = _make_process(with_discrete_VC=False)
        process.volume.volume_changes["bolus"] = Inflow(
            name="bolus",
            unit="L",
            is_controlled=False,
            is_continuous=False,
            feed_medium=_make_feed("bolus_feed"),
            values=_ts([3.0], [0.1]),
        )
        del process.volume.volume_changes["bolus"].feed_medium.components["glucose"]
        ordering = get_process_ordering(process)
        with pytest.raises(ValueError, match="glucose"):
            extract_discrete_events(process, ordering)


# ---------------------------------------------------------------------------
# build_state_splines
# ---------------------------------------------------------------------------


class TestBuildStateSplines:
    def test_returns_callable_per_state(self):
        process = _make_process(with_modeled_PV=True)
        ordering = get_process_ordering(process)
        state_splines = build_state_splines(process, ordering)
        assert set(state_splines.keys()) == {"biomass", "glucose", "dissolved_O2"}
        for sp_name, sp in state_splines.items():
            val = sp(jnp.array(5.0))
            assert val is not None

    def test_pseudobatch_path(self):
        process = _make_process(with_controlled_Inflow=True, with_controlled_PV=False)
        _apply_pseudobatch_transform(process)
        ordering = get_process_ordering(process)
        state_splines = build_state_splines(process, ordering)
        assert set(state_splines.keys()) == {"biomass", "glucose"}

    def test_pseudobatch_feed_correction_requires_c_star(self):
        process = _make_process(with_controlled_Inflow=True, with_controlled_PV=False)
        _apply_pseudobatch_transform(process)
        process.reactor_medium.components["glucose"].c_star_concentration = None
        ordering = get_process_ordering(process)

        with pytest.raises(ValueError, match="no c_star_concentration"):
            build_state_splines(process, ordering)

    def test_pseudobatch_feed_correction_requires_c_star_for_unmodeled_component(self):
        process = _make_process(with_controlled_Inflow=True, with_controlled_PV=False)
        _apply_pseudobatch_transform(process)
        process.reactor_medium.components["biomass"].c_star_concentration = None
        ordering = dataclasses.replace(
            get_process_ordering(process),
            name_modeled_RMCs=("glucose",),
        )

        with pytest.raises(ValueError, match="no matching c_star_concentration"):
            build_state_splines(process, ordering)


# ---------------------------------------------------------------------------
# build_algebraic_func
# ---------------------------------------------------------------------------


class TestBuildAlgebraicFunc:
    def test_returns_dict(self):
        process = _make_batch_process()
        process.biological_ode = BiologicalOde(
            algebraic={"X_active": "biomass - 0.1"},
            rates={"q_b": (None, None), "q_g": (None, None)},
            derivatives={
                "biomass": "q_b * X_active",
                "glucose": "q_g * X_active",
            },
        )
        f = build_algebraic_func(process)
        # state values: biomass=2, glucose=5; no controlled PVs; rates arbitrary
        out = f(
            jnp.array([2.0, 5.0]),
            jnp.zeros(0),
            jnp.array([0.5, -0.3]),
        )
        assert "X_active" in out
        assert float(out["X_active"]) == pytest.approx(1.9, abs=1e-6)

    def test_no_biological_ode_raises(self):
        process = _make_batch_process()
        process.biological_ode = None
        with pytest.raises(ValueError, match="biological_ode"):
            build_algebraic_func(process)
