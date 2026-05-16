"""Tests for bp_format.mechanistic post-P3 refactor.

Covers ``get_process_ordering``, ``get_control_splines``, ``get_rhs_ode``,
``extract_discrete_events``, ``build_state_splines``, and
``build_algebraic_func``. Forward integration moved to ``bp-train`` and is
not exercised here.

JAX-jit tests use ``eqx.filter_jit`` (the equinox-idiomatic way to JIT
modules that contain JAX-array fields).
"""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from bp_format import (
    BiologicalOde,
    BioProcess,
    BioProcessMetadata,
    FeedMedium,
    FeedMediumComponent,
    FeedVolumeChange,
    ProcessOrdering,
    ProcessVariable,
    ReactorMedium,
    ReactorMediumComponent,
    SampleVolumeChange,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)
from bp_format.mechanistic import (
    ControlSplines,
    RhsOde,
    build_algebraic_func,
    build_state_splines,
    extract_discrete_events,
    get_control_splines,
    get_process_ordering,
    get_rhs_ode,
)
from bp_format.splines import (
    build_pseudobatch_transform,
    make_cubic_ppoly,
)
from bp_format.time_series import PPoly


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _ts(t, v):
    return TimeSeries(times=jnp.array(t, dtype=float), values=jnp.array(v, dtype=float))


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
    with_controlled_FVC=True,
    with_controlled_PV=True,
    with_modeled_PV=False,
    with_modeled_SVC=False,
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
    if with_controlled_FVC:
        vc_dict["feed"] = FeedVolumeChange(
            name="feed",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            feed_medium=_make_feed("glucose_feed"),
            values=_ts([0.0, 5.0, 10.0, 20.0], [0.0, 0.25, 0.5, 1.0]),
        )
    if with_modeled_SVC:
        vc_dict["evaporation"] = SampleVolumeChange(
            name="evaporation",
            unit="L",
            is_controlled=False,
            is_continuous=True,
            values=_ts([0.0, 10.0, 20.0], [0.0, -0.01, -0.02]),
        )
    if with_discrete_VC:
        vc_dict["sampling"] = SampleVolumeChange(
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
# make_cubic_ppoly (passthrough sanity)
# ---------------------------------------------------------------------------


class TestMakeCubicPPoly:
    def test_returns_owned_ppoly(self):
        sp = make_cubic_ppoly(
            np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 4.0, 9.0])
        )
        assert isinstance(sp, PPoly)

    def test_eval_at_knots(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        v = np.array([0.0, 1.0, 4.0, 9.0])
        sp = make_cubic_ppoly(t, v)
        for ti, vi in zip(t, v):
            assert float(sp(ti)) == pytest.approx(float(vi), abs=1e-4)

    def test_derivative_of_linear_is_slope(self):
        sp = make_cubic_ppoly(np.array([0.0, 10.0]), np.array([0.0, 5.0]))
        assert float(sp.derivative()(5.0)) == pytest.approx(0.5, rel=1e-4)


# ---------------------------------------------------------------------------
# ProcessOrdering
# ---------------------------------------------------------------------------


class TestProcessOrdering:
    def test_fields_match_groups(self):
        process = _make_process(
            with_controlled_FVC=True,
            with_controlled_PV=True,
            with_modeled_PV=True,
            with_modeled_SVC=True,
            with_discrete_VC=True,
        )
        ordering = get_process_ordering(process)
        assert isinstance(ordering, ProcessOrdering)
        assert ordering.name_modeled_RMCs == ("biomass", "glucose")
        assert ordering.name_modeled_PVs == ("dissolved_O2",)
        assert ordering.name_controlled_PVs == ("pH",)
        assert ordering.name_controlled_FVCs == ("feed",)
        assert ordering.name_modeled_FVCs == ()
        assert ordering.name_modeled_SVCs == ("evaporation",)
        assert ordering.name_controlled_SVCs == ()

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

    def test_FVC_missing_feed_medium_raises(self):
        process = _make_process(with_controlled_FVC=False)
        process.volume.volume_changes["feed"] = FeedVolumeChange(
            name="feed",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            feed_medium=None,
            values=_ts([0.0, 20.0], [0.0, 0.5]),
        )
        with pytest.raises(ValueError, match="feed_medium"):
            get_process_ordering(process)

    def test_FVC_unknown_feed_component_raises(self):
        process = _make_process(with_controlled_FVC=False)
        process.volume.volume_changes["feed"] = FeedVolumeChange(
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

    def test_output_layout_FVC_SVC_PV(self):
        process = _make_process(
            with_controlled_FVC=True,
            with_controlled_PV=True,
            with_modeled_SVC=True,  # uncontrolled SVC — not in ControlSplines
        )
        ctrl = get_control_splines(process)
        assert ctrl.name_controlled_FVCs == ("feed",)
        assert ctrl.name_controlled_SVCs == ()
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

    def test_only_FVCs_returns_flows(self):
        process = _make_process(with_controlled_FVC=True, with_controlled_PV=False)
        ctrl = get_control_splines(process)
        u = ctrl(jnp.array(5.0))
        assert u.shape == (1,)
        # Pure flow rate
        assert float(u[0]) == pytest.approx(0.05, abs=1e-3)

    def test_only_PVs_returns_values(self):
        process = _make_process(with_controlled_FVC=False, with_controlled_PV=True)
        ctrl = get_control_splines(process)
        u = ctrl(jnp.array(5.0))
        assert u.shape == (1,)
        assert float(u[0]) == pytest.approx(7.0, abs=1e-3)


# ---------------------------------------------------------------------------
# RhsOde
# ---------------------------------------------------------------------------


class TestRhsOde:
    def test_class_type(self):
        process = _make_process()
        rhs = get_rhs_ode(process)
        assert isinstance(rhs, RhsOde)

    def test_field_names(self):
        process = _make_process(
            with_controlled_FVC=True,
            with_controlled_PV=True,
            with_modeled_SVC=True,
        )
        rhs = get_rhs_ode(process)
        assert rhs.name_modeled_RMCs == ("biomass", "glucose")
        assert rhs.name_modeled_PVs == ()
        assert rhs.name_controlled_PVs == ("pH",)
        assert rhs.name_controlled_FVCs == ("feed",)
        assert rhs.name_controlled_SVCs == ()
        assert rhs.name_modeled_SVCs == ("evaporation",)
        assert rhs.name_modeled_FVCs == ()
        # auto-rates: q_biomass, q_glucose
        assert set(rhs.name_modeled_rates) == {"q_biomass", "q_glucose"}

    def test_Cin_shapes(self):
        process = _make_process(with_controlled_FVC=True)
        rhs = get_rhs_ode(process)
        assert rhs.Cin_controlled_FVCs.shape == (1, 2)  # one feed × two RMCs
        assert rhs.Cin_modeled_FVCs.shape == (0, 2)

    def test_Cin_values(self):
        process = _make_process(with_controlled_FVC=True)
        rhs = get_rhs_ode(process)
        # biomass=0, glucose=500 in feed; RMC order is (biomass, glucose) alphabetical
        assert float(rhs.Cin_controlled_FVCs[0, 0]) == pytest.approx(0.0)
        assert float(rhs.Cin_controlled_FVCs[0, 1]) == pytest.approx(500.0)

    def test_call_output_shape(self):
        process = _make_process()
        rhs = get_rhs_ode(process)
        ctrl = get_control_splines(process)
        c = jnp.array([1.0, 5.0, 1.0])  # biomass, glucose, V
        rates = jnp.array([0.1, -0.5])  # q_biomass, q_glucose (auto order)
        u = ctrl(jnp.array(5.0))
        f_modeled_FVCs = jnp.zeros(0)
        f_modeled_SVCs = jnp.zeros(0)
        dc = rhs(c, rates, u, f_modeled_FVCs, f_modeled_SVCs)
        assert dc.shape == (3,)  # 2 RMCs + 0 PVs + V

    def test_dV_from_FVC(self):
        process = _make_process(with_controlled_FVC=True, with_controlled_PV=False)
        rhs = get_rhs_ode(process)
        c = jnp.array([1.0, 5.0, 1.0])
        rates = jnp.zeros(2)
        u = jnp.array([0.05])  # FVC flow
        dc = rhs(c, rates, u, jnp.zeros(0), jnp.zeros(0))
        # dV should equal +0.05 (FVC inflow only)
        assert float(dc[-1]) == pytest.approx(0.05, abs=1e-6)

    def test_dV_from_SVC_modeled(self):
        process = _make_process(
            with_controlled_FVC=False,
            with_controlled_PV=False,
            with_modeled_SVC=True,
        )
        rhs = get_rhs_ode(process)
        c = jnp.array([1.0, 5.0, 1.0])
        rates = jnp.zeros(2)
        f_modeled_SVCs = jnp.array([-0.001])  # outflow
        dc = rhs(c, rates, jnp.zeros(0), jnp.zeros(0), f_modeled_SVCs)
        # dV = total_in - total_out = 0 - 0.001 = -0.001
        assert float(dc[-1]) == pytest.approx(-0.001, abs=1e-9)

    def test_dV_balance_FVC_minus_SVC(self):
        process = _make_process(
            with_controlled_FVC=True,
            with_controlled_PV=False,
            with_modeled_SVC=True,
        )
        rhs = get_rhs_ode(process)
        c = jnp.array([1.0, 5.0, 1.0])
        rates = jnp.zeros(2)
        u = jnp.array([0.05])
        f_modeled_SVCs = jnp.array([-0.02])
        dc = rhs(c, rates, u, jnp.zeros(0), f_modeled_SVCs)
        assert float(dc[-1]) == pytest.approx(0.05 - 0.02, abs=1e-9)

    def test_pure_batch_no_feed_term(self):
        process = _make_batch_process()
        rhs = get_rhs_ode(process)
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
        rhs = get_rhs_ode(process)
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
        rhs = get_rhs_ode(process)
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
            get_rhs_ode(process)

    def test_feed_dilution_concentration(self):
        process = _make_process(with_controlled_FVC=True, with_controlled_PV=False)
        rhs = get_rhs_ode(process)
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
        process.volume.volume_changes["bolus"] = FeedVolumeChange(
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
        process.volume.volume_changes["sampling"] = SampleVolumeChange(
            name="sampling",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=_ts([5.0], [-0.05]),
        )
        process.volume.volume_changes["bolus"] = FeedVolumeChange(
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
        process = _make_process(with_controlled_FVC=True, with_controlled_PV=False)
        _apply_pseudobatch_transform(process)
        ordering = get_process_ordering(process)
        state_splines = build_state_splines(process, ordering)
        assert set(state_splines.keys()) == {"biomass", "glucose"}

    def test_pseudobatch_feed_correction_requires_c_star(self):
        process = _make_process(with_controlled_FVC=True, with_controlled_PV=False)
        _apply_pseudobatch_transform(process)
        process.reactor_medium.components["glucose"].c_star_concentration = None
        ordering = get_process_ordering(process)

        with pytest.raises(ValueError, match="no c_star_concentration"):
            build_state_splines(process, ordering)

    def test_pseudobatch_feed_correction_requires_c_star_for_unmodeled_component(self):
        process = _make_process(with_controlled_FVC=True, with_controlled_PV=False)
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
