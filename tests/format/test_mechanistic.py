"""
Tests for bpbench.mechanistic: get_control_splines and get_mass_balance.

All JAX-jit tests use eqx.filter_jit (the equinox-idiomatic way to JIT
modules that contain JAX-array fields).
"""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

import bpbench
from bpbench import (
    BioProcess, BioProcessMetadata, TimeAxis, TimeSeries, StaticVariable,
    ReactorMedium, ReactorMediumComponent, FeedMedium, FeedMediumComponent,
    FeedVolumeChange, SampleVolumeChange, Volume, ProcessVariable,
)
from bpbench.mechanistic import (
    ControlSplines, MassBalance, get_control_splines, get_mass_balance,
    _make_spline,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _ts(t, v):
    return TimeSeries(timepoints=jnp.array(t, dtype=float),
                      values=jnp.array(v, dtype=float))


def _make_feed(name, glucose_conc=500.0, biomass_conc=0.0):
    return FeedMedium(
        name=name, density=1.0, density_unit="kg/L",
        components={
            "biomass": FeedMediumComponent(
                name="biomass", unit="g/L",
                concentration=StaticVariable(value=biomass_conc),
                is_controlled=False,
            ),
            "glucose": FeedMediumComponent(
                name="glucose", unit="g/L",
                concentration=StaticVariable(value=glucose_conc),
                is_controlled=False,
            ),
        },
    )


def _make_process(
    with_controlled_flow=True, with_controlled_pv=True,
    with_discrete_vc=False, with_uncontrolled_pv=False,
    with_uncontrolled_flow=False,
):
    rm = ReactorMedium(
        name="medium", density=1.0, density_unit="kg/L",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass", unit="g/L",
                concentration=_ts([0., 5., 10., 20.], [0.5, 1.0, 2.0, 4.0]),
                is_intracellular=False,
            ),
            "glucose": ReactorMediumComponent(
                name="glucose", unit="g/L",
                concentration=_ts([0., 5., 10., 20.], [10.0, 8.0, 5.0, 1.0]),
                is_intracellular=False,
            ),
        },
    )
    vc_dict = {}
    if with_controlled_flow:
        vc_dict["feed"] = FeedVolumeChange(
            name="feed", unit="L", is_controlled=True, is_continuous=True,
            feed_medium=_make_feed("glucose_feed"),
            values=_ts([0., 5., 10., 20.], [0.0, 0.25, 0.5, 1.0]),
        )
    if with_uncontrolled_flow:
        vc_dict["evaporation"] = FeedVolumeChange(
            name="evaporation", unit="L", is_controlled=False, is_continuous=True,
            feed_medium=_make_feed("water"),
            values=_ts([0., 10., 20.], [0.0, -0.01, -0.02]),
        )
    if with_discrete_vc:
        vc_dict["sampling"] = SampleVolumeChange(
            name="sampling", unit="L", is_controlled=True, is_continuous=False,
            values=_ts([5., 10.], [-0.05, -0.05]),
        )
    pv_dict = {}
    if with_controlled_pv:
        pv_dict["pH"] = ProcessVariable(
            name="pH", unit="", is_controlled=True,
            values=_ts([0., 5., 10., 20.], [7.0, 7.0, 7.0, 7.0]),
        )
    if with_uncontrolled_pv:
        pv_dict["dissolved_O2"] = ProcessVariable(
            name="dissolved_O2", unit="%", is_controlled=False,
            values=_ts([0., 5., 10., 20.], [100., 80., 60., 40.]),
        )
    return BioProcess(
        metadata=BioProcessMetadata(name="test_fb", process_type="fed_batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=20.0,
                           time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes=vc_dict),
        reactor_medium=rm,
        process_variables=pv_dict,
    )


def _make_batch_process():
    rm = ReactorMedium(
        name="medium", density=1.0, density_unit="kg/L",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass", unit="g/L",
                concentration=_ts([0., 5., 10.], [0.5, 1.5, 4.0]),
                is_intracellular=False,
            ),
            "glucose": ReactorMediumComponent(
                name="glucose", unit="g/L",
                concentration=_ts([0., 5., 10.], [10.0, 6.0, 1.0]),
                is_intracellular=False,
            ),
        },
    )
    return BioProcess(
        metadata=BioProcessMetadata(name="batch", process_type="batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=10.0,
                           time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=rm,
    )


# ---------------------------------------------------------------------------
# _make_spline helper tests
# ---------------------------------------------------------------------------

class TestMakeSpline:
    def test_returns_interpax_spline(self):
        import interpax
        sp = _make_spline(jnp.array([0., 1., 2., 3.]), jnp.array([0., 1., 4., 9.]))
        assert isinstance(sp, interpax.CubicSpline)

    def test_is_eqx_module(self):
        sp = _make_spline(jnp.array([0., 1., 2., 3.]), jnp.array([0., 1., 4., 9.]))
        assert isinstance(sp, eqx.Module)

    def test_eval_at_knots(self):
        t = jnp.array([0., 1., 2., 3.])
        v = jnp.array([0., 1., 4., 9.])
        sp = _make_spline(t, v)
        for ti, vi in zip(t, v):
            assert float(sp(ti)) == pytest.approx(float(vi), abs=1e-4)

    def test_single_point_constant(self):
        sp = _make_spline(jnp.array([5.0]), jnp.array([42.0]))
        assert float(sp(5.0)) == pytest.approx(42.0, rel=1e-5)

    def test_two_point_linear(self):
        sp = _make_spline(jnp.array([0., 10.]), jnp.array([0., 5.]))
        assert float(sp(5.0)) == pytest.approx(2.5, rel=1e-4)

    def test_derivative_of_linear_is_slope(self):
        sp = _make_spline(jnp.array([0., 10.]), jnp.array([0., 5.]))
        assert float(sp.derivative()(5.0)) == pytest.approx(0.5, rel=1e-4)

    def test_derivative_of_cumulative_gives_flow(self):
        """Derivative of a linear cumulative-volume curve = constant flow rate."""
        flow = 0.05
        t = jnp.array([0., 5., 10., 15., 20.])
        cum = jnp.array([flow * float(ti) for ti in [0., 5., 10., 15., 20.]])
        sp = _make_spline(t, cum)
        for ti in [2.0, 7.0, 12.0, 17.0]:
            assert float(sp.derivative()(ti)) == pytest.approx(flow, rel=1e-3)

    def test_spline_jittable(self):
        sp = _make_spline(jnp.array([0., 1., 2., 3.]), jnp.array([0., 1., 4., 9.]))
        fn = eqx.filter_jit(sp)
        result = fn(jnp.array(1.5))
        assert result.shape == ()

    def test_derivative_jittable(self):
        sp = _make_spline(jnp.array([0., 1., 2., 3.]), jnp.array([0., 1., 4., 9.]))
        fn = eqx.filter_jit(sp.derivative())
        result = fn(jnp.array(1.5))
        assert result.shape == ()


# ---------------------------------------------------------------------------
# ControlSplines tests
# ---------------------------------------------------------------------------

class TestGetControlSplines:

    def test_returns_control_splines_instance(self):
        assert isinstance(get_control_splines(_make_process()), ControlSplines)

    def test_is_eqx_module(self):
        assert isinstance(get_control_splines(_make_process()), eqx.Module)

    def test_control_names_flow_first_then_ctrl(self):
        cs = get_control_splines(_make_process())
        assert cs.control_names == ("feed", "pH")

    def test_flow_indices(self):
        cs = get_control_splines(_make_process())
        assert cs.flow_indices == (0,)

    def test_ctrl_indices(self):
        cs = get_control_splines(_make_process())
        assert cs.ctrl_indices == (1,)

    def test_flow_only_metadata(self):
        cs = get_control_splines(_make_process(with_controlled_pv=False))
        assert cs.control_names == ("feed",)
        assert cs.flow_indices == (0,)
        assert cs.ctrl_indices == ()

    def test_ctrl_only_metadata(self):
        cs = get_control_splines(_make_process(with_controlled_flow=False))
        assert cs.control_names == ("pH",)
        assert cs.flow_indices == ()
        assert cs.ctrl_indices == (0,)

    def test_discrete_vc_excluded(self):
        cs = get_control_splines(_make_process(with_discrete_vc=True,
                                               with_controlled_pv=False))
        assert "sampling" not in cs.control_names

    def test_uncontrolled_pv_excluded(self):
        cs = get_control_splines(_make_process(with_controlled_flow=False,
                                               with_controlled_pv=False,
                                               with_uncontrolled_pv=True))
        assert "dissolved_O2" not in cs.control_names

    def test_uncontrolled_flow_excluded(self):
        cs = get_control_splines(_make_process(with_controlled_flow=False,
                                               with_controlled_pv=False,
                                               with_uncontrolled_flow=True))
        assert "evaporation" not in cs.control_names

    def test_output_shape(self):
        cs = get_control_splines(_make_process())
        assert cs(jnp.array(10.0)).shape == (2,)

    def test_output_shape_empty(self):
        cs = get_control_splines(_make_process(with_controlled_flow=False,
                                               with_controlled_pv=False))
        assert cs(jnp.array(5.0)).shape == (0,)

    def test_flow_rate_nonnegative_for_monotone_feed(self):
        cs = get_control_splines(_make_process(with_controlled_pv=False))
        for t in [1.0, 5.0, 10.0, 15.0, 19.0]:
            rate = float(cs(jnp.array(t))[0])
            assert rate >= -1e-6, f"Negative flow rate {rate} at t={t}"

    def test_flow_rate_integrated_matches_cumulative_volume(self):
        """Integrating the spline flow rate must reproduce cumulative volume."""
        from scipy.integrate import quad
        cs = get_control_splines(_make_process(with_controlled_pv=False))
        total, _ = quad(lambda t: float(cs(jnp.array(t))[0]), 0.0, 20.0)
        assert total == pytest.approx(1.0, rel=1e-3)

    def test_ctrl_pv_value_at_knots(self):
        cs = get_control_splines(_make_process(with_controlled_flow=False))
        for t in [0., 5., 10., 20.]:
            assert float(cs(jnp.array(t))[0]) == pytest.approx(7.0, abs=1e-3)

    def test_static_pv_constant(self):
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        process.process_variables["temp"] = ProcessVariable(
            name="temp", unit="C", is_controlled=True,
            values=StaticVariable(value=37.0),
        )
        cs = get_control_splines(process)
        assert "temp" in cs.control_names
        for t in [0.0, 10.0, 20.0]:
            assert float(cs(jnp.array(t))[0]) == pytest.approx(37.0, rel=1e-5)

    def test_single_timepoint_pv(self):
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        process.process_variables["initial_S"] = ProcessVariable(
            name="initial_S", unit="g/L", is_controlled=True,
            values=_ts([0.0], [29.2]),
        )
        cs = get_control_splines(process)
        assert "initial_S" in cs.control_names
        assert float(cs(jnp.array(0.0))[0]) == pytest.approx(29.2, rel=1e-4)

    def test_callable_under_filter_jit(self):
        cs = get_control_splines(_make_process())
        out = eqx.filter_jit(cs)(jnp.array(5.0))
        assert out.shape == (2,)

    def test_filter_jit_matches_eager(self):
        cs = get_control_splines(_make_process())
        t = jnp.array(7.3)
        assert jnp.allclose(cs(t), eqx.filter_jit(cs)(t), atol=1e-5)

    def test_grad_through_ctrl_spline(self):
        cs = get_control_splines(_make_process(with_controlled_flow=False))
        g = eqx.filter_jit(jax.grad(lambda t: cs(t)[0]))(jnp.array(10.0))
        assert float(g) == pytest.approx(0.0, abs=1e-4)  # constant pH => slope=0

    def test_grad_through_flow_spline(self):
        cs = get_control_splines(_make_process(with_controlled_pv=False))
        g = eqx.filter_jit(jax.grad(lambda t: cs(t)[0]))(jnp.array(10.0))
        assert g.shape == ()  # scalar gradient (second deriv of cumulative vol)


# ---------------------------------------------------------------------------
# MassBalance tests
# ---------------------------------------------------------------------------

class TestGetMassBalance:

    def test_returns_mass_balance_instance(self):
        assert isinstance(get_mass_balance(_make_process()), MassBalance)

    def test_is_eqx_module(self):
        assert isinstance(get_mass_balance(_make_process()), eqx.Module)

    def test_c_size(self):
        assert get_mass_balance(_make_process()).c_size == 3

    def test_q_size(self):
        assert get_mass_balance(_make_process()).q_size == 2

    def test_u_flow_size_fedbatch(self):
        assert get_mass_balance(_make_process()).u_flow_size == 1

    def test_u_flow_size_batch(self):
        assert get_mass_balance(_make_batch_process()).u_flow_size == 0

    def test_output_size(self):
        assert get_mass_balance(_make_process()).output_size == 3

    def test_species_names(self):
        assert get_mass_balance(_make_process()).species_names == ("biomass", "glucose")

    def test_flow_names(self):
        assert get_mass_balance(_make_process()).flow_names == ("feed",)

    def test_flow_names_empty_batch(self):
        assert get_mass_balance(_make_batch_process()).flow_names == ()

    def test_biomass_idx(self):
        assert get_mass_balance(_make_process()).biomass_idx == 0

    def test_cin_shape(self):
        assert get_mass_balance(_make_process()).Cin.shape == (1, 2)

    def test_cin_values_biomass_and_glucose(self):
        mb = get_mass_balance(_make_process())
        assert float(mb.Cin[0, 0]) == pytest.approx(0.0)
        assert float(mb.Cin[0, 1]) == pytest.approx(500.0)

    def test_cin_empty_for_batch(self):
        assert get_mass_balance(_make_batch_process()).Cin.shape == (0, 2)

    def test_no_biomass_raises(self):
        rm = ReactorMedium(
            name="m", density=1.0, density_unit="kg/L",
            components={"glucose": ReactorMediumComponent(
                name="glucose", unit="g/L",
                concentration=_ts([0., 1.], [10., 5.]),
                is_intracellular=False,
            )},
        )
        p = BioProcess(
            metadata=BioProcessMetadata(name="p", process_type="batch"),
            time_axis=TimeAxis(unit="hours", start=0., end=10.,
                               time_reference="inoculation"),
            volume=Volume(initial_volume=1.0, unit="L"),
            reactor_medium=rm,
        )
        with pytest.raises(ValueError, match="biomass"):
            get_mass_balance(p)

    def test_call_shape_fedbatch(self):
        mb = get_mass_balance(_make_process())
        dc = mb(jnp.array([0.5, 10.0, 1.0]),
                jnp.array([0.3, -0.15]),
                jnp.array([0.05]),
                jnp.zeros(0))
        assert dc.shape == (3,)

    def test_call_shape_batch(self):
        mb = get_mass_balance(_make_batch_process())
        dc = mb(jnp.array([1.0, 5.0, 1.0]),
                jnp.array([0.2, -0.1]),
                jnp.zeros(0),
                jnp.zeros(0))
        assert dc.shape == (3,)

    def test_dV_equals_sum_u_flow(self):
        mb = get_mass_balance(_make_process())
        dc = mb(jnp.array([0.5, 10.0, 1.0]),
                jnp.array([0.3, -0.15]),
                jnp.array([0.05]),
                jnp.zeros(0))
        assert float(dc[-1]) == pytest.approx(0.05, rel=1e-5)

    def test_dV_zero_in_batch(self):
        mb = get_mass_balance(_make_batch_process())
        dc = mb(jnp.array([1.0, 5.0, 1.0]),
                jnp.array([0.2, -0.1]),
                jnp.zeros(0),
                jnp.zeros(0))
        assert float(dc[-1]) == pytest.approx(0.0)

    def test_reaction_only_batch(self):
        mb = get_mass_balance(_make_batch_process())
        X = 2.0
        dc = mb(jnp.array([X, 5.0, 1.0]),
                jnp.array([0.3, -0.15]),
                jnp.zeros(0),
                jnp.zeros(0))
        assert float(dc[0]) == pytest.approx(0.3 * X, rel=1e-5)
        assert float(dc[1]) == pytest.approx(-0.15 * X, rel=1e-5)

    def test_dilution_term_zero_q(self):
        mb = get_mass_balance(_make_process())
        X, S, V, F = 1.0, 10.0, 1.0, 0.1
        dc = mb(jnp.array([X, S, V]), jnp.zeros(2), jnp.array([F]), jnp.zeros(0))
        assert float(dc[0]) == pytest.approx((F / V) * (0.0 - X), rel=1e-4)
        assert float(dc[1]) == pytest.approx((F / V) * (500.0 - S), rel=1e-4)
        assert float(dc[2]) == pytest.approx(F, rel=1e-5)

    def test_full_balance_combined(self):
        mb = get_mass_balance(_make_process())
        X, S, V, F, qX, qS = 2.0, 5.0, 1.5, 0.08, 0.4, -0.2
        dc = mb(jnp.array([X, S, V]), jnp.array([qX, qS]), jnp.array([F]), jnp.zeros(0))
        assert float(dc[0]) == pytest.approx(qX * X + (F/V)*(0.0 - X), rel=1e-5)
        assert float(dc[1]) == pytest.approx(qS * X + (F/V)*(500.0 - S), rel=1e-5)
        assert float(dc[2]) == pytest.approx(F, rel=1e-5)

    def test_callable_under_filter_jit(self):
        mb = get_mass_balance(_make_process())
        dc = eqx.filter_jit(mb)(jnp.array([0.5, 10.0, 1.0]),
                                jnp.array([0.3, -0.15]),
                                jnp.array([0.05]),
                                jnp.zeros(0))
        assert dc.shape == (3,)

    def test_filter_jit_matches_eager(self):
        mb = get_mass_balance(_make_process())
        args = (jnp.array([0.5, 10.0, 1.0]),
                jnp.array([0.3, -0.15]),
                jnp.array([0.05]),
                jnp.zeros(0))
        assert jnp.allclose(mb(*args), eqx.filter_jit(mb)(*args), atol=1e-6)

    def test_grad_wrt_c(self):
        mb = get_mass_balance(_make_process())
        q = jnp.array([0.3, -0.15])
        u_flow = jnp.array([0.05])
        f_mod = jnp.zeros(0)
        g = eqx.filter_jit(jax.grad(lambda c: jnp.sum(mb(c, q, u_flow, f_mod))))(
            jnp.array([0.5, 10.0, 1.0]))
        assert g.shape == (3,)

    def test_grad_wrt_q(self):
        mb = get_mass_balance(_make_process())
        c = jnp.array([0.5, 10.0, 1.0])
        u_flow = jnp.array([0.05])
        f_mod = jnp.zeros(0)
        g = eqx.filter_jit(jax.grad(lambda q: jnp.sum(mb(c, q, u_flow, f_mod))))(
            jnp.array([0.3, -0.15]))
        assert g.shape == (2,)

    def test_vmap_over_batch_of_states(self):
        mb = get_mass_balance(_make_process())
        fn = eqx.filter_jit(jax.vmap(mb, in_axes=(0, 0, 0, None)))
        B = 4
        c = jnp.stack([jnp.array([0.5 + i*0.3, 10.0 - i, 1.0 + i*0.1])
                       for i in range(B)])
        q = jnp.tile(jnp.array([0.3, -0.15]), (B, 1))
        u = jnp.tile(jnp.array([0.05]), (B, 1))
        dc = fn(c, q, u, jnp.zeros(0))
        assert dc.shape == (B, 3)


# ---------------------------------------------------------------------------
# Biomass-at-index-0 reordering tests
# ---------------------------------------------------------------------------

class TestBiomassAtIndexZero:

    def _make_process_glucose_first(self):
        """Process where glucose is inserted before biomass in the dict."""
        rm = ReactorMedium(
            name="medium", density=1.0, density_unit="kg/L",
            components={
                "glucose": ReactorMediumComponent(
                    name="glucose", unit="g/L",
                    concentration=_ts([0., 10.], [10.0, 1.0]),
                    is_intracellular=False,
                ),
                "biomass": ReactorMediumComponent(
                    name="biomass", unit="g/L",
                    concentration=_ts([0., 10.], [0.5, 4.0]),
                    is_intracellular=False,
                ),
            },
        )
        return BioProcess(
            metadata=BioProcessMetadata(name="reorder", process_type="batch"),
            time_axis=TimeAxis(unit="hours", start=0.0, end=10.0,
                               time_reference="inoculation"),
            volume=Volume(initial_volume=1.0, unit="L"),
            reactor_medium=rm,
        )

    def test_biomass_is_always_first(self):
        mb = get_mass_balance(self._make_process_glucose_first())
        assert mb.species_names[0] == "biomass"

    def test_biomass_idx_is_zero(self):
        mb = get_mass_balance(self._make_process_glucose_first())
        assert mb.biomass_idx == 0

    def test_glucose_is_second(self):
        mb = get_mass_balance(self._make_process_glucose_first())
        assert mb.species_names[1] == "glucose"

    def test_reaction_uses_biomass_at_index_0(self):
        """dc[biomass]/dt = qX * X when biomass is reordered to index 0."""
        mb = get_mass_balance(self._make_process_glucose_first())
        X, S, V = 2.0, 5.0, 1.0
        # state is [biomass, glucose, V] after reordering
        dc = mb(jnp.array([X, S, V]), jnp.array([0.3, -0.15]), jnp.zeros(0), jnp.zeros(0))
        assert float(dc[0]) == pytest.approx(0.3 * X, rel=1e-5)
        assert float(dc[1]) == pytest.approx(-0.15 * X, rel=1e-5)


# ---------------------------------------------------------------------------
# Intracellular components tests
# ---------------------------------------------------------------------------

def _make_process_with_intracellular():
    """Process with an intracellular product component."""
    feed = FeedMedium(
        name="glucose_feed", density=1.0, density_unit="kg/L",
        components={
            "biomass": FeedMediumComponent(
                name="biomass", unit="g/L",
                concentration=StaticVariable(value=0.0),
                is_controlled=False,
            ),
            "glucose": FeedMediumComponent(
                name="glucose", unit="g/L",
                concentration=StaticVariable(value=500.0),
                is_controlled=False,
            ),
        },
    )
    rm = ReactorMedium(
        name="medium", density=1.0, density_unit="kg/L",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass", unit="g/L",
                concentration=_ts([0., 10.], [0.5, 4.0]),
                is_intracellular=False,
            ),
            "product": ReactorMediumComponent(
                name="product", unit="g/L",
                concentration=_ts([0., 10.], [0.0, 1.0]),
                is_intracellular=True,
            ),
            "glucose": ReactorMediumComponent(
                name="glucose", unit="g/L",
                concentration=_ts([0., 10.], [10.0, 1.0]),
                is_intracellular=False,
            ),
        },
    )
    vc = FeedVolumeChange(
        name="feed", unit="L", is_controlled=True, is_continuous=True,
        feed_medium=feed,
        values=_ts([0., 5., 10.], [0.0, 0.25, 0.5]),
    )
    return BioProcess(
        metadata=BioProcessMetadata(name="intracellular", process_type="fed_batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=10.0,
                           time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L",
                      volume_changes={"feed": vc}),
        reactor_medium=rm,
    )


class TestIntracellular:

    def test_intracellular_indices_populated(self):
        mb = get_mass_balance(_make_process_with_intracellular())
        assert len(mb.intracellular_indices) == 1

    def test_intracellular_indices_correct(self):
        mb = get_mass_balance(_make_process_with_intracellular())
        # biomass is at 0, product (intracellular) should be at 1
        assert mb.species_names[mb.intracellular_indices[0]] == "product"

    def test_no_intracellular_indices_for_normal_process(self):
        mb = get_mass_balance(_make_process())
        assert mb.intracellular_indices == ()

    def test_biomass_still_at_index_0(self):
        mb = get_mass_balance(_make_process_with_intracellular())
        assert mb.biomass_idx == 0
        assert mb.species_names[0] == "biomass"

    def test_x_active_used_in_reaction(self):
        """Reaction uses X_active = biomass - product, not biomass_measured."""
        mb = get_mass_balance(_make_process_with_intracellular())
        # state: [biomass_active=2.0, product=0.5, glucose=5.0, V=1.0]
        X_active = 2.0 - 0.5  # = 1.5
        qX, qP, qS = 0.4, 0.1, -0.2
        dc = mb(jnp.array([2.0, 0.5, 5.0, 1.0]),
                jnp.array([qX, qP, qS]),
                jnp.zeros(0),
                jnp.zeros(0))
        # No flow: pure reaction
        assert float(dc[0]) == pytest.approx(qX * X_active, rel=1e-5)
        assert float(dc[1]) == pytest.approx(qP * X_active, rel=1e-5)
        assert float(dc[2]) == pytest.approx(qS * X_active, rel=1e-5)

    def test_x_active_with_flow(self):
        """Full balance with flow uses X_active."""
        mb = get_mass_balance(_make_process_with_intracellular())
        X_meas, P, S, V, F = 2.0, 0.5, 5.0, 1.0, 0.1
        X_active = X_meas - P
        qX, qP, qS = 0.4, 0.1, -0.2
        Cin_glucose = 500.0
        dc = mb(jnp.array([X_meas, P, S, V]),
                jnp.array([qX, qP, qS]),
                jnp.array([F]),
                jnp.zeros(0))
        # biomass in feed = 0, product not in feed = 0
        assert float(dc[0]) == pytest.approx(
            qX * X_active + (F/V) * (0.0 - X_meas), rel=1e-5)
        assert float(dc[1]) == pytest.approx(
            qP * X_active + (F/V) * (0.0 - P), rel=1e-5)
        assert float(dc[2]) == pytest.approx(
            qS * X_active + (F/V) * (Cin_glucose - S), rel=1e-5)
        assert float(dc[3]) == pytest.approx(F, rel=1e-5)

    def test_intracellular_jit_compatible(self):
        mb = get_mass_balance(_make_process_with_intracellular())
        dc = eqx.filter_jit(mb)(
            jnp.array([2.0, 0.5, 5.0, 1.0]),
            jnp.array([0.4, 0.1, -0.2]),
            jnp.zeros(0),
            jnp.zeros(0),
        )
        assert dc.shape == (4,)

    def test_no_intracellular_backward_compat(self):
        """Without intracellular components, behaviour is unchanged (X_active == X_measured)."""
        mb = get_mass_balance(_make_batch_process())
        X = 2.0
        dc = mb(jnp.array([X, 5.0, 1.0]),
                jnp.array([0.3, -0.15]),
                jnp.zeros(0),
                jnp.zeros(0))
        assert float(dc[0]) == pytest.approx(0.3 * X, rel=1e-5)
        assert float(dc[1]) == pytest.approx(-0.15 * X, rel=1e-5)


# ---------------------------------------------------------------------------
# Modeled (uncontrolled continuous) flow tests
# ---------------------------------------------------------------------------

def _make_process_with_modeled_flow():
    """Process with a controlled carbon feed and an uncontrolled base feed."""
    carbon_feed_medium = FeedMedium(
        name="carbon_feed", density=1.0, density_unit="kg/L",
        components={
            "biomass": FeedMediumComponent(
                name="biomass", unit="g/L",
                concentration=StaticVariable(value=0.0),
                is_controlled=False,
            ),
            "glucose": FeedMediumComponent(
                name="glucose", unit="g/L",
                concentration=StaticVariable(value=500.0),
                is_controlled=False,
            ),
        },
    )
    base_feed_medium = FeedMedium(
        name="base_feed", density=1.0, density_unit="kg/L",
        components={
            "biomass": FeedMediumComponent(
                name="biomass", unit="g/L",
                concentration=StaticVariable(value=0.0),
                is_controlled=False,
            ),
            "glucose": FeedMediumComponent(
                name="glucose", unit="g/L",
                concentration=StaticVariable(value=0.0),
                is_controlled=False,
            ),
        },
    )
    rm = ReactorMedium(
        name="medium", density=1.0, density_unit="kg/L",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass", unit="g/L",
                concentration=_ts([0., 5., 10., 20.], [0.5, 1.0, 2.0, 4.0]),
                is_intracellular=False,
            ),
            "glucose": ReactorMediumComponent(
                name="glucose", unit="g/L",
                concentration=_ts([0., 5., 10., 20.], [10.0, 8.0, 5.0, 1.0]),
                is_intracellular=False,
            ),
        },
    )
    vc_dict = {
        "carbon_feed": FeedVolumeChange(
            name="carbon_feed", unit="L", is_controlled=True, is_continuous=True,
            feed_medium=carbon_feed_medium,
            values=_ts([0., 5., 10., 20.], [0.0, 0.25, 0.5, 1.0]),
        ),
        "base_feed": FeedVolumeChange(
            name="base_feed", unit="L", is_controlled=False, is_continuous=True,
            feed_medium=base_feed_medium,
            values=_ts([0., 5., 10., 20.], [0.0, 0.1, 0.2, 0.4]),
        ),
    }
    return BioProcess(
        metadata=BioProcessMetadata(name="test_modeled", process_type="fed_batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=20.0,
                           time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes=vc_dict),
        reactor_medium=rm,
    )


class TestModeledFlow:

    def test_modeled_flow_names(self):
        mb = get_mass_balance(_make_process_with_modeled_flow())
        assert mb.modeled_flow_names == ("base_feed",)

    def test_f_modeled_size(self):
        mb = get_mass_balance(_make_process_with_modeled_flow())
        assert mb.f_modeled_size == 1

    def test_cin_modeled_shape(self):
        mb = get_mass_balance(_make_process_with_modeled_flow())
        assert mb.Cin_modeled.shape == (1, 2)

    def test_cin_modeled_values(self):
        mb = get_mass_balance(_make_process_with_modeled_flow())
        assert float(mb.Cin_modeled[0, 0]) == pytest.approx(0.0)
        assert float(mb.Cin_modeled[0, 1]) == pytest.approx(0.0)

    def test_no_modeled_flow_for_simple_process(self):
        mb = get_mass_balance(_make_process())
        assert mb.modeled_flow_names == ()
        assert mb.f_modeled_size == 0
        assert mb.Cin_modeled.shape == (0, 2)

    def test_no_modeled_flow_for_batch(self):
        mb = get_mass_balance(_make_batch_process())
        assert mb.modeled_flow_names == ()
        assert mb.f_modeled_size == 0

    def test_uncontrolled_flow_is_modeled_not_controlled(self):
        """Uncontrolled continuous flow appears in modeled_flow_names, not flow_names."""
        mb = get_mass_balance(_make_process_with_modeled_flow())
        assert "base_feed" not in mb.flow_names
        assert "base_feed" in mb.modeled_flow_names
        assert "carbon_feed" in mb.flow_names
        assert "carbon_feed" not in mb.modeled_flow_names

    def test_call_shape_with_modeled_flow(self):
        mb = get_mass_balance(_make_process_with_modeled_flow())
        dc = mb(jnp.array([0.5, 10.0, 1.0]),
                jnp.array([0.3, -0.15]),
                jnp.array([0.05]),
                jnp.array([0.02]))
        assert dc.shape == (3,)

    def test_dV_includes_modeled_flow(self):
        mb = get_mass_balance(_make_process_with_modeled_flow())
        F_ctrl, F_mod = 0.05, 0.02
        dc = mb(jnp.array([0.5, 10.0, 1.0]),
                jnp.array([0.0, 0.0]),
                jnp.array([F_ctrl]),
                jnp.array([F_mod]))
        assert float(dc[-1]) == pytest.approx(F_ctrl + F_mod, rel=1e-5)

    def test_dilution_with_modeled_flow(self):
        """Modeled flow dilutes species in the reactor."""
        mb = get_mass_balance(_make_process_with_modeled_flow())
        X, S, V = 1.0, 10.0, 1.0
        F_ctrl, F_mod = 0.1, 0.05
        # Both feeds: carbon has Cin_glucose=500, base has Cin_glucose=0
        dc = mb(jnp.array([X, S, V]), jnp.zeros(2),
                jnp.array([F_ctrl]), jnp.array([F_mod]))
        # Expected: (F_ctrl/V)*(500-S) + (F_mod/V)*(0-S) for glucose
        expected_glucose = (F_ctrl / V) * (500.0 - S) + (F_mod / V) * (0.0 - S)
        assert float(dc[1]) == pytest.approx(expected_glucose, rel=1e-4)

    def test_full_balance_with_modeled_flow(self):
        mb = get_mass_balance(_make_process_with_modeled_flow())
        X, S, V = 2.0, 5.0, 1.5
        F_ctrl, F_mod = 0.08, 0.03
        qX, qS = 0.4, -0.2
        dc = mb(jnp.array([X, S, V]), jnp.array([qX, qS]),
                jnp.array([F_ctrl]), jnp.array([F_mod]))
        expected_X = qX * X + (F_ctrl/V)*(0.0-X) + (F_mod/V)*(0.0-X)
        expected_S = qS * X + (F_ctrl/V)*(500.0-S) + (F_mod/V)*(0.0-S)
        expected_dV = F_ctrl + F_mod
        assert float(dc[0]) == pytest.approx(expected_X, rel=1e-5)
        assert float(dc[1]) == pytest.approx(expected_S, rel=1e-5)
        assert float(dc[2]) == pytest.approx(expected_dV, rel=1e-5)

    def test_jit_with_modeled_flow(self):
        mb = get_mass_balance(_make_process_with_modeled_flow())
        dc = eqx.filter_jit(mb)(
            jnp.array([0.5, 10.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.array([0.05]),
            jnp.array([0.02]),
        )
        assert dc.shape == (3,)

    def test_jit_matches_eager_with_modeled_flow(self):
        mb = get_mass_balance(_make_process_with_modeled_flow())
        args = (jnp.array([0.5, 10.0, 1.0]),
                jnp.array([0.3, -0.15]),
                jnp.array([0.05]),
                jnp.array([0.02]))
        assert jnp.allclose(mb(*args), eqx.filter_jit(mb)(*args), atol=1e-6)

    def test_grad_wrt_f_modeled(self):
        mb = get_mass_balance(_make_process_with_modeled_flow())
        c = jnp.array([0.5, 10.0, 1.0])
        q = jnp.array([0.3, -0.15])
        u_flow = jnp.array([0.05])
        g = eqx.filter_jit(
            jax.grad(lambda fm: jnp.sum(mb(c, q, u_flow, fm)))
        )(jnp.array([0.02]))
        assert g.shape == (1,)

    def test_vmap_with_modeled_flow(self):
        mb = get_mass_balance(_make_process_with_modeled_flow())
        fn = eqx.filter_jit(jax.vmap(mb, in_axes=(0, 0, 0, 0)))
        B = 4
        c = jnp.stack([jnp.array([0.5 + i*0.3, 10.0 - i, 1.0 + i*0.1])
                       for i in range(B)])
        q = jnp.tile(jnp.array([0.3, -0.15]), (B, 1))
        u = jnp.tile(jnp.array([0.05]), (B, 1))
        f_mod = jnp.tile(jnp.array([0.02]), (B, 1))
        dc = fn(c, q, u, f_mod)
        assert dc.shape == (B, 3)


# ---------------------------------------------------------------------------
# Integration: ControlSplines + MassBalance wired together
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_wired_ode_step(self):
        process = _make_process()
        ctrl = get_control_splines(process)
        mb = get_mass_balance(process)

        @eqx.filter_jit
        def ode_rhs(t, c, q):
            u = ctrl(t)
            u_flow = u[jnp.array(list(ctrl.flow_indices))]
            return mb(c, q, u_flow, jnp.zeros(0))

        dc = ode_rhs(jnp.array(5.0),
                     jnp.array([0.5, 10.0, 1.0]),
                     jnp.array([0.3, -0.15]))
        assert dc.shape == (3,)

    def test_flow_index_aligns_with_flow_names(self):
        process = _make_process()
        ctrl = get_control_splines(process)
        mb = get_mass_balance(process)
        assert len(ctrl.flow_indices) == mb.u_flow_size
        assert ctrl.control_names[ctrl.flow_indices[0]] == mb.flow_names[0]

    def test_dV_from_wired_ode_equals_flow_rate(self):
        process = _make_process()
        ctrl = get_control_splines(process)
        mb = get_mass_balance(process)
        t = jnp.array(5.0)
        u = ctrl(t)
        u_flow = u[jnp.array(list(ctrl.flow_indices))]
        dc = mb(jnp.array([1.0, 8.0, 1.2]), jnp.zeros(2), u_flow, jnp.zeros(0))
        assert float(dc[-1]) == pytest.approx(float(u_flow[0]), rel=1e-4)


# ---------------------------------------------------------------------------
# Module export tests
# ---------------------------------------------------------------------------

class TestModuleExport:
    def test_functions_accessible(self):
        import bpbench.mechanistic as mech
        assert hasattr(mech, "get_control_splines")
        assert hasattr(mech, "get_mass_balance")
        assert hasattr(mech, "ControlSplines")
        assert hasattr(mech, "MassBalance")

    def test_mechanistic_in_bpbench_namespace(self):
        assert hasattr(bpbench, "mechanistic")

    def test_mechanistic_in_all(self):
        assert "mechanistic" in bpbench.__all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
