"""
Tests for bp_format.mechanistic: get_control_splines and get_rhs_ode.

All JAX-jit tests use eqx.filter_jit (the equinox-idiomatic way to JIT
modules that contain JAX-array fields).
"""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

import bp_format
from bp_format import (
    BiologicalOde,
    BioProcess,
    BioProcessMetadata,
    FeedMediumComponent,
    FeedMedium,
    FeedVolumeChange,
    ProcessVariable,
    RateDecl,
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
    UserDefinedRhsOde,
    build_derived_func,
    build_q_func,
    build_rates_func,
    build_state_splines,
    extract_discrete_events,
    estimate_specific_rates,
    get_control_splines,
    get_rhs_ode,
    integrate_process,
    integrate_process_pseudospace,
)
from bp_format.splines import (
    make_interpax_spline,
    build_pseudobatch_transform,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _ts(t, v):
    return TimeSeries(times=jnp.array(t, dtype=float), values=jnp.array(v, dtype=float))


def _apply_pseudobatch_transform(process, species_names=("biomass", "glucose")):
    transform = build_pseudobatch_transform(process, list(species_names))
    process.pseudobatch_transform = transform
    for sp_name in species_names:
        process.reactor_medium.components[sp_name].concentration = transform.species[
            sp_name
        ].c_star_ts
    return transform


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
    with_controlled_flow=True,
    with_controlled_pv=True,
    with_discrete_vc=False,
    with_uncontrolled_pv=False,
    with_uncontrolled_flow=False,
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
                is_intracellular=False,
            ),
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=_ts([0.0, 5.0, 10.0, 20.0], [10.0, 8.0, 5.0, 1.0]),
                is_intracellular=False,
            ),
        },
    )
    vc_dict = {}
    if with_controlled_flow:
        vc_dict["feed"] = FeedVolumeChange(
            name="feed",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            feed_medium=_make_feed("glucose_feed"),
            values=_ts([0.0, 5.0, 10.0, 20.0], [0.0, 0.25, 0.5, 1.0]),
        )
    if with_uncontrolled_flow:
        vc_dict["evaporation"] = SampleVolumeChange(
            name="evaporation",
            unit="L",
            is_controlled=False,
            is_continuous=True,
            values=_ts([0.0, 10.0, 20.0], [0.0, -0.01, -0.02]),
        )
    if with_discrete_vc:
        vc_dict["sampling"] = SampleVolumeChange(
            name="sampling",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=_ts([5.0, 10.0], [-0.05, -0.05]),
        )
    pv_dict = {}
    if with_controlled_pv:
        pv_dict["pH"] = ProcessVariable(
            name="pH",
            unit="",
            is_controlled=True,
            values=_ts([0.0, 5.0, 10.0, 20.0], [7.0, 7.0, 7.0, 7.0]),
        )
    if with_uncontrolled_pv:
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
                is_intracellular=False,
            ),
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=_ts([0.0, 5.0, 10.0], [10.0, 6.0, 1.0]),
                is_intracellular=False,
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


def _sample_on_observation_grid(
    sim_t: jnp.ndarray,
    sim_y: jnp.ndarray,
    t_obs: jnp.ndarray,
) -> jnp.ndarray:
    distances = jnp.abs(sim_t[:, None] - t_obs[None, :])
    nearest_indices = jnp.argmin(distances, axis=0)
    return sim_y[nearest_indices]


def _interp_on_grid(
    sim_t: jnp.ndarray,
    sim_y: jnp.ndarray,
    t_obs: jnp.ndarray,
) -> jnp.ndarray:
    out = jnp.zeros((len(t_obs), sim_y.shape[1]))
    for i in range(sim_y.shape[1]):
        out = out.at[:, i].set(jnp.interp(t_obs, sim_t, sim_y[:, i]))
    return out


def _wrap_q_as_rates(mb: RhsOde, q_func):
    def rates_func(t, state, controls):
        del state, controls
        q = jnp.asarray(q_func(t), dtype=float)
        r = jnp.zeros(mb.r_size, dtype=float)
        return q, r

    return rates_func


# ---------------------------------------------------------------------------
# make_interpax_spline helper tests
# ---------------------------------------------------------------------------


class TestMakeInterpaxSpline:
    def test_returns_interpax_spline(self):
        import interpax

        sp = make_interpax_spline(
            np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 4.0, 9.0])
        )
        assert isinstance(sp, interpax.CubicSpline)

    def test_is_eqx_module(self):
        sp = make_interpax_spline(
            np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 4.0, 9.0])
        )
        assert isinstance(sp, eqx.Module)

    def test_eval_at_knots(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        v = np.array([0.0, 1.0, 4.0, 9.0])
        sp = make_interpax_spline(t, v)
        for ti, vi in zip(t, v):
            assert float(sp(ti)) == pytest.approx(float(vi), abs=1e-4)

    def test_single_point_constant(self):
        sp = make_interpax_spline(np.array([5.0]), np.array([42.0]))
        assert float(sp(5.0)) == pytest.approx(42.0, rel=1e-5)

    def test_two_point_linear(self):
        sp = make_interpax_spline(np.array([0.0, 10.0]), np.array([0.0, 5.0]))
        assert float(sp(5.0)) == pytest.approx(2.5, rel=1e-4)

    def test_derivative_of_linear_is_slope(self):
        sp = make_interpax_spline(np.array([0.0, 10.0]), np.array([0.0, 5.0]))
        assert float(sp.derivative()(5.0)) == pytest.approx(0.5, rel=1e-4)

    def test_derivative_of_cumulative_gives_flow(self):
        """Derivative of a linear cumulative-volume curve = constant flow rate."""
        flow = 0.05
        t = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        cum = np.array([flow * ti for ti in [0.0, 5.0, 10.0, 15.0, 20.0]])
        sp = make_interpax_spline(t, cum)
        for ti in [2.0, 7.0, 12.0, 17.0]:
            assert float(sp.derivative()(ti)) == pytest.approx(flow, rel=1e-3)

    def test_spline_jittable(self):
        sp = make_interpax_spline(
            np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 4.0, 9.0])
        )
        fn = eqx.filter_jit(sp)
        result = fn(jnp.array(1.5))
        assert result.shape == ()

    def test_derivative_jittable(self):
        sp = make_interpax_spline(
            np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 4.0, 9.0])
        )
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
        cs = get_control_splines(
            _make_process(with_discrete_vc=True, with_controlled_pv=False)
        )
        assert "sampling" not in cs.control_names

    def test_uncontrolled_pv_excluded(self):
        cs = get_control_splines(
            _make_process(
                with_controlled_flow=False,
                with_controlled_pv=False,
                with_uncontrolled_pv=True,
            )
        )
        assert "dissolved_O2" not in cs.control_names

    def test_uncontrolled_flow_excluded(self):
        cs = get_control_splines(
            _make_process(
                with_controlled_flow=False,
                with_controlled_pv=False,
                with_uncontrolled_flow=True,
            )
        )
        assert "evaporation" not in cs.control_names

    def test_output_shape(self):
        cs = get_control_splines(_make_process())
        assert cs(jnp.array(10.0)).shape == (2,)

    def test_output_shape_empty(self):
        cs = get_control_splines(
            _make_process(with_controlled_flow=False, with_controlled_pv=False)
        )
        assert cs(jnp.array(5.0)).shape == (0,)

    def test_flow_rate_nonnegative_for_monotone_feed(self):
        cs = get_control_splines(_make_process(with_controlled_pv=False))
        for t in [1.0, 5.0, 10.0, 15.0, 19.0]:
            rate = float(cs(jnp.array(t))[0])
            assert rate >= -1e-6, f"Negative flow rate {rate} at t={t}"

    def test_flow_rate_integrated_matches_cumulative_volume(self):
        """Integrating the spline flow rate must reproduce cumulative volume."""
        cs = get_control_splines(_make_process(with_controlled_pv=False))
        t_dense = np.linspace(0.0, 20.0, 10_000)
        rates = np.array([float(cs(jnp.array(t))[0]) for t in t_dense])
        total = np.trapezoid(rates, t_dense)
        assert total == pytest.approx(1.0, rel=1e-3)

    def test_ctrl_pv_value_at_knots(self):
        cs = get_control_splines(_make_process(with_controlled_flow=False))
        for t in [0.0, 5.0, 10.0, 20.0]:
            assert float(cs(jnp.array(t))[0]) == pytest.approx(7.0, abs=1e-3)

    def test_static_pv_constant(self):
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        process.process_variables["temp"] = ProcessVariable(
            name="temp",
            unit="C",
            is_controlled=True,
            values=StaticVariable(value=37.0),
        )
        cs = get_control_splines(process)
        assert "temp" in cs.control_names
        for t in [0.0, 10.0, 20.0]:
            assert float(cs(jnp.array(t))[0]) == pytest.approx(37.0, rel=1e-5)

    def test_single_timepoint_pv(self):
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        process.process_variables["initial_S"] = ProcessVariable(
            name="initial_S",
            unit="g/L",
            is_controlled=True,
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

    def test_static_control_variable_supported_without_interpolator_field(self):
        process = _make_process(with_controlled_flow=False)
        cs = get_control_splines(process)
        assert float(cs(jnp.array(10.0))[0]) == pytest.approx(7.0, rel=1e-6)

    def test_prefers_timeseries_spline_state_over_raw_samples(self):
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        pv_series = TimeSeries(
            times=jnp.array([0.0, 10.0, 20.0], dtype=float),
            values=jnp.array([0.0, 1.0, 0.0], dtype=float),
            breaks=jnp.array([0.0, 10.0, 20.0], dtype=float),
            coeffs=jnp.array(
                [
                    [0.0, 0.1, 0.0, 0.0],
                    [1.0, -0.1, 0.0, 0.0],
                ],
                dtype=float,
            ),
            segment_start_piece_idx=jnp.array([0], dtype=jnp.int32),
        )
        process.process_variables["smooth_ctrl"] = ProcessVariable(
            name="smooth_ctrl",
            unit="-",
            is_controlled=True,
            values=pv_series,
        )

        cs = get_control_splines(process)
        idx = cs.control_names.index("smooth_ctrl")

        assert float(cs(jnp.array(5.0))[idx]) == pytest.approx(0.5, abs=2e-2)
        assert float(cs(jnp.array(10.0))[idx]) == pytest.approx(1.0, abs=2e-2)
        assert float(cs(jnp.array(15.0))[idx]) == pytest.approx(0.5, abs=2e-2)


# ---------------------------------------------------------------------------
# RhsOde tests
# ---------------------------------------------------------------------------


class TestGetRhsOde:
    def test_returns_rhs_ode_instance(self):
        assert isinstance(get_rhs_ode(_make_process()), RhsOde)

    def test_is_eqx_module(self):
        assert isinstance(get_rhs_ode(_make_process()), eqx.Module)

    def test_c_size(self):
        assert get_rhs_ode(_make_process()).c_size == 3

    def test_q_size(self):
        assert get_rhs_ode(_make_process()).q_size == 2

    def test_u_flow_size_fedbatch(self):
        assert get_rhs_ode(_make_process()).u_flow_size == 1

    def test_u_flow_size_batch(self):
        assert get_rhs_ode(_make_batch_process()).u_flow_size == 0

    def test_output_size(self):
        assert get_rhs_ode(_make_process()).output_size == 3

    def test_reactor_component_state_names(self):
        mb = get_rhs_ode(_make_process())
        assert mb.reactor_component_state_names == ("biomass", "glucose")

    def test_process_variable_state_names(self):
        mb = get_rhs_ode(_make_process(with_uncontrolled_pv=True))
        assert mb.process_variable_state_names == ("dissolved_O2",)

    def test_flow_names(self):
        assert get_rhs_ode(_make_process()).flow_names == ("feed",)

    def test_flow_names_empty_batch(self):
        assert get_rhs_ode(_make_batch_process()).flow_names == ()

    def test_biomass_idx(self):
        assert get_rhs_ode(_make_process()).biomass_idx == 0

    def test_cin_shape(self):
        assert get_rhs_ode(_make_process()).Cin.shape == (1, 2)

    def test_cin_values_biomass_and_glucose(self):
        mb = get_rhs_ode(_make_process())
        assert float(mb.Cin[0, 0]) == pytest.approx(0.0)
        assert float(mb.Cin[0, 1]) == pytest.approx(500.0)

    def test_cin_empty_for_batch(self):
        assert get_rhs_ode(_make_batch_process()).Cin.shape == (0, 2)

    def test_unknown_feed_component_raises(self):
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        process.volume.volume_changes["bad_feed"] = FeedVolumeChange(
            name="bad_feed",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            feed_medium=FeedMedium(
                name="bad_feed",
                density=1.0,
                density_unit="kg/L",
                components={
                    "biomass": FeedMediumComponent(
                        name="biomass",
                        unit="g/L",
                        concentration=StaticVariable(value=0.0),
                        is_controlled=False,
                    ),
                    "unknown_component": FeedMediumComponent(
                        name="unknown_component",
                        unit="g/L",
                        concentration=StaticVariable(value=1.0),
                        is_controlled=False,
                    ),
                },
            ),
            values=_ts([0.0, 20.0], [0.0, 0.5]),
        )

        with pytest.raises(ValueError, match="Unknown feed component"):
            get_rhs_ode(process)

    def test_no_biomass_raises(self):
        rm = ReactorMedium(
            name="m",
            density=1.0,
            density_unit="kg/L",
            components={
                "glucose": ReactorMediumComponent(
                    name="glucose",
                    unit="g/L",
                    concentration=_ts([0.0, 1.0], [10.0, 5.0]),
                    is_intracellular=False,
                )
            },
        )
        p = BioProcess(
            metadata=BioProcessMetadata(name="p", process_type="batch"),
            time_axis=TimeAxis(
                unit="hours", start=0.0, end=10.0, time_reference="inoculation"
            ),
            volume=Volume(initial_volume=1.0, unit="L"),
            reactor_medium=rm,
        )
        with pytest.raises(ValueError, match="biomass"):
            get_rhs_ode(p)

    def test_call_shape_fedbatch(self):
        mb = get_rhs_ode(_make_process())
        dc = mb(
            jnp.array([0.5, 10.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.array([0.05]),
            jnp.zeros(0),
            jnp.zeros(mb.r_size),
        )
        assert dc.shape == (3,)

    def test_call_shape_batch(self):
        mb = get_rhs_ode(_make_batch_process())
        dc = mb(
            jnp.array([1.0, 5.0, 1.0]),
            jnp.array([0.2, -0.1]),
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.zeros(mb.r_size),
        )
        assert dc.shape == (3,)

    def test_dV_equals_sum_u_flow(self):
        mb = get_rhs_ode(_make_process())
        dc = mb(
            jnp.array([0.5, 10.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.array([0.05]),
            jnp.zeros(0),
            jnp.zeros(mb.r_size),
        )
        assert float(dc[-1]) == pytest.approx(0.05, rel=1e-5)

    def test_dV_zero_in_batch(self):
        mb = get_rhs_ode(_make_batch_process())
        dc = mb(
            jnp.array([1.0, 5.0, 1.0]),
            jnp.array([0.2, -0.1]),
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.zeros(mb.r_size),
        )
        assert float(dc[-1]) == pytest.approx(0.0)

    def test_reaction_only_batch(self):
        mb = get_rhs_ode(_make_batch_process())
        X = 2.0
        dc = mb(
            jnp.array([X, 5.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.zeros(mb.r_size),
        )
        assert float(dc[0]) == pytest.approx(0.3 * X, rel=1e-5)
        assert float(dc[1]) == pytest.approx(-0.15 * X, rel=1e-5)

    def test_dilution_term_zero_q(self):
        mb = get_rhs_ode(_make_process())
        X, S, V, F = 1.0, 10.0, 1.0, 0.1
        dc = mb(
            jnp.array([X, S, V]),
            jnp.zeros(2),
            jnp.array([F]),
            jnp.zeros(0),
            jnp.zeros(mb.r_size),
        )
        assert float(dc[0]) == pytest.approx((F / V) * (0.0 - X), rel=1e-4)
        assert float(dc[1]) == pytest.approx((F / V) * (500.0 - S), rel=1e-4)
        assert float(dc[2]) == pytest.approx(F, rel=1e-5)

    def test_full_balance_combined(self):
        mb = get_rhs_ode(_make_process())
        X, S, V, F, qX, qS = 2.0, 5.0, 1.5, 0.08, 0.4, -0.2
        dc = mb(
            jnp.array([X, S, V]),
            jnp.array([qX, qS]),
            jnp.array([F]),
            jnp.zeros(0),
            jnp.zeros(mb.r_size),
        )
        assert float(dc[0]) == pytest.approx(qX * X + (F / V) * (0.0 - X), rel=1e-5)
        assert float(dc[1]) == pytest.approx(qS * X + (F / V) * (500.0 - S), rel=1e-5)
        assert float(dc[2]) == pytest.approx(F, rel=1e-5)

    def test_callable_under_filter_jit(self):
        mb = get_rhs_ode(_make_process())
        dc = eqx.filter_jit(mb)(
            jnp.array([0.5, 10.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.array([0.05]),
            jnp.zeros(0),
            jnp.zeros(mb.r_size),
        )
        assert dc.shape == (3,)

    def test_filter_jit_matches_eager(self):
        mb = get_rhs_ode(_make_process())
        args = (
            jnp.array([0.5, 10.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.array([0.05]),
            jnp.zeros(0),
            jnp.zeros(mb.r_size),
        )
        assert jnp.allclose(mb(*args), eqx.filter_jit(mb)(*args), atol=1e-6)

    def test_grad_wrt_c(self):
        mb = get_rhs_ode(_make_process())
        q = jnp.array([0.3, -0.15])
        u_flow = jnp.array([0.05])
        f_mod = jnp.zeros(0)
        r = jnp.zeros(mb.r_size)
        g = eqx.filter_jit(jax.grad(lambda c: jnp.sum(mb(c, q, u_flow, f_mod, r))))(
            jnp.array([0.5, 10.0, 1.0])
        )
        assert g.shape == (3,)

    def test_grad_wrt_q(self):
        mb = get_rhs_ode(_make_process())
        c = jnp.array([0.5, 10.0, 1.0])
        u_flow = jnp.array([0.05])
        f_mod = jnp.zeros(0)
        r = jnp.zeros(mb.r_size)
        g = eqx.filter_jit(jax.grad(lambda q: jnp.sum(mb(c, q, u_flow, f_mod, r))))(
            jnp.array([0.3, -0.15])
        )
        assert g.shape == (2,)

    def test_vmap_over_batch_of_states(self):
        mb = get_rhs_ode(_make_process())
        fn = eqx.filter_jit(eqx.filter_vmap(mb, in_axes=(0, 0, 0, None, None)))
        B = 4
        c = jnp.stack(
            [jnp.array([0.5 + i * 0.3, 10.0 - i, 1.0 + i * 0.1]) for i in range(B)]
        )
        q = jnp.tile(jnp.array([0.3, -0.15]), (B, 1))
        u = jnp.tile(jnp.array([0.05]), (B, 1))
        dc = fn(c, q, u, jnp.zeros(0), jnp.zeros(mb.r_size))
        assert dc.shape == (B, 3)

    def test_static_uncontrolled_pv_forced_zero_dynamics(self):
        process = _make_process(
            with_controlled_flow=False,
            with_controlled_pv=False,
            with_uncontrolled_pv=False,
        )
        process.process_variables["kLa"] = ProcessVariable(
            name="kLa",
            unit="1/h",
            is_controlled=False,
            values=StaticVariable(value=80.0),
        )
        mb = get_rhs_ode(process)

        state = jnp.array([1.0, 10.0, 80.0, 1.0])
        q = jnp.array([0.2, -0.1])
        r = jnp.array([0.0, 0.0, 5.0])
        dc = mb(state, q, jnp.zeros(0), jnp.zeros(0), r)

        assert float(dc[2]) == pytest.approx(0.0)

    def test_no_uncontrolled_pv_regression(self):
        mb = get_rhs_ode(_make_process(with_uncontrolled_pv=False))
        assert mb.n_pv_states == 0
        assert mb.c_size == 3
        dc = mb(
            jnp.array([1.0, 10.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.array([0.05]),
            jnp.zeros(0),
            jnp.zeros(mb.r_size),
        )
        assert dc.shape == (3,)

    def test_event_and_feed_effects_are_reactor_only(self):
        process = _make_process(
            with_controlled_flow=True,
            with_controlled_pv=False,
            with_uncontrolled_pv=True,
        )
        mb = get_rhs_ode(process)
        X, S, DO, V = 1.0, 10.0, 80.0, 1.0
        F = 0.1
        dc = mb(
            jnp.array([X, S, DO, V]),
            jnp.zeros(mb.q_size),
            jnp.array([F]),
            jnp.zeros(0),
            jnp.zeros(mb.r_size),
        )
        assert float(dc[2]) == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Biomass-at-index-0 reordering tests
# ---------------------------------------------------------------------------


class TestBiomassAtIndexZero:
    def _make_process_glucose_first(self):
        """Process where glucose is inserted before biomass in the dict."""
        rm = ReactorMedium(
            name="medium",
            density=1.0,
            density_unit="kg/L",
            components={
                "glucose": ReactorMediumComponent(
                    name="glucose",
                    unit="g/L",
                    concentration=_ts([0.0, 10.0], [10.0, 1.0]),
                    is_intracellular=False,
                ),
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=_ts([0.0, 10.0], [0.5, 4.0]),
                    is_intracellular=False,
                ),
            },
        )
        return BioProcess(
            metadata=BioProcessMetadata(name="reorder", process_type="batch"),
            time_axis=TimeAxis(
                unit="hours", start=0.0, end=10.0, time_reference="inoculation"
            ),
            volume=Volume(initial_volume=1.0, unit="L"),
            reactor_medium=rm,
        )

    def test_biomass_is_always_first(self):
        mb = get_rhs_ode(self._make_process_glucose_first())
        assert mb.reactor_component_state_names[0] == "biomass"

    def test_biomass_idx_is_zero(self):
        mb = get_rhs_ode(self._make_process_glucose_first())
        assert mb.biomass_idx == 0

    def test_glucose_is_second(self):
        mb = get_rhs_ode(self._make_process_glucose_first())
        assert mb.reactor_component_state_names[1] == "glucose"

    def test_reaction_uses_biomass_at_index_0(self):
        """dc[biomass]/dt = qX * X when biomass is reordered to index 0."""
        mb = get_rhs_ode(self._make_process_glucose_first())
        X, S, V = 2.0, 5.0, 1.0
        # state is [biomass, glucose, V] after reordering
        dc = mb(
            jnp.array([X, S, V]),
            jnp.array([0.3, -0.15]),
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.zeros(mb.r_size),
        )
        assert float(dc[0]) == pytest.approx(0.3 * X, rel=1e-5)
        assert float(dc[1]) == pytest.approx(-0.15 * X, rel=1e-5)


# ---------------------------------------------------------------------------
# Intracellular components tests
# ---------------------------------------------------------------------------


def _make_process_with_intracellular():
    """Process with an intracellular product component."""
    feed = FeedMedium(
        name="glucose_feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": FeedMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=StaticVariable(value=0.0),
                is_controlled=False,
            ),
            "glucose": FeedMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=StaticVariable(value=500.0),
                is_controlled=False,
            ),
        },
    )
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=_ts([0.0, 10.0], [0.5, 4.0]),
                is_intracellular=False,
            ),
            "product": ReactorMediumComponent(
                name="product",
                unit="g/L",
                concentration=_ts([0.0, 10.0], [0.0, 1.0]),
                is_intracellular=True,
            ),
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=_ts([0.0, 10.0], [10.0, 1.0]),
                is_intracellular=False,
            ),
        },
    )
    vc = FeedVolumeChange(
        name="feed",
        unit="L",
        is_controlled=True,
        is_continuous=True,
        feed_medium=feed,
        values=_ts([0.0, 5.0, 10.0], [0.0, 0.25, 0.5]),
    )
    return BioProcess(
        metadata=BioProcessMetadata(name="intracellular", process_type="fed_batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=10.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes={"feed": vc}),
        reactor_medium=rm,
    )


class TestIntracellular:
    def test_intracellular_indices_populated(self):
        mb = get_rhs_ode(_make_process_with_intracellular())
        assert len(mb.intracellular_indices) == 1

    def test_intracellular_indices_correct(self):
        mb = get_rhs_ode(_make_process_with_intracellular())
        # biomass is at 0, product (intracellular) should be at 1
        assert (
            mb.reactor_component_state_names[mb.intracellular_indices[0]] == "product"
        )

    def test_no_intracellular_indices_for_normal_process(self):
        mb = get_rhs_ode(_make_process())
        assert mb.intracellular_indices == ()

    def test_biomass_still_at_index_0(self):
        mb = get_rhs_ode(_make_process_with_intracellular())
        assert mb.biomass_idx == 0
        assert mb.reactor_component_state_names[0] == "biomass"

    def test_x_active_used_in_reaction(self):
        """Reaction uses X_active = biomass - product, and the measured-biomass
        derivative absorbs the intracellular accumulation rate so mass balance
        holds: dX_meas/dt = (q_X_active + q_P) * X_active.
        """
        mb = get_rhs_ode(_make_process_with_intracellular())
        # state: [biomass_measured=2.0, product=0.5, glucose=5.0, V=1.0]
        X_active = 2.0 - 0.5  # = 1.5
        qX, qP, qS = 0.4, 0.1, -0.2
        dc = mb(
            jnp.array([2.0, 0.5, 5.0, 1.0]),
            jnp.array([qX, qP, qS]),
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.zeros(mb.r_size),
        )
        # No flow: pure reaction. Biomass entry includes intracellular term.
        assert float(dc[0]) == pytest.approx((qX + qP) * X_active, rel=1e-5)
        assert float(dc[1]) == pytest.approx(qP * X_active, rel=1e-5)
        assert float(dc[2]) == pytest.approx(qS * X_active, rel=1e-5)
        # Mass-balance identity: dX_meas/dt == dX_active/dt + dP/dt.
        assert float(dc[0]) == pytest.approx(qX * X_active + float(dc[1]), rel=1e-5)

    def test_x_active_with_flow(self):
        """Full balance with flow: biomass derivative includes intracellular q
        accumulation so dX_meas/dt = dX_active/dt + dP/dt at any feed rate.
        """
        mb = get_rhs_ode(_make_process_with_intracellular())
        X_meas, P, S, V, F = 2.0, 0.5, 5.0, 1.0, 0.1
        X_active = X_meas - P
        qX, qP, qS = 0.4, 0.1, -0.2
        Cin_glucose = 500.0
        dc = mb(
            jnp.array([X_meas, P, S, V]),
            jnp.array([qX, qP, qS]),
            jnp.array([F]),
            jnp.zeros(0),
            jnp.zeros(mb.r_size),
        )
        # biomass in feed = 0, product not in feed = 0
        assert float(dc[0]) == pytest.approx(
            (qX + qP) * X_active + (F / V) * (0.0 - X_meas), rel=1e-5
        )
        assert float(dc[1]) == pytest.approx(
            qP * X_active + (F / V) * (0.0 - P), rel=1e-5
        )
        assert float(dc[2]) == pytest.approx(
            qS * X_active + (F / V) * (Cin_glucose - S), rel=1e-5
        )
        assert float(dc[3]) == pytest.approx(F, rel=1e-5)

    def test_intracellular_jit_compatible(self):
        mb = get_rhs_ode(_make_process_with_intracellular())
        dc = eqx.filter_jit(mb)(
            jnp.array([2.0, 0.5, 5.0, 1.0]),
            jnp.array([0.4, 0.1, -0.2]),
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.zeros(mb.r_size),
        )
        assert dc.shape == (4,)

    def test_no_intracellular_backward_compat(self):
        """Without intracellular components, X_active remains X_measured."""
        mb = get_rhs_ode(_make_batch_process())
        X = 2.0
        dc = mb(
            jnp.array([X, 5.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.zeros(mb.r_size),
        )
        assert float(dc[0]) == pytest.approx(0.3 * X, rel=1e-5)
        assert float(dc[1]) == pytest.approx(-0.15 * X, rel=1e-5)


def _make_batch_process_with_intracellular():
    """Batch process (no feed/sample, V constant) with intracellular product."""
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=_ts([0.0, 10.0], [1.0, 5.0]),
                is_intracellular=False,
            ),
            "product": ReactorMediumComponent(
                name="product",
                unit="g/L",
                concentration=_ts([0.0, 10.0], [0.0, 1.0]),
                is_intracellular=True,
            ),
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=_ts([0.0, 10.0], [10.0, 1.0]),
                is_intracellular=False,
            ),
        },
    )
    return BioProcess(
        metadata=BioProcessMetadata(name="batch_intra", process_type="batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=10.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=rm,
    )


class TestIntracellularRoundTrip:
    """End-to-end round-trip on a batch process with intracellular product.

    Closed-form analytic batch with constant specific rates:
      dX_active/dt = qXa * X_active   →   X_active(t) = X0 * exp(qXa * t)
      dP/dt        = qP  * X_active   →   P(t) = P0 + (qP/qXa) * X0 * (exp(qXa*t)-1)
      dS/dt        = qS  * X_active   →   S(t) = S0 + (qS/qXa) * X0 * (exp(qXa*t)-1)
      X_meas(t)    = X_active(t) + P(t)

    These checks would silently pass under the pre-fix apparent-rate
    convention because forward and inversion were symmetrically wrong.
    After the Phase 1 fix, inversion must recover the *active* growth
    rate qXa rather than the apparent rate (qXa + qP).
    """

    qXa, qP, qS = 0.30, 0.10, -0.20
    X0_active, P0, S0 = 1.0, 0.0, 10.0

    def _analytic(self, t):
        t = np.asarray(t, dtype=float)
        X_active = self.X0_active * np.exp(self.qXa * t)
        P = self.P0 + (self.qP / self.qXa) * self.X0_active * (
            np.exp(self.qXa * t) - 1.0
        )
        X_meas = X_active + P
        S = np.maximum(
            self.S0
            + (self.qS / self.qXa) * self.X0_active * (np.exp(self.qXa * t) - 1.0),
            0.0,
        )
        return X_meas, P, S, X_active

    def test_inversion_recovers_active_specific_growth_rate(self):
        t = np.linspace(0.0, 10.0, 51)
        X_meas, P, S, _ = self._analytic(t)

        process = _make_batch_process_with_intracellular()
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)

        state_splines = {
            "biomass": make_interpax_spline(t, X_meas),
            "product": make_interpax_spline(t, P),
            "glucose": make_interpax_spline(t, S),
        }

        # Stay away from data edges; cubic-spline derivative noise near
        # boundaries is unrelated to the inversion correctness being tested.
        t_eval = np.linspace(1.0, 8.0, 15)
        q_est = estimate_specific_rates(process, ctrl, mb, state_splines, t_eval)

        np.testing.assert_allclose(q_est[:, 0], self.qXa, rtol=0.02)
        np.testing.assert_allclose(q_est[:, 1], self.qP, rtol=0.02)
        np.testing.assert_allclose(q_est[:, 2], self.qS, rtol=0.02)

        apparent = self.qXa + self.qP
        assert not np.allclose(q_est[:, 0], apparent, rtol=0.02), (
            "q[biomass] must NOT equal apparent rate (qXa + qP) after Phase 1 fix"
        )

    def test_forward_with_known_active_rates_matches_analytic(self):
        process = _make_batch_process_with_intracellular()
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)

        q_arr = jnp.array([self.qXa, self.qP, self.qS])

        def q_func(_t):
            return q_arr

        rates_func = _wrap_q_as_rates(mb, q_func)
        t_eval = np.linspace(0.0, 10.0, 21)
        result = integrate_process(process, ctrl, mb, rates_func, t_eval)

        X_meas_true, P_true, S_true, _ = self._analytic(np.asarray(result["t"]))
        np.testing.assert_allclose(result["c"][:, 0], X_meas_true, rtol=5e-3, atol=1e-3)
        np.testing.assert_allclose(result["c"][:, 1], P_true, rtol=5e-3, atol=1e-3)
        np.testing.assert_allclose(result["c"][:, 2], S_true, rtol=5e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# User-defined biological ODE
# ---------------------------------------------------------------------------


def _make_batch_with_biological_ode_intracellular():
    """Batch process with the same intracellular shape as
    :func:`_make_batch_process_with_intracellular`, but with a user-defined
    ``biological_ode`` block that explicitly writes the active-vs-measured
    biomass mass balance.
    """
    p = _make_batch_process_with_intracellular()
    p.biological_ode = BiologicalOde(
        derived={"X_active": "biomass - product"},
        rates={
            "q_X_active": RateDecl(),
            "q_P": RateDecl(),
            "q_S": RateDecl(),
        },
        derivatives={
            "biomass": "q_X_active * X_active + q_P * X_active",
            "product": "q_P * X_active",
            "glucose": "q_S * X_active",
        },
    )
    return p


class TestUserDefinedRhsOde:
    def test_get_rhs_ode_dispatches_to_user_defined(self):
        p = _make_batch_with_biological_ode_intracellular()
        mb = get_rhs_ode(p)
        assert isinstance(mb, UserDefinedRhsOde)

    def test_get_rhs_ode_falls_back_to_auto_when_block_absent(self):
        p = _make_batch_process_with_intracellular()
        assert p.biological_ode is None
        mb = get_rhs_ode(p)
        assert isinstance(mb, RhsOde)

    def test_user_defined_rhs_matches_auto_on_intracellular_state(self):
        """Hand-written biological_ode that matches the auto-RHS semantics
        must produce identical dc/dt at sample states (no flow)."""
        p_user = _make_batch_with_biological_ode_intracellular()
        p_auto = _make_batch_process_with_intracellular()
        mb_user = get_rhs_ode(p_user)
        mb_auto = get_rhs_ode(p_auto)

        c = jnp.array([2.0, 0.5, 5.0, 1.0])
        rates = jnp.array([0.4, 0.1, -0.2])
        dc_user = mb_user(c, rates, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0))
        dc_auto = mb_auto(
            c, rates, jnp.zeros(0), jnp.zeros(0), jnp.zeros(mb_auto.r_size)
        )
        np.testing.assert_allclose(np.asarray(dc_user), np.asarray(dc_auto), atol=1e-6)

    def test_derived_func_returns_x_active(self):
        p = _make_batch_with_biological_ode_intracellular()
        df = build_derived_func(p)
        state_values = jnp.array([2.0, 0.5, 5.0])
        out = df(state_values, jnp.zeros(0), jnp.array([0.0, 0.0, 0.0]))
        assert "X_active" in out
        assert float(out["X_active"]) == pytest.approx(1.5, rel=1e-6)

    def test_user_defined_jit_compatible(self):
        p = _make_batch_with_biological_ode_intracellular()
        mb = get_rhs_ode(p)
        dc = eqx.filter_jit(mb)(
            jnp.array([2.0, 0.5, 5.0, 1.0]),
            jnp.array([0.4, 0.1, -0.2]),
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.zeros(0),
        )
        assert dc.shape == (4,)

    def test_zero_derivative_means_no_biological_dynamics(self):
        """An entry of '0' in derivatives keeps the state's biological term
        at zero; physical (feed) contributions still apply on reactor states.
        """
        p = _make_batch_with_biological_ode_intracellular()
        # Override glucose to have no biological dynamics
        p.biological_ode.derivatives["glucose"] = "0"
        mb = get_rhs_ode(p)
        c = jnp.array([2.0, 0.5, 5.0, 1.0])
        rates = jnp.array([0.4, 0.1, -0.2])
        dc = mb(c, rates, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0))
        # No flow → glucose entry is purely the biological term, which is 0.
        assert float(dc[2]) == pytest.approx(0.0, abs=1e-7)

    def test_rate_size_follows_user_declaration(self):
        """`len(rates)` is whatever the user declared, not pinned to
        n_reactor_states. Three rates for three states matches today; an
        extra unused rate just adds to rate_size without changing dc/dt.
        """
        p = _make_batch_with_biological_ode_intracellular()
        p.biological_ode.rates["q_unused"] = RateDecl()
        mb = get_rhs_ode(p)
        assert mb.rate_size == 4
        # Adding the unused rate must not change dc/dt as long as no
        # expression references it.
        c = jnp.array([2.0, 0.5, 5.0, 1.0])
        rates_4 = jnp.array([0.4, 0.1, -0.2, 99.0])
        dc = mb(c, rates_4, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0))
        X_active = 1.5
        assert float(dc[0]) == pytest.approx((0.4 + 0.1) * X_active, rel=1e-5)
        assert float(dc[1]) == pytest.approx(0.1 * X_active, rel=1e-5)
        assert float(dc[2]) == pytest.approx(-0.2 * X_active, rel=1e-5)


# ---------------------------------------------------------------------------
# Modeled (uncontrolled continuous) flow tests
# ---------------------------------------------------------------------------


def _make_process_with_modeled_flow():
    """Process with a controlled carbon feed and an uncontrolled base feed."""
    carbon_feed_medium = FeedMedium(
        name="carbon_feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": FeedMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=StaticVariable(value=0.0),
                is_controlled=False,
            ),
            "glucose": FeedMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=StaticVariable(value=500.0),
                is_controlled=False,
            ),
        },
    )
    base_feed_medium = FeedMedium(
        name="base_feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": FeedMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=StaticVariable(value=0.0),
                is_controlled=False,
            ),
            "glucose": FeedMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=StaticVariable(value=0.0),
                is_controlled=False,
            ),
        },
    )
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=_ts([0.0, 5.0, 10.0, 20.0], [0.5, 1.0, 2.0, 4.0]),
                is_intracellular=False,
            ),
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=_ts([0.0, 5.0, 10.0, 20.0], [10.0, 8.0, 5.0, 1.0]),
                is_intracellular=False,
            ),
        },
    )
    vc_dict = {
        "carbon_feed": FeedVolumeChange(
            name="carbon_feed",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            feed_medium=carbon_feed_medium,
            values=_ts([0.0, 5.0, 10.0, 20.0], [0.0, 0.25, 0.5, 1.0]),
        ),
        "base_feed": FeedVolumeChange(
            name="base_feed",
            unit="L",
            is_controlled=False,
            is_continuous=True,
            feed_medium=base_feed_medium,
            values=_ts([0.0, 5.0, 10.0, 20.0], [0.0, 0.1, 0.2, 0.4]),
        ),
    }
    return BioProcess(
        metadata=BioProcessMetadata(name="test_modeled", process_type="fed_batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=20.0, time_reference="inoculation"
        ),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes=vc_dict),
        reactor_medium=rm,
    )


class TestModeledFlow:
    def test_modeled_flow_names(self):
        mb = get_rhs_ode(_make_process_with_modeled_flow())
        assert mb.modeled_flow_names == ("base_feed",)

    def test_f_modeled_size(self):
        mb = get_rhs_ode(_make_process_with_modeled_flow())
        assert mb.f_modeled_size == 1

    def test_cin_modeled_shape(self):
        mb = get_rhs_ode(_make_process_with_modeled_flow())
        assert mb.Cin_modeled.shape == (1, 2)

    def test_cin_modeled_values(self):
        mb = get_rhs_ode(_make_process_with_modeled_flow())
        assert float(mb.Cin_modeled[0, 0]) == pytest.approx(0.0)
        assert float(mb.Cin_modeled[0, 1]) == pytest.approx(0.0)

    def test_no_modeled_flow_for_simple_process(self):
        mb = get_rhs_ode(_make_process())
        assert mb.modeled_flow_names == ()
        assert mb.f_modeled_size == 0
        assert mb.Cin_modeled.shape == (0, 2)

    def test_no_modeled_flow_for_batch(self):
        mb = get_rhs_ode(_make_batch_process())
        assert mb.modeled_flow_names == ()
        assert mb.f_modeled_size == 0

    def test_uncontrolled_flow_is_modeled_not_controlled(self):
        """Uncontrolled continuous flow appears in modeled_flow_names."""
        mb = get_rhs_ode(_make_process_with_modeled_flow())
        assert "base_feed" not in mb.flow_names
        assert "base_feed" in mb.modeled_flow_names
        assert "carbon_feed" in mb.flow_names
        assert "carbon_feed" not in mb.modeled_flow_names

    def test_call_shape_with_modeled_flow(self):
        mb = get_rhs_ode(_make_process_with_modeled_flow())
        dc = mb(
            jnp.array([0.5, 10.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.array([0.05]),
            jnp.array([0.02]),
            jnp.zeros(mb.r_size),
        )
        assert dc.shape == (3,)

    def test_dV_includes_modeled_flow(self):
        mb = get_rhs_ode(_make_process_with_modeled_flow())
        F_ctrl, F_mod = 0.05, 0.02
        dc = mb(
            jnp.array([0.5, 10.0, 1.0]),
            jnp.array([0.0, 0.0]),
            jnp.array([F_ctrl]),
            jnp.array([F_mod]),
            jnp.zeros(mb.r_size),
        )
        assert float(dc[-1]) == pytest.approx(F_ctrl + F_mod, rel=1e-5)

    def test_dilution_with_modeled_flow(self):
        """Modeled flow dilutes species in the reactor."""
        mb = get_rhs_ode(_make_process_with_modeled_flow())
        X, S, V = 1.0, 10.0, 1.0
        F_ctrl, F_mod = 0.1, 0.05
        # Both feeds: carbon has Cin_glucose=500, base has Cin_glucose=0
        dc = mb(
            jnp.array([X, S, V]),
            jnp.zeros(2),
            jnp.array([F_ctrl]),
            jnp.array([F_mod]),
            jnp.zeros(mb.r_size),
        )
        # Expected: (F_ctrl/V)*(500-S) + (F_mod/V)*(0-S) for glucose
        expected_glucose = (F_ctrl / V) * (500.0 - S) + (F_mod / V) * (0.0 - S)
        assert float(dc[1]) == pytest.approx(expected_glucose, rel=1e-4)

    def test_full_balance_with_modeled_flow(self):
        mb = get_rhs_ode(_make_process_with_modeled_flow())
        X, S, V = 2.0, 5.0, 1.5
        F_ctrl, F_mod = 0.08, 0.03
        qX, qS = 0.4, -0.2
        dc = mb(
            jnp.array([X, S, V]),
            jnp.array([qX, qS]),
            jnp.array([F_ctrl]),
            jnp.array([F_mod]),
            jnp.zeros(mb.r_size),
        )
        expected_X = qX * X + (F_ctrl / V) * (0.0 - X) + (F_mod / V) * (0.0 - X)
        expected_S = qS * X + (F_ctrl / V) * (500.0 - S) + (F_mod / V) * (0.0 - S)
        expected_dV = F_ctrl + F_mod
        assert float(dc[0]) == pytest.approx(expected_X, rel=1e-5)
        assert float(dc[1]) == pytest.approx(expected_S, rel=1e-5)
        assert float(dc[2]) == pytest.approx(expected_dV, rel=1e-5)

    def test_jit_with_modeled_flow(self):
        mb = get_rhs_ode(_make_process_with_modeled_flow())
        dc = eqx.filter_jit(mb)(
            jnp.array([0.5, 10.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.array([0.05]),
            jnp.array([0.02]),
            jnp.zeros(mb.r_size),
        )
        assert dc.shape == (3,)

    def test_jit_matches_eager_with_modeled_flow(self):
        mb = get_rhs_ode(_make_process_with_modeled_flow())
        args = (
            jnp.array([0.5, 10.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.array([0.05]),
            jnp.array([0.02]),
            jnp.zeros(mb.r_size),
        )
        assert jnp.allclose(mb(*args), eqx.filter_jit(mb)(*args), atol=1e-6)

    def test_grad_wrt_f_modeled(self):
        mb = get_rhs_ode(_make_process_with_modeled_flow())
        c = jnp.array([0.5, 10.0, 1.0])
        q = jnp.array([0.3, -0.15])
        u_flow = jnp.array([0.05])
        g = eqx.filter_jit(
            jax.grad(lambda fm: jnp.sum(mb(c, q, u_flow, fm, jnp.zeros(mb.r_size))))
        )(jnp.array([0.02]))
        assert g.shape == (1,)

    def test_vmap_with_modeled_flow(self):
        mb = get_rhs_ode(_make_process_with_modeled_flow())
        fn = eqx.filter_jit(eqx.filter_vmap(mb, in_axes=(0, 0, 0, 0, None)))
        B = 4
        c = jnp.stack(
            [jnp.array([0.5 + i * 0.3, 10.0 - i, 1.0 + i * 0.1]) for i in range(B)]
        )
        q = jnp.tile(jnp.array([0.3, -0.15]), (B, 1))
        u = jnp.tile(jnp.array([0.05]), (B, 1))
        f_mod = jnp.tile(jnp.array([0.02]), (B, 1))
        dc = fn(c, q, u, f_mod, jnp.zeros(mb.r_size))
        assert dc.shape == (B, 3)


# ---------------------------------------------------------------------------
# Integration: ControlSplines + RhsOde wired together
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_wired_ode_step(self):
        process = _make_process()
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)

        @eqx.filter_jit
        def ode_rhs(t, c, q):
            u = ctrl(t)
            u_flow = u[jnp.array(list(ctrl.flow_indices))]
            return mb(c, q, u_flow, jnp.zeros(0), jnp.zeros(mb.r_size))

        dc = ode_rhs(
            jnp.array(5.0), jnp.array([0.5, 10.0, 1.0]), jnp.array([0.3, -0.15])
        )
        assert dc.shape == (3,)

    def test_flow_index_aligns_with_flow_names(self):
        process = _make_process()
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        assert len(ctrl.flow_indices) == mb.u_flow_size
        assert ctrl.control_names[ctrl.flow_indices[0]] == mb.flow_names[0]

    def test_dV_from_wired_ode_equals_flow_rate(self):
        process = _make_process()
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        t = jnp.array(5.0)
        u = ctrl(t)
        u_flow = u[jnp.array(list(ctrl.flow_indices))]
        dc = mb(
            jnp.array([1.0, 8.0, 1.2]),
            jnp.zeros(2),
            u_flow,
            jnp.zeros(0),
            jnp.zeros(mb.r_size),
        )
        assert float(dc[-1]) == pytest.approx(float(u_flow[0]), rel=1e-4)


# ---------------------------------------------------------------------------
# Module export tests
# ---------------------------------------------------------------------------


class TestModuleExport:
    def test_functions_accessible(self):
        import bp_format.mechanistic as mech

        assert hasattr(mech, "get_control_splines")
        assert hasattr(mech, "get_rhs_ode")
        assert hasattr(mech, "ControlSplines")
        assert hasattr(mech, "RhsOde")

    def test_new_functions_accessible(self):
        import bp_format.mechanistic as mech

        assert hasattr(mech, "extract_discrete_events")
        assert hasattr(mech, "estimate_specific_rates")
        assert hasattr(mech, "integrate_process")

    def test_mechanistic_in_bp_format_namespace(self):
        assert hasattr(bp_format, "mechanistic")

    def test_mechanistic_in_all(self):
        assert "mechanistic" in bp_format.__all__


# ---------------------------------------------------------------------------
# extract_discrete_events tests
# ---------------------------------------------------------------------------


class TestExtractDiscreteEvents:
    def test_sampling_only(self):
        process = _make_process(
            with_controlled_flow=True, with_controlled_pv=False, with_discrete_vc=True
        )
        mb = get_rhs_ode(process)
        events = extract_discrete_events(process, mb)
        assert len(events) == 2
        for ev in events:
            assert ev["kind"] == "sample"
            assert ev["dV"] < 0
            assert ev["Cin"] is None

    def test_bolus_feed(self):
        """Process with a discrete bolus feed."""
        feed_medium = FeedMedium(
            name="bolus_feed",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": FeedMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(value=0.0),
                    is_controlled=False,
                ),
                "glucose": FeedMediumComponent(
                    name="glucose",
                    unit="g/L",
                    concentration=StaticVariable(value=300.0),
                    is_controlled=False,
                ),
            },
        )
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        process.volume.volume_changes["bolus"] = FeedVolumeChange(
            name="bolus",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            feed_medium=feed_medium,
            values=_ts([5.0], [0.1]),
        )
        mb = get_rhs_ode(process)
        events = extract_discrete_events(process, mb)
        assert len(events) == 1
        assert events[0]["kind"] == "bolus_feed"
        assert events[0]["dV"] == pytest.approx(0.1)
        # Cin aligned with reactor_component_state_names
        bio_idx = mb.reactor_component_state_names.index("biomass")
        glu_idx = mb.reactor_component_state_names.index("glucose")
        assert events[0]["Cin"][bio_idx] == pytest.approx(0.0)
        assert events[0]["Cin"][glu_idx] == pytest.approx(300.0)

    def test_no_discrete_events(self):
        process = _make_process(
            with_controlled_flow=True, with_controlled_pv=False, with_discrete_vc=False
        )
        mb = get_rhs_ode(process)
        events = extract_discrete_events(process, mb)
        assert events == []

    def test_events_sorted_by_time(self):
        process = _make_process(
            with_controlled_flow=False, with_controlled_pv=False, with_discrete_vc=True
        )
        mb = get_rhs_ode(process)
        events = extract_discrete_events(process, mb)
        times = [ev["t"] for ev in events]
        assert times == sorted(times)

    def test_cin_length_matches_species(self):
        feed_medium = FeedMedium(
            name="bf",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": FeedMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=StaticVariable(value=0.0),
                    is_controlled=False,
                ),
                "glucose": FeedMediumComponent(
                    name="glucose",
                    unit="g/L",
                    concentration=StaticVariable(value=100.0),
                    is_controlled=False,
                ),
            },
        )
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        process.volume.volume_changes["bolus"] = FeedVolumeChange(
            name="bolus",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            feed_medium=feed_medium,
            values=_ts([3.0], [0.05]),
        )
        mb = get_rhs_ode(process)
        events = extract_discrete_events(process, mb)
        assert len(events[0]["Cin"]) == len(mb.reactor_component_state_names)

    def test_same_timestamp_orders_sample_before_bolus(self):
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        process.volume.volume_changes["sampling"] = SampleVolumeChange(
            name="sampling",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=_ts([5.0], [-0.1]),
        )
        process.volume.volume_changes["bolus"] = FeedVolumeChange(
            name="bolus",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            feed_medium=_make_feed("bolus_feed", glucose_conc=300.0, biomass_conc=0.0),
            values=_ts([5.0], [0.05]),
        )
        mb = get_rhs_ode(process)

        events = extract_discrete_events(process, mb)
        assert len(events) == 2
        assert events[0]["kind"] == "sample"
        assert events[1]["kind"] == "bolus_feed"

    def test_duplicate_sampling_at_same_timestamp_raises(self):
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        process.volume.volume_changes["sampling_a"] = SampleVolumeChange(
            name="sampling_a",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=_ts([5.0], [-0.05]),
        )
        process.volume.volume_changes["sampling_b"] = SampleVolumeChange(
            name="sampling_b",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=_ts([5.0], [-0.03]),
        )
        mb = get_rhs_ode(process)

        with pytest.raises(ValueError, match="At most one discrete event per kind"):
            extract_discrete_events(process, mb)

    def test_duplicate_bolus_at_same_timestamp_raises(self):
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        process.volume.volume_changes["bolus_a"] = FeedVolumeChange(
            name="bolus_a",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            feed_medium=_make_feed("bolus_a_feed", glucose_conc=50.0, biomass_conc=0.0),
            values=_ts([5.0], [0.02]),
        )
        process.volume.volume_changes["bolus_b"] = FeedVolumeChange(
            name="bolus_b",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            feed_medium=_make_feed(
                "bolus_b_feed", glucose_conc=100.0, biomass_conc=0.0
            ),
            values=_ts([5.0], [0.03]),
        )
        mb = get_rhs_ode(process)

        with pytest.raises(ValueError, match="At most one discrete event per kind"):
            extract_discrete_events(process, mb)


# ---------------------------------------------------------------------------
# estimate_specific_rates tests
# ---------------------------------------------------------------------------


class TestEstimateSpecificRates:
    def test_constant_q_exponential_growth(self):
        """With constant q and no feed, exponential growth gives back q."""
        # Simple batch: dX/dt = q_X * X, dS/dt = q_S * X, V=1 constant
        q_X_true = 0.3
        q_S_true = -0.1
        t = np.linspace(0, 10, 50)
        X0, S0 = 1.0, 10.0
        X = X0 * np.exp(q_X_true * t)
        S = S0 + (q_S_true / q_X_true) * X0 * (np.exp(q_X_true * t) - 1)
        S = np.maximum(S, 0.0)

        process = _make_batch_process()
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)

        state_splines = {
            "biomass": make_interpax_spline(t, X),
            "glucose": make_interpax_spline(t, S),
        }

        t_eval = np.linspace(0.5, 9.5, 20)
        q_est = estimate_specific_rates(process, ctrl, mb, state_splines, t_eval)

        # q_X should be ~0.3, q_S should be ~-0.1
        assert q_est.shape == (20, 2)
        np.testing.assert_allclose(q_est[:, 0], q_X_true, rtol=0.05)
        np.testing.assert_allclose(q_est[:, 1], q_S_true, rtol=0.05)

    def test_build_q_func_q_r_partition_no_overlap(self):
        q_x_true = 0.3
        r_s = -0.2
        t = np.linspace(0.0, 10.0, 101)
        x0, s0 = 1.0, 10.0
        x = x0 * np.exp(q_x_true * t)
        s = s0 + r_s * t

        process = _make_batch_process()
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        state_splines = {
            "biomass": make_interpax_spline(t, x),
            "glucose": make_interpax_spline(t, s),
        }

        q_func = build_q_func(
            process,
            ctrl,
            mb,
            state_splines,
            q_state_indices=[0],
            r_state_indices=[1],
        )
        q_t = q_func(4.0)
        assert float(q_t[0]) == pytest.approx(q_x_true, rel=0.05)
        assert float(q_t[1]) == pytest.approx(0.0, abs=1e-8)

    def test_build_q_func_overlap_requires_r_func(self):
        process = _make_batch_process()
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        state_splines = build_state_splines(process, mb)

        with pytest.raises(
            ValueError, match="Overlapping q/r state indices require r_func"
        ):
            build_q_func(
                process,
                ctrl,
                mb,
                state_splines,
                q_state_indices=[0, 1],
                r_state_indices=[1],
            )

    def test_build_q_func_r_func_requires_explicit_indices(self):
        process = _make_batch_process()
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        state_splines = build_state_splines(process, mb)

        def r_func(_t):
            return jnp.zeros(mb.r_size)

        with pytest.raises(
            ValueError, match="r_func requires explicit q_state_indices"
        ):
            build_q_func(
                process,
                ctrl,
                mb,
                state_splines,
                r_func=r_func,
            )

    def test_build_q_func_overlap_with_r_func(self):
        q_x_true = 0.25
        q_s_true = -0.1
        r_s = -0.2
        t = np.linspace(0.0, 10.0, 201)
        x0, s0 = 1.0, 10.0
        x = x0 * np.exp(q_x_true * t)
        s = s0 + (q_s_true / q_x_true) * x0 * (np.exp(q_x_true * t) - 1.0) + r_s * t

        process = _make_batch_process()
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        state_splines = {
            "biomass": make_interpax_spline(t, x),
            "glucose": make_interpax_spline(t, s),
        }

        def r_func(_t):
            return jnp.array([0.0, r_s])

        q_func = build_q_func(
            process,
            ctrl,
            mb,
            state_splines,
            q_state_indices=[0, 1],
            r_state_indices=[1],
            r_func=r_func,
        )
        q_t = q_func(5.0)
        assert float(q_t[0]) == pytest.approx(q_x_true, rel=0.05)
        assert float(q_t[1]) == pytest.approx(q_s_true, rel=0.1)

    def test_legacy_conc_splines_keyword_still_supported(self):
        t = np.linspace(0.0, 10.0, 101)
        x = np.exp(0.3 * t)
        s = 10.0 - t
        process = _make_batch_process()
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        conc_splines = {
            "biomass": make_interpax_spline(t, x),
            "glucose": make_interpax_spline(t, s),
        }

        q_func = build_q_func(process, ctrl, mb, conc_splines=conc_splines)
        q_est = estimate_specific_rates(
            process,
            ctrl,
            mb,
            conc_splines=conc_splines,
            t_eval=t,
        )

        assert q_func(5.0).shape == (mb.q_size,)
        assert q_est.shape == (len(t), mb.q_size)

    def test_build_rates_func_default_infers_pv_r_from_state_splines(self):
        process = _make_process(
            with_controlled_flow=False,
            with_controlled_pv=False,
            with_uncontrolled_pv=True,
        )
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        state_splines = build_state_splines(process, mb)

        rates_func = build_rates_func(process, ctrl, mb, state_splines)
        t_eval = 5.0
        q, r = rates_func(t_eval, jnp.zeros(mb.c_size), ctrl(t_eval))

        pv_name = mb.process_variable_state_names[0]
        expected_pv_r = state_splines[pv_name].derivative()(t_eval)

        assert q.shape == (mb.q_size,)
        assert r.shape == (mb.r_size,)
        np.testing.assert_allclose(r[: mb.n_reactor_states], 0.0, atol=1e-10)
        assert float(r[mb.n_reactor_states]) == pytest.approx(
            float(expected_pv_r), rel=1e-6
        )

    def test_build_rates_func_default_without_pv_keeps_r_zero(self):
        process = _make_batch_process()
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        state_splines = build_state_splines(process, mb)

        rates_func = build_rates_func(process, ctrl, mb, state_splines)
        q, r = rates_func(2.0, jnp.zeros(mb.c_size), ctrl(2.0))

        assert q.shape == (mb.q_size,)
        np.testing.assert_allclose(r, 0.0, atol=1e-10)

    def test_build_rates_func_legacy_conc_splines_keyword_still_supported(self):
        process = _make_process(
            with_controlled_flow=False,
            with_controlled_pv=False,
            with_uncontrolled_pv=True,
        )
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        conc_splines = build_state_splines(process, mb)

        rates_func = build_rates_func(process, ctrl, mb, conc_splines=conc_splines)
        t_eval = 5.0
        q, r = rates_func(t_eval, jnp.zeros(mb.c_size), ctrl(t_eval))

        pv_name = mb.process_variable_state_names[0]
        expected_pv_r = conc_splines[pv_name].derivative()(t_eval)

        assert q.shape == (mb.q_size,)
        assert r.shape == (mb.r_size,)
        np.testing.assert_allclose(r[: mb.n_reactor_states], 0.0, atol=1e-10)
        assert float(r[mb.n_reactor_states]) == pytest.approx(
            float(expected_pv_r), rel=1e-6
        )


# ---------------------------------------------------------------------------
# integrate_process tests
# ---------------------------------------------------------------------------


class TestIntegrateProcess:
    def _setup_batch_integration(self):
        """Set up a batch process with known constant q for testing."""
        q_X = 0.3
        q_S = -0.1

        process = _make_batch_process()
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)

        q_arr = jnp.array([q_X, q_S])

        def q_func(t):
            return q_arr

        rates_func = _wrap_q_as_rates(mb, q_func)
        return process, ctrl, mb, rates_func, q_X, q_S

    def test_batch_accuracy(self):
        """Forward integration with known q recovers analytical solution."""
        process, ctrl, mb, rates_func, q_X, q_S = self._setup_batch_integration()
        t_eval = np.linspace(0, 10, 50)

        result = integrate_process(process, ctrl, mb, rates_func, t_eval)

        # Analytical solution
        X0 = 0.5  # from _make_batch_process
        S0 = 10.0
        X_true = X0 * np.exp(q_X * result["t"])
        S_true = S0 + (q_S / q_X) * X0 * (np.exp(q_X * result["t"]) - 1)

        # RMSE should be very small
        rmse_X = np.sqrt(np.mean((result["c"][:, 0] - X_true) ** 2))
        rmse_S = np.sqrt(np.mean((result["c"][:, 1] - S_true) ** 2))
        assert rmse_X < 1e-3, f"Biomass RMSE = {rmse_X}"
        assert rmse_S < 1e-3, f"Glucose RMSE = {rmse_S}"

    def test_rates_func_can_use_state_argument(self):
        process = _make_batch_process()
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        t_eval = np.linspace(0, 4, 40)

        def rates_func(_t, state, controls):
            del controls
            q = jnp.array([0.2 * state[0], 0.0])
            r = jnp.zeros(mb.r_size)
            return q, r

        result = integrate_process(process, ctrl, mb, rates_func, t_eval)
        assert float(result["c"][-1, 0]) > float(result["c"][0, 0])

    def test_rates_func_can_use_controls_argument(self):
        process = _make_process(with_controlled_flow=True, with_controlled_pv=False)
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        t_eval = np.linspace(0, 10, 60)

        def rates_from_controls(_t, _state, controls):
            q = jnp.array([jnp.maximum(controls[0], 0.0), 0.0])
            r = jnp.zeros(mb.r_size)
            return q, r

        def zero_rates(_t, _state, _controls):
            return jnp.zeros(mb.q_size), jnp.zeros(mb.r_size)

        out_controls = integrate_process(process, ctrl, mb, rates_from_controls, t_eval)
        out_zero = integrate_process(process, ctrl, mb, zero_rates, t_eval)

        assert float(out_controls["c"][-1, 0]) > float(out_zero["c"][-1, 0])

    def test_integrate_process_requires_rates_func(self):
        process, ctrl, mb, _, _, _ = self._setup_batch_integration()
        t_eval = np.linspace(0, 2, 5)

        with pytest.raises(ValueError, match="rates_func is required"):
            integrate_process(process, ctrl, mb, None, t_eval)

    def test_volume_constant_in_batch(self):
        """Volume should stay constant in batch mode."""
        process, ctrl, mb, rates_func, _, _ = self._setup_batch_integration()
        t_eval = np.linspace(0, 10, 20)
        result = integrate_process(process, ctrl, mb, rates_func, t_eval)
        np.testing.assert_allclose(result["V"], 1.0, atol=1e-6)

    def test_with_sampling_events(self):
        """Volume drops at sampling events, concentrations stay continuous."""
        process = _make_process(
            with_controlled_flow=True, with_controlled_pv=False, with_discrete_vc=True
        )
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)

        zero_rates = _wrap_q_as_rates(mb, lambda t: jnp.zeros(mb.q_size))
        t_eval = np.linspace(0, 20, 100)

        result = integrate_process(process, ctrl, mb, zero_rates, t_eval)
        assert "t" in result
        assert "c" in result
        assert "V" in result

        # Volume should be lower after sampling
        V_early = result["V"][result["t"] < 4.0]
        V_late = result["V"][result["t"] > 11.0]
        assert np.mean(V_late) > np.mean(V_early) - 0.2  # feed increases volume

    def test_output_format(self):
        """Check returned dict has expected keys and shapes."""
        process, ctrl, mb, rates_func, _, _ = self._setup_batch_integration()
        t_eval = np.linspace(0, 10, 30)
        result = integrate_process(process, ctrl, mb, rates_func, t_eval)
        assert {"t", "c", "V", "stats"} == set(result.keys())
        assert result["c"].shape[1] == mb.q_size
        assert result["V"].shape[0] == result["c"].shape[0]
        assert result["t"].shape[0] == result["c"].shape[0]

    def test_default_settings_accuracy(self):
        """Default rtol/atol settings produce RMSE < 1e-3 for a simple batch.

        Default tolerances are rtol=1e-4, atol=1e-6 (float32-friendly).
        """
        process, ctrl, mb, rates_func, q_X, q_S = self._setup_batch_integration()
        t_eval = np.linspace(0, 10, 100)

        result = integrate_process(process, ctrl, mb, rates_func, t_eval)

        X0 = 0.5
        S0 = 10.0
        X_true = X0 * np.exp(q_X * result["t"])
        S_true = S0 + (q_S / q_X) * X0 * (np.exp(q_X * result["t"]) - 1)

        rmse = np.sqrt(
            np.mean(
                (result["c"][:, 0] - X_true) ** 2 + (result["c"][:, 1] - S_true) ** 2
            )
        )
        assert rmse < 1e-3, f"Overall RMSE = {rmse}"

    def test_fedbatch_volume_increases(self):
        """Fed-batch integration should show increasing volume."""
        process = _make_process(with_controlled_flow=True, with_controlled_pv=False)
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)

        zero_rates = _wrap_q_as_rates(mb, lambda t: jnp.zeros(mb.q_size))
        t_eval = np.linspace(0, 20, 50)
        result = integrate_process(process, ctrl, mb, zero_rates, t_eval)

        # Volume should increase over time due to feed
        assert result["V"][-1] > result["V"][0]

    def test_discrete_bolus_event_applies_mixing_jump(self):
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        process.volume.volume_changes["bolus"] = FeedVolumeChange(
            name="bolus",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            feed_medium=_make_feed("bolus_feed", glucose_conc=300.0, biomass_conc=0.0),
            values=_ts([5.0], [0.1]),
        )

        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        zero_rates = _wrap_q_as_rates(mb, lambda t: jnp.zeros(mb.q_size))
        t_eval = jnp.array([0.0, 4.9, 5.0, 5.1])
        result = integrate_process(process, ctrl, mb, zero_rates, t_eval)

        V_new = 1.1
        X_new = (0.5 * 1.0 + 0.0 * 0.1) / V_new
        S_new = (10.0 * 1.0 + 300.0 * 0.1) / V_new

        # pre-event at t_b: state is unchanged
        assert float(result["c"][1, 0]) == pytest.approx(0.5, rel=1e-6)
        assert float(result["c"][1, 1]) == pytest.approx(10.0, rel=1e-6)
        assert float(result["V"][1]) == pytest.approx(1.0, rel=1e-6)
        assert float(result["c"][2, 0]) == pytest.approx(0.5, rel=1e-6)
        assert float(result["c"][2, 1]) == pytest.approx(10.0, rel=1e-6)
        assert float(result["V"][2]) == pytest.approx(1.0, rel=1e-6)
        # post-event at t_b + eps: mixed state
        assert float(result["c"][3, 0]) == pytest.approx(X_new, rel=1e-3)
        assert float(result["c"][3, 1]) == pytest.approx(S_new, rel=1e-3)
        assert float(result["V"][3]) == pytest.approx(V_new, rel=1e-3)

    def test_discrete_sampling_event_preserves_concentrations(self):
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        process.volume.volume_changes["sampling"] = SampleVolumeChange(
            name="sampling",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=_ts([5.0], [-0.1]),
        )

        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        zero_rates = _wrap_q_as_rates(mb, lambda t: jnp.zeros(mb.q_size))
        t_eval = jnp.array([0.0, 4.9, 5.0, 5.1])
        result = integrate_process(process, ctrl, mb, zero_rates, t_eval)

        # pre-event at t_b: volume and concentrations unchanged
        assert float(result["c"][1, 0]) == pytest.approx(0.5, rel=1e-6)
        assert float(result["c"][1, 1]) == pytest.approx(10.0, rel=1e-6)
        assert float(result["V"][1]) == pytest.approx(1.0, rel=1e-6)
        assert float(result["c"][2, 0]) == pytest.approx(0.5, rel=1e-6)
        assert float(result["c"][2, 1]) == pytest.approx(10.0, rel=1e-6)
        assert float(result["V"][2]) == pytest.approx(1.0, rel=1e-6)
        # post-event at t_b + eps: volume reduced, concentrations preserved
        assert float(result["c"][3, 0]) == pytest.approx(0.5, rel=1e-3)
        assert float(result["c"][3, 1]) == pytest.approx(10.0, rel=1e-3)
        assert float(result["V"][3]) == pytest.approx(0.9, rel=1e-3)

    def test_discrete_events_affect_only_reactor_block_with_pv_state(self):
        process = _make_process(
            with_controlled_flow=False,
            with_controlled_pv=False,
            with_uncontrolled_pv=True,
        )
        process.volume.volume_changes["bolus"] = FeedVolumeChange(
            name="bolus",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            feed_medium=_make_feed("bolus_feed", glucose_conc=300.0, biomass_conc=0.0),
            values=_ts([5.0], [0.1]),
        )

        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        assert mb.process_variable_state_names == ("dissolved_O2",)

        zero_rates = _wrap_q_as_rates(mb, lambda t: jnp.zeros(mb.q_size))
        t_eval = jnp.array([4.9, 5.0, 5.1])
        result = integrate_process(process, ctrl, mb, zero_rates, t_eval)

        pv_idx = mb.pv_indices[0]
        assert float(result["c"][0, pv_idx]) == pytest.approx(100.0, rel=1e-6)
        assert float(result["c"][1, pv_idx]) == pytest.approx(100.0, rel=1e-6)
        assert float(result["c"][2, pv_idx]) == pytest.approx(100.0, rel=1e-6)

    def test_same_timestamp_sampling_then_bolus_mixing(self):
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        process.volume.volume_changes["sampling"] = SampleVolumeChange(
            name="sampling",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=_ts([5.0], [-0.2]),
        )
        process.volume.volume_changes["bolus"] = FeedVolumeChange(
            name="bolus",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            feed_medium=_make_feed("bolus_feed", glucose_conc=300.0, biomass_conc=0.0),
            values=_ts([5.0], [0.1]),
        )

        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        zero_rates = _wrap_q_as_rates(mb, lambda t: jnp.zeros(mb.q_size))
        t_eval = jnp.array([0.0, 4.9, 5.0, 5.1])
        result = integrate_process(process, ctrl, mb, zero_rates, t_eval)

        # sample first (1.0 -> 0.8), then bolus (+0.1 with mixing, final V=0.9)
        V_expected = 0.9
        X_expected = (0.5 * 0.8 + 0.0 * 0.1) / V_expected
        G_expected = (10.0 * 0.8 + 300.0 * 0.1) / V_expected

        # pre-event at t_b: state unchanged
        assert float(result["V"][2]) == pytest.approx(1.0, rel=1e-6)
        assert float(result["c"][2, 0]) == pytest.approx(0.5, rel=1e-6)
        assert float(result["c"][2, 1]) == pytest.approx(10.0, rel=1e-6)
        # post-event at t_b + eps: sample-then-bolus mass balance applied
        assert float(result["V"][3]) == pytest.approx(V_expected, rel=1e-3)
        assert float(result["c"][3, 0]) == pytest.approx(X_expected, rel=1e-3)
        assert float(result["c"][3, 1]) == pytest.approx(G_expected, rel=1e-3)

    def test_event_at_t_end_output_is_pre_event(self):
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        process.time_axis = TimeAxis(
            unit="hours",
            start=0.0,
            end=20.0,
            time_reference="inoculation",
        )
        process.volume.volume_changes["sampling"] = SampleVolumeChange(
            name="sampling",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=_ts([20.0], [-0.2]),
        )

        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        zero_rates = _wrap_q_as_rates(mb, lambda t: jnp.zeros(mb.q_size))
        t_eval = jnp.array([19.9, 20.0])
        result = integrate_process(process, ctrl, mb, zero_rates, t_eval)

        i_pre = int(jnp.argmin(jnp.abs(result["t"] - 19.9)))
        i_end = int(jnp.argmin(jnp.abs(result["t"] - 20.0)))
        # Left-continuous: exact event timestamp is pre-event.
        assert float(result["V"][i_pre]) == pytest.approx(1.0, rel=1e-6)
        assert float(result["V"][i_end]) == pytest.approx(1.0, rel=1e-6)

    def test_event_at_t_start_output_is_pre_event(self):
        """Left-continuous: bolus at t_start gives pre-event state at t_start."""
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        process.volume.volume_changes["bolus"] = FeedVolumeChange(
            name="bolus",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            feed_medium=_make_feed("bolus_feed", glucose_conc=300.0, biomass_conc=0.0),
            values=_ts([0.0], [0.1]),
        )

        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        zero_rates = _wrap_q_as_rates(mb, lambda t: jnp.zeros(mb.q_size))
        t_eval = jnp.array([0.0, 0.1, 1.0])
        result = integrate_process(process, ctrl, mb, zero_rates, t_eval)

        V_post = 1.1
        X_post = (0.5 * 1.0 + 0.0 * 0.1) / V_post
        S_post = (10.0 * 1.0 + 300.0 * 0.1) / V_post

        # pre-event at t_start: output must be the initial state
        assert float(result["V"][0]) == pytest.approx(1.0, rel=1e-6)
        assert float(result["c"][0, 0]) == pytest.approx(0.5, rel=1e-6)
        assert float(result["c"][0, 1]) == pytest.approx(10.0, rel=1e-6)
        # post-event visible from next point onward
        assert float(result["V"][1]) == pytest.approx(V_post, rel=1e-3)
        assert float(result["c"][1, 0]) == pytest.approx(X_post, rel=1e-3)
        assert float(result["c"][1, 1]) == pytest.approx(S_post, rel=1e-3)

    def test_pseudospace_matches_segmented_with_sampling_and_bolus(self):
        """Single-pass pseudo-space integration should match segmented integration."""
        process = _make_process(with_controlled_flow=True, with_controlled_pv=False)

        process.volume.volume_changes["sampling"] = SampleVolumeChange(
            name="sampling",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=_ts([4.0, 9.0, 14.0, 18.0], [-0.003, -0.004, -0.003, -0.002]),
        )
        process.volume.volume_changes["bolus"] = FeedVolumeChange(
            name="bolus",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            feed_medium=_make_feed("bolus_feed", glucose_conc=25.0, biomass_conc=0.0),
            values=_ts([6.0, 12.0, 16.0], [0.003, 0.002, 0.002]),
        )

        t_obs = jnp.linspace(0.0, 20.0, 121)
        biomass = 0.4 * jnp.exp(0.08 * t_obs)
        glucose = jnp.maximum(40.0 - 1.4 * t_obs - 0.03 * (t_obs**2), 0.5)
        process.reactor_medium.components["biomass"].concentration = _ts(t_obs, biomass)
        process.reactor_medium.components["glucose"].concentration = _ts(t_obs, glucose)

        _apply_pseudobatch_transform(process)

        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        state_splines = build_state_splines(process, mb)
        q_func = build_q_func(process, ctrl, mb, state_splines)
        rates_func = _wrap_q_as_rates(mb, q_func)

        t_eval = jnp.linspace(0.0, 20.0, 181)
        ref = integrate_process(
            process,
            ctrl,
            mb,
            rates_func,
            t_eval,
        )
        pseudo = integrate_process_pseudospace(
            process,
            ctrl,
            mb,
            rates_func,
            t_eval,
            state_splines=state_splines,
        )

        c_ref = _sample_on_observation_grid(ref["t"], ref["c"], t_eval)
        V_ref = _sample_on_observation_grid(ref["t"], ref["V"][:, None], t_eval)[:, 0]
        c_pseudo = _sample_on_observation_grid(pseudo["t"], pseudo["c"], t_eval)
        V_pseudo = _sample_on_observation_grid(
            pseudo["t"], pseudo["V"][:, None], t_eval
        )[:, 0]

        max_c_diff = float(jnp.max(jnp.abs(c_ref - c_pseudo)))
        max_v_diff = float(jnp.max(jnp.abs(V_ref - V_pseudo)))

        assert max_c_diff < 20.0
        assert max_v_diff < 1e-5

        for t_bolus in [6.0, 12.0, 16.0]:
            t_pre = t_bolus - 0.01
            t_post = t_bolus + 0.01
            i_pre = int(jnp.argmin(jnp.abs(t_eval - t_pre)))
            i_post = int(jnp.argmin(jnp.abs(t_eval - t_post)))
            assert float(jnp.max(jnp.abs(c_ref[i_pre] - c_pseudo[i_pre]))) < 20.0
            assert float(jnp.max(jnp.abs(c_ref[i_post] - c_pseudo[i_post]))) < 20.0

    def test_pseudospace_matches_segmented_with_same_time_sampling_and_bolus(self):
        process = _make_process(with_controlled_flow=True, with_controlled_pv=False)

        process.volume.volume_changes["sampling"] = SampleVolumeChange(
            name="sampling",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=_ts([6.0, 12.0], [-0.1, -0.2]),
        )
        process.volume.volume_changes["bolus"] = FeedVolumeChange(
            name="bolus",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            feed_medium=_make_feed("bolus_feed", glucose_conc=25.0, biomass_conc=0.0),
            values=_ts([6.0, 12.0], [0.2, 0.1]),
        )

        t_obs = jnp.linspace(0.0, 20.0, 121)
        biomass = 0.4 * jnp.exp(0.08 * t_obs)
        glucose = jnp.maximum(40.0 - 1.4 * t_obs - 0.03 * (t_obs**2), 0.5)
        process.reactor_medium.components["biomass"].concentration = _ts(t_obs, biomass)
        process.reactor_medium.components["glucose"].concentration = _ts(t_obs, glucose)

        _apply_pseudobatch_transform(process)

        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        state_splines = build_state_splines(process, mb)
        q_func = build_q_func(process, ctrl, mb, state_splines)
        rates_func = _wrap_q_as_rates(mb, q_func)

        t_eval = jnp.linspace(0.0, 20.0, 181)
        ref = integrate_process(
            process,
            ctrl,
            mb,
            rates_func,
            t_eval,
        )
        pseudo = integrate_process_pseudospace(
            process,
            ctrl,
            mb,
            rates_func,
            t_eval,
            state_splines=state_splines,
        )

        c_ref = _sample_on_observation_grid(ref["t"], ref["c"], t_eval)
        V_ref = _sample_on_observation_grid(ref["t"], ref["V"][:, None], t_eval)[:, 0]
        c_pseudo = _sample_on_observation_grid(pseudo["t"], pseudo["c"], t_eval)
        V_pseudo = _sample_on_observation_grid(
            pseudo["t"], pseudo["V"][:, None], t_eval
        )[:, 0]

        max_c_diff = float(jnp.max(jnp.abs(c_ref - c_pseudo)))
        max_v_diff = float(jnp.max(jnp.abs(V_ref - V_pseudo)))

        # Tolerance covers the numerical divergence between real-space and
        # pseudo-space integration paths under the physics-correct
        # sample-compensation ADF (dense piecewise-linear rather than the
        # earlier step-table encoding).
        assert max_c_diff < 12.0
        assert max_v_diff < 1e-4

    def test_pseudospace_volume_same_time_bolus_matches_segmented_at_tb(self):
        """Left-continuous: exact event timestamp is pre-event; pseudo-space matches."""
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        process.volume.volume_changes["sampling"] = SampleVolumeChange(
            name="sampling",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=_ts([5.0], [-0.2]),
        )
        process.volume.volume_changes["bolus"] = FeedVolumeChange(
            name="bolus",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            feed_medium=_make_feed("bolus_feed", glucose_conc=25.0, biomass_conc=0.0),
            values=_ts([5.0], [0.1]),
        )

        t_obs = jnp.linspace(0.0, 20.0, 121)
        biomass = 0.4 * jnp.exp(0.08 * t_obs)
        glucose = jnp.maximum(40.0 - 1.4 * t_obs - 0.03 * (t_obs**2), 0.5)
        process.reactor_medium.components["biomass"].concentration = _ts(t_obs, biomass)
        process.reactor_medium.components["glucose"].concentration = _ts(t_obs, glucose)

        _apply_pseudobatch_transform(process)

        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        state_splines = build_state_splines(process, mb)
        q_func = build_q_func(process, ctrl, mb, state_splines)
        rates_func = _wrap_q_as_rates(mb, q_func)

        t_b = 5.0
        # Tiny probes check left/right event sides without relying on spline knots.
        eps = 5e-4
        t_eval = jnp.array([0.0, t_b - eps, t_b, t_b + eps, 10.0, 20.0])

        ref = integrate_process(
            process,
            ctrl,
            mb,
            rates_func,
            t_eval,
        )
        pseudo = integrate_process_pseudospace(
            process,
            ctrl,
            mb,
            rates_func,
            t_eval,
            state_splines=state_splines,
        )

        v_ref = _sample_on_observation_grid(ref["t"], ref["V"][:, None], t_eval)[:, 0]
        v_pseudo = _sample_on_observation_grid(
            pseudo["t"], pseudo["V"][:, None], t_eval
        )[:, 0]
        c_ref = _sample_on_observation_grid(ref["t"], ref["c"], t_eval)
        c_pseudo = _sample_on_observation_grid(pseudo["t"], pseudo["c"], t_eval)

        v_expected = 1.0 - 0.2 + 0.1  # initial - sample + bolus
        # Left-continuous contract: exact event timestamp is pre-event.
        assert float(v_ref[1]) == pytest.approx(1.0, rel=1e-6)  # t_b - eps
        assert float(v_ref[2]) == pytest.approx(1.0, rel=1e-6)  # t_b: pre-event
        assert float(v_ref[3]) == pytest.approx(v_expected, rel=1e-6)  # t_b + eps

        # Pseudo-space must match the same semantics.
        assert float(v_pseudo[1]) == pytest.approx(float(v_ref[1]), abs=1e-6)
        assert float(v_pseudo[2]) == pytest.approx(float(v_ref[2]), abs=1e-6)
        assert float(v_pseudo[3]) == pytest.approx(float(v_ref[3]), abs=1e-6)

        # Use pre-event values (index 2, t_b pre-event) for mass balance.
        x_pre = float(c_ref[2, 0])
        s_pre = float(c_ref[2, 1])
        x_post_expected = (x_pre * 0.8 + 0.0 * 0.1) / v_expected
        s_post_expected = (s_pre * 0.8 + 25.0 * 0.1) / v_expected

        # Post-event jump appears at index 3 (t_b + eps).
        assert float(c_ref[3, 0]) == pytest.approx(x_post_expected, rel=1e-3, abs=1e-3)
        assert float(c_ref[3, 1]) == pytest.approx(s_post_expected, rel=1e-3, abs=1e-3)

        # Pseudo-space vs honest c-space concentration agreement, away
        # from the discontinuity. The previous absolute bound (abs=2e-2)
        # was tuned when the segmented path also used the spline-X_active
        # trick. With that trick removed, `integrate_process` is honest
        # c-space integration while `integrate_process_pseudospace` still
        # anchors to splines via the c* transform, so the two paths
        # diverge by ~1–2% on each species. Switched to a relative bound
        # to cover both species (biomass ~0.6 g/L and glucose ~30 g/L)
        # at the same percentage tolerance.
        assert float(c_pseudo[1, 0]) == pytest.approx(float(c_ref[1, 0]), rel=5e-2)
        assert float(c_pseudo[3, 0]) == pytest.approx(float(c_ref[3, 0]), rel=5e-2)
        assert float(c_pseudo[1, 1]) == pytest.approx(float(c_ref[1, 1]), rel=5e-2)
        assert float(c_pseudo[3, 1]) == pytest.approx(float(c_ref[3, 1]), rel=5e-2)

    def test_pseudospace_same_time_sampling_and_bolus_from_transformed_timeseries(self):
        """Mechanistic code reads transform from the process-level bundle."""
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        process.volume.volume_changes["sampling"] = SampleVolumeChange(
            name="sampling",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=_ts([5.0], [-0.2]),
        )
        process.volume.volume_changes["bolus"] = FeedVolumeChange(
            name="bolus",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            feed_medium=_make_feed("bolus_feed", glucose_conc=25.0, biomass_conc=0.0),
            values=_ts([5.0], [0.1]),
        )

        t_obs = jnp.linspace(0.0, 20.0, 121)
        biomass = 0.4 * jnp.exp(0.08 * t_obs)
        glucose = jnp.maximum(40.0 - 1.4 * t_obs - 0.03 * (t_obs**2), 0.5)
        process.reactor_medium.components["biomass"].concentration = _ts(t_obs, biomass)
        process.reactor_medium.components["glucose"].concentration = _ts(t_obs, glucose)

        _apply_pseudobatch_transform(process)

        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        state_splines = build_state_splines(process, mb)
        q_func = build_q_func(process, ctrl, mb, state_splines)
        rates_func = _wrap_q_as_rates(mb, q_func)

        t_b = 5.0
        eps = 5e-4
        t_eval = jnp.array([0.0, t_b - eps, t_b, t_b + eps, 10.0, 20.0])

        ref = integrate_process(
            process,
            ctrl,
            mb,
            rates_func,
            t_eval,
        )
        pseudo = integrate_process_pseudospace(
            process,
            ctrl,
            mb,
            rates_func,
            t_eval,
            state_splines=state_splines,
        )

        v_ref = _sample_on_observation_grid(ref["t"], ref["V"][:, None], t_eval)[:, 0]
        v_pseudo = _sample_on_observation_grid(
            pseudo["t"], pseudo["V"][:, None], t_eval
        )[:, 0]
        c_ref = _sample_on_observation_grid(ref["t"], ref["c"], t_eval)
        c_pseudo = _sample_on_observation_grid(pseudo["t"], pseudo["c"], t_eval)

        # Use relative concentration tolerance for the same reason as the
        # previous test: honest c-space and pseudo-space integration solve
        # different state coordinates once the old spline-X_active shortcut is
        # gone.
        assert float(v_pseudo[1]) == pytest.approx(float(v_ref[1]), abs=1e-6)
        assert float(v_pseudo[2]) == pytest.approx(float(v_ref[2]), abs=1e-6)
        assert float(v_pseudo[3]) == pytest.approx(float(v_ref[3]), abs=1e-6)
        assert float(c_pseudo[1, 0]) == pytest.approx(float(c_ref[1, 0]), rel=5e-2)
        assert float(c_pseudo[3, 0]) == pytest.approx(float(c_ref[3, 0]), rel=5e-2)
        assert float(c_pseudo[1, 1]) == pytest.approx(float(c_ref[1, 1]), rel=5e-2)
        assert float(c_pseudo[3, 1]) == pytest.approx(float(c_ref[3, 1]), rel=5e-2)

    def test_integrate_process_fails_when_event_empties_reactor(self):
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        process.volume.volume_changes["sampling"] = SampleVolumeChange(
            name="sampling",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=_ts([5.0], [-2.0]),
        )
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)

        def q_func(t):
            del t
            return jnp.zeros(mb.n_reactor_states)

        rates_func = _wrap_q_as_rates(mb, q_func)
        with pytest.raises(Exception, match="reactor volume"):
            integrate_process(
                process,
                ctrl,
                mb,
                rates_func,
                jnp.array([0.0, 5.0, 10.0]),
            )

    def test_pseudobatch_transform_fails_when_sampling_empties_reactor(self):
        process = _make_process(with_controlled_flow=False, with_controlled_pv=False)
        process.volume.volume_changes["sampling"] = SampleVolumeChange(
            name="sampling",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=_ts([5.0], [-1.0]),
        )
        with pytest.raises(ValueError, match="reactor volume"):
            _apply_pseudobatch_transform(process)

    def test_pseudospace_runs_without_transform_metadata(self):
        process = _make_process(with_controlled_flow=True, with_controlled_pv=False)
        process.volume.volume_changes["bolus"] = FeedVolumeChange(
            name="bolus",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            feed_medium=_make_feed("bolus_feed", glucose_conc=200.0, biomass_conc=0.0),
            values=_ts([6.0], [0.05]),
        )

        t_obs = jnp.linspace(0.0, 20.0, 81)
        biomass = 0.4 * jnp.exp(0.08 * t_obs)
        glucose = jnp.maximum(40.0 - 1.4 * t_obs - 0.03 * (t_obs**2), 0.5)
        process.reactor_medium.components["biomass"].concentration = _ts(t_obs, biomass)
        process.reactor_medium.components["glucose"].concentration = _ts(t_obs, glucose)

        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        rates_func = _wrap_q_as_rates(mb, lambda t: jnp.array([0.1, -0.2]))
        t_eval_coarse = jnp.linspace(0.0, 20.0, 41)
        out_coarse = integrate_process_pseudospace(
            process=process,
            ctrl=ctrl,
            mb=mb,
            rates_func=rates_func,
            t_eval=t_eval_coarse,
        )
        out_dense = integrate_process_pseudospace(
            process=process,
            ctrl=ctrl,
            mb=mb,
            rates_func=rates_func,
            t_eval=t_obs,
        )
        ref = integrate_process(
            process=process,
            ctrl=ctrl,
            mb=mb,
            rates_func=rates_func,
            t_eval=t_eval_coarse,
        )

        assert out_coarse["c"].shape == (len(t_eval_coarse), mb.q_size)
        assert out_coarse["V"].shape == (len(t_eval_coarse),)
        assert jnp.all(jnp.isfinite(out_coarse["c"]))
        assert jnp.all(jnp.isfinite(out_coarse["V"]))

        y_dense_on_coarse = _interp_on_grid(
            out_dense["t"], out_dense["c"], t_eval_coarse
        )
        max_grid_sensitivity = float(
            jnp.max(jnp.abs(out_coarse["c"] - y_dense_on_coarse))
        )
        assert max_grid_sensitivity < 1.0

        V_dense_on_coarse = jnp.interp(t_eval_coarse, out_dense["t"], out_dense["V"])
        max_V_grid_sensitivity = float(
            jnp.max(jnp.abs(out_coarse["V"] - V_dense_on_coarse))
        )
        assert max_V_grid_sensitivity < 1e-4

        y_ref = _interp_on_grid(ref["t"], ref["c"], t_eval_coarse)
        max_ref_diff = float(jnp.max(jnp.abs(out_coarse["c"] - y_ref)))
        assert max_ref_diff < 25.0

        V_ref = jnp.interp(t_eval_coarse, ref["t"], ref["V"])
        max_V_ref_diff = float(jnp.max(jnp.abs(out_coarse["V"] - V_ref)))
        assert max_V_ref_diff < 1e-4

    def test_pseudospace_uses_rates_func_signature(self):
        process = _make_batch_process()
        ctrl = get_control_splines(process)
        mb = get_rhs_ode(process)
        t_eval = jnp.linspace(0.0, 3.0, 21)

        def rates_func(_t, state, controls):
            del controls
            q = jnp.array([0.1 + 0.01 * state[0], 0.0])
            r = jnp.zeros(mb.r_size)
            return q, r

        out = integrate_process_pseudospace(
            process=process,
            ctrl=ctrl,
            mb=mb,
            rates_func=rates_func,
            t_eval=t_eval,
        )
        assert out["c"].shape[0] == t_eval.shape[0]
        assert float(out["c"][-1, 0]) > float(out["c"][0, 0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
