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
    build_algebraic_func,
    build_state_splines,
    extract_discrete_events,
    get_control_splines,
    get_rhs_ode,
    integrate_process,
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
            ),
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=_ts([0.0, 5.0, 10.0, 20.0], [10.0, 8.0, 5.0, 1.0]),
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


def _wrap_q_as_rates(rhs_ode: RhsOde, q_func):
    """Build a flat rates_func that wraps a reactor-only q(t) producer.

    The returned ``rates_func`` produces a flat array of shape
    ``(rhs_ode.rate_size,)`` aligned with ``rhs_ode.rate_names``: the first
    ``n_reactor_states`` entries are ``q_func(t)``, remaining (PV) entries
    default to zero.
    """

    n_reactor = rhs_ode.n_reactor_states
    n_pv_rates = rhs_ode.rate_size - n_reactor

    def rates_func(t, state, controls):
        del state, controls
        q = jnp.asarray(q_func(t), dtype=float)
        if n_pv_rates == 0:
            return q
        return jnp.concatenate([q, jnp.zeros(n_pv_rates, dtype=float)])

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

    def test_rate_size(self):
        # Auto-generated BiologicalOde: q_<rmc> for each reactor component
        # plus r_<pv> for each dynamic PV. _make_process has 2 RMCs and 0
        # dynamic PVs by default.
        assert get_rhs_ode(_make_process()).rate_size == 2

    def test_u_flow_size_fedbatch(self):
        assert get_rhs_ode(_make_process()).u_flow_size == 1

    def test_u_flow_size_batch(self):
        assert get_rhs_ode(_make_batch_process()).u_flow_size == 0

    def test_output_size(self):
        assert get_rhs_ode(_make_process()).output_size == 3

    def test_reactor_component_state_names(self):
        rhs_ode = get_rhs_ode(_make_process())
        assert rhs_ode.reactor_component_state_names == ("biomass", "glucose")

    def test_process_variable_state_names(self):
        rhs_ode = get_rhs_ode(_make_process(with_uncontrolled_pv=True))
        assert rhs_ode.process_variable_state_names == ("dissolved_O2",)

    def test_flow_names(self):
        assert get_rhs_ode(_make_process()).flow_names == ("feed",)

    def test_flow_names_empty_batch(self):
        assert get_rhs_ode(_make_batch_process()).flow_names == ()

    def test_cin_shape(self):
        assert get_rhs_ode(_make_process()).Cin.shape == (1, 2)

    def test_cin_values_biomass_and_glucose(self):
        rhs_ode = get_rhs_ode(_make_process())
        assert float(rhs_ode.Cin[0, 0]) == pytest.approx(0.0)
        assert float(rhs_ode.Cin[0, 1]) == pytest.approx(500.0)

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
                )
            },
        )
        # Auto-generation runs in BioProcess.__post_init__, so the missing
        # biomass component is reported there.
        with pytest.raises(ValueError, match="biomass"):
            BioProcess(
                metadata=BioProcessMetadata(name="p", process_type="batch"),
                time_axis=TimeAxis(
                    unit="hours", start=0.0, end=10.0, time_reference="inoculation"
                ),
                volume=Volume(initial_volume=1.0, unit="L"),
                reactor_medium=rm,
            )

    def test_call_shape_fedbatch(self):
        rhs_ode = get_rhs_ode(_make_process())
        dc = rhs_ode(
            jnp.array([0.5, 10.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.array([0.05]),
            jnp.zeros(0),
            jnp.zeros(rhs_ode.n_controlled_pv),
        )
        assert dc.shape == (3,)

    def test_call_shape_batch(self):
        rhs_ode = get_rhs_ode(_make_batch_process())
        dc = rhs_ode(
            jnp.array([1.0, 5.0, 1.0]),
            jnp.array([0.2, -0.1]),
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.zeros(rhs_ode.n_controlled_pv),
        )
        assert dc.shape == (3,)

    def test_dV_equals_sum_u_flow(self):
        rhs_ode = get_rhs_ode(_make_process())
        dc = rhs_ode(
            jnp.array([0.5, 10.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.array([0.05]),
            jnp.zeros(0),
            jnp.zeros(rhs_ode.n_controlled_pv),
        )
        assert float(dc[-1]) == pytest.approx(0.05, rel=1e-5)

    def test_dV_zero_in_batch(self):
        rhs_ode = get_rhs_ode(_make_batch_process())
        dc = rhs_ode(
            jnp.array([1.0, 5.0, 1.0]),
            jnp.array([0.2, -0.1]),
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.zeros(rhs_ode.n_controlled_pv),
        )
        assert float(dc[-1]) == pytest.approx(0.0)

    def test_reaction_only_batch(self):
        rhs_ode = get_rhs_ode(_make_batch_process())
        X = 2.0
        dc = rhs_ode(
            jnp.array([X, 5.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.zeros(rhs_ode.n_controlled_pv),
        )
        assert float(dc[0]) == pytest.approx(0.3 * X, rel=1e-5)
        assert float(dc[1]) == pytest.approx(-0.15 * X, rel=1e-5)

    def test_dilution_term_zero_q(self):
        rhs_ode = get_rhs_ode(_make_process())
        X, S, V, F = 1.0, 10.0, 1.0, 0.1
        dc = rhs_ode(
            jnp.array([X, S, V]),
            jnp.zeros(2),
            jnp.array([F]),
            jnp.zeros(0),
            jnp.zeros(rhs_ode.n_controlled_pv),
        )
        assert float(dc[0]) == pytest.approx((F / V) * (0.0 - X), rel=1e-4)
        assert float(dc[1]) == pytest.approx((F / V) * (500.0 - S), rel=1e-4)
        assert float(dc[2]) == pytest.approx(F, rel=1e-5)

    def test_full_balance_combined(self):
        rhs_ode = get_rhs_ode(_make_process())
        X, S, V, F, qX, qS = 2.0, 5.0, 1.5, 0.08, 0.4, -0.2
        dc = rhs_ode(
            jnp.array([X, S, V]),
            jnp.array([qX, qS]),
            jnp.array([F]),
            jnp.zeros(0),
            jnp.zeros(rhs_ode.n_controlled_pv),
        )
        assert float(dc[0]) == pytest.approx(qX * X + (F / V) * (0.0 - X), rel=1e-5)
        assert float(dc[1]) == pytest.approx(qS * X + (F / V) * (500.0 - S), rel=1e-5)
        assert float(dc[2]) == pytest.approx(F, rel=1e-5)

    def test_callable_under_filter_jit(self):
        rhs_ode = get_rhs_ode(_make_process())
        dc = eqx.filter_jit(rhs_ode)(
            jnp.array([0.5, 10.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.array([0.05]),
            jnp.zeros(0),
            jnp.zeros(rhs_ode.n_controlled_pv),
        )
        assert dc.shape == (3,)

    def test_filter_jit_matches_eager(self):
        rhs_ode = get_rhs_ode(_make_process())
        args = (
            jnp.array([0.5, 10.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.array([0.05]),
            jnp.zeros(0),
            jnp.zeros(rhs_ode.n_controlled_pv),
        )
        assert jnp.allclose(rhs_ode(*args), eqx.filter_jit(rhs_ode)(*args), atol=1e-6)

    def test_grad_wrt_c(self):
        rhs_ode = get_rhs_ode(_make_process())
        q = jnp.array([0.3, -0.15])
        u_flow = jnp.array([0.05])
        f_mod = jnp.zeros(0)
        r = jnp.zeros(rhs_ode.n_controlled_pv)
        g = eqx.filter_jit(jax.grad(lambda c: jnp.sum(rhs_ode(c, q, u_flow, f_mod, r))))(
            jnp.array([0.5, 10.0, 1.0])
        )
        assert g.shape == (3,)

    def test_grad_wrt_q(self):
        rhs_ode = get_rhs_ode(_make_process())
        c = jnp.array([0.5, 10.0, 1.0])
        u_flow = jnp.array([0.05])
        f_mod = jnp.zeros(0)
        r = jnp.zeros(rhs_ode.n_controlled_pv)
        g = eqx.filter_jit(jax.grad(lambda q: jnp.sum(rhs_ode(c, q, u_flow, f_mod, r))))(
            jnp.array([0.3, -0.15])
        )
        assert g.shape == (2,)

    def test_vmap_over_batch_of_states(self):
        rhs_ode = get_rhs_ode(_make_process())
        fn = eqx.filter_jit(eqx.filter_vmap(rhs_ode, in_axes=(0, 0, 0, None, None)))
        B = 4
        c = jnp.stack(
            [jnp.array([0.5 + i * 0.3, 10.0 - i, 1.0 + i * 0.1]) for i in range(B)]
        )
        q = jnp.tile(jnp.array([0.3, -0.15]), (B, 1))
        u = jnp.tile(jnp.array([0.05]), (B, 1))
        dc = fn(c, q, u, jnp.zeros(0), jnp.zeros(rhs_ode.n_controlled_pv))
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
        rhs_ode = get_rhs_ode(process)

        state = jnp.array([1.0, 10.0, 80.0, 1.0])
        q = jnp.array([0.2, -0.1])
        r = jnp.array([0.0, 0.0, 5.0])
        dc = rhs_ode(state, q, jnp.zeros(0), jnp.zeros(0), r)

        assert float(dc[2]) == pytest.approx(0.0)

    def test_no_uncontrolled_pv_regression(self):
        rhs_ode = get_rhs_ode(_make_process(with_uncontrolled_pv=False))
        assert rhs_ode.n_pv_states == 0
        assert rhs_ode.c_size == 3
        dc = rhs_ode(
            jnp.array([1.0, 10.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.array([0.05]),
            jnp.zeros(0),
            jnp.zeros(rhs_ode.n_controlled_pv),
        )
        assert dc.shape == (3,)

    def test_event_and_feed_effects_are_reactor_only(self):
        process = _make_process(
            with_controlled_flow=True,
            with_controlled_pv=False,
            with_uncontrolled_pv=True,
        )
        rhs_ode = get_rhs_ode(process)
        X, S, DO, V = 1.0, 10.0, 80.0, 1.0
        F = 0.1
        dc = rhs_ode(
            jnp.array([X, S, DO, V]),
            jnp.zeros(rhs_ode.n_reactor_states),
            jnp.array([F]),
            jnp.zeros(0),
            jnp.zeros(rhs_ode.n_controlled_pv),
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
                ),
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=_ts([0.0, 10.0], [0.5, 4.0]),
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
        rhs_ode = get_rhs_ode(self._make_process_glucose_first())
        assert rhs_ode.reactor_component_state_names[0] == "biomass"

    def test_glucose_is_second(self):
        rhs_ode = get_rhs_ode(self._make_process_glucose_first())
        assert rhs_ode.reactor_component_state_names[1] == "glucose"

    def test_reaction_uses_biomass_at_index_0(self):
        """dc[biomass]/dt = qX * X when biomass is reordered to index 0."""
        rhs_ode = get_rhs_ode(self._make_process_glucose_first())
        X, S, V = 2.0, 5.0, 1.0
        # state is [biomass, glucose, V] after reordering
        dc = rhs_ode(
            jnp.array([X, S, V]),
            jnp.array([0.3, -0.15]),
            jnp.zeros(0),
            jnp.zeros(0),
            jnp.zeros(rhs_ode.n_controlled_pv),
        )
        assert float(dc[0]) == pytest.approx(0.3 * X, rel=1e-5)
        assert float(dc[1]) == pytest.approx(-0.15 * X, rel=1e-5)


# ---------------------------------------------------------------------------
# Helpers used by TestUserDefinedBiologicalOde — formerly intracellular fixtures
# ---------------------------------------------------------------------------


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
            ),
            "product": ReactorMediumComponent(
                name="product",
                unit="g/L",
                concentration=_ts([0.0, 10.0], [0.0, 1.0]),
            ),
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=_ts([0.0, 10.0], [10.0, 1.0]),
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
        algebraic={"X_active": "biomass - product"},
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


class TestUserDefinedBiologicalOde:
    def test_get_rhs_ode_returns_rhs_ode(self):
        p = _make_batch_with_biological_ode_intracellular()
        rhs_ode = get_rhs_ode(p)
        assert isinstance(rhs_ode, RhsOde)

    def test_user_defined_rhs_uses_x_active_in_derivatives(self):
        """User-defined RHS evaluates derivatives over X_active = biomass -
        product, and routes the product accumulation back into the biomass
        derivative so dX_meas/dt = dX_active/dt + dP/dt."""
        p_user = _make_batch_with_biological_ode_intracellular()
        mb_user = get_rhs_ode(p_user)

        c = jnp.array([2.0, 0.5, 5.0, 1.0])
        rates = jnp.array([0.4, 0.1, -0.2])  # q_X_active, q_P, q_S
        dc_user = mb_user(c, rates, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0))

        X_active = 2.0 - 0.5
        np.testing.assert_allclose(
            np.asarray(dc_user),
            np.array([(0.4 + 0.1) * X_active, 0.1 * X_active, -0.2 * X_active, 0.0]),
            atol=1e-6,
        )

    def test_algebraic_func_returns_x_active(self):
        p = _make_batch_with_biological_ode_intracellular()
        df = build_algebraic_func(p)
        state_values = jnp.array([2.0, 0.5, 5.0])
        out = df(state_values, jnp.zeros(0), jnp.array([0.0, 0.0, 0.0]))
        assert "X_active" in out
        assert float(out["X_active"]) == pytest.approx(1.5, rel=1e-6)

    def test_user_defined_jit_compatible(self):
        p = _make_batch_with_biological_ode_intracellular()
        rhs_ode = get_rhs_ode(p)
        dc = eqx.filter_jit(rhs_ode)(
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
        rhs_ode = get_rhs_ode(p)
        c = jnp.array([2.0, 0.5, 5.0, 1.0])
        rates = jnp.array([0.4, 0.1, -0.2])
        dc = rhs_ode(c, rates, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0))
        # No flow → glucose entry is purely the biological term, which is 0.
        assert float(dc[2]) == pytest.approx(0.0, abs=1e-7)

    def test_rate_size_follows_user_declaration(self):
        """`len(rates)` is whatever the user declared, not pinned to
        n_reactor_states. Three rates for three states matches today; an
        extra unused rate just adds to rate_size without changing dc/dt.
        """
        p = _make_batch_with_biological_ode_intracellular()
        p.biological_ode.rates["q_unused"] = RateDecl()
        rhs_ode = get_rhs_ode(p)
        assert rhs_ode.rate_size == 4
        # Adding the unused rate must not change dc/dt as long as no
        # expression references it.
        c = jnp.array([2.0, 0.5, 5.0, 1.0])
        rates_4 = jnp.array([0.4, 0.1, -0.2, 99.0])
        dc = rhs_ode(c, rates_4, jnp.zeros(0), jnp.zeros(0), jnp.zeros(0))
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
            ),
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=_ts([0.0, 5.0, 10.0, 20.0], [10.0, 8.0, 5.0, 1.0]),
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
        rhs_ode = get_rhs_ode(_make_process_with_modeled_flow())
        assert rhs_ode.modeled_flow_names == ("base_feed",)

    def test_f_modeled_size(self):
        rhs_ode = get_rhs_ode(_make_process_with_modeled_flow())
        assert rhs_ode.f_modeled_size == 1

    def test_cin_modeled_shape(self):
        rhs_ode = get_rhs_ode(_make_process_with_modeled_flow())
        assert rhs_ode.Cin_modeled.shape == (1, 2)

    def test_cin_modeled_values(self):
        rhs_ode = get_rhs_ode(_make_process_with_modeled_flow())
        assert float(rhs_ode.Cin_modeled[0, 0]) == pytest.approx(0.0)
        assert float(rhs_ode.Cin_modeled[0, 1]) == pytest.approx(0.0)

    def test_no_modeled_flow_for_simple_process(self):
        rhs_ode = get_rhs_ode(_make_process())
        assert rhs_ode.modeled_flow_names == ()
        assert rhs_ode.f_modeled_size == 0
        assert rhs_ode.Cin_modeled.shape == (0, 2)

    def test_no_modeled_flow_for_batch(self):
        rhs_ode = get_rhs_ode(_make_batch_process())
        assert rhs_ode.modeled_flow_names == ()
        assert rhs_ode.f_modeled_size == 0

    def test_uncontrolled_flow_is_modeled_not_controlled(self):
        """Uncontrolled continuous flow appears in modeled_flow_names."""
        rhs_ode = get_rhs_ode(_make_process_with_modeled_flow())
        assert "base_feed" not in rhs_ode.flow_names
        assert "base_feed" in rhs_ode.modeled_flow_names
        assert "carbon_feed" in rhs_ode.flow_names
        assert "carbon_feed" not in rhs_ode.modeled_flow_names

    def test_call_shape_with_modeled_flow(self):
        rhs_ode = get_rhs_ode(_make_process_with_modeled_flow())
        dc = rhs_ode(
            jnp.array([0.5, 10.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.array([0.05]),
            jnp.array([0.02]),
            jnp.zeros(rhs_ode.n_controlled_pv),
        )
        assert dc.shape == (3,)

    def test_dV_includes_modeled_flow(self):
        rhs_ode = get_rhs_ode(_make_process_with_modeled_flow())
        F_ctrl, F_mod = 0.05, 0.02
        dc = rhs_ode(
            jnp.array([0.5, 10.0, 1.0]),
            jnp.array([0.0, 0.0]),
            jnp.array([F_ctrl]),
            jnp.array([F_mod]),
            jnp.zeros(rhs_ode.n_controlled_pv),
        )
        assert float(dc[-1]) == pytest.approx(F_ctrl + F_mod, rel=1e-5)

    def test_dilution_with_modeled_flow(self):
        """Modeled flow dilutes species in the reactor."""
        rhs_ode = get_rhs_ode(_make_process_with_modeled_flow())
        X, S, V = 1.0, 10.0, 1.0
        F_ctrl, F_mod = 0.1, 0.05
        # Both feeds: carbon has Cin_glucose=500, base has Cin_glucose=0
        dc = rhs_ode(
            jnp.array([X, S, V]),
            jnp.zeros(2),
            jnp.array([F_ctrl]),
            jnp.array([F_mod]),
            jnp.zeros(rhs_ode.n_controlled_pv),
        )
        # Expected: (F_ctrl/V)*(500-S) + (F_mod/V)*(0-S) for glucose
        expected_glucose = (F_ctrl / V) * (500.0 - S) + (F_mod / V) * (0.0 - S)
        assert float(dc[1]) == pytest.approx(expected_glucose, rel=1e-4)

    def test_full_balance_with_modeled_flow(self):
        rhs_ode = get_rhs_ode(_make_process_with_modeled_flow())
        X, S, V = 2.0, 5.0, 1.5
        F_ctrl, F_mod = 0.08, 0.03
        qX, qS = 0.4, -0.2
        dc = rhs_ode(
            jnp.array([X, S, V]),
            jnp.array([qX, qS]),
            jnp.array([F_ctrl]),
            jnp.array([F_mod]),
            jnp.zeros(rhs_ode.n_controlled_pv),
        )
        expected_X = qX * X + (F_ctrl / V) * (0.0 - X) + (F_mod / V) * (0.0 - X)
        expected_S = qS * X + (F_ctrl / V) * (500.0 - S) + (F_mod / V) * (0.0 - S)
        expected_dV = F_ctrl + F_mod
        assert float(dc[0]) == pytest.approx(expected_X, rel=1e-5)
        assert float(dc[1]) == pytest.approx(expected_S, rel=1e-5)
        assert float(dc[2]) == pytest.approx(expected_dV, rel=1e-5)

    def test_jit_with_modeled_flow(self):
        rhs_ode = get_rhs_ode(_make_process_with_modeled_flow())
        dc = eqx.filter_jit(rhs_ode)(
            jnp.array([0.5, 10.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.array([0.05]),
            jnp.array([0.02]),
            jnp.zeros(rhs_ode.n_controlled_pv),
        )
        assert dc.shape == (3,)

    def test_jit_matches_eager_with_modeled_flow(self):
        rhs_ode = get_rhs_ode(_make_process_with_modeled_flow())
        args = (
            jnp.array([0.5, 10.0, 1.0]),
            jnp.array([0.3, -0.15]),
            jnp.array([0.05]),
            jnp.array([0.02]),
            jnp.zeros(rhs_ode.n_controlled_pv),
        )
        assert jnp.allclose(rhs_ode(*args), eqx.filter_jit(rhs_ode)(*args), atol=1e-6)

    def test_grad_wrt_f_modeled(self):
        rhs_ode = get_rhs_ode(_make_process_with_modeled_flow())
        c = jnp.array([0.5, 10.0, 1.0])
        q = jnp.array([0.3, -0.15])
        u_flow = jnp.array([0.05])
        g = eqx.filter_jit(
            jax.grad(lambda fm: jnp.sum(rhs_ode(c, q, u_flow, fm, jnp.zeros(rhs_ode.n_controlled_pv))))
        )(jnp.array([0.02]))
        assert g.shape == (1,)

    def test_vmap_with_modeled_flow(self):
        rhs_ode = get_rhs_ode(_make_process_with_modeled_flow())
        fn = eqx.filter_jit(eqx.filter_vmap(rhs_ode, in_axes=(0, 0, 0, 0, None)))
        B = 4
        c = jnp.stack(
            [jnp.array([0.5 + i * 0.3, 10.0 - i, 1.0 + i * 0.1]) for i in range(B)]
        )
        q = jnp.tile(jnp.array([0.3, -0.15]), (B, 1))
        u = jnp.tile(jnp.array([0.05]), (B, 1))
        f_mod = jnp.tile(jnp.array([0.02]), (B, 1))
        dc = fn(c, q, u, f_mod, jnp.zeros(rhs_ode.n_controlled_pv))
        assert dc.shape == (B, 3)


# ---------------------------------------------------------------------------
# Integration: ControlSplines + RhsOde wired together
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_wired_ode_step(self):
        process = _make_process()
        ctrl = get_control_splines(process)
        rhs_ode = get_rhs_ode(process)

        @eqx.filter_jit
        def ode_rhs(t, c, q):
            u = ctrl(t)
            u_flow = u[jnp.array(list(ctrl.flow_indices))]
            return rhs_ode(c, q, u_flow, jnp.zeros(0), jnp.zeros(rhs_ode.n_controlled_pv))

        dc = ode_rhs(
            jnp.array(5.0), jnp.array([0.5, 10.0, 1.0]), jnp.array([0.3, -0.15])
        )
        assert dc.shape == (3,)

    def test_flow_index_aligns_with_flow_names(self):
        process = _make_process()
        ctrl = get_control_splines(process)
        rhs_ode = get_rhs_ode(process)
        assert len(ctrl.flow_indices) == rhs_ode.u_flow_size
        assert ctrl.control_names[ctrl.flow_indices[0]] == rhs_ode.flow_names[0]

    def test_dV_from_wired_ode_equals_flow_rate(self):
        process = _make_process()
        ctrl = get_control_splines(process)
        rhs_ode = get_rhs_ode(process)
        t = jnp.array(5.0)
        u = ctrl(t)
        u_flow = u[jnp.array(list(ctrl.flow_indices))]
        dc = rhs_ode(
            jnp.array([1.0, 8.0, 1.2]),
            jnp.zeros(2),
            u_flow,
            jnp.zeros(0),
            jnp.zeros(rhs_ode.n_controlled_pv),
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
        assert hasattr(mech, "build_state_splines")
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
        rhs_ode = get_rhs_ode(process)
        events = extract_discrete_events(process, rhs_ode)
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
        rhs_ode = get_rhs_ode(process)
        events = extract_discrete_events(process, rhs_ode)
        assert len(events) == 1
        assert events[0]["kind"] == "bolus_feed"
        assert events[0]["dV"] == pytest.approx(0.1)
        # Cin aligned with reactor_component_state_names
        bio_idx = rhs_ode.reactor_component_state_names.index("biomass")
        glu_idx = rhs_ode.reactor_component_state_names.index("glucose")
        assert events[0]["Cin"][bio_idx] == pytest.approx(0.0)
        assert events[0]["Cin"][glu_idx] == pytest.approx(300.0)

    def test_no_discrete_events(self):
        process = _make_process(
            with_controlled_flow=True, with_controlled_pv=False, with_discrete_vc=False
        )
        rhs_ode = get_rhs_ode(process)
        events = extract_discrete_events(process, rhs_ode)
        assert events == []

    def test_events_sorted_by_time(self):
        process = _make_process(
            with_controlled_flow=False, with_controlled_pv=False, with_discrete_vc=True
        )
        rhs_ode = get_rhs_ode(process)
        events = extract_discrete_events(process, rhs_ode)
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
        rhs_ode = get_rhs_ode(process)
        events = extract_discrete_events(process, rhs_ode)
        assert len(events[0]["Cin"]) == len(rhs_ode.reactor_component_state_names)

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
        rhs_ode = get_rhs_ode(process)

        events = extract_discrete_events(process, rhs_ode)
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
        rhs_ode = get_rhs_ode(process)

        with pytest.raises(ValueError, match="At most one discrete event per kind"):
            extract_discrete_events(process, rhs_ode)

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
        rhs_ode = get_rhs_ode(process)

        with pytest.raises(ValueError, match="At most one discrete event per kind"):
            extract_discrete_events(process, rhs_ode)


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
        rhs_ode = get_rhs_ode(process)

        q_arr = jnp.array([q_X, q_S])

        def q_func(t):
            return q_arr

        rates_func = _wrap_q_as_rates(rhs_ode, q_func)
        return process, ctrl, rhs_ode, rates_func, q_X, q_S

    def test_batch_accuracy(self):
        """Forward integration with known q recovers analytical solution."""
        process, ctrl, rhs_ode, rates_func, q_X, q_S = self._setup_batch_integration()
        t_eval = np.linspace(0, 10, 50)

        result = integrate_process(process, ctrl, rhs_ode, rates_func, t_eval)

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
        rhs_ode = get_rhs_ode(process)
        t_eval = np.linspace(0, 4, 40)

        def rates_func(_t, state, controls):
            del controls
            q = jnp.array([0.2 * state[0], 0.0])
            return q  # flat rates: this batch process has no dynamic PVs

        result = integrate_process(process, ctrl, rhs_ode, rates_func, t_eval)
        assert float(result["c"][-1, 0]) > float(result["c"][0, 0])

    def test_rates_func_can_use_controls_argument(self):
        process = _make_process(with_controlled_flow=True, with_controlled_pv=False)
        ctrl = get_control_splines(process)
        rhs_ode = get_rhs_ode(process)
        t_eval = np.linspace(0, 10, 60)

        def rates_from_controls(_t, _state, controls):
            q = jnp.array([jnp.maximum(controls[0], 0.0), 0.0])
            return q  # flat rates: this fixture has no dynamic PVs

        def zero_rates(_t, _state, _controls):
            return jnp.zeros(rhs_ode.rate_size)

        out_controls = integrate_process(process, ctrl, rhs_ode, rates_from_controls, t_eval)
        out_zero = integrate_process(process, ctrl, rhs_ode, zero_rates, t_eval)

        assert float(out_controls["c"][-1, 0]) > float(out_zero["c"][-1, 0])

    def test_integrate_process_requires_rates_func(self):
        process, ctrl, rhs_ode, _, _, _ = self._setup_batch_integration()
        t_eval = np.linspace(0, 2, 5)

        with pytest.raises(ValueError, match="rates_func is required"):
            integrate_process(process, ctrl, rhs_ode, None, t_eval)

    def test_volume_constant_in_batch(self):
        """Volume should stay constant in batch mode."""
        process, ctrl, rhs_ode, rates_func, _, _ = self._setup_batch_integration()
        t_eval = np.linspace(0, 10, 20)
        result = integrate_process(process, ctrl, rhs_ode, rates_func, t_eval)
        np.testing.assert_allclose(result["V"], 1.0, atol=1e-6)

    def test_with_sampling_events(self):
        """Volume drops at sampling events, concentrations stay continuous."""
        process = _make_process(
            with_controlled_flow=True, with_controlled_pv=False, with_discrete_vc=True
        )
        ctrl = get_control_splines(process)
        rhs_ode = get_rhs_ode(process)

        zero_rates = _wrap_q_as_rates(rhs_ode, lambda t: jnp.zeros(rhs_ode.n_reactor_states))
        t_eval = np.linspace(0, 20, 100)

        result = integrate_process(process, ctrl, rhs_ode, zero_rates, t_eval)
        assert "t" in result
        assert "c" in result
        assert "V" in result

        # Volume should be lower after sampling
        V_early = result["V"][result["t"] < 4.0]
        V_late = result["V"][result["t"] > 11.0]
        assert np.mean(V_late) > np.mean(V_early) - 0.2  # feed increases volume

    def test_output_format(self):
        """Check returned dict has expected keys and shapes."""
        process, ctrl, rhs_ode, rates_func, _, _ = self._setup_batch_integration()
        t_eval = np.linspace(0, 10, 30)
        result = integrate_process(process, ctrl, rhs_ode, rates_func, t_eval)
        assert {"t", "c", "V", "stats"} == set(result.keys())
        assert result["c"].shape[1] == rhs_ode.n_reactor_states
        assert result["V"].shape[0] == result["c"].shape[0]
        assert result["t"].shape[0] == result["c"].shape[0]

    def test_default_settings_accuracy(self):
        """Default rtol/atol settings produce RMSE < 1e-3 for a simple batch.

        Default tolerances are rtol=1e-4, atol=1e-6 (float32-friendly).
        """
        process, ctrl, rhs_ode, rates_func, q_X, q_S = self._setup_batch_integration()
        t_eval = np.linspace(0, 10, 100)

        result = integrate_process(process, ctrl, rhs_ode, rates_func, t_eval)

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
        rhs_ode = get_rhs_ode(process)

        zero_rates = _wrap_q_as_rates(rhs_ode, lambda t: jnp.zeros(rhs_ode.n_reactor_states))
        t_eval = np.linspace(0, 20, 50)
        result = integrate_process(process, ctrl, rhs_ode, zero_rates, t_eval)

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
        rhs_ode = get_rhs_ode(process)
        zero_rates = _wrap_q_as_rates(rhs_ode, lambda t: jnp.zeros(rhs_ode.n_reactor_states))
        t_eval = jnp.array([0.0, 4.9, 5.0, 5.1])
        result = integrate_process(process, ctrl, rhs_ode, zero_rates, t_eval)

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
        rhs_ode = get_rhs_ode(process)
        zero_rates = _wrap_q_as_rates(rhs_ode, lambda t: jnp.zeros(rhs_ode.n_reactor_states))
        t_eval = jnp.array([0.0, 4.9, 5.0, 5.1])
        result = integrate_process(process, ctrl, rhs_ode, zero_rates, t_eval)

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
        rhs_ode = get_rhs_ode(process)
        assert rhs_ode.process_variable_state_names == ("dissolved_O2",)

        zero_rates = _wrap_q_as_rates(rhs_ode, lambda t: jnp.zeros(rhs_ode.n_reactor_states))
        t_eval = jnp.array([4.9, 5.0, 5.1])
        result = integrate_process(process, ctrl, rhs_ode, zero_rates, t_eval)

        pv_idx = rhs_ode.pv_indices[0]
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
        rhs_ode = get_rhs_ode(process)
        zero_rates = _wrap_q_as_rates(rhs_ode, lambda t: jnp.zeros(rhs_ode.n_reactor_states))
        t_eval = jnp.array([0.0, 4.9, 5.0, 5.1])
        result = integrate_process(process, ctrl, rhs_ode, zero_rates, t_eval)

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
        rhs_ode = get_rhs_ode(process)
        zero_rates = _wrap_q_as_rates(rhs_ode, lambda t: jnp.zeros(rhs_ode.n_reactor_states))
        t_eval = jnp.array([19.9, 20.0])
        result = integrate_process(process, ctrl, rhs_ode, zero_rates, t_eval)

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
        rhs_ode = get_rhs_ode(process)
        zero_rates = _wrap_q_as_rates(rhs_ode, lambda t: jnp.zeros(rhs_ode.n_reactor_states))
        t_eval = jnp.array([0.0, 0.1, 1.0])
        result = integrate_process(process, ctrl, rhs_ode, zero_rates, t_eval)

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
        rhs_ode = get_rhs_ode(process)

        def q_func(t):
            del t
            return jnp.zeros(rhs_ode.n_reactor_states)

        rates_func = _wrap_q_as_rates(rhs_ode, q_func)
        with pytest.raises(Exception, match="reactor volume"):
            integrate_process(
                process,
                ctrl,
                rhs_ode,
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