"""Tests for hybrax.format.splines.fit_monotone_timeseries_spline."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import BSpline

from hybrax.format import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    ProcessVariable,
    ReactorMedium,
    TimeAxis,
    TimeSeries,
    Volume,
    monotone_spline,
)
from hybrax.format.serialization import load_process_collection, save_process_collection
from hybrax.format.splines import fit_monotone_timeseries_spline, fit_timeseries_spline

T = np.linspace(0.0, 12.0, 241)
RAMP = np.clip((T - 3.0) / 0.2, 0.0, 1.0)
RNG = np.random.default_rng(0)
T_IRREGULAR = np.sort(np.r_[0.0, 12.0, RNG.uniform(0.0, 12.0, 200)])


def _ts(t, v, **kwargs):
    return TimeSeries(
        times=jnp.asarray(t, dtype=float), values=jnp.asarray(v, dtype=float), **kwargs
    )


def _assert_contract(fitted, times, values, rms_tol):
    """Check the guarantees on the stored representation independently."""
    times, values = np.asarray(times), np.asarray(values)
    net = values[-1] - values[0]
    breaks = np.asarray(fitted.breaks)
    coeffs = np.asarray(fitted.coeffs)
    assert breaks.dtype == np.float64 and coeffs.dtype == np.float64
    np.testing.assert_array_equal(np.asarray(fitted.times), times)
    np.testing.assert_array_equal(np.asarray(fitted.values), values)
    assert breaks[0] == times[0] and breaks[-1] == times[-1]

    # RMS budget, through public evaluation.
    evaluated = np.asarray(fitted.evaluate_many(fitted.times))
    assert np.sqrt(np.mean((evaluated - values) ** 2)) <= rms_tol
    scale = max(np.max(np.abs(values)), 1.0)
    assert abs(evaluated[0] - values[0]) <= 1e-12 * scale
    assert abs(evaluated[-1] - values[-1]) <= 1e-12 * scale

    # Derivative: exact sign at piece ends and interior extrema of each quadratic.
    rate = np.asarray(fitted.deriv().coeffs)[:, :3]
    widths = np.diff(breaks)
    rates = [rate[:, 0], rate[:, 0] + rate[:, 1] * widths + rate[:, 2] * widths**2]
    with np.errstate(divide="ignore", invalid="ignore"):
        vertex = -rate[:, 1] / (2 * rate[:, 2])
    inside = (rate[:, 2] != 0) & (vertex > 0) & (vertex < widths)
    v = vertex[inside]
    r = rate[inside]
    rates.append(r[:, 0] + r[:, 1] * v + r[:, 2] * v**2)
    signed = np.sign(net) * np.concatenate(rates) if net else np.concatenate(rates)
    if net:
        assert np.all(signed >= 0.0)
    else:
        assert np.all(signed == 0.0)

    # The derivative integrates to the net change.
    integral = np.sum(
        rate[:, 0] * widths + rate[:, 1] * widths**2 / 2 + rate[:, 2] * widths**3 / 3
    )
    # Widths come from absolute breaks, so a large time offset limits precision.
    width_precision = 4 * np.spacing(np.max(np.abs(breaks))) / np.min(widths)
    assert integral == pytest.approx(
        net, rel=1e-12 + width_precision, abs=1e-15 * scale
    )

    # C0, C1 and C2 continuity at interior breaks.
    h = widths[:-1]
    left_value = (
        coeffs[:-1, 0]
        + coeffs[:-1, 1] * h
        + coeffs[:-1, 2] * h**2
        + coeffs[:-1, 3] * h**3
    )
    value_scale = max(np.max(np.abs(values)), rms_tol)
    np.testing.assert_allclose(
        left_value,
        coeffs[1:, 0],
        rtol=0.0,
        atol=1e-12 * value_scale + 1e-9 * abs(net),
    )
    left_rate = coeffs[:-1, 1] + 2 * coeffs[:-1, 2] * h + 3 * coeffs[:-1, 3] * h**2
    left_curvature = 2 * coeffs[:-1, 2] + 6 * coeffs[:-1, 3] * h
    rate_scale = max(abs(net), rms_tol) / (times[-1] - times[0])
    np.testing.assert_allclose(
        left_rate, coeffs[1:, 1], rtol=width_precision, atol=1e-9 * rate_scale
    )
    curvature_scale = np.max(np.abs(left_curvature), initial=0.0) + rate_scale
    np.testing.assert_allclose(
        left_curvature,
        2 * coeffs[1:, 2],
        rtol=width_precision,
        atol=(1e-6 + 10 * width_precision) * curvature_scale,
    )


CASES = {  # times, values, rms_tol
    "linear_increasing": (T, 2.0 * T + 1.0, 1e-3),
    "smooth_decreasing": (T, -3.0 * np.tanh((T - 6.0) / 2.0) + 7.0, 6e-4),
    "narrow_ramp": (T, 5.0 * RAMP, 5e-3),
    "decreasing_staircase": (T, -np.floor(T / 3.0), 0.2),
    "step_in_first_interval": (T, np.r_[0.0, np.full(len(T) - 1, 3.7e-5)], 3.7e-8),
    "step_in_last_interval": (T, np.r_[np.zeros(len(T) - 1), 2.0], 2e-3),
    "irregular_noisy": (
        T_IRREGULAR,
        T_IRREGULAR + 0.01 * RNG.standard_normal(len(T_IRREGULAR)),
        0.012,
    ),
    "large_value_offset": (T, 1e8 + RAMP, 1e-3),
    "moderate_time_offset": (100.0 + T, RAMP, 1e-3),
    "late_short_stretch": (
        800.0 + np.linspace(0.0, 1.0, 61),
        np.linspace(0.0, 1.0, 61) ** 2,
        1e-3,
    ),
    "tiny_negative_change": (T, -5.0 - 1e-9 * RAMP, 1e-12),
    "two_samples": (np.array([0.0, 1.0]), np.array([1.0, 3.0]), 1e-6),
    "three_samples": (np.array([0.0, 1.0, 5.0]), np.array([0.0, 0.1, 2.0]), 1e-6),
}


@pytest.mark.parametrize("name", CASES)
def test_fit_meets_contract(name):
    times, values, rms_tol = CASES[name]
    fitted = fit_monotone_timeseries_spline(
        _ts(times, values), rms_tol=rms_tol, increasing=bool(values[-1] > values[0])
    )
    _assert_contract(fitted, times, values, rms_tol)
    info = fitted.metadata["monotone_fit"]
    assert info["rms_tol"] == rms_tol
    assert info["rms_error"] <= rms_tol
    expected = "increasing" if values[-1] > values[0] else "decreasing"
    assert info["direction"] == expected


def test_smooth_data_needs_few_knots_and_tight_budget_more():
    values = np.tanh((T - 6.0) / 2.0)
    loose = fit_monotone_timeseries_spline(
        _ts(T, values), rms_tol=1e-2, increasing=True
    )
    tight = fit_monotone_timeseries_spline(
        _ts(T, values), rms_tol=1e-5, increasing=True
    )
    assert loose.metadata["monotone_fit"]["interior_knots"] <= 8
    assert (
        tight.metadata["monotone_fit"]["interior_knots"]
        > loose.metadata["monotone_fit"]["interior_knots"]
    )
    _assert_contract(tight, T, values, 1e-5)


def _roughness(fitted):
    """Integrated squared second derivative of the stored spline."""
    coeffs = np.asarray(fitted.coeffs)
    widths = np.diff(np.asarray(fitted.breaks))
    p, q = 2 * coeffs[:, 2], 6 * coeffs[:, 3]
    return np.sum(p**2 * widths + p * q * widths**2 + q**2 * widths**3 / 3)


def test_smoothing_uses_the_budget_to_reduce_curvature(monkeypatch):
    values = np.tanh((T - 6.0) / 2.0) + 0.002 * RNG.standard_normal(len(T))
    values[[0, -1]] = np.tanh(np.array([-3.0, 3.0]))
    smoothed = fit_monotone_timeseries_spline(
        _ts(T, values), rms_tol=5e-3, increasing=True
    )
    info = smoothed.metadata["monotone_fit"]
    assert info["rms_error"] >= 0.99 * 5e-3
    monkeypatch.setattr(
        monotone_spline,
        "_smoothest",
        lambda design, z, budget, *args: monotone_spline._solve_conic(
            design, z, budget, None, None
        ),
    )
    closest = fit_monotone_timeseries_spline(
        _ts(T, values), rms_tol=5e-3, increasing=True
    )
    assert _roughness(smoothed) < _roughness(closest)


def test_knots_concentrate_at_local_features():
    fitted = fit_monotone_timeseries_spline(
        _ts(T, 5.0 * RAMP), rms_tol=5e-3, increasing=True
    )
    breaks = np.asarray(fitted.breaks)
    near_ramp = np.sum((breaks > 2.5) & (breaks < 3.7))
    assert near_ramp >= (len(breaks) - 2) / 2


def test_unit_rescaling_preserves_accuracy():
    values = 4.0 * RAMP + 0.3 * T
    base = fit_monotone_timeseries_spline(_ts(T, values), rms_tol=1e-3, increasing=True)
    # Time in seconds and values in grams: scale rms_tol with the value unit.
    scaled = fit_monotone_timeseries_spline(
        _ts(3600.0 * T, 1000.0 * values), rms_tol=1.0, increasing=True
    )
    np.testing.assert_allclose(
        np.asarray(scaled.evaluate_many(scaled.times)) / 1000.0,
        np.asarray(base.evaluate_many(base.times)),
        atol=1e-9,
    )
    _assert_contract(scaled, 3600.0 * T, 1000.0 * values, 1.0)


def test_unconstrained_fit_would_stop_too_early():
    # This monotone cubic has unordered B-spline coefficients [0, 1, 0, 1]. An
    # unconstrained fit without interior knots is exact; the monotone fit is not
    # and must refine.
    times = np.linspace(0.0, 1.0, 101)
    values = 4.0 * (times - 0.5) ** 3 + 0.5
    knots = np.r_[[0.0] * 4, [1.0] * 4]
    coef = np.linalg.lstsq(
        BSpline.design_matrix(times, knots, 3).toarray(), values, rcond=None
    )[0]
    np.testing.assert_allclose(coef, [0.0, 1.0, 0.0, 1.0], atol=1e-12)
    fitted = fit_monotone_timeseries_spline(
        _ts(times, values), rms_tol=1e-3, increasing=True
    )
    assert fitted.metadata["monotone_fit"]["interior_knots"] > 0
    _assert_contract(fitted, times, values, 1e-3)


def test_budget_is_pooled_so_isolated_dips_are_smoothed_over():
    values = T.copy()
    values[120] -= 0.05  # a reversal well below the previous sample
    fitted = fit_monotone_timeseries_spline(
        _ts(T, values), rms_tol=0.01, increasing=True
    )
    _assert_contract(fitted, T, values, 0.01)
    assert fitted.metadata["monotone_fit"]["max_error"] > 0.01
    np.testing.assert_array_equal(np.asarray(fitted.values), values)


@pytest.mark.parametrize("increasing", [True, False])
def test_constant_data_gives_constant_spline(increasing):
    fitted = fit_monotone_timeseries_spline(
        _ts(T, np.full_like(T, 2.0)), rms_tol=1e-12, increasing=increasing
    )
    np.testing.assert_array_equal(np.asarray(fitted.coeffs), [[2.0, 0.0, 0.0, 0.0]])
    assert fitted.metadata["monotone_fit"]["direction"] == (
        "increasing" if increasing else "decreasing"
    )


@pytest.mark.parametrize("increasing", [True, False])
def test_equal_endpoints_accept_excursion_within_budget(increasing):
    values = 1e-3 * np.sin(np.pi * T / 12.0) ** 2
    values[-1] = values[0]
    fitted = fit_monotone_timeseries_spline(
        _ts(T, values), rms_tol=7e-4, increasing=increasing
    )
    np.testing.assert_array_equal(np.asarray(fitted.coeffs)[:, 1:], 0.0)
    _assert_contract(fitted, T, values, 7e-4)


def test_equal_endpoints_reject_excursion_beyond_budget():
    values = np.r_[np.sin(np.pi * T[:-1] / 12.0), 0.0]
    with pytest.raises(ValueError, match="Endpoints are equal"):
        fit_monotone_timeseries_spline(_ts(T, values), rms_tol=1e-3, increasing=True)


@pytest.mark.parametrize(
    ("series", "kwargs", "match"),
    [
        (_ts(T, T), {"rms_tol": 0.0}, "rms_tol must be"),
        (_ts(T, T), {"rms_tol": -1.0}, "rms_tol must be"),
        (_ts(T, T), {"rms_tol": float("nan")}, "rms_tol must be"),
        (_ts(T, T), {"rms_tol": float("inf")}, "rms_tol must be"),
        (_ts([0.0], [1.0]), {"rms_tol": 1.0}, "two samples"),
        (_ts(T, np.r_[np.nan, T[1:]]), {"rms_tol": 1e-3}, "finite"),
        (_ts(T, T, jump_times=jnp.asarray([6.0])), {"rms_tol": 1e-3}, "jumps"),
        (_ts(T, T), {"rms_tol": 1e-3, "max_knots": -1}, "max_knots"),
        (_ts(T, T), {"rms_tol": 1e-3, "max_knots": 2.5}, "max_knots"),
        (_ts(T, T), {"rms_tol": 1e-3, "max_knots": 2.0}, "max_knots"),
        (_ts(T, T), {"rms_tol": 1e-3, "max_knots": True}, "max_knots"),
    ],
)
def test_invalid_input_is_rejected(series, kwargs, match):
    with pytest.raises(ValueError, match=match):
        fit_monotone_timeseries_spline(series, **kwargs, increasing=True)


def test_budget_below_noise_hits_default_knot_cap():
    # 241 samples give a default cap of 24 interior knots.
    noisy = T + 0.01 * RNG.standard_normal(len(T))
    noisy[[0, -1]] = T[[0, -1]]
    with pytest.raises(
        RuntimeError, match=r"max_knots=24 .*noise level.*not be monotone"
    ):
        fit_monotone_timeseries_spline(_ts(T, noisy), rms_tol=1e-4, increasing=True)


def test_short_series_default_cap_allows_a_sharp_step():
    times = np.arange(7.0)
    values = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    fitted = fit_monotone_timeseries_spline(
        _ts(times, values), rms_tol=1e-3, increasing=True
    )
    assert 7 < fitted.metadata["monotone_fit"]["interior_knots"] <= 16
    _assert_contract(fitted, times, values, 1e-3)


def test_explicit_knot_cap_is_exact():
    values = np.floor(T * 4.0) / 4.0  # many steps between adjacent samples
    with pytest.raises(RuntimeError, match=r"max_knots=5 with 5 interior knots"):
        fit_monotone_timeseries_spline(
            _ts(T, values), rms_tol=1e-4, max_knots=5, increasing=True
        )


def test_accepts_budget_when_refinement_margin_is_out_of_reach(monkeypatch):
    monkeypatch.setattr(monotone_spline, "_ACCEPT_FRACTION", 0.0)
    values = np.tanh((T - 6.0) / 2.0)
    fitted = fit_monotone_timeseries_spline(
        _ts(T, values), rms_tol=0.2, max_knots=2, increasing=True
    )
    assert fitted.metadata["monotone_fit"]["interior_knots"] == 2
    _assert_contract(fitted, T, values, 0.2)


def test_solver_failure_is_reported(monkeypatch):
    monkeypatch.setattr(monotone_spline, "_solve_conic", lambda *args: None)
    with pytest.raises(
        RuntimeError,
        match="did not converge with rms_tol=0.005.*noise standard deviation",
    ):
        fit_monotone_timeseries_spline(
            _ts(T, 5.0 * RAMP), rms_tol=5e-3, increasing=True
        )


def test_smoothing_failure_is_reported(monkeypatch):
    monkeypatch.setattr(monotone_spline, "_smoothest", lambda *args: None)
    with pytest.raises(RuntimeError, match="smoothing problem did not solve.*clustered samples"):
        fit_monotone_timeseries_spline(
            _ts(T, 5.0 * RAMP), rms_tol=5e-3, increasing=True
        )


def test_out_of_budget_solution_is_never_returned(monkeypatch):
    def linear(design, z, budget, *args):
        return np.linspace(0.0, 1.0, design.shape[1])  # monotone, misses the ramp

    monkeypatch.setattr(monotone_spline, "_solve_conic", linear)
    monkeypatch.setattr(monotone_spline, "_ACCEPT_FRACTION", 1e9)
    with pytest.raises(RuntimeError, match="RMS error .* exceeds rms_tol"):
        fit_monotone_timeseries_spline(
            _ts(T, 5.0 * RAMP), rms_tol=5e-3, increasing=True
        )


def test_endpoint_drift_is_never_returned(monkeypatch):
    project = monotone_spline._project
    monkeypatch.setattr(monotone_spline, "_project", lambda c: project(c) * (1 - 1e-9))
    with pytest.raises(RuntimeError, match="endpoint error"):
        fit_monotone_timeseries_spline(
            _ts(T, 5.0 * RAMP), rms_tol=5e-3, increasing=True
        )


def test_preserves_input_and_discards_its_metadata():
    smooth = fit_timeseries_spline(_ts(T, 5.0 * RAMP), smoothing_s=0.5)
    series = TimeSeries(
        times=smooth.times,
        values=smooth.values,
        breaks=smooth.breaks,
        coeffs=smooth.coeffs,
        segment_start_piece_idx=smooth.segment_start_piece_idx,
        continuity_side="left",
        metadata={**smooth.metadata, "source": "scale"},
    )
    before = np.asarray(series.coeffs).copy()
    fitted = fit_monotone_timeseries_spline(
        series, rms_tol=np.float64(5e-3), increasing=True
    )
    np.testing.assert_array_equal(np.asarray(series.coeffs), before)
    assert fitted.continuity_side == "left"
    assert type(fitted.metadata["monotone_fit"]["rms_tol"]) is float
    assert set(fitted.metadata) == {"monotone_fit"}
    assert set(fitted.metadata["monotone_fit"]) == {
        "rms_tol",
        "direction",
        "interior_knots",
        "refinements",
        "rms_error",
        "max_error",
    }
    json.dumps(fitted.metadata)


def test_json_roundtrip_keeps_fit():
    fitted = fit_monotone_timeseries_spline(
        _ts(T, 5.0 * RAMP), rms_tol=5e-3, increasing=True
    )
    process = BioProcess(
        metadata=BioProcessMetadata(name="p", process_type="fed-batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=12.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="m", density=1.0, density_unit="kg/L"),
        process_variables={
            "x": ProcessVariable(name="x", unit="g", is_controlled=True, values=fitted)
        },
    )
    collection = BioProcessCollection(
        case_id="cs", organism="E. coli", citation="test", processes={"p": process}
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "c.json"
        save_process_collection(collection, path)
        loaded = load_process_collection(path).processes["p"].process_variables["x"]
    restored = loaded.values
    np.testing.assert_array_equal(
        np.asarray(restored.breaks), np.asarray(fitted.breaks)
    )
    np.testing.assert_array_equal(
        np.asarray(restored.coeffs), np.asarray(fitted.coeffs)
    )
    assert restored.metadata["monotone_fit"] == fitted.metadata["monotone_fit"]
    _assert_contract(restored, T, 5.0 * RAMP, 5e-3)


@pytest.mark.parametrize("increasing", [True, False])
def test_endpoint_direction_must_match_request(increasing):
    values = [3.0, 1.0] if increasing else [1.0, 3.0]
    with pytest.raises(ValueError, match=rf"{values[0]} and {values[1]}.*increasing"):
        fit_monotone_timeseries_spline(
            _ts([0.0, 1.0], values), rms_tol=0.1, increasing=increasing
        )


@pytest.mark.parametrize("rms_tol", [1e-3, 1e3])
def test_linear_fit_has_negligible_curvature_across_relative_budgets(rms_tol):
    times = np.linspace(0.0, 1.0, 101)
    fitted = fit_monotone_timeseries_spline(
        _ts(times, times), rms_tol=rms_tol, increasing=True
    )
    assert fitted.metadata["monotone_fit"]["interior_knots"] == 0
    assert _roughness(fitted) < 1e-6
    _assert_contract(fitted, times, times, rms_tol)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_extreme_value_units_preserve_constant_rms_budget(scale):
    series = _ts([0.0, 1.0, 2.0], [0.0, scale, 0.0])
    fitted = fit_monotone_timeseries_spline(series, rms_tol=scale, increasing=True)
    assert fitted.metadata["monotone_fit"]["rms_error"] / scale == pytest.approx(
        1 / np.sqrt(3)
    )
    with pytest.raises(ValueError, match="Endpoints are equal"):
        fit_monotone_timeseries_spline(series, rms_tol=scale / 10, increasing=True)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_extreme_value_units_preserve_nonconstant_rms_budget(scale):
    times = np.linspace(0.0, 1.0, 101)
    values = scale * times**2
    fitted = fit_monotone_timeseries_spline(
        _ts(times, values), rms_tol=scale * 1e-3, increasing=True
    )
    evaluated = np.asarray(fitted.evaluate_many(fitted.times)) / scale
    expected_rms = np.linalg.norm(evaluated - times**2) / np.sqrt(len(times))
    assert expected_rms <= 1e-3
    assert fitted.metadata["monotone_fit"]["rms_error"] / scale == pytest.approx(
        expected_rms
    )


def test_project_enforces_anchors_and_increment_floor():
    projected = monotone_spline._project(np.array([0.01, -0.01, -0.01, 0.5, 1.02]))
    assert projected[0] == 0.0 and projected[-1] == 1.0
    np.testing.assert_allclose(projected[1:3], [1e-9, 2e-9], rtol=1e-15, atol=0.0)
    assert np.all(np.diff(projected) >= 1e-9)
    # The two positive increments retain their 51:52 ratio up to the floor.
    assert projected[3] == pytest.approx(51 / 103, abs=3e-9, rel=0.0)


def test_stored_derivative_sign_is_checked():
    # Endpoints and RMS pass, but the rate is negative at the start.
    assert (
        monotone_spline._check_stored(
            np.array([0.0, 1.0]),
            np.array([[0.0, -1.0, 2.0, 0.0]]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            0.1,
            1.0,
        )
        == "derivative changes sign"
    )


def test_stored_value_continuity_is_checked():
    # Samples, endpoints and rates pass despite a jump between pieces.
    problem = monotone_spline._check_stored(
        np.array([0.0, 0.5, 1.0]),
        np.array([[0.0, 1.0, 0.0, 0.0], [0.75, 0.5, 0.0, 0.0]]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        0.1,
        1.0,
    )
    assert "value discontinuity" in problem


EXACT_SHAPES = {
    "line": lambda x: x,
    "quadratic": lambda x: 0.1 * x + 0.9 * x**2,
    "quadratic_zero_initial_slope": lambda x: x**2,
    "cubic": lambda x: x + x**3,
    "cubic_zero_initial_slope": lambda x: x**3,
}


@pytest.mark.parametrize("shape", EXACT_SHAPES)
@pytest.mark.parametrize("n_samples", [6, 11, 101, 1000])
@pytest.mark.parametrize("relative_tol", [1e-2, 1e-3, 2e-4, 1e-5])
def test_noise_free_polynomials_fit(shape, n_samples, relative_tol):
    # A single cubic reproduces these exactly or almost exactly, which puts the
    # least-RMS optimum at or next to the tip of its cone.
    times = np.linspace(0.0, 12.0, n_samples)
    values = EXACT_SHAPES[shape](times / 12.0)
    rms_tol = relative_tol * values[-1]
    fitted = fit_monotone_timeseries_spline(
        _ts(times, values), rms_tol=rms_tol, increasing=True
    )
    _assert_contract(fitted, times, values, rms_tol)


@pytest.mark.parametrize(
    ("times", "values", "rms_tol"),
    [
        (np.arange(6.0), np.arange(6.0), 1e-3),
        (np.linspace(0.0, 24.0, 1000), 0.5 * np.linspace(0.0, 24.0, 1000) ** 2, 0.01),
    ],
    ids=["constant_rate_counter", "linearly_ramped_pump"],
)
def test_noise_free_pump_traces_fit(times, values, rms_tol):
    fitted = fit_monotone_timeseries_spline(
        _ts(times, values), rms_tol=rms_tol, increasing=True
    )
    _assert_contract(fitted, times, values, rms_tol)


@pytest.mark.parametrize("n_samples", [12, 17, 20, 40])
@pytest.mark.parametrize("relative_tol", [1e-3, 1e-4, 2e-5])
@pytest.mark.parametrize(
    "shape",
    [
        lambda t: np.clip((t - 6.0) / 0.1, 0.0, 1.0),  # transfer between readings
        lambda t: np.clip((t - 18.0) / 0.1, 0.0, 1.0),
        lambda t: np.minimum(t, 12.0) / 12.0,  # feed, then pause
    ],
    ids=["step_early", "step_late", "feed_then_pause"],
)
def test_noise_free_plateaus_fit(shape, n_samples, relative_tol):
    # Exact plateaus can stop the least-RMS solve before it fully converges.
    times = np.linspace(0.0, 24.0, n_samples)
    values = 100.0 * shape(times)
    rms_tol = relative_tol * 100.0
    fitted = fit_monotone_timeseries_spline(
        _ts(times, values), rms_tol=rms_tol, increasing=True, max_knots=100
    )
    _assert_contract(fitted, times, values, rms_tol)


@pytest.mark.parametrize(
    ("times", "values", "rms_tol"),
    [
        (
            np.linspace(0.0, 24.0, 20),
            100.0 * np.clip((np.linspace(0.0, 24.0, 20) - 6.0) / 0.1, 0.0, 1.0),
            0.01,
        ),
        (np.arange(17.0), 1000.0 * np.minimum(np.arange(17.0), 8.0), 0.16),
    ],
    ids=["step_transfer", "feed_then_pause_counter"],
)
def test_review_plateau_cases_fit(times, values, rms_tol):
    fitted = fit_monotone_timeseries_spline(
        _ts(times, values), rms_tol=rms_tol, increasing=True
    )
    _assert_contract(fitted, times, values, rms_tol)


@pytest.mark.parametrize("status", ["AlmostSolved", "MaxIterations"])
def test_unconverged_solves_are_accepted_only_for_the_least_rms_step(
    monkeypatch, status
):
    class Unconverged:
        def __init__(self, P, q, A, b, cones, settings):
            self.n = len(q)

        def solve(self):
            solver_status = getattr(monotone_spline.clarabel.SolverStatus, status)
            return SimpleNamespace(status=solver_status, x=np.zeros(self.n))

    monkeypatch.setattr(monotone_spline.clarabel, "DefaultSolver", Unconverged)
    knots = np.r_[[0.0] * 4, [1.0] * 4]
    x = np.linspace(0.0, 1.0, 11)
    design = BSpline.design_matrix(x, knots, 3).tocsc()
    least_rms = monotone_spline._solve_conic(design, x, 1e-3, None, None)
    smoothest = monotone_spline._smoothest(design, x, 1e-3, 0.999, knots)
    assert least_rms is not None
    assert smoothest is None


def test_nonfinite_least_rms_result_is_reported(monkeypatch):
    class NonFinite:
        def __init__(self, P, q, A, b, cones, settings):
            self.n = len(q)

        def solve(self):
            status = monotone_spline.clarabel.SolverStatus.NumericalError
            return SimpleNamespace(status=status, x=np.full(self.n, np.nan))

    monkeypatch.setattr(monotone_spline.clarabel, "DefaultSolver", NonFinite)
    with pytest.raises(RuntimeError, match="did not converge"):
        fit_monotone_timeseries_spline(_ts(T, T), rms_tol=1e-3, increasing=True)


@pytest.mark.parametrize("rms_tol", [0.0065, 0.065])
def test_burst_logging_is_smoothed_to_optimality(monkeypatch, rms_tol):
    # A 144-second logging burst clusters knots, which makes the curvature matrix
    # badly scaled. The fit must still reach the curvature minimum: a re-solve
    # with the objective scaled up (same minimizer) must not find a smoother curve.
    times = np.unique(np.r_[np.linspace(0.0, 24.0, 30), np.linspace(5.99, 6.03, 301)])
    values = 10.0 * np.minimum(times, 4.0) + 5.0 * np.clip(times - 12.0, 0.0, 5.0)
    fitted = fit_monotone_timeseries_spline(
        _ts(times, values), rms_tol=rms_tol, increasing=True
    )
    _assert_contract(fitted, times, values, rms_tol)
    solve = monotone_spline._solve_conic

    def scaled_objective(design, z, budget, P, rms_bound):
        return solve(design, z, budget, None if P is None else 1e3 * P, rms_bound)

    monkeypatch.setattr(monotone_spline, "_solve_conic", scaled_objective)
    reference = fit_monotone_timeseries_spline(
        _ts(times, values), rms_tol=rms_tol, increasing=True
    )
    assert _roughness(fitted) <= 1.01 * _roughness(reference)


def test_increasing_must_be_a_bool():
    with pytest.raises(TypeError, match="increasing must be a bool"):
        fit_monotone_timeseries_spline(_ts(T, T), rms_tol=1e-3, increasing=None)


@pytest.mark.parametrize(
    "times",
    [12001.0 - 12.0 + T, 2.3e5 + T, 4.5e4 + T / 24.0, 1.7e9 + 3600.0 * T],
    ids=["just_over_limit", "hours_since_2000", "excel_days", "epoch_seconds"],
)
def test_large_time_offsets_are_rejected_upfront(times):
    with pytest.raises(ValueError, match="subtract the start time"):
        fit_monotone_timeseries_spline(_ts(times, RAMP), rms_tol=1e-3, increasing=True)


def test_time_offset_at_the_limit_is_accepted():
    times = 12000.0 - 12.0 + T  # largest time exactly 1000 durations from zero
    fitted = fit_monotone_timeseries_spline(
        _ts(times, RAMP), rms_tol=1e-3, increasing=True
    )
    _assert_contract(fitted, times, RAMP, 1e-3)


def test_refinement_measures_the_projected_least_rms_curve(monkeypatch):
    # An inexact solver result that overshoots the end anchor must be judged as
    # its projected (exactly anchored) curve, which already fits the line.
    solve = monotone_spline._solve_conic

    def overshooting(design, z, budget, P, rms_bound):
        c = solve(design, z, budget, P, rms_bound)
        return c * 1.05 if rms_bound is None else c

    monkeypatch.setattr(monotone_spline, "_solve_conic", overshooting)
    fitted = fit_monotone_timeseries_spline(_ts(T, T), rms_tol=1e-3, increasing=True)
    assert fitted.metadata["monotone_fit"]["interior_knots"] == 0


def test_left_continuous_output_is_checked_as_stored():
    # At a large value offset, the two pieces at a break differ by roundoff. The
    # check must use the side the returned series evaluates with.
    values = 1e11 + 3.7 * RAMP
    series = _ts(T, values, continuity_side="left")
    with pytest.raises(RuntimeError, match="exceeds rms_tol"):
        fit_monotone_timeseries_spline(series, rms_tol=3.7e-4, increasing=True)
    fitted = fit_monotone_timeseries_spline(series, rms_tol=3.8e-4, increasing=True)
    evaluated = np.asarray(fitted.evaluate_many(fitted.times))
    rms = np.sqrt(np.mean((evaluated - values) ** 2))
    assert rms <= 3.8e-4
    assert fitted.metadata["monotone_fit"]["rms_error"] == pytest.approx(rms, rel=1e-9)


def test_nan_endpoint_tolerance_fails():
    with np.errstate(over="ignore", invalid="ignore"):
        problem = monotone_spline._check_stored(
            np.array([0.0, 1e110]),
            np.array([[0.0, 1e-110, 0.0, 0.0]]),
            np.array([0.0, 1e110]),
            np.array([0.0, 1.0]),
            0.1,
            1.0,
        )
    assert problem is not None and "endpoint error" in problem
