"""Tests for hybrax.format.splines.fit_monotone_timeseries_spline."""

import json
import tempfile
from pathlib import Path

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


def _assert_contract(fitted, times, values, atol):
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
    assert np.sqrt(np.mean((evaluated - values) ** 2)) <= atol
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

    # C1 and C2 continuity at interior breaks.
    h = widths[:-1]
    left_rate = coeffs[:-1, 1] + 2 * coeffs[:-1, 2] * h + 3 * coeffs[:-1, 3] * h**2
    left_curvature = 2 * coeffs[:-1, 2] + 6 * coeffs[:-1, 3] * h
    rate_scale = max(abs(net), atol) / (times[-1] - times[0])
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


CASES = {  # times, values, atol
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
    "large_time_offset": (1e9 + T, RAMP, 1e-3),
    "tiny_negative_change": (T, -5.0 - 1e-9 * RAMP, 1e-12),
    "two_samples": (np.array([0.0, 1.0]), np.array([1.0, 3.0]), 1e-6),
    "three_samples": (np.array([0.0, 1.0, 5.0]), np.array([0.0, 0.1, 2.0]), 1e-6),
}


@pytest.mark.parametrize("name", CASES)
def test_fit_meets_contract(name):
    times, values, atol = CASES[name]
    fitted = fit_monotone_timeseries_spline(_ts(times, values), atol=atol)
    _assert_contract(fitted, times, values, atol)
    info = fitted.metadata["monotone_fit"]
    assert info["atol"] == atol
    assert info["rms_error"] <= atol
    expected = "increasing" if values[-1] > values[0] else "decreasing"
    assert info["direction"] == expected


def test_smooth_data_needs_few_knots_and_tight_budget_more():
    values = np.tanh((T - 6.0) / 2.0)
    loose = fit_monotone_timeseries_spline(_ts(T, values), atol=1e-2)
    tight = fit_monotone_timeseries_spline(_ts(T, values), atol=1e-5)
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
    smoothed = fit_monotone_timeseries_spline(_ts(T, values), atol=5e-3)
    info = smoothed.metadata["monotone_fit"]
    assert info["coefficient_source"] == "smoothed"
    assert info["rms_error"] >= 0.99 * 5e-3
    monkeypatch.setattr(monotone_spline, "_smoothest", lambda *args: None)
    closest = fit_monotone_timeseries_spline(_ts(T, values), atol=5e-3)
    assert _roughness(smoothed) < _roughness(closest)


def test_knots_concentrate_at_local_features():
    fitted = fit_monotone_timeseries_spline(_ts(T, 5.0 * RAMP), atol=5e-3)
    breaks = np.asarray(fitted.breaks)
    near_ramp = np.sum((breaks > 2.5) & (breaks < 3.7))
    assert near_ramp >= (len(breaks) - 2) / 2


def test_unit_rescaling_preserves_accuracy():
    values = 4.0 * RAMP + 0.3 * T
    base = fit_monotone_timeseries_spline(_ts(T, values), atol=1e-3)
    # Time in seconds and values in grams: scale atol with the value unit.
    scaled = fit_monotone_timeseries_spline(_ts(3600.0 * T, 1000.0 * values), atol=1.0)
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
    fitted = fit_monotone_timeseries_spline(_ts(times, values), atol=1e-3)
    assert fitted.metadata["monotone_fit"]["interior_knots"] > 0
    _assert_contract(fitted, times, values, 1e-3)


def test_budget_is_pooled_so_isolated_dips_are_smoothed_over():
    values = T.copy()
    values[120] -= 0.05  # a reversal well below the previous sample
    fitted = fit_monotone_timeseries_spline(_ts(T, values), atol=0.01)
    _assert_contract(fitted, T, values, 0.01)
    assert fitted.metadata["monotone_fit"]["max_error"] > 0.01
    np.testing.assert_array_equal(np.asarray(fitted.values), values)


def test_constant_data_gives_constant_spline():
    fitted = fit_monotone_timeseries_spline(_ts(T, np.full_like(T, 2.0)), atol=1e-12)
    np.testing.assert_array_equal(np.asarray(fitted.coeffs), [[2.0, 0.0, 0.0, 0.0]])
    assert fitted.metadata["monotone_fit"]["direction"] == "constant"


def test_equal_endpoints_accept_excursion_within_budget():
    values = 1e-3 * np.sin(np.pi * T / 12.0) ** 2
    values[-1] = values[0]
    fitted = fit_monotone_timeseries_spline(_ts(T, values), atol=7e-4)
    np.testing.assert_array_equal(np.asarray(fitted.coeffs)[:, 1:], 0.0)
    _assert_contract(fitted, T, values, 7e-4)


def test_equal_endpoints_reject_excursion_beyond_budget():
    values = np.r_[np.sin(np.pi * T[:-1] / 12.0), 0.0]
    with pytest.raises(ValueError, match="Endpoints are equal"):
        fit_monotone_timeseries_spline(_ts(T, values), atol=1e-3)


@pytest.mark.parametrize(
    ("series", "kwargs", "match"),
    [
        (_ts(T, T), {"atol": 0.0}, "atol must be"),
        (_ts(T, T), {"atol": -1.0}, "atol must be"),
        (_ts(T, T), {"atol": float("nan")}, "atol must be"),
        (_ts(T, T), {"atol": float("inf")}, "atol must be"),
        (_ts([0.0], [1.0]), {"atol": 1.0}, "two samples"),
        (_ts(T, np.r_[np.nan, T[1:]]), {"atol": 1e-3}, "finite"),
        (_ts(T, T, jump_times=jnp.asarray([6.0])), {"atol": 1e-3}, "jumps"),
        (_ts(T, T), {"atol": 1e-3, "max_knots": -1}, "max_knots"),
    ],
)
def test_invalid_input_is_rejected(series, kwargs, match):
    with pytest.raises(ValueError, match=match):
        fit_monotone_timeseries_spline(series, **kwargs)


def test_budget_below_noise_hits_default_knot_cap():
    # 241 samples give a default cap of 24 interior knots.
    noisy = T + 0.01 * RNG.standard_normal(len(T))
    noisy[[0, -1]] = T[[0, -1]]
    with pytest.raises(RuntimeError, match=r"max_knots=24 .*noise level"):
        fit_monotone_timeseries_spline(_ts(T, noisy), atol=1e-4)


def test_short_series_default_cap_allows_a_sharp_step():
    times = np.arange(7.0)
    values = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    fitted = fit_monotone_timeseries_spline(_ts(times, values), atol=1e-3)
    assert 7 < fitted.metadata["monotone_fit"]["interior_knots"] <= 16
    _assert_contract(fitted, times, values, 1e-3)


def test_explicit_knot_cap_is_exact():
    values = np.floor(T * 4.0) / 4.0  # many steps between adjacent samples
    with pytest.raises(RuntimeError, match=r"max_knots=5 with 5 interior knots"):
        fit_monotone_timeseries_spline(_ts(T, values), atol=1e-4, max_knots=5)


def test_accepts_budget_when_refinement_margin_is_out_of_reach(monkeypatch):
    monkeypatch.setattr(monotone_spline, "_ACCEPT_FRACTION", 0.0)
    values = np.tanh((T - 6.0) / 2.0)
    fitted = fit_monotone_timeseries_spline(_ts(T, values), atol=0.2, max_knots=2)
    assert fitted.metadata["monotone_fit"]["interior_knots"] == 2
    _assert_contract(fitted, T, values, 0.2)


def test_solver_failure_is_reported(monkeypatch):
    monkeypatch.setattr(monotone_spline, "_solve_conic", lambda *args: None)
    with pytest.raises(RuntimeError, match="least-RMS problem did not solve"):
        fit_monotone_timeseries_spline(_ts(T, 5.0 * RAMP), atol=5e-3)


def test_falls_back_to_least_rms_solution_when_smoothing_fails(monkeypatch):
    monkeypatch.setattr(monotone_spline, "_smoothest", lambda *args: None)
    fitted = fit_monotone_timeseries_spline(_ts(T, 5.0 * RAMP), atol=5e-3)
    assert fitted.metadata["monotone_fit"]["coefficient_source"] == "least_rms"
    _assert_contract(fitted, T, 5.0 * RAMP, 5e-3)


def test_out_of_budget_solution_is_never_returned(monkeypatch):
    def linear(design, z, budget):
        return np.linspace(0.0, 1.0, design.shape[1])  # monotone, misses the ramp

    monkeypatch.setattr(monotone_spline, "_least_rms", linear)
    monkeypatch.setattr(monotone_spline, "_smoothest", lambda *args: None)
    monkeypatch.setattr(monotone_spline, "_ACCEPT_FRACTION", 1e9)
    with pytest.raises(RuntimeError, match="RMS error .* exceeds atol"):
        fit_monotone_timeseries_spline(_ts(T, 5.0 * RAMP), atol=5e-3)


def test_endpoint_drift_is_never_returned(monkeypatch):
    project = monotone_spline._project
    monkeypatch.setattr(monotone_spline, "_project", lambda c: project(c) * (1 - 1e-9))
    with pytest.raises(RuntimeError, match="endpoint error"):
        fit_monotone_timeseries_spline(_ts(T, 5.0 * RAMP), atol=5e-3)


def test_preserves_input_and_replaces_fit_metadata():
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
    fitted = fit_monotone_timeseries_spline(series, atol=5e-3)
    np.testing.assert_array_equal(np.asarray(series.coeffs), before)
    assert fitted.continuity_side == "left"
    assert fitted.metadata["source"] == "scale"
    assert set(fitted.metadata) == {"source", "monotone_fit"}
    json.dumps(fitted.metadata)


def test_json_roundtrip_keeps_fit():
    fitted = fit_monotone_timeseries_spline(_ts(T, 5.0 * RAMP), atol=5e-3)
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
