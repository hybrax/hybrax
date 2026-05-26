from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from bp_format.time_series import PPoly, TimeSeries


def test_ppoly_scalar_side_extrapolation_and_derivative() -> None:
    poly = PPoly(
        [0.0, 1.0, 2.0],
        [[1.0, 1.0, 0.0, 0.0], [10.0, 2.0, 0.0, 0.0]],
    )
    assert float(poly(1.0, side="left")) == pytest.approx(2.0)
    assert float(poly(1.0, side="right")) == pytest.approx(10.0)
    assert float(poly(-1.0)) == pytest.approx(0.0)
    assert float(poly(3.0)) == pytest.approx(14.0)
    np.testing.assert_allclose(np.asarray(poly([0.0, 1.5], nu=1)), [1.0, 2.0])
    np.testing.assert_allclose(np.asarray(poly.derivative()([0.0, 1.5])), [1.0, 2.0])


def test_ppoly_multi_output_grid_shape() -> None:
    coeffs = np.zeros((2, 4, 2, 3), dtype=float)
    coeffs[:, 0, :, :] = np.arange(6).reshape(2, 3)
    coeffs[:, 1, :, :] = 1.0
    poly = PPoly([0.0, 1.0, 2.0], coeffs)
    out = poly(np.asarray([[0.0, 0.5], [1.0, 1.5]]))
    assert out.shape == (2, 2, 2, 3)
    np.testing.assert_allclose(out[0, 1], coeffs[0, 0] + 0.5)
    np.testing.assert_allclose(out[1, 1], coeffs[1, 0] + 0.5)


def test_ppoly_from_scipy_ppoly_matches_scipy_cubic_spline() -> None:
    t = np.asarray([0.0, 1.0, 2.0, 3.0])
    y = np.asarray([0.0, 1.0, 0.0, 1.0])
    scipy_poly = CubicSpline(t, y, bc_type="natural")
    poly = PPoly.from_scipy_ppoly(scipy_poly)
    probe = np.linspace(0.0, 3.0, 13)
    np.testing.assert_allclose(np.asarray(poly(probe)), scipy_poly(probe), atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(poly(probe, nu=1)),
        scipy_poly(probe, nu=1),
        atol=1e-12,
    )


def test_ppoly_sample_constructor_matches_samples() -> None:
    t = np.asarray([0.0, 1.0, 2.0, 3.0])
    y = np.asarray([0.0, 1.0, 0.0, 1.0])
    poly = PPoly.from_samples_pchip(t, y)
    np.testing.assert_allclose(np.asarray(poly(t)), y, atol=1e-12)


def test_timeseries_uses_owned_ppoly_without_serializing_it() -> None:
    ts = TimeSeries(
        times=[0.0, 1.0],
        values=[1.0, 3.0],
        derived=True,
        breaks=[0.0, 1.0],
        coeffs=[[1.0, 2.0, 0.0, 0.0]],
        segment_start_piece_idx=[0],
        metadata={"source": "test"},
    )

    assert isinstance(ts.poly, PPoly)
    assert float(ts.evaluate(0.5)) == pytest.approx(2.0)
    np.testing.assert_allclose(
        np.asarray(ts.evaluate_many([0.0, 0.5, 1.0])),
        [1.0, 2.0, 3.0],
    )

    derivative = ts.deriv()
    assert isinstance(derivative.poly, PPoly)
    assert derivative.derived is True
    assert derivative.metadata == {"source": "test"}
    assert float(derivative.evaluate(0.25)) == pytest.approx(2.0)

    payload = ts.to_dict()
    assert "poly" not in payload
    roundtrip = TimeSeries.from_dict(payload)
    assert isinstance(roundtrip.poly, PPoly)
    np.testing.assert_allclose(np.asarray(roundtrip.breaks), np.asarray(ts.breaks))
    np.testing.assert_allclose(np.asarray(roundtrip.coeffs), np.asarray(ts.coeffs))
    assert roundtrip.metadata == {"source": "test"}


def test_timeseries_accepts_owned_ppoly_with_canonical_segment_start() -> None:
    poly = PPoly([0.0, 1.0], [[1.0, 2.0, 0.0, 0.0]])
    ts = TimeSeries(poly=poly, segment_start_piece_idx=[0])

    np.testing.assert_allclose(np.asarray(ts.breaks), np.asarray(poly.breaks))
    np.testing.assert_allclose(np.asarray(ts.coeffs), np.asarray(poly.coeffs))
    assert float(ts.evaluate(0.5)) == pytest.approx(2.0)


def test_timeseries_rejects_mismatched_owned_ppoly_and_spline_state() -> None:
    poly = PPoly([0.0, 1.0], [[1.0, 2.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="breaks"):
        TimeSeries(
            poly=poly,
            breaks=[0.0, 2.0],
            coeffs=[[1.0, 2.0, 0.0, 0.0]],
            segment_start_piece_idx=[0],
        )

    with pytest.raises(ValueError, match="coeffs"):
        TimeSeries(
            poly=poly,
            breaks=[0.0, 1.0],
            coeffs=[[1.0, 3.0, 0.0, 0.0]],
            segment_start_piece_idx=[0],
        )
