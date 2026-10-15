"""Analytic checks for pilot-guided monotone-spline refinement."""

import warnings

import numpy as np
import pytest

from hybrax.format import TimeSeries, monotone_spline
from hybrax.format.splines import fit_monotone_timeseries_spline


@pytest.mark.parametrize("sign", [-1, 1])
def test_cubic_pilot_curvature_variation(sign):
    x = np.linspace(0, 1, 21)
    edges, density = monotone_spline._pilot_guide(x, sign * x**3, 0.01, 32)
    np.testing.assert_allclose(edges, [0.0, 1.0])
    np.testing.assert_allclose(density, [6.0], rtol=1e-10)


def test_short_pilot_has_no_curvature_information():
    edges, density = monotone_spline._pilot_guide(
        np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.1, 64
    )
    np.testing.assert_array_equal(edges, [0.0, 1.0])
    np.testing.assert_array_equal(density, [0.0])


def test_pilot_does_not_suppress_unrelated_runtime_warnings(monkeypatch):
    smooth = monotone_spline.make_splrep

    def warn(*args, **kwargs):
        # Attribute it to SciPy so only the message restriction can distinguish
        # this warning from the known, acceptable pilot iteration-limit warning.
        warnings.warn_explicit(
            "unrelated pilot problem",
            RuntimeWarning,
            filename=__file__,
            lineno=1,
            module="scipy.interpolate._fitpack_repro",
        )
        return smooth(*args, **kwargs)

    monkeypatch.setattr(monotone_spline, "make_splrep", warn)
    x = np.linspace(0, 1, 21)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning, match="unrelated pilot problem"):
            monotone_spline._fit_pilot(x, x**3, 0.01, 32)


def test_pilot_target_is_clipped_for_four_samples():
    x = np.linspace(0, 1, 4)
    edges, density = monotone_spline._pilot_guide(x, x**3, 0.01, 64)
    np.testing.assert_allclose(edges, [0.0, 1.0])
    np.testing.assert_allclose(density, [6.0], rtol=1e-10)


def test_guide_median_integrates_in_time_and_crops_to_interval():
    # Mass 1 on [0, .25], mass 3 on [.25, 1]; half lies at .5.
    guide = np.array([0.0, 0.25, 1.0]), np.array([4.0, 4.0])
    assert monotone_spline._guide_location(0.0, 1.0, guide) == pytest.approx(0.5)
    # Discard the large mass outside [.6, .8], leaving a uniform density there.
    guide = np.array([0.0, 0.5, 1.0]), np.array([100.0, 2.0])
    assert monotone_spline._guide_location(0.6, 0.8, guide) == pytest.approx(0.7)


def test_guide_without_curvature_variation_returns_none():
    guide = np.array([0.0, 1.0]), np.zeros(1)
    assert monotone_spline._guide_location(0.0, 1.0, guide) is None


def test_curvature_guides_location_independently_of_sample_burst():
    x = np.array([0.0, 0.1, 0.11, 0.12, 0.13, 0.14, 0.9, 1.0])
    guide = np.array([0.0, 0.6, 0.7, 1.0]), np.array([0.0, 1.0, 0.0])
    knots = monotone_spline._choose_knots(x, np.ones(len(x)), np.array([]), 1, guide)
    assert knots == pytest.approx([0.65])


def test_scores_count_samples_equally_and_locations_are_guarded():
    x = np.array([0, 0.1, 0.11, 0.12, 0.13, 0.14, 0.5, 0.7, 0.8, 1])
    residual = np.array([0, 1, 1, 1, 1, 1, 0, 1, 1, 0])
    guide = np.array([0.0, 0.5, 0.8, 1.0]), np.array([0.0, 1.0, 0.0])
    assert monotone_spline._choose_knots(x, residual, np.array([0.5]), 1, guide) == [
        0.125
    ]
    guide = np.array([0.0, 0.1, 1.0]), np.array([1.0, 0.0])
    assert monotone_spline._choose_knots(
        x, np.ones(len(x)), np.array([]), 1, guide
    ) == [0.25]


@pytest.mark.parametrize(
    "samples, expected",
    [([0, 0.2, 0.3, 0.4, 0.8, 1], 0.35), ([0, 0.2, 0.8, 1], 0.5), ([0, 1], 0.5)],
)
def test_zero_mass_uses_original_median_or_midpoint(samples, expected):
    x = np.array(samples)
    guide = np.array([0.0, 1.0]), np.zeros(1)
    knots = monotone_spline._choose_knots(x, np.ones(len(x)), np.array([]), 1, guide)
    assert knots == pytest.approx([expected])


@pytest.mark.parametrize("target", [32, 64])
def test_infeasible_pilot_budget_uses_bounded_lsq(monkeypatch, target):
    x = np.linspace(0, 1, 401)
    y = x + 0.01 * np.sin(2 * np.pi * 100 * x)
    generate = monotone_spline.generate_knots
    nests = []

    def capture(*args, **kwargs):
        nests.append(kwargs["nest"])
        return generate(*args, **kwargs)

    def impossible_smoothing(*args, **kwargs):
        pytest.fail("Do not ask smoothing to meet an infeasible pilot budget")

    monkeypatch.setattr(monotone_spline, "generate_knots", capture)
    monkeypatch.setattr(monotone_spline, "make_splrep", impossible_smoothing)
    pilot = monotone_spline._fit_pilot(x, y, 1e-6, target)
    assert nests == [target + 8]
    assert len(np.unique(pilot.t)) - 2 < len(x) - 4
    assert np.isfinite(pilot.c).all()
    assert np.linalg.norm(pilot(x) - y) / np.sqrt(len(x)) > 1e-6


@pytest.mark.parametrize("constant", [False, True])
def test_initially_feasible_curve_does_not_construct_pilot(monkeypatch, constant):
    def unexpected_pilot(*args):
        pytest.fail("A feasible initial fit does not need a pilot")

    monkeypatch.setattr(monotone_spline, "_pilot_guide", unexpected_pilot)
    x = np.linspace(0, 1, 21)
    y = np.ones_like(x) if constant else x
    fit_monotone_timeseries_spline(
        TimeSeries(times=x, values=y), rms_tol=1e-3, increasing=True
    )


@pytest.mark.parametrize(
    "n, cap, expected", [(101, None, 64), (701, None, 70), (101, 16, 16)]
)
def test_pilot_receives_resolved_final_cap(monkeypatch, n, cap, expected):
    received = []

    def guide(x, z, budget, target):
        received.append(target)
        return np.array([0.0, 1.0]), np.zeros(1)

    monkeypatch.setattr(monotone_spline, "_pilot_guide", guide)
    monkeypatch.setattr(monotone_spline, "_choose_knots", lambda *args: [])
    x = np.linspace(0, 1, n)
    y = 0.1 * x + 0.9 * np.clip((x - 0.45) / 0.1, 0, 1)
    with pytest.raises(RuntimeError, match="cannot place further knots") as error:
        fit_monotone_timeseries_spline(
            TimeSeries(times=x, values=y),
            rms_tol=1e-4,
            increasing=True,
            max_knots=cap,
        )
    assert received == [expected]
    assert "minimum knot spacing" in str(error.value)
    assert "Increase max_knots" not in str(error.value)


@pytest.mark.parametrize("unit_scale", [1.0, 100.0])
def test_pilot_is_reused_with_original_normalized_budget(monkeypatch, unit_scale):
    x = np.linspace(0, 1, 101)
    y = 10 * unit_scale * (0.1 * x + 0.9 * np.clip((x - 0.45) / 0.1, 0, 1))
    tolerance = 0.01 * unit_scale
    guide = monotone_spline._pilot_guide
    received = []

    def capture(x, z, budget, target):
        received.append((budget, target))
        return guide(x, z, budget, target)

    monkeypatch.setattr(monotone_spline, "_pilot_guide", capture)
    fitted = fit_monotone_timeseries_spline(
        TimeSeries(times=x, values=y), rms_tol=tolerance, increasing=True
    )
    assert received == pytest.approx([(0.001, 64)])
    assert fitted.metadata["monotone_fit"]["refinements"] > 1
    rms = np.linalg.norm(np.asarray(fitted.evaluate_many(fitted.times)) - y) / np.sqrt(
        len(x)
    )
    assert rms <= tolerance


@pytest.mark.parametrize("width_in_gaps", [1.5, 3.0])
def test_narrow_interval_tries_midpoint_only_when_it_fits(width_in_gaps):
    gap = monotone_spline._MIN_KNOT_GAP
    a, b = 0.5, 0.5 + width_in_gaps * gap
    x = np.array([0.0, a + gap / 10, b - gap / 10, 1.0])
    residual = np.array([0.0, 1.0, 1.0, 0.0])
    guide = np.array([0.0, a, a + gap / 10, b, 1.0]), np.array([0.0, 1.0, 0.0, 0.0])
    knots = monotone_spline._choose_knots(x, residual, np.array([a, b]), 1, guide)
    if width_in_gaps > 2:
        assert knots == pytest.approx([(a + b) / 2], rel=0, abs=1e-15)
        assert min(knots[0] - a, b - knots[0]) > gap
    else:
        assert knots == []
