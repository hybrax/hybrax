"""Divergence-tolerant LOO scoring: an unscoreable fold scores NaN, never crashes, and the
measurement-node guard still fails loudly for a genuinely-old node-omitting predictions.csv."""

from __future__ import annotations

import numpy as np
import pytest

from bp_train.loo_metrics import (
    _evaluate_predictions_for_process,
    _prediction_unscoreable,
)


def test_prediction_unscoreable_cases():
    t = np.array([0.0, 5.0, 10.0])
    # finite + in-range -> scoreable
    assert _prediction_unscoreable(t, np.array([1.0, 2.0, 3.0]), np.array([0.0, 5.0, 10.0])) is False
    # non-finite values -> unscoreable
    assert _prediction_unscoreable(t, np.array([1.0, np.nan, 3.0]), np.array([0.0, 10.0])) is True
    # measurement beyond the (truncated) prediction grid -> unscoreable
    assert _prediction_unscoreable(np.array([0.0, 5.0]), np.array([1.0, 2.0]), np.array([0.0, 10.0])) is True
    # empty grid -> unscoreable
    assert _prediction_unscoreable(np.array([]), np.array([]), np.array([1.0])) is True


def test_diverged_nonfinite_scores_nan_not_raise():
    out = _evaluate_predictions_for_process(
        pred_t=np.array([0.0, 5.0, 10.0]),
        pred_columns={"c_biomass": np.array([1.0, np.nan, 3.0])},
        measurements={"biomass": (np.array([0.0, 5.0, 10.0]), np.array([1.0, 2.0, 3.0]))},
    )
    assert out["biomass"]["n_measured"] == 0
    assert np.isnan(out["biomass"]["nmae"])


def test_truncated_grid_scores_nan_not_raise():
    # diverged solve stopped at t=5 but measurements go to t=10 -> NaN, NOT a clamped-interp value.
    out = _evaluate_predictions_for_process(
        pred_t=np.array([0.0, 5.0]),
        pred_columns={"c_biomass": np.array([1.0, 2.0])},
        measurements={"biomass": (np.array([0.0, 5.0, 10.0]), np.array([1.0, 2.0, 9.0]))},
    )
    assert np.isnan(out["biomass"]["nmae"])


def test_finite_in_range_nodeless_still_raises():
    # finite prediction whose grid omits an interior measurement node -> the guard must fire loudly.
    with pytest.raises(ValueError, match="no grid node"):
        _evaluate_predictions_for_process(
            pred_t=np.array([0.0, 5.0, 10.0]),
            pred_columns={"c_biomass": np.array([1.0, 2.0, 3.0])},
            measurements={"biomass": (np.array([0.0, 3.0, 10.0]), np.array([1.0, 2.0, 3.0]))},
        )


def test_finite_scoreable_computes_metrics():
    out = _evaluate_predictions_for_process(
        pred_t=np.array([0.0, 5.0, 10.0]),
        pred_columns={"c_biomass": np.array([1.0, 2.0, 3.0])},
        measurements={"biomass": (np.array([0.0, 5.0, 10.0]), np.array([1.0, 2.0, 3.0]))},
    )
    assert out["biomass"]["n_measured"] == 3
    assert out["biomass"]["nmae"] == pytest.approx(0.0)
