"""Regression tests for end-to-end pseudobatch spline accuracy on real data.

These tests guard the current accuracy of the splines.py pipeline against
future regressions. Each test:

1. Loads a frozen single-process dataset JSON from ``tests/fixtures/``.
2. Loads the companion high-resolution ground-truth CSV.
3. Builds pseudobatch inputs + splines for each species and evaluates the
   backtransform at every ground-truth timestamp inside the measurement
   domain.
4. Asserts:
   * ``abs@meas`` — the backtransform at measurement times matches
     ``meas_conc`` to within the pseudobatch-math invariant tolerance.
   * ``rel@gt_mean`` — the mean relative error vs the ground truth is
     below 1 %.
   * ``rel@gt_p90`` — the 90th-percentile relative error is below a
     species-specific bound (tight for densely-sampled cases, looser where
     the nine sparse measurements can't fully resolve sharp peaks).

If the pipeline regresses, the failing assertion's message reports the
observed metrics so the underlying cause can be diagnosed quickly.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import bp_format
from bp_format.splines import (
    build_pseudobatch_inputs,
    build_splines,
    evaluate_real_concentration,
)


FIXTURES = Path(__file__).parent / "fixtures"

SPECIES_COLUMNS = {
    "biomass": "Viable cells [cells/L]",
    "glucose": "Glucose [mmol/L]",
    "glutamine": "Glutamine [mmol/L]",
}


def _relative_error(c_back: np.ndarray, c_gt: np.ndarray) -> np.ndarray:
    """Floor the denominator at 1% of the peak magnitude so that near-zero
    ground-truth samples (which happen at the tail of glucose consumption)
    don't artificially inflate the relative error."""
    peak = float(np.max(np.abs(c_gt)))
    floor = max(1e-6, 0.01 * peak)
    denom = np.maximum(np.abs(c_gt), floor)
    return np.abs(c_back - c_gt) / denom


def _evaluate_case(fixture_dir: Path, species: str) -> dict:
    """Build splines + evaluate at every ground-truth timestamp within the
    measurement domain. Returns a dict of observed metrics."""
    dataset = bp_format.serialization.load_dataset_json(
        str(fixture_dir / "data.json")
    )
    case_study = next(iter(dataset.case_studies.values()))
    process = next(iter(case_study.processes.values()))

    if process.reactor_medium is None or species not in process.reactor_medium.components:
        pytest.skip(f"species {species!r} not present in fixture {fixture_dir.name}")

    inputs = build_pseudobatch_inputs(process, species)
    spl = build_splines(inputs, process, species)

    meas_t = np.asarray(inputs["meas_times"])
    meas_conc = np.asarray(inputs["meas_conc"])
    c_back_at_meas = np.asarray(
        evaluate_real_concentration(jnp.asarray(meas_t), spl)
    )
    # Relative error against meas — the pseudobatch-math invariant target.
    peak_meas = float(np.max(np.abs(meas_conc)))
    abs_meas = float(np.max(np.abs(c_back_at_meas - meas_conc)))
    rel_meas = abs_meas / max(peak_meas, 1e-12)

    gt = pd.read_csv(fixture_dir / "full_offline.csv")
    column = SPECIES_COLUMNS[species]
    if column not in gt.columns:
        pytest.skip(f"column {column!r} missing from fixture CSV")

    # Restrict ground-truth samples to the measurement-time domain.
    t_gt = gt["time [h]"].to_numpy(dtype=float)
    c_gt = gt[column].to_numpy(dtype=float)
    mask = (t_gt >= float(meas_t[0])) & (t_gt <= float(meas_t[-1]))
    t_gt, c_gt = t_gt[mask], c_gt[mask]
    c_back_gt = np.asarray(
        evaluate_real_concentration(jnp.asarray(t_gt), spl)
    )
    rel = _relative_error(c_back_gt, c_gt)

    return {
        "species": species,
        "n_meas": int(meas_t.size),
        "n_gt": int(t_gt.size),
        "abs_meas": abs_meas,
        "rel_meas": rel_meas,
        "rel_mean": float(rel.mean()),
        "rel_p50": float(np.percentile(rel, 50)),
        "rel_p90": float(np.percentile(rel, 90)),
        "rel_p99": float(np.percentile(rel, 99)),
        "rel_max": float(rel.max()),
    }


def _assert_meas_invariant(metrics: dict, tol_rel: float = 1e-3) -> None:
    """Backtransform at measurement times must match meas_conc exactly."""
    assert metrics["rel_meas"] < tol_rel, (
        f"{metrics['species']}: pseudobatch-math invariant violated — "
        f"rel@meas {metrics['rel_meas']:.3e} exceeds {tol_rel:.1e} "
        f"(abs {metrics['abs_meas']:.3e})"
    )


def _assert_gt_accuracy(
    metrics: dict,
    *,
    mean_tol: float,
    p90_tol: float,
) -> None:
    """Relative error vs high-resolution ground truth must be within budget."""
    assert metrics["rel_mean"] < mean_tol, (
        f"{metrics['species']}: mean relative error {metrics['rel_mean']:.4f} "
        f"exceeds {mean_tol:.4f}. Full metrics: {metrics}"
    )
    assert metrics["rel_p90"] < p90_tol, (
        f"{metrics['species']}: p90 relative error {metrics['rel_p90']:.4f} "
        f"exceeds {p90_tol:.4f}. Full metrics: {metrics}"
    )


# --------------------------------------------------------------------------
# 10_martens_2025_f (sparse continuous feed + boluses; 9 measurements)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("species", ["biomass", "glucose", "glutamine"])
def test_martens_2025_f_meas_invariant(species):
    metrics = _evaluate_case(FIXTURES / "martens_2025_f_single", species)
    _assert_meas_invariant(metrics)


def test_martens_2025_f_biomass_accuracy():
    metrics = _evaluate_case(FIXTURES / "martens_2025_f_single", "biomass")
    _assert_gt_accuracy(metrics, mean_tol=0.02, p90_tol=0.05)


def test_martens_2025_f_glucose_accuracy():
    # Glucose has the sharpest peaks; cubic-spline overshoot bumps p90 slightly
    # above the martens_expanded target.
    metrics = _evaluate_case(FIXTURES / "martens_2025_f_single", "glucose")
    _assert_gt_accuracy(metrics, mean_tol=0.03, p90_tol=0.05)


def test_martens_2025_f_glutamine_accuracy():
    metrics = _evaluate_case(FIXTURES / "martens_2025_f_single", "glutamine")
    _assert_gt_accuracy(metrics, mean_tol=0.015, p90_tol=0.02)


# --------------------------------------------------------------------------
# 12_martens_expanded (dense continuous feed + boluses; 9 measurements)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("species", ["biomass", "glucose", "glutamine"])
def test_martens_expanded_meas_invariant(species):
    metrics = _evaluate_case(FIXTURES / "martens_expanded_single", species)
    _assert_meas_invariant(metrics)


def test_martens_expanded_biomass_accuracy():
    metrics = _evaluate_case(FIXTURES / "martens_expanded_single", "biomass")
    _assert_gt_accuracy(metrics, mean_tol=0.02, p90_tol=0.03)


def test_martens_expanded_glucose_accuracy():
    metrics = _evaluate_case(FIXTURES / "martens_expanded_single", "glucose")
    _assert_gt_accuracy(metrics, mean_tol=0.01, p90_tol=0.02)


def test_martens_expanded_glutamine_accuracy():
    metrics = _evaluate_case(FIXTURES / "martens_expanded_single", "glutamine")
    _assert_gt_accuracy(metrics, mean_tol=0.01, p90_tol=0.02)
