"""Regression tests for end-to-end mechanistic integration accuracy.

These tests guard against silent regressions in the mechanistic q-inversion
and forward-integration pipeline. Each test:

1. Loads a frozen single-process dataset JSON from ``tests/fixtures/``.
2. Builds pseudobatch ReactorMediumComponent interpolators (these are what
   ``build_state_splines`` consumes for the backtransform spline).
3. Builds the RHS ODE + estimates ``q(t)`` on a dense event-scaled grid.
4. Fits a cubic q-spline and forward-integrates via ``integrate_process``.
5. Asserts the per-species nRMSE vs the original measurements is below the
   bound we currently achieve.

These tests caught the ``BacktransformSpline.derivative()`` quotient-rule
regression (nRMSE 0.462 on kittler_2022 after the ADF-semantics fix). If the
mechanistic pipeline silently loses accuracy again, the failure message
surfaces the observed nRMSE so the cause is diagnosable from the test output
alone.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import bp_format
import bp_format.mechanistic as bpm
from bp_format.splines import (
    build_pseudobatch_inputs,
    build_splines,
    make_interpax_spline,
    to_interpolator,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _wrap_q_as_rates(mb, q_func):
    def rates_func(t, state, controls):
        del state, controls
        return q_func(t), jnp.zeros(mb.r_size)

    return rates_func


def _attach_interpolators(process):
    """Fit pseudobatch interpolators for every reactor-medium component whose
    concentration is a TimeSeries, and attach them in-place. Mirrors step 2a
    of ``examples/00_combined/04_spline_serialization/01_serialize_splines.py``.
    """
    for comp_name, comp in process.reactor_medium.components.items():
        if not hasattr(comp.concentration, "times"):
            continue
        inputs = build_pseudobatch_inputs(process, comp_name)
        spl = build_splines(inputs, process, comp_name)
        comp.interpolator = to_interpolator(inputs, spl, comp_name)


def _run_mechanistic_pipeline(fixture_dir: Path) -> dict:
    """Load fixture, build splines + q-spline, forward-integrate, compute
    per-species nRMSE against the stored measurements.

    Returns a metrics dict keyed by species name with per-species nRMSE plus
    the combined ``rms`` summary.
    """
    dataset = bp_format.serialization.load_dataset_json(
        str(fixture_dir / "data.json")
    )
    case_study = next(iter(dataset.case_studies.values()))
    process = next(iter(case_study.processes.values()))
    _attach_interpolators(process)

    ctrl = bpm.get_control_splines(process)
    mb = bpm.get_rhs_ode(process)
    state_splines = bpm.build_state_splines(process, mb)

    events = bpm.extract_discrete_events(process, mb)
    n_events = len(events)

    t_start = float(process.time_axis.start)
    t_end = float(process.time_axis.end)
    n_q_grid = max(500, 20 * n_events)
    t_eval_q = np.linspace(t_start, t_end, n_q_grid)
    q_est = bpm.estimate_specific_rates(process, ctrl, mb, state_splines, t_eval_q)
    q_spline = make_interpax_spline(t_eval_q, q_est)

    def q_func(t, _s=q_spline):
        return _s(t)

    rates_func = _wrap_q_as_rates(mb, q_func)

    t_eval_int = np.linspace(t_start, t_end, 200)
    result = bpm.integrate_process(
        process, ctrl, mb, rates_func, t_eval_int, state_splines=state_splines
    )

    metrics: dict = {"species": {}}
    rms_acc = []
    for i, sp_name in enumerate(mb.reactor_component_state_names):
        ts = process.reactor_medium.components[sp_name].concentration
        t_m = np.asarray(ts.times, dtype=float)
        v_m = np.asarray(ts.values, dtype=float)
        v_p = np.interp(t_m, np.asarray(result["t"]), np.asarray(result["c"][:, i]))
        rmse = float(np.sqrt(np.mean((v_p - v_m) ** 2)))
        rng = float(v_m.max() - v_m.min())
        nrmse = rmse / rng if rng > 1e-12 else 0.0
        metrics["species"][sp_name] = {
            "rmse": rmse,
            "range": rng,
            "nrmse": nrmse,
        }
        rms_acc.append(nrmse)
    metrics["rms"] = float(np.sqrt(np.mean(np.array(rms_acc) ** 2)))
    metrics["n_events"] = n_events
    return metrics


def _assert_per_species(metrics: dict, *, tol: float) -> None:
    """Per-species nRMSE must stay below ``tol``."""
    worst_sp = max(metrics["species"], key=lambda s: metrics["species"][s]["nrmse"])
    worst = metrics["species"][worst_sp]["nrmse"]
    assert worst < tol, (
        f"Mechanistic integration regression: species {worst_sp!r} nRMSE "
        f"{worst:.6f} exceeds {tol:.6f}. Full per-species metrics: "
        f"{metrics['species']}"
    )


# --------------------------------------------------------------------------
# 10_martens_2025_f — 11 discrete events, 9 measurements, 8 species.
# --------------------------------------------------------------------------


def test_martens_2025_f_mechanistic_integration_accuracy():
    metrics = _run_mechanistic_pipeline(FIXTURES / "martens_2025_f_single")
    # Current observed: worst species nRMSE < 0.001 across all 8 species.
    # Bound kept loose enough to absorb floating-point / spline-resampling
    # noise, tight enough to catch any systematic mechanistic regression
    # (which historically pushed nRMSE to ~0.5).
    _assert_per_species(metrics, tol=0.01)


# --------------------------------------------------------------------------
# 12_martens_expanded — 12 discrete events, 9 measurements, 10 species.
# --------------------------------------------------------------------------


def test_martens_expanded_mechanistic_integration_accuracy():
    metrics = _run_mechanistic_pipeline(FIXTURES / "martens_expanded_single")
    # Current observed: worst species nRMSE ≈ 0.0006 (glutamine).
    _assert_per_species(metrics, tol=0.01)


# --------------------------------------------------------------------------
# Sanity check on the combined RMS summary (catches cases where only one
# species drifts but the per-species bound is already quite tight).
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_dir,tol_rms",
    [
        ("martens_2025_f_single", 5e-3),
        ("martens_expanded_single", 5e-3),
    ],
)
def test_mechanistic_rms_summary(fixture_dir, tol_rms):
    metrics = _run_mechanistic_pipeline(FIXTURES / fixture_dir)
    assert metrics["rms"] < tol_rms, (
        f"{fixture_dir}: combined RMS nRMSE {metrics['rms']:.6f} exceeds "
        f"{tol_rms:.6f}. Full metrics: {metrics}"
    )
