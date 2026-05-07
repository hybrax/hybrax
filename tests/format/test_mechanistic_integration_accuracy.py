"""Regression tests for end-to-end mechanistic integration accuracy.

These tests guard against silent regressions in the mechanistic q-inversion
and forward-integration pipeline. Each test:

1. Loads a frozen single-process dataset JSON from ``tests/fixtures/``.
2. Builds process-level pseudobatch bundles and assigns c* ``TimeSeries``
   carriers to reactor components.
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
    build_pseudobatch_transform,
    make_interpax_spline,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _wrap_q_as_rates(mb, q_func):
    def rates_func(t, state, controls):
        del state, controls
        return q_func(t), jnp.zeros(mb.r_size)

    return rates_func


def _attach_pseudobatch_series(process):
    """Fit pseudobatch TimeSeries carriers for each measured reactor component."""
    species_names = [
        comp_name
        for comp_name, comp in process.reactor_medium.components.items()
        if hasattr(comp.concentration, "times")
    ]
    transform = build_pseudobatch_transform(process, species_names)
    process.pseudobatch_transform = transform
    for comp_name in species_names:
        process.reactor_medium.components[comp_name].concentration = transform.species[
            comp_name
        ].c_star_ts


def _run_mechanistic_pipeline(fixture_dir: Path, *, mutator=None) -> dict:
    """Load fixture, build splines + q-spline, forward-integrate, compute
    per-species nRMSE against the stored measurements.

    Returns a metrics dict keyed by species name with per-species nRMSE plus
    the combined ``rms`` summary.

    ``mutator`` is an optional callable invoked on the loaded ``BioProcess``
    before pseudobatch TimeSeries carriers are attached, used by
    intracellular-variant tests.
    """
    dataset = bp_format.serialization.load_dataset_json(str(fixture_dir / "data.json"))
    case_study = next(iter(dataset.case_studies.values()))
    process = next(iter(case_study.processes.values()))
    if mutator is not None:
        mutator(process)
    observations = {
        sp_name: (
            np.asarray(comp.concentration.times, dtype=float).copy(),
            np.asarray(comp.concentration.values, dtype=float).copy(),
        )
        for sp_name, comp in process.reactor_medium.components.items()
        if hasattr(comp.concentration, "times")
    }
    _attach_pseudobatch_series(process)

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
    result = bpm.integrate_process(process, ctrl, mb, rates_func, t_eval_int)

    metrics: dict = {"species": {}}
    rms_acc = []
    for i, sp_name in enumerate(mb.reactor_component_state_names):
        t_m, v_m = observations[sp_name]
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


# --------------------------------------------------------------------------
# Intracellular variants: re-run the same fixtures after marking the
# `product` component as intracellular. After the Phase 1 fix to the
# auto-generated RHS, the inversion + forward-integration self-consistency
# must still hold (the pipeline infers q from the measurements with the
# corrected formula, then re-integrates with the corrected forward, so the
# round-trip remains exact regardless of whether q[biomass] is active or
# apparent).
# --------------------------------------------------------------------------


def _mark_product_intracellular(process):
    process.reactor_medium.components["product"].is_intracellular = True


@pytest.mark.parametrize(
    "fixture_dir",
    ["martens_2025_f_single", "martens_expanded_single"],
)
def test_mechanistic_integration_with_intracellular_product(fixture_dir):
    metrics = _run_mechanistic_pipeline(
        FIXTURES / fixture_dir, mutator=_mark_product_intracellular
    )
    _assert_per_species(metrics, tol=0.01)


@pytest.mark.parametrize(
    "fixture_dir,tol_rms",
    [
        ("martens_2025_f_single", 5e-3),
        ("martens_expanded_single", 5e-3),
    ],
)
def test_mechanistic_rms_summary_with_intracellular_product(fixture_dir, tol_rms):
    metrics = _run_mechanistic_pipeline(
        FIXTURES / fixture_dir, mutator=_mark_product_intracellular
    )
    assert metrics["rms"] < tol_rms, (
        f"{fixture_dir} (intracellular product): combined RMS nRMSE "
        f"{metrics['rms']:.6f} exceeds {tol_rms:.6f}. Full metrics: {metrics}"
    )


# --------------------------------------------------------------------------
# Dual-path equivalence: build an auto RhsOde and a hand-crafted equivalent
# UserDefinedRhsOde from the same fixture, then assert their dc/dt match
# numerically at randomly sampled (c, rates) inputs. Proves the user-defined
# path is a strict generalization of the auto path on real fixtures.
# --------------------------------------------------------------------------


from bp_format import BiologicalOde, RateDecl
from bp_format.mechanistic import _build_auto_rhs_ode


def _make_equivalent_biological_ode(mb) -> BiologicalOde:
    """Build a `BiologicalOde` block that, layered on top of the same process
    structure, produces dc/dt identical to the auto RhsOde *mb*.

    Convention used:
    - One rate per reactor-component state, named ``q_<state_name>``.
    - ``X_active`` is declared as an algebraic variable (``biomass - sum(intra)``)
      when intracellular components exist, otherwise it equals biomass directly
      and we inline that.
    - Per-state biological RHS is ``q_i * X_active`` for non-biomass states;
      for biomass it additionally absorbs the intracellular accumulation rates
      to mirror the Phase-1 mass-balance correction inside RhsOde.
    - Process-variable states get ``"0"`` (auto path adds only the additive
      r-vector to PV states; with r=0 in this test that matches).
    """
    state_names = list(mb.reactor_component_state_names)
    biomass_idx = mb.biomass_idx
    intra_idxs = list(mb.intracellular_indices)
    intra_names = [state_names[i] for i in intra_idxs]
    biomass_name = state_names[biomass_idx]

    rates = {f"q_{n}": RateDecl() for n in state_names}

    if intra_names:
        algebraic = {"X_active": " - ".join([biomass_name] + intra_names)}
        x_active = "X_active"
    else:
        algebraic = {}
        x_active = biomass_name

    derivatives: dict = {}
    for i, name in enumerate(state_names):
        if i == biomass_idx and intra_names:
            terms = [f"q_{biomass_name} * {x_active}"]
            terms.extend(f"q_{nm} * {x_active}" for nm in intra_names)
            derivatives[name] = " + ".join(terms)
        else:
            derivatives[name] = f"q_{name} * {x_active}"

    for pv_name in mb.process_variable_state_names:
        derivatives[pv_name] = "0"

    return BiologicalOde(algebraic=algebraic, rates=rates, derivatives=derivatives)


def _assert_dual_path_dcdt_equivalence(process, *, n_samples: int = 5, seed: int = 7):
    """Build both RHS modules and assert they produce identical dc/dt at
    randomly sampled (c, rates, u_flow, f_modeled) inputs."""
    mb_auto = _build_auto_rhs_ode(process)
    process.biological_ode = _make_equivalent_biological_ode(mb_auto)
    mb_user = bp_format.mechanistic.get_rhs_ode(process)

    rng = np.random.default_rng(seed)
    for _ in range(n_samples):
        c = rng.uniform(0.1, 5.0, size=mb_auto.c_size).astype(np.float64)
        c[mb_auto.volume_idx] = float(rng.uniform(0.5, 2.0))  # ensure V > 0
        # Keep biomass > sum(intracellular) so X_active > 0
        if mb_auto.intracellular_indices:
            intra_sum = sum(c[i] for i in mb_auto.intracellular_indices)
            c[mb_auto.biomass_idx] = float(intra_sum + rng.uniform(0.5, 2.0))
        rates_auto = rng.uniform(-0.3, 0.3, size=mb_auto.q_size).astype(np.float64)
        u_flow = rng.uniform(0.0, 0.05, size=mb_auto.u_flow_size).astype(np.float64)
        f_modeled = rng.uniform(0.0, 0.05, size=mb_auto.f_modeled_size).astype(
            np.float64
        )
        r = jnp.zeros(mb_auto.r_size)
        ctrl_pv_values = jnp.zeros(mb_user.n_controlled_pv)

        dc_auto = mb_auto(
            jnp.asarray(c),
            jnp.asarray(rates_auto),
            jnp.asarray(u_flow),
            jnp.asarray(f_modeled),
            r,
        )
        dc_user = mb_user(
            jnp.asarray(c),
            jnp.asarray(rates_auto),
            jnp.asarray(u_flow),
            jnp.asarray(f_modeled),
            ctrl_pv_values,
        )
        np.testing.assert_allclose(
            np.asarray(dc_user),
            np.asarray(dc_auto),
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"dual-path dc/dt mismatch on sample c={c}, rates={rates_auto}",
        )


@pytest.mark.parametrize(
    "fixture_dir",
    ["martens_2025_f_single", "martens_expanded_single"],
)
def test_dual_path_dcdt_equivalence(fixture_dir):
    """Auto RhsOde and an equivalent hand-written biological_ode produce the
    same dc/dt on the no-intracellular fixtures."""
    dataset = bp_format.serialization.load_dataset_json(
        str(FIXTURES / fixture_dir / "data.json")
    )
    process = next(iter(next(iter(dataset.case_studies.values())).processes.values()))
    _assert_dual_path_dcdt_equivalence(process)


@pytest.mark.parametrize(
    "fixture_dir",
    ["martens_2025_f_single", "martens_expanded_single"],
)
def test_dual_path_dcdt_equivalence_with_intracellular_product(fixture_dir):
    """Same equivalence check after marking ``product`` as intracellular.
    Exercises the intracellular mass-balance term on the biomass derivative
    in both paths."""
    dataset = bp_format.serialization.load_dataset_json(
        str(FIXTURES / fixture_dir / "data.json")
    )
    process = next(iter(next(iter(dataset.case_studies.values())).processes.values()))
    _mark_product_intracellular(process)
    _assert_dual_path_dcdt_equivalence(process)
