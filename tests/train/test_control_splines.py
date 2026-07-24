"""Direct JAX spline controls, dense fallbacks, and prepare diagnostics."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bp_format.dataclasses import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    ProcessVariable,
    ReactorMedium,
    ReactorMediumComponent,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)
from bp_format.splines import fit_timeseries_spline

from bp_train.controls import select_control_sources
from bp_train.controls_store import ControlsStore


def _noisy_ph() -> TimeSeries:
    t = np.linspace(0.0, 1.0, 60)
    v = 7.0 + 0.05 * np.sin(20.0 * t) + 0.02 * np.cos(33.0 * t)
    return TimeSeries(times=jnp.asarray(t), values=jnp.asarray(v))


def _process_with_ph(ph_values) -> BioProcess:
    return BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=1.0, time_reference="start"),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes={}),
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass", unit="g/L", concentration=StaticVariable(0.1)
                )
            },
        ),
        process_variables={
            "pH": ProcessVariable(
                name="pH", unit="-", is_controlled=True, values=ph_values
            )
        },
    )


def test_spline_control_is_consumed_not_rejected():
    fitted = fit_timeseries_spline(_noisy_ph(), smoothing_s=0.5)
    assert fitted.breaks is not None
    # This previously raised because spline-backed controls were unsupported.
    bundle = select_control_sources(_process_with_ph(fitted))
    src = bundle.sources_by_name["pH"]
    assert src.metadata.get("source") == "spline"
    # Its numpy evaluator matches the bp-format PPoly (to float32).
    tq = np.linspace(0.05, 0.95, 25)
    np.testing.assert_allclose(
        np.asarray(src.evaluator(tq)),
        np.asarray(fitted.evaluate_many(jnp.asarray(tq))),
        atol=1e-4,
    )


def test_raw_control_keeps_linear_path():
    src = select_control_sources(_process_with_ph(_noisy_ph())).sources_by_name["pH"]
    assert src.metadata.get("source") == "timeseries"


def test_spline_control_builds_store_and_evaluates():
    fitted = fit_timeseries_spline(_noisy_ph(), smoothing_s=0.5)
    store = ControlsStore.from_collection(
        BioProcessCollection(metadata={}, processes={"p1": _process_with_ph(fitted)})
    )
    value = float(
        np.asarray(store.get_controls("p1").eval_controlled_PVs(0.5, None))[0]
    )
    assert 6.8 < value < 7.2  # smoothed pH near 7


def _global_cubic(breaks, *, scale=1.0, side="right") -> TimeSeries:
    breaks = np.asarray(breaks, dtype=float)
    x = breaks[:-1]
    coeffs = scale * np.column_stack(
        [
            1.0 + 2.0 * x + 3.0 * x**2 + 4.0 * x**3,
            2.0 + 6.0 * x + 12.0 * x**2,
            3.0 + 12.0 * x,
            np.full_like(x, 4.0),
        ]
    )
    return TimeSeries(
        times=breaks[[0, -1]],
        values=scale
        * np.asarray(
            [
                1.0 + 2.0 * breaks[0] + 3.0 * breaks[0] ** 2 + 4.0 * breaks[0] ** 3,
                1.0 + 2.0 * breaks[-1] + 3.0 * breaks[-1] ** 2 + 4.0 * breaks[-1] ** 3,
            ]
        ),
        breaks=breaks,
        coeffs=coeffs,
        segment_start_piece_idx=[0],
        continuity_side=side,
    )


def _process_with_controls(name, controls) -> BioProcess:
    process = _process_with_ph(next(iter(controls.values())).values)
    process.metadata.name = name
    process.process_variables = controls
    return process


def test_direct_splines_rebase_different_grids_and_match_bp_format():
    first = _global_cubic([0.0, 0.4, 1.0])
    second = _global_cubic([0.0, 0.7, 1.0], scale=2.0)
    process = _process_with_controls(
        "p1",
        {
            "a": ProcessVariable("a", "-", True, first),
            "b": ProcessVariable("b", "-", True, second),
        },
    )
    store = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": process})
    )
    controls = store.get_controls("p1")
    ts = jnp.asarray([-0.2, 0.0, 0.4, 0.7, 1.0, 1.2])

    assert controls.spline_indices == (0, 1)
    assert controls.fallback_indices == ()
    finite_breaks = np.asarray(controls.spline_breaks)
    np.testing.assert_array_equal(
        finite_breaks[np.isfinite(finite_breaks)], [0.0, 0.4, 0.7, 1.0]
    )
    expected_values = jnp.column_stack(
        [first.evaluate_many(ts), second.evaluate_many(ts)]
    )
    np.testing.assert_allclose(controls.eval_controlled_PVs(ts, None), expected_values)
    np.testing.assert_allclose(
        jax.jit(lambda query: controls.eval_controlled_PVs(query, None))(ts),
        expected_values,
    )
    np.testing.assert_allclose(
        controls._eval_derivatives(ts),
        jnp.column_stack(
            [first.deriv().evaluate_many(ts), second.deriv().evaluate_many(ts)]
        ),
    )
    assert store.control_values.shape[-1] == 0


def test_mixed_direct_raw_static_controls_preserve_order_and_batch_rows():
    processes = {}
    for index, name in enumerate(("p1", "p2"), start=1):
        breaks = [0.0, 1.0] if index == 1 else [0.0, 0.5, 1.0]
        spline = _global_cubic(breaks, scale=index)
        processes[name] = _process_with_controls(
            name,
            {
                "a_spline": ProcessVariable("a_spline", "-", True, spline),
                "b_raw": ProcessVariable(
                    "b_raw",
                    "-",
                    True,
                    TimeSeries(times=[0.0, 1.0], values=[index, index + 1.0]),
                ),
                "c_static": ProcessVariable(
                    "c_static", "-", True, StaticVariable(10.0 + index)
                ),
            },
        )
    store = ControlsStore.from_collection(BioProcessCollection(processes=processes))
    controls = store.get_controls("p2")

    assert controls.spline_indices == (0,)
    assert controls.fallback_indices == (1, 2)
    assert store.spline_coeffs.shape[2] == 1
    assert store.control_values.shape[2] == 2
    np.testing.assert_allclose(
        controls.eval_controlled_PVs(jnp.asarray([0.5]), None),
        [[float(_global_cubic([0.0, 0.5, 1.0], scale=2).evaluate(0.5)), 2.5, 12.0]],
    )

    batch = store.gather_batch(jnp.asarray([1, 0, 1]))
    for row, process_index in enumerate((1, 0, 1)):
        np.testing.assert_allclose(
            batch.eval_controlled_PVs(row, jnp.asarray([0.3, 0.8, 1.2]), None),
            store.get_controls(process_index).eval_controlled_PVs(
                jnp.asarray([0.3, 0.8, 1.2]), None
            ),
        )
    np.testing.assert_allclose(
        jax.jit(lambda row, query: batch.eval_controlled_PVs(row, query, None))(
            jnp.asarray(1), jnp.asarray([0.3, 0.8, 1.2])
        ),
        store.get_controls(0).eval_controlled_PVs(jnp.asarray([0.3, 0.8, 1.2]), None),
    )
    p1_spline = processes["p1"].process_variables["a_spline"].values
    assert batch.eval_controlled_PVs(1, 1.2, None)[0] == pytest.approx(
        p1_spline.evaluate(1.2)
    )


def test_control_is_direct_only_when_spline_backed_in_every_process():
    p1 = _process_with_controls(
        "p1", {"u": ProcessVariable("u", "-", True, _global_cubic([0.0, 1.0]))}
    )
    p2 = _process_with_controls(
        "p2",
        {
            "u": ProcessVariable(
                "u", "-", True, TimeSeries(times=[0.0, 1.0], values=[2.0, 3.0])
            )
        },
    )
    store = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": p1, "p2": p2})
    )

    assert store.spline_indices == ()
    assert store.fallback_indices == (0,)
    assert store.spline_coeffs.shape == (2, 0, 0, 4)
    assert store.control_values.shape[-1] == 1


def test_no_spline_store_has_zero_width_direct_payload():
    store = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": _process_with_ph(_noisy_ph())})
    )

    assert store.spline_breaks.shape == (1, 0)
    assert store.spline_coeffs.shape == (1, 0, 0, 4)
    assert store.control_values.shape[-1] == 1


def test_direct_spline_continuity_side_and_extrapolation():
    for side, expected_at_knot in (("right", (10.0, 2.0)), ("left", (0.5, 1.0))):
        series = TimeSeries(
            breaks=[0.0, 0.5, 1.0],
            coeffs=[[0.0, 1.0, 0.0, 0.0], [10.0, 2.0, 0.0, 0.0]],
            segment_start_piece_idx=[0],
            continuity_side=side,
        )
        process = _process_with_controls(
            "p1", {"u": ProcessVariable("u", "-", True, series)}
        )
        controls = ControlsStore.from_collection(
            BioProcessCollection(processes={"p1": process})
        ).get_controls("p1")

        values = controls.eval_controlled_PVs(jnp.asarray([-0.5, 0.5, 1.5]), None)
        rates = controls._eval_derivatives(jnp.asarray([-0.5, 0.5, 1.5]))
        np.testing.assert_allclose(values[:, 0], [-0.5, expected_at_knot[0], 12.0])
        np.testing.assert_allclose(rates[:, 0], [1.0, expected_at_knot[1], 2.0])


def test_direct_spline_ignores_dense_refinement_config():
    process = _process_with_controls(
        "p1", {"u": ProcessVariable("u", "-", True, _global_cubic([0.0, 1.0]))}
    )
    collection = BioProcessCollection(processes={"p1": process})
    coarse = ControlsStore.from_collection(collection)
    collection.metadata = {
        "bp-train": {
            "runtime_controls_config": {
                "initial_grid_points": 100,
                "max_rel_error": 1e-12,
                "max_refinement_rounds": 20,
            }
        }
    }
    refined = ControlsStore.from_collection(collection)

    np.testing.assert_array_equal(coarse.spline_breaks, refined.spline_breaks)
    np.testing.assert_array_equal(coarse.spline_coeffs, refined.spline_coeffs)
    assert coarse.control_values.shape[-1] == refined.control_values.shape[-1] == 0


def test_render_control_diagnostics_writes_png(tmp_path: Path):
    from bp_train.postprocessing import (
        ControlDiagnostic,
        ProcessControlDiagnostics,
        render_control_diagnostics,
    )

    t = np.linspace(0.0, 1.0, 50)
    diag = ProcessControlDiagnostics(
        process_name="p1",
        time_unit="h",
        controls=(
            ControlDiagnostic(
                name="pH",
                unit="-",
                raw_times=t,
                raw_values=7.0 + 0.05 * np.sin(20.0 * t),
                curve_t=t,
                curve_values=np.full(50, 7.0),
                grid_t=t[::5],
                is_spline=True,
                max_rel_dev=0.01,
            ),
        ),
    )
    render_control_diagnostics(diag, tmp_path)
    png = tmp_path / "p1_controls.png"
    assert png.is_file()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
