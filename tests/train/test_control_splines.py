"""Direct JAX spline controls, linear controls, and prepare diagnostics."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hybrax.format.dataclasses import (
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
from hybrax.format.splines import fit_timeseries_spline

from hybrax.train.controls import select_control_sources
from hybrax.train.controls_store import ControlsStore


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


def test_continuous_control_rejects_single_point_timeseries():
    process = _process_with_ph(TimeSeries(times=[0.0], values=[7.0]))

    with pytest.raises(ValueError, match="must contain at least two points"):
        select_control_sources(process)


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
    assert controls.linear_indices == ()
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
    assert controls.linear_indices == (1, 2)
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


def test_control_rejects_mixed_spline_availability_across_processes():
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

    with pytest.raises(
        ValueError,
        match="'u' must be spline-backed in every process or no process",
    ) as exc_info:
        ControlsStore.from_collection(
            BioProcessCollection(processes={"p1": p1, "p2": p2})
        )

    assert "spline-backed in ['p1'], but not ['p2']" in str(exc_info.value)


def test_control_rejects_mixed_raw_and_spline_continuity_sides():
    process = _process_with_controls(
        "p1",
        {
            "raw": ProcessVariable(
                "raw",
                "-",
                True,
                TimeSeries(
                    times=[0.0, 1.0],
                    values=[0.0, 1.0],
                    continuity_side="left",
                ),
            ),
            "spline": ProcessVariable(
                "spline",
                "-",
                True,
                _global_cubic([0.0, 1.0], side="right"),
            ),
        },
    )

    with pytest.raises(
        ValueError,
        match="all time-varying controls must use one continuity side",
    ):
        ControlsStore.from_collection(BioProcessCollection(processes={"p1": process}))


def test_control_rejects_mixed_spline_continuity_across_processes():
    processes = {
        name: _process_with_controls(
            name,
            {
                "u": ProcessVariable(
                    "u", "-", True, _global_cubic([0.0, 1.0], side=side)
                )
            },
        )
        for name, side in (("p1", "left"), ("p2", "right"))
    }

    with pytest.raises(
        ValueError,
        match="all time-varying controls must use one continuity side",
    ) as exc_info:
        ControlsStore.from_collection(BioProcessCollection(processes=processes))

    assert "'u': {'left': 'p1', 'right': 'p2'}" in str(exc_info.value)


@pytest.mark.parametrize(
    ("side", "expected_rates"),
    [("left", [1.0, 1.0, 2.0]), ("right", [1.0, 2.0, 2.0])],
)
def test_raw_control_values_and_rates_at_knot_follow_continuity_side(
    side, expected_rates
):
    process = _process_with_ph(
        TimeSeries(
            times=[0.0, 1.0, 2.0],
            values=[0.0, 1.0, 3.0],
            continuity_side=side,
        )
    )
    process.time_axis.end = 2.0
    controls = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": process})
    ).get_controls("p1")
    ts = jnp.asarray([0.5, 1.0, 1.5])

    np.testing.assert_allclose(
        controls.eval_controlled_PVs(ts, None)[:, 0], [0.5, 1.0, 2.0]
    )
    np.testing.assert_allclose(controls._eval_derivatives(ts)[:, 0], expected_rates)


@pytest.mark.parametrize("side", ["left", "right"])
def test_raw_control_rate_uses_in_domain_interval_at_support_endpoints(side):
    process = _process_with_ph(
        TimeSeries(
            times=[2.0, 5.0, 8.0],
            values=[0.0, 3.0, 9.0],
            continuity_side=side,
        )
    )
    process.time_axis.end = 10.0
    controls = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": process})
    ).get_controls("p1")

    controls.validate_support(2.0, 8.0)
    np.testing.assert_allclose(
        controls._eval_derivatives(jnp.asarray([2.0, 8.0]))[:, 0],
        [1.0, 2.0],
    )


def test_linear_control_padding_does_not_change_active_process():
    short = _process_with_controls(
        "short",
        {
            "u": ProcessVariable(
                "u",
                "-",
                True,
                TimeSeries(times=[0.0, 1.0, 2.0], values=[0.0, 1.0, 3.0]),
            )
        },
    )
    short.time_axis.end = 2.0
    long = _process_with_controls(
        "long",
        {
            "u": ProcessVariable(
                "u",
                "-",
                True,
                TimeSeries(
                    times=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
                    values=[0.0, 0.25, 0.5, 1.0, 2.0, 3.0],
                ),
            )
        },
    )
    long.time_axis.end = 2.0
    alone = ControlsStore.from_collection(
        BioProcessCollection(processes={"short": short})
    ).get_controls("short")
    padded_store = ControlsStore.from_collection(
        BioProcessCollection(processes={"short": short, "long": long})
    )
    padded = padded_store.gather_batch(jnp.asarray([0]))
    ts = jnp.asarray([0.5, 1.0, 1.5, 2.0])

    np.testing.assert_allclose(
        padded.eval_controlled_PVs(0, ts, None),
        alone.eval_controlled_PVs(ts, None),
    )
    np.testing.assert_allclose(
        padded._eval_derivatives(0, ts),
        alone._eval_derivatives(ts),
    )


def test_static_control_is_unbounded_and_has_zero_rate():
    process = _process_with_ph(StaticVariable(7.0))
    controls = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": process})
    ).get_controls("p1")
    ts = jnp.asarray([-1.0e6, 0.5, 1.0e6])

    controls.validate_support(-1.0e6, 1.0e6)
    assert controls.control_supports == {"pH": (-np.inf, np.inf)}
    np.testing.assert_allclose(controls.eval_controlled_PVs(ts, None)[:, 0], 7.0)
    np.testing.assert_allclose(controls._eval_derivatives(ts)[:, 0], 0.0)


def test_control_support_validation_allows_float32_endpoint_roundoff():
    rounded_end = float(np.float32(1000.1))
    process = _process_with_ph(
        TimeSeries(
            times=np.asarray([1000.0, rounded_end], dtype=np.float64),
            values=[6.9, 7.1],
        )
    )
    controls = ControlsStore.from_collection(
        BioProcessCollection(processes={"p1": process})
    ).get_controls("p1")

    controls.validate_support(1000.0, 1000.1)
    with pytest.raises(
        ValueError,
        match=r"representation='raw'.*violated_side='left'",
    ):
        controls.validate_support(999.9, 1000.1)
    with pytest.raises(
        ValueError,
        match=r"representation='raw'.*violated_side='right'",
    ):
        controls.validate_support(1000.0, 1000.11)


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
        with pytest.raises(
            ValueError,
            match=r"representation='spline'.*violated_side='right'",
        ):
            controls.validate_support(0.0, 1.1)


def test_render_control_diagnostics_writes_png(tmp_path: Path):
    from hybrax.train.postprocessing import (
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
