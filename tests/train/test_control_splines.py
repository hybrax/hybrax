"""Spline-aware controls: bp-train consumes spline-backed ``TimeSeries`` controls
(instead of rejecting them) and evaluates them via a numpy PPoly; raw-sample
controls keep the linear path. Plus the prepare control-diagnostic renderer.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
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
    # Before the change this raised "spline-backed TimeSeries controls are not supported".
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
