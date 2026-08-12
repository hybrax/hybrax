"""Regression test: panels for pseudobatch-transformed components are coherent.

After ``build_pseudobatch_transform`` is registered on a process and the
dataset is round-tripped through JSON, ``inspect.plot_process`` must show
scatter points and a curve that live in the same real-concentration space.

Earlier the ``examples/00_combined/04_spline_serialization`` script
overwrote ``comp.concentration`` with the c* (pseudobatch-transformed)
``TimeSeries``. After reload, ``_collect_process_panels`` then produced a
panel whose scatter ``(x, y)`` was in c* space while the curve resolved to
real concentration via ``build_backtransform_spline`` — visually the points
and the line ran on different scales.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bp_format import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    FeedMedium,
    FeedMediumComponent,
    Inflow,
    ReactorMedium,
    ReactorMediumComponent,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)
from bp_format.inspect import _collect_process_panels
from bp_format.serialization import load_process_collection, save_process_collection
from bp_format.splines import build_backtransform_spline, build_pseudobatch_transform


def _ts(times, values):
    return TimeSeries(
        times=jnp.array(times, dtype=float),
        values=jnp.array(values, dtype=float),
    )


def _make_process_with_continuous_feed():
    """Process with continuous feed so c* and real concentration diverge.

    Glucose is consumed (real concentration falls) while feed dilutes the
    reactor (ADF rises), so c* = c·ADF − feed_corr departs sharply from
    the real measurements after a few hours.
    """
    real_times = [0.0, 2.0, 4.0, 6.0]
    real_vals = [10.0, 8.0, 6.0, 4.0]
    feed_medium = FeedMedium(
        name="feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=StaticVariable(value=0.0),
                is_controlled=True,
            ),
        },
    )
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=_ts(real_times, real_vals),
            ),
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=_ts(real_times, real_vals),
            ),
        },
    )
    vol = Volume(
        initial_volume=1.0,
        unit="L",
        volume_changes={
            "feed": Inflow(
                name="feed",
                unit="L",
                is_controlled=True,
                is_continuous=True,
                feed_medium=feed_medium,
                values=_ts(
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                ),
            ),
        },
    )
    process = BioProcess(
        metadata=BioProcessMetadata(name="p", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=6.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )
    return process, real_times, real_vals


def test_plot_process_panel_scatter_matches_backtransform_after_roundtrip(tmp_path):
    process, real_times, real_vals = _make_process_with_continuous_feed()
    transform = build_pseudobatch_transform(process, ["glucose"])
    process.pseudobatch_transform = transform

    cstar_vals = np.asarray(
        process.reactor_medium.components["glucose"].c_star_concentration.values
    )
    assert not np.allclose(cstar_vals, real_vals, rtol=1e-2), (
        "Test fixture invalid: c* must differ from real measurements so that "
        "the regression check actually exercises the bug. Got c*="
        f"{cstar_vals.tolist()} vs real={list(real_vals)}."
    )

    collection = BioProcessCollection(
        case_id="cs",
        organism="test",
        citation="n/a",
        processes={"p": process},
    )
    out = tmp_path / "data.json"
    save_process_collection(collection, str(out))
    loaded_proc = load_process_collection(str(out)).processes["p"]

    panels = _collect_process_panels(loaded_proc)
    glucose = next(p for p in panels if p["title"].startswith("glucose"))

    assert glucose.get("series_type") == "backtransform"
    assert glucose.get("process") is loaded_proc
    assert glucose.get("species_name") == "glucose"

    np.testing.assert_allclose(np.asarray(glucose["x"]), np.asarray(real_times))

    np.testing.assert_allclose(
        np.asarray(glucose["y"]), np.asarray(real_vals), rtol=5e-3, atol=5e-3
    )

    bt = build_backtransform_spline(loaded_proc, "glucose")
    curve_at_meas = np.asarray(jax.vmap(bt)(jnp.asarray(glucose["x"], dtype=float)))
    np.testing.assert_allclose(
        np.asarray(glucose["y"]), curve_at_meas, rtol=5e-3, atol=5e-3
    )
