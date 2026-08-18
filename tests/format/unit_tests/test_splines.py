"""
Tests for bp_format.splines: spline fitting, serialization, and evaluation.
"""

import pytest
import jax.numpy as jnp
import numpy as np
import tempfile
from pathlib import Path
from scipy import interpolate

from bp_format import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    TimeAxis,
    TimeSeries,
    StaticVariable,
    ReactorMedium,
    ReactorMediumComponent,
    FeedMedium,
    FeedMediumComponent,
    Inflow,
    Volume,
    ProcessVariable,
    DiscreteEvents,
)
from bp_format.splines import (
    detect_discrete_state_events,
    make_segment_boundaries,
    split_timeseries,
    fit_timeseries_spline,
    make_constant_spline,
    make_cubic_ppoly,
)
from bp_format.serialization import (
    save_process_collection,
    load_process_collection,
)
from bp_format.time_series import PPoly


# ---------------------------------------------------------------------------
# make_cubic_ppoly
# ---------------------------------------------------------------------------


class TestMakeCubicPPoly:
    def test_returns_owned_ppoly(self):
        sp = make_cubic_ppoly(
            np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 4.0, 9.0])
        )
        assert isinstance(sp, PPoly)

    def test_eval_at_knots(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        v = np.array([0.0, 1.0, 4.0, 9.0])
        sp = make_cubic_ppoly(t, v)
        for ti, vi in zip(t, v, strict=True):
            assert float(sp(ti)) == pytest.approx(float(vi), abs=1e-4)

    def test_derivative_of_linear_is_slope(self):
        sp = make_cubic_ppoly(np.array([0.0, 10.0]), np.array([0.0, 5.0]))
        assert float(sp.derivative()(5.0)) == pytest.approx(0.5, rel=1e-4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(t, v):
    return TimeSeries(times=jnp.array(t, dtype=float), values=jnp.array(v, dtype=float))


def _make_feed(name="feed"):
    return FeedMedium(name=name, density=1.0, density_unit="kg/L")


def _make_process_with_discrete():
    """Process with both continuous and discrete volume changes."""
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=_ts([0.0, 5.0, 10.0, 20.0], [0.5, 1.0, 2.0, 4.0]),
            ),
        },
    )
    vol = Volume(
        initial_volume=1.0,
        unit="L",
        volume_changes={
            "continuous_feed": Inflow(
                name="continuous_feed",
                unit="L",
                is_controlled=True,
                is_continuous=True,
                feed_medium=_make_feed("cont"),
                values=_ts([0.0, 5.0, 10.0, 20.0], [0.0, 0.25, 0.5, 1.0]),
            ),
            "bolus": Inflow(
                name="bolus",
                unit="L",
                is_controlled=True,
                is_continuous=False,
                feed_medium=_make_feed("bolus"),
                values=_ts([3.0, 12.0], [0.1, 0.1]),
            ),
        },
    )
    return BioProcess(
        metadata=BioProcessMetadata(name="test_proc", process_type="fed_batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=20.0, time_reference="inoculation"
        ),
        volume=vol,
        reactor_medium=rm,
    )


# ---------------------------------------------------------------------------
# DiscreteEvents
# ---------------------------------------------------------------------------


def test_discrete_events_creation():
    de = DiscreteEvents(times=jnp.array([1.0, 2.0, 5.0]))
    assert de.times.shape == (3,)
    assert de.labels is None


def test_discrete_events_with_labels():
    de = DiscreteEvents(
        times=jnp.array([1.0, 3.0]),
        labels=["bolus_1", "bolus_2"],
    )
    assert len(de.labels) == 2


# ---------------------------------------------------------------------------
# detect_discrete_state_events
# ---------------------------------------------------------------------------


def test_detect_discrete_events():
    proc = _make_process_with_discrete()
    de = detect_discrete_state_events(proc)
    assert de.times.shape[0] == 2
    assert jnp.allclose(de.times, jnp.array([3.0, 12.0]))
    assert de.labels is not None
    assert len(de.labels) == 2


def test_detect_discrete_events_no_discrete():
    """Process with only continuous changes -> empty events."""
    rm = ReactorMedium(name="m", density=1.0, density_unit="kg/L")
    vol = Volume(
        initial_volume=1.0,
        unit="L",
        volume_changes={
            "feed": Inflow(
                name="feed",
                unit="L",
                is_controlled=True,
                is_continuous=True,
                feed_medium=_make_feed(),
                values=_ts([0.0, 10.0], [0.0, 1.0]),
            ),
        },
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="t", process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=10.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )
    de = detect_discrete_state_events(proc)
    assert de.times.shape[0] == 0


# ---------------------------------------------------------------------------
# make_segment_boundaries
# ---------------------------------------------------------------------------


def test_make_segment_boundaries_no_events():
    b = make_segment_boundaries(0.0, 10.0, jnp.zeros(0))
    np.testing.assert_array_equal(b, [0.0, 10.0])


def test_make_segment_boundaries_with_events():
    b = make_segment_boundaries(0.0, 20.0, jnp.array([3.0, 12.0]))
    np.testing.assert_array_equal(b, [0.0, 3.0, 12.0, 20.0])


def test_make_segment_boundaries_events_outside():
    """Events at or beyond boundaries should be excluded."""
    b = make_segment_boundaries(0.0, 10.0, jnp.array([0.0, 10.0, 15.0]))
    np.testing.assert_array_equal(b, [0.0, 10.0])


# ---------------------------------------------------------------------------
# split_timeseries
# ---------------------------------------------------------------------------


def test_split_timeseries_single_segment():
    ts = _ts([0.0, 2.0, 4.0, 6.0], [1.0, 2.0, 3.0, 4.0])
    segments = split_timeseries(ts, np.array([0.0, 6.0]))
    assert len(segments) == 1
    assert segments[0].times.shape[0] == 4


def test_split_timeseries_two_segments():
    ts = _ts([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    segments = split_timeseries(ts, np.array([0.0, 2.5, 5.0]))
    assert len(segments) == 2
    assert segments[0].times.shape[0] == 3
    assert segments[1].times.shape[0] == 3


# ---------------------------------------------------------------------------
# fit_timeseries_spline
# ---------------------------------------------------------------------------


def test_fit_simple_cubic():
    ts = _ts([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 1.0, 4.0, 9.0, 16.0, 25.0])
    rep = fit_timeseries_spline(ts)
    assert isinstance(rep, TimeSeries)
    assert rep.breaks is not None
    assert rep.coeffs is not None
    assert rep.metadata["fit_strategy"] == "smoothing_bspline"
    assert rep.metadata["fit_strategies"] == ["smoothing_bspline"]
    assert rep.metadata["actual_segments"] == 1
    assert rep.times.shape == (6,)


def test_fit_with_segmentation():
    ts = _ts([0.0, 1.0, 2.0, 5.0, 6.0, 7.0], [0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    boundaries = np.array([0.0, 3.0, 7.0])
    rep = fit_timeseries_spline(ts, boundaries=boundaries)
    assert rep.metadata["actual_segments"] == 2
    assert rep.metadata["fit_strategy"] == "cubic_interp"
    assert rep.metadata["fit_strategies"] == ["cubic_interp", "cubic_interp"]
    np.testing.assert_allclose(rep.metadata["segment_boundaries"], boundaries)
    assert len(rep.segment_start_piece_idx) == 2


def test_fit_single_point_segment():
    """A segment with only 1 point should not crash."""
    ts = _ts([0.0, 5.0, 10.0], [1.0, 2.0, 3.0])
    boundaries = np.array([0.0, 2.0, 10.0])
    rep = fit_timeseries_spline(ts, boundaries=boundaries)
    assert rep.metadata["actual_segments"] == 2


def test_fit_strategy_reports_mixed_segment_fits():
    """Metadata records when some segments need the cubic fallback."""
    ts = _ts(
        [0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0],
        [0.0, 1.0, 4.0, 9.0, 10.0, 11.0, 12.0],
    )
    rep = fit_timeseries_spline(ts, boundaries=np.array([0.0, 4.0, 7.0]))

    assert rep.metadata["fit_strategy"] == "mixed"
    assert rep.metadata["fit_strategies"] == ["smoothing_bspline", "cubic_interp"]


def test_fit_roundtrip_accuracy():
    """Fitted spline should pass through original (cubic interp) points."""
    ts = _ts([0.0, 2.0, 4.0, 6.0, 8.0], [0.0, 1.0, 0.0, 1.0, 0.0])
    rep = fit_timeseries_spline(ts)
    for t_val, expected in zip([0.0, 2.0, 4.0, 6.0, 8.0], [0.0, 1.0, 0.0, 1.0, 0.0]):
        result = float(rep.evaluate(t_val))
        assert abs(result - expected) < 1e-4, (
            f"At t={t_val}: got {result}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# TimeSeries.evaluate
# ---------------------------------------------------------------------------


def test_timeseries_evaluate_multi_segment_fit():
    ts = _ts([0.0, 1.0, 2.0, 5.0, 6.0, 7.0], [0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    boundaries = np.array([0.0, 3.0, 7.0])
    rep = fit_timeseries_spline(ts, boundaries=boundaries)
    val_seg1 = float(rep.evaluate(1.0))
    assert abs(val_seg1 - 1.0) < 1e-3
    val_seg2 = float(rep.evaluate(6.0))
    assert abs(val_seg2 - 11.0) < 1e-3


# ---------------------------------------------------------------------------
# TimeSeries spline serialization round-trip (JSON)
# ---------------------------------------------------------------------------


def test_spline_json_roundtrip():
    """Spline-backed TimeSeries survives JSON save/load."""
    ts = _ts([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 0.5, 1.5, 3.0, 5.0])
    rep = fit_timeseries_spline(ts)

    pv = ProcessVariable(
        name="test_var",
        unit="g/L",
        is_controlled=True,
        values=rep,
    )
    rm = ReactorMedium(name="m", density=1.0, density_unit="kg/L")
    proc = BioProcess(
        metadata=BioProcessMetadata(name="p", process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=4.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=rm,
        process_variables={"test_var": pv},
    )
    cs = BioProcessCollection(
        case_id="cs", organism="E. coli", citation="test", processes={"p": proc}
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_process_collection(cs, path)
        payload = path.read_text()
        loaded = load_process_collection(path)

    assert '"interpolator"' not in payload
    assert '"breaks"' in payload
    assert '"coeffs"' in payload
    loaded_pv = loaded.processes["p"].process_variables["test_var"]
    assert isinstance(loaded_pv.values, TimeSeries)
    np.testing.assert_allclose(loaded_pv.values.breaks, rep.breaks)
    np.testing.assert_allclose(loaded_pv.values.coeffs, rep.coeffs)
    np.testing.assert_allclose(
        loaded_pv.values.segment_start_piece_idx,
        rep.segment_start_piece_idx,
    )

    for t_val in [0.0, 1.0, 2.0, 3.0, 4.0]:
        orig = float(rep.evaluate(t_val))
        loaded_val = float(loaded_pv.values.evaluate(t_val))
        assert abs(orig - loaded_val) < 1e-6, (
            f"At t={t_val}: orig={orig}, loaded={loaded_val}"
        )


def test_discrete_events_json_roundtrip():
    """DiscreteEvents survive JSON save/load on BioProcess."""
    de = DiscreteEvents(times=jnp.array([3.0, 12.0]), labels=["a", "b"])
    rm = ReactorMedium(name="m", density=1.0, density_unit="kg/L")
    proc = BioProcess(
        metadata=BioProcessMetadata(name="p", process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=20.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=rm,
        discrete_events=de,
    )
    cs = BioProcessCollection(
        case_id="cs", organism="E. coli", citation="test", processes={"p": proc}
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_process_collection(cs, path)
        loaded = load_process_collection(path)

    loaded_proc = loaded.processes["p"]
    assert loaded_proc.discrete_events is not None
    assert jnp.allclose(loaded_proc.discrete_events.times, jnp.array([3.0, 12.0]))
    assert loaded_proc.discrete_events.labels == ["a", "b"]


def test_no_interpolator_field():
    """Datasets without legacy sibling interpolator payloads still load fine."""
    rm = ReactorMedium(name="m", density=1.0, density_unit="kg/L")
    pv = ProcessVariable(
        name="x", unit="g/L", is_controlled=False, values=_ts([0.0, 1.0], [1.0, 2.0])
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="p", process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=1.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=rm,
        process_variables={"x": pv},
    )
    cs = BioProcessCollection(
        case_id="cs", organism="E. coli", citation="test", processes={"p": proc}
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_process_collection(cs, path)
        payload = path.read_text()
        loaded = load_process_collection(path)

    loaded_pv = loaded.processes["p"].process_variables["x"]
    assert '"interpolator"' not in payload
    np.testing.assert_allclose(loaded_pv.values.times, pv.values.times)
    np.testing.assert_allclose(loaded_pv.values.values, pv.values.values)


def test_short_series_falls_back_to_cubic_interp():
    """Segments with fewer than four samples use CubicSpline fallback."""
    times = jnp.asarray([0.0, 1.0, 2.0])
    values = jnp.asarray([1.0, 3.0, 2.0])
    ts = TimeSeries(times=times, values=values)

    fitted = fit_timeseries_spline(ts, smoothing_s=6.0)

    assert fitted.metadata["fit_strategy"] == "cubic_interp"
    assert fitted.metadata["fit_strategies"] == ["cubic_interp"]
    np.testing.assert_allclose(fitted.evaluate_many(times), values, atol=1e-6)


def test_sparse_series_with_four_or_more_points_uses_smoothing_bspline():
    """Sparse series use smoothing B-splines when cubic smoothing is possible."""
    times = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    values = jnp.asarray([1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 8.0])
    ts = TimeSeries(times=times, values=values)

    fitted = fit_timeseries_spline(ts, smoothing_s=6.0)
    scipy_bspline = interpolate.make_splrep(
        np.asarray(times), np.asarray(values), s=6.0, k=3
    )

    assert fitted.metadata["fit_strategy"] == "smoothing_bspline"
    assert fitted.metadata["fit_strategies"] == ["smoothing_bspline"]
    assert fitted.metadata["smoothing_s"] == pytest.approx(6.0)
    assert np.max(np.abs(np.asarray(fitted.evaluate_many(times)) - values)) > 0.1
    for t_val in np.linspace(0.25, 5.75, 7):
        assert float(fitted.evaluate(t_val)) == pytest.approx(
            float(scipy_bspline(t_val)), abs=1e-6
        )


def test_smoothing_spline_uses_bspline_knots_without_uniform_resampling():
    """Smoothing path stores SciPy's fitted spline pieces directly."""
    n = 150
    times = jnp.linspace(0.0, 10.0, n)
    rng = np.random.default_rng(0)
    values = jnp.asarray(np.sin(np.asarray(times)) + 0.05 * rng.standard_normal(n))
    ts = TimeSeries(times=times, values=values)

    fitted = fit_timeseries_spline(ts, smoothing_s=2.0)
    scipy_bspline = interpolate.make_splrep(
        np.asarray(times), np.asarray(values), s=2.0, k=3
    )

    assert fitted.metadata["fit_strategy"] == "smoothing_bspline"
    assert "bc_type" not in fitted.metadata
    assert fitted.metadata["smoothing_storage"] == "direct_power_basis"
    assert "n_ctrl" not in fitted.metadata
    assert "n_ctrl_semantics" not in fitted.metadata
    for t_val in np.linspace(0.25, 9.75, 7):
        assert float(fitted.evaluate(t_val)) == pytest.approx(
            float(scipy_bspline(t_val)), abs=1e-6
        )


def test_exact_smoothing_bspline_uses_scipy_default_knots():
    """Exact smoothing path delegates knot selection to SciPy."""
    n = 150
    times = jnp.linspace(0.0, 10.0, n)
    values = jnp.sin(times)
    ts = TimeSeries(times=times, values=values)

    fitted = fit_timeseries_spline(ts, smoothing_s=0.0)
    scipy_bspline = interpolate.make_splrep(
        np.asarray(times), np.asarray(values), s=0.0, k=3
    )

    assert fitted.metadata["fit_strategy"] == "smoothing_bspline"
    assert "n_ctrl" not in fitted.metadata
    assert "n_ctrl_semantics" not in fitted.metadata
    for t_val in np.linspace(0.25, 9.75, 7):
        assert float(fitted.evaluate(t_val)) == pytest.approx(
            float(scipy_bspline(t_val)), abs=1e-6
        )


def test_segmented_smoothing_spline_matches_per_segment_scipy_fit():
    """Segmented smoothing preserves each segment's fitted BSpline pieces."""
    n = 302
    times = jnp.linspace(0.0, 20.0, n)
    values = jnp.sin(times) + 0.1 * jnp.cos(3.0 * times)
    boundaries = np.array([0.0, 10.0, 20.0])
    fitted = fit_timeseries_spline(
        TimeSeries(times=times, values=values),
        boundaries=boundaries,
        smoothing_s=1.0,
    )

    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        mask = (np.asarray(times) >= lo) & (np.asarray(times) <= hi)
        scipy_bspline = interpolate.make_splrep(
            np.asarray(times)[mask],
            np.asarray(values)[mask],
            s=1.0,
            k=3,
        )
        for t_val in np.linspace(lo + 0.25, hi - 0.25, 4):
            assert float(fitted.evaluate(t_val)) == pytest.approx(
                float(scipy_bspline(t_val)), abs=1e-6
            )


def test_smoothing_spline_metadata_round_trips_through_serialization():
    """fit_strategy/smoothing_s on TimeSeries metadata survive save/load."""
    n = 150
    times = jnp.linspace(0.0, 10.0, n)
    rng = np.random.default_rng(0)
    values = jnp.asarray(np.sin(np.asarray(times)) + 0.05 * rng.standard_normal(n))
    ts = TimeSeries(times=times, values=values)
    fitted = fit_timeseries_spline(ts, smoothing_s=2.0)
    assert fitted.metadata["fit_strategy"] == "smoothing_bspline"
    assert fitted.metadata["smoothing_s"] == 2.0

    process = _make_process_continuous_only()
    process.process_variables["temperature"] = ProcessVariable(
        name="temperature", unit="C", is_controlled=False, values=fitted
    )
    case_study = BioProcessCollection(
        case_id="cs", organism="test", citation="test", processes={"p": process}
    )

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+", delete=False) as f:
        save_process_collection(case_study, f.name)
        loaded = load_process_collection(f.name)

    reloaded = loaded.processes["p"].process_variables["temperature"].values
    assert reloaded.metadata["fit_strategy"] == "smoothing_bspline"
    assert reloaded.metadata["smoothing_s"] == 2.0


def _make_process_continuous_only(glucose_feed_conc=100.0):
    """Process with only a continuous cumulative feed and glucose measurements."""
    feed_medium = FeedMedium(
        name="cont_feed_medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=StaticVariable(value=glucose_feed_conc),
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
                concentration=_ts(
                    [0.0, 5.0, 10.0, 15.0, 20.0],
                    [10.0, 8.0, 6.0, 4.0, 2.0],
                ),
            ),
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=_ts(
                    [0.0, 5.0, 10.0, 15.0, 20.0],
                    [10.0, 8.0, 6.0, 4.0, 2.0],
                ),
            ),
        },
    )
    vol = Volume(
        initial_volume=1.0,
        unit="L",
        volume_changes={
            "cont_feed": Inflow(
                name="cont_feed",
                unit="L",
                is_controlled=True,
                is_continuous=True,
                feed_medium=feed_medium,
                values=_ts(
                    [0.0, 5.0, 10.0, 15.0, 20.0],
                    [0.0, 0.1, 0.2, 0.3, 0.4],
                ),
            ),
        },
    )
    return BioProcess(
        metadata=BioProcessMetadata(name="test_cont", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=20.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )


def test_load_rejects_legacy_process_variable_interpolator_payload():
    """Loader should reject legacy sibling interpolator payloads on PVs."""
    ts = fit_timeseries_spline(_ts([0.0, 1.0, 2.0], [0.0, 2.0, 4.0]))
    pv = ProcessVariable(
        name="linear_var",
        unit="g/L",
        is_controlled=False,
        values=ts,
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="p", process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="m", density=1.0, density_unit="kg/L"),
        process_variables={"linear_var": pv},
    )
    cs = BioProcessCollection(
        case_id="cs", organism="E. coli", citation="test", processes={"p": proc}
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "legacy-pv.json"
        save_process_collection(cs, path)
        payload = path.read_text()
        mutated = payload.replace(
            '"values": {',
            '"interpolator": {"kind": "legacy_linear"},\n          "values": {',
            1,
        )
        path.write_text(mutated)
        with pytest.raises(ValueError, match="Legacy sibling 'interpolator' payloads"):
            load_process_collection(path)


def test_load_rejects_legacy_reactor_component_interpolator_payload():
    """Loader should reject legacy sibling interpolator payloads on components."""
    ts = fit_timeseries_spline(_ts([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]))
    rm = ReactorMedium(
        name="m",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=ts,
            ),
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=ts,
            ),
        },
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="p", process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=rm,
    )
    cs = BioProcessCollection(
        case_id="cs", organism="E. coli", citation="test", processes={"p": proc}
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "legacy-comp.json"
        save_process_collection(cs, path)
        payload = path.read_text()
        mutated = payload.replace(
            '"concentration": {',
            '"interpolator": {"kind": "legacy_ppoly"},\n'
            '              "concentration": {',
            1,
        )
        path.write_text(mutated)
        with pytest.raises(ValueError, match="Legacy sibling 'interpolator' payloads"):
            load_process_collection(path)


def test_load_rejects_legacy_volume_change_interpolator_payload():
    """Loader should reject legacy sibling interpolator payloads on volume changes."""
    feed = Inflow(
        name="feed",
        unit="L",
        is_controlled=True,
        is_continuous=True,
        feed_medium=_make_feed(),
        values=fit_timeseries_spline(_ts([0.0, 1.0, 2.0], [0.0, 0.1, 0.3])),
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="p", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes={"feed": feed}),
        reactor_medium=ReactorMedium(name="m", density=1.0, density_unit="kg/L"),
    )
    cs = BioProcessCollection(
        case_id="cs", organism="E. coli", citation="test", processes={"p": proc}
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "legacy-vc.json"
        save_process_collection(cs, path)
        payload = path.read_text()
        mutated = payload.replace(
            '"values": {',
            '"interpolator": {"kind": "legacy_linear"},\n            "values": {',
            1,
        )
        path.write_text(mutated)
        with pytest.raises(ValueError, match="Legacy sibling 'interpolator' payloads"):
            load_process_collection(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
