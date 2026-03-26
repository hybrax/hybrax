"""
Tests for bpbench.splines: spline fitting, serialization, evaluation,
and pseudobatch transform pipeline.
"""

import pytest
import jax.numpy as jnp
import numpy as np
import tempfile
from pathlib import Path

from bpbench import (
    BioProcess, BioProcessMetadata, TimeAxis, TimeSeries, StaticVariable,
    ReactorMedium, ReactorMediumComponent, FeedMedium, FeedMediumComponent,
    FeedVolumeChange, SampleVolumeChange, Volume, ProcessVariable,
    Interpolator, DiscreteEvents,
)
from bpbench.splines import (
    detect_discrete_state_events, make_segment_boundaries, split_timeseries,
    choose_spline_kind, fit_timeseries_spline, build_interpax_spline,
    evaluate_spline_at, SMOOTHING_THRESHOLD,
    build_pseudobatch_inputs, build_splines, evaluate_real_concentration,
    to_interpolator, build_backtransform_spline, BacktransformSpline,
)
from bpbench.serialization import save_dataset, save_dataset_json, load_dataset, load_dataset_json
from bpbench import BenchmarkDataset, CaseStudy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(t, v):
    return TimeSeries(timepoints=jnp.array(t, dtype=float),
                      values=jnp.array(v, dtype=float))


def _make_feed(name="feed"):
    return FeedMedium(name=name, density=1.0, density_unit="kg/L")


def _make_process_with_discrete():
    """Process with both continuous and discrete volume changes."""
    rm = ReactorMedium(
        name="medium", density=1.0, density_unit="kg/L",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass", unit="g/L",
                concentration=_ts([0., 5., 10., 20.], [0.5, 1.0, 2.0, 4.0]),
                is_intracellular=False,
            ),
        },
    )
    vol = Volume(
        initial_volume=1.0, unit="L",
        volume_changes={
            "continuous_feed": FeedVolumeChange(
                name="continuous_feed", unit="L",
                is_controlled=True, is_continuous=True,
                feed_medium=_make_feed("cont"),
                values=_ts([0., 5., 10., 20.], [0.0, 0.25, 0.5, 1.0]),
            ),
            "bolus": FeedVolumeChange(
                name="bolus", unit="L",
                is_controlled=True, is_continuous=False,
                feed_medium=_make_feed("bolus"),
                values=_ts([3.0, 12.0], [0.1, 0.1]),
            ),
        },
    )
    return BioProcess(
        metadata=BioProcessMetadata(name="test_proc", process_type="fed_batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=20.0, time_reference="inoculation"),
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
        initial_volume=1.0, unit="L",
        volume_changes={
            "feed": FeedVolumeChange(
                name="feed", unit="L",
                is_controlled=True, is_continuous=True,
                feed_medium=_make_feed(),
                values=_ts([0., 10.], [0.0, 1.0]),
            ),
        },
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="t", process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=10.0, time_reference="inoculation"),
        volume=vol, reactor_medium=rm,
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
    ts = _ts([0., 2., 4., 6.], [1., 2., 3., 4.])
    segments = split_timeseries(ts, np.array([0.0, 6.0]))
    assert len(segments) == 1
    assert segments[0].timepoints.shape[0] == 4


def test_split_timeseries_two_segments():
    ts = _ts([0., 1., 2., 3., 4., 5.], [10., 20., 30., 40., 50., 60.])
    segments = split_timeseries(ts, np.array([0.0, 2.5, 5.0]))
    assert len(segments) == 2
    assert segments[0].timepoints.shape[0] == 3
    assert segments[1].timepoints.shape[0] == 3


# ---------------------------------------------------------------------------
# choose_spline_kind
# ---------------------------------------------------------------------------

def test_choose_cubic_interp():
    assert choose_spline_kind(50) == "cubic_interp"
    assert choose_spline_kind(100) == "cubic_interp"


def test_choose_smoothing():
    assert choose_spline_kind(101) == "smoothing_bspline"
    assert choose_spline_kind(500) == "smoothing_bspline"


# ---------------------------------------------------------------------------
# fit_timeseries_spline
# ---------------------------------------------------------------------------

def test_fit_simple_cubic():
    ts = _ts([0., 1., 2., 3., 4., 5.], [0., 1., 4., 9., 16., 25.])
    rep = fit_timeseries_spline(ts)
    assert isinstance(rep, Interpolator)
    assert rep.kind == "interpax_cubic"
    assert rep.n_segments == 1
    assert int(rep.n[0]) == 6
    assert rep.x.shape == (16, 128)  # default padding


def test_fit_with_segmentation():
    ts = _ts([0., 1., 2., 5., 6., 7.], [0., 1., 2., 10., 11., 12.])
    boundaries = np.array([0.0, 3.0, 7.0])
    rep = fit_timeseries_spline(ts, boundaries=boundaries)
    assert rep.n_segments == 2
    assert int(rep.n[0]) >= 2
    assert int(rep.n[1]) >= 2


def test_fit_single_point_segment():
    """A segment with only 1 point should not crash."""
    ts = _ts([0., 5., 10.], [1., 2., 3.])
    boundaries = np.array([0.0, 2.0, 10.0])
    rep = fit_timeseries_spline(ts, boundaries=boundaries)
    assert rep.n_segments == 2


def test_fit_roundtrip_accuracy():
    """Fitted spline should pass through original (cubic interp) points."""
    ts = _ts([0., 2., 4., 6., 8.], [0., 1., 0., 1., 0.])
    rep = fit_timeseries_spline(ts)
    for t_val, expected in zip([0., 2., 4., 6., 8.], [0., 1., 0., 1., 0.]):
        result = evaluate_spline_at(rep, t_val)
        assert abs(result - expected) < 1e-4, f"At t={t_val}: got {result}, expected {expected}"


# ---------------------------------------------------------------------------
# build_interpax_spline / evaluate_spline_at
# ---------------------------------------------------------------------------

def test_build_interpax_spline():
    ts = _ts([0., 1., 2., 3.], [0., 1., 4., 9.])
    rep = fit_timeseries_spline(ts)
    splines, boundaries = build_interpax_spline(rep)
    assert len(splines) == 1
    assert len(boundaries) == 2
    val = float(splines[0](jnp.array(1.0)))
    assert abs(val - 1.0) < 1e-4


def test_evaluate_spline_at_multi_segment():
    ts = _ts([0., 1., 2., 5., 6., 7.], [0., 1., 2., 10., 11., 12.])
    boundaries = np.array([0.0, 3.0, 7.0])
    rep = fit_timeseries_spline(ts, boundaries=boundaries)
    val_seg1 = evaluate_spline_at(rep, 1.0)
    assert abs(val_seg1 - 1.0) < 1e-3
    val_seg2 = evaluate_spline_at(rep, 6.0)
    assert abs(val_seg2 - 11.0) < 1e-3


# ---------------------------------------------------------------------------
# Interpolator serialization round-trip (JSON)
# ---------------------------------------------------------------------------

def test_spline_json_roundtrip():
    """Interpolator survives JSON save/load."""
    ts = _ts([0., 1., 2., 3., 4.], [0., 0.5, 1.5, 3.0, 5.0])
    rep = fit_timeseries_spline(ts)

    pv = ProcessVariable(
        name="test_var", unit="g/L", is_controlled=True,
        values=ts, interpolator=rep,
    )
    rm = ReactorMedium(name="m", density=1.0, density_unit="kg/L")
    proc = BioProcess(
        metadata=BioProcessMetadata(name="p", process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=4.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=rm,
        process_variables={"test_var": pv},
    )
    cs = CaseStudy(case_id="cs", organism="E. coli", citation="test",
                   processes={"p": proc})
    ds = BenchmarkDataset(metadata={"name": "test"}, case_studies={"cs": cs})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_dataset_json(ds, path)
        payload = path.read_text()
        loaded = load_dataset_json(path)

    assert '"interpolator"' in payload
    assert '"spline"' not in payload
    loaded_pv = loaded.case_studies["cs"].processes["p"].process_variables["test_var"]
    assert loaded_pv.interpolator is not None
    assert loaded_pv.interpolator.kind == rep.kind
    assert loaded_pv.interpolator.n_segments == rep.n_segments
    # Compare valid segment counts (loaded may have different padding)
    n_seg = rep.n_segments
    assert jnp.allclose(loaded_pv.interpolator.n[:n_seg], rep.n[:n_seg])

    for t_val in [0.0, 1.0, 2.0, 3.0, 4.0]:
        orig = evaluate_spline_at(rep, t_val)
        loaded_val = evaluate_spline_at(loaded_pv.interpolator, t_val)
        assert abs(orig - loaded_val) < 1e-6, f"At t={t_val}: orig={orig}, loaded={loaded_val}"


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
    cs = CaseStudy(case_id="cs", organism="E. coli", citation="test",
                   processes={"p": proc})
    ds = BenchmarkDataset(metadata={"name": "test"}, case_studies={"cs": cs})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_dataset_json(ds, path)
        loaded = load_dataset_json(path)

    loaded_proc = loaded.case_studies["cs"].processes["p"]
    assert loaded_proc.discrete_events is not None
    assert jnp.allclose(loaded_proc.discrete_events.times, jnp.array([3.0, 12.0]))
    assert loaded_proc.discrete_events.labels == ["a", "b"]


def test_no_interpolator_field():
    """Datasets without interpolator fields still load fine."""
    rm = ReactorMedium(name="m", density=1.0, density_unit="kg/L")
    pv = ProcessVariable(name="x", unit="g/L", is_controlled=False,
                         values=_ts([0., 1.], [1., 2.]))
    proc = BioProcess(
        metadata=BioProcessMetadata(name="p", process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=1.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=rm,
        process_variables={"x": pv},
    )
    cs = CaseStudy(case_id="cs", organism="E. coli", citation="test",
                   processes={"p": proc})
    ds = BenchmarkDataset(metadata={"name": "test"}, case_studies={"cs": cs})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_dataset_json(ds, path)
        loaded = load_dataset_json(path)

    loaded_pv = loaded.case_studies["cs"].processes["p"].process_variables["x"]
    assert loaded_pv.interpolator is None


# ---------------------------------------------------------------------------
# Pseudo-batch helpers
# ---------------------------------------------------------------------------

def _make_process_with_bolus_feed(
    V0=1.0,
    feed_times=None,
    feed_vols=None,
    glucose_feed_conc=500.0,
    glucose_times=None,
    glucose_values=None,
):
    """Helper: minimal BioProcess with a bolus feed and glucose measurements."""
    if feed_times is None:
        feed_times = [50.0]
    if feed_vols is None:
        feed_vols = [0.2]
    if glucose_times is None:
        glucose_times = [0.0, 25.0, 50.0, 75.0, 100.0]
    if glucose_values is None:
        glucose_values = [10.0, 8.0, 6.0, 7.5, 5.0]

    feed_medium = FeedMedium(
        name="bolus_feed_medium", density=1.0, density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose", unit="mmol/L",
                concentration=StaticVariable(value=glucose_feed_conc),
                is_controlled=True,
            ),
        },
    )
    rm = ReactorMedium(
        name="medium", density=1.0, density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose", unit="mmol/L",
                concentration=_ts(glucose_times, glucose_values),
                is_intracellular=False,
            ),
        },
    )
    vol = Volume(
        initial_volume=V0, unit="L",
        volume_changes={
            "bolus_feed": FeedVolumeChange(
                name="bolus_feed", unit="L",
                is_controlled=True, is_continuous=False,
                feed_medium=feed_medium,
                values=_ts(feed_times, feed_vols),
            ),
        },
    )
    return BioProcess(
        metadata=BioProcessMetadata(name="test_pb", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=max(glucose_times),
                           time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )


def _make_process_continuous_only(glucose_feed_conc=100.0):
    """Process with only a continuous cumulative feed and glucose measurements."""
    feed_medium = FeedMedium(
        name="cont_feed_medium", density=1.0, density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose", unit="mmol/L",
                concentration=StaticVariable(value=glucose_feed_conc),
                is_controlled=True,
            ),
        },
    )
    rm = ReactorMedium(
        name="medium", density=1.0, density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose", unit="mmol/L",
                concentration=_ts(
                    [0.0, 5.0, 10.0, 15.0, 20.0],
                    [10.0, 8.0, 6.0, 4.0, 2.0],
                ),
                is_intracellular=False,
            ),
        },
    )
    vol = Volume(
        initial_volume=1.0, unit="L",
        volume_changes={
            "cont_feed": FeedVolumeChange(
                name="cont_feed", unit="L",
                is_controlled=True, is_continuous=True,
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


# ---------------------------------------------------------------------------
# Pseudobatch pipeline: build_pseudobatch_inputs
# ---------------------------------------------------------------------------

def test_pseudobatch_inputs_discrete():
    """build_pseudobatch_inputs produces valid outputs for discrete bolus feed."""
    proc = _make_process_with_bolus_feed(
        V0=1.0, feed_times=[50.0], feed_vols=[0.2], glucose_feed_conc=500.0,
    )
    inputs = build_pseudobatch_inputs(proc, "glucose")

    np.testing.assert_array_equal(inputs["meas_times"], [0.0, 25.0, 50.0, 75.0, 100.0])
    assert inputs["c_star"].shape == (5,)
    assert np.all(np.isfinite(inputs["c_star"]))
    assert inputs["has_discrete_feed"] is True


def test_pseudobatch_inputs_continuous():
    """build_pseudobatch_inputs produces valid outputs for continuous feed."""
    proc = _make_process_continuous_only(glucose_feed_conc=100.0)
    inputs = build_pseudobatch_inputs(proc, "glucose")

    np.testing.assert_array_equal(inputs["meas_times"], [0.0, 5.0, 10.0, 15.0, 20.0])
    assert inputs["c_star"].shape == (5,)
    assert np.all(np.isfinite(inputs["c_star"]))
    assert inputs["has_discrete_feed"] is False


def test_pseudobatch_species_not_in_feed():
    """For a species absent from the feed medium, the feed correction is zero."""
    proc = _make_process_with_bolus_feed()
    proc.reactor_medium.components["biomass"] = ReactorMediumComponent(
        name="biomass", unit="cells/L",
        concentration=_ts([0.0, 25.0, 50.0, 75.0, 100.0],
                          [1.0, 2.0, 4.0, 8.0, 16.0]),
        is_intracellular=False,
    )
    inputs = build_pseudobatch_inputs(proc, "biomass")
    np.testing.assert_allclose(inputs["feed_corr_at_meas"], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Pseudobatch pipeline: to_interpolator + evaluate roundtrip
# ---------------------------------------------------------------------------

def test_interpolator_roundtrip_bolus():
    """to_interpolator -> build_backtransform_spline
    matches evaluate_real_concentration for a bolus feed process."""
    proc = _make_process_with_bolus_feed(
        V0=1.0, feed_times=[50.0], feed_vols=[0.2], glucose_feed_conc=500.0,
        glucose_times=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0],
        glucose_values=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 7.0, 6.0, 5.0, 4.0],
    )
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")

    rep = to_interpolator(inputs, splines, "glucose")
    assert rep.interpolator_metadata is not None
    assert rep.interpolator_metadata["transform"]["name"] == "pseudo_batch"
    assert rep.interpolator_metadata["transform"]["species"] == "glucose"
    assert rep.interpolator_metadata["transform"]["feed_corr_interp"] == "linear"

    bt = build_backtransform_spline(rep)
    assert isinstance(bt, BacktransformSpline)

    t_eval = np.linspace(0.0, 100.0, 50)
    direct = evaluate_real_concentration(t_eval, splines)
    from_rep = np.array([float(bt(jnp.array(t))) for t in t_eval])

    np.testing.assert_allclose(from_rep, direct, rtol=1e-4, atol=1e-6)


def test_interpolator_roundtrip_continuous():
    """to_interpolator -> build_backtransform_spline
    matches evaluate_real_concentration for a continuous feed process."""
    proc = _make_process_continuous_only(glucose_feed_conc=100.0)
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")

    rep = to_interpolator(inputs, splines, "glucose")
    assert rep.interpolator_metadata["transform"]["feed_corr_interp"] == "cubic"

    bt = build_backtransform_spline(rep)

    t_eval = np.linspace(0.0, 20.0, 30)
    direct = evaluate_real_concentration(t_eval, splines)
    from_rep = np.array([float(bt(jnp.array(t))) for t in t_eval])

    np.testing.assert_allclose(from_rep, direct, rtol=1e-4, atol=1e-6)


def test_interpolator_scalar():
    """BacktransformSpline works for scalar evaluation."""
    proc = _make_process_with_bolus_feed()
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_interpolator(inputs, splines, "glucose")

    bt = build_backtransform_spline(rep)
    val = float(bt(jnp.array(25.0)))
    assert np.isfinite(val)
    assert val > 0


def test_backtransform_has_jump_at_bolus():
    """Evaluating just before/after a bolus feed should show a discontinuity."""
    proc = _make_process_with_bolus_feed(
        V0=1.0, feed_times=[50.0], feed_vols=[0.2], glucose_feed_conc=500.0,
        glucose_times=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0],
        glucose_values=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 7.0, 6.0, 5.0, 4.0],
    )
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_interpolator(inputs, splines, "glucose")

    bt = build_backtransform_spline(rep)

    # Use eps > _EPS (1e-4) to cross the dense grid's pre-event epsilon point
    eps = 5e-4
    val_before = float(bt(jnp.array(50.0 - eps)))
    val_after = float(bt(jnp.array(50.0 + eps)))

    jump = abs(val_after - val_before)
    assert jump > 0.1, (
        f"Expected a jump at t_feed=50.0, got val_before={val_before}, "
        f"val_after={val_after}, jump={jump}"
    )


def test_continuous_backtransform_no_nan():
    """Fitting and back-transforming a continuous-only process gives finite values."""
    proc = _make_process_continuous_only(glucose_feed_conc=100.0)
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_interpolator(inputs, splines, "glucose")

    bt = build_backtransform_spline(rep)
    for t in [0.0, 5.0, 10.0, 15.0, 20.0]:
        val = float(bt(jnp.array(t)))
        assert np.isfinite(val), f"Non-finite value at t={t}: {val}"
        assert val >= 0, f"Negative concentration at t={t}: {val}"


# ---------------------------------------------------------------------------
# Interpolator JSON serialization with backtransform metadata
# ---------------------------------------------------------------------------

def test_pseudobatch_spline_json_roundtrip():
    """Interpolator with pseudobatch metadata survives JSON roundtrip."""
    proc = _make_process_with_bolus_feed()
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_interpolator(inputs, splines, "glucose")

    # Attach to a ReactorMediumComponent
    comp = proc.reactor_medium.components["glucose"]
    comp.interpolator = rep

    cs = CaseStudy(case_id="cs", organism="CHO", citation="test",
                   processes={"p": proc})
    ds = BenchmarkDataset(metadata={"name": "test"}, case_studies={"cs": cs})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pb.json"
        save_dataset_json(ds, path)
        payload = path.read_text()
        loaded = load_dataset_json(path)

    assert '"interpolator"' in payload
    loaded_comp = loaded.case_studies["cs"].processes["p"].reactor_medium.components["glucose"]
    assert loaded_comp.interpolator is not None
    loaded_tr = loaded_comp.interpolator.interpolator_metadata["transform"]
    assert loaded_tr["name"] == "pseudo_batch"

    bt_orig = build_backtransform_spline(rep)
    bt_loaded = build_backtransform_spline(loaded_comp.interpolator)
    for t_val in [0.0, 25.0, 75.0]:
        orig = float(bt_orig(jnp.array(t_val)))
        loaded_val = float(bt_loaded(jnp.array(t_val)))
        assert abs(orig - loaded_val) < 1e-4, (
            f"Roundtrip mismatch at t={t_val}: {orig} vs {loaded_val}"
        )


def test_pseudobatch_metadata_json_serializable():
    """Transform metadata is pure JSON-serializable (lists, not arrays)."""
    import json

    proc = _make_process_with_bolus_feed()
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_interpolator(inputs, splines, "glucose")

    assert "transform" in rep.interpolator_metadata
    tr = rep.interpolator_metadata["transform"]
    assert tr["name"] == "pseudo_batch"
    assert tr["species"] == "glucose"

    # Should be JSON-serializable
    meta_json = json.dumps(rep.interpolator_metadata)
    meta_loaded = json.loads(meta_json)
    assert meta_loaded["transform"]["name"] == "pseudo_batch"


def test_linear_interpolator_json_roundtrip():
    interp = Interpolator(
        kind="interpax_linear",
        x=jnp.array([[0.0, 1.0, 2.0]]),
        y=jnp.array([[0.0, 2.0, 4.0]]),
        n=jnp.array([3]),
        n_segments=1,
        segment_boundaries=jnp.array([0.0, 2.0]),
        bc_type=None,
        interpolator_metadata={"source": "test"},
    )
    pv = ProcessVariable(
        name="linear_var",
        unit="g/L",
        is_controlled=False,
        values=_ts([0.0, 1.0, 2.0], [0.0, 2.0, 4.0]),
        interpolator=interp,
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="p", process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="m", density=1.0, density_unit="kg/L"),
        process_variables={"linear_var": pv},
    )
    ds = BenchmarkDataset(
        metadata={"name": "test"},
        case_studies={"cs": CaseStudy(case_id="cs", organism="E. coli", citation="test", processes={"p": proc})},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "linear.json"
        save_dataset_json(ds, path)
        loaded = load_dataset_json(path)

    loaded_interp = loaded.case_studies["cs"].processes["p"].process_variables["linear_var"].interpolator
    assert loaded_interp is not None
    assert loaded_interp.kind == "interpax_linear"
    assert loaded_interp.bc_type is None
    assert loaded_interp.interpolator_metadata == {"source": "test"}


def test_ppoly_interpolator_json_roundtrip():
    interp = Interpolator(
        kind="interpax_ppoly",
        x=jnp.array([0.0, 1.0, 2.0]),
        coefficients=jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        extrapolate=False,
        interpolator_metadata={"axis": 0},
    )
    rm = ReactorMedium(
        name="m",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=_ts([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]),
                is_intracellular=False,
                interpolator=interp,
            )
        },
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="p", process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=rm,
    )
    ds = BenchmarkDataset(
        metadata={"name": "test"},
        case_studies={"cs": CaseStudy(case_id="cs", organism="E. coli", citation="test", processes={"p": proc})},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ppoly.json"
        save_dataset_json(ds, path)
        loaded = load_dataset_json(path)

    loaded_interp = loaded.case_studies["cs"].processes["p"].reactor_medium.components["glucose"].interpolator
    assert loaded_interp is not None
    assert loaded_interp.kind == "interpax_ppoly"
    assert loaded_interp.extrapolate is False
    np.testing.assert_allclose(loaded_interp.x, interp.x)
    np.testing.assert_allclose(loaded_interp.coefficients, interp.coefficients)


def test_linear_interpolator_hybrid_roundtrip():
    interp = Interpolator(
        kind="interpax_linear",
        x=jnp.array([[0.0, 1.0, 2.0]]),
        y=jnp.array([[0.0, 2.0, 4.0]]),
        n=jnp.array([3]),
        n_segments=1,
        segment_boundaries=jnp.array([0.0, 2.0]),
        bc_type=None,
        interpolator_metadata={"source": "test"},
    )
    pv = ProcessVariable(
        name="linear_var",
        unit="g/L",
        is_controlled=False,
        values=_ts([0.0, 1.0, 2.0], [0.0, 2.0, 4.0]),
        interpolator=interp,
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="p", process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="m", density=1.0, density_unit="kg/L"),
        process_variables={"linear_var": pv},
    )
    ds = BenchmarkDataset(
        metadata={"name": "test"},
        case_studies={"cs": CaseStudy(case_id="cs", organism="E. coli", citation="test", processes={"p": proc})},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "hybrid"
        save_dataset(ds, path)
        loaded = load_dataset(path)

    loaded_interp = loaded.case_studies["cs"].processes["p"].process_variables["linear_var"].interpolator
    assert loaded_interp is not None
    assert loaded_interp.kind == "interpax_linear"
    assert loaded_interp.interpolator_metadata == {"source": "test"}


def test_runtime_rejects_linear_interpolator():
    interp = Interpolator(
        kind="interpax_linear",
        x=jnp.array([[0.0, 1.0, 2.0]]),
        y=jnp.array([[0.0, 1.0, 2.0]]),
        n=jnp.array([3]),
        n_segments=1,
        segment_boundaries=jnp.array([0.0, 2.0]),
        bc_type=None,
    )

    with pytest.raises(NotImplementedError, match="interpax_cubic"):
        build_interpax_spline(interp)

    with pytest.raises(NotImplementedError, match="interpax_cubic"):
        evaluate_spline_at(interp, 1.0)


def test_runtime_rejects_ppoly_backtransform():
    interp = Interpolator(
        kind="interpax_ppoly",
        x=jnp.array([0.0, 1.0, 2.0]),
        coefficients=jnp.ones((4, 2)),
        interpolator_metadata={
            "transform": {
                "adf_times": [0.0, 2.0],
                "adf_values": [1.0, 1.0],
                "feed_corr_times": [0.0, 2.0],
                "feed_corr_values": [0.0, 0.0],
            }
        },
    )

    with pytest.raises(NotImplementedError, match="interpax_cubic"):
        build_backtransform_spline(interp)


# ---------------------------------------------------------------------------
# Pseudobatch with sample volume change
# ---------------------------------------------------------------------------

def test_pseudobatch_with_sample_volume_change():
    """Process with SampleVolumeChange should be handled correctly."""
    feed_medium = FeedMedium(
        name="feed", density=1.0, density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose", unit="mmol/L",
                concentration=StaticVariable(value=100.0),
                is_controlled=True,
            ),
        },
    )
    rm = ReactorMedium(
        name="medium", density=1.0, density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose", unit="mmol/L",
                concentration=_ts(
                    [0.0, 5.0, 10.0, 15.0, 20.0],
                    [10.0, 8.0, 6.0, 5.0, 4.0],
                ),
                is_intracellular=False,
            ),
        },
    )
    vol = Volume(
        initial_volume=1.0, unit="L",
        volume_changes={
            "feed": FeedVolumeChange(
                name="feed", unit="L",
                is_controlled=True, is_continuous=False,
                feed_medium=feed_medium,
                values=_ts([10.0], [0.2]),
            ),
            "sample": SampleVolumeChange(
                name="sample", unit="L",
                is_controlled=True, is_continuous=False,
                values=_ts([5.0], [-0.05]),
            ),
        },
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="test_sample", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=20.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )

    inputs = build_pseudobatch_inputs(proc, "glucose")
    assert inputs["c_star"].shape == (5,)
    assert np.all(np.isfinite(inputs["c_star"]))

    # Full pipeline should work
    splines = build_splines(inputs, proc, "glucose")
    rep = to_interpolator(inputs, splines, "glucose")
    bt = build_backtransform_spline(rep)
    t_eval = np.linspace(0.0, 20.0, 20)
    vals = np.array([float(bt(jnp.array(t))) for t in t_eval])
    assert np.all(np.isfinite(vals))


# ---------------------------------------------------------------------------
# Multiple feed streams
# ---------------------------------------------------------------------------

def test_pseudobatch_multiple_feed_streams():
    """Process with multiple feed streams should produce correct inputs."""
    feed_medium_1 = FeedMedium(
        name="feed1", density=1.0, density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose", unit="mmol/L",
                concentration=StaticVariable(value=200.0),
                is_controlled=True,
            ),
        },
    )
    feed_medium_2 = FeedMedium(
        name="feed2", density=1.0, density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose", unit="mmol/L",
                concentration=StaticVariable(value=50.0),
                is_controlled=True,
            ),
        },
    )
    rm = ReactorMedium(
        name="medium", density=1.0, density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose", unit="mmol/L",
                concentration=_ts([0.0, 10.0, 20.0], [10.0, 7.0, 5.0]),
                is_intracellular=False,
            ),
        },
    )
    vol = Volume(
        initial_volume=1.0, unit="L",
        volume_changes={
            "feed1": FeedVolumeChange(
                name="feed1", unit="L",
                is_controlled=True, is_continuous=False,
                feed_medium=feed_medium_1,
                values=_ts([5.0], [0.1]),
            ),
            "feed2": FeedVolumeChange(
                name="feed2", unit="L",
                is_controlled=True, is_continuous=False,
                feed_medium=feed_medium_2,
                values=_ts([15.0], [0.2]),
            ),
        },
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="test_multi", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=20.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )

    inputs = build_pseudobatch_inputs(proc, "glucose")
    assert inputs["c_star"].shape == (3,)
    assert np.all(np.isfinite(inputs["c_star"]))

    # Full pipeline
    splines = build_splines(inputs, proc, "glucose")
    rep = to_interpolator(inputs, splines, "glucose")
    bt = build_backtransform_spline(rep)
    t_eval = np.linspace(0.0, 20.0, 20)
    vals = np.array([float(bt(jnp.array(t))) for t in t_eval])
    assert np.all(np.isfinite(vals))


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------

def test_backtransform_spline_jit_bolus():
    """BacktransformSpline should be JIT-compilable (bolus/linear feed_corr)."""
    import equinox as eqx

    proc = _make_process_with_bolus_feed(
        V0=1.0, feed_times=[50.0], feed_vols=[0.2], glucose_feed_conc=500.0,
        glucose_times=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0],
        glucose_values=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 7.0, 6.0, 5.0, 4.0],
    )
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_interpolator(inputs, splines, "glucose")
    bt = build_backtransform_spline(rep)

    jit_fn = eqx.filter_jit(bt)
    val = jit_fn(jnp.array(25.0))
    assert np.isfinite(float(val))
    assert float(val) > 0


def test_backtransform_spline_jit_continuous():
    """BacktransformSpline should be JIT-compilable (continuous/cubic feed_corr)."""
    import equinox as eqx

    proc = _make_process_continuous_only(glucose_feed_conc=100.0)
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_interpolator(inputs, splines, "glucose")
    bt = build_backtransform_spline(rep)

    jit_fn = eqx.filter_jit(bt)
    val = jit_fn(jnp.array(10.0))
    assert np.isfinite(float(val))
    assert float(val) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
