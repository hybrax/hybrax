"""
Tests for bpbench.splines: spline fitting, serialization, and evaluation.
"""

import pytest
import jax.numpy as jnp
import numpy as np
import tempfile
from pathlib import Path

from bpbench import (
    BioProcess, BioProcessMetadata, TimeAxis, TimeSeries, StaticVariable,
    ReactorMedium, ReactorMediumComponent, FeedMedium, FeedMediumComponent,
    FeedVolumeChange, Volume, ProcessVariable, SplineRepresentation, DiscreteEvents,
)
from bpbench.splines import (
    detect_discrete_state_events, make_segment_boundaries, split_timeseries,
    choose_spline_kind, fit_timeseries_spline, build_interpax_spline,
    evaluate_spline_at, SMOOTHING_THRESHOLD,
    compute_volume_at_times, pseudo_batch_transform_timeseries,
    fit_state_timeseries_spline_pseudobatch, evaluate_timeseries_spline_at,
)
from bpbench.serialization import save_dataset_json, load_dataset_json
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
    """Process with only continuous changes → empty events."""
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
    # First segment: [0, 1, 2]
    assert segments[0].timepoints.shape[0] == 3
    # Second segment: [3, 4, 5]
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
    assert isinstance(rep, SplineRepresentation)
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
    # Evaluate at known point
    val = float(splines[0](jnp.float64(1.0)))
    assert abs(val - 1.0) < 1e-4


def test_evaluate_spline_at_multi_segment():
    ts = _ts([0., 1., 2., 5., 6., 7.], [0., 1., 2., 10., 11., 12.])
    boundaries = np.array([0.0, 3.0, 7.0])
    rep = fit_timeseries_spline(ts, boundaries=boundaries)
    # Evaluate at known points in each segment
    val_seg1 = evaluate_spline_at(rep, 1.0)
    assert abs(val_seg1 - 1.0) < 1e-3
    val_seg2 = evaluate_spline_at(rep, 6.0)
    assert abs(val_seg2 - 11.0) < 1e-3


# ---------------------------------------------------------------------------
# SplineRepresentation serialization round-trip (JSON)
# ---------------------------------------------------------------------------

def test_spline_json_roundtrip():
    """SplineRepresentation survives JSON save/load."""
    ts = _ts([0., 1., 2., 3., 4.], [0., 0.5, 1.5, 3.0, 5.0])
    rep = fit_timeseries_spline(ts)

    pv = ProcessVariable(
        name="test_var", unit="g/L", is_controlled=True,
        values=ts, spline=rep,
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
        loaded = load_dataset_json(path)

    loaded_pv = loaded.case_studies["cs"].processes["p"].process_variables["test_var"]
    assert loaded_pv.spline is not None
    assert loaded_pv.spline.kind == rep.kind
    assert loaded_pv.spline.n_segments == rep.n_segments
    assert jnp.allclose(loaded_pv.spline.n, rep.n)

    # Evaluate loaded spline and compare
    for t_val in [0.0, 1.0, 2.0, 3.0, 4.0]:
        orig = evaluate_spline_at(rep, t_val)
        loaded_val = evaluate_spline_at(loaded_pv.spline, t_val)
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


def test_no_spline_backward_compat():
    """Datasets without spline fields still load fine."""
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
    assert loaded_pv.spline is None


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


# ---------------------------------------------------------------------------
# Test A: evaluate_timeseries_spline_at falls back for non-pseudo-batch reps
# ---------------------------------------------------------------------------

def test_evaluate_timeseries_spline_at_fallback():
    """Without pseudo-batch metadata, evaluate_timeseries_spline_at equals evaluate_spline_at."""
    ts = _ts([0., 1., 2., 3., 4.], [0., 1., 4., 9., 16.])
    rep = fit_timeseries_spline(ts)
    for t_val in [0.0, 1.5, 3.0, 4.0]:
        expected = evaluate_spline_at(rep, t_val)
        actual = evaluate_timeseries_spline_at(rep, t_val)
        assert abs(actual - expected) < 1e-10, (
            f"Fallback mismatch at t={t_val}: {actual} vs {expected}"
        )


# ---------------------------------------------------------------------------
# Test B: Pseudo-batch metadata is JSON-serializable
# ---------------------------------------------------------------------------

def test_pseudobatch_metadata_json_serializable():
    """SplineRepresentation with pseudo-batch metadata survives JSON roundtrip."""
    import json

    proc = _make_process_with_bolus_feed()
    glucose_ts = proc.reactor_medium.components["glucose"].concentration
    rep = fit_state_timeseries_spline_pseudobatch(
        glucose_ts, proc, "glucose",
    )
    # Verify metadata is present
    assert "transform" in rep.spline_metadata
    tr = rep.spline_metadata["transform"]
    assert tr["name"] == "pseudo_batch"
    assert tr["species"] == "glucose"

    # Roundtrip via json
    meta_json = json.dumps(rep.spline_metadata)
    meta_loaded = json.loads(meta_json)
    assert meta_loaded["transform"]["name"] == "pseudo_batch"
    assert meta_loaded["transform"]["adf_step_values"] == tr["adf_step_values"]

    # Full dataset roundtrip
    pv = ProcessVariable(
        name="glucose", unit="mmol/L", is_controlled=False,
        values=glucose_ts, spline=rep,
    )
    proc.process_variables["glucose"] = pv
    cs = CaseStudy(case_id="cs", organism="CHO", citation="test",
                   processes={"p": proc})
    ds = BenchmarkDataset(metadata={"name": "test"}, case_studies={"cs": cs})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pb.json"
        save_dataset_json(ds, path)
        loaded = load_dataset_json(path)

    loaded_pv = loaded.case_studies["cs"].processes["p"].process_variables["glucose"]
    assert loaded_pv.spline is not None
    loaded_tr = loaded_pv.spline.spline_metadata["transform"]
    assert loaded_tr["name"] == "pseudo_batch"
    # Evaluate loaded spline with backtransform
    for t_val in [0.0, 25.0, 75.0]:
        orig = evaluate_timeseries_spline_at(rep, t_val)
        loaded_val = evaluate_timeseries_spline_at(loaded_pv.spline, t_val)
        assert abs(orig - loaded_val) < 1e-4, (
            f"Roundtrip mismatch at t={t_val}: {orig} vs {loaded_val}"
        )


# ---------------------------------------------------------------------------
# Test C: Backtransform reintroduces jumps at feed events
# ---------------------------------------------------------------------------

def test_pseudobatch_backtransform_has_jump():
    """Evaluating just before/after a bolus feed should show a discontinuity."""
    V0 = 1.0
    t_feed = 50.0
    delta_v = 0.2
    c_feed_glucose = 500.0

    # Simple scenario: glucose decreasing linearly, then bolus at t=50 adds
    # concentrated glucose, causing a jump up in true concentration.
    proc = _make_process_with_bolus_feed(
        V0=V0,
        feed_times=[t_feed],
        feed_vols=[delta_v],
        glucose_feed_conc=c_feed_glucose,
        glucose_times=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0],
        glucose_values=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 7.0, 6.0, 5.0, 4.0],
    )
    glucose_ts = proc.reactor_medium.components["glucose"].concentration
    rep = fit_state_timeseries_spline_pseudobatch(
        glucose_ts, proc, "glucose",
    )

    eps = 1e-6
    val_before = evaluate_timeseries_spline_at(rep, t_feed - eps)
    val_after = evaluate_timeseries_spline_at(rep, t_feed + eps)

    # There should be a jump (discontinuity) at the feed time.
    jump = val_after - val_before
    # The feed adds concentrated glucose to a diluted reactor, so the
    # concentration should jump up (positive direction).
    assert abs(jump) > 0.1, (
        f"Expected a jump at t_feed={t_feed}, got val_before={val_before}, "
        f"val_after={val_after}, jump={jump}"
    )
    # Direction: feed conc (500) >> reactor conc (~5), so concentration goes up
    assert jump > 0, (
        f"Expected positive jump from glucose bolus, got jump={jump}"
    )


# ---------------------------------------------------------------------------
# Test D: Species not in feed → feed term is 0, no crash
# ---------------------------------------------------------------------------

def test_pseudobatch_species_not_in_feed():
    """For a species absent from the feed medium, the transform should not crash
    and the feed term should be zero (transform reduces to ADF * c)."""
    proc = _make_process_with_bolus_feed()
    # Add a 'biomass' component to reactor medium (not in feed)
    proc.reactor_medium.components["biomass"] = ReactorMediumComponent(
        name="biomass", unit="cells/L",
        concentration=_ts([0.0, 25.0, 50.0, 75.0, 100.0],
                          [1.0, 2.0, 4.0, 8.0, 16.0]),
        is_intracellular=False,
    )
    bio_ts = proc.reactor_medium.components["biomass"].concentration
    pb = pseudo_batch_transform_timeseries(proc, "biomass", bio_ts)

    # Feed term should be zero everywhere
    assert np.allclose(pb["feed_term_step_values"], 0.0), (
        f"Feed term should be 0 for species not in feed, got {pb['feed_term_step_values']}"
    )

    # Should not crash when fitting and evaluating
    rep = fit_state_timeseries_spline_pseudobatch(bio_ts, proc, "biomass")
    val = evaluate_timeseries_spline_at(rep, 25.0)
    assert np.isfinite(val), f"Expected finite value, got {val}"


# ---------------------------------------------------------------------------
# Test: compute_volume_at_times
# ---------------------------------------------------------------------------

def test_compute_volume_at_times():
    """Volume should increase at bolus feed events."""
    proc = _make_process_with_bolus_feed(
        V0=1.0, feed_times=[50.0], feed_vols=[0.2],
    )
    times = np.array([0.0, 25.0, 49.9, 50.0, 75.0, 100.0])
    vol = compute_volume_at_times(proc, times)
    # Before feed: V = 1.0
    assert abs(vol[0] - 1.0) < 1e-6
    assert abs(vol[1] - 1.0) < 1e-6
    assert abs(vol[2] - 1.0) < 1e-6
    # At and after feed: V = 1.2
    assert abs(vol[3] - 1.2) < 1e-6
    assert abs(vol[4] - 1.2) < 1e-6


# ---------------------------------------------------------------------------
# Test: compute_volume_at_times with continuous feeds
# ---------------------------------------------------------------------------

def test_compute_volume_at_times_continuous():
    """Volume should increase smoothly with continuous cumulative feed."""
    rm = ReactorMedium(
        name="medium", density=1.0, density_unit="kg/L", components={},
    )
    vol = Volume(
        initial_volume=1.0, unit="L",
        volume_changes={
            "cont_feed": FeedVolumeChange(
                name="cont_feed", unit="L",
                is_controlled=True, is_continuous=True,
                feed_medium=_make_feed("cont"),
                values=_ts([0.0, 5.0, 10.0, 20.0], [0.0, 0.25, 0.5, 1.0]),
            ),
        },
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="test", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=20.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )
    times = np.array([0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0])
    v = compute_volume_at_times(proc, times)
    # At t=0: V = 1.0 + 0.0 = 1.0
    assert abs(v[0] - 1.0) < 1e-6
    # At t=5: V = 1.0 + 0.25 = 1.25
    assert abs(v[2] - 1.25) < 1e-6
    # At t=10: V = 1.0 + 0.5 = 1.5
    assert abs(v[4] - 1.5) < 1e-6
    # At t=20: V = 1.0 + 1.0 = 2.0
    assert abs(v[6] - 2.0) < 1e-6
    # Interpolated at t=2.5: V = 1.0 + interp(2.5, ...) = 1.0 + 0.125 = 1.125
    assert abs(v[1] - 1.125) < 1e-6


# ---------------------------------------------------------------------------
# Tests: pseudo-batch transform with continuous, mixed, and discrete feeds
# ---------------------------------------------------------------------------

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


def test_pseudobatch_continuous_only_adf():
    """For continuous-only feed, ADF should equal V(t)/V(t0) and increase smoothly."""
    proc = _make_process_continuous_only()
    glucose_ts = proc.reactor_medium.components["glucose"].concentration
    pb = pseudo_batch_transform_timeseries(proc, "glucose", glucose_ts)

    # ADF at measurement times via interpolation
    meas_times = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    adf_at_meas = np.interp(meas_times, pb["adf_times"], pb["adf_values"])

    # V(t) = 1.0 + cumulative feed
    expected_v = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
    expected_adf = expected_v / expected_v[0]
    np.testing.assert_allclose(adf_at_meas, expected_adf, rtol=1e-6)


def test_pseudobatch_continuous_only_cstar():
    """c* should account for continuous dilution and feed contribution."""
    proc = _make_process_continuous_only(glucose_feed_conc=100.0)
    glucose_ts = proc.reactor_medium.components["glucose"].concentration
    pb = pseudo_batch_transform_timeseries(proc, "glucose", glucose_ts)

    # c* should not be just adf*c (feed_term should be nonzero)
    conc = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    adf_at_meas = np.interp(pb["times"], pb["adf_times"], pb["adf_values"])
    feed_at_meas = np.interp(pb["times"], pb["feed_term_times"], pb["feed_term_values"])

    np.testing.assert_allclose(pb["c_star"], adf_at_meas * conc - feed_at_meas, rtol=1e-10)
    # feed_term should be > 0 at later times (glucose is in feed)
    assert feed_at_meas[-1] > 0, "Feed term should be positive for species in feed"


def test_pseudobatch_mixed_continuous_discrete():
    """Mixed continuous + discrete should include both in ADF and feed_term."""
    proc = _make_process_with_discrete()
    # Use biomass (not in feed) to check ADF only
    bio_ts = proc.reactor_medium.components["biomass"].concentration
    pb = pseudo_batch_transform_timeseries(proc, "biomass", bio_ts)

    # ADF should account for both continuous cumulative and discrete bolus
    meas_times = np.array([0.0, 5.0, 10.0, 20.0])
    adf_at_meas = np.interp(meas_times, pb["adf_times"], pb["adf_values"])

    # V(0) = 1.0 (initial)
    # Continuous: cum = [0.0, 0.25, 0.5, 1.0] at [0, 5, 10, 20]
    # Discrete bolus: +0.1 at t=3, +0.1 at t=12
    # V(0)  = 1.0 + 0.0 = 1.0
    # V(5)  = 1.0 + 0.25 + 0.1 = 1.35 (bolus at t=3)
    # V(10) = 1.0 + 0.5 + 0.1 = 1.6  (bolus at t=3 only)
    # V(20) = 1.0 + 1.0 + 0.1 + 0.1 = 2.2 (both boluses)
    expected_v = np.array([1.0, 1.35, 1.6, 2.2])
    expected_adf = expected_v / expected_v[0]
    np.testing.assert_allclose(adf_at_meas, expected_adf, rtol=1e-6)


def test_pseudobatch_discrete_only_regression():
    """Discrete-only transform should match the expected ADF and feed_term."""
    V0 = 1.0
    delta_v = 0.2
    c_feed = 500.0

    proc = _make_process_with_bolus_feed(
        V0=V0, feed_times=[50.0], feed_vols=[delta_v],
        glucose_feed_conc=c_feed,
    )
    glucose_ts = proc.reactor_medium.components["glucose"].concentration
    pb = pseudo_batch_transform_timeseries(proc, "glucose", glucose_ts)

    # Before feed (t=0, t=25): ADF = 1.0, feed_term = 0.0
    adf_before = np.interp(25.0, pb["adf_times"], pb["adf_values"])
    feed_before = np.interp(25.0, pb["feed_term_times"], pb["feed_term_values"])
    assert abs(adf_before - 1.0) < 1e-10
    assert abs(feed_before - 0.0) < 1e-10

    # After feed (t=75): ADF = V_after / V_before = 1.2 / 1.0 = 1.2
    adf_after = np.interp(75.0, pb["adf_times"], pb["adf_values"])
    assert abs(adf_after - 1.2) < 1e-6

    # feed_term after: ADF * c_feed * (delta_v / V_after) = 1.2 * 500 * (0.2/1.2)
    expected_feed = 1.2 * c_feed * (delta_v / 1.2)
    feed_after = np.interp(75.0, pb["feed_term_times"], pb["feed_term_values"])
    assert abs(feed_after - expected_feed) < 1e-4


def test_pseudobatch_continuous_not_cumulative_raises():
    """Non-cumulative continuous series should raise NotImplementedError."""
    rm = ReactorMedium(
        name="medium", density=1.0, density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose", unit="mmol/L",
                concentration=_ts([0.0, 5.0, 10.0], [5.0, 3.0, 1.0]),
                is_intracellular=False,
            ),
        },
    )
    vol = Volume(
        initial_volume=1.0, unit="L",
        volume_changes={
            "bad_feed": FeedVolumeChange(
                name="bad_feed", unit="L",
                is_controlled=True, is_continuous=True,
                feed_medium=_make_feed("bad"),
                # Decreasing values → not cumulative
                values=_ts([0.0, 5.0, 10.0], [0.5, 0.3, 0.1]),
            ),
        },
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="test_bad", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=10.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )
    glucose_ts = proc.reactor_medium.components["glucose"].concentration
    with pytest.raises(NotImplementedError, match="cumulative"):
        pseudo_batch_transform_timeseries(proc, "glucose", glucose_ts)


def test_pseudobatch_volume_positive_validation():
    """Volume <= 0 should raise ValueError."""
    rm = ReactorMedium(
        name="medium", density=1.0, density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose", unit="mmol/L",
                concentration=_ts([0.0, 5.0], [5.0, 3.0]),
                is_intracellular=False,
            ),
        },
    )
    vol = Volume(
        initial_volume=0.5, unit="L",
        volume_changes={
            # Discrete bolus that removes more than initial volume
            "big_sample": FeedVolumeChange(
                name="big_sample", unit="L",
                is_controlled=True, is_continuous=False,
                feed_medium=_make_feed("dummy"),
                values=_ts([2.0], [-1.0]),
            ),
        },
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="test_neg", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=5.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )
    glucose_ts = proc.reactor_medium.components["glucose"].concentration
    with pytest.raises(ValueError, match="positive"):
        pseudo_batch_transform_timeseries(proc, "glucose", glucose_ts)


def test_pseudobatch_continuous_backtransform_roundtrip():
    """Fitting and back-transforming a continuous-only process should give
    reasonable values (no NaN/Inf)."""
    proc = _make_process_continuous_only(glucose_feed_conc=100.0)
    glucose_ts = proc.reactor_medium.components["glucose"].concentration
    rep = fit_state_timeseries_spline_pseudobatch(glucose_ts, proc, "glucose")

    # Evaluate at several times
    for t in [0.0, 5.0, 10.0, 15.0, 20.0]:
        val = evaluate_timeseries_spline_at(rep, t)
        assert np.isfinite(val), f"Non-finite value at t={t}: {val}"
        assert val >= 0, f"Negative concentration at t={t}: {val}"


def test_pseudobatch_metadata_has_interp_key():
    """New metadata should contain 'interp': 'step'."""
    proc = _make_process_with_bolus_feed()
    glucose_ts = proc.reactor_medium.components["glucose"].concentration
    rep = fit_state_timeseries_spline_pseudobatch(glucose_ts, proc, "glucose")
    tr = rep.spline_metadata["transform"]
    assert tr["interp"] == "step"
    assert "adf_times" in tr
    assert "adf_values" in tr
    assert "feed_term_times" in tr
    assert "feed_term_values" in tr


# ---------------------------------------------------------------------------
# Tests: pseudobatch package integration – direct comparison
# ---------------------------------------------------------------------------

from pseudobatch import pseudobatch_transform as pb_transform
from pseudobatch.data_correction import accumulated_dilution_factor as pb_adf
from bpbench.splines import _prepare_pseudobatch_inputs


def test_prepare_pseudobatch_inputs_discrete_only():
    """_prepare_pseudobatch_inputs extracts correct arrays for discrete bolus."""
    proc = _make_process_with_bolus_feed(
        V0=1.0, feed_times=[50.0], feed_vols=[0.2], glucose_feed_conc=500.0,
    )
    glucose_ts = proc.reactor_medium.components["glucose"].concentration
    inputs = _prepare_pseudobatch_inputs(proc, "glucose", glucose_ts)

    np.testing.assert_array_equal(inputs["times"], [0.0, 25.0, 50.0, 75.0, 100.0])
    # Reactor volume: V0=1.0, bolus +0.2 at t=50
    expected_vol = np.array([1.0, 1.0, 1.2, 1.2, 1.2])
    np.testing.assert_allclose(inputs["reactor_volume"], expected_vol)
    # Accumulated feed
    expected_feed = np.array([0.0, 0.0, 0.2, 0.2, 0.2])
    np.testing.assert_allclose(inputs["accumulated_feed"], expected_feed)
    assert inputs["concentration_in_feed"] == 500.0
    np.testing.assert_array_equal(inputs["sample_volume"], np.zeros(5))


def test_prepare_pseudobatch_inputs_continuous_only():
    """_prepare_pseudobatch_inputs extracts correct arrays for continuous feed."""
    proc = _make_process_continuous_only(glucose_feed_conc=100.0)
    glucose_ts = proc.reactor_medium.components["glucose"].concentration
    inputs = _prepare_pseudobatch_inputs(proc, "glucose", glucose_ts)

    expected_vol = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
    np.testing.assert_allclose(inputs["reactor_volume"], expected_vol)
    expected_feed = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(inputs["accumulated_feed"], expected_feed)
    assert inputs["concentration_in_feed"] == 100.0


def test_prepare_pseudobatch_inputs_species_not_in_feed():
    """Species not present in feed should give concentration_in_feed = 0."""
    proc = _make_process_with_bolus_feed()
    proc.reactor_medium.components["biomass"] = ReactorMediumComponent(
        name="biomass", unit="cells/L",
        concentration=_ts([0.0, 25.0, 50.0, 75.0, 100.0],
                          [1.0, 2.0, 4.0, 8.0, 16.0]),
        is_intracellular=False,
    )
    bio_ts = proc.reactor_medium.components["biomass"].concentration
    inputs = _prepare_pseudobatch_inputs(proc, "biomass", bio_ts)
    assert inputs["concentration_in_feed"] == 0.0


def test_pseudobatch_cstar_matches_direct_call_discrete():
    """c_star from pseudo_batch_transform_timeseries matches direct pseudobatch call
    for a discrete bolus feed."""
    proc = _make_process_with_bolus_feed(
        V0=1.0, feed_times=[50.0], feed_vols=[0.2], glucose_feed_conc=500.0,
    )
    glucose_ts = proc.reactor_medium.components["glucose"].concentration
    pb = pseudo_batch_transform_timeseries(proc, "glucose", glucose_ts)

    # Direct pseudobatch call
    inputs = _prepare_pseudobatch_inputs(proc, "glucose", glucose_ts)
    c_star_direct = pb_transform(
        measured_concentration=inputs["measured_conc"],
        reactor_volume=inputs["reactor_volume"],
        accumulated_feed=inputs["accumulated_feed"],
        concentration_in_feed=inputs["concentration_in_feed"],
        sample_volume=inputs["sample_volume"],
    )
    np.testing.assert_allclose(pb["c_star"], c_star_direct, rtol=1e-10)


def test_pseudobatch_cstar_matches_direct_call_continuous():
    """c_star from pseudo_batch_transform_timeseries matches direct pseudobatch call
    for a continuous feed."""
    proc = _make_process_continuous_only(glucose_feed_conc=100.0)
    glucose_ts = proc.reactor_medium.components["glucose"].concentration
    pb = pseudo_batch_transform_timeseries(proc, "glucose", glucose_ts)

    inputs = _prepare_pseudobatch_inputs(proc, "glucose", glucose_ts)
    c_star_direct = pb_transform(
        measured_concentration=inputs["measured_conc"],
        reactor_volume=inputs["reactor_volume"],
        accumulated_feed=inputs["accumulated_feed"],
        concentration_in_feed=inputs["concentration_in_feed"],
        sample_volume=inputs["sample_volume"],
    )
    np.testing.assert_allclose(pb["c_star"], c_star_direct, rtol=1e-10)


def test_pseudobatch_cstar_matches_direct_call_mixed():
    """c_star matches for a process with both continuous and discrete feeds."""
    proc = _make_process_with_discrete()
    bio_ts = proc.reactor_medium.components["biomass"].concentration
    pb = pseudo_batch_transform_timeseries(proc, "biomass", bio_ts)

    inputs = _prepare_pseudobatch_inputs(proc, "biomass", bio_ts)
    c_star_direct = pb_transform(
        measured_concentration=inputs["measured_conc"],
        reactor_volume=inputs["reactor_volume"],
        accumulated_feed=inputs["accumulated_feed"],
        concentration_in_feed=inputs["concentration_in_feed"],
        sample_volume=inputs["sample_volume"],
    )
    np.testing.assert_allclose(pb["c_star"], c_star_direct, rtol=1e-10)


def test_pseudobatch_backtransform_identity_at_measurement_times():
    """Backtransform of c_star at measurement times should recover original concentrations."""
    proc = _make_process_with_bolus_feed(
        V0=1.0, feed_times=[50.0], feed_vols=[0.2], glucose_feed_conc=500.0,
    )
    glucose_ts = proc.reactor_medium.components["glucose"].concentration
    pb = pseudo_batch_transform_timeseries(proc, "glucose", glucose_ts)

    times = pb["times"]
    c_star = pb["c_star"]
    adf_at_meas = np.interp(times, pb["adf_times"], pb["adf_values"])
    feed_at_meas = np.interp(times, pb["feed_term_times"], pb["feed_term_values"])

    # ĉ(t) = (c_star + feed_term) / ADF should recover original concentrations
    conc_recovered = (c_star + feed_at_meas) / adf_at_meas
    np.testing.assert_allclose(
        conc_recovered,
        np.asarray(glucose_ts.values, dtype=float),
        rtol=1e-10,
    )


def test_pseudobatch_with_sample_volume_change():
    """Process with SampleVolumeChange should be handled correctly."""
    from bpbench import SampleVolumeChange

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

    glucose_ts = proc.reactor_medium.components["glucose"].concentration
    inputs = _prepare_pseudobatch_inputs(proc, "glucose", glucose_ts)

    # Sample volume should be positive where sampling occurs (t=5)
    assert abs(inputs["sample_volume"][1] - 0.05) < 1e-6
    assert inputs["sample_volume"][0] == 0.0

    # c_star should be computable without error
    pb = pseudo_batch_transform_timeseries(proc, "glucose", glucose_ts)
    assert len(pb["c_star"]) == 5
    assert np.all(np.isfinite(pb["c_star"]))


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

    glucose_ts = proc.reactor_medium.components["glucose"].concentration
    inputs = _prepare_pseudobatch_inputs(proc, "glucose", glucose_ts)

    # Multiple feeds → 2D accumulated_feed, array concentration_in_feed
    assert inputs["accumulated_feed"].ndim == 2
    assert inputs["accumulated_feed"].shape == (3, 2)
    np.testing.assert_array_equal(inputs["concentration_in_feed"], [200.0, 50.0])

    # c_star should be computable without error
    pb = pseudo_batch_transform_timeseries(proc, "glucose", glucose_ts)
    assert len(pb["c_star"]) == 3
    assert np.all(np.isfinite(pb["c_star"]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
