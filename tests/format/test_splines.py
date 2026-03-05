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
    VolumeChange, Volume, ProcessVariable, SplineRepresentation, DiscreteEvents,
)
from bpbench.splines import (
    detect_discrete_events, make_segment_boundaries, split_timeseries,
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
            "continuous_feed": VolumeChange(
                name="continuous_feed", unit="L",
                is_controlled=True, is_continuous=True,
                feed_medium=_make_feed("cont"),
                values=_ts([0., 5., 10., 20.], [0.0, 0.25, 0.5, 1.0]),
            ),
            "bolus": VolumeChange(
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
# detect_discrete_events
# ---------------------------------------------------------------------------

def test_detect_discrete_events():
    proc = _make_process_with_discrete()
    de = detect_discrete_events(proc)
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
            "feed": VolumeChange(
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
    de = detect_discrete_events(proc)
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
            "bolus_feed": VolumeChange(
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
