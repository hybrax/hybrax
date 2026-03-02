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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
