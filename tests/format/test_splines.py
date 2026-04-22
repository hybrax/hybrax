"""
Tests for bp_format.splines: spline fitting, serialization, evaluation,
and pseudobatch transform pipeline.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import tempfile
from pathlib import Path

from bp_format import (
    BioProcess,
    BioProcessMetadata,
    TimeAxis,
    TimeSeries,
    StaticVariable,
    ReactorMedium,
    ReactorMediumComponent,
    FeedMedium,
    FeedMediumComponent,
    FeedVolumeChange,
    SampleVolumeChange,
    Volume,
    ProcessVariable,
    Interpolator,
    DiscreteEvents,
)
from bp_format.splines import (
    detect_discrete_state_events,
    make_segment_boundaries,
    split_timeseries,
    choose_spline_kind,
    fit_timeseries_spline,
    build_interpax_spline,
    evaluate_spline_at,
    build_pseudobatch_inputs,
    build_splines,
    evaluate_real_concentration,
    to_interpolator,
    build_backtransform_spline,
    build_batched_conc_splines,
    BacktransformSpline,
    evaluate_left_continuous_step,
)
from bp_format.serialization import (
    save_dataset,
    save_dataset_json,
    load_dataset,
    load_dataset_json,
)
from bp_format import BenchmarkDataset, CaseStudy


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
                is_intracellular=False,
            ),
        },
    )
    vol = Volume(
        initial_volume=1.0,
        unit="L",
        volume_changes={
            "continuous_feed": FeedVolumeChange(
                name="continuous_feed",
                unit="L",
                is_controlled=True,
                is_continuous=True,
                feed_medium=_make_feed("cont"),
                values=_ts([0.0, 5.0, 10.0, 20.0], [0.0, 0.25, 0.5, 1.0]),
            ),
            "bolus": FeedVolumeChange(
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
            "feed": FeedVolumeChange(
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
    ts = _ts([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 1.0, 4.0, 9.0, 16.0, 25.0])
    rep = fit_timeseries_spline(ts)
    assert isinstance(rep, Interpolator)
    assert rep.kind == "interpax_cubic"
    assert rep.n_segments == 1
    assert int(rep.n[0]) == 6
    assert rep.x.shape == (16, 128)  # default padding


def test_fit_with_segmentation():
    ts = _ts([0.0, 1.0, 2.0, 5.0, 6.0, 7.0], [0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    boundaries = np.array([0.0, 3.0, 7.0])
    rep = fit_timeseries_spline(ts, boundaries=boundaries)
    assert rep.n_segments == 2
    assert int(rep.n[0]) >= 2
    assert int(rep.n[1]) >= 2


def test_fit_single_point_segment():
    """A segment with only 1 point should not crash."""
    ts = _ts([0.0, 5.0, 10.0], [1.0, 2.0, 3.0])
    boundaries = np.array([0.0, 2.0, 10.0])
    rep = fit_timeseries_spline(ts, boundaries=boundaries)
    assert rep.n_segments == 2


def test_fit_roundtrip_accuracy():
    """Fitted spline should pass through original (cubic interp) points."""
    ts = _ts([0.0, 2.0, 4.0, 6.0, 8.0], [0.0, 1.0, 0.0, 1.0, 0.0])
    rep = fit_timeseries_spline(ts)
    for t_val, expected in zip([0.0, 2.0, 4.0, 6.0, 8.0], [0.0, 1.0, 0.0, 1.0, 0.0]):
        result = evaluate_spline_at(rep, t_val)
        assert abs(result - expected) < 1e-4, (
            f"At t={t_val}: got {result}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# build_interpax_spline / evaluate_spline_at
# ---------------------------------------------------------------------------


def test_build_interpax_spline():
    ts = _ts([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 4.0, 9.0])
    rep = fit_timeseries_spline(ts)
    splines, boundaries = build_interpax_spline(rep)
    assert len(splines) == 1
    assert len(boundaries) == 2
    val = float(splines[0](jnp.array(1.0)))
    assert abs(val - 1.0) < 1e-4


def test_evaluate_spline_at_multi_segment():
    ts = _ts([0.0, 1.0, 2.0, 5.0, 6.0, 7.0], [0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
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
    ts = _ts([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 0.5, 1.5, 3.0, 5.0])
    rep = fit_timeseries_spline(ts)

    pv = ProcessVariable(
        name="test_var",
        unit="g/L",
        is_controlled=True,
        values=ts,
        interpolator=rep,
    )
    rm = ReactorMedium(name="m", density=1.0, density_unit="kg/L")
    proc = BioProcess(
        metadata=BioProcessMetadata(name="p", process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=4.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=rm,
        process_variables={"test_var": pv},
    )
    cs = CaseStudy(
        case_id="cs", organism="E. coli", citation="test", processes={"p": proc}
    )
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
    cs = CaseStudy(
        case_id="cs", organism="E. coli", citation="test", processes={"p": proc}
    )
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
    cs = CaseStudy(
        case_id="cs", organism="E. coli", citation="test", processes={"p": proc}
    )
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
        name="bolus_feed_medium",
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
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=_ts(glucose_times, glucose_values),
                is_intracellular=False,
            ),
        },
    )
    vol = Volume(
        initial_volume=V0,
        unit="L",
        volume_changes={
            "bolus_feed": FeedVolumeChange(
                name="bolus_feed",
                unit="L",
                is_controlled=True,
                is_continuous=False,
                feed_medium=feed_medium,
                values=_ts(feed_times, feed_vols),
            ),
        },
    )
    return BioProcess(
        metadata=BioProcessMetadata(name="test_pb", process_type="fed_batch"),
        time_axis=TimeAxis(
            unit="h", start=0.0, end=max(glucose_times), time_reference="inoculation"
        ),
        volume=vol,
        reactor_medium=rm,
    )


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
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=_ts(
                    [0.0, 5.0, 10.0, 15.0, 20.0],
                    [10.0, 8.0, 6.0, 4.0, 2.0],
                ),
                is_intracellular=False,
            ),
        },
    )
    vol = Volume(
        initial_volume=1.0,
        unit="L",
        volume_changes={
            "cont_feed": FeedVolumeChange(
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


# ---------------------------------------------------------------------------
# Pseudobatch pipeline: build_pseudobatch_inputs
# ---------------------------------------------------------------------------


def test_pseudobatch_inputs_discrete():
    """build_pseudobatch_inputs produces valid outputs for discrete bolus feed."""
    proc = _make_process_with_bolus_feed(
        V0=1.0,
        feed_times=[50.0],
        feed_vols=[0.2],
        glucose_feed_conc=500.0,
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
        name="biomass",
        unit="cells/L",
        concentration=_ts([0.0, 25.0, 50.0, 75.0, 100.0], [1.0, 2.0, 4.0, 8.0, 16.0]),
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
        V0=1.0,
        feed_times=[50.0],
        feed_vols=[0.2],
        glucose_feed_conc=500.0,
        glucose_times=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0],
        glucose_values=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 7.0, 6.0, 5.0, 4.0],
    )
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")

    rep = to_interpolator(inputs, splines, "glucose")
    assert rep.interpolator_metadata is not None
    assert rep.interpolator_metadata["transform"]["name"] == "pseudo_batch"
    assert rep.interpolator_metadata["transform"]["species"] == "glucose"
    assert rep.interpolator_metadata["transform"]["feed_corr_interp"] == (
        "linear_plus_step"
    )

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
        V0=1.0,
        feed_times=[50.0],
        feed_vols=[0.2],
        glucose_feed_conc=500.0,
        glucose_times=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0],
        glucose_values=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 7.0, 6.0, 5.0, 4.0],
    )
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_interpolator(inputs, splines, "glucose")

    bt = build_backtransform_spline(rep)

    t_b = 50.0
    post_probe = 5e-4
    pre_probe = 5e-4  # safely away from event edge
    val_before = float(bt(jnp.array(t_b - pre_probe)))
    val_at = float(bt(jnp.array(t_b)))
    val_after = float(bt(jnp.array(t_b + post_probe)))

    assert val_at == pytest.approx(val_before, abs=2e-2)

    jump = abs(val_after - val_before)
    assert jump > 0.1, (
        f"Expected a jump right after t_feed=50.0, got val_at={val_at}, "
        f"val_after={val_after}, jump={jump}"
    )


def test_batched_backtransform_preserves_bolus_jump():
    """Batched backtransform must preserve discrete feed jumps."""
    proc = _make_process_with_bolus_feed(
        V0=1.0,
        feed_times=[50.0],
        feed_vols=[0.2],
        glucose_feed_conc=500.0,
        glucose_times=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0],
        glucose_values=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 7.0, 6.0, 5.0, 4.0],
    )
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_interpolator(inputs, splines, "glucose")
    bt = build_backtransform_spline(rep)
    batched = build_batched_conc_splines(
        conc_splines={"glucose": bt},
        species_names=["glucose"],
        t_start=0.0,
        t_end=100.0,
    )

    t_b = 50.0
    delta = 5e-5
    val_at = float(batched(jnp.array(t_b))[0])
    val_post = float(batched(jnp.array(t_b + delta))[0])
    ref_at = float(bt(jnp.array(t_b)))
    ref_post = float(bt(jnp.array(t_b + delta)))

    assert abs(val_at - ref_at) < 2.0
    assert abs(val_post - ref_post) < 2.0
    assert abs(val_post - val_at) > 0.1


def test_backtransform_rejects_legacy_linear_feed_corr():
    """Legacy linear feed_corr metadata is intentionally unsupported."""
    proc = _make_process_with_bolus_feed(
        V0=1.0,
        feed_times=[50.0],
        feed_vols=[0.2],
        glucose_feed_conc=500.0,
        glucose_times=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0],
        glucose_values=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 7.0, 6.0, 5.0, 4.0],
    )
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_interpolator(inputs, splines, "glucose")

    tr = rep.interpolator_metadata["transform"]
    tr["feed_corr_interp"] = "linear"
    tr.pop("feed_corr_base_times", None)
    tr.pop("feed_corr_base_values", None)
    tr.pop("feed_corr_jump_times", None)
    tr.pop("feed_corr_jump_values", None)

    with pytest.raises(ValueError, match="feed_corr_interp='linear'"):
        _ = build_backtransform_spline(rep)


def test_backtransform_same_time_sampling_and_bolus_is_pre_event_at_tb():
    """Contract: pre-event at t_b, with upward jump only for t > t_b."""
    feed_medium = FeedMedium(
        name="feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=StaticVariable(value=300.0),
                is_controlled=True,
            ),
        },
    )
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=_ts([0.0, 5.0, 10.0, 15.0], [10.0, 9.0, 7.0, 6.0]),
                is_intracellular=False,
            ),
        },
    )
    vol = Volume(
        initial_volume=1.0,
        unit="L",
        volume_changes={
            "sampling": SampleVolumeChange(
                name="sampling",
                unit="L",
                is_controlled=True,
                is_continuous=False,
                values=_ts([10.0], [-0.2]),
            ),
            "bolus": FeedVolumeChange(
                name="bolus",
                unit="L",
                is_controlled=True,
                is_continuous=False,
                feed_medium=feed_medium,
                values=_ts([10.0], [0.1]),
            ),
        },
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="same_time", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=15.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )

    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_interpolator(inputs, splines, "glucose")
    bt = build_backtransform_spline(rep)

    t_b = 10.0
    post_probe = 5e-4
    pre_probe = 5e-4
    val_pre = float(bt(jnp.array(t_b - pre_probe)))
    val_at = float(bt(jnp.array(t_b)))
    val_post = float(bt(jnp.array(t_b + post_probe)))

    # sample first: V=1.0->0.8, then bolus +0.1 with C_feed=300 at same timestamp
    # expected post-event concentration if t_b is treated as pre-event:
    # C+ = (7.0*0.8 + 300.0*0.1) / 0.9
    expected_post = (7.0 * 0.8 + 300.0 * 0.1) / 0.9

    # Characterization contract at exact timestamp.
    assert val_at == pytest.approx(val_pre, abs=2e-2)
    # Directional jump: feed is much richer than broth, so jump should be upward.
    assert val_post > val_at
    # Post-event value should follow sample-then-bolus mass balance.
    assert val_post == pytest.approx(expected_post, rel=1e-2, abs=1e-2)


def test_backtransform_bolus_at_t_start_no_crash():
    """Left-continuous contract at t_start with a discrete bolus event.

    For a bolus at `t_start`, the backtransform should return pre-event at
    `t_start` and post-event for `t > t_start`.
    """
    proc = _make_process_with_bolus_feed(
        V0=1.0,
        feed_times=[0.0],
        feed_vols=[0.2],
        glucose_feed_conc=500.0,
        glucose_times=[0.0, 5.0, 10.0, 15.0],
        glucose_values=[10.0, 9.0, 8.0, 7.0],
    )
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_interpolator(inputs, splines, "glucose")
    bt = build_backtransform_spline(rep)

    vals = [float(bt(jnp.array(t))) for t in [0.0, 5e-4, 1.0]]
    assert np.all(np.isfinite(vals))
    assert np.all(np.array(vals) >= 0.0)

    # Pre-event at exact boundary timestamp.
    assert vals[0] == pytest.approx(10.0, abs=1e-4)

    # Post-event for t > t_start: V = 1.0 + 0.2, C_feed = 500.
    expected_post = (10.0 * 1.0 + 500.0 * 0.2) / 1.2
    assert vals[1] > vals[0]
    assert vals[1] == pytest.approx(expected_post, rel=1e-2, abs=1e-2)


def test_backtransform_bolus_at_t_end_no_crash():
    """Boundary-limited check at t_end, not a full jump-contract test.

    `build_splines` skips explicit boundary-event knot augmentation at `t_end`.
    So for a bolus at `t_end`, this test only verifies:
    1) backtransform remains finite/non-negative, and
    2) `bt(t_end)` equals the final raw measured value (pre-event sample).
    """
    proc = _make_process_with_bolus_feed(
        V0=1.0,
        feed_times=[15.0],
        feed_vols=[0.2],
        glucose_feed_conc=500.0,
        glucose_times=[0.0, 5.0, 10.0, 15.0],
        glucose_values=[10.0, 9.0, 8.0, 7.0],
    )
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_interpolator(inputs, splines, "glucose")
    bt = build_backtransform_spline(rep)

    vals = [float(bt(jnp.array(t))) for t in [14.999, 15.0]]
    assert np.all(np.isfinite(vals))
    assert np.all(np.array(vals) >= 0.0)
    # Known limitation: boundary bolus correction is skipped in build_splines.
    # Therefore bt(t_end) stays at the final raw measurement (pre-event).
    assert vals[1] == pytest.approx(7.0, abs=1e-4)


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

    cs = CaseStudy(case_id="cs", organism="CHO", citation="test", processes={"p": proc})
    ds = BenchmarkDataset(metadata={"name": "test"}, case_studies={"cs": cs})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pb.json"
        save_dataset_json(ds, path)
        payload = path.read_text()
        loaded = load_dataset_json(path)

    assert '"interpolator"' in payload
    loaded_comp = (
        loaded.case_studies["cs"].processes["p"].reactor_medium.components["glucose"]
    )
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

    # ADF step behaviour should survive serialization. ADF is stored as the
    # dense grid and evaluated via jnp.interp; the transition across an
    # event occupies the ε-pair window [t_b - _EPS, t_b + _EPS] so we probe
    # strictly outside that window.
    post_delta = 2e-4  # > _EPS = 1e-4
    tr_orig = rep.interpolator_metadata["transform"]
    tr_loaded = loaded_comp.interpolator.interpolator_metadata["transform"]
    orig_adf_t = jnp.asarray(tr_orig["adf_times"], dtype=float)
    orig_adf_v = jnp.asarray(tr_orig["adf_values"], dtype=float)
    loaded_adf_t = jnp.asarray(tr_loaded["adf_times"], dtype=float)
    loaded_adf_v = jnp.asarray(tr_loaded["adf_values"], dtype=float)
    assert orig_adf_t.size > 1
    # First bolus event time: the first dense knot where ADF changes.
    adf_diff = jnp.diff(orig_adf_v)
    jump_idx = int(jnp.argmax(jnp.abs(adf_diff))) + 1
    t_b = float(orig_adf_t[jump_idx])

    orig_pre = float(jnp.interp(jnp.array(t_b - post_delta), orig_adf_t, orig_adf_v))
    orig_post = float(jnp.interp(jnp.array(t_b + post_delta), orig_adf_t, orig_adf_v))
    loaded_pre = float(jnp.interp(jnp.array(t_b - post_delta), loaded_adf_t, loaded_adf_v))
    loaded_post = float(jnp.interp(jnp.array(t_b + post_delta), loaded_adf_t, loaded_adf_v))
    assert orig_post > orig_pre
    assert loaded_post > loaded_pre
    assert loaded_pre == pytest.approx(orig_pre, abs=1e-12)
    assert loaded_post == pytest.approx(orig_post, abs=1e-12)


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
        case_studies={
            "cs": CaseStudy(
                case_id="cs", organism="E. coli", citation="test", processes={"p": proc}
            )
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "linear.json"
        save_dataset_json(ds, path)
        loaded = load_dataset_json(path)

    loaded_interp = (
        loaded.case_studies["cs"]
        .processes["p"]
        .process_variables["linear_var"]
        .interpolator
    )
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
        case_studies={
            "cs": CaseStudy(
                case_id="cs", organism="E. coli", citation="test", processes={"p": proc}
            )
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ppoly.json"
        save_dataset_json(ds, path)
        loaded = load_dataset_json(path)

    loaded_interp = (
        loaded.case_studies["cs"]
        .processes["p"]
        .reactor_medium.components["glucose"]
        .interpolator
    )
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
        case_studies={
            "cs": CaseStudy(
                case_id="cs", organism="E. coli", citation="test", processes={"p": proc}
            )
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "hybrid"
        save_dataset(ds, path)
        loaded = load_dataset(path)

    loaded_interp = (
        loaded.case_studies["cs"]
        .processes["p"]
        .process_variables["linear_var"]
        .interpolator
    )
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
        name="feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=StaticVariable(value=100.0),
                is_controlled=True,
            ),
        },
    )
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=_ts(
                    [0.0, 5.0, 10.0, 15.0, 20.0],
                    [10.0, 8.0, 6.0, 5.0, 4.0],
                ),
                is_intracellular=False,
            ),
        },
    )
    vol = Volume(
        initial_volume=1.0,
        unit="L",
        volume_changes={
            "feed": FeedVolumeChange(
                name="feed",
                unit="L",
                is_controlled=True,
                is_continuous=False,
                feed_medium=feed_medium,
                values=_ts([10.0], [0.2]),
            ),
            "sample": SampleVolumeChange(
                name="sample",
                unit="L",
                is_controlled=True,
                is_continuous=False,
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
        name="feed1",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=StaticVariable(value=200.0),
                is_controlled=True,
            ),
        },
    )
    feed_medium_2 = FeedMedium(
        name="feed2",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=StaticVariable(value=50.0),
                is_controlled=True,
            ),
        },
    )
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=_ts([0.0, 10.0, 20.0], [10.0, 7.0, 5.0]),
                is_intracellular=False,
            ),
        },
    )
    vol = Volume(
        initial_volume=1.0,
        unit="L",
        volume_changes={
            "feed1": FeedVolumeChange(
                name="feed1",
                unit="L",
                is_controlled=True,
                is_continuous=False,
                feed_medium=feed_medium_1,
                values=_ts([5.0], [0.1]),
            ),
            "feed2": FeedVolumeChange(
                name="feed2",
                unit="L",
                is_controlled=True,
                is_continuous=False,
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
        V0=1.0,
        feed_times=[50.0],
        feed_vols=[0.2],
        glucose_feed_conc=500.0,
        glucose_times=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0],
        glucose_values=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 7.0, 6.0, 5.0, 4.0],
    )
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_interpolator(inputs, splines, "glucose")
    bt = build_backtransform_spline(rep)

    jit_fn = eqx.filter_jit(bt)
    val_mid = jit_fn(jnp.array(25.0))
    assert np.isfinite(float(val_mid))
    assert float(val_mid) > 0

    t_b = 50.0
    post_probe = 5e-4
    pre_probe = 5e-4
    val_pre = float(jit_fn(jnp.array(t_b - pre_probe)))
    val_at = float(jit_fn(jnp.array(t_b)))
    val_post = float(jit_fn(jnp.array(t_b + post_probe)))
    assert val_at == pytest.approx(val_pre, abs=2e-2)
    assert val_post - val_at > 0.01


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


# ---------------------------------------------------------------------------
# PCHIP fallback for spline oscillation
# ---------------------------------------------------------------------------


def _make_process_sharp_profile():
    """Process with a species that has a sharp 0→peak→0 profile (triggers PCHIP)."""
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "acetate": ReactorMediumComponent(
                name="acetate",
                unit="g/L",
                concentration=_ts(
                    [0.0, 2.0, 3.0, 4.5, 6.0, 7.5, 9.0, 10.0],
                    [0.0, 0.15, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                is_intracellular=False,
            ),
        },
    )
    vol = Volume(initial_volume=1.0, unit="L", volume_changes={})
    return BioProcess(
        metadata=BioProcessMetadata(name="test_pchip", process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=10.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )


def _make_process_smooth_profile():
    """Process with a smooth monotone species (no PCHIP needed)."""
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=_ts(
                    [0.0, 2.0, 5.0, 8.0, 10.0],
                    [0.1, 0.5, 2.0, 5.0, 8.0],
                ),
                is_intracellular=False,
            ),
        },
    )
    vol = Volume(initial_volume=1.0, unit="L", volume_changes={})
    return BioProcess(
        metadata=BioProcessMetadata(name="test_smooth", process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=10.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )


class TestPchipFallback:
    def test_pchip_fallback_triggered(self):
        """Sharp 0→peak→0 profile should trigger PCHIP fallback."""
        proc = _make_process_sharp_profile()
        inputs = build_pseudobatch_inputs(proc, "acetate")
        splines = build_splines(inputs, proc, "acetate")
        assert inputs.get("cstar_interp") == "pchip"

        # Verify the spline stays non-negative on a dense grid
        t_dense = jnp.linspace(0.0, 10.0, 500)
        c_dense = jax.vmap(splines["spline_cstar"])(t_dense)
        assert float(jnp.min(c_dense)) >= -1e-8

    def test_pchip_fallback_not_triggered(self):
        """Smooth monotone profile should NOT trigger PCHIP fallback."""
        proc = _make_process_smooth_profile()
        inputs = build_pseudobatch_inputs(proc, "biomass")
        splines = build_splines(inputs, proc, "biomass")
        assert inputs.get("cstar_interp", "cubic") == "cubic"

    def test_pchip_backtransform_roundtrip(self):
        """PCHIP spline serialized and rebuilt recovers measurement values."""
        proc = _make_process_sharp_profile()
        inputs = build_pseudobatch_inputs(proc, "acetate")
        splines = build_splines(inputs, proc, "acetate")
        rep = to_interpolator(inputs, splines, "acetate")

        assert rep.interpolator_metadata["transform"]["cstar_interp"] == "pchip"

        bt = build_backtransform_spline(rep)
        # Evaluate at measurement times
        meas_t = jnp.array(inputs["meas_times"])
        meas_c = jnp.array(inputs["meas_conc"])
        for t, c_expected in zip(meas_t, meas_c):
            c_bt = float(bt(t))
            assert abs(c_bt - float(c_expected)) < 1e-4, (
                f"At t={float(t):.2f}: backtransform={c_bt:.6f}, "
                f"expected={float(c_expected):.6f}"
            )

    def test_pchip_nonnegative_concentration(self):
        """BacktransformSpline from PCHIP should not go significantly negative."""
        proc = _make_process_sharp_profile()
        inputs = build_pseudobatch_inputs(proc, "acetate")
        splines = build_splines(inputs, proc, "acetate")
        rep = to_interpolator(inputs, splines, "acetate")
        bt = build_backtransform_spline(rep)

        t_dense = jnp.linspace(0.0, 10.0, 500)
        c_dense = jnp.array([float(bt(t)) for t in t_dense])
        assert float(jnp.min(c_dense)) >= -1e-6, (
            f"BacktransformSpline went negative: min={float(jnp.min(c_dense)):.6f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
