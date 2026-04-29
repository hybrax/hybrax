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
    build_pseudobatch_transform,
    build_splines,
    evaluate_real_concentration,
    to_timeseries,
    build_backtransform_spline,
    build_batched_conc_splines,
    BacktransformSpline,
)
from bp_format.serialization import (
    save_dataset_json,
    load_dataset_json,
)
from bp_format import BenchmarkDataset, CaseStudy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(t, v):
    return TimeSeries(times=jnp.array(t, dtype=float), values=jnp.array(v, dtype=float))


def _build_single_species_transform(process, species_name="glucose"):
    return build_pseudobatch_transform(process, [species_name])


def _build_single_species_backtransform(process, species_name="glucose"):
    transform = _build_single_species_transform(process, species_name)
    return build_backtransform_spline(transform, species_name)


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
    assert isinstance(rep, TimeSeries)
    assert rep.breaks is not None
    assert rep.coeffs is not None
    assert rep.metadata["kind"] == "interpax_cubic"
    assert rep.metadata["actual_segments"] == 1
    assert rep.times.shape == (6,)


def test_fit_with_segmentation():
    ts = _ts([0.0, 1.0, 2.0, 5.0, 6.0, 7.0], [0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    boundaries = np.array([0.0, 3.0, 7.0])
    rep = fit_timeseries_spline(ts, boundaries=boundaries)
    assert rep.metadata["actual_segments"] == 2
    np.testing.assert_allclose(rep.metadata["segment_boundaries"], boundaries)
    assert len(rep.segment_start_piece_idx) == 2


def test_fit_single_point_segment():
    """A segment with only 1 point should not crash."""
    ts = _ts([0.0, 5.0, 10.0], [1.0, 2.0, 3.0])
    boundaries = np.array([0.0, 2.0, 10.0])
    rep = fit_timeseries_spline(ts, boundaries=boundaries)
    assert rep.metadata["actual_segments"] == 2


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
    cs = CaseStudy(
        case_id="cs", organism="E. coli", citation="test", processes={"p": proc}
    )
    ds = BenchmarkDataset(metadata={"name": "test"}, case_studies={"cs": cs})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_dataset_json(ds, path)
        payload = path.read_text()
        loaded = load_dataset_json(path)

    assert '"interpolator"' not in payload
    assert '"breaks"' in payload
    assert '"coeffs"' in payload
    loaded_pv = loaded.case_studies["cs"].processes["p"].process_variables["test_var"]
    assert isinstance(loaded_pv.values, TimeSeries)
    np.testing.assert_allclose(loaded_pv.values.breaks, rep.breaks)
    np.testing.assert_allclose(loaded_pv.values.coeffs, rep.coeffs)
    np.testing.assert_allclose(
        loaded_pv.values.segment_start_piece_idx,
        rep.segment_start_piece_idx,
    )

    for t_val in [0.0, 1.0, 2.0, 3.0, 4.0]:
        orig = evaluate_spline_at(rep, t_val)
        loaded_val = evaluate_spline_at(loaded_pv.values, t_val)
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
    cs = CaseStudy(
        case_id="cs", organism="E. coli", citation="test", processes={"p": proc}
    )
    ds = BenchmarkDataset(metadata={"name": "test"}, case_studies={"cs": cs})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_dataset_json(ds, path)
        payload = path.read_text()
        loaded = load_dataset_json(path)

    loaded_pv = loaded.case_studies["cs"].processes["p"].process_variables["x"]
    assert '"interpolator"' not in payload
    np.testing.assert_allclose(loaded_pv.values.times, pv.values.times)
    np.testing.assert_allclose(loaded_pv.values.values, pv.values.values)


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
# Pseudobatch pipeline: TimeSeries carrier + evaluate roundtrip
# ---------------------------------------------------------------------------


def test_interpolator_roundtrip_bolus():
    """to_timeseries -> build_backtransform_spline
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

    rep = to_timeseries(inputs, splines, "glucose")
    assert rep.metadata is not None
    tr = rep.metadata["transform"]
    assert tr["name"] == "pseudo_batch"
    assert tr["species"] == "glucose"
    assert tr["cstar_interp"] in {"cubic", "pchip"}
    assert "series" not in tr

    bt = _build_single_species_backtransform(proc, "glucose")
    assert isinstance(bt, BacktransformSpline)

    t_eval = np.linspace(0.0, 100.0, 50)
    direct = evaluate_real_concentration(t_eval, splines)
    from_rep = np.array([float(bt(jnp.array(t))) for t in t_eval])

    np.testing.assert_allclose(from_rep, direct, rtol=1e-4, atol=1e-6)


def test_interpolator_roundtrip_continuous():
    """to_timeseries -> build_backtransform_spline
    matches evaluate_real_concentration for a continuous feed process."""
    proc = _make_process_continuous_only(glucose_feed_conc=100.0)
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")

    rep = to_timeseries(inputs, splines, "glucose")
    assert "series" not in rep.metadata["transform"]

    bt = _build_single_species_backtransform(proc, "glucose")

    t_eval = np.linspace(0.0, 20.0, 30)
    direct = evaluate_real_concentration(t_eval, splines)
    from_rep = np.array([float(bt(jnp.array(t))) for t in t_eval])

    np.testing.assert_allclose(from_rep, direct, rtol=1e-4, atol=1e-6)


def test_interpolator_scalar():
    """BacktransformSpline works for scalar evaluation."""
    proc = _make_process_with_bolus_feed()
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    to_timeseries(inputs, splines, "glucose")

    bt = _build_single_species_backtransform(proc, "glucose")
    val = float(bt(jnp.array(25.0)))
    assert np.isfinite(val)
    assert val > 0


def test_near_constant_nonzero_species_uses_constant_shortcut():
    proc = _make_process_continuous_only(glucose_feed_conc=100.0)
    proc.reactor_medium.components["glucose"].concentration = _ts(
        [0.0, 5.0, 10.0, 15.0, 20.0],
        [2.0, 2.0 + 1e-9, 2.0 - 1e-9, 2.0 + 1e-9, 2.0],
    )
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_timeseries(inputs, splines, "glucose")

    tr = rep.metadata["transform"]
    assert tr["is_constant"] is True
    assert tr["constant_value"] == pytest.approx(2.0, abs=1e-8)

    bt = _build_single_species_backtransform(proc, "glucose")
    vals = np.array([float(bt(jnp.array(t))) for t in np.linspace(0.0, 20.0, 7)])
    np.testing.assert_allclose(vals, np.full_like(vals, 2.0), atol=1e-8)


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
    to_timeseries(inputs, splines, "glucose")

    bt = _build_single_species_backtransform(proc, "glucose")

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
    to_timeseries(inputs, splines, "glucose")
    transform = _build_single_species_transform(proc, "glucose")
    bt = build_backtransform_spline(transform, "glucose")
    batched = build_batched_conc_splines(
        transform,
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


def test_batched_backtransform_matches_scalar_continuous_feed():
    proc = _make_process_continuous_only(glucose_feed_conc=100.0)
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    to_timeseries(inputs, splines, "glucose")
    transform = _build_single_species_transform(proc, "glucose")
    bt = build_backtransform_spline(transform, "glucose")
    batched = build_batched_conc_splines(
        transform,
        species_names=["glucose"],
        t_start=0.0,
        t_end=20.0,
    )

    t_eval = jnp.linspace(0.0, 20.0, 81)
    scalar = jnp.asarray([bt(t) for t in t_eval])
    vectorized = jax.vmap(lambda t: batched(t)[0])(t_eval)
    np.testing.assert_allclose(vectorized, scalar, rtol=1e-5, atol=1e-5)


def test_batched_backtransform_derivative_matches_scalar_continuous_feed():
    proc = _make_process_continuous_only(glucose_feed_conc=100.0)
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    to_timeseries(inputs, splines, "glucose")
    transform = _build_single_species_transform(proc, "glucose")
    bt = build_backtransform_spline(transform, "glucose")
    batched = build_batched_conc_splines(
        transform,
        species_names=["glucose"],
        t_start=0.0,
        t_end=20.0,
    )

    t_eval = jnp.linspace(0.0, 20.0, 81)
    scalar = jnp.asarray([bt.derivative()(t) for t in t_eval])
    vectorized = jax.vmap(lambda t: batched.eval_derivative(t)[0])(t_eval)
    np.testing.assert_allclose(vectorized, scalar, rtol=1e-4, atol=1e-4)


def test_batched_backtransform_preserves_start_boundary_bolus():
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
    to_timeseries(inputs, splines, "glucose")
    transform = _build_single_species_transform(proc, "glucose")
    bt = build_backtransform_spline(transform, "glucose")
    batched = build_batched_conc_splines(
        transform,
        species_names=["glucose"],
        t_start=0.0,
        t_end=15.0,
    )

    for t in [0.0, 5e-4, 1.0]:
        assert float(batched(jnp.array(t))[0]) == pytest.approx(
            float(bt(jnp.array(t))),
            rel=1e-5,
            abs=1e-5,
        )


def test_batched_backtransform_sums_same_time_bolus_jumps():
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
    proc = _make_process_with_bolus_feed(
        V0=1.0,
        feed_times=[],
        feed_vols=[],
        glucose_feed_conc=0.0,
        glucose_times=[0.0, 5.0, 10.0],
        glucose_values=[10.0, 9.0, 8.0],
    )
    proc.volume.volume_changes = {
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
            values=_ts([5.0], [0.2]),
        ),
    }
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    to_timeseries(inputs, splines, "glucose")
    transform = _build_single_species_transform(proc, "glucose")
    bt = build_backtransform_spline(transform, "glucose")
    batched = build_batched_conc_splines(
        transform,
        species_names=["glucose"],
        t_start=0.0,
        t_end=10.0,
    )

    for t in [5.0, 5.0005]:
        assert float(batched(jnp.array(t))[0]) == pytest.approx(
            float(bt(jnp.array(t))),
            rel=1e-5,
            abs=1e-5,
        )


def test_evaluate_real_concentration_vector_path_is_jittable():
    proc = _make_process_with_bolus_feed()
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    t_eval = jnp.asarray([0.0, 25.0, 50.0, 75.0])

    expected = evaluate_real_concentration(t_eval, splines)
    got = jax.jit(lambda t: evaluate_real_concentration(t, splines))(t_eval)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-10)


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
    transform = _build_single_species_transform(proc, "glucose")
    transform.species["glucose"].feed_corr_ts.metadata["interp"] = "linear"

    with pytest.raises(ValueError, match="feed_corr_interp='linear'"):
        _ = build_backtransform_spline(transform, "glucose")


def test_backtransform_requires_species_in_bundle():
    proc = _make_process_with_bolus_feed()
    transform = _build_single_species_transform(proc, "glucose")

    with pytest.raises(KeyError):
        _ = build_backtransform_spline(transform, "biomass")


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
    to_timeseries(inputs, splines, "glucose")
    bt = _build_single_species_backtransform(proc, "glucose")

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
    to_timeseries(inputs, splines, "glucose")
    bt = _build_single_species_backtransform(proc, "glucose")

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
    to_timeseries(inputs, splines, "glucose")
    bt = _build_single_species_backtransform(proc, "glucose")

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
    to_timeseries(inputs, splines, "glucose")

    bt = _build_single_species_backtransform(proc, "glucose")
    for t in [0.0, 5.0, 10.0, 15.0, 20.0]:
        val = float(bt(jnp.array(t)))
        assert np.isfinite(val), f"Non-finite value at t={t}: {val}"
        assert val >= 0, f"Negative concentration at t={t}: {val}"


def test_build_pseudobatch_transform_rejects_already_transformed_carrier():
    """Builder must not treat a c* carrier as raw measured concentration."""
    proc = _make_process_with_bolus_feed()
    transform = _build_single_species_transform(proc, "glucose")
    proc.pseudobatch_transform = transform
    proc.reactor_medium.components["glucose"].concentration = transform.species[
        "glucose"
    ].c_star_ts

    with pytest.raises(ValueError, match="already carries pseudobatch"):
        build_pseudobatch_transform(proc, ["glucose"])

    with pytest.raises(ValueError, match="already carries pseudobatch"):
        build_pseudobatch_transform(proc)


# ---------------------------------------------------------------------------
# TimeSeries JSON serialization with backtransform metadata
# ---------------------------------------------------------------------------


def test_pseudobatch_spline_json_roundtrip():
    """Pseudobatch bundle and lightweight c* metadata round-trip through JSON."""
    proc = _make_process_with_bolus_feed()
    transform = _build_single_species_transform(proc, "glucose")
    rep = transform.species["glucose"].c_star_ts
    proc.pseudobatch_transform = transform
    proc.reactor_medium.components["glucose"].concentration = rep

    cs = CaseStudy(case_id="cs", organism="CHO", citation="test", processes={"p": proc})
    ds = BenchmarkDataset(metadata={"name": "test"}, case_studies={"cs": cs})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_pb.json"
        save_dataset_json(ds, path)
        payload = path.read_text()
        loaded = load_dataset_json(path)

    assert '"interpolator"' not in payload
    loaded_comp = (
        loaded.case_studies["cs"].processes["p"].reactor_medium.components["glucose"]
    )
    loaded_transform = loaded.case_studies["cs"].processes["p"].pseudobatch_transform
    loaded_tr = loaded_comp.concentration.metadata["transform"]
    assert loaded_tr["name"] == "pseudo_batch"
    assert "series" not in loaded_tr
    assert loaded_transform is not None

    bt_orig = build_backtransform_spline(transform, "glucose")
    bt_loaded = build_backtransform_spline(loaded_transform, "glucose")
    for t_val in [0.0, 25.0, 75.0]:
        orig = float(bt_orig(jnp.array(t_val)))
        loaded_val = float(bt_loaded(jnp.array(t_val)))
        assert abs(orig - loaded_val) < 1e-4, (
            f"Roundtrip mismatch at t={t_val}: {orig} vs {loaded_val}"
        )

    post_delta = 5e-4
    orig_adf = transform.adf_ts
    loaded_adf = loaded_transform.adf_ts
    t_b = float(orig_adf.jump_times[0])

    orig_pre = float(orig_adf.evaluate(jnp.array(t_b), side="left"))
    orig_post = float(orig_adf.evaluate(jnp.array(t_b + post_delta), side="left"))
    loaded_pre = float(loaded_adf.evaluate(jnp.array(t_b), side="left"))
    loaded_post = float(loaded_adf.evaluate(jnp.array(t_b + post_delta), side="left"))
    assert orig_post > orig_pre
    assert loaded_post > loaded_pre
    assert loaded_pre == pytest.approx(orig_pre, abs=1e-12)
    assert loaded_post == pytest.approx(orig_post, abs=1e-12)


def test_pseudobatch_metadata_json_serializable():
    """Transform metadata is JSON-serializable lightweight provenance."""
    import json

    proc = _make_process_with_bolus_feed()
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_timeseries(inputs, splines, "glucose")

    assert "transform" in rep.metadata
    tr = rep.metadata["transform"]
    assert tr["name"] == "pseudo_batch"
    assert tr["species"] == "glucose"
    assert "series" not in tr
    assert tr["cstar_interp"] in {"cubic", "pchip"}

    meta_json = json.dumps(rep.metadata)
    meta_loaded = json.loads(meta_json)
    assert meta_loaded["transform"]["name"] == "pseudo_batch"


def test_bundle_adf_timeseries_evaluates_canonical_spline_state():
    proc = _make_process_with_bolus_feed()
    transform = _build_single_species_transform(proc, "glucose")
    adf_ts = transform.adf_ts
    t_eval = jnp.linspace(
        float(adf_ts.breaks[0]),
        float(adf_ts.breaks[-1]),
        64,
    )
    values = np.asarray(adf_ts.evaluate_many(t_eval))
    assert np.all(np.isfinite(values))
    t_b = jnp.array(float(adf_ts.jump_times[0]))
    assert float(adf_ts.evaluate(t_b, side="left")) < float(
        adf_ts.evaluate(t_b, side="right")
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
    ds = BenchmarkDataset(
        metadata={"name": "test"},
        case_studies={
            "cs": CaseStudy(
                case_id="cs", organism="E. coli", citation="test", processes={"p": proc}
            )
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "legacy-pv.json"
        save_dataset_json(ds, path)
        payload = path.read_text()
        mutated = payload.replace(
            '"values": {',
            '"interpolator": {"kind": "interpax_linear"},\n          "values": {',
            1,
        )
        path.write_text(mutated)
        with pytest.raises(ValueError, match="Legacy sibling 'interpolator' payloads"):
            load_dataset_json(path)


def test_load_rejects_legacy_reactor_component_interpolator_payload():
    """Loader should reject legacy sibling interpolator payloads on components."""
    ts = fit_timeseries_spline(_ts([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]))
    rm = ReactorMedium(
        name="m",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=ts,
                is_intracellular=False,
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
        path = Path(tmpdir) / "legacy-comp.json"
        save_dataset_json(ds, path)
        payload = path.read_text()
        mutated = payload.replace(
            '"concentration": {',
            '"interpolator": {"kind": "interpax_ppoly"},\n'
            '              "concentration": {',
            1,
        )
        path.write_text(mutated)
        with pytest.raises(ValueError, match="Legacy sibling 'interpolator' payloads"):
            load_dataset_json(path)


def test_load_rejects_legacy_volume_change_interpolator_payload():
    """Loader should reject legacy sibling interpolator payloads on volume changes."""
    feed = FeedVolumeChange(
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
    ds = BenchmarkDataset(
        metadata={"name": "test"},
        case_studies={
            "cs": CaseStudy(
                case_id="cs", organism="E. coli", citation="test", processes={"p": proc}
            )
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "legacy-vc.json"
        save_dataset_json(ds, path)
        payload = path.read_text()
        mutated = payload.replace(
            '"values": {',
            '"interpolator": {"kind": "interpax_linear"},\n            "values": {',
            1,
        )
        path.write_text(mutated)
        with pytest.raises(ValueError, match="Legacy sibling 'interpolator' payloads"):
            load_dataset_json(path)


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
    to_timeseries(inputs, splines, "glucose")
    bt = _build_single_species_backtransform(proc, "glucose")
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
    to_timeseries(inputs, splines, "glucose")
    bt = _build_single_species_backtransform(proc, "glucose")
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
    to_timeseries(inputs, splines, "glucose")
    bt = _build_single_species_backtransform(proc, "glucose")

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
    to_timeseries(inputs, splines, "glucose")
    bt = _build_single_species_backtransform(proc, "glucose")

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
        build_splines(inputs, proc, "biomass")
        assert inputs.get("cstar_interp", "cubic") == "cubic"

    def test_pchip_backtransform_roundtrip(self):
        """PCHIP spline serialized and rebuilt recovers measurement values."""
        proc = _make_process_sharp_profile()
        inputs = build_pseudobatch_inputs(proc, "acetate")
        splines = build_splines(inputs, proc, "acetate")
        rep = to_timeseries(inputs, splines, "acetate")

        assert rep.metadata["transform"]["cstar_interp"] == "pchip"

        bt = _build_single_species_backtransform(proc, "acetate")
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
        to_timeseries(inputs, splines, "acetate")
        bt = _build_single_species_backtransform(proc, "acetate")

        t_dense = jnp.linspace(0.0, 10.0, 500)
        c_dense = jnp.array([float(bt(t)) for t in t_dense])
        assert float(jnp.min(c_dense)) >= -1e-6, (
            f"BacktransformSpline went negative: min={float(jnp.min(c_dense)):.6f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
