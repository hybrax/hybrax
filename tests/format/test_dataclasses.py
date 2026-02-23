"""
Tests for BPbench data structures (current architecture).
"""

import pytest
import jax.numpy as jnp

from bpbench import (
    TimeAxis,
    TimeSeries,
    StaticVariable,
    BioProcessMetadata,
    ProcessVariable,
    FeedMediumComponent,
    ReactorMediumComponent,
    FeedMedium,
    ReactorMedium,
    VolumeChange,
    Volume,
    BioProcess,
    CaseStudy,
    BenchmarkDataset,
)


# ---------------------------------------------------------------------------
# Low-level structures
# ---------------------------------------------------------------------------

def test_time_axis_creation():
    ta = TimeAxis(unit="hours", start=0.0, end=48.0, time_reference="inoculation")
    assert ta.unit == "hours"
    assert ta.start == 0.0
    assert ta.end == 48.0
    assert ta.time_reference == "inoculation"


def test_timeseries_creation():
    ts = TimeSeries(
        timepoints=jnp.array([0., 12., 24., 36., 48.]),
        values=jnp.array([0.1, 1.2, 3.5, 5.8, 6.0]),
    )
    assert ts.timepoints.shape == (5,)
    assert ts.values.shape == (5,)


def test_static_variable_creation():
    sv = StaticVariable(value=0.5)
    assert sv.value == 0.5


def test_bioprocess_metadata_creation():
    meta = BioProcessMetadata(name="batch_001", process_type="batch")
    assert meta.name == "batch_001"
    assert meta.process_type == "batch"
    assert meta.notes is None


def test_bioprocess_metadata_with_notes():
    meta = BioProcessMetadata(name="fb_001", process_type="fed_batch", notes="Replicate A")
    assert meta.notes == "Replicate A"


# ---------------------------------------------------------------------------
# Component structures
# ---------------------------------------------------------------------------

def test_process_variable_timeseries():
    ts = TimeSeries(
        timepoints=jnp.array([0., 1., 2.]),
        values=jnp.array([37.0, 37.0, 37.0]),
    )
    pv = ProcessVariable(name="temperature", unit="°C", is_controlled=True, values=ts)
    assert pv.name == "temperature"
    assert pv.is_controlled is True
    assert pv.spline is None
    assert hasattr(pv.values, "timepoints")


def test_process_variable_static():
    sv = StaticVariable(value=7.0)
    pv = ProcessVariable(name="pH", unit="", is_controlled=False, values=sv)
    assert pv.name == "pH"
    assert isinstance(pv.values, StaticVariable)


def test_feed_medium_component_static():
    fmc = FeedMediumComponent(
        name="glucose", unit="g/L",
        concentration=StaticVariable(value=500.0),
        is_controlled=True,
    )
    assert fmc.name == "glucose"
    assert fmc.concentration.value == 500.0


def test_feed_medium_component_timeseries():
    ts = TimeSeries(timepoints=jnp.array([0., 1.]), values=jnp.array([100.0, 200.0]))
    fmc = FeedMediumComponent(name="glucose", unit="g/L", concentration=ts, is_controlled=True)
    assert hasattr(fmc.concentration, "timepoints")


def test_reactor_medium_component_timeseries():
    ts = TimeSeries(timepoints=jnp.array([0., 1., 2.]), values=jnp.array([0.1, 0.5, 1.0]))
    rc = ReactorMediumComponent(name="biomass", unit="g/L", concentration=ts, is_intracellular=False)
    assert rc.name == "biomass"
    assert rc.is_intracellular is False


def test_reactor_medium_component_static():
    rc = ReactorMediumComponent(
        name="biomass", unit="g/L",
        concentration=StaticVariable(value=1.0),
        is_intracellular=False,
    )
    assert isinstance(rc.concentration, StaticVariable)


# ---------------------------------------------------------------------------
# Medium-level structures
# ---------------------------------------------------------------------------

def test_feed_medium_empty_components():
    fm = FeedMedium(name="glucose_feed", density=1.0, density_unit="kg/L")
    assert fm.name == "glucose_feed"
    assert fm.components == {}


def test_feed_medium_with_components():
    fmc = FeedMediumComponent(
        name="glucose", unit="g/L",
        concentration=StaticVariable(value=500.0),
        is_controlled=True,
    )
    fm = FeedMedium(
        name="glucose_feed", density=1.1, density_unit="kg/L",
        components={"glucose": fmc},
    )
    assert "glucose" in fm.components
    assert fm.density == 1.1


def test_reactor_medium_empty_components():
    rm = ReactorMedium(name="medium", density=1.0, density_unit="kg/L")
    assert rm.components == {}


def test_reactor_medium_with_components():
    ts = TimeSeries(timepoints=jnp.array([0., 1.]), values=jnp.array([0.1, 0.5]))
    rc = ReactorMediumComponent(name="biomass", unit="g/L", concentration=ts, is_intracellular=False)
    rm = ReactorMedium(name="medium", density=1.0, density_unit="kg/L", components={"biomass": rc})
    assert "biomass" in rm.components


def test_volume_change_continuous():
    ts = TimeSeries(timepoints=jnp.array([0., 5., 10.]), values=jnp.array([0.0, 0.5, 1.0]))
    fm = FeedMedium(name="f", density=1.0, density_unit="kg/L")
    vc = VolumeChange(name="feed", unit="L", is_controlled=True, is_continuous=True,
                      feed_medium=fm, values=ts)
    assert vc.name == "feed"
    assert vc.is_continuous is True
    assert vc.values.timepoints.shape == (3,)


def test_volume_change_discrete():
    ts = TimeSeries(timepoints=jnp.array([2.0, 5.0]), values=jnp.array([0.5, 0.5]))
    fm = FeedMedium(name="f", density=1.0, density_unit="kg/L")
    vc = VolumeChange(name="bolus", unit="L", is_controlled=True, is_continuous=False,
                      feed_medium=fm, values=ts)
    assert vc.is_continuous is False


def test_volume_default_volume_changes():
    vol = Volume(initial_volume=1.0, unit="L")
    assert vol.initial_volume == 1.0
    assert vol.volume_changes == {}


def test_volume_with_changes():
    ts = TimeSeries(timepoints=jnp.array([0., 10.]), values=jnp.array([0.0, 0.5]))
    fm = FeedMedium(name="f", density=1.0, density_unit="kg/L")
    vc = VolumeChange(name="feed", unit="L", is_controlled=True, is_continuous=True,
                      feed_medium=fm, values=ts)
    vol = Volume(initial_volume=1.0, unit="L", volume_changes={"feed": vc})
    assert "feed" in vol.volume_changes


# ---------------------------------------------------------------------------
# Process-level structures
# ---------------------------------------------------------------------------

def test_bioprocess_minimal():
    process = BioProcess(
        metadata=BioProcessMetadata(name="batch_001", process_type="batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=24.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="medium", density=1.0, density_unit="kg/L"),
    )
    assert process.metadata.name == "batch_001"
    assert process.metadata.process_type == "batch"
    assert process.process_variables == {}


def test_bioprocess_with_process_variables():
    ts = TimeSeries(timepoints=jnp.array([0., 1.]), values=jnp.array([37.0, 37.0]))
    pv = ProcessVariable(name="temperature", unit="°C", is_controlled=True, values=ts)
    process = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=10.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="medium", density=1.0, density_unit="kg/L"),
        process_variables={"temperature": pv},
    )
    assert "temperature" in process.process_variables


def test_case_study_creation():
    process = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=24.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="medium", density=1.0, density_unit="kg/L"),
    )
    cs = CaseStudy(
        case_id="ecoli_study",
        organism="Escherichia coli",
        citation="Doe et al. 2024",
        processes={"p1": process},
    )
    assert cs.case_id == "ecoli_study"
    assert cs.organism == "Escherichia coli"
    assert "p1" in cs.processes


def test_benchmark_dataset_creation():
    process = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=24.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="medium", density=1.0, density_unit="kg/L"),
    )
    cs = CaseStudy(
        case_id="ecoli_study",
        organism="Escherichia coli",
        citation="Doe et al. 2024",
        processes={"p1": process},
    )
    dataset = BenchmarkDataset(
        metadata={"name": "Test Dataset", "version": "0.1.0"},
        case_studies={"ecoli": cs},
    )
    assert dataset.metadata["name"] == "Test Dataset"
    assert "ecoli" in dataset.case_studies
    assert len(dataset.case_studies) == 1


def test_benchmark_dataset_empty():
    dataset = BenchmarkDataset()
    assert dataset.metadata == {}
    assert dataset.case_studies == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
