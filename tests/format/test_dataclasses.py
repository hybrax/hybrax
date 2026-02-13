"""
Basic tests for BPbench data structures
"""

import pytest
import jax.numpy as jnp
from bpbench import (
    BenchmarkDataset, CaseStudy, Process,
    TimeSeries, RawTimeSeries, TimeAxis,
    ReactorProperties, StaticVariable,
    get_event_times, leave_one_process_out, iter_loocv
)


def test_time_axis_creation():
    """Test TimeAxis creation"""
    time_axis = TimeAxis(
        unit="hours",
        start=0.0,
        end=48.0,
        time_reference="inoculation"
    )
    assert time_axis.unit == "hours"
    assert time_axis.start == 0.0
    assert time_axis.end == 48.0
    assert time_axis.time_reference == "inoculation"


def test_raw_timeseries_creation():
    """Test RawTimeSeries creation"""
    raw = RawTimeSeries(
        timepoints=jnp.array([0., 12., 24., 36., 48.]),
        values=jnp.array([0.1, 1.2, 3.5, 5.8, 6.0])
    )
    assert raw.timepoints.shape == (5,)
    assert raw.values.shape == (5,)
    assert raw.measurement_std is None


def test_timeseries_creation():
    """Test TimeSeries creation"""
    raw = RawTimeSeries(
        timepoints=jnp.array([0., 12., 24., 36., 48.]),
        values=jnp.array([0.1, 1.2, 3.5, 5.8, 6.0])
    )
    
    timeseries = TimeSeries(
        name="Biomass",
        canonical_name="biomass",
        unit="g/L",
        role="state",
        raw=raw
    )
    
    assert timeseries.name == "Biomass"
    assert timeseries.canonical_name == "biomass"
    assert timeseries.unit == "g/L"
    assert timeseries.role == "state"
    assert timeseries.raw is not None


def test_process_creation():
    """Test Process creation"""
    time_axis = TimeAxis(
        unit="hours",
        start=0.0,
        end=48.0,
        time_reference="inoculation"
    )
    
    reactor = ReactorProperties(
        working_volume=1.0,
        volume_unit="L"
    )
    
    process = Process(
        process_id="batch_001",
        process_type="batch",
        time=time_axis,
        reactor=reactor
    )
    
    assert process.process_id == "batch_001"
    assert process.process_type == "batch"
    assert process.time is not None
    assert process.reactor is not None


def test_case_study_creation():
    """Test CaseStudy creation"""
    process = Process(
        process_id="batch_001",
        process_type="batch"
    )
    
    case_study = CaseStudy(
        case_id="ecoli_study",
        organism="Escherichia coli",
        citation="Doe et al. 2024",
        processes={"batch_001": process}
    )
    
    assert case_study.case_id == "ecoli_study"
    assert case_study.organism == "Escherichia coli"
    assert len(case_study.processes) == 1


def test_benchmark_dataset_creation():
    """Test BenchmarkDataset creation"""
    process = Process(
        process_id="batch_001",
        process_type="batch"
    )
    
    case_study = CaseStudy(
        case_id="ecoli_study",
        organism="Escherichia coli",
        citation="Doe et al. 2024",
        processes={"batch_001": process}
    )
    
    dataset = BenchmarkDataset(
        metadata={
            "name": "Test Dataset",
            "version": "0.1.0"
        },
        case_studies={"ecoli": case_study}
    )
    
    assert dataset.metadata["name"] == "Test Dataset"
    assert len(dataset.case_studies) == 1


def test_get_event_times():
    """Test get_event_times utility"""
    process = Process(
        process_id="batch_001",
        process_type="batch",
        event_times=jnp.array([12.0, 24.0, 36.0])
    )
    
    event_times = get_event_times(process)
    assert event_times.shape == (3,)
    
    # Test with None
    process_no_events = Process(
        process_id="batch_002",
        process_type="batch"
    )
    event_times_empty = get_event_times(process_no_events)
    assert event_times_empty.shape == (0,)


def test_leave_one_process_out():
    """Test leave_one_process_out utility"""
    processes = {
        "p1": Process(process_id="p1", process_type="batch"),
        "p2": Process(process_id="p2", process_type="batch"),
        "p3": Process(process_id="p3", process_type="batch")
    }
    
    case_study = CaseStudy(
        case_id="test",
        organism="Test organism",
        citation="Test citation",
        processes=processes
    )
    
    splits = list(leave_one_process_out(case_study))
    assert len(splits) == 3
    
    # Check first split
    train_ids, test_id = splits[0]
    assert len(train_ids) == 2
    assert test_id in ["p1", "p2", "p3"]
    assert test_id not in train_ids


def test_iter_loocv():
    """Test iter_loocv utility"""
    processes = {
        "p1": Process(process_id="p1", process_type="batch"),
        "p2": Process(process_id="p2", process_type="batch")
    }
    
    case_study = CaseStudy(
        case_id="test",
        organism="Test organism",
        citation="Test citation",
        processes=processes
    )
    
    dataset = BenchmarkDataset(
        metadata={"name": "Test"},
        case_studies={"case1": case_study}
    )
    
    cv_splits = list(iter_loocv(dataset))
    assert len(cv_splits) == 2
    
    case_id, train_ids, test_id = cv_splits[0]
    assert case_id == "case1"
    assert len(train_ids) == 1


def test_static_variable():
    """Test StaticVariable creation"""
    var = StaticVariable(value=0.5, unit="1/h")
    assert var.value == 0.5
    assert var.unit == "1/h"


def test_volume_change_creation():
    """Test VolumeChange creation"""
    from bpbench import VolumeChange
    
    # Test continuous, controlled volume change
    vc_continuous = VolumeChange(
        name="feed1",
        controlled=True,
        continuous=True,
        unit="L/h",
        feed_medium="glucose_feed"
    )
    assert vc_continuous.name == "feed1"
    assert vc_continuous.controlled is True
    assert vc_continuous.continuous is True
    assert vc_continuous.unit == "L/h"
    
    # Test discrete volume change
    vc_discrete = VolumeChange(
        name="bolus_addition",
        controlled=True,
        continuous=False,
        unit="L",
        timepoints=jnp.array([10.0, 20.0]),
        values=jnp.array([0.5, 0.3])
    )
    assert vc_discrete.continuous is False
    assert vc_discrete.timepoints.shape == (2,)
    assert vc_discrete.values.shape == (2,)


def test_volume_creation():
    """Test Volume creation"""
    from bpbench import Volume, VolumeChange
    
    vc = VolumeChange(
        name="feed1",
        controlled=True,
        continuous=True,
        unit="L/h"
    )
    
    volume = Volume(
        volume_changes={"feed1": vc},
        initial_volume=1.0,
        volume_unit="L"
    )
    
    assert volume.initial_volume == 1.0
    assert volume.volume_unit == "L"
    assert "feed1" in volume.volume_changes
    assert volume.volume_changes["feed1"].name == "feed1"


def test_volume_validation_continuous():
    """Test Volume validation with continuous feed"""
    from bpbench import Volume, VolumeChange
    
    # Create continuous feed with known rate
    times = jnp.array([0., 10., 20., 30.])
    rates = jnp.array([0.1, 0.1, 0.1, 0.1])  # constant 0.1 L/h
    
    raw = RawTimeSeries(timepoints=times, values=rates)
    ts = TimeSeries(name='feed_rate', unit='L/h', raw=raw)
    
    vc = VolumeChange(
        name='feed',
        controlled=True,
        continuous=True,
        unit='L/h',
        timeseries=ts
    )
    
    volume = Volume(
        volume_changes={'feed': vc},
        initial_volume=1.0,
        volume_unit='L'
    )
    
    time_axis = TimeAxis(unit='hours', start=0.0, end=30.0, time_reference='inoculation')
    
    # Expected: 1.0 + 0.1*30 = 4.0 L
    is_valid, msg = volume.validate_volume_consistency(time_axis=time_axis, final_volume=4.0)
    assert is_valid is True
    assert "4.00" in msg


def test_volume_validation_discrete():
    """Test Volume validation with discrete additions"""
    from bpbench import Volume, VolumeChange
    
    vc = VolumeChange(
        name='bolus',
        controlled=True,
        continuous=False,
        unit='L',
        timepoints=jnp.array([10.0, 20.0, 30.0]),
        values=jnp.array([0.5, 0.5, 0.5])
    )
    
    volume = Volume(
        volume_changes={'bolus': vc},
        initial_volume=2.0,
        volume_unit='L'
    )
    
    # Expected: 2.0 + 0.5 + 0.5 + 0.5 = 3.5 L
    is_valid, msg = volume.validate_volume_consistency(final_volume=3.5)
    assert is_valid is True
    assert "3.50" in msg


def test_volume_validation_inconsistency():
    """Test Volume validation detects inconsistencies"""
    from bpbench import Volume, VolumeChange
    
    vc = VolumeChange(
        name='bolus',
        controlled=True,
        continuous=False,
        unit='L',
        timepoints=jnp.array([10.0]),
        values=jnp.array([1.0])
    )
    
    volume = Volume(
        volume_changes={'bolus': vc},
        initial_volume=1.0,
        volume_unit='L'
    )
    
    # Expected final: 2.0, but we claim it's 3.0 (50% difference)
    is_valid, msg = volume.validate_volume_consistency(final_volume=3.0)
    assert is_valid is False
    assert "inconsistency" in msg.lower()


def test_process_with_volume():
    """Test Process creation with Volume"""
    from bpbench import Volume, VolumeChange
    
    vc = VolumeChange(name='feed', controlled=True, continuous=True, unit='L/h')
    volume = Volume(volume_changes={'feed': vc}, initial_volume=1.0)
    
    process = Process(
        process_id="fed_batch_001",
        process_type="fed_batch",
        volume=volume
    )
    
    assert process.volume is not None
    assert process.volume.initial_volume == 1.0
    assert "feed" in process.volume.volume_changes


def test_process_backward_compatibility():
    """Test Process with new field names"""
    raw = RawTimeSeries(
        timepoints=jnp.array([0., 12., 24.]),
        values=jnp.array([0.1, 1.2, 3.5])
    )
    
    biomass_ts = TimeSeries(name="Biomass", unit="g/L", role="state", raw=raw)
    temp_ts = TimeSeries(name="Temperature", unit="K", role="control", raw=raw)
    
    process = Process(
        process_id="test",
        process_type="batch",
        dynamic_variables={
            "biomass": biomass_ts,
            "temperature": temp_ts
        }
    )
    
    # Test new field names work correctly
    assert "biomass" in process.dynamic_variables
    assert "temperature" in process.dynamic_variables


def test_volume_feed_validation():
    """Test Volume feed component validation"""
    from bpbench import Volume, VolumeChange, Feed, FeedComponent
    
    # Create a feed with glucose component
    glucose_feed = Feed(
        name="glucose_feed",
        density=1.1,
        density_unit="kg/L",
        components={
            "glucose": FeedComponent(concentration=500.0, unit="g/L")
        }
    )
    
    # Create VolumeChange with feed reference
    vc = VolumeChange(
        name='feed',
        controlled=True,
        continuous=True,
        unit='L',
        feed_medium='glucose_feed'
    )
    
    volume = Volume(
        volume_changes={'feed': vc},
        initial_volume=1.0,
        volume_unit='L'
    )
    
    # Test with feed that exists but is missing some components
    process_feeds = {'glucose_feed': glucose_feed}
    dynamic_variables = {
        'biomass': TimeSeries(name='biomass', unit='g/L'),
        'glucose': TimeSeries(name='glucose', unit='g/L')
    }
    
    is_valid, msg = volume.validate_feed_components(process_feeds, dynamic_variables)
    # Should return True (no errors) but with warning about missing biomass
    assert is_valid is True, "Should be valid even with warnings"
    assert 'warning' in msg.lower(), "Should contain warning"
    assert 'biomass' in msg.lower(), "Should warn about missing biomass"
    
    # Test with missing feed reference - should return False
    is_valid, msg = volume.validate_feed_components({}, dynamic_variables)
    assert is_valid is False, "Should be invalid when feed reference doesn't exist"
    assert 'error' in msg.lower(), "Should contain error message"
    assert 'not defined' in msg.lower(), "Should indicate feed is not defined"


def test_volume_change_with_inline_feed():
    """Test VolumeChange with inline feed definition"""
    from bpbench import VolumeChange, Feed, FeedComponent
    
    inline_feed = Feed(
        name="inline_feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "substrate": FeedComponent(concentration=100.0, unit="g/L")
        }
    )
    
    vc = VolumeChange(
        name='feed',
        controlled=True,
        continuous=True,
        unit='L',
        feed=inline_feed
    )
    
    assert vc.feed is not None
    assert vc.feed.name == "inline_feed"
    assert "substrate" in vc.feed.components


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
