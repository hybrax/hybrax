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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
