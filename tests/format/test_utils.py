"""
Tests for BPbench utility functions
"""

import pytest
import jax.numpy as jnp
from io import StringIO
import sys
from bpbench import (
    Process, TimeSeries, RawTimeSeries, TimeAxis,
    ReactorProperties, StaticVariable, Feed, FeedComponent,
    Volume, VolumeChange,
    print_structure
)


@pytest.fixture
def simple_process():
    """Create a simple process for testing"""
    time_axis = TimeAxis(
        unit="hours",
        start=0.0,
        end=48.0,
        time_reference="inoculation"
    )
    
    biomass = TimeSeries(
        name="Biomass",
        canonical_name="biomass",
        unit="g/L",
        role="state",
        raw=RawTimeSeries(
            timepoints=jnp.array([0., 12., 24., 36., 48.]),
            values=jnp.array([0.1, 1.2, 3.5, 5.8, 6.0])
        )
    )
    
    reactor = ReactorProperties(
        working_volume=1.0,
        volume_unit="L"
    )
    
    process = Process(
        process_id="test_001",
        process_type="batch",
        time=time_axis,
        dynamic_states={"biomass": biomass},
        reactor=reactor
    )
    
    return process


@pytest.fixture
def complex_process():
    """Create a complex process with all features for testing"""
    time_axis = TimeAxis(
        unit="hours",
        start=0.0,
        end=48.0,
        time_reference="inoculation"
    )
    
    # Create states
    biomass = TimeSeries(
        name="Biomass",
        canonical_name="biomass",
        unit="g/L",
        role="state",
        raw=RawTimeSeries(
            timepoints=jnp.array([0., 12., 24., 36., 48.]),
            values=jnp.array([0.1, 1.2, 3.5, 5.8, 6.0])
        )
    )
    
    glucose = TimeSeries(
        name="Glucose",
        canonical_name="glucose",
        unit="g/L",
        role="state",
        raw=RawTimeSeries(
            timepoints=jnp.array([0., 12., 24., 36., 48.]),
            values=jnp.array([20.0, 15.0, 8.0, 2.0, 0.5])
        )
    )
    
    # Create controls
    temperature = TimeSeries(
        name="Temperature",
        canonical_name="temperature",
        unit="K",
        role="control",
        raw=RawTimeSeries(
            timepoints=jnp.array([0., 12., 24., 36., 48.]),
            values=jnp.array([310.0, 310.0, 310.0, 310.0, 310.0])
        )
    )
    
    # Create feed
    feed = Feed(
        name="Glucose feed",
        density=1.1,
        density_unit="kg/L",
        components={
            "glucose": FeedComponent(concentration=500.0, unit="g/L")
        }
    )
    
    # Create volume with feed tracking
    feed_rate_ts = TimeSeries(
        name="Feed_rate",
        canonical_name="feed_rate",
        unit="L/h",
        role="control",
        raw=RawTimeSeries(
            timepoints=jnp.array([0., 12., 24., 36., 48.]),
            values=jnp.array([0.0, 0.05, 0.05, 0.05, 0.0])
        )
    )
    
    volume_change = VolumeChange(
        name="glucose_feed",
        controlled=True,
        continuous=True,
        unit="L/h",
        feed_medium="Glucose feed",
        timeseries=feed_rate_ts
    )
    
    volume = Volume(
        volume_changes={"glucose_feed": volume_change},
        initial_volume=1.0,
        volume_unit="L"
    )
    
    # Create reactor
    reactor = ReactorProperties(
        working_volume=2.0,
        volume_unit="L",
        density=1.0
    )
    
    # Event times
    event_times = jnp.array([12.0, 24.0, 36.0])
    
    process = Process(
        process_id="test_fed_batch_001",
        process_type="fed_batch",
        replicate_id="rep1",
        time=time_axis,
        dynamic_states={"biomass": biomass, "glucose": glucose},
        dynamic_controls={"temperature": temperature},
        static_controls={"pH": StaticVariable(value=7.0, unit="pH")},
        volume=volume,
        feeds={"Glucose feed": feed},
        static_parameters={
            "mu_max": StaticVariable(value=0.5, unit="1/h"),
            "Ks": StaticVariable(value=0.1, unit="g/L")
        },
        event_times=event_times,
        reactor=reactor
    )
    
    return process


def test_print_structure_simple(simple_process, capsys):
    """Test print_structure with a simple process"""
    print_structure(simple_process)
    captured = capsys.readouterr()
    
    # Check that key information is present
    assert "Process Structure" in captured.out
    assert "test_001" in captured.out
    assert "batch" in captured.out
    assert "Biomass" in captured.out
    assert "biomass" in captured.out
    assert "g/L" in captured.out
    assert "0.00 to 48.00 hours" in captured.out
    assert "Working Volume: 1.0 L" in captured.out


def test_print_structure_complex(complex_process, capsys):
    """Test print_structure with a complex process"""
    print_structure(complex_process)
    captured = capsys.readouterr()
    
    # Check process info
    assert "test_fed_batch_001" in captured.out
    assert "fed_batch" in captured.out
    assert "rep1" in captured.out
    
    # Check dynamic states
    assert "Dynamic States: (2 total)" in captured.out
    assert "Biomass" in captured.out
    assert "Glucose" in captured.out
    
    # Check dynamic controls
    assert "Dynamic Controls: (1 total)" in captured.out
    assert "Temperature" in captured.out
    
    # Check static controls
    assert "Static Controls: (1 total)" in captured.out
    assert "pH" in captured.out
    
    # Check volume
    assert "Volume:" in captured.out
    assert "Initial: 1.0 L" in captured.out
    assert "Volume Changes: (1 total)" in captured.out
    assert "glucose_feed" in captured.out
    
    # Check feeds
    assert "Feeds: (1 total)" in captured.out
    assert "Glucose feed" in captured.out
    
    # Check static parameters
    assert "Static Parameters: (2 total)" in captured.out
    assert "mu_max" in captured.out
    assert "Ks" in captured.out
    
    # Check event times
    assert "Event Times: (3 total)" in captured.out
    
    # Check reactor
    assert "Reactor:" in captured.out
    assert "Working Volume: 2.0 L" in captured.out


def test_print_structure_with_values(simple_process, capsys):
    """Test print_structure with show_values=True"""
    print_structure(simple_process, show_values=True)
    captured = capsys.readouterr()
    
    # Check that values are shown
    assert "First 3:" in captured.out or "Values:" in captured.out


def test_print_structure_no_crash_minimal_process():
    """Test that print_structure doesn't crash with minimal process"""
    process = Process(
        process_id="minimal",
        process_type="batch"
    )
    
    # Should not raise any exception
    try:
        print_structure(process)
    except Exception as e:
        pytest.fail(f"print_structure raised {e} unexpectedly!")


def test_print_structure_with_many_event_times(capsys):
    """Test print_structure with many event times (>10)"""
    process = Process(
        process_id="test_many_events",
        process_type="batch",
        event_times=jnp.array([float(i) for i in range(20)])
    )
    
    print_structure(process)
    captured = capsys.readouterr()
    
    # Should show first 5 and last 5 for many events
    assert "First 5:" in captured.out
    assert "Last 5:" in captured.out
