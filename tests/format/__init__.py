"""
Test package initialization
"""

import bpbench


def test_version():
    """Test that version is defined"""
    assert hasattr(bpbench, '__version__')
    assert isinstance(bpbench.__version__, str)


def test_imports():
    """Test that all main exports are available"""
    # Dataclasses
    assert hasattr(bpbench, 'TimeAxis')
    assert hasattr(bpbench, 'RawTimeSeries')
    assert hasattr(bpbench, 'SplineRepresentation')
    assert hasattr(bpbench, 'TimeSeries')
    assert hasattr(bpbench, 'StaticVariable')
    assert hasattr(bpbench, 'FeedComponent')
    assert hasattr(bpbench, 'Feed')
    assert hasattr(bpbench, 'ReactorProperties')
    assert hasattr(bpbench, 'Process')
    assert hasattr(bpbench, 'CaseStudy')
    assert hasattr(bpbench, 'BenchmarkDataset')
    
    # Utils
    assert hasattr(bpbench, 'get_event_times')
    assert hasattr(bpbench, 'leave_one_process_out')
    assert hasattr(bpbench, 'iter_loocv')
    
    # Serialization
    assert hasattr(bpbench, 'save_dataset')
    assert hasattr(bpbench, 'load_dataset')
    assert hasattr(bpbench, 'save_dataset_json')
    assert hasattr(bpbench, 'load_dataset_json')


def test_all_exports():
    """Test __all__ is properly defined"""
    assert hasattr(bpbench, '__all__')
    assert isinstance(bpbench.__all__, list)
    assert len(bpbench.__all__) > 0
