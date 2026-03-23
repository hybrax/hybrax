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
    assert hasattr(bpbench, 'Interpolator')
    assert hasattr(bpbench, 'DiscreteEvents')
    assert hasattr(bpbench, 'TimeSeries')
    assert hasattr(bpbench, 'StaticVariable')
    assert hasattr(bpbench, 'BioProcessMetadata')
    assert hasattr(bpbench, 'ProcessVariable')
    assert hasattr(bpbench, 'FeedMediumComponent')
    assert hasattr(bpbench, 'ReactorMediumComponent')
    assert hasattr(bpbench, 'FeedMedium')
    assert hasattr(bpbench, 'ReactorMedium')
    assert hasattr(bpbench, 'FeedVolumeChange')
    assert hasattr(bpbench, 'SampleVolumeChange')
    assert hasattr(bpbench, 'VolumeChange')
    assert hasattr(bpbench, 'Volume')
    assert hasattr(bpbench, 'BioProcess')
    assert hasattr(bpbench, 'CaseStudy')
    assert hasattr(bpbench, 'BenchmarkDataset')
    
    # Utils
    assert hasattr(bpbench, 'print_process_structure')
    assert hasattr(bpbench, 'print_dataset_structure')
    assert hasattr(bpbench, 'plot_process')
    assert hasattr(bpbench, 'plot_case_study')
    assert hasattr(bpbench, 'validate_process')
    assert hasattr(bpbench, 'validate_case_study')
    
    # Modules
    assert hasattr(bpbench, 'serialization')
    assert hasattr(bpbench, 'validate')
    assert hasattr(bpbench, 'mechanistic')
    assert hasattr(bpbench, 'splines')


def test_all_exports():
    """Test __all__ is properly defined"""
    assert hasattr(bpbench, '__all__')
    assert isinstance(bpbench.__all__, list)
    assert len(bpbench.__all__) > 0
