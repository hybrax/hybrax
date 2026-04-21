"""
Test package initialization
"""

import bp_format


def test_version():
    """Test that version is defined"""
    assert hasattr(bp_format, '__version__')
    assert isinstance(bp_format.__version__, str)


def test_imports():
    """Test that all main exports are available"""
    # Dataclasses
    assert hasattr(bp_format, 'TimeAxis')
    assert hasattr(bp_format, 'Interpolator')
    assert hasattr(bp_format, 'DiscreteEvents')
    assert hasattr(bp_format, 'TimeSeries')
    assert hasattr(bp_format, 'StaticVariable')
    assert hasattr(bp_format, 'BioProcessMetadata')
    assert hasattr(bp_format, 'ProcessVariable')
    assert hasattr(bp_format, 'FeedMediumComponent')
    assert hasattr(bp_format, 'ReactorMediumComponent')
    assert hasattr(bp_format, 'FeedMedium')
    assert hasattr(bp_format, 'ReactorMedium')
    assert hasattr(bp_format, 'FeedVolumeChange')
    assert hasattr(bp_format, 'SampleVolumeChange')
    assert hasattr(bp_format, 'VolumeChange')
    assert hasattr(bp_format, 'Volume')
    assert hasattr(bp_format, 'BioProcess')
    assert hasattr(bp_format, 'CaseStudy')
    assert hasattr(bp_format, 'BenchmarkDataset')
    
    # Utils
    assert hasattr(bp_format, 'print_process_structure')
    assert hasattr(bp_format, 'print_dataset_structure')
    assert hasattr(bp_format, 'plot_process')
    assert hasattr(bp_format, 'plot_case_study')
    assert hasattr(bp_format, 'validate_process')
    assert hasattr(bp_format, 'validate_case_study')
    
    # Modules
    assert hasattr(bp_format, 'serialization')
    assert hasattr(bp_format, 'validate')
    assert hasattr(bp_format, 'mechanistic')
    assert hasattr(bp_format, 'splines')


def test_all_exports():
    """Test __all__ is properly defined"""
    assert hasattr(bp_format, '__all__')
    assert isinstance(bp_format.__all__, list)
    assert len(bp_format.__all__) > 0
