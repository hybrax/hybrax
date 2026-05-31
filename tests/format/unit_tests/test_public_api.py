"""Public package API tests."""

import bp_format


def test_version():
    assert hasattr(bp_format, "__version__")
    assert isinstance(bp_format.__version__, str)


def test_main_exports_are_available():
    for name in (
        "TimeAxis",
        "DiscreteEvents",
        "TimeSeries",
        "StaticVariable",
        "BioProcessMetadata",
        "ProcessVariable",
        "FeedMediumComponent",
        "ReactorMediumComponent",
        "FeedMedium",
        "ReactorMedium",
        "FeedVolumeChange",
        "SampleVolumeChange",
        "VolumeChange",
        "Volume",
        "BioProcess",
        "CaseStudy",
        "BenchmarkDataset",
        "print_process_structure",
        "print_dataset_structure",
        "plot_process",
        "plot_case_study",
        "validate_process",
        "validate_case_study",
        "serialization",
        "validate",
        "mechanistic",
        "splines",
    ):
        assert hasattr(bp_format, name)


def test_all_exports():
    assert hasattr(bp_format, "__all__")
    assert isinstance(bp_format.__all__, list)
    assert len(bp_format.__all__) > 0
