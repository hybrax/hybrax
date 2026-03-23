"""
BPbench: Bioprocess Benchmarking Dataset Structure

A JAX-compatible framework for standardized bioprocess data management
and benchmarking across multiple case studies.
"""

__version__ = "0.1.0"

# Import core dataclasses
from .dataclasses import (
    # Low-level structures
    TimeAxis,
    Interpolator,
    DiscreteEvents,
    TimeSeries,
    StaticVariable,
    BioProcessMetadata,
    # Process and medium components
    ProcessVariable,
    FeedMediumComponent,
    ReactorMediumComponent,
    # Volume-related structures
    VolumeChange,
    FeedVolumeChange,
    SampleVolumeChange,
    Volume,
    FeedMedium,
    ReactorMedium,
    # Higher-level structures
    BioProcess,
    BioProcessCollection,
    CaseStudy,
    BenchmarkDataset,
)


# Import inspect
from .inspect import (
    print_process_structure,
    print_dataset_structure,
    plot_process,
    plot_case_study,
)

# Import serialization functions
from . import serialization
from . import validate
from . import mechanistic
from . import splines
from .validate import (
    validate_timeseries_shape,
    validate_volume_change_sign,
    validate_volume_change_states,
    validate_biomass_in_reactor_medium,
    validate_process,
    validate_volume_consistency,
    validate_case_study,
)

__all__ = [
    # Version
    "__version__",
    # Dataclasses
    "TimeAxis",
    "Interpolator",
    "DiscreteEvents",
    "TimeSeries",
    "StaticVariable",
    "BioProcessMetadata",
    "ProcessVariable",
    "FeedMediumComponent",
    "ReactorMediumComponent",
    "FeedMedium",
    "ReactorMedium",
    "VolumeChange",
    "FeedVolumeChange",
    "SampleVolumeChange",
    "Volume",
    "BioProcess",
    "BioProcessCollection",
    "CaseStudy",
    "BenchmarkDataset",
    # Utils
    "print_process_structure",
    "print_dataset_structure",
    "plot_process",
    "plot_case_study",
    # Validate
    "validate_timeseries_shape",
    "validate_volume_change_sign",
    "validate_volume_change_states",
    "validate_biomass_in_reactor_medium",
    "validate_process",
    "validate_volume_consistency",
    "validate_case_study",
    # Modules
    "serialization",
    "validate",
    "mechanistic",
    "splines",
]
