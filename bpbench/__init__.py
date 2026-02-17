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
    SplineRepresentation,
    TimeSeries,
    StaticVariable,
    BioProcessMetadata,
    # Process and medium components
    ProcessVariable,
    FeedMediumComponent,
    ReactorMediumComponent,
    # Volume-related structures
    VolumeChange,
    Volume,
    FeedMedium,
    ReactorMedium,
    # Higher-level structures
    BioProcess,
    CaseStudy,
    BenchmarkDataset,
)

# Import utilities
# from .utils import (
#     get_event_times,
#     leave_one_process_out,
#     iter_loocv,
# )

# Import inspect
from .inspect import (
    print_structure,
    print_dataset_structure
)

# # Import spline utilities
# from .splines import (
#     fit_cubic_spline,
#     compute_rate_from_cumulative,
# )

# Import serialization functions
from . import serialization
from . import validate

__all__ = [
    # Version
    "__version__",
    # Dataclasses
    "TimeAxis",
    "SplineRepresentation",
    "TimeSeries",
    "StaticVariable",
    "BioProcessMetadata",
    "ProcessVariable",
    "FeedMediumComponent",
    "ReactorMediumComponent",
    "FeedMedium",
    "ReactorMedium",
    "VolumeChange",
    "Volume",
    "BioProcess",
    "CaseStudy",
    "BenchmarkDataset",
    # Utils
    "print_structure",
    "print_dataset_structure",
    # Modules
    "serialization",
    "validate",
]
