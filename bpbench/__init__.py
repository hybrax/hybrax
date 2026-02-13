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
    RawTimeSeries,
    SplineRepresentation,
    TimeSeries,
    StaticVariable,
    BioProcessMetadata,
    # Volume-related structures
    VolumeChange,
    Volume,
    FeedComponent,
    FeedMedium,
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
    "RawTimeSeries",
    "SplineRepresentation",
    "TimeSeries",
    "StaticVariable",
    "FeedComponent",
    "Feed",
    "ReactorProperties",
    "VolumeChange",
    "Volume",
    "Process",
    "CaseStudy",
    "BenchmarkDataset",
    # Utils
    "get_event_times",
    "leave_one_process_out",
    "iter_loocv",
    "print_structure",
    "fit_cubic_spline",
    "compute_rate_from_cumulative",
    # Modules
    "serialization",
    "validate",
]
