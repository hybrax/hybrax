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
    FeedComponent,
    Feed,
    ReactorProperties,
    # Volume-related structures
    VolumeChange,
    Volume,
    # Higher-level structures
    Process,
    CaseStudy,
    BenchmarkDataset,
)

# Import utilities
from .utils import (
    get_event_times,
    leave_one_process_out,
    iter_loocv,
    print_structure,
)

# Import spline utilities
from .splines import (
    fit_cubic_spline,
    compute_rate_from_cumulative,
)

# Import serialization functions
from .serialization import (
    save_dataset,
    load_dataset,
    save_dataset_json,
    load_dataset_json,
)

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
    # Serialization
    "save_dataset",
    "load_dataset",
    "save_dataset_json",
    "load_dataset_json",
]
