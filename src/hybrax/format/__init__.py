"""
bp-format: Bioprocess Benchmarking Dataset Structure

A JAX-compatible framework for standardized bioprocess data management
and benchmarking across multiple case studies.
"""

__version__ = "0.1.0"

# Import core dataclasses
from .dataclasses import (
    # Low-level structures
    TimeAxis,
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
    # User-defined biological ODE
    Bounds,
    BiologicalOde,
    ProcessOrdering,
    PseudobatchTransform,
    # Higher-level structures
    BioProcess,
    AugmentedBioProcess,
    BioProcessCollection,
    CaseStudy,
)

# Import simulation helpers
from .simulation import Simulation, SimulationEvent, SimulationResult


# Import inspect
from .inspect import (
    print_process_structure,
    print_case_study_structure,
    print_rhs_ode,
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
    validate_measurement_sampling_alignment,
    validate_process,
    validate_volume_consistency,
    validate_case_study,
    validate_augmented_parent_refs,
    validate_biological_ode,
    validate_biological_ode_equivalence,
    validate_bounds,
)

__all__ = [
    # Version
    "__version__",
    # Dataclasses
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
    "PseudobatchTransform",
    "VolumeChange",
    "FeedVolumeChange",
    "SampleVolumeChange",
    "Volume",
    "BioProcess",
    "AugmentedBioProcess",
    "BioProcessCollection",
    "CaseStudy",
    "Bounds",
    "BiologicalOde",
    "ProcessOrdering",
    "Simulation",
    "SimulationEvent",
    "SimulationResult",
    # Utils
    "print_process_structure",
    "print_case_study_structure",
    "print_rhs_ode",
    "plot_process",
    "plot_case_study",
    # Validate
    "validate_timeseries_shape",
    "validate_volume_change_sign",
    "validate_volume_change_states",
    "validate_biomass_in_reactor_medium",
    "validate_measurement_sampling_alignment",
    "validate_process",
    "validate_volume_consistency",
    "validate_case_study",
    "validate_augmented_parent_refs",
    "validate_biological_ode",
    "validate_biological_ode_equivalence",
    "validate_bounds",
    # Modules
    "serialization",
    "validate",
    "mechanistic",
    "splines",
]
