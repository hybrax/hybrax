"""Bioprocess benchmarking data structures and utilities."""

import importlib
import os
import sys
from typing import TYPE_CHECKING, Any

# bp-format requires float64 throughout. Configure future JAX imports without
# importing JAX just to initialize this package.
os.environ["JAX_ENABLE_X64"] = "true"
if "jax" in sys.modules:
    importlib.import_module("jax").config.update("jax_enable_x64", True)

__version__ = "0.1.0"

if TYPE_CHECKING:
    from . import mechanistic, serialization, splines, validate  # noqa: F401
    from .dataclasses import (  # noqa: F401
        AugmentedBioProcess,
        BiologicalOde,
        BioProcess,
        BioProcessCollection,
        BioProcessMetadata,
        Bounds,
        DiscreteEvents,
        FeedMedium,
        FeedMediumComponent,
        FeedVolumeChange,
        ProcessOrdering,
        ProcessVariable,
        PseudobatchTransform,
        ReactorMedium,
        ReactorMediumComponent,
        SampleVolumeChange,
        StaticVariable,
        TimeAxis,
        TimeSeries,
        Volume,
        VolumeChange,
    )
    from .inspect import (  # noqa: F401
        plot_collection,
        plot_process,
        print_collection_structure,
        print_process_structure,
        print_rhs_ode,
    )
    from .simulation import Simulation, SimulationEvent, SimulationResult  # noqa: F401
    from .validate import (  # noqa: F401
        validate_augmented_parent_refs,
        validate_biological_ode,
        validate_biological_ode_equivalence,
        validate_biomass_in_reactor_medium,
        validate_bounds,
        validate_bounds_against_data,
        validate_cross_process_consistency,
        validate_discrete_events,
        validate_for_publication,
        validate_initial_state_alignment,
        validate_measurement_sampling_alignment,
        validate_process,
        validate_time_axis,
        validate_timeseries_shape,
        validate_timestamp_bounds,
        validate_volume_change_sign,
        validate_volume_change_states,
        validate_volume_consistency,
        validate_volume_units,
    )

_EXPORT_MODULES = {
    ".dataclasses": (
        "TimeAxis",
        "DiscreteEvents",
        "TimeSeries",
        "StaticVariable",
        "BioProcessMetadata",
        "ProcessVariable",
        "FeedMediumComponent",
        "ReactorMediumComponent",
        "VolumeChange",
        "FeedVolumeChange",
        "SampleVolumeChange",
        "Volume",
        "FeedMedium",
        "ReactorMedium",
        "Bounds",
        "BiologicalOde",
        "ProcessOrdering",
        "PseudobatchTransform",
        "BioProcess",
        "AugmentedBioProcess",
        "BioProcessCollection",
    ),
    ".simulation": ("Simulation", "SimulationEvent", "SimulationResult"),
    ".inspect": (
        "print_process_structure",
        "print_collection_structure",
        "print_rhs_ode",
        "plot_process",
        "plot_collection",
    ),
    ".validate": (
        "validate_time_axis",
        "validate_timeseries_shape",
        "validate_discrete_events",
        "validate_timestamp_bounds",
        "validate_volume_change_sign",
        "validate_volume_change_states",
        "validate_volume_units",
        "validate_biomass_in_reactor_medium",
        "validate_initial_state_alignment",
        "validate_measurement_sampling_alignment",
        "validate_process",
        "validate_volume_consistency",
        "validate_for_publication",
        "validate_augmented_parent_refs",
        "validate_biological_ode",
        "validate_biological_ode_equivalence",
        "validate_bounds",
        "validate_bounds_against_data",
        "validate_cross_process_consistency",
    ),
}
_EXPORTS = {
    name: module_name
    for module_name, names in _EXPORT_MODULES.items()
    for name in names
}
_EXPORTS.update(
    {
        "serialization": ".serialization",
        "validate": ".validate",
        "mechanistic": ".mechanistic",
        "splines": ".splines",
    }
)

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str) -> Any:
    """Load public objects only when accessed."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = module if name == module_name.removeprefix(".") else getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy exports in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
