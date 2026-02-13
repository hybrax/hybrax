"""
Bioprocess Benchmarking Dataset Structure
JAX-compatible dataclasses for standardized bioprocess data
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import jax.numpy as jnp

# TODO for later: standardize unit spelling so it might be used for unit checks


# ============================================================
# Modelling Structures
# ============================================================

@dataclass
class SplineRepresentation:
    """
    Fitted spline representation of time series data.
    
    The spline is segmented based on discontinuities (e.g., process.event_times).
    Each segment has its own coefficients, allowing for different polynomial
    behavior in different time regions.
    
    Note: K breakpoints define K-1 segments, so M = K-1 where M is the number
    of segments (first dimension of coefficients array).
    
    For cumulative data (like feed volumes), this representation allows:
    - Smooth interpolation of cumulative values
    - Computation of rates (derivatives) at any time point
    - Handling of discontinuities in the rate profile
    """
    # TODO for later: This is a placehold for processed data that can be filled later.
    # type: str  # e.g. "cubic_hermite", "linear", "zero_order_hold"
    # breakpoints: jnp.ndarray  # shape (K,), segment boundaries (including start and end)
    # coefficients: jnp.ndarray  # shape (M, C), M=K-1 segments with C coefficients each
    # discontinuous: bool = False  # True if spline has discontinuities
    # fit_residual_std: Optional[float] = None  # goodness of fit
    # notes: Optional[str] = None  # any additional info about fitting

@dataclass
class DiscreteEvents:
    """
    Stores discrete events
    """
    # TODO for later: This is a placehold for processed data that can be filled later.


# ============================================================
# Low-Level Structures
# ============================================================

@dataclass
class TimeAxis:
    """
    Time axis definition for a bioprocess
    """
    unit:  str  # e.g. "hours", "days"
    start: float
    end:   float
    time_reference: str  # e.g. "inoculation", "first_feed", "operator_defined"


@dataclass
class RawTimeSeries:
    """
    Raw experimental measurements
    """
    timepoints: jnp.ndarray          # shape (N,)
    values:     jnp.ndarray          # shape (N,)
    measurement_std: Optional[jnp.ndarray] = None  # measurement uncertainty


@dataclass
class TimeSeries:
    """
    Time-dependent variable (state or control)
    """
    name: str  # original name from paper
    unit: str  # e.g. "g/L", "g/L/h", "°C" 
    controlled: bool # True if this variable is a control input, False if it's a state variable
    raw: Optional[RawTimeSeries] = None
    spline: Optional[SplineRepresentation] = None


@dataclass
class StaticVariable:
    """
    Time-independent parameter
    """
    name: str
    value: float
    unit: str


@dataclass
class FeedComponent:
    """
    Single component in a feed medium
    """
    name: str # eg. "glucose", "ammonium", "inductor"
    concentration: float
    unit: str  # e.g. "g/L", "mM"


@dataclass
class FeedMedium:
    """
    Feed medium definition for mass balance calculations
    """
    name: str
    density: float
    density_unit: str  # typically "kg/L"
    components: Dict[str, FeedComponent] = field(default_factory=dict)


@dataclass
class BioProcessMetadata:
    """
    Static reactor characteristics
    """
    name: str
    process_type: str  # "batch", "fed_batch", "continuous"
    notes: Optional[str] = None

@dataclass
class VolumeChange:
    """
    Represents a volume change event (discrete or continuous).
    
    Note: volume changes are saved in the volume unit (i.e., L, m3, kg), not as a rate.
    """
    name: str
    unit: str  # e.g. "L", "m3", "kg", not allowed is "L/h" or "kg/h" as they are ususally derived values
    controlled: bool  # True if controlled, False if modeled
    continuous: bool  # True if continuous, False if discrete
    feed_medium: Optional[FeedMedium] = None  # Reference to feed name in Process.feeds
    timeseries: Optional[TimeSeries] = None  # For continuous changes (cumulative or rate)


@dataclass
class Volume:
    """
    Container for all volume-related information in a process.
    
    This separates volume from states and controls since volume is special:
    - Not a classic "state" variable
    - Not purely a "control" (can be both controlled and modeled)
    - Affected by multiple operations (feeds, sampling, evaporation)
    """
    initial_volume: float
    volume_unit: str  # e.g. "L", "m3", "kg"
    density: float    # often implicitly 1 kg/L
    density_unit: str # typically "kg/L"
    volume_changes: Dict[str, VolumeChange] = field(default_factory=dict)

# ============================================================
# Process Level
# ============================================================

@dataclass
class BioProcess:
    """
    Single experimental bioprocess run.
    
    Structure:
    - metadata: Static information about the process (type, notes, etc.)
    - time: Time axis definition
    - dynamic_variables: Time-varying variables (biomass, substrate, product, temperature, etc.)
    - static_variables: Time-constant variables (inductor strength, initial concentrations, etc.)
    - volume: Volume tracking with all volume changes (feeds, sampling, evaporation)
    """
    metadata: BioProcessMetadata
    time: TimeAxis
    volume: Volume
    dynamic_variables: Dict[str, TimeSeries] = field(default_factory=dict)
    static_variables: Dict[str, StaticVariable] = field(default_factory=dict)
    

    # TODO for later: event_times is already a preprocessing step and will be considered in the future.
    # event_times: Optional[jnp.ndarray] = None  # sorted discontinuity times
    # - event_times: Discontinuity times, these should not be inferred without visual inspection, therefore here they are parsed directly


# ============================================================
# Case Study Level
# ============================================================

@dataclass
class CaseStudy:
    """Collection of processes from one publication/dataset"""
    case_id: str
    organism: str
    citation: str
    processes: Dict[str, BioProcess] = field(default_factory=dict)


# ============================================================
# Benchmark Dataset Level
# ============================================================

@dataclass
class BenchmarkDataset:
    """Top-level benchmarking dataset"""
    metadata: Dict[str, str] = field(default_factory=dict)
    # metadata should include: name, version, description, preprocessing_version
    
    case_studies: Dict[str, CaseStudy] = field(default_factory=dict)


# # ============================================================
# # PyTree Registration (critical for JAX compatibility)
# # ============================================================

# tree_util.register_pytree_node(
#     TimeAxis,
#     lambda obj: ((obj.start, obj.end), (obj.unit, obj.time_reference)),
#     lambda data, children: TimeAxis(data[0], children[0], children[1], data[1])
# )

# tree_util.register_pytree_node(
#     RawTimeSeries,
#     lambda obj: ((obj.timepoints, obj.values, obj.measurement_std), ()),
#     lambda data, children: RawTimeSeries(*children)
# )

# tree_util.register_pytree_node(
#     SplineRepresentation,
#     lambda obj: (
#         (obj.breakpoints, obj.coefficients, obj.fit_residual_std),
#         (obj.type, obj.discontinuous, obj.notes)
#     ),
#     lambda data, children: SplineRepresentation(
#         data[0], children[0], children[1], data[1], children[2], data[2]
#     )
# )

# tree_util.register_pytree_node(
#     TimeSeries,
#     lambda obj: ((obj.raw, obj.spline), (obj.name, obj.canonical_name, obj.unit, obj.role)),
#     lambda data, children: TimeSeries(data[0], data[1], data[2], data[3], *children)
# )

# tree_util.register_pytree_node(
#     StaticVariable,
#     lambda obj: ((obj.value,), (obj.unit,)),
#     lambda data, children: StaticVariable(children[0], data[0])
# )

# tree_util.register_pytree_node(
#     FeedComponent,
#     lambda obj: ((obj.concentration,), (obj.unit,)),
#     lambda data, children: FeedComponent(children[0], data[0])
# )

# tree_util.register_pytree_node(
#     Feed,
#     lambda obj: ((obj.density, obj.components), (obj.name, obj.density_unit)),
#     lambda data, children: Feed(data[0], children[0], data[1], children[1])
# )

# tree_util.register_pytree_node(
#     ReactorProperties,
#     lambda obj: ((obj.working_volume, obj.density), (obj.volume_unit,)),
#     lambda data, children: ReactorProperties(children[0], data[0], children[1])
# )

# tree_util.register_pytree_node(
#     VolumeChange,
#     lambda obj: (
#         (obj.timeseries, obj.timepoints, obj.values, obj.feed),
#         (obj.name, obj.controlled, obj.continuous, obj.unit, obj.feed_medium)
#     ),
#     lambda data, children: VolumeChange(
#         data[0], data[1], data[2], data[3], data[4], *children
#     )
# )

# tree_util.register_pytree_node(
#     Volume,
#     lambda obj: (
#         (obj.volume_changes, obj.initial_volume),
#         (obj.volume_unit,)
#     ),
#     lambda data, children: Volume(children[0], children[1], data[0])
# )

# tree_util.register_pytree_node(
#     Process,
#     lambda obj: (
#         (obj.time, obj.dynamic_variables, obj.static_variables,
#          obj.volume, obj.feeds, obj.event_times, obj.reactor),
#         (obj.process_id, obj.process_type, obj.replicate_id)
#     ),
#     lambda data, children: Process(
#         process_id=data[0], 
#         process_type=data[1], 
#         replicate_id=data[2],
#         time=children[0],
#         dynamic_variables=children[1],
#         static_variables=children[2],
#         volume=children[3],
#         feeds=children[4],
#         event_times=children[5],
#         reactor=children[6]
#     )
# )

# tree_util.register_pytree_node(
#     CaseStudy,
#     lambda obj: ((obj.processes,), (obj.case_id, obj.organism, obj.citation)),
#     lambda data, children: CaseStudy(data[0], data[1], data[2], children[0])
# )

# tree_util.register_pytree_node(
#     BenchmarkDataset,
#     lambda obj: ((obj.case_studies,), (obj.metadata,)),
#     lambda data, children: BenchmarkDataset(data[0], children[0])
# )
