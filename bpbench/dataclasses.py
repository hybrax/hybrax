"""
Bioprocess Benchmarking Dataset Structure
JAX-compatible dataclasses for standardized bioprocess data
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import jax.numpy as jnp
from jax import tree_util


# ============================================================
# Low-Level Structures
# ============================================================

@dataclass
class TimeAxis:
    """Time axis definition for a bioprocess"""
    unit: str  # e.g. "hours", "days"
    start: float
    end: float
    time_reference: str  # e.g. "inoculation", "first_feed", "operator_defined"


@dataclass
class RawTimeSeries:
    """Raw experimental measurements"""
    timepoints: jnp.ndarray          # shape (N,)
    values: jnp.ndarray              # shape (N,)
    measurement_std: Optional[jnp.ndarray] = None  # measurement uncertainty


@dataclass
class SplineRepresentation:
    """Fitted spline representation of time series data"""
    type: str  # e.g. "cubic_hermite", "linear", "zero_order_hold"
    breakpoints: jnp.ndarray  # shape (M+1,), sorted
    coefficients: jnp.ndarray  # shape (M, C), outer dim = num segments
    discontinuous: bool  # if True, contributes to process.event_times
    fit_residual_std: Optional[float] = None  # goodness of fit
    notes: Optional[str] = None  # any additional info about fitting


@dataclass
class TimeSeries:
    """Time-dependent variable (state or control)"""
    name: str  # original name from paper
    canonical_name: Optional[str] = None  # standardized cross-study name
    unit: str = ""
    role: str = "state"  # "state" or "control"
    raw: Optional[RawTimeSeries] = None
    spline: Optional[SplineRepresentation] = None


@dataclass
class StaticVariable:
    """Time-independent parameter"""
    value: float
    unit: str


@dataclass
class FeedComponent:
    """Single component in a feed medium"""
    concentration: float
    unit: str  # e.g. "g/L", "mM"


@dataclass
class Feed:
    """Feed medium definition for mass balance calculations"""
    name: str
    density: float
    density_unit: str  # typically "kg/L"
    components: Dict[str, FeedComponent] = field(default_factory=dict)


@dataclass
class ReactorProperties:
    """Static reactor characteristics"""
    working_volume: float  # nominal/geometric volume
    volume_unit: str
    density: Optional[float] = None  # often implicitly 1 kg/L


# ============================================================
# Process Level
# ============================================================

@dataclass
class Process:
    """Single experimental bioprocess run"""
    process_id: str
    process_type: str  # "batch", "fed_batch", "continuous"
    replicate_id: Optional[str] = None  # e.g. "rep1", "rep2"

    time: Optional[TimeAxis] = None

    states: Dict[str, TimeSeries] = field(default_factory=dict)
    controls: Dict[str, TimeSeries] = field(default_factory=dict)

    feeds: Dict[str, Feed] = field(default_factory=dict)
    static_parameters: Dict[str, StaticVariable] = field(default_factory=dict)

    event_times: Optional[jnp.ndarray] = None  # sorted discontinuity times

    reactor: Optional[ReactorProperties] = None


# ============================================================
# Case Study Level
# ============================================================

@dataclass
class CaseStudy:
    """Collection of processes from one publication/dataset"""
    case_id: str
    organism: str
    citation: str
    processes: Dict[str, Process] = field(default_factory=dict)


# ============================================================
# Benchmark Dataset Level
# ============================================================

@dataclass
class BenchmarkDataset:
    """Top-level benchmarking dataset"""
    metadata: Dict[str, str] = field(default_factory=dict)
    # metadata should include: name, version, description, preprocessing_version
    
    case_studies: Dict[str, CaseStudy] = field(default_factory=dict)


# ============================================================
# PyTree Registration (critical for JAX compatibility)
# ============================================================

tree_util.register_pytree_node(
    TimeAxis,
    lambda obj: ((obj.start, obj.end), (obj.unit, obj.time_reference)),
    lambda data, children: TimeAxis(data[0], children[0], children[1], data[1])
)

tree_util.register_pytree_node(
    RawTimeSeries,
    lambda obj: ((obj.timepoints, obj.values, obj.measurement_std), ()),
    lambda data, children: RawTimeSeries(*children)
)

tree_util.register_pytree_node(
    SplineRepresentation,
    lambda obj: (
        (obj.breakpoints, obj.coefficients, obj.fit_residual_std),
        (obj.type, obj.discontinuous, obj.notes)
    ),
    lambda data, children: SplineRepresentation(
        data[0], children[0], children[1], data[1], children[2], data[2]
    )
)

tree_util.register_pytree_node(
    TimeSeries,
    lambda obj: ((obj.raw, obj.spline), (obj.name, obj.canonical_name, obj.unit, obj.role)),
    lambda data, children: TimeSeries(data[0], data[1], data[2], data[3], *children)
)

tree_util.register_pytree_node(
    StaticVariable,
    lambda obj: ((obj.value,), (obj.unit,)),
    lambda data, children: StaticVariable(children[0], data[0])
)

tree_util.register_pytree_node(
    FeedComponent,
    lambda obj: ((obj.concentration,), (obj.unit,)),
    lambda data, children: FeedComponent(children[0], data[0])
)

tree_util.register_pytree_node(
    Feed,
    lambda obj: ((obj.density, obj.components), (obj.name, obj.density_unit)),
    lambda data, children: Feed(data[0], children[0], data[1], children[1])
)

tree_util.register_pytree_node(
    ReactorProperties,
    lambda obj: ((obj.working_volume, obj.density), (obj.volume_unit,)),
    lambda data, children: ReactorProperties(children[0], data[0], children[1])
)

tree_util.register_pytree_node(
    Process,
    lambda obj: (
        (obj.time, obj.states, obj.controls, obj.feeds, 
         obj.static_parameters, obj.event_times, obj.reactor),
        (obj.process_id, obj.process_type, obj.replicate_id)
    ),
    lambda data, children: Process(
        data[0], data[1], data[2], *children
    )
)

tree_util.register_pytree_node(
    CaseStudy,
    lambda obj: ((obj.processes,), (obj.case_id, obj.organism, obj.citation)),
    lambda data, children: CaseStudy(data[0], data[1], data[2], children[0])
)

tree_util.register_pytree_node(
    BenchmarkDataset,
    lambda obj: ((obj.case_studies,), (obj.metadata,)),
    lambda data, children: BenchmarkDataset(data[0], children[0])
)
