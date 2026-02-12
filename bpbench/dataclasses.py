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
    density_unit: str  # typically "kg/L"


@dataclass
class VolumeChange:
    """
    Represents a volume change event (discrete or continuous).
    
    Volume changes can be:
    - Controlled (e.g., programmed feed rate) or modeled (e.g., evaporation)
    - Discrete (e.g., bolus addition) or continuous (e.g., continuous feed)
    """
    name: str
    controlled: bool  # True if controlled, False if modeled
    continuous: bool  # True if continuous, False if discrete
    unit: str  # e.g. "L/h" for continuous, "L" for discrete
    feed_medium: Optional[str] = None  # Reference to feed name in Process.feeds
    timeseries: Optional[TimeSeries] = None  # For continuous changes
    timepoints: Optional[jnp.ndarray] = None  # For discrete changes (when they occur)
    values: Optional[jnp.ndarray] = None  # For discrete changes (volumes added)


@dataclass
class Volume:
    """
    Container for all volume-related information in a process.
    
    This separates volume from states and controls since volume is special:
    - Not a classic "state" variable
    - Not purely a "control" (can be both controlled and modeled)
    - Affected by multiple operations (feeds, sampling, evaporation)
    """
    volume_changes: Dict[str, VolumeChange] = field(default_factory=dict)
    initial_volume: Optional[float] = None
    volume_unit: str = "L"
    
    def validate_volume_consistency(self, time_axis: Optional[TimeAxis] = None, 
                                   final_volume: Optional[float] = None) -> tuple[bool, str]:
        """
        Validate that volume changes sum to expected final volume.
        
        Returns:
            (is_valid, message): Tuple of validation result and descriptive message
        """
        if self.initial_volume is None:
            return (True, "No initial volume specified, skipping validation")
        
        if not self.volume_changes:
            return (True, "No volume changes to validate")
        
        # Calculate total volume change
        total_change = 0.0
        messages = []
        
        for name, change in self.volume_changes.items():
            if change.continuous and change.timeseries is not None:
                # For continuous changes, integrate over time
                if time_axis is not None and change.timeseries.raw is not None:
                    # Simple trapezoidal integration
                    times = change.timeseries.raw.timepoints
                    rates = change.timeseries.raw.values
                    if len(times) > 1:
                        dt = jnp.diff(times)
                        avg_rates = (rates[:-1] + rates[1:]) / 2.0
                        change_vol = jnp.sum(dt * avg_rates)
                        total_change += float(change_vol)
                        messages.append(f"  {name}: +{change_vol:.2f} {self.volume_unit} (continuous)")
            elif not change.continuous and change.values is not None:
                # For discrete changes, sum all values
                change_vol = float(jnp.sum(change.values))
                total_change += change_vol
                messages.append(f"  {name}: +{change_vol:.2f} {self.volume_unit} (discrete)")
        
        calculated_final = self.initial_volume + total_change
        
        if final_volume is not None:
            diff = abs(calculated_final - final_volume)
            rel_diff = diff / final_volume if final_volume > 0 else 0
            
            messages.insert(0, f"Initial volume: {self.initial_volume:.2f} {self.volume_unit}")
            messages.append(f"Total change: {total_change:.2f} {self.volume_unit}")
            messages.append(f"Calculated final: {calculated_final:.2f} {self.volume_unit}")
            messages.append(f"Expected final: {final_volume:.2f} {self.volume_unit}")
            messages.append(f"Difference: {diff:.2f} {self.volume_unit} ({rel_diff*100:.1f}%)")
            
            if rel_diff > 0.05:  # More than 5% difference
                return (False, "Volume inconsistency detected:\n" + "\n".join(messages))
            else:
                return (True, "Volume balance OK:\n" + "\n".join(messages))
        else:
            messages.insert(0, f"Initial volume: {self.initial_volume:.2f} {self.volume_unit}")
            messages.append(f"Calculated final: {calculated_final:.2f} {self.volume_unit}")
            return (True, "Volume changes calculated:\n" + "\n".join(messages))


# ============================================================
# Process Level
# ============================================================

@dataclass
class Process:
    """
    Single experimental bioprocess run.
    
    Structure:
    - process_id: Unique identifier
    - process_type: "batch", "fed_batch", "continuous"
    - time: Time axis definition
    - dynamic_states: Time-varying biological states (biomass, substrate, product)
    - dynamic_controls: Time-varying controlled variables (pH, temperature if changed)
    - static_controls: Time-constant controlled variables (temperature if constant)
    - volume: Volume tracking with all volume changes
    - reactor: Reactor properties
    - feeds: Feed medium definitions
    - event_times: Discontinuity times
    """
    process_id: str
    process_type: str  # "batch", "fed_batch", "continuous"
    replicate_id: Optional[str] = None  # e.g. "rep1", "rep2"

    time: Optional[TimeAxis] = None

    # New organization: separate dynamic states, dynamic controls, and static controls
    dynamic_states: Dict[str, TimeSeries] = field(default_factory=dict)
    dynamic_controls: Dict[str, TimeSeries] = field(default_factory=dict)
    static_controls: Dict[str, StaticVariable] = field(default_factory=dict)
    
    # Volume gets its own special handling
    volume: Optional[Volume] = None

    feeds: Dict[str, Feed] = field(default_factory=dict)
    
    # Keep static_parameters for other static values
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
    VolumeChange,
    lambda obj: (
        (obj.timeseries, obj.timepoints, obj.values),
        (obj.name, obj.controlled, obj.continuous, obj.unit, obj.feed_medium)
    ),
    lambda data, children: VolumeChange(
        data[0], data[1], data[2], data[3], data[4], *children
    )
)

tree_util.register_pytree_node(
    Volume,
    lambda obj: (
        (obj.volume_changes, obj.initial_volume),
        (obj.volume_unit,)
    ),
    lambda data, children: Volume(children[0], children[1], data[0])
)

tree_util.register_pytree_node(
    Process,
    lambda obj: (
        (obj.time, obj.dynamic_states, obj.dynamic_controls, obj.static_controls,
         obj.volume, obj.feeds, obj.static_parameters, obj.event_times, obj.reactor),
        (obj.process_id, obj.process_type, obj.replicate_id)
    ),
    lambda data, children: Process(
        process_id=data[0], 
        process_type=data[1], 
        replicate_id=data[2],
        time=children[0],
        dynamic_states=children[1],
        dynamic_controls=children[2],
        static_controls=children[3],
        volume=children[4],
        feeds=children[5],
        static_parameters=children[6],
        event_times=children[7],
        reactor=children[8]
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
