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
    """
    Fitted spline representation of time series data.
    
    The spline is segmented based on discontinuities (e.g., process.event_times).
    Each segment has its own coefficients, allowing for different polynomial
    behavior in different time regions.
    
    For cumulative data (like feed volumes), this representation allows:
    - Smooth interpolation of cumulative values
    - Computation of rates (derivatives) at any time point
    - Handling of discontinuities in the rate profile
    """
    type: str  # e.g. "cubic_hermite", "linear", "zero_order_hold"
    breakpoints: jnp.ndarray  # shape (K,), segment boundaries (including start and end)
    coefficients: jnp.ndarray  # shape (M, C), M segments with C coefficients each
    discontinuous: bool = False  # True if spline has discontinuities
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
    density_unit: str = "kg/L"  # typically "kg/L"


@dataclass
class VolumeChange:
    """
    Represents a volume change event (discrete or continuous).
    
    Volume changes can be:
    - Controlled (e.g., programmed feed rate) or modeled (e.g., evaporation)
    - Discrete (e.g., bolus addition) or continuous (e.g., continuous feed)
    
    For continuous changes:
    - timeseries.raw should contain the originally measured data (cumulative volumes in L or rates in L/h)
    - timeseries.spline should contain the fitted spline representation
    - The unit field indicates whether data is cumulative ("L") or rate ("L/h")
    
    Feed validation:
    - If feed_medium is specified, the referenced Feed must exist in Process.feeds
    - All dynamic state concentrations in the feed should be defined
    """
    name: str
    controlled: bool  # True if controlled, False if modeled
    continuous: bool  # True if continuous, False if discrete
    unit: str  # e.g. "L" for cumulative volumes, "L/h" for rates, "L" for discrete
    feed_medium: Optional[str] = None  # Reference to feed name in Process.feeds
    timeseries: Optional[TimeSeries] = None  # For continuous changes (cumulative or rate)
    timepoints: Optional[jnp.ndarray] = None  # For discrete changes (when they occur)
    values: Optional[jnp.ndarray] = None  # For discrete changes (volumes added)
    feed: Optional[Feed] = None  # Feed composition (inline definition alternative to feed_medium)


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
                # For continuous changes, check if data is cumulative or rate
                if change.timeseries.raw is not None:
                    times = change.timeseries.raw.timepoints
                    values = change.timeseries.raw.values
                    
                    if len(times) > 1:
                        # Check unit to determine if cumulative or rate
                        if change.unit == "L" or change.unit == self.volume_unit:
                            # Cumulative volume: final - initial
                            change_vol = float(values[-1] - values[0])
                        elif "/" in change.unit:
                            # Rate (e.g., "L/h"): integrate using trapezoidal rule
                            dt = jnp.diff(times)
                            avg_rates = (values[:-1] + values[1:]) / 2.0
                            change_vol = float(jnp.sum(dt * avg_rates))
                        else:
                            # Unknown unit, assume cumulative
                            change_vol = float(values[-1] - values[0])
                        
                        total_change += change_vol
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
    
    def validate_feed_components(self, process_feeds: Dict[str, Feed], 
                                 dynamic_variables: Dict[str, TimeSeries]) -> tuple[bool, str]:
        """
        Validate that feed compositions are properly defined for volume changes.
        
        For each VolumeChange with a feed_medium reference:
        - The referenced feed must exist in process_feeds
        - Warning if feed components don't cover all dynamic variables
        
        Args:
            process_feeds: Dictionary of Feed objects from Process.feeds
            dynamic_variables: Dictionary of TimeSeries from Process.dynamic_variables
            
        Returns:
            (is_valid, message): Tuple of validation result and descriptive message
        """
        messages = []
        all_valid = True
        
        for vc_name, vc in self.volume_changes.items():
            # Check if this volume change has a feed
            feed = None
            if vc.feed_medium is not None:
                # Reference to Process.feeds
                if vc.feed_medium not in process_feeds:
                    messages.append(f"ERROR: VolumeChange '{vc_name}' references feed '{vc.feed_medium}' "
                                  f"which is not defined in Process.feeds")
                    all_valid = False
                else:
                    feed = process_feeds[vc.feed_medium]
            elif vc.feed is not None:
                # Inline feed definition
                feed = vc.feed
            
            # If there's a feed, validate component coverage
            if feed is not None:
                missing_components = []
                for var_name in dynamic_variables.keys():
                    if var_name not in feed.components:
                        missing_components.append(var_name)
                
                if missing_components:
                    messages.append(f"WARNING: VolumeChange '{vc_name}' feed '{feed.name}' "
                                  f"is missing concentrations for dynamic variables: {missing_components}")
        
        if not messages:
            return (True, "All feed components properly defined")
        
        return (all_valid, "\n".join(messages))


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
    - replicate_id: Replicate identifier (e.g., "R1", "rep1")
    - time: Time axis definition
    - dynamic_variables: Time-varying variables (biomass, substrate, product, temperature, etc.)
    - static_variables: Time-constant variables (inductor strength, initial concentrations, etc.)
    - volume: Volume tracking with all volume changes (feeds, sampling, evaporation)
    - reactor: Reactor properties
    - feeds: Feed medium definitions
    - event_times: Discontinuity times
    """
    process_id: str
    process_type: str  # "batch", "fed_batch", "continuous"
    replicate_id: Optional[str] = None  # e.g. "rep1", "rep2", "R1"

    time: Optional[TimeAxis] = None

    # Unified dynamic variables (combines what were states and controls)
    dynamic_variables: Dict[str, TimeSeries] = field(default_factory=dict)
    
    # Unified static variables (combines what were static_controls and static_parameters)
    static_variables: Dict[str, StaticVariable] = field(default_factory=dict)
    
    # Volume gets its own special handling (includes cumulative feed data)
    volume: Optional[Volume] = None

    feeds: Dict[str, Feed] = field(default_factory=dict)

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
        (obj.timeseries, obj.timepoints, obj.values, obj.feed),
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
        (obj.time, obj.dynamic_variables, obj.static_variables,
         obj.volume, obj.feeds, obj.event_times, obj.reactor),
        (obj.process_id, obj.process_type, obj.replicate_id)
    ),
    lambda data, children: Process(
        process_id=data[0], 
        process_type=data[1], 
        replicate_id=data[2],
        time=children[0],
        dynamic_variables=children[1],
        static_variables=children[2],
        volume=children[3],
        feeds=children[4],
        event_times=children[5],
        reactor=children[6]
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
