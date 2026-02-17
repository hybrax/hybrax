"""
Bioprocess Benchmarking Dataset Structure
JAX-compatible dataclasses for standardized bioprocess data
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import jax.numpy as jnp

# TODO for later: standardize unit spelling so it might be used for unit checks


# ============================================================
# Modelling Structures - CURRENTLY PLACEHOLDERS
# ============================================================

@dataclass
class SplineRepresentation:
    """
    Fitted spline representation of TimeSeries and StaticVariable data.
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
    Time axis definition for a bioprocess. Here critically 
      * the unit of time is referenced,
      * the start and end times are defined.
    """
    unit:  str  # e.g. "hours", "days"
    start: float
    end:   float
    time_reference: str  # e.g. "inoculation", "first_feed", "operator_defined"


@dataclass
class StaticVariable:
    """
    Raw experimental value for a time-independent parameter
    """
    value: float


@dataclass
class TimeSeries:
    """
    Raw experimental time-dependent measurements
    """
    timepoints: jnp.ndarray
    values:     jnp.ndarray


@dataclass
class ProcessVariable:
    """
    Process variable
    """
    name: str  # original name from paper
    unit: str  # e.g. "g/L", "g/L/h", "°C" 
    is_controlled: bool # True if this variable is a control input, False if it's a state variable
    values: TimeSeries | StaticVariable
    spline: Optional[SplineRepresentation] = None

@dataclass
class FeedMediumComponent:
    """
    Single component in a feed medium
    """
    name: str # eg. "glucose", "ammonium", "inductor"
    unit: str  # e.g. "g/L", "mM"
    concentration: TimeSeries | StaticVariable
    is_controlled: bool

@dataclass
class ReactorMediumComponent:
    """
    Single component in the bioreactor
    """
    name: str # eg. "glucose", "ammonium", "inductor"
    unit: str  # e.g. "g/L", "mM"
    concentration: TimeSeries | StaticVariable
    is_intracellular: bool # if True, this component is intracellular (e.g., X_measured = X_active + P) and should be treated differently in mass balance calculations


@dataclass
class FeedMedium:
    """
    Feed medium definition for mass balance calculations
    """
    name: str
    density: float # often assumed as 1 kg/L for aqueous solutions, but can be specified if known
    density_unit: str  # typically "kg/L"
    components: Dict[str, FeedMediumComponent] = field(default_factory=dict)

@dataclass
class ReactorMedium:
    """
    Feed medium definition for mass balance calculations
    """
    name: str
    density: float
    density_unit: str  # typically "kg/L"
    components: Dict[str, ReactorMediumComponent] = field(default_factory=dict)

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
    is_controlled: bool  # True if controlled, False if modeled
    is_continuous: bool  # True if continuous, False if discrete
    feed_medium: FeedMedium  # Reference to feed name in Process.feeds
    values: TimeSeries  # For continuous changes (cumulative or rate)


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
    unit: str  # e.g. "L", "m3", "kg"
    volume_changes: Dict[str, VolumeChange] = field(default_factory=dict)

# ============================================================
# Process Level
# ============================================================

@dataclass
class BioProcess:
    """
    Single experimental bioprocess run.
    
    Structure: # TODO for later!
    """
    metadata: BioProcessMetadata
    time_axis: TimeAxis
    volume: Volume
    reactor_medium: ReactorMedium
    process_variables: Dict[str, ProcessVariable] = field(default_factory=dict)
    

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
