"""
Bioprocess Benchmarking Dataset Structure
JAX-compatible dataclasses for standardized bioprocess data
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union
import jax.numpy as jnp

# TODO for later: standardize unit spelling so it might be used for unit checks


# ============================================================
# Modelling Structures - CURRENTLY PLACEHOLDERS
# ============================================================

@dataclass
class SplineRepresentation:
    """
    Serializable spline representation that can reconstruct an interpax spline.

    Stores per-segment control points (x, y) padded to fixed shapes so that
    all splines in a dataset share the same array dimensions.  This is critical
    for JAX JIT compilation (no shape-driven recompilation).

    Reconstruction:
        For each segment *i* (``i < n_segments``), build
        ``interpax.CubicSpline(x[i, :n[i]], y[i, :n[i]])`` and evaluate
        within ``[segment_boundaries[i], segment_boundaries[i+1]]``.
    """
    kind: str  # e.g. "interpax_cubic", "smoothing_bspline_approx"
    x: jnp.ndarray  # shape (max_segments, max_ctrl_points) – padded
    y: jnp.ndarray  # shape (max_segments, max_ctrl_points) – padded
    n: jnp.ndarray  # shape (max_segments,) – valid point count per segment
    n_segments: int
    segment_boundaries: jnp.ndarray  # shape (max_segments + 1,) – padded
    bc_type: str = "natural"
    spline_metadata: Optional[dict] = None  # e.g. {"s": 0.1, "n_ctrl": 128}


@dataclass
class DiscreteEvents:
    """
    Stores discrete event times (bolus feeds, sampling, volume jumps, etc.).
    """
    times: jnp.ndarray  # sorted, unique event times
    labels: Optional[list] = None
    metadata: Optional[dict] = None


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
    spline: Optional[SplineRepresentation] = None


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
class BaseVolumeChange:
    """
    Base class for volume change events (discrete or continuous).

    Note: volume changes are saved in the volume unit (i.e., L, m3, kg), not as a rate.
    """
    name: str
    unit: str  # e.g. "L", "m3", "kg", not allowed is "L/h" or "kg/h" as they are usually derived values
    is_controlled: bool  # True if controlled, False if modeled
    is_continuous: bool  # True if continuous, False if discrete
    values: TimeSeries  # For continuous changes (cumulative or rate)


@dataclass
class FeedVolumeChange(BaseVolumeChange):
    """
    Volume change from a feed (inflow). Includes the feed medium composition.

    All delta values should be >= 0.
    """
    feed_medium: FeedMedium


@dataclass
class SampleVolumeChange(BaseVolumeChange):
    """
    Volume change from sampling (outflow). No feed medium.

    All delta values should be <= 0.
    """
    pass


# Union type alias for convenience
VolumeChange = Union[FeedVolumeChange, SampleVolumeChange]


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
    discrete_events: Optional[DiscreteEvents] = None


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
