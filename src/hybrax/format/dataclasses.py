"""
Bioprocess Benchmarking Dataset Structure
JAX-compatible dataclasses for standardized bioprocess data
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import jax.numpy as jnp

from .time_series import TimeSeries

# Bounds metadata: ``(lower, upper)`` with ``None`` on either side meaning
# unbounded. Default ``(None, None)`` is unbounded on both sides. Bounds are
# pure metadata — not enforced inside RhsOde / UserDefinedRhsOde / build_q_func
# / integrator. Downstream consumers (e.g. bp-train's loss generator) read
# them off the process to build soft-constraint penalties.
Bounds = Tuple[Optional[float], Optional[float]]
_NO_BOUNDS: Bounds = (None, None)

# TODO for later: standardize unit spelling so it might be used for unit checks


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

    unit: str  # e.g. "hours", "days"
    start: float
    end: float
    time_reference: str  # e.g. "inoculation", "first_feed", "operator_defined"


@dataclass
class StaticVariable:
    """
    Raw experimental value for a time-independent parameter
    """

    value: float


@dataclass
class ProcessVariable:
    """
    Process variable
    """

    name: str  # original name from paper
    unit: str  # e.g. "g/L", "g/L/h", "°C"
    is_controlled: (
        bool  # True if this variable is a control input, False if it's a state variable
    )
    values: TimeSeries | StaticVariable
    bounds: Bounds = _NO_BOUNDS  # (lo, hi); None on either side = unbounded


@dataclass
class FeedMediumComponent:
    """
    Single component in a feed medium
    """

    name: str  # eg. "glucose", "ammonium", "inductor"
    unit: str  # e.g. "g/L", "mM"
    concentration: TimeSeries | StaticVariable
    is_controlled: bool = False


@dataclass
class ReactorMediumComponent:
    """
    Single component in the bioreactor
    """

    name: str  # eg. "glucose", "ammonium", "inductor"
    unit: str  # e.g. "g/L", "mM"
    concentration: TimeSeries | StaticVariable
    is_intracellular: bool  # if True, this component is intracellular (e.g., X_measured = X_active + P) and should be treated differently in ODE RHS calculations
    bounds: Bounds = _NO_BOUNDS  # (lo, hi); None on either side = unbounded


@dataclass
class FeedMedium:
    """
    Feed medium definition for ODE RHS calculations.
    """

    name: str
    density: float  # often assumed as 1 kg/L for aqueous solutions, but can be specified if known
    density_unit: str  # typically "kg/L"
    components: Dict[str, FeedMediumComponent] = field(default_factory=dict)


@dataclass
class ReactorMedium:
    """
    Reactor medium definition for ODE RHS calculations.
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
    Base class for volume change events (discrete or continuous).

    Note: volume changes are saved in the volume unit (i.e., L, m3, kg), not as a rate.
    """

    name: str
    unit: str  # e.g. "L", "m3", "kg", not allowed is "L/h" or "kg/h" as they are usually derived values
    is_controlled: bool  # True if controlled, False if modeled
    is_continuous: bool  # True if continuous, False if discrete
    values: TimeSeries  # For continuous changes (cumulative or rate)


@dataclass
class FeedVolumeChange(VolumeChange):
    """
    Volume change from a feed (inflow). Includes the feed medium composition.

    All delta values should be >= 0.
    """

    feed_medium: FeedMedium


@dataclass
class SampleVolumeChange(VolumeChange):
    """
    Volume change from sampling (outflow). No feed medium.

    All delta values should be <= 0.
    """


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
    bounds: Bounds = _NO_BOUNDS  # (lo, hi) on V; None on either side = unbounded


# ============================================================
# User-defined biological ODE
# ============================================================


@dataclass
class RateDecl:
    """Declaration of one abstract specific-rate symbol used in a
    :class:`BiologicalOde` block.

    Rates are abstract placeholders: their values are supplied at call time
    by the runtime (today: spline-fitted from concentrations; later: NN output
    from bp-train). Bounds are pure metadata for downstream loss generators.
    """

    bounds: Bounds = _NO_BOUNDS


@dataclass
class BiologicalOde:
    """User-defined per-state biological RHS expressions.

    Describes only the *biological* part of ``dc/dt``. Physical contributions
    (feed inflow, dilution, sample outflow, volume dynamics) continue to be
    added by bp-format from the existing :class:`VolumeChange` machinery and
    are not part of this block.

    Attributes
    ----------
    derived:
        Mapping ``name -> expression string``. Algebraic (no time derivative);
        recomputed every RHS call. Must be acyclic.
    rates:
        Mapping ``name -> RateDecl``. Names of abstract specific rates that
        the runtime supplies; ``len(rates)`` is the rate-vector dimension.
    derivatives:
        Mapping ``state_name -> expression string`` giving the biological
        contribution to ``d(state)/dt``. Every dynamic state must have an
        entry; use ``"0"`` to declare "no biological dynamics".
    """

    derived: Dict[str, str] = field(default_factory=dict)
    rates: Dict[str, RateDecl] = field(default_factory=dict)
    derivatives: Dict[str, str] = field(default_factory=dict)


# ============================================================
# Process Level
# ============================================================


@dataclass
class BioProcess:
    """
    Single experimental bioprocess run.

    Structure: # TODO for later!
    """

    metadata: Optional[BioProcessMetadata]
    time_axis: TimeAxis
    volume: Volume
    reactor_medium: ReactorMedium
    process_variables: Dict[str, ProcessVariable] = field(default_factory=dict)
    discrete_events: Optional[DiscreteEvents] = None
    biological_ode: Optional[BiologicalOde] = None


@dataclass(kw_only=True)
class AugmentedBioProcess(BioProcess):
    """A synthetic variant of an existing :class:`BioProcess`.

    Same fields as :class:`BioProcess` plus a mandatory ``parent_process``
    string referencing the parent's key in the enclosing
    :class:`BioProcessCollection` / :class:`CaseStudy`. Augmented children
    inherit the parent's structural identity (control/state schema,
    medium semantics) and must be grouped with the parent for any
    train/eval split so synthetic siblings cannot leak into a fold whose
    parent is held out.

    This is a placeholder in v1: no augmentation logic produces these
    objects yet, but the data shape is fixed so downstream packages
    (e.g. ``bp-train``'s LOO orchestrator) can rely on it.
    """

    parent_process: str


# ============================================================
# Case Study Level
# ============================================================


@dataclass
class BioProcessCollection:
    """
    Wrapper for a dict of `BioProcess` instances and optional metadata. Useful for raw
    data that's not a full-fledged case-study.
    """

    metadata: Optional[Dict] = None
    processes: Dict[str, BioProcess] = field(default_factory=dict)


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
