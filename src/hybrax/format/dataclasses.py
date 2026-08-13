"""
Bioprocess Benchmarking Dataset Structure
JAX-compatible dataclasses for standardized bioprocess data
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union
import jax.numpy as jnp

from .time_series import TimeSeries

# Bounds metadata: ``(lower, upper)`` with ``None`` on either side meaning
# unbounded. RMCs default to nonnegative; other carriers default unbounded.
# Bounds are pure metadata — not enforced inside RhsOde / integrator. Consumers
# (e.g. bp-train's loss generator) read them off the process to build
# soft-constraint penalties.
Bounds = Tuple[Optional[float], Optional[float]]
_NO_BOUNDS: Bounds = (None, None)
_DEFAULT_RMC_BOUNDS: Bounds = (0.0, None)


@dataclass(frozen=True)
class ProcessOrdering:
    """Canonical name ordering across all derived mechanistic modules.

    Built once from a :class:`BioProcess` via
    :func:`bp_format.mechanistic.get_process_ordering`. ``RhsOde``,
    ``ControlSplines``, and the spline/event helpers all consume this object
    so the layout of every state/control/rate vector is determined in exactly
    one place.

    Ordering rules:

    - ``name_modeled_rates`` preserves the user-supplied insertion order of
      ``BiologicalOde.rates`` (downstream consumers such as ``bp-train``
      pass rate vectors in this order).
    - ``name_modeled_algebraic`` is topo-sorted by inter-algebraic
      dependencies; ties broken alphabetically.
    - All other tuples are alphabetical within their sub-group.

    State vector layout:
        ``c = [name_modeled_RMCs... | name_modeled_PVs... | V]``

    Control vector layout (output of ``ControlSplines.__call__``):
        ``u = [name_controlled_FVCs... | name_controlled_SVCs... |
        name_controlled_PVs...]``

    The first ``len(FVCs)+len(SVCs)`` entries of ``u`` are flow rates (spline
    derivatives); the remaining ``len(PVs)`` entries are direct values.
    """

    name_modeled_rates: Tuple[str, ...]
    name_modeled_algebraic: Tuple[str, ...]
    name_modeled_RMCs: Tuple[str, ...]
    name_modeled_PVs: Tuple[str, ...]
    name_modeled_FVCs: Tuple[str, ...]
    name_modeled_SVCs: Tuple[str, ...]
    name_controlled_PVs: Tuple[str, ...]
    name_controlled_FVCs: Tuple[str, ...]
    name_controlled_SVCs: Tuple[str, ...]


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
    c_star_concentration: TimeSeries | StaticVariable | None = None
    bounds: Bounds = _DEFAULT_RMC_BOUNDS


@dataclass
class FeedMedium:
    """
    Feed medium definition for ODE RHS calculations.
    """

    name: str
    # Often assumed as 1 kg/L for aqueous solutions, but can be specified.
    density: float
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
    # e.g. "L", "m3", "kg"; not "L/h" or "kg/h" derived units.
    unit: str
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
    total_volume: TimeSeries | None = None
    bounds: Bounds = _NO_BOUNDS  # (lo, hi) on V; None on either side = unbounded


# ============================================================
# User-defined biological ODE
# ============================================================


@dataclass
class BiologicalOde:
    """User-defined per-state biological RHS expressions.

    Describes only the *biological* part of ``dc/dt``. Physical contributions
    (feed inflow, dilution, sample outflow, volume dynamics) continue to be
    added by bp-format from the existing :class:`VolumeChange` machinery and
    are not part of this block.

    Attributes
    ----------
    algebraic:
        Mapping ``name -> expression string``. Algebraic (no time derivative);
        recomputed every RHS call. Must be acyclic.
    rates:
        Mapping ``name -> Bounds``. Names of abstract specific rates that the
        runtime supplies; ``len(rates)`` is the rate-vector dimension. Each
        value is a ``(lower, upper)`` tuple with ``None`` on either side
        meaning unbounded; use ``(None, None)`` (i.e. ``_NO_BOUNDS``) for
        rates without bounds.
    derivatives:
        Mapping ``state_name -> expression string`` giving the biological
        contribution to ``d(state)/dt``. Every dynamic state must have an
        entry; use ``"0"`` to declare "no biological dynamics".
    """

    algebraic: Dict[str, str] = field(default_factory=dict)
    rates: Dict[str, Bounds] = field(default_factory=dict)
    derivatives: Dict[str, str] = field(default_factory=dict)


def _auto_generate_biological_ode(process: "BioProcess") -> BiologicalOde:
    """Generate the minimal default :class:`BiologicalOde` for a process.

    For every reactor-medium component ``c`` the derivative is
    ``q_<c> * <biomass>``; for every dynamic (non-controlled, non-static)
    process variable ``p`` it is ``r_<p>``. Static process variables
    (``StaticVariable`` values) are skipped — they have no biological
    derivative, matching the legacy auto-RHS semantics that zeroed their
    rate contribution.

    Rate insertion order is biomass-first reactor components followed by
    dynamic process variables, so the flat rates array layout matches the
    insertion order of :attr:`BiologicalOde.rates`.

    Auto-generation requires a ``"biomass"`` reactor-medium component
    (case-insensitive). Users who do not have a biomass component must
    define ``BioProcess.biological_ode`` themselves; the
    :meth:`BioProcess.__post_init__` skips auto-generation when the user
    already supplied a block.
    """
    if not process.reactor_medium.components:
        return BiologicalOde()
    biomass_name = next(
        (
            n
            for n in process.reactor_medium.components
            if n.strip().lower() == "biomass"
        ),
        None,
    )
    if biomass_name is None:
        raise ValueError(
            "auto-generated BiologicalOde requires a 'biomass' component "
            "in process.reactor_medium.components. Pass an explicit "
            "BioProcess.biological_ode to skip auto-generation. "
            f"Available reactor components: "
            f"{list(process.reactor_medium.components)}"
        )

    rmc_names = [biomass_name] + [
        n for n in process.reactor_medium.components if n != biomass_name
    ]
    pv_dynamic = [
        name
        for name, pv in process.process_variables.items()
        if (not pv.is_controlled) and isinstance(pv.values, TimeSeries)
    ]

    rates: Dict[str, Bounds] = {}
    for rmc in rmc_names:
        rates[f"q_{rmc}"] = _NO_BOUNDS
    for pv in pv_dynamic:
        rates[f"r_{pv}"] = _NO_BOUNDS

    derivatives: Dict[str, str] = {
        rmc: f"q_{rmc} * {biomass_name}" for rmc in rmc_names
    }
    derivatives.update({pv: f"r_{pv}" for pv in pv_dynamic})

    return BiologicalOde(algebraic={}, rates=rates, derivatives=derivatives)


# ============================================================
# Process Level
# ============================================================


@dataclass
class PseudobatchTransform:
    """
    Process-level pseudobatch transform bundle.

    `adf` stores the shared accumulated dilution factor. `feed_corrections`
    stores species-specific feed corrections needed to map between transformed
    and real concentration. `sample_compensation` and `accumulated_feeds` store
    optional helper traces for transparency/debugging.
    """

    adf: TimeSeries
    feed_corrections: Dict[str, TimeSeries]
    sample_compensation: Optional[TimeSeries] = None
    accumulated_feeds: Dict[str, TimeSeries] = field(default_factory=dict)


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
    pseudobatch_transform: Optional[PseudobatchTransform] = None

    def __post_init__(self):
        if self.volume is None:
            raise ValueError("BioProcess.volume is required")
        if self.biological_ode is None:
            self.biological_ode = _auto_generate_biological_ode(self)


@dataclass(kw_only=True)
class AugmentedBioProcess(BioProcess):
    """A synthetic variant of an existing :class:`BioProcess`.

    Same fields as :class:`BioProcess` plus a mandatory ``parent_process``
    string referencing the parent's key in the enclosing
    :class:`BioProcessCollection`. Augmented children
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
    Collection of processes, optionally identified as a published case study.

    ``case_id``/``organism``/``citation`` set (all non-empty) mark this as a
    full case study from one publication/dataset. Left ``None`` (the default),
    this is raw/intermediate data with no case-study identity. ``metadata``
    remains a free-form dict for arbitrary provenance (e.g. bp-train's
    namespaced metadata block).
    """

    case_id: Optional[str] = None
    organism: Optional[str] = None
    citation: Optional[str] = None
    metadata: Optional[Dict] = None
    processes: Dict[str, BioProcess] = field(default_factory=dict)
