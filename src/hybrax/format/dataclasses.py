"""
Bioprocess Benchmarking Dataset Structure
JAX-compatible dataclasses for standardized bioprocess data
"""

import contextlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
import jax.numpy as jnp

from ._logging import get_logger
from .time_series import TimeSeries

_logger = get_logger(__name__)

# Bounds metadata: ``(lower, upper)`` with ``None`` on either side meaning
# unbounded. RMCs default to nonnegative; other carriers default unbounded.
# Bounds are pure metadata: not enforced inside RhsOde / integrator. Consumers
# (e.g. hybrax.train's loss generator) read them off the process to build
# soft-constraint penalties.
Bounds = Tuple[Optional[float], Optional[float]]
_NO_BOUNDS: Bounds = (None, None)
_DEFAULT_RMC_BOUNDS: Bounds = (0.0, None)


def _format_bounds(bounds: Bounds) -> str:
    """Format a bounds tuple for display, hiding the unbounded default.

    Shared by ``inspect.print_process_structure`` and this module's own
    auto-assumption notices. Kept here, not in ``inspect.py``, since
    ``inspect.py`` already imports from this module and the reverse would be
    a circular import.
    """
    if bounds is None or (bounds[0] is None and bounds[1] is None):
        return "unbounded"
    lo = "-inf" if bounds[0] is None else f"{bounds[0]:g}"
    hi = "+inf" if bounds[1] is None else f"{bounds[1]:g}"
    return f"[{lo}, {hi}]"


# ============================================================
# Auto-assumption notices
# ============================================================
#
# hybrax.format fills a small number of required-but-unspecified values with a
# documented default (e.g. a feed medium's missing reactor-component
# concentration) instead of erroring. Every such fill is announced through
# `_announce_assumption` so it is never silent. Use `silence_assumptions()`
# to suppress these notices when constructing/loading many processes in a
# loop, where a per-process notice would otherwise repeat once per dataset.

_ANNOUNCE_ASSUMPTIONS = True


def _announce_assumption(message: str) -> None:
    if _ANNOUNCE_ASSUMPTIONS:
        _logger.info("Assumption: %s", message)


@contextlib.contextmanager
def silence_assumptions():
    """Temporarily suppress hybrax.format's auto-assumption notices.

    Useful when constructing or loading many ``BioProcess`` objects in a
    loop (e.g. building a ``BioProcessCollection``), where a per-process
    notice would otherwise repeat once per dataset.
    """
    global _ANNOUNCE_ASSUMPTIONS
    previous = _ANNOUNCE_ASSUMPTIONS
    _ANNOUNCE_ASSUMPTIONS = False
    try:
        yield
    finally:
        _ANNOUNCE_ASSUMPTIONS = previous


@dataclass(frozen=True)
class ProcessOrdering:
    """Canonical name ordering across all derived mechanistic modules.

    Built once from a :class:`BioProcess` via
    :func:`hybrax.format.mechanistic.get_process_ordering`. ``RhsOde``,
    ``ControlSplines``, and the spline/event helpers all consume this object
    so the layout of every state/control/rate vector is determined in exactly
    one place.

    Ordering rules:

    - ``name_modeled_rates`` preserves the user-supplied insertion order of
      ``BiologicalOde.rates`` (downstream consumers such as ``hybrax.train``
      pass rate vectors in this order).
    - ``name_modeled_algebraic`` is topo-sorted by inter-algebraic
      dependencies; ties broken alphabetically.
    - All other tuples are alphabetical within their sub-group.

    State vector layout:
        ``c = [name_modeled_RMCs... | name_modeled_PVs... | V]``

    Control vector layout (output of ``ControlSplines.__call__``):
        ``u = [name_controlled_Inflows... | name_controlled_Outflows... |
        name_controlled_PVs...]``

    The first ``len(Inflows)+len(Outflows)`` entries of ``u`` are flow rates (spline
    derivatives); the remaining ``len(PVs)`` entries are direct values.
    """

    name_modeled_rates: Tuple[str, ...]
    name_modeled_algebraic: Tuple[str, ...]
    name_modeled_RMCs: Tuple[str, ...]
    name_modeled_PVs: Tuple[str, ...]
    name_modeled_Inflows: Tuple[str, ...]
    name_modeled_Outflows: Tuple[str, ...]
    name_controlled_PVs: Tuple[str, ...]
    name_controlled_Inflows: Tuple[str, ...]
    name_controlled_Outflows: Tuple[str, ...]


# TODO for later: standardize unit spelling so it might be used for unit checks


@dataclass
class DiscreteEvents:
    """Discrete event times: bolus feeds, sampling, volume jumps, and similar."""

    times: jnp.ndarray  # sorted, unique event times
    labels: Optional[list] = None
    metadata: Optional[dict] = None


# ============================================================
# Low-Level Structures
# ============================================================


@dataclass
class TimeAxis:
    """Time axis for a process: its unit, start/end bounds, and
    time_reference anchor.
    """

    unit: str  # e.g. "hours", "days"
    start: float
    end: float
    time_reference: str  # e.g. "inoculation", "first_feed", "operator_defined"


@dataclass
class StaticVariable:
    """Raw experimental value for a time-independent parameter."""

    value: float


@dataclass
class ProcessVariable:
    """One process variable: a controlled input, or a modeled state."""

    name: str  # original name from paper
    unit: str  # e.g. "g/L", "g/L/h", "°C"
    is_controlled: (
        bool  # True if this variable is a control input, False if it's a state variable
    )
    values: TimeSeries | StaticVariable
    bounds: Bounds = _NO_BOUNDS  # (lo, hi); None on either side = unbounded


@dataclass
class FeedMediumComponent:
    """One named component of a feed medium's composition."""

    name: str  # eg. "glucose", "ammonium", "inductor"
    unit: str  # e.g. "g/L", "mM"
    concentration: TimeSeries | StaticVariable
    is_controlled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.concentration, (TimeSeries, StaticVariable)):
            raise TypeError(
                "FeedMediumComponent.concentration must be a TimeSeries or "
                f"StaticVariable, got {type(self.concentration).__name__}"
            )


@dataclass
class ReactorMediumComponent:
    """One reactor-medium component: its concentration, optional pseudobatch
    series, and bounds.
    """

    name: str  # eg. "glucose", "ammonium", "inductor"
    unit: str  # e.g. "g/L", "mM"
    concentration: TimeSeries | StaticVariable
    pseudobatch_concentration: TimeSeries | StaticVariable | None = None
    bounds: Bounds = _DEFAULT_RMC_BOUNDS

    def __post_init__(self) -> None:
        if not isinstance(self.concentration, (TimeSeries, StaticVariable)):
            raise TypeError(
                "ReactorMediumComponent.concentration must be a TimeSeries or "
                f"StaticVariable, got {type(self.concentration).__name__}"
            )


@dataclass
class FeedMedium:
    """
    Feed medium definition for ODE RHS calculations.

    ``density``/``density_unit`` are bookkeeping metadata only: the
    RHS/mass-balance math (mechanistic.py) never reads them, so they default
    to the common aqueous-solution assumption (1 kg/L) rather than being
    required.
    """

    name: str
    density: float = 1.0
    density_unit: str = "kg/L"
    components: Dict[str, FeedMediumComponent] = field(default_factory=dict)


@dataclass
class ReactorMedium:
    """
    Reactor medium definition for ODE RHS calculations.

    ``density``/``density_unit`` are bookkeeping metadata only, with the same
    default as :class:`FeedMedium`.
    """

    name: str
    density: float = 1.0
    density_unit: str = "kg/L"
    components: Dict[str, ReactorMediumComponent] = field(default_factory=dict)


@dataclass
class BioProcessMetadata:
    """Static reactor characteristics."""

    name: str
    process_type: str  # "batch", "fed_batch", "continuous"
    notes: Optional[str] = None


@dataclass
class VolumeChange:
    """Base class for volume change events, discrete or continuous.

    Volume changes are saved in the volume unit (L, m3, or kg), not as a rate.
    """

    name: str
    # e.g. "L", "m3", "kg"; not "L/h" or "kg/h" derived units.
    unit: str
    is_controlled: bool  # True if controlled, False if modeled
    is_continuous: bool  # True if continuous, False if discrete
    values: TimeSeries  # For continuous changes (cumulative or rate)


@dataclass
class Inflow(VolumeChange):
    """
    Volume change from a feed (inflow). Includes the feed medium composition.

    All delta values should be >= 0.
    """

    feed_medium: FeedMedium


@dataclass
class Outflow(VolumeChange):
    """
    Volume change from sampling/removal (outflow). No feed medium.

    All delta values should be <= 0.
    """

    retention: Dict[str, float] = field(default_factory=dict)
    # sigma in [0, 1] per reactor-medium component name. Missing key => 0
    # (species fully leaves with the outflow: sampling/harvest, the only
    # behavior before this field existed). sigma=1 => fully retained (e.g.
    # cells in perfusion; solutes in evaporation). sigma models *reduced
    # removal relative to a well-mixed outlet only*: it cannot represent a
    # component being enriched in the outflow above bulk concentration (e.g.
    # preferential stripping/volatilization). Only meaningful on a
    # continuous Outflow (is_continuous=True); a non-empty value here on a
    # discrete Outflow is rejected at construction/validation time. No
    # physical meaning on Inflow (feeds add, never retain).


def _check_outflow_retention(
    vc_name: str, vc: "Outflow", rmc_names: Union[Set[str], frozenset]
) -> List[str]:
    """Return validation errors for one Outflow's retention (empty = valid).

    Single source of truth for the three retention rules, shared by
    :func:`BioProcess.__post_init__`, :func:`mechanistic.get_process_ordering`,
    and :func:`validate.validate_outflow_retention`. Each enforces it at a
    different point (construction, RHS-build, opt-in audit) since ``Outflow``
    is a plain mutable dataclass and construction-time validity does not
    guarantee ongoing validity.
    """
    if not vc.retention:
        return []
    if not vc.is_continuous:
        return [
            (
                f"Outflow {vc_name!r} sets retention {vc.retention!r} but is "
                "discrete (is_continuous=False). Retention is only implemented "
                "for continuous Outflows; setting it on a discrete Outflow would "
                "be silently ignored by the RHS ODE."
            )
        ]
    errors: List[str] = []
    unknown = [name for name in vc.retention if name not in rmc_names]
    if unknown:
        errors.append(
            f"Outflow {vc_name!r} retention references unknown reactor "
            f"component(s): {unknown}."
        )
    out_of_range = {
        name: value for name, value in vc.retention.items() if not 0.0 <= value <= 1.0
    }
    if out_of_range:
        errors.append(
            f"Outflow {vc_name!r} retention value(s) out of [0, 1]: {out_of_range}."
        )
    return errors


@dataclass
class Volume:
    """Container for all volume-related information in a process.

    Kept separate from states and controls because volume can be both
    controlled and modeled, and is affected by multiple operations (feeds,
    sampling, evaporation).
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
    added by hybrax.format from the existing :class:`VolumeChange` machinery and
    are not part of this block.

    Attributes:
        algebraic: Mapping ``name -> expression string``. Algebraic (no time
            derivative), recomputed every RHS call. Must be acyclic.
        rates: Mapping ``name -> Bounds``. Names of abstract specific rates
            that the runtime supplies; ``len(rates)`` is the rate-vector
            dimension. Each value is a ``(lower, upper)`` tuple with ``None``
            on either side meaning unbounded; use ``(None, None)`` (i.e.
            ``_NO_BOUNDS``) for rates without bounds.
        derivatives: Mapping ``state_name -> expression string`` giving the
            biological contribution to ``d(state)/dt``. Every dynamic state
            must have an entry; use ``"0"`` to declare no biological
            dynamics.
    """

    algebraic: Dict[str, str] = field(default_factory=dict)
    rates: Dict[str, Bounds] = field(default_factory=dict)
    derivatives: Dict[str, str] = field(default_factory=dict)


def _format_biological_ode_lines(bo: BiologicalOde, prefix: str = "") -> List[str]:
    """Render a BiologicalOde's algebraic/rates/derivatives as display lines.

    Shared by `_auto_generate_biological_ode`'s assumption notice and
    `inspect.print_process_structure`. Kept here, next to `BiologicalOde`
    itself, for the same import-direction reason as `_format_bounds`.
    """
    lines: List[str] = []
    if bo.algebraic:
        lines.append(f"{prefix}Algebraic ({len(bo.algebraic)}):")
        for name, expr in bo.algebraic.items():
            lines.append(f"{prefix}  {name} = {expr}")
    if bo.rates:
        lines.append(f"{prefix}Rates ({len(bo.rates)}):")
        for name, bounds in bo.rates.items():
            lines.append(f"{prefix}  {name}: bounds={_format_bounds(bounds)}")
    if bo.derivatives:
        lines.append(f"{prefix}Derivatives ({len(bo.derivatives)}):")
        for name, expr in bo.derivatives.items():
            lines.append(f"{prefix}  d({name})/dt|biological = {expr}")
    return lines


def _auto_generate_biological_ode(process: "BioProcess") -> BiologicalOde:
    """Generate the minimal default :class:`BiologicalOde` for a process.

    For every reactor-medium component ``c`` the derivative is
    ``q_<c> * <biomass>``; for every dynamic (non-controlled, non-static)
    process variable ``p`` it is ``r_<p>``. Static process variables
    (``StaticVariable`` values) are skipped: they have no biological
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

    bo = BiologicalOde(algebraic={}, rates=rates, derivatives=derivatives)
    if _ANNOUNCE_ASSUMPTIONS:
        detail = "\n".join(_format_biological_ode_lines(bo, prefix="  "))
        _logger.info(
            "Assumption: process.biological_ode not provided; auto-generating "
            f"a default using reactor component {biomass_name!r} as the "
            f"biomass/growth driver. Pass an explicit BioProcess.biological_ode "
            f"to override:\n{detail}"
        )
    return bo


def _announce_missing_inflow_concentrations(process: "BioProcess") -> None:
    """Announce reactor components omitted from each Inflow feed medium.

    Omitted components remain absent from the caller-owned mapping and are
    interpreted as zero when mechanistic feed-composition vectors are built.
    An entirely missing ``feed_medium`` remains a hard error elsewhere.
    """
    rmc_names = tuple(process.reactor_medium.components)
    for vc_name, vc in process.volume.volume_changes.items():
        if not isinstance(vc, Inflow) or vc.feed_medium is None:
            continue
        feed = vc.feed_medium
        missing = [rmc for rmc in rmc_names if rmc not in feed.components]
        if missing:
            _announce_assumption(
                f"feed medium {feed.name!r} of Inflow {vc_name!r} did not define "
                f"a concentration for reactor component(s) {missing}; assuming 0."
            )


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


# ============================================================
# Process Level
# ============================================================


@dataclass
class BioProcess:
    """Single experimental bioprocess run: time axis, volume, media, and variables.

    ``biological_ode`` auto-generates from ``reactor_medium``/
    ``process_variables`` if not supplied; see
    ``_auto_generate_biological_ode``.
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
        if self.reactor_medium is None:
            raise ValueError("BioProcess.reactor_medium is required")
        if self.time_axis is None:
            raise ValueError("BioProcess.time_axis is required")
        if self.biological_ode is None:
            self.biological_ode = _auto_generate_biological_ode(self)
        _announce_missing_inflow_concentrations(self)
        rmc_names = set(self.reactor_medium.components)
        retention_errors = [
            error
            for vc_name, vc in self.volume.volume_changes.items()
            if isinstance(vc, Outflow)
            for error in _check_outflow_retention(vc_name, vc, rmc_names)
        ]
        if retention_errors:
            raise ValueError("\n".join(retention_errors))


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
    (e.g. ``hybrax.train``'s LOO orchestrator) can rely on it.
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
    remains a free-form dict for arbitrary provenance (e.g. hybrax.train's
    namespaced metadata block).
    """

    case_id: Optional[str] = None
    organism: Optional[str] = None
    citation: Optional[str] = None
    metadata: Optional[Dict] = None
    processes: Dict[str, BioProcess] = field(default_factory=dict)
