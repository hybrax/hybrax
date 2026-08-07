from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np


TRAINABLE_METADATA_KEY = "bp_train_trainable"


def trainable_field(**kwargs: Any) -> Any:
    """eqx.field whose array leaves are included in optimizer updates.

    An explicit tag on a parent field overrides every descendant's own tag.
    Untagged array leaves default to frozen.
    """
    metadata = dict(kwargs.pop("metadata", {}) or {})
    metadata[TRAINABLE_METADATA_KEY] = True
    return eqx.field(metadata=metadata, **kwargs)


def frozen_field(**kwargs: Any) -> Any:
    """eqx.field whose leaves are masked out of optimizer updates."""
    metadata = dict(kwargs.pop("metadata", {}) or {})
    metadata[TRAINABLE_METADATA_KEY] = False
    return eqx.field(metadata=metadata, **kwargs)


def _resolve_effective_tag(module: eqx.Module, path: tuple) -> bool | None:
    """Walk ``module`` along ``path`` and return the effective trainable tag.

    Returns True (trainable), False (frozen), or None (untagged anywhere on
    the path; caller maps None to frozen for arrays).

    Inheritance rule: the first explicit tag encountered along the path wins;
    deeper tags are ignored.
    """
    inherited: bool | None = None
    current: Any = module
    for entry in path:
        if isinstance(entry, jtu.GetAttrKey):
            attr_name = entry.name
            if isinstance(current, eqx.Module):
                field = current.__dataclass_fields__.get(attr_name)
                if field is not None:
                    field_tag = field.metadata.get(TRAINABLE_METADATA_KEY)
                    if inherited is None and field_tag is not None:
                        inherited = field_tag
            current = getattr(current, attr_name)
        elif isinstance(entry, jtu.SequenceKey):
            current = current[entry.idx]
        elif isinstance(entry, jtu.DictKey):
            current = current[entry.key]
        else:
            current = None
    return inherited


def _partition_trainable_from_metadata(
    module: eqx.Module,
) -> tuple[eqx.Module, eqx.Module]:
    """Partition a module into (trainable, static) pytrees from field metadata."""
    leaves_with_paths, treedef = jtu.tree_flatten_with_path(module)
    bool_leaves: list[bool] = []
    for path, leaf in leaves_with_paths:
        if not eqx.is_inexact_array(leaf):
            # Activation callables, ints, dtypes, etc. cannot receive gradients;
            # always static regardless of any inherited trainable tag.
            bool_leaves.append(False)
            continue
        tag = _resolve_effective_tag(module, path)
        bool_leaves.append(bool(tag) if tag is not None else False)
    filter_spec = jtu.tree_unflatten(treedef, bool_leaves)
    return eqx.partition(module, filter_spec)


def partition_trainable(module: eqx.Module) -> tuple[eqx.Module, eqx.Module]:
    """Return (trainable, static) pytrees from field metadata.

    Array leaves under ``trainable_field()`` are trainable; everything else
    (including untagged array leaves) is static. Trainability is declared
    solely through field tags — there is no per-module custom override.
    Advanced sub-field control (e.g. freezing some MLP layers) belongs in the
    ``build_optimizer`` hook via ``optax.masked`` / ``optax.multi_transform``.

    Works on any ``eqx.Module``, including the whole ``HybridOdeWrapper``: the
    tag-inheritance rule (first explicit tag on the path wins) means untagged
    container fields let their children's tags through, so the wrapper's
    ``reaction_module`` and ``loss_module`` contribute exactly their own
    ``trainable_field()`` leaves and every other leaf stays frozen.
    """
    return _partition_trainable_from_metadata(module)


# ---------------------------------------------------------------------------
# Scalers: RAW <-> SCL transforms, one per semantic axis. Frozen, never trained.
# ---------------------------------------------------------------------------


def _offset_is_nonzero(offset: jax.Array) -> bool | None:
    """Return a concrete offset predicate, or ``None`` for a traced offset.

    NumPy is intentional: JAX operations stage even on a concrete array closed
    over by ``jit``, while ``np.asarray`` yields a Python decision for that
    normal library path. A genuinely dynamic offset rejects NumPy conversion;
    callers then use a runtime ``jnp.where`` so one compiled function handles
    both zero and non-zero replacements.
    """
    try:
        return bool(np.any(np.asarray(offset) != 0))
    except jax.errors.TracerArrayConversionError:
        return None


def _scale_with_optional_offset(
    RAW: jax.Array,
    scale: jax.Array,
    offset: jax.Array,
    *,
    offset_flag: bool | None,
) -> jax.Array:
    """Subtract a concrete offset branch, or select for a traced offset."""
    # Do not touch `offset` on the false branch: a wider zero offset would
    # promote the result dtype despite being numerically inactive.
    if offset_flag is False:
        return RAW / scale
    if offset_flag is True:
        return (RAW - offset) / scale
    # Both signs of zero matter, so unknown offsets need selection in both
    # directions: x+(+0) flips -0, x+(-0) preserves it; x-(+0) preserves -0,
    # x-(-0) flips it.
    centered = jnp.where(jnp.any(offset != 0), RAW - offset, RAW)
    return centered / scale


def _unscale_with_optional_offset(
    SCL: jax.Array,
    scale: jax.Array,
    offset: jax.Array,
    *,
    offset_flag: bool | None,
) -> jax.Array:
    """Add a concrete offset branch, or select for a traced offset."""
    unscaled = SCL * scale
    if offset_flag is False:
        return unscaled
    if offset_flag is True:
        return unscaled + offset
    return jnp.where(jnp.any(offset != 0), unscaled + offset, unscaled)


class Scaler(eqx.Module):
    """RAW <-> SCL transform for one semantic axis. Frozen, never trained.

    A scaler encodes the reparametrisation between RAW physical space and the
    SCL space the ODE solver integrates in. The default and overwhelmingly
    common case is pure division (``LinearScaler``); ``AffineScaler`` adds an
    offset (``SCL = (RAW - b) / s``) as an opt-in via the ``estimate_all_scales``
    hook return value.

    Two operation kinds are named explicitly because they diverge once an
    offset exists. **Value** ops (``RAW / scaler``, ``SCL * scaler``) map a
    quantity and its time-derivative identically under pure division, but under
    an affine transform the value subtracts the offset and the derivative does
    not (``d((RAW-b)/s)/dt = (dRAW/dt)/s``). Subtracting the offset from a
    derivative is a silent, green-suite ODE-RHS corruption, so the derivative
    ops are separate and must be called by name at every derivative site.

    Value ops are exposed via the dunders ``__rtruediv__`` (``RAW / scaler``
    -> SCL) and ``__rmul__`` (``SCL * scaler`` -> RAW); there is deliberately
    no ``__truediv__`` / ``__mul__``, so ``scaler / x`` and ``scaler * x`` raise
    ``TypeError`` loudly. Derivative ops (:meth:`scale_derivative` /
    :meth:`unscale_derivative`) are offset-free and named, so a derivative site
    cannot accidentally pick up a value transform through ``/`` or ``*`` — it
    must call the method by name. A scaler is **not** silently array-coercible:
    its rejecting ``__array__`` and NumPy dispatch guards make
    ``jnp.asarray(scaler)`` / ``np.asarray(scaler)`` raise, so every coercion
    site is a loud ``TypeError`` that must become an explicit ``.scale`` /
    ``.astype`` access (the one library coercion site is ``physical_solve.py``).
    """

    __array_priority__ = 1000
    __array_ufunc__ = None

    def __array__(self, dtype=None):
        del dtype
        raise TypeError("A Scaler cannot be coerced to a NumPy array")

    def scale_value(self, RAW: jax.Array) -> jax.Array:
        """RAW -> SCL for a value without reversed-operator dispatch."""
        return self.__rtruediv__(jnp.asarray(RAW))

    def unscale_value(self, SCL: jax.Array) -> jax.Array:
        """SCL -> RAW for a value without reversed-operator dispatch."""
        return self.__rmul__(jnp.asarray(SCL))

    # Value ops: RAW / scaler -> SCL, SCL * scaler -> RAW.
    def __rtruediv__(self, RAW: jax.Array) -> jax.Array:
        """RAW -> SCL for a VALUE (``RAW / scaler``)."""
        raise NotImplementedError

    def __rmul__(self, SCL: jax.Array) -> jax.Array:
        """SCL -> RAW for a VALUE (``SCL * scaler``)."""
        raise NotImplementedError

    def scale_derivative(self, RAW_rate: jax.Array) -> jax.Array:
        """RAW rate -> SCL rate (offset-free; ``d((RAW-b)/s)/dt = (dRAW/dt)/s``)."""
        raise NotImplementedError

    def unscale_derivative(self, SCL_rate: jax.Array) -> jax.Array:
        """SCL rate -> RAW rate (offset-free)."""
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape of one element on this axis (NOT a scalar ``width``).

        Axes are not all 1-D: ``V_in_cumulative`` is scalar ``()``, the two
        ``*_Cin`` axes are 2-D ``(n_FVCs, n_RMCs)``, and several axes are
        legitimately zero-width.
        """
        raise NotImplementedError

    # Static contract for elementwise composition. Type-only so concrete eqx
    # modules can store ``scale`` / ``offset`` as fields (a runtime @property
    # here would block those field assignments). ``_compose_scalers`` validates
    # the contract loudly at runtime.
    if TYPE_CHECKING:
        scale: jax.Array
        offset: jax.Array

    def astype(self, dtype: jnp.dtype) -> "Scaler":
        """Return a scaler with the underlying arrays cast to ``dtype``."""
        raise NotImplementedError

    def __getitem__(self, idx) -> "Scaler":
        """Sub-scaler by index (state target subsetting; value semantics)."""
        raise NotImplementedError

    # Deliberately NO __truediv__ / __mul__: see class docstring.


class LinearScaler(Scaler):
    """Pure-division scaler: ``SCL = RAW / s``. The default.

    Value (``__rtruediv__`` / ``__rmul__``) and derivative ops are all ``/s`` or
    ``*s``, so this is bit-identical to the pre-scaler ``RAW / SCALE`` /
    ``SCL * SCALE`` by construction. The underlying array (``.scale``) is
    stored verbatim — no dtype normalisation or unification — so
    ``jnp.concatenate`` promotion over composed state axes is reproduced
    exactly (a zero-width float64 axis still promotes the concat to float64).
    """

    scale: jax.Array = frozen_field()

    def __init__(self, scale: jax.Array):
        if not hasattr(scale, "shape") or not hasattr(scale, "dtype"):
            raise TypeError("LinearScaler.scale must be a JAX/NumPy array")
        self.scale = scale

    def __rtruediv__(self, RAW: jax.Array) -> jax.Array:
        if isinstance(RAW, np.ndarray):
            raise TypeError("Use Scaler.scale_value() for NumPy arrays")
        return RAW / self.scale

    def __rmul__(self, SCL: jax.Array) -> jax.Array:
        return SCL * self.scale

    def scale_derivative(self, RAW_rate: jax.Array) -> jax.Array:
        return RAW_rate / self.scale

    def unscale_derivative(self, SCL_rate: jax.Array) -> jax.Array:
        return SCL_rate * self.scale

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.scale.shape)

    @property
    def offset(self) -> jax.Array:
        return jnp.zeros_like(self.scale)

    def astype(self, dtype: jnp.dtype) -> "LinearScaler":
        return LinearScaler(jnp.asarray(self.scale, dtype=dtype))

    def __getitem__(self, idx) -> "LinearScaler":
        return LinearScaler(self.scale[idx])


class AffineScaler(Scaler):
    """Affine scaler: ``SCL = (RAW - b) / s``. Opt-in via the hook return value.

    Value ops subtract the offset (``RAW / scaler`` → ``(RAW - b)/s``;
    ``SCL * scaler`` → ``SCL * s + b``); derivative ops are offset-free
    (``/s`` and ``*s``), because ``d((RAW-b)/s)/dt = (dRAW/dt)/s``. This
    divergence is the whole point of the explicit derivative ops — a value
    ``/`` applied to a rate would silently inject ``-b/s`` into the ODE RHS.

    The default path never constructs this; a user opts in by returning an
    ``AffineScaler`` for one axis from ``estimate_all_scales``. Offsets are
    meaningful only on VALUE axes — the harness rejects a non-zero offset on
    the three rate axes. Closed-over concrete zero offsets retain
    ``LinearScaler`` bit identity through reverse-mode training. A genuinely
    dynamic traced offset uses runtime selection: values retain signed-zero
    identity, but reverse-mode signed-zero identity is not guaranteed.

    A bare ``AffineScaler.astype`` staged inside a trace uses the dynamic
    fallback. Composed zero-offset state scalers instead collapse to
    ``LinearScaler`` before casting, preserving the solver's concrete
    offset-free branch without cached public state.

    A composed **non-zero** offset does take that dynamic fallback, and the
    reason is not obvious: ``_compose_scalers`` reads the predicate from
    concrete child leaves, but the ``AffineScaler`` it returns holds the
    *concatenated* arrays, and ``jnp.concatenate`` stages inside a trace even
    when every child is a concrete closed-over constant. The composed scaler's
    ops therefore re-derive the predicate from a tracer and select at runtime.
    Measured over a real ``solve_physical_states`` with a non-zero offset and
    the module closed over: 7 of 10 composed value ops select rather than
    branch, at both float32 and float64. That is numerically correct and there
    is no bit-identity claim for a non-zero offset, so it is accepted rather
    than worked around — carrying the predicate across would reintroduce the
    public cached flag that ``eqx.tree_at`` can stale. Note it costs extra
    selects on closed-over forward paths (export, inference). The training path
    is unaffected in the sense that it already selects: there the module is a
    differentiated argument, so the offset is a genuine tracer either way.
    """

    scale: jax.Array = frozen_field()
    offset: jax.Array = frozen_field()

    def __init__(self, scale: jax.Array, offset: jax.Array):
        # Preserve user dtypes exactly; broadcast only (incompatible shapes
        # raise loudly). Rate-axis non-zero-offset rejection lives at the
        # wrapper boundary, where the semantic axis name is known.
        if not hasattr(scale, "shape") or not hasattr(scale, "dtype"):
            raise TypeError("AffineScaler.scale must be a JAX/NumPy array")
        if not hasattr(offset, "shape") or not hasattr(offset, "dtype"):
            raise TypeError("AffineScaler.offset must be a JAX/NumPy array")
        self.scale = scale
        self.offset = jnp.broadcast_to(offset, scale.shape)

    def __rtruediv__(self, RAW: jax.Array) -> jax.Array:
        if isinstance(RAW, np.ndarray):
            raise TypeError("Use Scaler.scale_value() for NumPy arrays")
        return _scale_with_optional_offset(
            RAW,
            self.scale,
            self.offset,
            offset_flag=_offset_is_nonzero(self.offset),
        )

    def __rmul__(self, SCL: jax.Array) -> jax.Array:
        return _unscale_with_optional_offset(
            SCL,
            self.scale,
            self.offset,
            offset_flag=_offset_is_nonzero(self.offset),
        )

    def scale_derivative(self, RAW_rate: jax.Array) -> jax.Array:
        return RAW_rate / self.scale

    def unscale_derivative(self, SCL_rate: jax.Array) -> jax.Array:
        return SCL_rate * self.scale

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.scale.shape)

    def astype(self, dtype: jnp.dtype) -> "AffineScaler":
        return AffineScaler(
            jnp.asarray(self.scale, dtype=dtype),
            jnp.asarray(self.offset, dtype=dtype),
        )

    def __getitem__(self, idx) -> "AffineScaler":
        return AffineScaler(self.scale[idx], self.offset[idx])


def _compose_scalers(*parts: Scaler) -> LinearScaler | AffineScaler:
    """Compose exact elementwise scalers into one concrete scaler.

    Zero-width parts stay in both concatenations because they participate in
    JAX dtype promotion. Arrays are materialized once here; value and derivative
    operations then reuse the existing concrete scaler implementations.
    """
    for part in parts:
        if type(part) not in (LinearScaler, AffineScaler):
            raise TypeError(
                "State scaler composition supports exact LinearScaler or "
                f"AffineScaler; got {type(part).__name__}"
            )

    scale = jnp.concatenate([jnp.atleast_1d(part.scale) for part in parts])
    offset = jnp.concatenate([jnp.atleast_1d(part.offset) for part in parts])
    flags = [
        _offset_is_nonzero(part.offset) for part in parts if type(part) is AffineScaler
    ]
    offset_flag = (
        True
        if any(flag is True for flag in flags)
        else None
        if None in flags
        else False
    )
    if offset_flag is False:
        return LinearScaler(scale)
    return AffineScaler(scale, offset)


def _as_scaler(value: jax.Array | Scaler) -> Scaler:
    """Promote a bare array to ``LinearScaler``; pass a ``Scaler`` through.

    The ``estimate_all_scales`` hook and the no-hook defaults both return bare
    arrays (the 21 live+frozen hooks keep doing so); this promotes them to
    ``LinearScaler`` so the default path is pure division, bit-identical to the
    pre-scaler code. A hook opts into affine scaling by returning a ``Scaler``
    (e.g. ``AffineScaler``) for the axis it wants to transform.
    """
    if isinstance(value, Scaler):
        return value
    return LinearScaler(value)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class ReactionInputs(eqx.Module):
    """All inputs to a UserReactionModule call, one per semantic axis, in SCL space.

    Built by HybridOdeWrapper at each RHS evaluation. The module reads only the
    axes it cares about; unused fields cost nothing under JIT.

    State-slice axes (from the integrated SCL state):
    - ``SCL_modeled_RMCs``: species concentrations, UNCLIPPED so MLP gradient flow
      survives transient negative excursions near depletion.
    - ``SCL_modeled_PVs``: modeled (uncontrolled, dynamic) process-variable
      states, integrated alongside the RMCs. Empty when the process has none.
    - ``SCL_modeled_V``: real reactor volume at time t (already includes the
      ``min_V`` floor applied by the wrapper).
    - ``SCL_modeled_FVCs_cumulative``: per-feed cumulative volume of each
      MODELED feed (integrated state).

    Continuous controlled FVC axes (from the controls' per-axis accessors):
    - ``SCL_controlled_FVCs_cumulative`` — per-feed cumulative volume.
    - ``SCL_controlled_FVCs_rates`` — per-feed instantaneous flow rate.
    - ``SCL_controlled_FVCs_Cin`` — per-feed Cin matrix [n_FVC, n_RMCs].

    Process variables and modeled-feed composition:
    - ``SCL_controlled_PVs`` — process-variable signals (pH, DO, T, …).
    - ``SCL_modeled_FVCs_Cin`` — Cin matrix [n_modeled_FVCs, n_RMCs] for the
      modeled feeds.
    """

    SCL_modeled_RMCs: jax.Array
    SCL_modeled_V: jax.Array
    SCL_modeled_FVCs_cumulative: jax.Array
    SCL_controlled_FVCs_cumulative: jax.Array
    SCL_controlled_FVCs_rates: jax.Array
    SCL_controlled_FVCs_Cin: jax.Array
    SCL_modeled_FVCs_Cin: jax.Array
    SCL_modeled_PVs: jax.Array = eqx.field(
        default_factory=lambda: jnp.zeros(0, dtype=jnp.float64)
    )
    SCL_controlled_PVs: jax.Array = eqx.field(
        default_factory=lambda: jnp.zeros(0, dtype=jnp.float64)
    )
    SCL_latent: jax.Array = eqx.field(
        default_factory=lambda: jnp.zeros(0, dtype=jnp.float64)
    )


class ReactionOutputs(eqx.Module):
    """Structured return value from a UserReactionModule.__call__.

    Both rate fields are in SCALED space — the module divides its physical
    output by the matching SCALE_* axis (typically via ``self.scale_*`` helpers)
    before returning. The wrapper then unscales these on the way into the
    physical RhsOde mass balance.

    Attributes
    ----------
    SCL_modeled_BiologicalOde_rates:
        Rates emitted by the BiologicalOde block, aligned with
        ``rhs_ode.name_modeled_rates``. Not 1:1 with RMCs — algebraic rates
        (e.g. ``q_X_active`` in TUB) live here without a corresponding RMC.
    SCL_modeled_FVCs_rates:
        Flow rates for modeled FeedVolumeChanges, aligned with
        ``rhs_ode.name_modeled_FVCs``. Must be non-negative — the module
        applies its own positivity transform (typically softplus) before
        scaling.
    SCL_latent_derivative:
        Continuous latent-state derivative, aligned with ``SCL_latent``.
    auxiliary:
        Optional model-defined observables that follow the solver-time save
        path. ``None`` or ``dict[str, array]`` with scalar or 1D-array leaves
        and stable keys across calls.
    """

    SCL_modeled_BiologicalOde_rates: jax.Array
    SCL_modeled_FVCs_rates: jax.Array
    SCL_latent_derivative: jax.Array = eqx.field(
        default_factory=lambda: jnp.zeros(0, dtype=jnp.float64)
    )
    auxiliary: dict[str, jax.Array] | None = None


@dataclass
class EstimatedScales:
    """Return shape for the ``estimate_all_scales`` user hook.

    The harness unpacks these into ``build_reaction_module`` kwargs and the
    constructed module stores them as frozen fields. Together they cover every
    semantic axis the module / wrapper / trainer needs to scale.

    Fields accept a bare ``jax.Array`` (promoted to ``LinearScaler`` by the
    harness — the default, bit-identical to pure division) **or** a ``Scaler``
    (e.g. ``AffineScaler`` to opt into affine scaling for one axis). Hooks keep
    returning arrays for zero-edit compatibility; a hook opts in by returning a
    ``Scaler`` for the axis it wants to transform.
    """

    SCALE_modeled_RMCs: jax.Array | Scaler
    SCALE_V_in_cumulative: jax.Array | Scaler
    SCALE_modeled_FVCs_cumulative: jax.Array | Scaler
    SCALE_controlled_FVCs_cumulative: jax.Array | Scaler
    SCALE_controlled_FVCs_rates: jax.Array | Scaler
    SCALE_controlled_FVCs_Cin: jax.Array | Scaler
    SCALE_controlled_PVs: jax.Array | Scaler
    SCALE_modeled_FVCs_Cin: jax.Array | Scaler
    SCALE_modeled_BiologicalOde_rates: jax.Array | Scaler
    SCALE_modeled_FVCs_rates: jax.Array | Scaler
    # Defaults to empty (no modeled PVs); processes with uncontrolled,
    # dynamic process variables supply a real (n_modeled_PVs,) scale.
    SCALE_modeled_PVs: jax.Array | Scaler = field(
        default_factory=lambda: jnp.zeros(0, dtype=jnp.float64)
    )


class UserReactionModule(eqx.Module):
    """Base for user-defined reaction modules.

    The module is the single source of truth for every ``SCALE_*`` vector in
    bp-train. The wrapper queries scales via ``self.reaction_module.SCALE_*``;
    the trainer reads ``SCALE_state`` to convert measurements to SCL space.

    Subclasses inherit the scale fields below — they do NOT redeclare them;
    instead they pass values to ``super().__init__(**scale_kwargs)``. Each
    ``SCALE_*`` field holds a :class:`Scaler` (frozen, never trained); the
    harness promotes bare arrays from the ``estimate_all_scales`` hook to
    :class:`LinearScaler` so the default path is pure division.

    Most ``scale_*`` / ``unscale_*`` helpers below operate on VALUES. The three
    ``*_rates`` helper pairs use derivative semantics. Under pure division
    (``LinearScaler``) a value and its time-derivative scale identically, but
    that interchangeability is NOT general: under an affine transform the
    derivative must NOT subtract the offset (``d((RAW-b)/s)/dt = (dRAW/dt)/s``).
    ``wrapper.py`` unscales the reaction module's latent derivative;
    ``physical_solve.py`` then scales the full RAW ODE derivative.
    """

    # SCALE_* fields default to zero-sized placeholders so subclasses can be
    # instantiated without passing them (handy for tests + tooling). The
    # ``HybridOdeWrapper`` constructor validates per-axis shapes; supplying
    # real values via ``estimate_all_scales`` / ``super().__init__(**kwargs)``
    # is the production path.
    SCALE_modeled_RMCs: Scaler = frozen_field(
        default_factory=lambda: LinearScaler(jnp.zeros(0, dtype=jnp.float64))
    )
    SCALE_modeled_PVs: Scaler = frozen_field(
        default_factory=lambda: LinearScaler(jnp.zeros(0, dtype=jnp.float64))
    )
    SCALE_V_in_cumulative: Scaler = frozen_field(
        default_factory=lambda: LinearScaler(jnp.asarray(1.0, dtype=jnp.float64))
    )
    SCALE_modeled_FVCs_cumulative: Scaler = frozen_field(
        default_factory=lambda: LinearScaler(jnp.zeros(0, dtype=jnp.float64))
    )
    SCALE_controlled_FVCs_cumulative: Scaler = frozen_field(
        default_factory=lambda: LinearScaler(jnp.zeros(0, dtype=jnp.float64))
    )
    SCALE_controlled_FVCs_rates: Scaler = frozen_field(
        default_factory=lambda: LinearScaler(jnp.zeros(0, dtype=jnp.float64))
    )
    SCALE_controlled_FVCs_Cin: Scaler = frozen_field(
        default_factory=lambda: LinearScaler(jnp.zeros((0, 0), dtype=jnp.float64))
    )
    SCALE_controlled_PVs: Scaler = frozen_field(
        default_factory=lambda: LinearScaler(jnp.zeros(0, dtype=jnp.float64))
    )
    SCALE_modeled_FVCs_Cin: Scaler = frozen_field(
        default_factory=lambda: LinearScaler(jnp.zeros((0, 0), dtype=jnp.float64))
    )
    SCALE_modeled_BiologicalOde_rates: Scaler = frozen_field(
        default_factory=lambda: LinearScaler(jnp.zeros(0, dtype=jnp.float64))
    )
    SCALE_modeled_FVCs_rates: Scaler = frozen_field(
        default_factory=lambda: LinearScaler(jnp.zeros(0, dtype=jnp.float64))
    )
    SCALE_latent: Scaler = frozen_field(
        default_factory=lambda: LinearScaler(jnp.zeros(0, dtype=jnp.float64))
    )

    def __init__(self, **kwargs: Any) -> None:
        """Set base scale fields (applying defaults); promote bare arrays.

        The module is the single source of truth for scales, so it normalizes
        whatever it receives — a bare array becomes ``LinearScaler``; a
        ``Scaler`` passes through. Unknown keywords raise immediately (a scale
        typo must never silently fall back to the default). Declared subclass
        fields are accepted and their defaults are applied; required fields
        remain available for custom subclass constructors to set after
        ``super().__init__()``.
        """
        module_fields = {f.name: f for f in fields(type(self))}
        scale_fields = {f.name for f in fields(UserReactionModule)}
        unknown = sorted(set(kwargs) - set(module_fields))
        if unknown:
            raise TypeError(
                f"Unknown UserReactionModule field(s): {', '.join(unknown)}"
            )
        for f in module_fields.values():
            if f.name in kwargs:
                value = kwargs[f.name]
            elif f.default is not MISSING:
                value = f.default
            elif f.default_factory is not MISSING:
                value = f.default_factory()
            else:
                continue
            if f.name in scale_fields:
                value = _as_scaler(value)
            object.__setattr__(self, f.name, value)

    # ------------------------------------------------------------------
    # Axis-dimension properties. Subclass authors size their MLPs /
    # mechanistic equations from these after ``super().__init__()``; no
    # need to thread `n_*` kwargs through `build_reaction_module`.
    # ------------------------------------------------------------------

    @property
    def n_modeled_RMCs(self) -> int:
        """Number of modeled RMCs (species)."""
        return int(self.SCALE_modeled_RMCs.shape[0])

    @property
    def n_modeled_PVs(self) -> int:
        """Number of modeled (uncontrolled, dynamic) process variables."""
        return int(self.SCALE_modeled_PVs.shape[0])

    @property
    def n_modeled_FVCs(self) -> int:
        """Number of modeled feed volume changes (per-feed state slice)."""
        return int(self.SCALE_modeled_FVCs_cumulative.shape[0])

    @property
    def n_modeled_BiologicalOde_rates(self) -> int:
        """Number of rates emitted by the BiologicalOde block."""
        return int(self.SCALE_modeled_BiologicalOde_rates.shape[0])

    @property
    def n_controlled_FVCs(self) -> int:
        """Number of continuous controlled feeds."""
        return int(self.SCALE_controlled_FVCs_cumulative.shape[0])

    @property
    def n_controlled_PVs(self) -> int:
        """Number of controlled process-variable signals."""
        return int(self.SCALE_controlled_PVs.shape[0])

    @property
    def n_latent(self) -> int:
        """Number of integrated latent-state dimensions."""
        return int(self.SCALE_latent.shape[0])

    @property
    def _state_scalers(self) -> tuple[Scaler, ...]:
        return (
            self.SCALE_modeled_RMCs,
            self.SCALE_modeled_PVs,
            self.SCALE_V_in_cumulative,
            self.SCALE_modeled_FVCs_cumulative,
        )

    @property
    def SCALE_state(self) -> Scaler:
        """Composed state scaler.

        Layout: ``[modeled_RMCs | modeled_PVs | V | modeled_FVCs_cumulative]``.
        Zero-width parts (e.g. an empty PV axis) are preserved in the concat to
        reproduce ``jnp.concatenate`` dtype promotion exactly — do NOT filter.
        """
        return _compose_scalers(*self._state_scalers)

    @property
    def SCALE_modeled_V(self) -> Scaler:
        """Real-volume scale shares the V_in_cumulative scale (same units, L)."""
        return self.SCALE_V_in_cumulative

    @property
    def SCALE_integrated_state(self) -> Scaler:
        """State scaler used by the ODE solver, incl. trailing latent state."""
        return _compose_scalers(*self._state_scalers, self.SCALE_latent)

    def __call__(self, t: jax.Array, inputs: ReactionInputs) -> ReactionOutputs:
        """Override. Inputs in SCL space; return rates in SCL space.

        Use the ``.unscale_*`` helpers when you need RAW physical values for
        chemistry / FBA / kinetic-law evaluation. Use ``.scale_*`` on the way
        back out.
        """
        raise NotImplementedError

    @property
    def latent_observables(self) -> tuple[str, ...]:
        """Names of observables that require live latent state during the solve."""
        return ()

    def observe(self, states: jax.Array) -> jax.Array:
        """Optional post-hoc observation map; default identity on physical state."""
        return states

    def initial_latent(self, RAW_phys_y0: jax.Array) -> jax.Array:
        """Initial RAW latent state appended to the physical initial state."""
        return jnp.zeros(self.n_latent, dtype=jnp.asarray(RAW_phys_y0).dtype)

    # ------------------------------------------------------------------
    # Value scale/unscale helpers (all except ``*_rates``) operate on VALUES.
    # The three rate helper pairs use ``Scaler.scale_derivative`` /
    # ``unscale_derivative`` (offset-free).
    # ------------------------------------------------------------------

    def scale_state(self, RAW_state):
        return self.SCALE_state.scale_value(RAW_state)

    def unscale_state(self, SCL_state):
        return self.SCALE_state.unscale_value(SCL_state)

    def scale_latent(self, RAW_latent):
        return self.SCALE_latent.scale_value(RAW_latent)

    def unscale_latent(self, SCL_latent):
        return self.SCALE_latent.unscale_value(SCL_latent)

    def scale_modeled_RMCs(self, RAW_modeled_RMCs):
        return self.SCALE_modeled_RMCs.scale_value(RAW_modeled_RMCs)

    def unscale_modeled_RMCs(self, SCL_modeled_RMCs):
        return self.SCALE_modeled_RMCs.unscale_value(SCL_modeled_RMCs)

    def scale_modeled_PVs(self, RAW_modeled_PVs):
        return self.SCALE_modeled_PVs.scale_value(RAW_modeled_PVs)

    def unscale_modeled_PVs(self, SCL_modeled_PVs):
        return self.SCALE_modeled_PVs.unscale_value(SCL_modeled_PVs)

    def scale_modeled_V(self, RAW_modeled_V):
        return self.SCALE_modeled_V.scale_value(RAW_modeled_V)

    def unscale_modeled_V(self, SCL_modeled_V):
        return self.SCALE_modeled_V.unscale_value(SCL_modeled_V)

    def scale_V_in_cumulative(self, RAW_V_in_cumulative):
        return self.SCALE_V_in_cumulative.scale_value(RAW_V_in_cumulative)

    def unscale_V_in_cumulative(self, SCL_V_in_cumulative):
        return self.SCALE_V_in_cumulative.unscale_value(SCL_V_in_cumulative)

    def scale_modeled_FVCs_cumulative(self, RAW_modeled_FVCs_cumulative):
        return self.SCALE_modeled_FVCs_cumulative.scale_value(
            RAW_modeled_FVCs_cumulative
        )

    def unscale_modeled_FVCs_cumulative(self, SCL_modeled_FVCs_cumulative):
        return self.SCALE_modeled_FVCs_cumulative.unscale_value(
            SCL_modeled_FVCs_cumulative
        )

    def scale_controlled_FVCs_cumulative(self, RAW_controlled_FVCs_cumulative):
        return self.SCALE_controlled_FVCs_cumulative.scale_value(
            RAW_controlled_FVCs_cumulative
        )

    def unscale_controlled_FVCs_cumulative(self, SCL_controlled_FVCs_cumulative):
        return self.SCALE_controlled_FVCs_cumulative.unscale_value(
            SCL_controlled_FVCs_cumulative
        )

    def scale_controlled_FVCs_rates(self, RAW_controlled_FVCs_rates):
        return self.SCALE_controlled_FVCs_rates.scale_derivative(
            RAW_controlled_FVCs_rates
        )

    def unscale_controlled_FVCs_rates(self, SCL_controlled_FVCs_rates):
        return self.SCALE_controlled_FVCs_rates.unscale_derivative(
            SCL_controlled_FVCs_rates
        )

    def scale_controlled_FVCs_Cin(self, RAW_controlled_FVCs_Cin):
        return self.SCALE_controlled_FVCs_Cin.scale_value(RAW_controlled_FVCs_Cin)

    def unscale_controlled_FVCs_Cin(self, SCL_controlled_FVCs_Cin):
        return self.SCALE_controlled_FVCs_Cin.unscale_value(SCL_controlled_FVCs_Cin)

    def scale_controlled_PVs(self, RAW_controlled_PVs):
        return self.SCALE_controlled_PVs.scale_value(RAW_controlled_PVs)

    def unscale_controlled_PVs(self, SCL_controlled_PVs):
        return self.SCALE_controlled_PVs.unscale_value(SCL_controlled_PVs)

    def scale_modeled_FVCs_Cin(self, RAW_modeled_FVCs_Cin):
        return self.SCALE_modeled_FVCs_Cin.scale_value(RAW_modeled_FVCs_Cin)

    def unscale_modeled_FVCs_Cin(self, SCL_modeled_FVCs_Cin):
        return self.SCALE_modeled_FVCs_Cin.unscale_value(SCL_modeled_FVCs_Cin)

    def scale_modeled_BiologicalOde_rates(self, RAW_modeled_BiologicalOde_rates):
        return self.SCALE_modeled_BiologicalOde_rates.scale_derivative(
            RAW_modeled_BiologicalOde_rates
        )

    def unscale_modeled_BiologicalOde_rates(self, SCL_modeled_BiologicalOde_rates):
        return self.SCALE_modeled_BiologicalOde_rates.unscale_derivative(
            SCL_modeled_BiologicalOde_rates
        )

    def scale_modeled_FVCs_rates(self, RAW_modeled_FVCs_rates):
        return self.SCALE_modeled_FVCs_rates.scale_derivative(RAW_modeled_FVCs_rates)

    def unscale_modeled_FVCs_rates(self, SCL_modeled_FVCs_rates):
        return self.SCALE_modeled_FVCs_rates.unscale_derivative(SCL_modeled_FVCs_rates)


# ---------------------------------------------------------------------------
# Loss API: LossInputs / LossOutputs / UserLossModule
# ---------------------------------------------------------------------------


class LossInputs(eqx.Module):
    """Everything a loss term needs for one sample, on the measurement grid.

    Built once per sample by the trainer after the shared ODE solve. Predicted
    trajectories are provided in both SCL and RAW space (scaling is a cheap
    elementwise broadcast over the leading time axis); the loss module picks
    whichever space it needs.

    Scales are NOT duplicated here — they live on ``reaction_module`` (the
    single source of truth), reachable via ``inputs.reaction_module.SCALE_*``
    or its ``scale_*`` / ``unscale_*`` helpers.

    The masks handle sparse, unaligned measurements (species sampled on
    different time grids, padded to a common length per batch):
    - ``mask_measured`` ``(n_meas, n_target)``: per-cell validity. True iff the
      ``(timepoint, species)`` pair is a real measurement; False for padding
      rows and for species not sampled at that timestamp.
    - ``mask_measured_any`` ``(n_meas,)``: ``any(mask_measured, axis=1)`` cast to
      float — per-row "is this timestep real". Multiply trajectory-wide
      penalties (e.g. bounds hinges) by this.
    - ``n_measured``: the unpadded row count for this sample.

    Failure handling: if the ODE solve bailed partway (a stiff segment hit the
    step cap), every point past the failure time is dropped BEFORE this struct is
    built — ``mask_measured`` / ``mask_measured_any`` are already zeroed on those
    rows, and ``dense_valid_time`` (below) marks the valid dense rows. All predicted
    trajectories here (measurement AND dense) are guaranteed finite: post-failure
    rows carry a finite fallback, never ``inf``/``nan``. So the ``penalty * mask``
    idiom is safe (no ``0 * inf``); just remember to gate DENSE penalties by
    ``dense_valid_time`` (the dense grid has no other failure mask).
    """

    # Predictions over measurement times (n_meas, ...)
    SCL_states: jax.Array
    RAW_states: jax.Array
    SCL_modeled_BiologicalOde_rates: jax.Array
    RAW_modeled_BiologicalOde_rates: jax.Array
    SCL_modeled_FVCs_rates: jax.Array
    RAW_modeled_FVCs_rates: jax.Array
    SCL_V: jax.Array
    RAW_V: jax.Array
    # Integrated volume before the wrapper's ``min_V`` floor. Use for physical
    # constraint losses; ``RAW_V`` is the safe value passed to the reaction model.
    RAW_V_unclamped: jax.Array
    auxiliary: dict[str, jax.Array]

    # Convenience target slice: SCL_states[:, target_state_indices]
    SCL_target_pred: jax.Array

    # Ground truth + masks
    SCL_target_measured: jax.Array
    mask_measured: jax.Array
    mask_measured_any: jax.Array
    t_measured: jax.Array
    n_measured: jax.Array

    # Single source of SCALE_* (frozen reference, not a copy)
    reaction_module: UserReactionModule

    # Training step; -1 in forward-eval contexts. For schedules / annealing.
    step: jax.Array

    # Controls-discontinuity times for this sample (from
    # ``controls.active_jump_ts``). ``None`` when the trainer ran without
    # jump-ts. Useful for masking dense points near discontinuities; passed
    # whether or not the dense grid is enabled.
    jump_ts: jax.Array | None = None

    # ---- Dense-grid view. Populated iff ``UserLossModule.dense_grid_n`` is
    # not None. All have leading dim ``dense_grid_n + n_meas_padded`` is wrong
    # -- the dense linspace has exactly ``dense_grid_n`` points and these
    # arrays are the index-gathered dense rows from the union solve, so the
    # leading dim is ``dense_grid_n``. ``dense_t`` is the linspace itself.
    dense_t: jax.Array | None = None
    dense_SCL_states: jax.Array | None = None
    dense_RAW_states: jax.Array | None = None
    dense_SCL_modeled_BiologicalOde_rates: jax.Array | None = None
    dense_RAW_modeled_BiologicalOde_rates: jax.Array | None = None
    dense_SCL_modeled_FVCs_rates: jax.Array | None = None
    dense_RAW_modeled_FVCs_rates: jax.Array | None = None
    dense_SCL_V: jax.Array | None = None
    dense_RAW_V: jax.Array | None = None
    dense_auxiliary: dict[str, jax.Array] | None = None
    # (dense_grid_n,) bool: True for dense rows at/before the solve's failure time,
    # False for post-failure rows (whose trajectory values are a finite fallback, not
    # real predictions). All-True when the solve succeeded. Gate dense penalties
    # (smoothness, curvature, bounds over the dense grid) by this the way measurement
    # terms use ``mask_measured``. ``None`` when the dense grid is disabled.
    dense_valid_time: jax.Array | None = None


class LossOutputs(eqx.Module):
    """Named loss scalars. Plot/log panel names = dict keys.

    Total loss for backprop = ``mean(named_losses.values())`` (the harness
    stacks the values in ``loss_names`` order and takes the mean). Mean keeps
    gradients in the same range as bp-train's historical default, so a tuned
    ``grad_clip_norm`` keeps behaving the same as the term count grows. The set
    of keys is fixed per run and declared up front via
    ``UserLossModule.loss_names``; ``__call__`` must return exactly those keys
    in that order.
    """

    named_losses: dict[str, jax.Array]


class UserLossModule(eqx.Module):
    """Base for user-defined loss modules. Mirrors ``UserReactionModule``.

    Subclasses declare any trainable / frozen state via ``trainable_field()`` /
    ``frozen_field()`` (e.g. Kendall uncertainty weights), exactly like a
    reaction module. They are optimized alongside the reaction module because
    the harness partitions the whole wrapper by field tags.
    """

    @property
    def loss_names(self) -> tuple[str, ...]:
        """Stable ordered term/panel names. MUST equal ``named_losses`` keys."""
        raise NotImplementedError

    @property
    def dense_grid_n(self) -> int | None:
        """Optional dense-grid opt-in.

        Return an int N to ask the trainer to solve once on the union of the
        measurement times and ``linspace(t_start, t_end, N)``; the dense view
        is then populated on :class:`LossInputs` as ``dense_*`` fields. Return
        ``None`` (default) to stay on the measurement-grid-only path.

        A dense time is NOT a segment boundary — ``physical_solve`` splits segments
        only at bolus/sample events and reads the grid off ``SaveAt(ts=...)`` inside
        each segment, which is interpolation and costs no solver steps. So a finer
        dense grid does not subdivide the integration, does not change ``fail_time``,
        and does not change which samples bail. (It used to do all three: on a
        240 h / 11-measurement / 10-sample-event process the solve went from 10
        segments / 38 ODE steps at ``dense_grid_n=None`` to 108 / 229 at 100.)

        It is not free, though: each dense point still costs one interpolant
        evaluation per segment it is windowed into, and the ``dense_*`` views it
        populates are real arrays in the loss.
        """
        return None

    def __call__(self, inputs: LossInputs) -> LossOutputs:
        raise NotImplementedError
