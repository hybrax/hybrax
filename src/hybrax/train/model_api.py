from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu


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
# Public types
# ---------------------------------------------------------------------------


class ReactionInputs(eqx.Module):
    """All inputs to a UserReactionModule call, one per semantic axis, in SCL space.

    Built by HybridOdeWrapper at each RHS evaluation. The module reads only the
    axes it cares about; unused fields cost nothing under JIT.

    State-slice axes (from the integrated SCL state):
    - ``SCL_modeled_RMCs``: species concentrations, UNCLIPPED so MLP gradient flow
      survives transient negative excursions near depletion.
    - ``SCL_modeled_V``: real reactor volume at time t (already includes the
      ``min_V`` floor applied by the wrapper).
    - ``SCL_modeled_FVCs_cumulative``: per-feed cumulative volume of each
      MODELED feed (integrated state).

    Continuous controlled FVC axes (controls.eval(t) + controls.eval_u(t)):
    - ``SCL_controlled_FVCs_cumulative`` — per-feed cumulative volume.
    - ``SCL_controlled_FVCs_rates`` — per-feed instantaneous flow rate.
    - ``SCL_controlled_FVCs_Cin`` — per-feed Cin matrix [n_FVC, n_RMCs].

    Discrete bolus controlled FVCs (split out of the old `extras`):
    - ``SCL_controlled_FVCs_bolus_rates`` — instantaneous triangle-wave rate of each
      bolus event at time t.

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
    SCL_controlled_FVCs_bolus_rates: jax.Array
    SCL_controlled_PVs: jax.Array
    SCL_modeled_FVCs_Cin: jax.Array


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
    auxiliary:
        Optional model-defined observables that follow the solver-time save
        path. ``None`` or ``dict[str, array]`` with scalar or 1D-array leaves
        and stable keys across calls.
    """

    SCL_modeled_BiologicalOde_rates: jax.Array
    SCL_modeled_FVCs_rates: jax.Array
    auxiliary: dict[str, jax.Array] | None = None


@dataclass
class EstimatedScales:
    """Return shape for the ``estimate_all_scales`` user hook.

    The harness unpacks these into ``build_reaction_module`` kwargs and the
    constructed module stores them as frozen fields. Together they cover every
    semantic axis the module / wrapper / trainer needs to scale.
    """

    SCALE_modeled_RMCs: jax.Array
    SCALE_V_in_cumulative: jax.Array
    SCALE_modeled_FVCs_cumulative: jax.Array
    SCALE_controlled_FVCs_cumulative: jax.Array
    SCALE_controlled_FVCs_rates: jax.Array
    SCALE_controlled_FVCs_Cin: jax.Array
    SCALE_controlled_FVCs_bolus_rates: jax.Array
    SCALE_controlled_PVs: jax.Array
    SCALE_modeled_FVCs_Cin: jax.Array
    SCALE_modeled_BiologicalOde_rates: jax.Array
    SCALE_modeled_FVCs_rates: jax.Array


class UserReactionModule(eqx.Module):
    """Base for user-defined reaction modules.

    The module is the single source of truth for every ``SCALE_*`` vector in
    bp-train. The wrapper queries scales via ``self.reaction_module.SCALE_*``;
    the trainer reads ``SCALE_state`` to convert measurements to SCL space.

    Subclasses inherit the scale fields below — they do NOT redeclare them;
    instead they pass values to ``super().__init__(**scale_kwargs)``.

    The ``scale_*`` / ``unscale_*`` helpers below are linear and work
    identically for state values AND state derivatives (since
    ``d(x/k)/dt = (dx/dt)/k`` for constant ``k``).
    """

    # SCALE_* fields default to zero-sized placeholders so subclasses can be
    # instantiated without passing them (handy for tests + tooling). The
    # ``HybridOdeWrapper`` constructor validates per-axis shapes; supplying
    # real values via ``estimate_all_scales`` / ``super().__init__(**kwargs)``
    # is the production path.
    SCALE_modeled_RMCs: jax.Array = frozen_field(
        default_factory=lambda: jnp.zeros(0, dtype=jnp.float32)
    )
    SCALE_V_in_cumulative: jax.Array = frozen_field(
        default_factory=lambda: jnp.asarray(1.0, dtype=jnp.float32)
    )
    SCALE_modeled_FVCs_cumulative: jax.Array = frozen_field(
        default_factory=lambda: jnp.zeros(0, dtype=jnp.float32)
    )
    SCALE_controlled_FVCs_cumulative: jax.Array = frozen_field(
        default_factory=lambda: jnp.zeros(0, dtype=jnp.float32)
    )
    SCALE_controlled_FVCs_rates: jax.Array = frozen_field(
        default_factory=lambda: jnp.zeros(0, dtype=jnp.float32)
    )
    SCALE_controlled_FVCs_Cin: jax.Array = frozen_field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.float32)
    )
    SCALE_controlled_FVCs_bolus_rates: jax.Array = frozen_field(
        default_factory=lambda: jnp.zeros(0, dtype=jnp.float32)
    )
    SCALE_controlled_PVs: jax.Array = frozen_field(
        default_factory=lambda: jnp.zeros(0, dtype=jnp.float32)
    )
    SCALE_modeled_FVCs_Cin: jax.Array = frozen_field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.float32)
    )
    SCALE_modeled_BiologicalOde_rates: jax.Array = frozen_field(
        default_factory=lambda: jnp.zeros(0, dtype=jnp.float32)
    )
    SCALE_modeled_FVCs_rates: jax.Array = frozen_field(
        default_factory=lambda: jnp.zeros(0, dtype=jnp.float32)
    )

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
    def n_controlled_FVCs_bolus(self) -> int:
        """Number of discrete bolus controlled-FVC events."""
        return int(self.SCALE_controlled_FVCs_bolus_rates.shape[0])

    @property
    def n_controlled_PVs(self) -> int:
        """Number of controlled process-variable signals."""
        return int(self.SCALE_controlled_PVs.shape[0])

    @property
    def SCALE_state(self) -> jax.Array:
        """Concatenated state-scale: ``[modeled_RMCs | V_in_cumulative | modeled_FVCs_cumulative]``."""
        return jnp.concatenate(
            [
                self.SCALE_modeled_RMCs,
                jnp.atleast_1d(self.SCALE_V_in_cumulative),
                self.SCALE_modeled_FVCs_cumulative,
            ]
        )

    @property
    def SCALE_modeled_V(self) -> jax.Array:
        """Real-volume scale shares the V_in_cumulative scale (same units, L)."""
        return self.SCALE_V_in_cumulative

    def __call__(self, t: jax.Array, inputs: ReactionInputs) -> ReactionOutputs:
        """Override. Inputs in SCL space; return rates in SCL space.

        Use the ``.unscale_*`` helpers when you need RAW physical values for
        chemistry / FBA / kinetic-law evaluation. Use ``.scale_*`` on the way
        back out.
        """
        raise NotImplementedError

    def observe(self, states: jax.Array) -> jax.Array:
        """Optional observation map; default identity."""
        return states

    # ------------------------------------------------------------------
    # Linear scale/unscale helpers. Argument-name suffix matches method name.
    # Work for state values AND state derivatives identically.
    # ------------------------------------------------------------------

    def scale_state(self, RAW_state):
        return RAW_state / self.SCALE_state

    def unscale_state(self, SCL_state):
        return SCL_state * self.SCALE_state

    def scale_modeled_RMCs(self, RAW_modeled_RMCs):
        return RAW_modeled_RMCs / self.SCALE_modeled_RMCs

    def unscale_modeled_RMCs(self, SCL_modeled_RMCs):
        return SCL_modeled_RMCs * self.SCALE_modeled_RMCs

    def scale_modeled_V(self, RAW_modeled_V):
        return RAW_modeled_V / self.SCALE_modeled_V

    def unscale_modeled_V(self, SCL_modeled_V):
        return SCL_modeled_V * self.SCALE_modeled_V

    def scale_V_in_cumulative(self, RAW_V_in_cumulative):
        return RAW_V_in_cumulative / self.SCALE_V_in_cumulative

    def unscale_V_in_cumulative(self, SCL_V_in_cumulative):
        return SCL_V_in_cumulative * self.SCALE_V_in_cumulative

    def scale_modeled_FVCs_cumulative(self, RAW_modeled_FVCs_cumulative):
        return RAW_modeled_FVCs_cumulative / self.SCALE_modeled_FVCs_cumulative

    def unscale_modeled_FVCs_cumulative(self, SCL_modeled_FVCs_cumulative):
        return SCL_modeled_FVCs_cumulative * self.SCALE_modeled_FVCs_cumulative

    def scale_controlled_FVCs_cumulative(self, RAW_controlled_FVCs_cumulative):
        return RAW_controlled_FVCs_cumulative / self.SCALE_controlled_FVCs_cumulative

    def unscale_controlled_FVCs_cumulative(self, SCL_controlled_FVCs_cumulative):
        return SCL_controlled_FVCs_cumulative * self.SCALE_controlled_FVCs_cumulative

    def scale_controlled_FVCs_rates(self, RAW_controlled_FVCs_rates):
        return RAW_controlled_FVCs_rates / self.SCALE_controlled_FVCs_rates

    def unscale_controlled_FVCs_rates(self, SCL_controlled_FVCs_rates):
        return SCL_controlled_FVCs_rates * self.SCALE_controlled_FVCs_rates

    def scale_controlled_FVCs_Cin(self, RAW_controlled_FVCs_Cin):
        return RAW_controlled_FVCs_Cin / self.SCALE_controlled_FVCs_Cin

    def unscale_controlled_FVCs_Cin(self, SCL_controlled_FVCs_Cin):
        return SCL_controlled_FVCs_Cin * self.SCALE_controlled_FVCs_Cin

    def scale_controlled_FVCs_bolus_rates(self, RAW_controlled_FVCs_bolus_rates):
        return RAW_controlled_FVCs_bolus_rates / self.SCALE_controlled_FVCs_bolus_rates

    def unscale_controlled_FVCs_bolus_rates(self, SCL_controlled_FVCs_bolus_rates):
        return SCL_controlled_FVCs_bolus_rates * self.SCALE_controlled_FVCs_bolus_rates

    def scale_controlled_PVs(self, RAW_controlled_PVs):
        return RAW_controlled_PVs / self.SCALE_controlled_PVs

    def unscale_controlled_PVs(self, SCL_controlled_PVs):
        return SCL_controlled_PVs * self.SCALE_controlled_PVs

    def scale_modeled_FVCs_Cin(self, RAW_modeled_FVCs_Cin):
        return RAW_modeled_FVCs_Cin / self.SCALE_modeled_FVCs_Cin

    def unscale_modeled_FVCs_Cin(self, SCL_modeled_FVCs_Cin):
        return SCL_modeled_FVCs_Cin * self.SCALE_modeled_FVCs_Cin

    def scale_modeled_BiologicalOde_rates(self, RAW_modeled_BiologicalOde_rates):
        return RAW_modeled_BiologicalOde_rates / self.SCALE_modeled_BiologicalOde_rates

    def unscale_modeled_BiologicalOde_rates(self, SCL_modeled_BiologicalOde_rates):
        return SCL_modeled_BiologicalOde_rates * self.SCALE_modeled_BiologicalOde_rates

    def scale_modeled_FVCs_rates(self, RAW_modeled_FVCs_rates):
        return RAW_modeled_FVCs_rates / self.SCALE_modeled_FVCs_rates

    def unscale_modeled_FVCs_rates(self, SCL_modeled_FVCs_rates):
        return SCL_modeled_FVCs_rates * self.SCALE_modeled_FVCs_rates


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
    # ``controls.active_step_ts``). ``None`` when the trainer ran without
    # jump-ts. Useful for masking dense points near discontinuities; passed
    # whether or not the dense grid is enabled.
    jump_ts: jax.Array | None = None

    # ---- Dense-grid view. Populated iff ``UserLossModule.dense_grid_n`` is
    # not None. All have leading dim ``dense_grid_n``. 
    # ``dense_t`` is the linspace itself.
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

        The dense path costs ~zero extra ODE work — it just adds ``SaveAt``
        evaluations of ``wrapper.save_outputs`` at the dense times.
        """
        return None

    def __call__(self, inputs: LossInputs) -> LossOutputs:
        raise NotImplementedError
