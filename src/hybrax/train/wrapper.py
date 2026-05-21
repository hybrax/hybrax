from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from bp_format.dataclasses import BioProcess, FeedVolumeChange, StaticVariable
from bp_format.mechanistic import RhsOde, build_rhs_ode

from .controls_store import PerProcessControls
from .model_api import ReactionInputs




class WrapperEvaluation(eqx.Module):
    """Shared wrapper evaluation results for RHS and save-time exports.

    State is carried in SCL space (matches the solver's integration space);
    rates are carried in RAW physical units (the form they take after the
    module's ``unscale_*`` helpers). Plot/CSV exporters convert state to RAW
    at write time via ``module.unscale_state(SCL_states)``.
    """

    SCL_states: jax.Array
    RAW_RMC_rhs: jax.Array
    RAW_V_export: jax.Array
    RAW_V: jax.Array
    RAW_u_rhs_full: jax.Array
    RAW_controlled_FVCs_bolus_rates_at_indices: jax.Array
    RAW_modeled_BiologicalOde_rates: jax.Array
    RAW_modeled_FVCs_rates: jax.Array
    auxiliary: dict[str, jax.Array] | None = None


class SaveOutputs(eqx.Module):
    """Diffrax-saveable wrapper outputs.

    ``SCL_states`` are in the solver's scaled state space; rates are RAW. Use
    ``module.unscale_state`` to convert ``SCL_states`` to physical units when
    exporting to plots/CSV/JSONL.
    """

    SCL_states: jax.Array
    RAW_V_export: jax.Array
    RAW_V: jax.Array
    RAW_modeled_BiologicalOde_rates: jax.Array
    RAW_modeled_FVCs_rates: jax.Array
    auxiliary: dict[str, jax.Array] | None = None


def _normalize_auxiliary_outputs(
    auxiliary: Any,
) -> dict[str, jax.Array] | None:
    """Validate and normalize reaction auxiliary outputs.

    Conservative contract for now:

    - ``None`` or ``dict[str, array]``
    - values must be scalar or 1D arrays at a single save time
    """
    if auxiliary is None:
        return None
    if not isinstance(auxiliary, dict):
        raise TypeError("ReactionOutputs.auxiliary must be None or dict[str, array]")

    normalized: dict[str, jax.Array] = {}
    for key, value in auxiliary.items():
        if not isinstance(key, str):
            raise TypeError("ReactionOutputs.auxiliary keys must be strings")
        arr = jnp.asarray(value)
        if arr.ndim not in (0, 1):
            raise ValueError(
                "ReactionOutputs.auxiliary values must be scalars or 1D arrays, "
                f"got key {key!r} with shape {tuple(arr.shape)}"
            )
        normalized[key] = arr
    return normalized


class HybridOdeWrapper(eqx.Module):
    """ODE wrapper integrating in **scaled state space**.

    The integration state ``SCL_state`` is the SCL-space view of the physical
    state ``RAW_state`` (= ``SCL_state * SCALE_state`` using the scale vector
    that lives on the attached ``UserReactionModule``):

        RAW_state = [modeled_RMCs (n_RMCs)
                     | V_in_cumulative (1)
                     | modeled_VCs_cumulative (n_VCs)]

    The wrapper holds **no scale fields of its own** — every scale comes from
    ``self.reaction_module.SCALE_*``.

    Inside each RHS evaluation:

    1. Unscale ``SCL_state`` → ``RAW_state`` via the module.
    2. Decompose ``controls.eval(t)`` and ``controls.eval_u(t)`` into their
       semantic axes (controlled_FVCs_cumulative, controlled_SVCs_cumulative,
       controlled_PVs, extras, controlled_FVC_rates, controlled_SVC_rates).
    3. Compute ``RAW_V = max(V_in_cumulative - V_sample_acc, min_V)``.
    4. Scale every per-axis input via the module's ``scale_*`` helpers and
       build a ``ReactionInputs`` instance.
    5. Call ``reaction_module(t, inputs) → ReactionOutputs`` (SCL rates).
    6. Unscale rates via the module to RAW for the physical ``RhsOde`` call.
    7. ``RhsOde`` yields ``RAW_d_RMCs_V_dt`` of shape ``(n_RMCs + 1,)``.
    8. Append the cumulative-modeled-feed derivatives (= ``RAW_modeled_FVCs_rates``)
       to form the full ``RAW_d_state_dt``.
    9. Rescale via the module to return ``SCL_d_state_dt``.

    The wrapper applies one safety clip — ``RAW_RMC_rhs = max(RAW_state[:n_RMCs], 0)``
    — only on the path into ``RhsOde``. The module receives the **unclipped**
    SCL slice so MLP gradient flow survives transient negative excursions.

    Modeled PVs and continuous SVCs are not supported (the constructor raises).
    """

    rhs_ode: RhsOde
    reaction_module: Any
    controls: PerProcessControls

    controlled_FVCs_bolus_control_indices: jax.Array
    controlled_FVCs_bolus_Cin: jax.Array
    sample_acc_control_index: int = eqx.field(static=True)
    min_V: float = eqx.field(static=True)

    modeled_RMC_names: tuple[str, ...] = eqx.field(static=True)
    modeled_VC_names: tuple[str, ...] = eqx.field(static=True)

    # Cached slice sizes for the canonical controls vector; sourced from
    # PerProcessControls at construction time so the runtime path doesn't have
    # to introspect ``self.controls`` (which may be swapped to a
    # ``_BatchIndexedControls`` adapter that doesn't expose name tuples).
    n_controlled_FVCs: int = eqx.field(static=True)
    n_controlled_PVs: int = eqx.field(static=True)
    n_controlled_FVCs_bolus: int = eqx.field(static=True)

    target_state_indices: jax.Array  # which state columns are loss targets

    @classmethod
    def from_process(
        cls,
        *,
        reaction_module: Any,
        process: BioProcess,
        controls: PerProcessControls,
        target_state_indices: jax.Array | None = None,
        min_V: float = 1e-8,
    ) -> HybridOdeWrapper:
        """Build a wrapper from a BioProcess and per-process controls.

        Scales are read from ``reaction_module`` (a ``UserReactionModule``
        subclass with ``SCALE_*`` fields). The constructor validates each
        ``SCALE_*`` shape against the RhsOde / controls layout.
        """
        rhs_ode = build_rhs_ode(process)

        if rhs_ode.name_modeled_PVs:
            raise NotImplementedError(
                "HybridOdeWrapper does not support modeled PVs "
                f"({rhs_ode.name_modeled_PVs}); extend the RhsOde input first."
            )
        if rhs_ode.name_controlled_SVCs:
            raise NotImplementedError(
                "HybridOdeWrapper does not support continuous controlled SVCs "
                f"({rhs_ode.name_controlled_SVCs})."
            )
        if rhs_ode.name_modeled_SVCs:
            raise NotImplementedError(
                "HybridOdeWrapper does not support continuous modeled SVCs "
                f"({rhs_ode.name_modeled_SVCs})."
            )

        if rhs_ode.name_controlled_FVCs != controls.name_controlled_FVCs:
            raise ValueError(
                "RhsOde and controls disagree on name_controlled_FVCs: "
                f"{rhs_ode.name_controlled_FVCs} vs "
                f"{controls.name_controlled_FVCs}"
            )

        n_RMCs = len(rhs_ode.name_modeled_RMCs)
        n_VCs = len(rhs_ode.name_modeled_FVCs)
        n_rates = len(rhs_ode.name_modeled_rates)
        n_u = controls.n_u

        # name_extras = (bolus FVCs...) + (V_sample_acc,) — last column is always sample_acc.
        n_controlled_FVCs_bolus = len(controls.name_extras) - 1
        n_controlled_FVCs_count = len(controls.name_controlled_FVCs)

        # Validate reaction_module SCALE_* shapes match the layout we'll feed.
        _expected_shapes: dict[str, tuple[int, ...]] = {
            "SCALE_modeled_RMCs": (n_RMCs,),
            "SCALE_modeled_FVCs_cumulative": (n_VCs,),
            "SCALE_controlled_FVCs_cumulative": (n_controlled_FVCs_count,),
            "SCALE_controlled_FVCs_rates": (n_controlled_FVCs_count,),
            "SCALE_controlled_FVCs_Cin": (n_controlled_FVCs_count, n_RMCs),
            "SCALE_controlled_FVCs_bolus_rates": (n_controlled_FVCs_bolus,),
            "SCALE_controlled_PVs": (len(controls.name_controlled_PVs),),
            "SCALE_modeled_FVCs_Cin": (n_VCs, n_RMCs),
            "SCALE_modeled_BiologicalOde_rates": (n_rates,),
            "SCALE_modeled_FVCs_rates": (n_VCs,),
        }
        for field_name, expected in _expected_shapes.items():
            if not hasattr(reaction_module, field_name):
                raise TypeError(
                    f"reaction_module is missing SCALE field {field_name!r}; "
                    "subclass UserReactionModule and pass all 11 SCALE_* fields "
                    "to super().__init__(...)."
                )
            arr = getattr(reaction_module, field_name)
            if tuple(arr.shape) != expected:
                raise ValueError(
                    f"reaction_module.{field_name} has shape {tuple(arr.shape)}, "
                    f"expected {expected}"
                )
        # V_in_cumulative is a scalar — accept any ndim-0 array.
        if not hasattr(reaction_module, "SCALE_V_in_cumulative"):
            raise TypeError(
                "reaction_module is missing SCALE_V_in_cumulative; subclass "
                "UserReactionModule and pass all 11 SCALE_* fields to "
                "super().__init__(...)."
            )

        # Bolus FVCs live in the extras block; sample_acc is the last extras column.
        bolus_extras = controls.name_extras[:-1]
        bolus_index_in_columns = {
            name: n_u + idx for idx, name in enumerate(bolus_extras)
        }
        bolus_control_indices: list[int] = []
        bolus_Cin_rows: list[list[float]] = []
        for flow_name, volume_change in process.volume.volume_changes.items():
            if not isinstance(volume_change, FeedVolumeChange):
                continue
            if not volume_change.is_controlled or volume_change.is_continuous:
                continue
            if flow_name not in bolus_index_in_columns:
                raise ValueError(
                    f"non-continuous feed '{flow_name}' not found in controls extras; "
                    f"available bolus extras: {list(bolus_extras)}"
                )
            if volume_change.feed_medium is None:
                raise ValueError(
                    "FeedVolumeChange must define feed_medium for wrapper feed "
                    f"transport. Missing for volume change '{flow_name}'."
                )
            cin_row: list[float] = []
            for species_name in rhs_ode.name_modeled_RMCs:
                if species_name not in volume_change.feed_medium.components:
                    cin_row.append(0.0)
                    continue
                concentration = volume_change.feed_medium.components[
                    species_name
                ].concentration
                if isinstance(concentration, StaticVariable):
                    cin_row.append(float(concentration.value))
                else:
                    raise NotImplementedError(
                        "TimeSeries feed concentrations are not supported in "
                        "HybridOdeWrapper for non-continuous feeds. "
                        f"Found TimeSeries for species '{species_name}' in "
                        f"feed '{flow_name}'."
                    )
            bolus_control_indices.append(bolus_index_in_columns[flow_name])
            bolus_Cin_rows.append(cin_row)

        n_modeled = n_VCs + len(rhs_ode.name_modeled_SVCs)

        # Default target_state_indices: species columns + modeled-cumulative columns
        # (V_in_cumulative at index n_RMCs is in the state but not a loss target).
        if target_state_indices is None:
            default_indices = list(range(n_RMCs)) + list(
                range(n_RMCs + 1, n_RMCs + 1 + n_modeled)
            )
            _target_state_indices = jnp.asarray(default_indices, dtype=jnp.int32)
        else:
            _target_state_indices = jnp.asarray(target_state_indices, dtype=jnp.int32)

        if bolus_Cin_rows:
            _bolus_Cin = jnp.asarray(bolus_Cin_rows, dtype=jnp.float32)
        else:
            _bolus_Cin = jnp.zeros((0, n_RMCs), dtype=jnp.float32)

        return cls(
            rhs_ode=rhs_ode,
            reaction_module=reaction_module,
            controls=controls,
            controlled_FVCs_bolus_control_indices=jnp.asarray(
                bolus_control_indices, dtype=jnp.int32
            ),
            controlled_FVCs_bolus_Cin=_bolus_Cin,
            sample_acc_control_index=int(controls.sample_acc_global_index),
            min_V=float(min_V),
            modeled_RMC_names=rhs_ode.name_modeled_RMCs,
            modeled_VC_names=rhs_ode.name_modeled_FVCs,
            n_controlled_FVCs=n_controlled_FVCs_count,
            n_controlled_PVs=len(controls.name_controlled_PVs),
            n_controlled_FVCs_bolus=n_controlled_FVCs_bolus,
            target_state_indices=_target_state_indices,
        )

    def _validate_state_vector(self, SCL_state: jax.Array) -> None:
        """Validate scaled wrapper state layout."""
        if SCL_state.ndim != 1:
            raise ValueError("state vector ndim must be 1")
        n_RMCs = len(self.modeled_RMC_names)
        n_VCs = len(self.modeled_VC_names)
        expected_state_size = n_RMCs + 1 + n_VCs
        if SCL_state.shape[0] != expected_state_size:
            raise ValueError(
                f"state vector must have shape ({expected_state_size},), "
                f"got {tuple(SCL_state.shape)}"
            )

    def _evaluate_wrapper_terms(
        self,
        t: float | jax.Array,
        SCL_state: jax.Array,
    ) -> WrapperEvaluation:
        """Evaluate shared wrapper quantities used by RHS and save path."""
        self._validate_state_vector(SCL_state)
        module = self.reaction_module
        n_RMCs = len(self.modeled_RMC_names)
        t_arr = jnp.asarray(t, dtype=SCL_state.dtype)

        # Unscale the integrated state to physical units for RhsOde-facing work.
        RAW_state = module.unscale_state(SCL_state)
        RAW_RMC_rhs = jnp.maximum(RAW_state[:n_RMCs], 0.0)
        RAW_V_in_cumulative = jnp.maximum(
            RAW_state[n_RMCs], jnp.asarray(0.0, dtype=SCL_state.dtype)
        )

        # Decompose the canonical control vector into semantic axes.
        #   [FVC_cum | SVC_cum (always size 0) | controlled_PVs | extras]
        # where extras = (bolus FVCs ...) + (V_sample_acc,).
        n_FVC = self.n_controlled_FVCs
        n_SVC = 0  # name_controlled_SVCs is hardcoded `()` in bp-train; kept for slice math
        n_PV = self.n_controlled_PVs
        n_bolus = self.n_controlled_FVCs_bolus

        RAW_u_canonical_full = self.controls.eval(t_arr)
        RAW_controlled_FVCs_cumulative = RAW_u_canonical_full[:n_FVC]
        # SVC block (always size 0) skipped at index [n_FVC : n_FVC + n_SVC].
        RAW_controlled_PVs = RAW_u_canonical_full[
            n_FVC + n_SVC : n_FVC + n_SVC + n_PV
        ]
        RAW_controlled_FVCs_bolus_rates = RAW_u_canonical_full[
            n_FVC + n_SVC + n_PV : n_FVC + n_SVC + n_PV + n_bolus
        ]
        # V_sample_acc is the trailing extras column. Wrapper-internal only.
        RAW_V_sample_acc = RAW_u_canonical_full[self.sample_acc_control_index]

        # RhsOde u vector: [FVC_flows | SVC_flows | PV_values]. Decompose flows
        # for the module; the full vector is passed through to RhsOde.
        RAW_u_rhs_full = self.controls.eval_u(t_arr)
        RAW_controlled_FVCs_rates = RAW_u_rhs_full[:n_FVC]

        # Bolus FVC flow rates at their canonical-controls column indices —
        # used by the wrapper for the bolus mass-balance contribution.
        RAW_controlled_FVCs_bolus_rates_at_indices = RAW_u_canonical_full[
            self.controlled_FVCs_bolus_control_indices
        ]

        # V_real, with the min_V floor for dilution-term safety.
        RAW_V_export = RAW_state[n_RMCs] - RAW_V_sample_acc
        RAW_V = jnp.maximum(
            RAW_V_in_cumulative - RAW_V_sample_acc,
            jnp.asarray(self.min_V, dtype=SCL_state.dtype),
        )

        # Feed-medium concentrations (static-ish per process).
        RAW_controlled_FVCs_Cin = self.rhs_ode.Cin_controlled_FVCs
        RAW_modeled_FVCs_Cin = self.rhs_ode.Cin_modeled_FVCs

        # Build ReactionInputs: every axis scaled by the module's own helpers.
        # State-slice axes are pulled from SCL_state directly so the module sees
        # the UNCLIPPED SCL species (gradient flow preserved near depletion).
        SCL_modeled_RMCs = SCL_state[:n_RMCs]
        SCL_modeled_FVCs_cumulative = SCL_state[n_RMCs + 1 :]
        SCL_modeled_V = module.scale_modeled_V(RAW_V)
        inputs = ReactionInputs(
            SCL_modeled_RMCs=SCL_modeled_RMCs,
            SCL_modeled_V=SCL_modeled_V,
            SCL_modeled_FVCs_cumulative=SCL_modeled_FVCs_cumulative,
            SCL_controlled_FVCs_cumulative=module.scale_controlled_FVCs_cumulative(
                RAW_controlled_FVCs_cumulative
            ),
            SCL_controlled_FVCs_rates=module.scale_controlled_FVCs_rates(
                RAW_controlled_FVCs_rates
            ),
            SCL_controlled_FVCs_Cin=module.scale_controlled_FVCs_Cin(
                RAW_controlled_FVCs_Cin
            ),
            SCL_controlled_FVCs_bolus_rates=module.scale_controlled_FVCs_bolus_rates(
                RAW_controlled_FVCs_bolus_rates
            ),
            SCL_controlled_PVs=module.scale_controlled_PVs(RAW_controlled_PVs),
            SCL_modeled_FVCs_Cin=module.scale_modeled_FVCs_Cin(
                RAW_modeled_FVCs_Cin
            ),
        )

        outputs = module(t_arr, inputs)
        if not hasattr(outputs, "SCL_modeled_BiologicalOde_rates") or not hasattr(
            outputs, "SCL_modeled_FVCs_rates"
        ):
            raise TypeError(
                "reaction_module output must expose `SCL_modeled_BiologicalOde_rates` "
                "and `SCL_modeled_FVCs_rates`"
            )
        SCL_modeled_BiologicalOde_rates = jnp.asarray(
            outputs.SCL_modeled_BiologicalOde_rates, dtype=SCL_state.dtype
        )
        SCL_modeled_FVCs_rates = jnp.asarray(
            outputs.SCL_modeled_FVCs_rates, dtype=SCL_state.dtype
        )

        expected_rates_shape = (len(self.rhs_ode.name_modeled_rates),)
        if SCL_modeled_BiologicalOde_rates.shape != expected_rates_shape:
            raise ValueError(
                "SCL_modeled_BiologicalOde_rates must match name_modeled_rates "
                f"shape {expected_rates_shape}, got "
                f"{tuple(SCL_modeled_BiologicalOde_rates.shape)}"
            )
        expected_modeled_FVCs_shape = (len(self.rhs_ode.name_modeled_FVCs),)
        if SCL_modeled_FVCs_rates.shape != expected_modeled_FVCs_shape:
            raise ValueError(
                f"SCL_modeled_FVCs_rates must have shape {expected_modeled_FVCs_shape}, "
                f"got {tuple(SCL_modeled_FVCs_rates.shape)}"
            )

        # Cross to RAW for the physical RhsOde and the saved diagnostic.
        RAW_modeled_BiologicalOde_rates = module.unscale_modeled_BiologicalOde_rates(
            SCL_modeled_BiologicalOde_rates
        )
        RAW_modeled_FVCs_rates = module.unscale_modeled_FVCs_rates(
            SCL_modeled_FVCs_rates
        )

        return WrapperEvaluation(
            SCL_states=SCL_state,
            RAW_RMC_rhs=RAW_RMC_rhs,
            RAW_V_export=RAW_V_export,
            RAW_V=RAW_V,
            RAW_u_rhs_full=RAW_u_rhs_full,
            RAW_controlled_FVCs_bolus_rates_at_indices=RAW_controlled_FVCs_bolus_rates_at_indices,
            RAW_modeled_BiologicalOde_rates=RAW_modeled_BiologicalOde_rates,
            RAW_modeled_FVCs_rates=RAW_modeled_FVCs_rates,
            auxiliary=_normalize_auxiliary_outputs(getattr(outputs, "auxiliary", None)),
        )

    # ------ ODE RHS ------

    def __call__(self, t: float | jax.Array, SCL_state: jax.Array) -> jax.Array:
        """Compute ``d(SCL_state)/dt`` in **scaled** state space."""
        n_RMCs = len(self.modeled_RMC_names)
        eval_terms = self._evaluate_wrapper_terms(t, SCL_state)

        # ---- 6. Mechanistic RHS in physical space ----
        RAW_RMCs_V = jnp.concatenate(
            [eval_terms.RAW_RMC_rhs, eval_terms.RAW_V[None]]
        )
        f_modeled_SVCs = jnp.zeros(
            (len(self.rhs_ode.name_modeled_SVCs),), dtype=SCL_state.dtype
        )
        RAW_d_RMCs_V_dt = self.rhs_ode(
            RAW_RMCs_V,
            eval_terms.RAW_modeled_BiologicalOde_rates,
            eval_terms.RAW_u_rhs_full,
            eval_terms.RAW_modeled_FVCs_rates,
            f_modeled_SVCs,
        )
        if self.controlled_FVCs_bolus_Cin.shape[0] > 0:
            RAW_controlled_FVCs_bolus_contrib = (
                eval_terms.RAW_controlled_FVCs_bolus_rates_at_indices[:, None] * (
                    self.controlled_FVCs_bolus_Cin.astype(SCL_state.dtype)
                    - eval_terms.RAW_RMC_rhs[None, :]
                )
            )
            RAW_d_RMCs_V_dt = RAW_d_RMCs_V_dt.at[:n_RMCs].add(
                jnp.sum(RAW_controlled_FVCs_bolus_contrib, axis=0) / eval_terms.RAW_V
            )
            RAW_d_RMCs_V_dt = RAW_d_RMCs_V_dt.at[n_RMCs].add(
                jnp.sum(eval_terms.RAW_controlled_FVCs_bolus_rates_at_indices)
            )

        # ---- 7. Append cumulative-modeled-feed derivatives ----
        RAW_d_state_dt = jnp.concatenate(
            [RAW_d_RMCs_V_dt, eval_terms.RAW_modeled_FVCs_rates]
        )

        # ---- 8. Re-scale derivative via the module ----
        SCL_d_state_dt = self.reaction_module.scale_state(RAW_d_state_dt)
        return SCL_d_state_dt

    def save_outputs(
        self,
        t: float | jax.Array,
        SCL_state: jax.Array,
        args: Any = None,
    ) -> SaveOutputs:
        """Return solver-time outputs for Diffrax ``SaveAt(fn=...)``."""
        del args
        eval_terms = self._evaluate_wrapper_terms(t, SCL_state)
        return SaveOutputs(
            SCL_states=eval_terms.SCL_states,
            RAW_V_export=eval_terms.RAW_V_export,
            RAW_V=eval_terms.RAW_V,
            RAW_modeled_BiologicalOde_rates=eval_terms.RAW_modeled_BiologicalOde_rates,
            RAW_modeled_FVCs_rates=eval_terms.RAW_modeled_FVCs_rates,
            auxiliary=eval_terms.auxiliary,
        )


def validate_rhs_ode_compatibility(
    reference_name: str,
    reference_rhs_ode: RhsOde,
    candidate_name: str,
    candidate_rhs_ode: RhsOde,
) -> None:
    """Validate that two RhsOde instances have compatible structure."""
    if (
        reference_rhs_ode.name_modeled_RMCs
        != candidate_rhs_ode.name_modeled_RMCs
    ):
        raise ValueError(
            f"RhsOde name_modeled_RMCs differ between "
            f"{reference_name!r} and {candidate_name!r}: "
            f"{reference_rhs_ode.name_modeled_RMCs} vs "
            f"{candidate_rhs_ode.name_modeled_RMCs}"
        )
    if (
        reference_rhs_ode.name_controlled_FVCs
        != candidate_rhs_ode.name_controlled_FVCs
    ):
        raise ValueError(
            f"RhsOde name_controlled_FVCs differ between {reference_name!r} and "
            f"{candidate_name!r}: {reference_rhs_ode.name_controlled_FVCs} vs "
            f"{candidate_rhs_ode.name_controlled_FVCs}"
        )
    if reference_rhs_ode.name_modeled_FVCs != candidate_rhs_ode.name_modeled_FVCs:
        raise ValueError(
            f"RhsOde name_modeled_FVCs differ between {reference_name!r} and "
            f"{candidate_name!r}: {reference_rhs_ode.name_modeled_FVCs} vs "
            f"{candidate_rhs_ode.name_modeled_FVCs}"
        )
    if (
        reference_rhs_ode.Cin_controlled_FVCs.shape
        != candidate_rhs_ode.Cin_controlled_FVCs.shape
    ):
        raise ValueError(
            f"RhsOde Cin_controlled_FVCs shapes differ between {reference_name!r} and "
            f"{candidate_name!r}: {reference_rhs_ode.Cin_controlled_FVCs.shape} vs "
            f"{candidate_rhs_ode.Cin_controlled_FVCs.shape}"
        )
    if (
        reference_rhs_ode.Cin_modeled_FVCs.shape
        != candidate_rhs_ode.Cin_modeled_FVCs.shape
    ):
        raise ValueError(
            f"RhsOde Cin_modeled_FVCs shapes differ between {reference_name!r} and "
            f"{candidate_name!r}: {reference_rhs_ode.Cin_modeled_FVCs.shape} vs "
            f"{candidate_rhs_ode.Cin_modeled_FVCs.shape}"
        )
