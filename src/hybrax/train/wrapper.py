from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from bp_format.dataclasses import BioProcess
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
    RAW_controlled_FVCs_rates: jax.Array
    RAW_controlled_FVCs_Cin: jax.Array
    RAW_modeled_FVCs_Cin: jax.Array
    ADF: jax.Array
    RAW_state: jax.Array
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
                     | modeled_FVCs_cumulative (n_FVCs)]

    The wrapper holds **no scale fields of its own** — every scale comes from
    ``self.reaction_module.SCALE_*``.

    Inside each RHS evaluation:

    1. Unscale ``SCL_state`` → ``RAW_state`` via the module.
    2. Read each control axis via the controls' per-axis accessors
       (``eval_controlled_FVCs_cumulative`` / ``eval_controlled_FVCs_rates`` /
       ``eval_controlled_PVs``), which return RAW values.
    3. Scale every per-axis input via the module's ``scale_*`` helpers and
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

    min_V: float = eqx.field(static=True)

    modeled_RMC_names: tuple[str, ...] = eqx.field(static=True)
    modeled_FVC_names: tuple[str, ...] = eqx.field(static=True)

    # Cached slice sizes for the canonical controls vector; sourced from
    # PerProcessControls at construction time so the runtime path doesn't have
    # to introspect ``self.controls`` (which may be swapped to a
    # ``_BatchIndexedControls`` adapter that doesn't expose name tuples).
    n_controlled_FVCs: int = eqx.field(static=True)
    n_controlled_PVs: int = eqx.field(static=True)

    target_state_indices: jax.Array  # which state columns are loss targets

    # User loss module (a UserLossModule). Untagged so the whole-wrapper
    # partition_trainable walk reads its own trainable_field()/frozen_field()
    # tags. Default None for direct constructors / forward-only paths; the
    # train harness always attaches a real module.
    loss_module: Any = None

    @classmethod
    def from_process(
        cls,
        *,
        reaction_module: Any,
        process: BioProcess,
        controls: PerProcessControls,
        target_state_indices: jax.Array | None = None,
        min_V: float = 1e-8,
        loss_module: Any = None,
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
        n_FVCs = len(rhs_ode.name_modeled_FVCs)
        n_rates = len(rhs_ode.name_modeled_rates)
        n_u = controls.n_u

        n_controlled_FVCs_count = len(controls.name_controlled_FVCs)

        # Validate reaction_module SCALE_* shapes match the layout we'll feed.
        _expected_shapes: dict[str, tuple[int, ...]] = {
            "SCALE_modeled_RMCs": (n_RMCs,),
            "SCALE_modeled_FVCs_cumulative": (n_FVCs,),
            "SCALE_controlled_FVCs_cumulative": (n_controlled_FVCs_count,),
            "SCALE_controlled_FVCs_rates": (n_controlled_FVCs_count,),
            "SCALE_controlled_FVCs_Cin": (n_controlled_FVCs_count, n_RMCs),
            "SCALE_controlled_PVs": (len(controls.name_controlled_PVs),),
            "SCALE_modeled_FVCs_Cin": (n_FVCs, n_RMCs),
            "SCALE_modeled_BiologicalOde_rates": (n_rates,),
            "SCALE_modeled_FVCs_rates": (n_FVCs,),
        }
        for field_name, expected in _expected_shapes.items():
            if not hasattr(reaction_module, field_name):
                raise TypeError(
                    f"reaction_module is missing SCALE field {field_name!r}; "
                    "subclass UserReactionModule and pass all 10 SCALE_* fields "
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
                "UserReactionModule and pass all 10 SCALE_* fields to "
                "super().__init__(...)."
            )

        n_modeled = n_FVCs + len(rhs_ode.name_modeled_SVCs)

        # Default target_state_indices: species columns + modeled-cumulative columns
        # (V_in_cumulative at index n_RMCs is in the state but not a loss target).
        if target_state_indices is None:
            default_indices = list(range(n_RMCs)) + list(
                range(n_RMCs + 1, n_RMCs + 1 + n_modeled)
            )
            _target_state_indices = jnp.asarray(default_indices, dtype=jnp.int32)
        else:
            _target_state_indices = jnp.asarray(target_state_indices, dtype=jnp.int32)

        return cls(
            rhs_ode=rhs_ode,
            reaction_module=reaction_module,
            controls=controls,
            min_V=float(min_V),
            modeled_RMC_names=rhs_ode.name_modeled_RMCs,
            modeled_FVC_names=rhs_ode.name_modeled_FVCs,
            n_controlled_FVCs=n_controlled_FVCs_count,
            n_controlled_PVs=len(controls.name_controlled_PVs),
            target_state_indices=_target_state_indices,
            loss_module=loss_module,
        )

    # ------ Physical-state RHS (continuous part of the diffrax_callbacks solve) ------
    #
    # Integrates the *physical* state ``y = [RAW_RMCs | RAW_V | RAW_modeled_cum]``
    # directly; discrete bolus/sample events are applied as state jumps between
    # segments (``physical_solve.solve_physical_states``), not folded into the
    # vector field. Between events only continuous dynamics act
    # (biology, plus continuous feeds/dilution if any), so the integrated state stays
    # O(C) and the reverse-mode adjoint is well-conditioned. (Replaces the earlier
    # pseudobatch state, whose unbounded accumulator corrupted the gradient.)

    def physical_rhs(self, t: float | jax.Array, y_phys: jax.Array) -> jax.Array:
        """d/dt of the physical state ``[RAW_RMCs | RAW_V | RAW_modeled_cum]``.

        Continuous part only — discrete boluses/samples are handled as jumps.
        """
        module = self.reaction_module
        n_RMCs = len(self.modeled_RMC_names)
        n_FVCs = len(self.modeled_FVC_names)
        dtype = y_phys.dtype
        t_arr = jnp.asarray(t, dtype=dtype)

        RAW_RMCs = y_phys[:n_RMCs]
        RAW_V = jnp.maximum(y_phys[n_RMCs], jnp.asarray(self.min_V, dtype=dtype))
        RAW_modeled_cum = y_phys[n_RMCs + 1 : n_RMCs + 1 + n_FVCs]
        RAW_RMC_rhs = jnp.maximum(RAW_RMCs, 0.0)

        n_FVC = self.n_controlled_FVCs
        n_SVC_ctrl = len(self.rhs_ode.name_controlled_SVCs)

        RAW_controlled_FVCs_cumulative = self.controls.eval_controlled_FVCs_cumulative(
            t_arr, y_phys
        )
        RAW_controlled_FVCs_rates = self.controls.eval_controlled_FVCs_rates(
            t_arr, y_phys
        )
        RAW_controlled_PVs = self.controls.eval_controlled_PVs(t_arr, y_phys)
        RAW_controlled_FVCs_Cin = self.rhs_ode.Cin_controlled_FVCs
        RAW_modeled_FVCs_Cin = self.rhs_ode.Cin_modeled_FVCs

        inputs = ReactionInputs(
            SCL_modeled_RMCs=module.scale_modeled_RMCs(RAW_RMCs),
            SCL_modeled_V=module.scale_modeled_V(RAW_V),
            SCL_modeled_FVCs_cumulative=module.scale_modeled_FVCs_cumulative(
                RAW_modeled_cum
            ),
            SCL_controlled_FVCs_cumulative=module.scale_controlled_FVCs_cumulative(
                RAW_controlled_FVCs_cumulative
            ),
            SCL_controlled_FVCs_rates=module.scale_controlled_FVCs_rates(
                RAW_controlled_FVCs_rates
            ),
            SCL_controlled_FVCs_Cin=module.scale_controlled_FVCs_Cin(
                RAW_controlled_FVCs_Cin
            ),
            SCL_controlled_PVs=module.scale_controlled_PVs(RAW_controlled_PVs),
            SCL_modeled_FVCs_Cin=module.scale_modeled_FVCs_Cin(RAW_modeled_FVCs_Cin),
        )
        outputs = module(t_arr, inputs)
        RAW_bio_rates = module.unscale_modeled_BiologicalOde_rates(
            jnp.asarray(outputs.SCL_modeled_BiologicalOde_rates, dtype=dtype)
        )
        RAW_modeled_FVCs_rates = module.unscale_modeled_FVCs_rates(
            jnp.asarray(outputs.SCL_modeled_FVCs_rates, dtype=dtype)
        )

        RAW_RMCs_V = jnp.concatenate([RAW_RMC_rhs, RAW_V[None]])
        # RhsOde's u = [FVC_flows | SVC_flows | PV_values]; the wrapper zeroes the
        # flows (it applies feed/dilution itself below) and forwards only the PVs.
        RAW_u_bio = jnp.concatenate(
            [jnp.zeros(n_FVC + n_SVC_ctrl, dtype=dtype), RAW_controlled_PVs]
        )
        RAW_zero_modeled_FVCs_rates = jnp.zeros_like(RAW_modeled_FVCs_rates)
        RAW_modeled_SVCs_rates = jnp.zeros(
            (len(self.rhs_ode.name_modeled_SVCs),), dtype=dtype
        )
        RAW_d_RMCs_V_dt = self.rhs_ode(
            RAW_RMCs_V,
            RAW_bio_rates,
            RAW_u_bio,
            RAW_zero_modeled_FVCs_rates,
            RAW_modeled_SVCs_rates,
        )
        dC = RAW_d_RMCs_V_dt[:n_RMCs]
        # Continuous-feed volume/dilution (zero for a bolus-only process).
        controlled_addition = jnp.sum(
            RAW_controlled_FVCs_rates[:, None]
            * RAW_controlled_FVCs_Cin.astype(dtype),
            axis=0,
        )
        modeled_addition = jnp.sum(
            RAW_modeled_FVCs_rates[:, None] * RAW_modeled_FVCs_Cin.astype(dtype),
            axis=0,
        )
        dV_cont = jnp.sum(RAW_controlled_FVCs_rates) + jnp.sum(RAW_modeled_FVCs_rates)
        dilution = RAW_RMCs * (dV_cont / RAW_V)
        dC = dC + (controlled_addition + modeled_addition) / RAW_V - dilution
        return jnp.concatenate(
            [dC, jnp.atleast_1d(dV_cont).astype(dtype), RAW_modeled_FVCs_rates]
        )

    def initial_physical_state_from_raw(self, RAW_state: jax.Array) -> jax.Array:
        """Identity: the physical solve integrates the raw state layout directly."""
        n_RMCs = len(self.modeled_RMC_names)
        n_FVCs = len(self.modeled_FVC_names)
        return RAW_state[: n_RMCs + 1 + n_FVCs]

    def physical_save_outputs(
        self, t: float | jax.Array, y_phys: jax.Array
    ) -> "SaveOutputs":
        """``SaveOutputs`` computed from the physical state (for loss/exports)."""
        module = self.reaction_module
        n_RMCs = len(self.modeled_RMC_names)
        n_FVCs = len(self.modeled_FVC_names)
        dtype = y_phys.dtype
        t_arr = jnp.asarray(t, dtype=dtype)
        RAW_RMCs = y_phys[:n_RMCs]
        # Clamped V feeds the reaction module (the concentration denominator can't
        # go <= 0); the *export* keeps the true (possibly <min_V) volume so the
        # human-facing v_real reflects the sampled volume directly.
        RAW_V = jnp.maximum(y_phys[n_RMCs], jnp.asarray(self.min_V, dtype=dtype))
        RAW_V_export = y_phys[n_RMCs]
        RAW_modeled_cum = y_phys[n_RMCs + 1 : n_RMCs + 1 + n_FVCs]

        RAW_controlled_FVCs_cumulative = self.controls.eval_controlled_FVCs_cumulative(
            t_arr, y_phys
        )
        RAW_controlled_FVCs_rates = self.controls.eval_controlled_FVCs_rates(
            t_arr, y_phys
        )
        RAW_controlled_PVs = self.controls.eval_controlled_PVs(t_arr, y_phys)
        inputs = ReactionInputs(
            SCL_modeled_RMCs=module.scale_modeled_RMCs(RAW_RMCs),
            SCL_modeled_V=module.scale_modeled_V(RAW_V),
            SCL_modeled_FVCs_cumulative=module.scale_modeled_FVCs_cumulative(
                RAW_modeled_cum
            ),
            SCL_controlled_FVCs_cumulative=module.scale_controlled_FVCs_cumulative(
                RAW_controlled_FVCs_cumulative
            ),
            SCL_controlled_FVCs_rates=module.scale_controlled_FVCs_rates(
                RAW_controlled_FVCs_rates
            ),
            SCL_controlled_FVCs_Cin=module.scale_controlled_FVCs_Cin(
                self.rhs_ode.Cin_controlled_FVCs
            ),
            SCL_controlled_PVs=module.scale_controlled_PVs(RAW_controlled_PVs),
            SCL_modeled_FVCs_Cin=module.scale_modeled_FVCs_Cin(
                self.rhs_ode.Cin_modeled_FVCs
            ),
        )
        outputs = module(t_arr, inputs)
        RAW_bio_rates = module.unscale_modeled_BiologicalOde_rates(
            jnp.asarray(outputs.SCL_modeled_BiologicalOde_rates, dtype=dtype)
        )
        RAW_modeled_FVCs_rates = module.unscale_modeled_FVCs_rates(
            jnp.asarray(outputs.SCL_modeled_FVCs_rates, dtype=dtype)
        )
        RAW_state = jnp.concatenate(
            [RAW_RMCs, RAW_V[None], RAW_modeled_cum]
        )
        return SaveOutputs(
            SCL_states=module.scale_state(RAW_state),
            RAW_V_export=RAW_V_export,
            RAW_V=RAW_V,
            RAW_modeled_BiologicalOde_rates=RAW_bio_rates,
            RAW_modeled_FVCs_rates=RAW_modeled_FVCs_rates,
            auxiliary=_normalize_auxiliary_outputs(getattr(outputs, "auxiliary", None)),
        )


def validate_rhs_ode_compatibility(
    reference_name: str,
    reference_rhs_ode: RhsOde,
    candidate_name: str,
    candidate_rhs_ode: RhsOde,
) -> None:
    """Validate that two RhsOde instances have compatible structure."""
    if reference_rhs_ode.name_modeled_RMCs != candidate_rhs_ode.name_modeled_RMCs:
        raise ValueError(
            f"RhsOde name_modeled_RMCs differ between "
            f"{reference_name!r} and {candidate_name!r}: "
            f"{reference_rhs_ode.name_modeled_RMCs} vs "
            f"{candidate_rhs_ode.name_modeled_RMCs}"
        )
    if reference_rhs_ode.name_controlled_FVCs != candidate_rhs_ode.name_controlled_FVCs:
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
