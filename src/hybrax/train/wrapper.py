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
    RAW_controlled_FVCs_rates: jax.Array
    RAW_controlled_FVCs_Cin: jax.Array
    RAW_modeled_FVCs_Cin: jax.Array
    RAW_controlled_FVCs_bolus_rates_at_indices: jax.Array
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
    modeled_FVC_names: tuple[str, ...] = eqx.field(static=True)

    # Cached slice sizes for the canonical controls vector; sourced from
    # PerProcessControls at construction time so the runtime path doesn't have
    # to introspect ``self.controls`` (which may be swapped to a
    # ``_BatchIndexedControls`` adapter that doesn't expose name tuples).
    n_controlled_FVCs: int = eqx.field(static=True)
    n_controlled_PVs: int = eqx.field(static=True)
    n_controlled_FVCs_bolus: int = eqx.field(static=True)

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

        # name_extras = (bolus FVCs...) + (V_sample_acc,).
        # Last column is always sample_acc.
        n_controlled_FVCs_bolus = len(controls.name_extras) - 1
        n_controlled_FVCs_count = len(controls.name_controlled_FVCs)

        # Validate reaction_module SCALE_* shapes match the layout we'll feed.
        _expected_shapes: dict[str, tuple[int, ...]] = {
            "SCALE_modeled_RMCs": (n_RMCs,),
            "SCALE_modeled_FVCs_cumulative": (n_FVCs,),
            "SCALE_controlled_FVCs_cumulative": (n_controlled_FVCs_count,),
            "SCALE_controlled_FVCs_rates": (n_controlled_FVCs_count,),
            "SCALE_controlled_FVCs_Cin": (n_controlled_FVCs_count, n_RMCs),
            "SCALE_controlled_FVCs_bolus_rates": (n_controlled_FVCs_bolus,),
            "SCALE_controlled_PVs": (len(controls.name_controlled_PVs),),
            "SCALE_modeled_FVCs_Cin": (n_FVCs, n_RMCs),
            "SCALE_modeled_BiologicalOde_rates": (n_rates,),
            "SCALE_modeled_FVCs_rates": (n_FVCs,),
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
            modeled_FVC_names=rhs_ode.name_modeled_FVCs,
            n_controlled_FVCs=n_controlled_FVCs_count,
            n_controlled_PVs=len(controls.name_controlled_PVs),
            n_controlled_FVCs_bolus=n_controlled_FVCs_bolus,
            target_state_indices=_target_state_indices,
            loss_module=loss_module,
        )

    def _pseudo_state_size(self) -> int:
        n_RMCs = len(self.modeled_RMC_names)
        n_FVCs = len(self.modeled_FVC_names)
        n_samples = int(self.controls.sample_event_times.shape[0])
        return n_RMCs + 1 + n_FVCs + n_RMCs + 1 + n_samples

    def _pseudo_scale(self) -> jax.Array:
        module = self.reaction_module
        n_samples = int(self.controls.sample_event_times.shape[0])
        return jnp.concatenate(
            [
                module.SCALE_modeled_RMCs,
                jnp.atleast_1d(module.SCALE_V_in_cumulative),
                module.SCALE_modeled_FVCs_cumulative,
                module.SCALE_modeled_RMCs,
                jnp.atleast_1d(module.SCALE_V_in_cumulative),
                jnp.full(
                    (n_samples,),
                    module.SCALE_V_in_cumulative,
                    dtype=module.SCALE_modeled_RMCs.dtype,
                ),
            ]
        )

    def scale_pseudo_state(self, RAW_pseudo_state: jax.Array) -> jax.Array:
        return RAW_pseudo_state / self._pseudo_scale()

    def unscale_pseudo_state(self, SCL_pseudo_state: jax.Array) -> jax.Array:
        return SCL_pseudo_state * self._pseudo_scale()

    def initial_pseudo_state_from_raw(self, RAW_state: jax.Array) -> jax.Array:
        """Build scaled pseudobatch initial state from physical RAW state."""
        n_RMCs = len(self.modeled_RMC_names)
        n_FVCs = len(self.modeled_FVC_names)
        n_samples = int(self.controls.sample_event_times.shape[0])
        RAW_RMCs = RAW_state[:n_RMCs]
        RAW_V0 = RAW_state[n_RMCs]
        RAW_modeled_cum = RAW_state[n_RMCs + 1 : n_RMCs + 1 + n_FVCs]
        RAW_pseudo = jnp.concatenate(
            [
                RAW_RMCs,
                RAW_V0[None],
                RAW_modeled_cum,
                jnp.zeros((n_RMCs,), dtype=RAW_state.dtype),
                RAW_V0[None],
                jnp.full((n_samples,), RAW_V0, dtype=RAW_state.dtype),
            ]
        )
        return self.scale_pseudo_state(RAW_pseudo)

    def _validate_state_vector(self, SCL_state: jax.Array) -> None:
        """Validate scaled pseudobatch wrapper state layout."""
        if SCL_state.ndim != 1:
            raise ValueError("state vector ndim must be 1")
        expected_state_size = self._pseudo_state_size()
        if SCL_state.shape[0] != expected_state_size:
            raise ValueError(
                f"state vector must have shape ({expected_state_size},), "
                f"got {tuple(SCL_state.shape)}"
            )

    def _split_pseudo_state(
        self,
        RAW_pseudo_state: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        n_RMCs = len(self.modeled_RMC_names)
        n_FVCs = len(self.modeled_FVC_names)
        rmc_star = RAW_pseudo_state[:n_RMCs]
        v_cont = RAW_pseudo_state[n_RMCs]
        modeled_cum = RAW_pseudo_state[n_RMCs + 1 : n_RMCs + 1 + n_FVCs]
        feed_corr = RAW_pseudo_state[n_RMCs + 1 + n_FVCs : 2 * n_RMCs + 1 + n_FVCs]
        v0 = RAW_pseudo_state[2 * n_RMCs + 1 + n_FVCs]
        sample_dummy = RAW_pseudo_state[2 * n_RMCs + 2 + n_FVCs :]
        return rmc_star, v_cont, modeled_cum, feed_corr, v0, sample_dummy

    def _transform_terms(
        self,
        t_arr: jax.Array,
        RAW_pseudo_state: jax.Array,
        *,
        bolus_right_continuous: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        rmc_star, v_cont, modeled_cum, feed_corr, RAW_V0, sample_dummy = (
            self._split_pseudo_state(RAW_pseudo_state)
        )
        dtype = RAW_pseudo_state.dtype
        sample_t = self.controls.sample_event_times.astype(dtype)
        sample_v = self.controls.sample_event_volumes.astype(dtype)
        sample_mask = self.controls.sample_event_mask
        bolus_t = self.controls.bolus_event_times.astype(dtype)
        bolus_v = self.controls.bolus_event_volumes.astype(dtype)
        bolus_Cin = self.controls.bolus_event_Cin.astype(dtype)
        bolus_mask = self.controls.bolus_event_mask

        samples_done = (t_arr >= sample_t) & sample_mask
        if bolus_right_continuous:
            boluses_done = (t_arr >= bolus_t) & bolus_mask
        else:
            boluses_done = (t_arr > bolus_t) & bolus_mask
        sample_before_sample = (
            (sample_t[None, :] < sample_t[:, None])
            & sample_mask[None, :]
            & sample_mask[:, None]
        )
        bolus_before_sample = (
            (bolus_t[None, :] < sample_t[:, None])
            & bolus_mask[None, :]
            & sample_mask[:, None]
        )
        sample_pre = sample_dummy
        sample_pre = sample_pre + jnp.sum(
            jnp.where(bolus_before_sample, bolus_v[None, :], 0.0), axis=1
        )
        sample_pre = sample_pre - jnp.sum(
            jnp.where(sample_before_sample, sample_v[None, :], 0.0), axis=1
        )
        sample_factor_raw = sample_pre / (sample_pre - sample_v)
        sample_factor = jnp.where(sample_mask, sample_factor_raw, 1.0)

        v_disc = jnp.sum(jnp.where(boluses_done, bolus_v, 0.0))
        v_disc = v_disc - jnp.sum(jnp.where(samples_done, sample_v, 0.0))
        RAW_V_export = v_cont + v_disc
        RAW_V = jnp.maximum(RAW_V_export, jnp.asarray(self.min_V, dtype=dtype))
        v0 = jnp.maximum(RAW_V0, jnp.asarray(self.min_V, dtype=dtype))
        sample_product = jnp.prod(jnp.where(samples_done, sample_factor, 1.0))
        ADF = RAW_V / v0 * sample_product

        samples_before_bolus = (
            (sample_t[None, :] <= bolus_t[:, None])
            & sample_mask[None, :]
            & bolus_mask[:, None]
        )
        bolus_sample_factor = jnp.prod(
            jnp.where(samples_before_bolus, sample_factor[None, :], 1.0), axis=1
        )
        bolus_piece = bolus_Cin * (bolus_v * bolus_sample_factor / v0)[:, None]
        bolus_corr = jnp.sum(jnp.where(boluses_done[:, None], bolus_piece, 0.0), axis=0)

        RAW_RMCs = (rmc_star + feed_corr + bolus_corr) / ADF
        RAW_state = jnp.concatenate([RAW_RMCs, RAW_V_export[None], modeled_cum])
        return ADF, RAW_V, RAW_state

    def _evaluate_wrapper_terms(
        self,
        t: float | jax.Array,
        SCL_state: jax.Array,
        *,
        bolus_right_continuous: bool = True,
    ) -> WrapperEvaluation:
        """Evaluate shared wrapper quantities used by RHS and save path."""
        self._validate_state_vector(SCL_state)
        module = self.reaction_module
        n_RMCs = len(self.modeled_RMC_names)
        t_arr = jnp.asarray(t, dtype=SCL_state.dtype)

        RAW_pseudo_state = self.unscale_pseudo_state(SCL_state)
        ADF, RAW_V, RAW_state = self._transform_terms(
            t_arr,
            RAW_pseudo_state,
            bolus_right_continuous=bolus_right_continuous,
        )
        RAW_RMC_rhs = jnp.maximum(RAW_state[:n_RMCs], 0.0)
        RAW_V_export = RAW_state[n_RMCs]

        n_FVC = self.n_controlled_FVCs
        n_SVC = 0
        n_PV = self.n_controlled_PVs
        n_bolus = self.n_controlled_FVCs_bolus

        RAW_u_canonical_full = self.controls.eval(t_arr)
        RAW_controlled_FVCs_cumulative = RAW_u_canonical_full[:n_FVC]
        RAW_controlled_PVs = RAW_u_canonical_full[n_FVC + n_SVC : n_FVC + n_SVC + n_PV]
        RAW_controlled_FVCs_bolus_rates = jnp.zeros((n_bolus,), dtype=SCL_state.dtype)

        RAW_u_rhs_full = self.controls.eval_u(t_arr)
        RAW_controlled_FVCs_rates = RAW_u_rhs_full[:n_FVC]
        RAW_controlled_FVCs_bolus_rates_at_indices = jnp.zeros(
            (self.controlled_FVCs_bolus_control_indices.shape[0],),
            dtype=SCL_state.dtype,
        )

        RAW_controlled_FVCs_Cin = self.rhs_ode.Cin_controlled_FVCs
        RAW_modeled_FVCs_Cin = self.rhs_ode.Cin_modeled_FVCs
        _, _, RAW_modeled_cum, _, _, _ = self._split_pseudo_state(RAW_pseudo_state)

        inputs = ReactionInputs(
            SCL_modeled_RMCs=module.scale_modeled_RMCs(RAW_state[:n_RMCs]),
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
            SCL_controlled_FVCs_bolus_rates=module.scale_controlled_FVCs_bolus_rates(
                RAW_controlled_FVCs_bolus_rates
            ),
            SCL_controlled_PVs=module.scale_controlled_PVs(RAW_controlled_PVs),
            SCL_modeled_FVCs_Cin=module.scale_modeled_FVCs_Cin(RAW_modeled_FVCs_Cin),
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
                "SCL_modeled_FVCs_rates must have shape "
                f"{expected_modeled_FVCs_shape}, got "
                f"{tuple(SCL_modeled_FVCs_rates.shape)}"
            )

        RAW_modeled_BiologicalOde_rates = module.unscale_modeled_BiologicalOde_rates(
            SCL_modeled_BiologicalOde_rates
        )
        RAW_modeled_FVCs_rates = module.unscale_modeled_FVCs_rates(
            SCL_modeled_FVCs_rates
        )

        return WrapperEvaluation(
            SCL_states=module.scale_state(RAW_state),
            RAW_RMC_rhs=RAW_RMC_rhs,
            RAW_V_export=RAW_V_export,
            RAW_V=RAW_V,
            RAW_u_rhs_full=RAW_u_rhs_full,
            RAW_controlled_FVCs_rates=RAW_controlled_FVCs_rates,
            RAW_controlled_FVCs_Cin=RAW_controlled_FVCs_Cin,
            RAW_modeled_FVCs_Cin=RAW_modeled_FVCs_Cin,
            RAW_controlled_FVCs_bolus_rates_at_indices=RAW_controlled_FVCs_bolus_rates_at_indices,
            ADF=ADF,
            RAW_state=RAW_state,
            RAW_modeled_BiologicalOde_rates=RAW_modeled_BiologicalOde_rates,
            RAW_modeled_FVCs_rates=RAW_modeled_FVCs_rates,
            auxiliary=_normalize_auxiliary_outputs(getattr(outputs, "auxiliary", None)),
        )

    # ------ ODE RHS ------

    def __call__(self, t: float | jax.Array, SCL_state: jax.Array) -> jax.Array:
        """Compute pseudobatch-state derivative in **scaled** state space."""
        self._validate_state_vector(SCL_state)
        n_RMCs = len(self.modeled_RMC_names)
        eval_terms = self._evaluate_wrapper_terms(t, SCL_state)

        RAW_RMCs_V = jnp.concatenate([eval_terms.RAW_RMC_rhs, eval_terms.RAW_V[None]])
        RAW_modeled_SVCs_rates = jnp.zeros(
            (len(self.rhs_ode.name_modeled_SVCs),), dtype=SCL_state.dtype
        )
        n_u = eval_terms.RAW_u_rhs_full.shape[0]
        n_pv = self.n_controlled_PVs
        RAW_u_bio = jnp.zeros_like(eval_terms.RAW_u_rhs_full)
        RAW_u_bio = RAW_u_bio.at[n_u - n_pv :].set(
            eval_terms.RAW_u_rhs_full[n_u - n_pv :]
        )
        RAW_zero_modeled_FVCs_rates = jnp.zeros_like(eval_terms.RAW_modeled_FVCs_rates)
        RAW_d_RMCs_V_dt = self.rhs_ode(
            RAW_RMCs_V,
            eval_terms.RAW_modeled_BiologicalOde_rates,
            RAW_u_bio,
            RAW_zero_modeled_FVCs_rates,
            RAW_modeled_SVCs_rates,
        )
        RAW_biological_dRMCs = RAW_d_RMCs_V_dt[:n_RMCs]

        controlled_addition = jnp.sum(
            eval_terms.RAW_controlled_FVCs_rates[:, None]
            * eval_terms.RAW_controlled_FVCs_Cin.astype(SCL_state.dtype),
            axis=0,
        )
        modeled_addition = jnp.sum(
            eval_terms.RAW_modeled_FVCs_rates[:, None]
            * eval_terms.RAW_modeled_FVCs_Cin.astype(SCL_state.dtype),
            axis=0,
        )
        RAW_dfeed_corr = (
            eval_terms.ADF * (controlled_addition + modeled_addition) / eval_terms.RAW_V
        )
        RAW_dV_cont = jnp.sum(eval_terms.RAW_controlled_FVCs_rates) + jnp.sum(
            eval_terms.RAW_modeled_FVCs_rates
        )
        RAW_dRMC_star = eval_terms.ADF * RAW_biological_dRMCs

        RAW_pseudo_state = self.unscale_pseudo_state(SCL_state)
        _, _, _, _, _, sample_dummy = self._split_pseudo_state(RAW_pseudo_state)
        sample_t = self.controls.sample_event_times.astype(SCL_state.dtype)
        sample_mask = self.controls.sample_event_mask
        t_arr = jnp.asarray(t, dtype=SCL_state.dtype)
        RAW_dsample_dummy = jnp.where(
            (t_arr < sample_t) & sample_mask,
            RAW_dV_cont,
            jnp.zeros_like(sample_dummy),
        )

        RAW_d_pseudo_state = jnp.concatenate(
            [
                RAW_dRMC_star,
                RAW_dV_cont[None],
                eval_terms.RAW_modeled_FVCs_rates,
                RAW_dfeed_corr,
                jnp.zeros((1,), dtype=SCL_state.dtype),
                RAW_dsample_dummy,
            ]
        )
        return self.scale_pseudo_state(RAW_d_pseudo_state)

    def save_outputs(
        self,
        t: float | jax.Array,
        SCL_state: jax.Array,
        args: Any = None,
    ) -> SaveOutputs:
        """Return solver-time outputs for Diffrax ``SaveAt(fn=...)``."""
        del args
        eval_terms = self._evaluate_wrapper_terms(
            t,
            SCL_state,
            bolus_right_continuous=False,
        )
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
