from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from bp_format.dataclasses import BioProcess, FeedVolumeChange, StaticVariable
from bp_format.mechanistic import RhsOde, get_rhs_ode

from .controls_store import PerProcessControls


def _build_augmented_controls_names(
    control_names: list[str],
    controlled_flow_names: tuple[str, ...],
    modeled_flow_names: tuple[str, ...],
    species_names: tuple[str, ...],
    *,
    include_v_real_feature: bool = False,
) -> tuple[str, ...]:
    """Build descriptive names for each element of the augmented controls vector."""
    names: list[str] = list(control_names)
    for flow_name in controlled_flow_names:
        for species_name in species_names:
            names.append(f"cin:{flow_name}:{species_name}")
    for flow_name in modeled_flow_names:
        for species_name in species_names:
            names.append(f"cin:{flow_name}:{species_name}")
    if include_v_real_feature:
        names.append("v_real")
    return tuple(names)


def _build_augmented_controls_units(
    control_metadata: dict[str, dict[str, Any]],
    control_names: list[str],
    process: BioProcess,
    controlled_flow_names: tuple[str, ...],
    modeled_flow_names: tuple[str, ...],
    species_names: tuple[str, ...],
    *,
    include_v_real_feature: bool = False,
) -> tuple[str, ...]:
    """Build unit strings for each element of the augmented controls vector."""
    units: list[str] = []

    for name in control_names:
        md = control_metadata.get(name, {})
        unit = md.get("unit", "")
        if not unit:
            if name in process.volume.volume_changes:
                unit = process.volume.volume_changes[name].unit
            elif name in process.process_variables:
                unit = process.process_variables[name].unit
        units.append(str(unit))

    for flow_name in controlled_flow_names:
        vc = process.volume.volume_changes.get(flow_name)
        for species_name in species_names:
            if (
                vc is not None
                and isinstance(vc, FeedVolumeChange)
                and vc.feed_medium is not None
                and species_name in vc.feed_medium.components
            ):
                units.append(str(vc.feed_medium.components[species_name].unit))
            else:
                units.append("")

    for flow_name in modeled_flow_names:
        vc = process.volume.volume_changes.get(flow_name)
        for species_name in species_names:
            if (
                vc is not None
                and isinstance(vc, FeedVolumeChange)
                and vc.feed_medium is not None
                and species_name in vc.feed_medium.components
            ):
                units.append(str(vc.feed_medium.components[species_name].unit))
            else:
                units.append("")
    if include_v_real_feature:
        units.append(str(process.volume.unit))

    return tuple(units)


class WrapperEvaluation(eqx.Module):
    """Shared wrapper evaluation results for RHS and save-time exports."""

    states_physical: jax.Array
    c_species_runtime: jax.Array
    v_real_export: jax.Array
    v_real_runtime: jax.Array
    u_flow: jax.Array
    u_flow_extra: jax.Array
    specific_rates_physical: jax.Array
    modeled_feed_rates_physical: jax.Array
    auxiliary: dict[str, jax.Array] | None = None


class SaveOutputs(eqx.Module):
    """Diffrax-saveable wrapper outputs in physical units."""

    states_physical: jax.Array
    v_real_export: jax.Array
    v_real_runtime: jax.Array
    specific_rates_physical: jax.Array
    modeled_feed_rates_physical: jax.Array
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

    The ODE state vector ``y`` is normalised:  ``y = Y / state_scale`` where
    ``Y`` is the physical state vector

        Y = [c_species_0, ..., c_species_{n_sp-1}, V_cont, B_modeled_cum_0, ...]

    layout:
      - indices ``0..n_species-1``       → species concentrations
      - index   ``n_species``            → V_cont (cumulative inflow volume)
      - indices ``n_species+1..end``     → cumulative modeled feed amounts
                                           (one per modeled flow, in
                                           ``rhs_ode.modeled_flow_names`` order)

    V_cont is in the state because the wrapper needs to compute
    ``V_real = V_cont - V_sample_acc(t)`` for the dilution denominator inside
    the RhsOde.  ``B_modeled_cum_k`` is also in the state so it can be matched
    against the measured cumulative for that feed (a much more direct training
    signal than V_cont itself, which is mostly determined by *known* controlled
    feeds).

    Inside each RHS evaluation the wrapper:

    1. Un-scales ``y`` → ``Y``.
    2. Evaluates controls (values + derivatives) at time ``t``.
    3. Builds the augmented controls vector for the MLP and scales it.
    4. Calls the reaction module with **scaled** inputs only:
       ``reaction_module(t, c_scaled, u_scaled) → (q_scaled, f_scaled)``.
    5. Un-scales the MLP outputs using ``q_scale`` / ``f_scale``.
    6. Delegates to ``RhsOde`` (physical space) for ``[dc/dt, dV_cont/dt]``.
    7. Appends ``dB_k/dt = F_modeled_k`` for the cumulative-modeled-feed states.
    8. Re-scales the full derivative: ``dy/dt = dY/dt / state_scale``.

    Note that the controlled feed rates passed into ``RhsOde`` come from
    ``controls.eval_derivative(t)``, **not** ``controls.eval(t)``.  The control
    values for feed channels are *cumulative volumes*, not flow rates; the
    derivative gives the actual flow rate that ``RhsOde`` expects.

    All ``*_scale`` arrays and ``target_state_indices`` are **frozen**
    (not trainable).
    """

    rhs_ode: RhsOde
    reaction_module: Any
    controls: PerProcessControls

    flow_control_indices: jax.Array
    extra_flow_control_indices: jax.Array
    extra_flow_cin: jax.Array
    sample_acc_control_index: int = eqx.field(static=True)
    min_real_volume: float = eqx.field(static=True)

    species_names: tuple[str, ...] = eqx.field(static=True)
    modeled_flow_names: tuple[str, ...] = eqx.field(static=True)
    augmented_controls_names: tuple[str, ...] = eqx.field(static=True)
    augmented_controls_units: tuple[str, ...] = eqx.field(static=True)
    include_v_real_feature: bool = eqx.field(static=True)

    # --- Scaling vectors (frozen, not trainable) ---
    state_scale: jax.Array  # [n_species + 1 + n_modeled]
    controls_scale: jax.Array  # [len(augmented_controls)]
    q_scale: jax.Array  # [n_species]
    f_scale: jax.Array  # [n_modeled_feeds]
    target_variance: jax.Array  # [len(target_state_indices)]
    target_state_indices: jax.Array  # which state columns are loss targets

    @classmethod
    def from_process(
        cls,
        *,
        reaction_module: Any,
        process: BioProcess,
        controls: PerProcessControls,
        state_scale: jax.Array | None = None,
        controls_scale: jax.Array | None = None,
        q_scale: jax.Array | None = None,
        f_scale: jax.Array | None = None,
        target_variance: jax.Array | None = None,
        target_state_indices: jax.Array | None = None,
        min_real_volume: float = 1e-8,
    ) -> HybridOdeWrapper:
        """Build a wrapper from a BioProcess and per-process controls."""
        rhs_ode = get_rhs_ode(process)
        include_v_real_feature = bool(
            getattr(reaction_module, "expects_v_real_feature", False)
        )

        if rhs_ode.process_variable_state_names:
            raise NotImplementedError(
                "HybridOdeWrapper does not yet support processes with PV states "
                f"({rhs_ode.process_variable_state_names}). "
                "Extend C_rhs construction in __call__ first."
            )

        flow_control_indices: list[int] = []
        for flow_name in rhs_ode.flow_names:
            if flow_name not in controls.control_name_to_index:
                raise ValueError(
                    f"RhsOde flow '{flow_name}' not found in controls; "
                    f"available: {list(controls.control_name_to_index.keys())}"
                )
            flow_control_indices.append(controls.control_name_to_index[flow_name])

        n_species = len(rhs_ode.reactor_component_state_names)
        extra_flow_control_indices: list[int] = []
        extra_flow_cin_rows: list[list[float]] = []
        for flow_name, volume_change in process.volume.volume_changes.items():
            if not isinstance(volume_change, FeedVolumeChange):
                continue
            if not volume_change.is_controlled or volume_change.is_continuous:
                continue
            if flow_name not in controls.control_name_to_index:
                raise ValueError(
                    f"non-continuous feed '{flow_name}' not found in controls; "
                    f"available: {list(controls.control_name_to_index.keys())}"
                )
            if volume_change.feed_medium is None:
                raise ValueError(
                    "FeedVolumeChange must define feed_medium for wrapper feed "
                    f"transport. Missing for volume change '{flow_name}'."
                )
            cin_row: list[float] = []
            for species_name in rhs_ode.reactor_component_state_names:
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
            extra_flow_control_indices.append(controls.control_name_to_index[flow_name])
            extra_flow_cin_rows.append(cin_row)

        aug_names = _build_augmented_controls_names(
            control_names=controls.control_names,
            controlled_flow_names=rhs_ode.flow_names,
            modeled_flow_names=rhs_ode.modeled_flow_names,
            species_names=rhs_ode.reactor_component_state_names,
            include_v_real_feature=include_v_real_feature,
        )
        aug_units = _build_augmented_controls_units(
            control_metadata=controls.control_metadata,
            control_names=controls.control_names,
            process=process,
            controlled_flow_names=rhs_ode.flow_names,
            modeled_flow_names=rhs_ode.modeled_flow_names,
            species_names=rhs_ode.reactor_component_state_names,
            include_v_real_feature=include_v_real_feature,
        )

        n_aug = len(aug_names)
        n_modeled = rhs_ode.f_modeled_size
        full_state_size = n_species + 1 + n_modeled

        # Default scales: ones (no scaling)
        _state_scale = (
            jnp.asarray(state_scale, dtype=jnp.float32)
            if state_scale is not None
            else jnp.ones(full_state_size, dtype=jnp.float32)
        )
        _controls_scale = (
            jnp.asarray(controls_scale, dtype=jnp.float32)
            if controls_scale is not None
            else jnp.ones(n_aug, dtype=jnp.float32)
        )
        _q_scale = (
            jnp.asarray(q_scale, dtype=jnp.float32)
            if q_scale is not None
            else jnp.ones(n_species, dtype=jnp.float32)
        )
        _f_scale = (
            jnp.asarray(f_scale, dtype=jnp.float32)
            if f_scale is not None
            else jnp.ones(n_modeled, dtype=jnp.float32)
        )
        # Default target_state_indices: species columns + modeled-cumulative columns
        # (V_cont at index n_species is in the state but not a loss target).
        if target_state_indices is None:
            default_indices = list(range(n_species)) + list(
                range(n_species + 1, n_species + 1 + n_modeled)
            )
            _target_state_indices = jnp.asarray(default_indices, dtype=jnp.int32)
        else:
            _target_state_indices = jnp.asarray(target_state_indices, dtype=jnp.int32)
        n_targets = int(_target_state_indices.shape[0])
        _target_variance = (
            jnp.asarray(target_variance, dtype=jnp.float32)
            if target_variance is not None
            else jnp.ones(n_targets, dtype=jnp.float32)
        )
        if extra_flow_cin_rows:
            _extra_flow_cin = jnp.asarray(extra_flow_cin_rows, dtype=jnp.float32)
        else:
            _extra_flow_cin = jnp.zeros((0, n_species), dtype=jnp.float32)

        return cls(
            rhs_ode=rhs_ode,
            reaction_module=reaction_module,
            controls=controls,
            flow_control_indices=jnp.asarray(flow_control_indices, dtype=jnp.int32),
            extra_flow_control_indices=jnp.asarray(
                extra_flow_control_indices, dtype=jnp.int32
            ),
            extra_flow_cin=_extra_flow_cin,
            sample_acc_control_index=int(controls.sample_acc_global_index),
            min_real_volume=float(min_real_volume),
            species_names=rhs_ode.reactor_component_state_names,
            modeled_flow_names=rhs_ode.modeled_flow_names,
            augmented_controls_names=aug_names,
            augmented_controls_units=aug_units,
            include_v_real_feature=include_v_real_feature,
            state_scale=_state_scale,
            controls_scale=_controls_scale,
            q_scale=_q_scale,
            f_scale=_f_scale,
            target_variance=_target_variance,
            target_state_indices=_target_state_indices,
        )

    # ------ helpers exposed to callers (e.g. plotting, harness) ------

    def scale_state(self, Y: jax.Array) -> jax.Array:
        """Physical state → scaled state."""
        return Y / self.state_scale

    def unscale_state(self, y: jax.Array) -> jax.Array:
        """Scaled state → physical state."""
        return y * self.state_scale

    def _validate_state_vector(self, y: jax.Array) -> None:
        """Validate scaled wrapper state layout."""
        if y.ndim != 1:
            raise ValueError("state vector y ndim must be 1")
        n_species = len(self.species_names)
        n_modeled = len(self.modeled_flow_names)
        expected_state_size = n_species + 1 + n_modeled
        if y.shape[0] != expected_state_size:
            raise ValueError(
                f"state vector y must have shape ({expected_state_size},), "
                f"got {tuple(y.shape)}"
            )

    def _evaluate_wrapper_terms(
        self,
        t: float | jax.Array,
        y: jax.Array,
    ) -> WrapperEvaluation:
        """Evaluate shared wrapper quantities used by RHS and save path."""
        self._validate_state_vector(y)

        n_species = len(self.species_names)
        t_arr = jnp.asarray(t, dtype=y.dtype)

        # Keep raw physical state for export; runtime math still clamps the
        # quantities that must remain non-negative inside the mechanistic ODE.
        Y = self.unscale_state(y)
        C_species_runtime = jnp.clip(Y[:n_species], 0.0)
        V_cont_runtime = jnp.maximum(Y[n_species], jnp.asarray(0.0, dtype=y.dtype))

        # Values are interpolated from the dense grid: feed channels store the
        # CUMULATIVE volume, process variables store the actual signal value.
        controls_vector = self.controls.eval(t_arr)
        # Derivatives at the same time: feed channels become flow rates (kg/h),
        # which is what RhsOde expects for `u_flow`.
        controls_derivatives = self.controls.eval_derivative(t_arr)

        V_sample_acc = controls_vector[self.sample_acc_control_index]
        V_real_export = Y[n_species] - V_sample_acc
        V_real_runtime = jnp.maximum(
            V_cont_runtime - V_sample_acc,
            jnp.asarray(self.min_real_volume, dtype=y.dtype),
        )

        U_flow = controls_derivatives[self.flow_control_indices]
        # Non-continuous controlled feeds are represented in prep as short
        # rate ramps (not cumulative traces), so use control values directly.
        U_flow_extra = controls_vector[self.extra_flow_control_indices]

        # Build augmented controls for the MLP. Use the *values* (cumulative for
        # feeds) — these are perfectly fine MLP features and match the existing
        # `augmented_controls_names` semantics.
        cin_flat = jnp.concatenate(
            [
                self.rhs_ode.Cin.reshape(-1),
                self.rhs_ode.Cin_modeled.reshape(-1),
            ]
        )
        U_augmented = jnp.concatenate([controls_vector, cin_flat])
        if self.include_v_real_feature:
            U_augmented = jnp.concatenate(
                [U_augmented, jnp.asarray([V_real_runtime], dtype=y.dtype)]
            )

        # Reaction module still sees scaled inputs only.
        c_scaled = y[:n_species]
        u_scaled = U_augmented / self.controls_scale
        outputs = self.reaction_module(t_arr, c_scaled, u_scaled)
        if not hasattr(outputs, "specific_rates") or not hasattr(
            outputs, "modeled_feed_rates"
        ):
            raise TypeError(
                "reaction_module output must expose `specific_rates` and "
                "`modeled_feed_rates`"
            )
        q_scaled = jnp.asarray(outputs.specific_rates, dtype=y.dtype)
        f_scaled = jnp.asarray(outputs.modeled_feed_rates, dtype=y.dtype)

        if q_scaled.shape != C_species_runtime.shape:
            raise ValueError(
                f"specific_rates must match species shape "
                f"{tuple(C_species_runtime.shape)}, got {tuple(q_scaled.shape)}"
            )
        expected_modeled_shape = (self.rhs_ode.f_modeled_size,)
        if f_scaled.shape != expected_modeled_shape:
            raise ValueError(
                f"modeled_feed_rates must have shape {expected_modeled_shape}, "
                f"got {tuple(f_scaled.shape)}"
            )

        return WrapperEvaluation(
            states_physical=Y,
            c_species_runtime=C_species_runtime,
            v_real_export=V_real_export,
            v_real_runtime=V_real_runtime,
            u_flow=U_flow,
            u_flow_extra=U_flow_extra,
            specific_rates_physical=q_scaled * self.q_scale,
            modeled_feed_rates_physical=jax.nn.softplus(f_scaled) * self.f_scale,
            auxiliary=_normalize_auxiliary_outputs(getattr(outputs, "auxiliary", None)),
        )

    # ------ ODE RHS ------

    def __call__(self, t: float | jax.Array, y: jax.Array) -> jax.Array:
        """Compute ``dy/dt`` in **scaled** state space.

        Parameters
        ----------
        t : scalar time
        y : scaled state vector with layout
            ``[c_species, V_cont, B_modeled_cum_0, ...] / state_scale``

        Returns
        -------
        dy/dt in scaled space, same layout as ``y``.
        """
        n_species = len(self.species_names)
        eval_terms = self._evaluate_wrapper_terms(t, y)

        # ---- 6. Mechanistic RHS in physical space ----
        # RhsOde returns [dc_species/dt, dV/dt] where dV/dt = sum(U_flow) +
        # sum(F_modeled).  By construction this equals dV_cont/dt because
        # V_cont = V0 + ∫(inflows) (sampling lives in V_sample_acc, not in V_cont).
        C_rhs = jnp.concatenate(
            [eval_terms.c_species_runtime, eval_terms.v_real_runtime[None]]
        )
        r = jnp.zeros(self.rhs_ode.r_size, dtype=y.dtype)
        dY_rhs = self.rhs_ode(
            C_rhs,
            eval_terms.specific_rates_physical,
            eval_terms.u_flow,
            eval_terms.modeled_feed_rates_physical,
            r,
        )
        if self.extra_flow_cin.shape[0] > 0:
            extra_contrib = eval_terms.u_flow_extra[:, None] * (
                self.extra_flow_cin.astype(y.dtype)
                - eval_terms.c_species_runtime[None, :]
            )
            dY_rhs = dY_rhs.at[:n_species].add(
                jnp.sum(extra_contrib, axis=0) / eval_terms.v_real_runtime
            )
            dY_rhs = dY_rhs.at[n_species].add(jnp.sum(eval_terms.u_flow_extra))
        # dY_rhs has length n_species + 1 (species + V_cont).

        # ---- 7. Append cumulative-modeled-feed derivatives ----
        # dB_k/dt = F_modeled_k by definition.
        dY_full = jnp.concatenate([dY_rhs, eval_terms.modeled_feed_rates_physical])

        # ---- 8. Re-scale derivative ----
        dy_dt = dY_full / self.state_scale

        return dy_dt

    def save_outputs(
        self,
        t: float | jax.Array,
        y: jax.Array,
        args: Any = None,
    ) -> SaveOutputs:
        """Return physical solver-time outputs for Diffrax ``SaveAt(fn=...)``."""
        del args
        eval_terms = self._evaluate_wrapper_terms(t, y)
        return SaveOutputs(
            states_physical=eval_terms.states_physical,
            v_real_export=eval_terms.v_real_export,
            v_real_runtime=eval_terms.v_real_runtime,
            specific_rates_physical=eval_terms.specific_rates_physical,
            modeled_feed_rates_physical=eval_terms.modeled_feed_rates_physical,
            auxiliary=eval_terms.auxiliary,
        )


def validate_rhs_ode_compatibility(
    reference_name: str,
    reference_rhs: RhsOde,
    candidate_name: str,
    candidate_rhs: RhsOde,
) -> None:
    """Validate that two RhsOde instances have compatible structure."""
    if (
        reference_rhs.reactor_component_state_names
        != candidate_rhs.reactor_component_state_names
    ):
        raise ValueError(
            f"RhsOde reactor_component_state_names differ between "
            f"{reference_name!r} and {candidate_name!r}: "
            f"{reference_rhs.reactor_component_state_names} vs "
            f"{candidate_rhs.reactor_component_state_names}"
        )
    if reference_rhs.flow_names != candidate_rhs.flow_names:
        raise ValueError(
            f"RhsOde flow_names differ between {reference_name!r} and "
            f"{candidate_name!r}: {reference_rhs.flow_names} vs "
            f"{candidate_rhs.flow_names}"
        )
    if reference_rhs.modeled_flow_names != candidate_rhs.modeled_flow_names:
        raise ValueError(
            f"RhsOde modeled_flow_names differ between {reference_name!r} and "
            f"{candidate_name!r}: {reference_rhs.modeled_flow_names} vs "
            f"{candidate_rhs.modeled_flow_names}"
        )
    if reference_rhs.Cin.shape != candidate_rhs.Cin.shape:
        raise ValueError(
            f"RhsOde Cin shapes differ between {reference_name!r} and "
            f"{candidate_name!r}: {reference_rhs.Cin.shape} vs "
            f"{candidate_rhs.Cin.shape}"
        )
    if reference_rhs.Cin_modeled.shape != candidate_rhs.Cin_modeled.shape:
        raise ValueError(
            f"RhsOde Cin_modeled shapes differ between {reference_name!r} and "
            f"{candidate_name!r}: {reference_rhs.Cin_modeled.shape} vs "
            f"{candidate_rhs.Cin_modeled.shape}"
        )
