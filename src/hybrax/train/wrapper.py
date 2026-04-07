from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from bpbench.dataclasses import BioProcess, FeedVolumeChange
from bpbench.mechanistic import RhsOde, get_rhs_ode

from .controls_store import PerProcessControls
from .model_api import UserReactionModule


def _build_augmented_controls_names(
    control_names: list[str],
    controlled_flow_names: tuple[str, ...],
    modeled_flow_names: tuple[str, ...],
    species_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Build descriptive names for each element of the augmented controls vector."""
    names: list[str] = list(control_names)
    for flow_name in controlled_flow_names:
        for species_name in species_names:
            names.append(f"cin:{flow_name}:{species_name}")
    for flow_name in modeled_flow_names:
        for species_name in species_names:
            names.append(f"cin:{flow_name}:{species_name}")
    return tuple(names)


def _build_augmented_controls_units(
    control_metadata: dict[str, dict[str, Any]],
    control_names: list[str],
    process: BioProcess,
    controlled_flow_names: tuple[str, ...],
    modeled_flow_names: tuple[str, ...],
    species_names: tuple[str, ...],
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

    return tuple(units)


class HybridOdeWrapper(eqx.Module):
    """ODE wrapper integrating in **scaled state space**.

    The ODE state vector ``y`` is normalised:  ``y = Y / state_scale`` where
    ``Y`` is the physical state ``[c_species..., V_cont]``.  Inside each
    RHS evaluation the wrapper:

    1. Un-scales ``y`` → ``Y`` (physical concentrations + volume).
    2. Evaluates controls and builds an augmented controls vector.
    3. Scales the augmented vector (``u = U / controls_scale``).
    4. Calls the reaction module with **scaled** inputs only:
       ``reaction_module(t, c_scaled, u_scaled) → (q_scaled, f_scaled)``.
    5. Un-scales the MLP outputs using ``q_scale`` / ``f_scale``.
    6. Delegates to ``RhsOde`` (physical space) for the mechanistic RHS.
    7. Re-scales the derivative: ``dy/dt = dY/dt / state_scale``.

    All ``*_scale`` arrays are **frozen** (not trainable).
    """

    rhs_ode: RhsOde
    reaction_module: Any
    controls: PerProcessControls

    flow_control_indices: jax.Array
    sample_acc_control_index: int = eqx.field(static=True)
    min_real_volume: float = eqx.field(static=True)

    species_names: tuple[str, ...] = eqx.field(static=True)
    augmented_controls_names: tuple[str, ...] = eqx.field(static=True)
    augmented_controls_units: tuple[str, ...] = eqx.field(static=True)

    # --- Scaling vectors (frozen, not trainable) ---
    state_scale: jax.Array          # [n_species + 1]
    controls_scale: jax.Array       # [len(augmented_controls)]
    q_scale: jax.Array              # [n_species]
    f_scale: jax.Array              # [n_modeled_feeds]
    target_variance: jax.Array      # [n_species] — per-species loss normalization

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
        min_real_volume: float = 1e-8,
    ) -> HybridOdeWrapper:
        """Build a wrapper from a BioProcess and per-process controls."""
        rhs_ode = get_rhs_ode(process)

        flow_control_indices: list[int] = []
        for flow_name in rhs_ode.flow_names:
            if flow_name not in controls.control_name_to_index:
                raise ValueError(
                    f"RhsOde flow '{flow_name}' not found in controls; "
                    f"available: {list(controls.control_name_to_index.keys())}"
                )
            flow_control_indices.append(controls.control_name_to_index[flow_name])

        aug_names = _build_augmented_controls_names(
            control_names=controls.control_names,
            controlled_flow_names=rhs_ode.flow_names,
            modeled_flow_names=rhs_ode.modeled_flow_names,
            species_names=rhs_ode.species_names,
        )
        aug_units = _build_augmented_controls_units(
            control_metadata=controls.control_metadata,
            control_names=controls.control_names,
            process=process,
            controlled_flow_names=rhs_ode.flow_names,
            modeled_flow_names=rhs_ode.modeled_flow_names,
            species_names=rhs_ode.species_names,
        )

        n_species = len(rhs_ode.species_names)
        n_aug = len(aug_names)
        n_modeled = rhs_ode.f_modeled_size

        # Default scales: ones (no scaling)
        _state_scale = (
            jnp.asarray(state_scale, dtype=jnp.float32)
            if state_scale is not None
            else jnp.ones(n_species + 1, dtype=jnp.float32)
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
        _target_variance = (
            jnp.asarray(target_variance, dtype=jnp.float32)
            if target_variance is not None
            else jnp.ones(n_species, dtype=jnp.float32)
        )

        return cls(
            rhs_ode=rhs_ode,
            reaction_module=reaction_module,
            controls=controls,
            flow_control_indices=jnp.asarray(flow_control_indices, dtype=jnp.int32),
            sample_acc_control_index=int(controls.sample_acc_global_index),
            min_real_volume=float(min_real_volume),
            species_names=rhs_ode.species_names,
            augmented_controls_names=aug_names,
            augmented_controls_units=aug_units,
            state_scale=_state_scale,
            controls_scale=_controls_scale,
            q_scale=_q_scale,
            f_scale=_f_scale,
            target_variance=_target_variance,
        )

    # ------ helpers exposed to callers (e.g. plotting, harness) ------

    def scale_state(self, Y: jax.Array) -> jax.Array:
        """Physical state → scaled state."""
        return Y / self.state_scale

    def unscale_state(self, y: jax.Array) -> jax.Array:
        """Scaled state → physical state."""
        return y * self.state_scale

    # ------ ODE RHS ------

    def __call__(self, t: float | jax.Array, y: jax.Array) -> jax.Array:
        """Compute ``dy/dt`` in **scaled** state space.

        Parameters
        ----------
        t : scalar time
        y : scaled state vector ``[c_species / state_scale, V / V_scale]``

        Returns
        -------
        dy/dt in scaled space.
        """
        if y.ndim != 1:
            raise ValueError("state vector y ndim must be 1")
        expected_state_size = len(self.species_names) + 1
        if y.shape[0] != expected_state_size:
            raise ValueError(
                f"state vector y must have shape ({expected_state_size},), "
                f"got {tuple(y.shape)}"
            )

        t_arr = jnp.asarray(t, dtype=y.dtype)

        # ---- 1. Un-scale state to physical space ----
        Y = self.unscale_state(y)
        C_species = jnp.clip(Y[:-1], 0.0)
        V_cont = jnp.maximum(Y[-1], jnp.asarray(0.0, dtype=y.dtype))

        # ---- 2. Evaluate controls (physical) ----
        controls_vector = self.controls.eval(t_arr)

        V_sample_acc = controls_vector[self.sample_acc_control_index]
        V_real = jnp.maximum(
            V_cont - V_sample_acc, jnp.asarray(self.min_real_volume)
        )

        U_flow = controls_vector[self.flow_control_indices]

        # Build augmented controls (physical)
        cin_flat = jnp.concatenate(
            [
                self.rhs_ode.Cin.reshape(-1),
                self.rhs_ode.Cin_modeled.reshape(-1),
            ]
        )
        U_augmented = jnp.concatenate([controls_vector, cin_flat])

        # ---- 3. Scale inputs for MLP ----
        c_scaled = y[:-1]  # already scaled (part of y)
        u_scaled = U_augmented / self.controls_scale

        # ---- 4. MLP predicts in scaled space ----
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

        if q_scaled.shape != C_species.shape:
            raise ValueError(
                f"specific_rates must match species shape {tuple(C_species.shape)}, "
                f"got {tuple(q_scaled.shape)}"
            )
        expected_modeled_shape = (self.rhs_ode.f_modeled_size,)
        if f_scaled.shape != expected_modeled_shape:
            raise ValueError(
                f"modeled_feed_rates must have shape {expected_modeled_shape}, "
                f"got {tuple(f_scaled.shape)}"
            )

        # ---- 5. Un-scale MLP outputs to physical rates ----
        Q = q_scaled * self.q_scale
        F_modeled = f_scaled * self.f_scale

        # ---- 6. Mechanistic RHS in physical space ----
        C_rhs = jnp.concatenate([C_species, jnp.asarray([V_real], dtype=y.dtype)])
        dY_dt = self.rhs_ode(C_rhs, Q, U_flow, F_modeled)

        # ---- 7. Re-scale derivative ----
        dy_dt = dY_dt / self.state_scale

        return dy_dt


def validate_rhs_ode_compatibility(
    reference_name: str,
    reference_rhs: RhsOde,
    candidate_name: str,
    candidate_rhs: RhsOde,
) -> None:
    """Validate that two RhsOde instances have compatible structure."""
    if reference_rhs.species_names != candidate_rhs.species_names:
        raise ValueError(
            f"RhsOde species_names differ between {reference_name!r} and "
            f"{candidate_name!r}: {reference_rhs.species_names} vs "
            f"{candidate_rhs.species_names}"
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
