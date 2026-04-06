from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from bpbench.dataclasses import BioProcess, FeedVolumeChange
from bpbench.mechanistic import RhsOde, get_rhs_ode

from .controls_store import PerProcessControls


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

    # Units for base controls
    for name in control_names:
        md = control_metadata.get(name, {})
        # Try to get unit from metadata, then from volume change or process variable
        unit = md.get("unit", "")
        if not unit:
            if name in process.volume.volume_changes:
                unit = process.volume.volume_changes[name].unit
            elif name in process.process_variables:
                unit = process.process_variables[name].unit
        units.append(str(unit))

    # Units for controlled feed Cin entries
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

    # Units for modeled feed Cin entries
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
    """ODE wrapper that delegates mechanistic transport to bpbench's RhsOde.

    The user's reaction module receives the species state plus an augmented
    controls vector (base controls + flattened Cin) and returns specific rates
    ``q`` and modeled feed rates.  The mechanistic ODE then computes the full
    state derivative including ``q * X_active``, transport, and dilution.
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

    @classmethod
    def from_process(
        cls,
        *,
        reaction_module: Any,
        process: BioProcess,
        controls: PerProcessControls,
        min_real_volume: float = 1e-8,
    ) -> HybridOdeWrapper:
        """Build a wrapper from a BioProcess and per-process controls."""
        rhs_ode = get_rhs_ode(process)

        # Map RhsOde controlled flow names → controls vector indices
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
        )

    def __call__(self, t: float | jax.Array, y: jax.Array) -> jax.Array:
        """Compute full state derivative ``[dc_species/dt..., dV/dt]``."""
        if y.ndim != 1:
            raise ValueError("state vector y must be rank-1")
        expected_state_size = len(self.species_names) + 1
        if y.shape[0] != expected_state_size:
            raise ValueError(
                f"state vector y must have shape ({expected_state_size},), "
                f"got {tuple(y.shape)}"
            )

        t_arr = jnp.asarray(t, dtype=y.dtype)
        c_species = jnp.clip(y[:-1], 0.0)
        v_cont = jnp.maximum(y[-1], jnp.asarray(self.min_real_volume))

        # Evaluate controls at time t
        controls_vector = self.controls.eval(t_arr)

        # Real volume = container volume - sampled volume
        v_sample_acc = controls_vector[self.sample_acc_control_index]
        v_real = jnp.maximum(v_cont - v_sample_acc, jnp.asarray(self.min_real_volume))

        # Controlled flow rates from controls vector
        u_flow = controls_vector[self.flow_control_indices]

        # Flatten Cin/Cin_modeled and append to controls for MLP input
        cin_flat = jnp.concatenate([
            self.rhs_ode.Cin.reshape(-1),
            self.rhs_ode.Cin_modeled.reshape(-1),
        ])
        augmented_controls = jnp.concatenate([controls_vector, cin_flat])

        # User's MLP: specific rates + modeled feed rates
        outputs = self.reaction_module(t_arr, c_species, augmented_controls)
        if not hasattr(outputs, "specific_rates") or not hasattr(
            outputs, "modeled_feed_rates"
        ):
            raise TypeError(
                "reaction_module output must expose `specific_rates` and "
                "`modeled_feed_rates`"
            )
        specific_rates = jnp.asarray(outputs.specific_rates, dtype=y.dtype)
        modeled_feed_rates = jnp.asarray(outputs.modeled_feed_rates, dtype=y.dtype)

        if specific_rates.shape != c_species.shape:
            raise ValueError(
                f"specific_rates must match species shape {tuple(c_species.shape)}, "
                f"got {tuple(specific_rates.shape)}"
            )
        expected_modeled_shape = (self.rhs_ode.f_modeled_size,)
        if modeled_feed_rates.shape != expected_modeled_shape:
            raise ValueError(
                f"modeled_feed_rates must have shape {expected_modeled_shape}, "
                f"got {tuple(modeled_feed_rates.shape)}"
            )

        # Build RhsOde state vector [c_species..., V_real]
        c_rhs = jnp.concatenate([c_species, jnp.asarray([v_real], dtype=y.dtype)])

        # RhsOde: dc/dt = q * X_active + transport(Cin, u_flow, f_modeled)
        dc_dt = self.rhs_ode(c_rhs, specific_rates, u_flow, modeled_feed_rates)

        # dc_dt from RhsOde has dV/dt as last element (sum of all flows).
        # We return this as dV_cont/dt — the container volume derivative.
        return dc_dt


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
