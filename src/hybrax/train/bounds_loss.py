from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from bp_format.mechanistic import build_rhs_ode

from .defaults import DefaultLossModule
from .model_api import LossInputs, LossOutputs


class BoundsViolationLossModule(DefaultLossModule):
    """Default measurement MSE plus bp-format state and rate bound penalties.

    Violations are measured in RAW physical space, normalized by the matching
    offset-free derivative scale, and evaluated on real measurement-grid rows.
    Each finite bound side becomes one named loss term.
    """

    bound_records: tuple[tuple[str, str, int, float, float], ...] = eqx.field(
        static=True
    )
    weight: float = eqx.field(static=True)

    def __init__(self, *, target_names, collection, weight):
        super().__init__(target_names=target_names)
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("weight must be finite and nonnegative")
        records = _collect_bound_records(collection)
        self.bound_records = records
        self.weight = weight
        own_loss_names = self.target_names + tuple(record[0] for record in records)
        if len(own_loss_names) != len(set(own_loss_names)):
            raise ValueError(
                f"Bounds loss names must be unique; got {own_loss_names!r}"
            )

    @property
    def loss_names(self) -> tuple[str, ...]:
        return self.target_names + tuple(record[0] for record in self.bound_records)

    def __call__(self, inputs: LossInputs) -> LossOutputs:
        named_losses = dict(super().__call__(inputs).named_losses)
        mask = inputs.mask_measured_any
        n_active = jnp.maximum(jnp.sum(mask), 1.0)
        reaction_module = inputs.reaction_module

        for label, source, idx, sign, threshold in self.bound_records:
            if source == "state":
                values = inputs.RAW_states[:, idx]
                scaler = reaction_module.SCALE_state[idx]
            elif source == "volume":
                values = inputs.RAW_V_unclamped
                scaler = reaction_module.SCALE_state[idx]
            else:
                values = inputs.RAW_modeled_BiologicalOde_rates[:, idx]
                scaler = reaction_module.SCALE_modeled_BiologicalOde_rates[idx]
            violation = jax.nn.relu(sign * (threshold - values))
            normalized = scaler.scale_derivative(violation)
            named_losses[label] = self.weight * (
                jnp.sum(jnp.square(normalized) * mask) / n_active
            )

        return LossOutputs(named_losses=named_losses)


def _collect_bound_records(collection):
    processes = tuple(collection.processes.items())
    if not processes:
        raise ValueError("BoundsViolationLossModule requires a non-empty collection")

    reference = processes[0][1]
    rhs_ode = build_rhs_ode(reference)
    sources = []

    def add_source(label, source, idx, getter):
        sources.append((label, source, idx, getter))

    for idx, name in enumerate(rhs_ode.name_modeled_RMCs):
        add_source(
            name,
            "state",
            idx,
            lambda p, n=name: p.reactor_medium.components[n].bounds,
        )

    pv_offset = len(rhs_ode.name_modeled_RMCs)
    for idx, name in enumerate(rhs_ode.name_modeled_PVs, start=pv_offset):
        add_source(
            name,
            "state",
            idx,
            lambda p, n=name: p.process_variables[n].bounds,
        )

    volume_idx = pv_offset + len(rhs_ode.name_modeled_PVs)
    state_names = rhs_ode.name_modeled_RMCs + rhs_ode.name_modeled_PVs
    volume_label = "volume/V" if "V" in state_names else "V"
    add_source(volume_label, "volume", volume_idx, lambda p: p.volume.bounds)

    for idx, name in enumerate(rhs_ode.name_modeled_rates):
        add_source(
            f"rate/{name}",
            "rate",
            idx,
            lambda p, n=name: (
                (None, None) if p.biological_ode is None else p.biological_ode.rates[n]
            ),
        )

    records = []
    for label, source, idx, getter in sources:
        bounds = tuple(getter(reference))
        for process_name, process in processes[1:]:
            try:
                other = tuple(getter(process))
            except KeyError as error:
                raise ValueError(
                    f"Bounds source {label!r} is missing from process {process_name!r}"
                ) from error
            if other != bounds:
                reference_name = processes[0][0]
                raise ValueError(
                    f"Bounds for {label!r} differ across processes: "
                    f"{bounds!r} in {reference_name!r} vs {other!r} "
                    f"in {process_name!r}"
                )
        lower, upper = (None if bound is None else float(bound) for bound in bounds)
        for description, threshold in (("Lower", lower), ("Upper", upper)):
            if threshold is not None and not math.isfinite(threshold):
                raise ValueError(
                    f"{description} bound for {label!r} must be finite or None"
                )
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(
                f"Lower bound for {label!r} must not exceed its upper bound"
            )

        for prefix, sign, threshold in (
            ("lwr_bnd", 1.0, lower),
            ("upr_bnd", -1.0, upper),
        ):
            if threshold is not None:
                records.append((f"{prefix}/{label}", source, idx, sign, threshold))

    return tuple(records)
