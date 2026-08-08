from __future__ import annotations

import math
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
from bp_format.mechanistic import build_rhs_ode

from .defaults import DefaultLossModule
from .model_api import LossInputs, LossOutputs


class BoundsViolationLossModule(DefaultLossModule):
    """Default measurement MSE plus bp-format state and rate bound penalties.

    Violations are measured in RAW physical space and normalized by the matching
    offset-free derivative scale. By default they are evaluated on real
    measurement-grid rows; ``dense_grid_n`` opts into the deduplicated union of
    measurement and dense-grid rows. Each finite bound side becomes one named
    loss term.
    """

    bound_records: tuple[tuple[str, str, int, float, float], ...] = eqx.field(
        static=True
    )
    weight: float = eqx.field(static=True)
    _dense_grid_n: int | None = eqx.field(static=True)

    def __init__(self, *, target_names, collection, weight, dense_grid_n=None):
        super().__init__(target_names=target_names)
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("weight must be finite and nonnegative")
        if dense_grid_n is not None:
            if isinstance(dense_grid_n, bool) or not isinstance(dense_grid_n, Integral):
                raise ValueError("dense_grid_n must be an integer or None")
            if dense_grid_n < 2:
                raise ValueError("dense_grid_n must be at least 2")
            dense_grid_n = int(dense_grid_n)
        records = _collect_bound_records(collection)
        self.bound_records = records
        self.weight = weight
        self._dense_grid_n = dense_grid_n
        own_loss_names = self.target_names + tuple(record[0] for record in records)
        if len(own_loss_names) != len(set(own_loss_names)):
            raise ValueError(
                f"Bounds loss names must be unique; got {own_loss_names!r}"
            )

    @property
    def dense_grid_n(self) -> int | None:
        return self._dense_grid_n

    @property
    def loss_names(self) -> tuple[str, ...]:
        return self.target_names + tuple(record[0] for record in self.bound_records)

    def __call__(self, inputs: LossInputs) -> LossOutputs:
        named_losses = dict(super().__call__(inputs).named_losses)
        measurement_mask = inputs.mask_measured_any
        if self.dense_grid_n is None:
            dense_mask = None
            n_active = jnp.maximum(jnp.sum(measurement_mask), 1.0)
        else:
            overlaps_dense = jnp.any(
                inputs.t_measured[:, None] == inputs.dense_t[None, :], axis=1
            )
            measurement_mask = measurement_mask * ~overlaps_dense
            dense_mask = inputs.dense_valid_time
            n_active = jnp.maximum(jnp.sum(measurement_mask) + jnp.sum(dense_mask), 1.0)
        reaction_module = inputs.reaction_module

        for label, source, idx, sign, threshold in self.bound_records:
            if source == "state":
                values = inputs.RAW_states[:, idx]
                dense_values = (
                    None if dense_mask is None else inputs.dense_RAW_states[:, idx]
                )
                scaler = reaction_module.SCALE_state[idx]
            elif source == "volume":
                values = inputs.RAW_V_unclamped
                dense_values = inputs.dense_RAW_V_unclamped
                scaler = reaction_module.SCALE_state[idx]
            else:
                values = inputs.RAW_modeled_BiologicalOde_rates[:, idx]
                dense_values = (
                    None
                    if dense_mask is None
                    else inputs.dense_RAW_modeled_BiologicalOde_rates[:, idx]
                )
                scaler = reaction_module.SCALE_modeled_BiologicalOde_rates[idx]
            normalized = scaler.scale_derivative(
                jax.nn.relu(sign * (threshold - values))
            )
            squared_sum = jnp.sum(jnp.square(normalized) * measurement_mask)
            if dense_mask is not None:
                dense_normalized = scaler.scale_derivative(
                    jax.nn.relu(sign * (threshold - dense_values))
                )
                squared_sum += jnp.sum(jnp.square(dense_normalized) * dense_mask)
            named_losses[label] = self.weight * squared_sum / n_active

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
