from __future__ import annotations

import math
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp

from .defaults import DefaultLossModule
from .model_api import LossInputs, LossOutputs
from .runtime_context import BoundSnapshot, collect_bound_records


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

    def __init__(
        self,
        *,
        target_names,
        bound_snapshots: tuple[BoundSnapshot, ...],
        weight,
        dense_grid_n=None,
    ):
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
        self.bound_records = collect_bound_records(bound_snapshots)
        self.weight = weight
        self._dense_grid_n = dense_grid_n
        own_loss_names = self.target_names + tuple(
            record[0] for record in self.bound_records
        )
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
