"""Optional state/rate bound-violation penalty on top of the default measurement loss."""

from __future__ import annotations

import math
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
from hybrax.format.dataclasses import BioProcessCollection

from .defaults import DefaultLossModule
from .model_api import LossInputs, LossOutputs
from .runtime_context import rhs_ode_from_training_parents


BoundDeclaration = tuple[str, str, int, float | None, float | None]
BoundSnapshot = tuple[BoundDeclaration, ...]
BoundRecord = tuple[str, str, int, float, float]


class BoundsViolationLossModule(DefaultLossModule):
    """Default measurement MSE plus hybrax.format state and rate bound penalties.

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
        bound_records: tuple[BoundRecord, ...],
        weight,
        dense_grid_n=None,
    ):
        """Build the bounds loss on top of the default measurement MSE.

        Args:
            target_names: Measurement target names, forwarded to
                :class:`DefaultLossModule`.
            bound_records: Per-axis bound declarations from
                :func:`bound_records_from_collection`; each becomes one named
                loss term (``lwr_bnd/<label>`` / ``upr_bnd/<label>``).
            weight: Nonnegative scalar weight applied to every bound-violation
                term.
            dense_grid_n: Forwarded as :attr:`dense_grid_n`; see
                :attr:`UserLossModule.dense_grid_n`.

        Raises:
            ValueError: If ``weight`` is not finite and nonnegative,
                ``dense_grid_n`` is not ``None``/an integer ``>= 2``, or a
                bound record's name collides with a target name or another
                bound record.
        """
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
        self.bound_records = tuple(bound_records)
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
        """Dense-grid opt-in set at construction; see :attr:`UserLossModule.dense_grid_n`."""
        return self._dense_grid_n

    @property
    def loss_names(self) -> tuple[str, ...]:
        """Measurement target names followed by every bound record's label."""
        return self.target_names + tuple(record[0] for record in self.bound_records)

    def __call__(self, inputs: LossInputs) -> LossOutputs:
        """Default measurement MSE plus one squared-hinge term per bound record."""
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


def bound_records_from_collection(
    collection: BioProcessCollection,
) -> tuple[BoundRecord, ...]:
    """Resolve consistent bound-loss records from represented parents."""
    rhs_ode = rhs_ode_from_training_parents(
        collection, empty_message="bounds loss requires a non-empty collection"
    )
    snapshots = tuple(
        _bound_snapshot(process, rhs_ode) for process in collection.processes.values()
    )
    return collect_bound_records(snapshots)


def _bound_snapshot(process, rhs_ode) -> BoundSnapshot:
    declarations: list[BoundDeclaration] = []
    for index, name in enumerate(rhs_ode.name_modeled_RMCs):
        declarations.append(
            (
                name,
                "state",
                index,
                *_bounds(process.reactor_medium.components[name].bounds),
            )
        )
    pv_offset = len(rhs_ode.name_modeled_RMCs)
    for index, name in enumerate(rhs_ode.name_modeled_PVs, start=pv_offset):
        declarations.append(
            (name, "state", index, *_bounds(process.process_variables[name].bounds))
        )
    state_names = rhs_ode.name_modeled_RMCs + rhs_ode.name_modeled_PVs
    volume_label = "volume/V" if "V" in state_names else "V"
    declarations.append(
        (
            volume_label,
            "volume",
            pv_offset + len(rhs_ode.name_modeled_PVs),
            *_bounds(process.volume.bounds),
        )
    )
    for index, name in enumerate(rhs_ode.name_modeled_rates):
        bounds = (
            (None, None)
            if process.biological_ode is None
            else process.biological_ode.rates[name]
        )
        declarations.append((f"rate/{name}", "rate", index, *_bounds(bounds)))
    return tuple(declarations)


def _bounds(bounds) -> tuple[float | None, float | None]:
    lower, upper = tuple(bounds)
    return (
        None if lower is None else float(lower),
        None if upper is None else float(upper),
    )


def collect_bound_records(
    snapshots: tuple[BoundSnapshot, ...],
) -> tuple[BoundRecord, ...]:
    """Validate per-process bound declarations when bounds loss is requested."""
    if not snapshots:
        raise ValueError("bounds loss requires a non-empty bounds snapshot")
    records: list[BoundRecord] = []
    reference = snapshots[0]
    for index, declaration in enumerate(reference):
        label, source, axis, lower, upper = declaration
        for process_index, snapshot in enumerate(snapshots[1:], start=1):
            try:
                other = snapshot[index]
            except IndexError as error:
                raise ValueError(
                    f"Bounds source {label!r} is missing from process index "
                    f"{process_index}"
                ) from error
            if other != declaration:
                raise ValueError(
                    f"Bounds for {label!r} differ across processes: "
                    f"{declaration[3:]!r} "
                    f"vs {other[3:]!r}"
                )
        for description, threshold in (("Lower", lower), ("Upper", upper)):
            if threshold is not None and not math.isfinite(threshold):
                raise ValueError(
                    f"{description} bound for {label!r} must be finite or None"
                )
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(
                f"Lower bound for {label!r} must not exceed its upper bound"
            )
        if lower is not None:
            records.append((f"lwr_bnd/{label}", source, axis, 1.0, lower))
        if upper is not None:
            records.append((f"upr_bnd/{label}", source, axis, -1.0, upper))
    return tuple(records)
