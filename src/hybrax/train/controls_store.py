from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from bpbench.dataclasses import BioProcessCollection
from bpbench.serialization import load_process_collection_json


def _as_jax_array(values: Any, *, dtype: Any = jnp.float32) -> jax.Array:
    """Convert JSON-loaded values into a JAX array."""
    return jnp.asarray(values, dtype=dtype)


def _coerce_index(
    process: str | int,
    process_order: list[str],
) -> tuple[str, int]:
    """Resolve a process key or integer index to a canonical process key."""
    if isinstance(process, str):
        if process not in process_order:
            raise KeyError(f"unknown process name: {process}")
        return process, process_order.index(process)

    index = int(process)
    if index < 0 or index >= len(process_order):
        raise IndexError(f"process index out of range: {index}")
    return process_order[index], index


def _interp_columns(
    ts: jax.Array,
    grid: jax.Array,
    values: jax.Array,
) -> jax.Array:
    """Linearly interpolate a `[n_grid, n_controls]` payload at one or more times."""

    def _interp_column(column: jax.Array) -> jax.Array:
        return jnp.interp(ts, grid, column, left=column[0], right=column[-1])

    return jax.vmap(_interp_column, in_axes=1, out_axes=1)(values)


class PerProcessControls(eqx.Module):
    """Per-process runtime view over padded, global-axis dense-grid controls."""

    # Canonical prepared process identifier used by the collection metadata.
    process_name: str
    # Integer row index into the collection-level padded tensors.
    process_index: int
    # Shared control names/order for this prepared artifact.
    control_names: list[str]
    # Mapping from control name to shared control column index.
    control_name_to_index: dict[str, int]
    # Shared control names/order across the entire collection.
    global_control_names: list[str]
    # Mapping from global control name to global control column index.
    global_control_name_to_index: dict[str, int]
    # Padded dense time grid for this process, shape `[max_grid_length]`.
    dense_grid: jax.Array
    # Padded control values in shared control order, shape `[max_grid_length, max_controls]`.
    control_values: jax.Array
    # Padded control derivatives in shared control order, same shape as `control_values`.
    control_derivatives: jax.Array
    # Padded step-boundary times used to guide the ODE solver.
    step_ts: jax.Array
    # Number of active dense-grid points for this process.
    grid_length: int
    # Number of active step-boundary entries for this process.
    step_ts_length: int
    # Per-control metadata persisted during preparation.
    control_metadata: dict[str, dict[str, Any]]
    # Reserved name of the cumulative sampled-volume control.
    sample_acc_name: str
    # Global control index of `sample_acc_name`.
    sample_acc_global_index: int

    @property
    def active_dense_grid(self) -> jax.Array:
        """Return the active, unpadded dense grid prefix for this process."""
        return self.dense_grid[: self.grid_length]

    @property
    def active_step_ts(self) -> jax.Array:
        """Return the active, unpadded step-boundary prefix for this process."""
        return self.step_ts[: self.step_ts_length]

    @property
    def active_control_values(self) -> jax.Array:
        """Return active control values in global control order."""
        return self.control_values[: self.grid_length]

    @property
    def active_control_derivatives(self) -> jax.Array:
        """Return active control derivatives in global control order."""
        return self.control_derivatives[: self.grid_length]

    def eval(self, ts: float | np.ndarray | jax.Array) -> jax.Array:
        """Evaluate controls at one or more times in global control order."""
        query = jnp.asarray(ts, dtype=self.dense_grid.dtype)
        scalar_input = query.ndim == 0
        query_1d = jnp.atleast_1d(query)
        values = _interp_columns(
            query_1d,
            self.active_dense_grid,
            self.active_control_values,
        )
        if scalar_input:
            return values[0]
        return values

    def eval_derivative(self, ts: float | np.ndarray | jax.Array) -> jax.Array:
        """Evaluate precomputed control derivatives in global control order."""
        query = jnp.asarray(ts, dtype=self.dense_grid.dtype)
        scalar_input = query.ndim == 0
        query_1d = jnp.atleast_1d(query)
        values = _interp_columns(
            query_1d,
            self.active_dense_grid,
            self.active_control_derivatives,
        )
        if scalar_input:
            return values[0]
        return values


class ControlsStore(eqx.Module):
    """Collection-level loader and index for prepared, padded JAX control tensors."""

    # Canonical prepared process keys in stable collection order.
    process_order: list[str]
    # Shared control names/order across all processes.
    global_control_names: list[str]
    # Mapping from control name to shared control column index.
    global_control_name_to_index: dict[str, int]
    # Shape metadata persisted in `prepared.json`.
    shape_metadata: dict[str, Any]
    # Stacked padded dense grids, shape `[n_processes, max_grid_length]`.
    dense_grid: jax.Array
    # Stacked padded control values, shape `[n_processes, max_grid_length, max_controls]`.
    control_values: jax.Array
    # Stacked padded control derivatives, same shape as `control_values`.
    control_derivatives: jax.Array
    # Stacked padded step-boundary arrays, shape `[n_processes, max_step_ts_length]`.
    step_ts: jax.Array
    # Active dense-grid lengths per process.
    grid_lengths: jax.Array
    # Active `step_ts` lengths per process.
    step_ts_lengths: jax.Array
    # Shared control index of the cumulative sampled-volume control.
    sample_acc_global_index: int
    # Raw per-process metadata entries needed to construct thin runtime views.
    _process_md_by_name: dict[str, dict[str, Any]]

    @classmethod
    def from_collection(
        cls,
        collection: BioProcessCollection,
        *,
        metadata_namespace: str = "bp_train",
    ) -> ControlsStore:
        """Build a JAX-backed runtime store from a prepared `BioProcessCollection`."""
        metadata = dict(collection.metadata or {})
        if metadata_namespace not in metadata:
            raise KeyError(
                f"metadata namespace '{metadata_namespace}' "
                "missing from prepared collection"
            )

        bp_train = metadata[metadata_namespace]
        process_order = list(bp_train["process_order"])
        global_control_names = list(bp_train["global_control_names"])
        global_control_name_to_index = dict(bp_train["global_control_name_to_index"])
        processes_metadata = dict(bp_train["processes"])

        dense_grid_rows = []
        control_value_rows = []
        control_derivative_rows = []
        step_ts_rows = []
        grid_lengths = []
        step_ts_lengths = []
        reference_control_names: list[str] | None = None

        for process_name in process_order:
            process_md = processes_metadata[process_name]
            local_control_names = list(process_md["local_control_names"])
            if reference_control_names is None:
                reference_control_names = local_control_names
            elif local_control_names != reference_control_names:
                raise ValueError(
                    "controls store requires identical control names/order across "
                    f"processes; {process_name!r} has {local_control_names!r} but "
                    f"expected {reference_control_names!r}"
                )

            dense_grid_rows.append(process_md["dense_grid"])
            control_value_rows.append(process_md["control_values"])
            control_derivative_rows.append(process_md["control_derivatives"])
            step_ts_rows.append(process_md["step_ts"])
            grid_lengths.append(int(process_md["grid_length"]))
            step_ts_lengths.append(
                int(sum(bool(flag) for flag in process_md["step_ts_mask"]))
            )

        return cls(
            process_order=process_order,
            global_control_names=global_control_names,
            global_control_name_to_index=global_control_name_to_index,
            shape_metadata=dict(bp_train["shape_metadata"]),
            dense_grid=_as_jax_array(dense_grid_rows),
            control_values=_as_jax_array(control_value_rows),
            control_derivatives=_as_jax_array(control_derivative_rows),
            step_ts=_as_jax_array(step_ts_rows),
            grid_lengths=jnp.asarray(grid_lengths, dtype=jnp.int32),
            step_ts_lengths=jnp.asarray(step_ts_lengths, dtype=jnp.int32),
            sample_acc_global_index=global_control_name_to_index["V_sample_acc"],
            _process_md_by_name=processes_metadata,
        )

    @classmethod
    def from_json(
        cls,
        prepared_json: str | Path,
        *,
        metadata_namespace: str = "bp_train",
    ) -> ControlsStore:
        """Load a prepared JSON artifact and construct a `ControlsStore`."""
        collection = load_process_collection_json(Path(prepared_json))
        return cls.from_collection(collection, metadata_namespace=metadata_namespace)

    def get_controls(self, process: str | int) -> PerProcessControls:
        """Return per-process controls by canonical prepared key or index."""
        process_name, process_index = _coerce_index(process, self.process_order)
        process_md = self._process_md_by_name[process_name]
        sample_acc_name = str(process_md["sample_acc_name"])
        local_names = list(process_md["local_control_names"])

        return PerProcessControls(
            process_name=process_name,
            process_index=process_index,
            control_names=local_names,
            control_name_to_index={name: idx for idx, name in enumerate(local_names)},
            global_control_names=self.global_control_names,
            global_control_name_to_index=self.global_control_name_to_index,
            dense_grid=self.dense_grid[process_index],
            control_values=self.control_values[process_index],
            control_derivatives=self.control_derivatives[process_index],
            step_ts=self.step_ts[process_index],
            grid_length=int(self.grid_lengths[process_index]),
            step_ts_length=int(self.step_ts_lengths[process_index]),
            control_metadata=dict(process_md["control_metadata"]),
            sample_acc_name=sample_acc_name,
            sample_acc_global_index=self.sample_acc_global_index,
        )
