from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from bpbench.dataclasses import BioProcessCollection
from bpbench.serialization import load_process_collection_json

from .controls import (
    BP_TRAIN_SAMPLE_ACC_NAME,
    SignalSource,
    build_dense_payload,
    build_sample_acc_source_default,
    compute_signal_spreads,
    select_control_sources,
)


DEFAULT_RUNTIME_CONTROLS_CONFIG: dict[str, Any] = {
    "initial_grid_points": 16,
    "max_rel_error": 1e-4,
    "max_refinement_rounds": 8,
}


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


def _piecewise_linear_derivative(
    ts: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
) -> np.ndarray:
    if xp.size <= 1:
        return np.zeros_like(ts, dtype=float)
    dx = np.diff(xp)
    slopes = np.divide(np.diff(fp), dx, out=np.zeros_like(dx), where=dx != 0)
    indices = np.searchsorted(xp[1:], ts, side="right")
    indices = np.clip(indices, 0, slopes.size - 1)
    return slopes[indices]


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
    # Padded control values in shared control order, shape
    # `[max_grid_length, max_controls]`.
    control_values: jax.Array
    # Padded control derivatives in shared control order, same shape as
    # `control_values`.
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


class BatchControls(eqx.Module):
    """All-process controls evaluator with index-based runtime lookup."""

    # Padded dense grids `[n_processes, max_grid_length]` with right-clamped tail.
    dense_grid: jax.Array
    # Padded control values `[n_processes, max_grid_length, max_controls]`.
    control_values: jax.Array

    def eval(self, process_idx: int, t: jax.Array) -> jax.Array:
        """Evaluate controls for one process index at one or more times."""
        if isinstance(process_idx, (int, np.integer)):
            idx = int(process_idx)
            n_processes = int(self.dense_grid.shape[0])
            if idx < 0 or idx >= n_processes:
                raise IndexError(f"process index out of range: {idx}")

        grid = self.dense_grid[process_idx]
        values = self.control_values[process_idx]

        query = jnp.asarray(t, dtype=grid.dtype)
        scalar_input = query.ndim == 0
        query_1d = jnp.atleast_1d(query)
        out = _interp_columns(query_1d, grid, values)
        if scalar_input:
            return out[0]
        return out


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
    # Stacked padded control values, shape
    # `[n_processes, max_grid_length, max_controls]`.
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
    # Runtime-built per-process metadata entries needed to construct thin views.
    _process_md_by_name: dict[str, dict[str, Any]]

    @staticmethod
    def _runtime_controls_config(
        metadata: dict[str, Any],
        metadata_namespace: str,
    ) -> dict[str, Any]:
        bp_train = metadata.get(metadata_namespace, {})
        cfg = dict(DEFAULT_RUNTIME_CONTROLS_CONFIG)
        cfg.update(dict(bp_train.get("runtime_controls_config", {})))
        return cfg

    @staticmethod
    def _process_order(
        collection: BioProcessCollection,
        metadata: dict[str, Any],
        metadata_namespace: str,
    ) -> list[str]:
        bp_train = metadata.get(metadata_namespace, {})
        process_order = bp_train.get("process_order")
        if process_order is None:
            return list(collection.processes.keys())
        return list(process_order)

    @staticmethod
    def _ordered_control_sources(
        process_name: str,
        sources: list[Any],
        process_md: dict[str, Any] | None,
    ) -> tuple[list[str], list[Any]]:
        source_by_name = {source.name: source for source in sources}
        source_names = list(source_by_name.keys())

        if not process_md or "local_control_names" not in process_md:
            return source_names, [source_by_name[name] for name in source_names]

        local_control_names = list(process_md["local_control_names"])
        expected_names = [
            name for name in local_control_names if name != BP_TRAIN_SAMPLE_ACC_NAME
        ]

        if set(expected_names) != set(source_names):
            raise ValueError(
                f"{process_name}: controls derived from prepared process do not match "
                "prepared metadata local_control_names"
            )

        return expected_names, [source_by_name[name] for name in expected_names]

    @staticmethod
    def _sample_source_from_prepared_metadata(
        process_name: str,
        process_md: dict[str, Any] | None,
    ) -> SignalSource | None:
        if not process_md:
            return None
        sample_md = process_md.get("sample_acc_source")
        if sample_md is None:
            return None
        times = np.asarray(sample_md["times"], dtype=float)
        values = np.asarray(sample_md["values"], dtype=float)
        if times.size == 0 or values.size == 0 or times.size != values.size:
            raise ValueError(
                f"{process_name}: invalid sample_acc_source in prepared metadata"
            )
        return SignalSource(
            name=BP_TRAIN_SAMPLE_ACC_NAME,
            kind="derived_control",
            times=times,
            values=values,
            evaluator=lambda ts: np.interp(
                np.asarray(ts, dtype=float),
                times,
                values,
                left=float(values[0]),
                right=float(values[-1]),
            ),
            derivative=lambda ts: _piecewise_linear_derivative(
                np.asarray(ts, dtype=float), times, values
            ),
            step_ts=[float(v) for v in sample_md.get("step_ts", [])],
            metadata=dict(sample_md.get("metadata", {})),
        )

    @staticmethod
    def _pad_payload(
        *,
        payload: dict[str, Any],
        local_control_names: list[str],
        global_control_names: list[str],
        max_grid_length: int,
        max_step_ts_length: int,
    ) -> tuple[
        list[float], list[list[float]], list[list[float]], list[float], int, int
    ]:
        grid = list(payload["grid"])
        values = [list(row) for row in payload["values"]]
        derivatives = [list(row) for row in payload["derivatives"]]
        step_ts = list(payload["step_ts"])

        grid_length = len(grid)
        step_ts_length = len(step_ts)
        max_controls = len(global_control_names)
        global_index = {name: idx for idx, name in enumerate(global_control_names)}

        dense_grid = grid + [0.0] * (max_grid_length - grid_length)

        control_values = []
        control_derivatives = []
        for row, deriv_row in zip(values, derivatives, strict=False):
            global_row = [0.0] * max_controls
            global_deriv = [0.0] * max_controls
            for local_idx, control_name in enumerate(local_control_names):
                idx = global_index[control_name]
                global_row[idx] = row[local_idx]
                global_deriv[idx] = deriv_row[local_idx]
            control_values.append(global_row)
            control_derivatives.append(global_deriv)

        zero_row = [0.0] * max_controls
        for _ in range(max_grid_length - grid_length):
            control_values.append(list(zero_row))
            control_derivatives.append(list(zero_row))

        step_ts_padded = step_ts + [0.0] * (max_step_ts_length - step_ts_length)
        return (
            dense_grid,
            control_values,
            control_derivatives,
            step_ts_padded,
            grid_length,
            step_ts_length,
        )

    @classmethod
    def from_collection(
        cls,
        collection: BioProcessCollection,
        *,
        metadata_namespace: str = "bp_train",
    ) -> ControlsStore:
        """Build a JAX-backed runtime store from a prepared `BioProcessCollection`."""
        metadata = dict(collection.metadata or {})
        cfg = cls._runtime_controls_config(metadata, metadata_namespace)
        process_order = cls._process_order(collection, metadata, metadata_namespace)
        bp_train = dict(metadata.get(metadata_namespace, {}))
        prepared_process_md = dict(bp_train.get("processes", {}))

        process_sources: dict[str, list[Any]] = {}
        process_sample_sources: dict[str, Any] = {}
        process_control_names: dict[str, list[str]] = {}
        process_control_metadata: dict[str, dict[str, Any]] = {}
        reference_control_names: list[str] | None = None

        for process_name in process_order:
            process = collection.processes[process_name]
            selected = select_control_sources(
                process_name=process_name,
                process=process,
                config=cfg,
            )
            prepared_md = prepared_process_md.get(process_name)
            ordered_names, ordered_sources = cls._ordered_control_sources(
                process_name=process_name,
                sources=selected,
                process_md=prepared_md,
            )
            sample_source = cls._sample_source_from_prepared_metadata(
                process_name=process_name,
                process_md=prepared_md,
            )
            if sample_source is None:
                sample_source = build_sample_acc_source_default(process)
            if sample_source.name != BP_TRAIN_SAMPLE_ACC_NAME:
                raise ValueError(
                    f"{process_name}: sample control must be named "
                    f"{BP_TRAIN_SAMPLE_ACC_NAME}"
                )

            local_names = [*ordered_names, sample_source.name]
            if reference_control_names is None:
                reference_control_names = local_names
            elif local_names != reference_control_names:
                raise ValueError(
                    "controls store requires identical control names/order across "
                    f"processes; {process_name!r} has {local_names!r} but "
                    f"expected {reference_control_names!r}"
                )

            process_sources[process_name] = ordered_sources
            process_sample_sources[process_name] = sample_source
            process_control_names[process_name] = local_names
            process_control_metadata[process_name] = {
                source.name: source.metadata
                for source in [*ordered_sources, sample_source]
            }

        if reference_control_names is None:
            raise ValueError("process collection is empty")

        global_control_names = list(reference_control_names)

        global_control_name_to_index = {
            name: idx for idx, name in enumerate(global_control_names)
        }

        spread_inputs = {
            process_name: [
                *process_sources[process_name],
                process_sample_sources[process_name],
            ]
            for process_name in process_order
        }
        spreads = compute_signal_spreads(spread_inputs)

        payloads_by_process: dict[str, dict[str, Any]] = {}
        max_grid_length = 0
        max_step_ts_length = 0
        for process_name in process_order:
            process = collection.processes[process_name]
            sources = [
                *process_sources[process_name],
                process_sample_sources[process_name],
            ]
            payload = build_dense_payload(
                process=process,
                sources=sources,
                spreads=spreads,
                config=cfg,
            )
            payloads_by_process[process_name] = payload
            max_grid_length = max(max_grid_length, len(payload["grid"]))
            max_step_ts_length = max(max_step_ts_length, len(payload["step_ts"]))

        dense_grid_rows = []
        control_value_rows = []
        control_derivative_rows = []
        step_ts_rows = []
        grid_lengths = []
        step_ts_lengths = []
        processes_metadata: dict[str, dict[str, Any]] = {}

        for process_name in process_order:
            payload = payloads_by_process[process_name]
            local_names = process_control_names[process_name]
            (
                dense_grid,
                control_values,
                control_derivatives,
                step_ts,
                grid_length,
                step_ts_length,
            ) = cls._pad_payload(
                payload=payload,
                local_control_names=local_names,
                global_control_names=global_control_names,
                max_grid_length=max_grid_length,
                max_step_ts_length=max_step_ts_length,
            )

            dense_grid_rows.append(dense_grid)
            control_value_rows.append(control_values)
            control_derivative_rows.append(control_derivatives)
            step_ts_rows.append(step_ts)
            grid_lengths.append(grid_length)
            step_ts_lengths.append(step_ts_length)
            processes_metadata[process_name] = {
                "local_control_names": local_names,
                "control_metadata": process_control_metadata[process_name],
                "sample_acc_name": BP_TRAIN_SAMPLE_ACC_NAME,
            }

        shape_metadata = {
            "n_processes": len(process_order),
            "max_grid_length": max_grid_length,
            "max_controls": len(global_control_names),
            "max_step_ts_length": max_step_ts_length,
        }

        return cls(
            process_order=process_order,
            global_control_names=global_control_names,
            global_control_name_to_index=global_control_name_to_index,
            shape_metadata=shape_metadata,
            dense_grid=_as_jax_array(dense_grid_rows),
            control_values=_as_jax_array(control_value_rows),
            control_derivatives=_as_jax_array(control_derivative_rows),
            step_ts=_as_jax_array(step_ts_rows),
            grid_lengths=jnp.asarray(grid_lengths, dtype=jnp.int32),
            step_ts_lengths=jnp.asarray(step_ts_lengths, dtype=jnp.int32),
            sample_acc_global_index=global_control_name_to_index[
                BP_TRAIN_SAMPLE_ACC_NAME
            ],
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

        return PerProcessControls(
            process_name=process_name,
            process_index=process_index,
            control_names=self.global_control_names,
            control_name_to_index=self.global_control_name_to_index,
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

    def as_batch_controls(self) -> BatchControls:
        """Build a minimal index-based controls evaluator for batch training."""
        max_grid_length = self.dense_grid.shape[1]
        tail_mask = (
            jnp.arange(max_grid_length, dtype=jnp.int32)[None, :]
            >= self.grid_lengths[:, None]
        )
        last_index = jnp.clip(self.grid_lengths - 1, 0, max_grid_length - 1)
        last_grid = self.dense_grid[
            jnp.arange(self.dense_grid.shape[0], dtype=jnp.int32),
            last_index,
        ]
        last_values = self.control_values[
            jnp.arange(self.control_values.shape[0], dtype=jnp.int32),
            last_index,
            :,
        ]
        grid_clamped = jnp.where(tail_mask, last_grid[:, None], self.dense_grid)
        values_clamped = jnp.where(
            tail_mask[:, :, None],
            last_values[:, None, :],
            self.control_values,
        )
        return BatchControls(
            dense_grid=grid_clamped,
            control_values=values_clamped,
        )
