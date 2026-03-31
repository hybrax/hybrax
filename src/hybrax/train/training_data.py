from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from bpbench.dataclasses import BioProcessCollection
from bpbench.serialization import load_process_collection_json

from .controls_store import ControlsStore, PerProcessControls


TARGET_SOURCE_AUTO = "auto"
TARGET_SOURCE_PROCESS_VARIABLES = "process_variables"
TARGET_SOURCE_REACTOR_COMPONENTS = "reactor_components"
TARGET_SOURCES = {
    TARGET_SOURCE_AUTO,
    TARGET_SOURCE_PROCESS_VARIABLES,
    TARGET_SOURCE_REACTOR_COMPONENTS,
}


def _normalize_target_source(target_source: str) -> str:
    value = str(target_source)
    if value not in TARGET_SOURCES:
        raise ValueError(
            f"target_source must be one of {sorted(TARGET_SOURCES)!r}, got {value!r}"
        )
    return value


def _coerce_process_index(
    process: str | int,
    process_order: list[str],
) -> tuple[str, int]:
    """Resolve process key/index to canonical process name and index."""
    if isinstance(process, str):
        if process not in process_order:
            raise KeyError(f"unknown process name: {process}")
        return process, process_order.index(process)

    index = int(process)
    if index < 0 or index >= len(process_order):
        raise IndexError(f"process index out of range: {index}")
    return process_order[index], index


def _process_variable_targets(process) -> list[str]:
    return [
        name
        for name, variable in process.process_variables.items()
        if not bool(variable.is_controlled)
    ]


def _is_timeseries_compatible(values) -> bool:
    if not hasattr(values, "times") or not hasattr(values, "values"):
        return False
    ts = np.asarray(values.times)
    ys = np.asarray(values.values)
    return (
        ts.ndim == 1
        and ys.ndim == 1
        and ts.size > 0
        and ys.size > 0
        and ts.size == ys.size
    )


def _reactor_component_timeseries_targets(process) -> list[str]:
    components = getattr(process.reactor_medium, "components", {}) or {}
    targets: list[str] = []
    for name, component in components.items():
        if _is_timeseries_compatible(component.concentration):
            targets.append(name)
    return targets


def _supports_configured_process_variables(
    process,
    configured_order: list[str],
) -> bool:
    for name in configured_order:
        variable = process.process_variables.get(name)
        if variable is None or bool(variable.is_controlled):
            return False
    return True


def _supports_configured_reactor_components(
    process,
    configured_order: list[str],
) -> bool:
    components = getattr(process.reactor_medium, "components", {}) or {}
    for name in configured_order:
        component = components.get(name)
        if component is None:
            return False
        if not _is_timeseries_compatible(component.concentration):
            return False
    return True


def _resolve_target_source(
    collection: BioProcessCollection,
    process_order: list[str],
    configured_order: list[str] | None,
    target_source: str,
) -> str:
    requested = _normalize_target_source(target_source)
    if requested != TARGET_SOURCE_AUTO:
        return requested

    if configured_order is not None:
        process_variable_ok = all(
            _supports_configured_process_variables(
                collection.processes[process_name],
                configured_order,
            )
            for process_name in process_order
        )
        if process_variable_ok:
            return TARGET_SOURCE_PROCESS_VARIABLES

        reactor_component_ok = all(
            _supports_configured_reactor_components(
                collection.processes[process_name],
                configured_order,
            )
            for process_name in process_order
        )
        if reactor_component_ok:
            return TARGET_SOURCE_REACTOR_COMPONENTS

        raise ValueError(
            "target_source='auto' could not resolve configured targets across "
            "all processes as either measured process variables or reactor "
            "components"
        )

    if all(
        len(_process_variable_targets(collection.processes[process_name])) > 0
        for process_name in process_order
    ):
        return TARGET_SOURCE_PROCESS_VARIABLES

    if all(
        len(_reactor_component_timeseries_targets(collection.processes[process_name]))
        > 0
        for process_name in process_order
    ):
        return TARGET_SOURCE_REACTOR_COMPONENTS

    raise ValueError(
        "target_source='auto' could not find a valid shared target source "
        "across processes"
    )


def _measurement_targets(
    process,
    configured_order: list[str] | None,
    target_source: str,
) -> list[str]:
    """Determine ordered measured target names for one process."""
    if target_source == TARGET_SOURCE_PROCESS_VARIABLES:
        measured = _process_variable_targets(process)
        if configured_order is None:
            return measured
        missing = [
            name for name in configured_order if name not in process.process_variables
        ]
        if missing:
            raise ValueError(
                f"{process.metadata.name}: configured target variables missing from process "
                f"variables: {missing}"
            )
        controlled = [
            name
            for name in configured_order
            if bool(process.process_variables[name].is_controlled)
        ]
        if controlled:
            raise ValueError(
                f"{process.metadata.name}: configured target variables must be measured "
                f"(is_controlled=False), got controlled targets: {controlled}"
            )
        return list(configured_order)

    if target_source == TARGET_SOURCE_REACTOR_COMPONENTS:
        measured = _reactor_component_timeseries_targets(process)
        if configured_order is None:
            return measured
        components = getattr(process.reactor_medium, "components", {}) or {}
        missing = [name for name in configured_order if name not in components]
        if missing:
            raise ValueError(
                f"{process.metadata.name}: configured target components missing from "
                f"reactor_medium.components: {missing}"
            )
        invalid = [
            name
            for name in configured_order
            if not _is_timeseries_compatible(components[name].concentration)
        ]
        if invalid:
            raise ValueError(
                f"{process.metadata.name}: configured target components must be "
                "time-series compatible (times+values), got: "
                f"{invalid}"
            )
        return list(configured_order)

    raise ValueError(f"unsupported target_source: {target_source!r}")


def _timeseries_numpy(
    process,
    target_name: str,
    target_source: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract measurement time/value arrays from one target source."""
    if target_source == TARGET_SOURCE_PROCESS_VARIABLES:
        values = process.process_variables[target_name].values
    elif target_source == TARGET_SOURCE_REACTOR_COMPONENTS:
        values = process.reactor_medium.components[target_name].concentration
    else:
        raise ValueError(f"unsupported target_source: {target_source!r}")

    if not hasattr(values, "times") or not hasattr(values, "values"):
        raise ValueError(
            f"{process.metadata.name}: target {target_name!r} must be a "
            "time-series variable with times and values"
        )
    ts = np.asarray(values.times, dtype=float)
    ys = np.asarray(values.values, dtype=float)
    if ts.ndim != 1 or ys.ndim != 1 or ts.size != ys.size:
        raise ValueError(
            f"{process.metadata.name}: target {target_name!r} has invalid "
            "time/value shape"
        )
    if ts.size == 0:
        raise ValueError(
            f"{process.metadata.name}: target {target_name!r} has no measurement points"
        )
    return ts, ys


class PerProcessTrainingData(eqx.Module):
    """Per-process active view over padded training-data tensors."""

    # Canonical process key from prepared metadata.
    process_name: str
    # Integer row index into collection-level stacked arrays.
    process_index: int
    # Ordered measured target names.
    target_names: list[str]
    # Source family of targets (`process_variables` or `reactor_components`).
    target_source: str
    # Number of active measurement rows for this process.
    n_meas: int
    # Padded measurement times for this process.
    t_meas: jax.Array
    # Padded measurement values for this process, columns follow `target_names`.
    y_meas: jax.Array
    # Padded boolean mask for active measurement rows.
    meas_mask: jax.Array
    # Initial state `[targets..., V_cont]`.
    y0: jax.Array
    # Per-process controls view from ControlsStore.
    controls: PerProcessControls

    @property
    def active_t_meas(self) -> jax.Array:
        """Active measurement-time prefix."""
        return self.t_meas[: self.n_meas]

    @property
    def active_y_meas(self) -> jax.Array:
        """Active measurement/value prefix."""
        return self.y_meas[: self.n_meas]

    @property
    def active_meas_mask(self) -> jax.Array:
        """Active measurement-mask prefix."""
        return self.meas_mask[: self.n_meas]


class BatchTrainingData(eqx.Module):
    """Batch view over stacked training tensors for process-index batches."""

    # Process indices used to gather this batch view.
    process_indices: jax.Array
    # Gathered measurement times `[batch_size, max_n_meas]`.
    t_meas: jax.Array
    # Gathered measurement values `[batch_size, max_n_meas, n_targets]`.
    y_meas: jax.Array
    # Gathered measurement masks `[batch_size, max_n_meas]`.
    meas_mask: jax.Array
    # Gathered active measurement counts `[batch_size]`.
    n_meas: jax.Array
    # Gathered initial state vectors `[batch_size, n_targets + 1]`.
    y0: jax.Array


class TrainingDataStore(eqx.Module):
    """Collection-level training-data store built from a prepared collection."""

    # Stable process order across all stacked arrays.
    process_order: list[str]
    # Ordered measured target names (shared across processes).
    target_names: list[str]
    # Source family of targets (`process_variables` or `reactor_components`).
    target_source: str
    # Mapping from target name to column index in y-measurement arrays.
    target_name_to_index: dict[str, int]
    # Shared controls store for this prepared artifact.
    controls_store: ControlsStore
    # Padded measurement times `[n_processes, max_n_meas]`.
    t_meas: jax.Array
    # Padded measurement values `[n_processes, max_n_meas, n_targets]`.
    y_meas: jax.Array
    # Padded measurement mask `[n_processes, max_n_meas]`.
    meas_mask: jax.Array
    # Active measurement counts per process.
    n_meas: jax.Array
    # Initial state matrix `[n_processes, n_targets + 1]` where last entry is `V_cont(0)`.
    y0: jax.Array

    @classmethod
    def from_collection(
        cls,
        collection: BioProcessCollection,
        *,
        target_variable_order: list[str] | None = None,
        target_source: str = TARGET_SOURCE_PROCESS_VARIABLES,
        metadata_namespace: str = "bp_train",
    ) -> TrainingDataStore:
        """Build training-data tensors from a prepared process collection."""
        controls_store = ControlsStore.from_collection(
            collection,
            metadata_namespace=metadata_namespace,
        )
        process_order = list(controls_store.process_order)

        target_order = list(target_variable_order) if target_variable_order else None
        resolved_target_source = _resolve_target_source(
            collection=collection,
            process_order=process_order,
            configured_order=target_order,
            target_source=target_source,
        )
        per_process_targets: dict[str, list[str]] = {}
        reference_targets: list[str] | None = None

        for process_name in process_order:
            process = collection.processes[process_name]
            current_targets = _measurement_targets(
                process,
                target_order,
                resolved_target_source,
            )
            per_process_targets[process_name] = current_targets

            if reference_targets is None:
                reference_targets = list(current_targets)
            elif current_targets != reference_targets:
                raise ValueError(
                    "training data requires identical measured target names/order "
                    f"across processes; {process_name!r} has {current_targets!r} "
                    f"but expected {reference_targets!r}"
                )

        if reference_targets is None:
            raise ValueError("process collection is empty")
        if len(reference_targets) == 0:
            raise ValueError("no measured target variables found in process collection")

        target_names = list(reference_targets)
        target_name_to_index = {name: idx for idx, name in enumerate(target_names)}

        per_process_times: list[np.ndarray] = []
        per_process_values: list[np.ndarray] = []
        per_process_y0: list[np.ndarray] = []
        n_meas_list: list[int] = []
        max_n_meas = 0

        for process_name in process_order:
            process = collection.processes[process_name]
            process_targets = per_process_targets[process_name]

            target_columns = []
            shared_ts: np.ndarray | None = None
            for target_name in process_targets:
                ts, ys = _timeseries_numpy(
                    process,
                    target_name,
                    resolved_target_source,
                )
                if shared_ts is None:
                    shared_ts = ts
                elif not np.array_equal(ts, shared_ts):
                    raise ValueError(
                        f"{process_name}: measurement times differ across targets; "
                        "V1 expects aligned measurement timestamps per process"
                    )
                target_columns.append(ys)

            if shared_ts is None:
                raise ValueError(f"{process_name}: no measurement data for targets")

            y_matrix = np.stack(target_columns, axis=1)
            n_meas = int(shared_ts.size)
            max_n_meas = max(max_n_meas, n_meas)

            y0_targets = y_matrix[0]
            y0 = np.concatenate(
                [y0_targets, np.asarray([float(process.volume.initial_volume)])],
                axis=0,
            )

            per_process_times.append(shared_ts)
            per_process_values.append(y_matrix)
            per_process_y0.append(y0)
            n_meas_list.append(n_meas)

        n_processes = len(process_order)
        n_targets = len(target_names)
        t_meas = np.full((n_processes, max_n_meas), np.nan, dtype=np.float32)
        y_meas = np.full((n_processes, max_n_meas, n_targets), np.nan, dtype=np.float32)
        meas_mask = np.zeros((n_processes, max_n_meas), dtype=bool)

        for index, (ts, ys) in enumerate(
            zip(per_process_times, per_process_values, strict=False)
        ):
            n_meas = n_meas_list[index]
            t_meas[index, :n_meas] = ts.astype(np.float32)
            y_meas[index, :n_meas, :] = ys.astype(np.float32)
            meas_mask[index, :n_meas] = True

        return cls(
            process_order=process_order,
            target_names=target_names,
            target_source=resolved_target_source,
            target_name_to_index=target_name_to_index,
            controls_store=controls_store,
            t_meas=jnp.asarray(t_meas),
            y_meas=jnp.asarray(y_meas),
            meas_mask=jnp.asarray(meas_mask),
            n_meas=jnp.asarray(n_meas_list, dtype=jnp.int32),
            y0=jnp.asarray(np.asarray(per_process_y0, dtype=np.float32)),
        )

    @classmethod
    def from_json(
        cls,
        prepared_json: str | Path,
        *,
        target_variable_order: list[str] | None = None,
        target_source: str = TARGET_SOURCE_PROCESS_VARIABLES,
        metadata_namespace: str = "bp_train",
    ) -> TrainingDataStore:
        """Load a prepared JSON artifact and construct a training-data store."""
        collection = load_process_collection_json(Path(prepared_json))
        return cls.from_collection(
            collection,
            target_variable_order=target_variable_order,
            target_source=target_source,
            metadata_namespace=metadata_namespace,
        )

    def get_process(self, process: str | int) -> PerProcessTrainingData:
        """Return per-process training data by canonical name or integer index."""
        process_name, process_index = _coerce_process_index(process, self.process_order)
        return PerProcessTrainingData(
            process_name=process_name,
            process_index=process_index,
            target_names=self.target_names,
            target_source=self.target_source,
            n_meas=int(self.n_meas[process_index]),
            t_meas=self.t_meas[process_index],
            y_meas=self.y_meas[process_index],
            meas_mask=self.meas_mask[process_index],
            y0=self.y0[process_index],
            controls=self.controls_store.get_controls(process_name),
        )

    def gather_batch(
        self, process_indices: jax.Array | np.ndarray
    ) -> BatchTrainingData:
        """Gather a process-index batch view over stacked training-data tensors."""
        indices = jnp.asarray(process_indices, dtype=jnp.int32)
        if indices.ndim != 1:
            raise ValueError("process_indices must be a 1D array")
        if indices.size == 0:
            raise ValueError("process_indices must be non-empty")

        n_processes = len(self.process_order)
        if bool(jnp.any(indices < 0)) or bool(jnp.any(indices >= n_processes)):
            raise IndexError("process index out of range in process_indices")

        return BatchTrainingData(
            process_indices=indices,
            t_meas=self.t_meas[indices],
            y_meas=self.y_meas[indices],
            meas_mask=self.meas_mask[indices],
            n_meas=self.n_meas[indices],
            y0=self.y0[indices],
        )
