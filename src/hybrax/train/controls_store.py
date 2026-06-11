from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from bp_format.dataclasses import (
    BioProcessCollection,
)
from bp_format.serialization import load_process_collection_json

from .constants import METADATA_NAMESPACE
from .controls import (
    BP_TRAIN_SAMPLE_ACC_NAME,
    EVENT_RUN_MIN_DT_CONFIG_KEY,
    ControlSourceBundle,
    SignalSource,
    build_dense_payload,
    build_sample_acc_source_default,
    compute_signal_spreads,
    get_collection_event_min_dt_if_needed,
    run_min_dt_from_runtime_controls,
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
    """Per-process runtime view over padded, canonical-axis dense-grid controls.

    Column axis follows
    ``[name_controlled_FVCs | name_controlled_SVCs | name_controlled_PVs | name_extras]``
    matching bp-format ``ControlSplines`` plus bp-train's extras block
    (bolus-FVC triangle ramps + the cumulative sample-acc trace at the
    very end). The first ``len(name_controlled_FVCs) +
    len(name_controlled_SVCs) + len(name_controlled_PVs)`` columns are
    consumed by ``eval_u(t)`` to build RhsOde's ``u`` argument; the
    extras tail carries bp-train-specific signals (bolus dilution and
    sample-volume bookkeeping) that the wrapper indexes directly.

    All non-array fields are ``eqx.field(static=True)`` so they live in
    the pytree treedef rather than as dynamic leaves.
    """

    process_name: str = eqx.field(static=True)
    process_index: int = eqx.field(static=True)
    name_controlled_FVCs: tuple[str, ...] = eqx.field(static=True)
    name_controlled_SVCs: tuple[str, ...] = eqx.field(static=True)
    name_controlled_PVs: tuple[str, ...] = eqx.field(static=True)
    name_extras: tuple[str, ...] = eqx.field(static=True)
    dense_grid: jax.Array
    control_values: jax.Array
    control_derivatives: jax.Array
    step_ts: jax.Array
    grid_length: int = eqx.field(static=True)
    step_ts_length: int = eqx.field(static=True)
    control_metadata: dict[str, dict[str, Any]] = eqx.field(static=True)
    sample_acc_name: str = eqx.field(static=True)
    sample_acc_global_index: int = eqx.field(static=True)

    @property
    def n_u(self) -> int:
        return (
            len(self.name_controlled_FVCs)
            + len(self.name_controlled_SVCs)
            + len(self.name_controlled_PVs)
        )

    @property
    def active_dense_grid(self) -> jax.Array:
        return self.dense_grid[: self.grid_length]

    @property
    def active_step_ts(self) -> jax.Array:
        return self.step_ts[: self.step_ts_length]

    @property
    def active_control_values(self) -> jax.Array:
        return self.control_values[: self.grid_length]

    @property
    def active_control_derivatives(self) -> jax.Array:
        return self.control_derivatives[: self.grid_length]

    def eval(self, ts: float | np.ndarray | jax.Array) -> jax.Array:
        """Evaluate all controls at one or more times in canonical order."""
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
        """Evaluate precomputed control derivatives in canonical order."""
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

    def eval_u(self, ts: float | np.ndarray | jax.Array) -> jax.Array:
        """Evaluate RhsOde's u vector: ``[FVC_flows | SVC_flows | PV_values]``.

        Flows come from precomputed derivatives of the cumulative-volume
        signals; PV values come from the raw signal trace.
        """
        n_fvc = len(self.name_controlled_FVCs)
        n_svc = len(self.name_controlled_SVCs)
        n_pv = len(self.name_controlled_PVs)
        derivatives = self.eval_derivative(ts)
        values = self.eval(ts)
        flows = derivatives[..., : n_fvc + n_svc]
        pvs = values[..., n_fvc + n_svc : n_fvc + n_svc + n_pv]
        return jnp.concatenate([flows, pvs], axis=-1)


class BatchControls(eqx.Module):
    """All-process controls evaluator with index-based runtime lookup.

    Column axis follows the same canonical
    ``[name_controlled_FVCs | name_controlled_SVCs | name_controlled_PVs | name_extras]``
    order as :class:`PerProcessControls`.
    """

    # Padded dense grids `[n_processes, max_grid_length]` with right-clamped tail.
    dense_grid: jax.Array
    # Padded control values `[n_processes, max_grid_length, max_controls]`.
    control_values: jax.Array
    # Padded control derivatives, same shape as control_values.
    control_derivatives: jax.Array
    name_controlled_FVCs: tuple[str, ...] = eqx.field(static=True)
    name_controlled_SVCs: tuple[str, ...] = eqx.field(static=True)
    name_controlled_PVs: tuple[str, ...] = eqx.field(static=True)
    name_extras: tuple[str, ...] = eqx.field(static=True)

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

    def eval_derivative(self, process_idx: int, t: jax.Array) -> jax.Array:
        """Evaluate control derivatives for one process index at one or more times."""
        grid = self.dense_grid[process_idx]
        derivatives = self.control_derivatives[process_idx]

        query = jnp.asarray(t, dtype=grid.dtype)
        scalar_input = query.ndim == 0
        query_1d = jnp.atleast_1d(query)
        out = _interp_columns(query_1d, grid, derivatives)
        if scalar_input:
            return out[0]
        return out

    def eval_u(self, process_idx: int, t: jax.Array) -> jax.Array:
        """Evaluate RhsOde's u vector for one process index."""
        n_fvc = len(self.name_controlled_FVCs)
        n_svc = len(self.name_controlled_SVCs)
        n_pv = len(self.name_controlled_PVs)
        derivatives = self.eval_derivative(process_idx, t)
        values = self.eval(process_idx, t)
        flows = derivatives[..., : n_fvc + n_svc]
        pvs = values[..., n_fvc + n_svc : n_fvc + n_svc + n_pv]
        return jnp.concatenate([flows, pvs], axis=-1)


class ControlsStore(eqx.Module):
    """Collection-level loader and index for prepared, padded JAX control tensors.

    Column axis follows
    ``[name_controlled_FVCs | name_controlled_SVCs | name_controlled_PVs | name_extras]``
    consistently across every process; the wrapper consumes the leading
    u-block via :meth:`PerProcessControls.eval_u`.
    """

    # Canonical prepared process keys in stable collection order.
    process_order: list[str]
    # Categorised name tuples (must be identical across processes).
    name_controlled_FVCs: tuple[str, ...]
    name_controlled_SVCs: tuple[str, ...]
    name_controlled_PVs: tuple[str, ...]
    name_extras: tuple[str, ...]
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
    # Column index of the cumulative sampled-volume signal (always at the very end of name_extras).
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
    def _validate_bundle_against_metadata(
        process_name: str,
        bundle: ControlSourceBundle,
        process_md: dict[str, Any] | None,
    ) -> None:
        if not process_md:
            return
        expected_extras = bundle.name_extras_bolus + (BP_TRAIN_SAMPLE_ACC_NAME,)
        for key, expected in (
            ("name_controlled_FVCs", bundle.name_controlled_FVCs),
            ("name_controlled_SVCs", bundle.name_controlled_SVCs),
            ("name_controlled_PVs", bundle.name_controlled_PVs),
            ("name_extras", expected_extras),
        ):
            if key not in process_md:
                continue
            stored = tuple(process_md[key])
            if stored != expected:
                raise ValueError(
                    f"{process_name}: prepared metadata {key}={stored!r} "
                    f"does not match derived {expected!r}"
                )

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
        payload_source_names: list[str],
        canonical_names: list[str],
        max_grid_length: int,
        max_step_ts_length: int,
    ) -> tuple[
        list[float], list[list[float]], list[list[float]], list[float], int, int
    ]:
        """Reorder payload columns from build order to canonical order, then pad."""
        grid = list(payload["grid"])
        values = [list(row) for row in payload["values"]]
        derivatives = [list(row) for row in payload["derivatives"]]
        step_ts = list(payload["step_ts"])

        grid_length = len(grid)
        step_ts_length = len(step_ts)
        max_controls = len(canonical_names)
        canonical_index = {name: idx for idx, name in enumerate(canonical_names)}

        dense_grid = grid + [0.0] * (max_grid_length - grid_length)

        control_values = []
        control_derivatives = []
        for row, deriv_row in zip(values, derivatives, strict=False):
            canonical_row = [0.0] * max_controls
            canonical_deriv = [0.0] * max_controls
            for local_idx, control_name in enumerate(payload_source_names):
                idx = canonical_index[control_name]
                canonical_row[idx] = row[local_idx]
                canonical_deriv[idx] = deriv_row[local_idx]
            control_values.append(canonical_row)
            control_derivatives.append(canonical_deriv)

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
    ) -> ControlsStore:
        """Build a JAX-backed runtime store from a prepared `BioProcessCollection`."""
        metadata = dict(collection.metadata or {})
        cfg = cls._runtime_controls_config(metadata, METADATA_NAMESPACE)
        process_order = cls._process_order(collection, metadata, METADATA_NAMESPACE)
        bp_train = dict(metadata.get(METADATA_NAMESPACE, {}))
        prepared_process_md = dict(bp_train.get("processes", {}))
        needs_default_sample_sources = any(
            "sample_acc_source" not in dict(prepared_process_md.get(process_name) or {})
            for process_name in process_order
        )
        if EVENT_RUN_MIN_DT_CONFIG_KEY not in cfg:
            run_min_dt = get_collection_event_min_dt_if_needed(
                collection,
                include_samples=needs_default_sample_sources,
            )
            if run_min_dt is not None:
                cfg[EVENT_RUN_MIN_DT_CONFIG_KEY] = run_min_dt

        process_bundles: dict[str, ControlSourceBundle] = {}
        process_sample_sources: dict[str, Any] = {}
        process_control_metadata: dict[str, dict[str, Any]] = {}
        reference_categorised: tuple[
            tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]
        ] | None = None

        for process_name in process_order:
            process = collection.processes[process_name]
            bundle = select_control_sources(
                process_name=process_name,
                process=process,
                config=cfg,
            )
            prepared_md = prepared_process_md.get(process_name)
            cls._validate_bundle_against_metadata(
                process_name=process_name,
                bundle=bundle,
                process_md=prepared_md,
            )
            sample_source = cls._sample_source_from_prepared_metadata(
                process_name=process_name,
                process_md=prepared_md,
            )
            if sample_source is None:
                sample_source = build_sample_acc_source_default(
                    process,
                    run_min_dt=run_min_dt_from_runtime_controls(cfg),
                )
            if sample_source.name != BP_TRAIN_SAMPLE_ACC_NAME:
                raise ValueError(
                    f"{process_name}: sample control must be named "
                    f"{BP_TRAIN_SAMPLE_ACC_NAME}"
                )

            categorised = (
                bundle.name_controlled_FVCs,
                bundle.name_controlled_SVCs,
                bundle.name_controlled_PVs,
                bundle.name_extras_bolus,
            )
            if reference_categorised is None:
                reference_categorised = categorised
            elif categorised != reference_categorised:
                raise ValueError(
                    "controls store requires identical categorised control "
                    f"layouts across processes; {process_name!r} has "
                    f"{categorised!r} but expected {reference_categorised!r}"
                )

            process_bundles[process_name] = bundle
            process_sample_sources[process_name] = sample_source
            process_control_metadata[process_name] = {
                source.name: source.metadata
                for source in [*bundle.all_sources, sample_source]
            }

        if reference_categorised is None:
            raise ValueError("process collection is empty")

        (
            name_controlled_FVCs,
            name_controlled_SVCs,
            name_controlled_PVs,
            name_extras_bolus,
        ) = reference_categorised
        # Extras layout: bolus FVCs first (alphabetical), then sample_acc at the very end.
        name_extras = name_extras_bolus + (BP_TRAIN_SAMPLE_ACC_NAME,)
        canonical_names: list[str] = list(
            name_controlled_FVCs
            + name_controlled_SVCs
            + name_controlled_PVs
            + name_extras
        )
        sample_acc_global_index = len(canonical_names) - 1

        spread_inputs = {
            process_name: [
                *process_bundles[process_name].all_sources,
                process_sample_sources[process_name],
            ]
            for process_name in process_order
        }
        spreads = compute_signal_spreads(spread_inputs)

        payloads_by_process: dict[str, dict[str, Any]] = {}
        payload_source_names_by_process: dict[str, list[str]] = {}
        max_grid_length = 0
        max_step_ts_length = 0
        for process_name in process_order:
            process = collection.processes[process_name]
            bundle = process_bundles[process_name]
            sample_source = process_sample_sources[process_name]
            sources = [*bundle.all_sources, sample_source]
            payload_source_names_by_process[process_name] = [
                source.name for source in sources
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
            payload_source_names = payload_source_names_by_process[process_name]
            (
                dense_grid,
                control_values,
                control_derivatives,
                step_ts,
                grid_length,
                step_ts_length,
            ) = cls._pad_payload(
                payload=payload,
                payload_source_names=payload_source_names,
                canonical_names=canonical_names,
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
                "name_controlled_FVCs": list(name_controlled_FVCs),
                "name_controlled_SVCs": list(name_controlled_SVCs),
                "name_controlled_PVs": list(name_controlled_PVs),
                "name_extras": list(name_extras),
                "control_metadata": process_control_metadata[process_name],
                "sample_acc_name": BP_TRAIN_SAMPLE_ACC_NAME,
            }

        shape_metadata = {
            "n_processes": len(process_order),
            "max_grid_length": max_grid_length,
            "max_controls": len(canonical_names),
            "max_step_ts_length": max_step_ts_length,
        }

        return cls(
            process_order=process_order,
            name_controlled_FVCs=name_controlled_FVCs,
            name_controlled_SVCs=name_controlled_SVCs,
            name_controlled_PVs=name_controlled_PVs,
            name_extras=name_extras,
            shape_metadata=shape_metadata,
            dense_grid=_as_jax_array(dense_grid_rows),
            control_values=_as_jax_array(control_value_rows),
            control_derivatives=_as_jax_array(control_derivative_rows),
            step_ts=_as_jax_array(step_ts_rows),
            grid_lengths=jnp.asarray(grid_lengths, dtype=jnp.int32),
            step_ts_lengths=jnp.asarray(step_ts_lengths, dtype=jnp.int32),
            sample_acc_global_index=sample_acc_global_index,
            _process_md_by_name=processes_metadata,
        )

    @classmethod
    def from_json(
        cls,
        prepared_json: str | Path,
    ) -> ControlsStore:
        """Load a prepared JSON artifact and construct a `ControlsStore`."""
        collection = load_process_collection_json(Path(prepared_json))
        return cls.from_collection(collection)

    def get_controls(self, process: str | int) -> PerProcessControls:
        """Return per-process controls by canonical prepared key or index."""
        process_name, process_index = _coerce_index(process, self.process_order)
        process_md = self._process_md_by_name[process_name]
        sample_acc_name = str(process_md["sample_acc_name"])

        return PerProcessControls(
            process_name=process_name,
            process_index=process_index,
            name_controlled_FVCs=self.name_controlled_FVCs,
            name_controlled_SVCs=self.name_controlled_SVCs,
            name_controlled_PVs=self.name_controlled_PVs,
            name_extras=self.name_extras,
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
        # Clamp derivatives in the padded tail to the last active value as well.
        last_derivatives = self.control_derivatives[
            jnp.arange(self.control_derivatives.shape[0], dtype=jnp.int32),
            last_index,
            :,
        ]
        grid_clamped = jnp.where(tail_mask, last_grid[:, None], self.dense_grid)
        values_clamped = jnp.where(
            tail_mask[:, :, None],
            last_values[:, None, :],
            self.control_values,
        )
        derivatives_clamped = jnp.where(
            tail_mask[:, :, None],
            last_derivatives[:, None, :],
            self.control_derivatives,
        )
        return BatchControls(
            dense_grid=grid_clamped,
            control_values=values_clamped,
            control_derivatives=derivatives_clamped,
            name_controlled_FVCs=self.name_controlled_FVCs,
            name_controlled_SVCs=self.name_controlled_SVCs,
            name_controlled_PVs=self.name_controlled_PVs,
            name_extras=self.name_extras,
        )
