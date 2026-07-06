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
from bp_format.mechanistic import build_rhs_ode
from bp_format.serialization import load_process_collection

from .constants import METADATA_NAMESPACE
from .controls import (
    ControlSourceBundle,
    build_dense_payload,
    collect_discrete_event_metadata,
    compute_signal_spreads,
    select_control_sources,
)


DEFAULT_RUNTIME_CONTROLS_CONFIG: dict[str, Any] = {
    "initial_grid_points": 16,
    "max_rel_error": 1e-4,
    "max_refinement_rounds": 8,
}


def _as_jax_array(values: Any, *, dtype: Any = jnp.float64) -> jax.Array:
    """Convert JSON-loaded values into a JAX array."""
    return jnp.asarray(values, dtype=dtype)


def _discrete_event_jump_ts(process: Any) -> list[float]:
    """Sorted unique vector-field discontinuity times from ``discrete_events``.

    These are genuine jumps in the controls/vector field (e.g. discrete steps in
    controlled process variables) wired to ``PIDController(jump_ts=...)`` — NOT
    the bolus/sample state-jump events, which the callbacks solve handles via its
    own ``*_event_*`` arrays. Empty when the process declares no discrete events.
    """
    de = process.discrete_events
    if de is None or de.times is None:
        return []
    return sorted({float(t) for t in np.asarray(de.times).reshape(-1).tolist()})


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
    """Per-process runtime view over padded, canonical-axis dense-grid controls.

    Column axis follows
    ``[name_controlled_FVCs | name_controlled_SVCs | name_controlled_PVs]``
    matching bp-format ``ControlSplines``. All columns are consumed by
    ``eval_u(t)`` to build RhsOde's ``u`` argument. Discrete bolus/sample events
    are NOT controls here — they are applied as state jumps by the callbacks
    solve from the ``*_event_*`` arrays. ``jump_ts`` carries genuine vector-field
    discontinuity times (``BioProcess.discrete_events``) for the adaptive solver.

    All non-array fields are ``eqx.field(static=True)`` so they live in
    the pytree treedef rather than as dynamic leaves.
    """

    process_name: str = eqx.field(static=True)
    process_index: int = eqx.field(static=True)
    name_controlled_FVCs: tuple[str, ...] = eqx.field(static=True)
    name_controlled_SVCs: tuple[str, ...] = eqx.field(static=True)
    name_controlled_PVs: tuple[str, ...] = eqx.field(static=True)
    dense_grid: jax.Array
    control_values: jax.Array
    control_derivatives: jax.Array
    jump_ts: jax.Array
    grid_length: int = eqx.field(static=True)
    jump_ts_length: int = eqx.field(static=True)
    control_metadata: dict[str, dict[str, Any]] = eqx.field(static=True)
    sample_event_times: jax.Array
    sample_event_volumes: jax.Array
    sample_event_mask: jax.Array
    bolus_event_times: jax.Array
    bolus_event_volumes: jax.Array
    bolus_event_Cin: jax.Array
    bolus_event_mask: jax.Array

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
    def active_jump_ts(self) -> jax.Array:
        return self.jump_ts[: self.jump_ts_length]

    @property
    def active_control_values(self) -> jax.Array:
        return self.control_values[: self.grid_length]

    @property
    def active_control_derivatives(self) -> jax.Array:
        return self.control_derivatives[: self.grid_length]

    def _eval_values(self, ts: float | np.ndarray | jax.Array) -> jax.Array:
        """Interpolate all control VALUES at one or more times, in canonical
        column order ``[FVCs_cum | SVCs_cum | PVs]``. Private — public access is
        via the per-axis ``eval_controlled_*`` accessors."""
        query = jnp.asarray(ts, dtype=self.dense_grid.dtype)
        scalar_input = query.ndim == 0
        query_1d = jnp.atleast_1d(query)
        values = _interp_columns(
            query_1d,
            self.active_dense_grid,
            self.active_control_values,
        )
        return values[0] if scalar_input else values

    def _eval_derivatives(self, ts: float | np.ndarray | jax.Array) -> jax.Array:
        """Interpolate all control DERIVATIVES (flow rates) in canonical order.
        Private — sliced by the per-axis ``eval_controlled_*_rates`` accessors."""
        query = jnp.asarray(ts, dtype=self.dense_grid.dtype)
        scalar_input = query.ndim == 0
        query_1d = jnp.atleast_1d(query)
        values = _interp_columns(
            query_1d,
            self.active_dense_grid,
            self.active_control_derivatives,
        )
        return values[0] if scalar_input else values

    # ------------------------------------------------------------------
    # Semantic, non-overlapping per-axis accessors. Each returns RAW
    # (physical, unscaled) values for a single control axis. ``states`` is a
    # placeholder for future state-dependent controls (e.g. pH feedback) and
    # is currently unused. The wrapper scales each result to SCL space via the
    # module's ``scale_controlled_*`` helpers before building ReactionInputs.
    # ------------------------------------------------------------------
    def eval_controlled_FVCs_cumulative(self, t_arr, states) -> jax.Array:
        n_fvc = len(self.name_controlled_FVCs)
        return self._eval_values(t_arr)[..., :n_fvc]

    def eval_controlled_FVCs_rates(self, t_arr, states) -> jax.Array:
        n_fvc = len(self.name_controlled_FVCs)
        return self._eval_derivatives(t_arr)[..., :n_fvc]

    def eval_controlled_SVCs_rates(self, t_arr, states) -> jax.Array:
        n_fvc = len(self.name_controlled_FVCs)
        n_svc = len(self.name_controlled_SVCs)
        return self._eval_derivatives(t_arr)[..., n_fvc : n_fvc + n_svc]

    def eval_controlled_PVs(self, t_arr, states) -> jax.Array:
        n_fvc = len(self.name_controlled_FVCs)
        n_svc = len(self.name_controlled_SVCs)
        n_pv = len(self.name_controlled_PVs)
        return self._eval_values(t_arr)[..., n_fvc + n_svc : n_fvc + n_svc + n_pv]


class BatchControls(eqx.Module):
    """All-process controls evaluator with index-based runtime lookup.

    Column axis follows the same canonical
    ``[name_controlled_FVCs | name_controlled_SVCs | name_controlled_PVs]``
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
    sample_event_times: jax.Array
    sample_event_volumes: jax.Array
    sample_event_mask: jax.Array
    bolus_event_times: jax.Array
    bolus_event_volumes: jax.Array
    bolus_event_Cin: jax.Array
    bolus_event_mask: jax.Array

    def _eval_values(self, process_idx: int, t: jax.Array) -> jax.Array:
        """Interpolate all control VALUES for one process index, canonical order.
        Private — sliced by the per-axis ``eval_controlled_*`` accessors."""
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
        return out[0] if scalar_input else out

    def _eval_derivatives(self, process_idx: int, t: jax.Array) -> jax.Array:
        """Interpolate all control DERIVATIVES for one process index, canonical
        order. Private — sliced by the ``eval_controlled_*_rates`` accessors."""
        grid = self.dense_grid[process_idx]
        derivatives = self.control_derivatives[process_idx]

        query = jnp.asarray(t, dtype=grid.dtype)
        scalar_input = query.ndim == 0
        query_1d = jnp.atleast_1d(query)
        out = _interp_columns(query_1d, grid, derivatives)
        return out[0] if scalar_input else out

    # Semantic, non-overlapping per-axis accessors (RAW values). ``states`` is a
    # placeholder for future state-dependent controls and is currently unused.
    def eval_controlled_FVCs_cumulative(self, process_idx, t_arr, states) -> jax.Array:
        n_fvc = len(self.name_controlled_FVCs)
        return self._eval_values(process_idx, t_arr)[..., :n_fvc]

    def eval_controlled_FVCs_rates(self, process_idx, t_arr, states) -> jax.Array:
        n_fvc = len(self.name_controlled_FVCs)
        return self._eval_derivatives(process_idx, t_arr)[..., :n_fvc]

    def eval_controlled_SVCs_rates(self, process_idx, t_arr, states) -> jax.Array:
        n_fvc = len(self.name_controlled_FVCs)
        n_svc = len(self.name_controlled_SVCs)
        return self._eval_derivatives(process_idx, t_arr)[..., n_fvc : n_fvc + n_svc]

    def eval_controlled_PVs(self, process_idx, t_arr, states) -> jax.Array:
        n_fvc = len(self.name_controlled_FVCs)
        n_svc = len(self.name_controlled_SVCs)
        n_pv = len(self.name_controlled_PVs)
        return self._eval_values(process_idx, t_arr)[
            ..., n_fvc + n_svc : n_fvc + n_svc + n_pv
        ]


class ControlsStore(eqx.Module):
    """Collection-level loader and index for prepared, padded JAX control tensors.

    Column axis follows
    ``[name_controlled_FVCs | name_controlled_SVCs | name_controlled_PVs]``
    consistently across every process; the wrapper consumes the full
    u-block via :meth:`PerProcessControls.eval_u`.
    """

    # Canonical prepared process keys in stable collection order.
    process_order: list[str]
    # Categorised name tuples (must be identical across processes).
    name_controlled_FVCs: tuple[str, ...]
    name_controlled_SVCs: tuple[str, ...]
    name_controlled_PVs: tuple[str, ...]
    # Shape metadata persisted in `prepared.json`.
    shape_metadata: dict[str, Any]
    # Stacked padded dense grids, shape `[n_processes, max_grid_length]`.
    dense_grid: jax.Array
    # Stacked padded control values, shape
    # `[n_processes, max_grid_length, max_controls]`.
    control_values: jax.Array
    # Stacked padded control derivatives, same shape as `control_values`.
    control_derivatives: jax.Array
    # Stacked padded jump-time arrays, shape `[n_processes, max_jump_ts_length]`.
    jump_ts: jax.Array
    # Active dense-grid lengths per process.
    grid_lengths: jax.Array
    # Active `jump_ts` lengths per process.
    jump_ts_lengths: jax.Array
    sample_event_times: jax.Array
    sample_event_volumes: jax.Array
    sample_event_mask: jax.Array
    bolus_event_times: jax.Array
    bolus_event_volumes: jax.Array
    bolus_event_Cin: jax.Array
    bolus_event_mask: jax.Array
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
        for key, expected in (
            ("name_controlled_FVCs", bundle.name_controlled_FVCs),
            ("name_controlled_SVCs", bundle.name_controlled_SVCs),
            ("name_controlled_PVs", bundle.name_controlled_PVs),
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
    def _pad_payload(
        *,
        payload: dict[str, Any],
        payload_source_names: list[str],
        canonical_names: list[str],
        max_grid_length: int,
        max_jump_ts_length: int,
    ) -> tuple[
        list[float], list[list[float]], list[list[float]], list[float], int, int
    ]:
        """Reorder payload columns from build order to canonical order, then pad."""
        grid = list(payload["grid"])
        values = [list(row) for row in payload["values"]]
        derivatives = [list(row) for row in payload["derivatives"]]
        jump_ts = list(payload["jump_ts"])

        grid_length = len(grid)
        jump_ts_length = len(jump_ts)
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

        jump_ts_padded = jump_ts + [0.0] * (max_jump_ts_length - jump_ts_length)
        return (
            dense_grid,
            control_values,
            control_derivatives,
            jump_ts_padded,
            grid_length,
            jump_ts_length,
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

        process_bundles: dict[str, ControlSourceBundle] = {}
        process_control_metadata: dict[str, dict[str, Any]] = {}
        reference_categorised: (
            tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None
        ) = None

        for process_name in process_order:
            process = collection.processes[process_name]
            bundle = select_control_sources(process)
            prepared_md = prepared_process_md.get(process_name)
            cls._validate_bundle_against_metadata(
                process_name=process_name,
                bundle=bundle,
                process_md=prepared_md,
            )

            categorised = (
                bundle.name_controlled_FVCs,
                bundle.name_controlled_SVCs,
                bundle.name_controlled_PVs,
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
            process_control_metadata[process_name] = {
                source.name: source.metadata for source in bundle.all_sources
            }

        if reference_categorised is None:
            raise ValueError("process collection is empty")

        (
            name_controlled_FVCs,
            name_controlled_SVCs,
            name_controlled_PVs,
        ) = reference_categorised
        canonical_names: list[str] = list(
            name_controlled_FVCs + name_controlled_SVCs + name_controlled_PVs
        )

        spread_inputs = {
            process_name: list(process_bundles[process_name].all_sources)
            for process_name in process_order
        }
        spreads = compute_signal_spreads(spread_inputs)

        event_metadata_by_process: dict[str, dict[str, Any]] = {}
        reference_species: tuple[str, ...] | None = None
        max_sample_events = 0
        max_bolus_events = 0
        for process_name in process_order:
            process = collection.processes[process_name]
            species_names = tuple(build_rhs_ode(process).name_modeled_RMCs)
            if reference_species is None:
                reference_species = species_names
            elif species_names != reference_species:
                raise ValueError(
                    "controls store requires identical modeled RMC layout across "
                    f"processes for event metadata; {process_name!r} has "
                    f"{species_names!r} but expected {reference_species!r}"
                )
            event_md = collect_discrete_event_metadata(process, species_names)
            event_metadata_by_process[process_name] = event_md
            max_sample_events = max(max_sample_events, len(event_md["sample_times"]))
            max_bolus_events = max(max_bolus_events, len(event_md["bolus_times"]))

        payloads_by_process: dict[str, dict[str, Any]] = {}
        payload_source_names_by_process: dict[str, list[str]] = {}
        max_grid_length = 0
        max_jump_ts_length = 0
        for process_name in process_order:
            process = collection.processes[process_name]
            bundle = process_bundles[process_name]
            sources = list(bundle.all_sources)
            payload_source_names_by_process[process_name] = [
                source.name for source in sources
            ]
            payload = build_dense_payload(
                process=process,
                sources=sources,
                spreads=spreads,
                config=cfg,
            )
            # jump_ts = genuine vector-field discontinuity times from
            # ``BioProcess.discrete_events`` (e.g. discrete steps in controlled
            # process variables). State-jump events (bolus/sample) are NOT here —
            # the callbacks solve already segments the solve at them. Passed to
            # ``diffrax.PIDController(jump_ts=...)`` so the adaptive controller
            # re-inits its step across the discontinuity.
            payload["jump_ts"] = _discrete_event_jump_ts(process)
            payloads_by_process[process_name] = payload
            max_grid_length = max(max_grid_length, len(payload["grid"]))
            max_jump_ts_length = max(max_jump_ts_length, len(payload["jump_ts"]))

        dense_grid_rows = []
        control_value_rows = []
        control_derivative_rows = []
        jump_ts_rows = []
        grid_lengths = []
        jump_ts_lengths = []
        processes_metadata: dict[str, dict[str, Any]] = {}
        sample_event_time_rows = []
        sample_event_volume_rows = []
        sample_event_mask_rows = []
        bolus_event_time_rows = []
        bolus_event_volume_rows = []
        bolus_event_Cin_rows = []
        bolus_event_mask_rows = []
        n_species = 0 if reference_species is None else len(reference_species)

        for process_name in process_order:
            payload = payloads_by_process[process_name]
            payload_source_names = payload_source_names_by_process[process_name]
            (
                dense_grid,
                control_values,
                control_derivatives,
                jump_ts,
                grid_length,
                jump_ts_length,
            ) = cls._pad_payload(
                payload=payload,
                payload_source_names=payload_source_names,
                canonical_names=canonical_names,
                max_grid_length=max_grid_length,
                max_jump_ts_length=max_jump_ts_length,
            )

            dense_grid_rows.append(dense_grid)
            control_value_rows.append(control_values)
            control_derivative_rows.append(control_derivatives)
            jump_ts_rows.append(jump_ts)
            grid_lengths.append(grid_length)
            jump_ts_lengths.append(jump_ts_length)
            event_md = event_metadata_by_process[process_name]
            n_samples = len(event_md["sample_times"])
            n_bolus = len(event_md["bolus_times"])
            sample_event_time_rows.append(
                event_md["sample_times"] + [0.0] * (max_sample_events - n_samples)
            )
            sample_event_volume_rows.append(
                event_md["sample_volumes"] + [0.0] * (max_sample_events - n_samples)
            )
            sample_event_mask_rows.append(
                [True] * n_samples + [False] * (max_sample_events - n_samples)
            )
            bolus_event_time_rows.append(
                event_md["bolus_times"] + [0.0] * (max_bolus_events - n_bolus)
            )
            bolus_event_volume_rows.append(
                event_md["bolus_volumes"] + [0.0] * (max_bolus_events - n_bolus)
            )
            bolus_event_Cin_rows.append(
                event_md["bolus_Cin"]
                + [[0.0] * n_species for _ in range(max_bolus_events - n_bolus)]
            )
            bolus_event_mask_rows.append(
                [True] * n_bolus + [False] * (max_bolus_events - n_bolus)
            )

            processes_metadata[process_name] = {
                "name_controlled_FVCs": list(name_controlled_FVCs),
                "name_controlled_SVCs": list(name_controlled_SVCs),
                "name_controlled_PVs": list(name_controlled_PVs),
                "control_metadata": process_control_metadata[process_name],
            }

        shape_metadata = {
            "n_processes": len(process_order),
            "max_grid_length": max_grid_length,
            "max_controls": len(canonical_names),
            "max_jump_ts_length": max_jump_ts_length,
            "max_sample_events": max_sample_events,
            "max_bolus_events": max_bolus_events,
        }

        return cls(
            process_order=process_order,
            name_controlled_FVCs=name_controlled_FVCs,
            name_controlled_SVCs=name_controlled_SVCs,
            name_controlled_PVs=name_controlled_PVs,
            shape_metadata=shape_metadata,
            dense_grid=_as_jax_array(dense_grid_rows),
            control_values=_as_jax_array(control_value_rows),
            control_derivatives=_as_jax_array(control_derivative_rows),
            jump_ts=_as_jax_array(jump_ts_rows),
            grid_lengths=jnp.asarray(grid_lengths, dtype=jnp.int32),
            jump_ts_lengths=jnp.asarray(jump_ts_lengths, dtype=jnp.int32),
            sample_event_times=_as_jax_array(sample_event_time_rows),
            sample_event_volumes=_as_jax_array(sample_event_volume_rows),
            sample_event_mask=jnp.asarray(sample_event_mask_rows, dtype=bool),
            bolus_event_times=_as_jax_array(bolus_event_time_rows),
            bolus_event_volumes=_as_jax_array(bolus_event_volume_rows),
            bolus_event_Cin=(
                jnp.zeros(
                    (len(process_order), 0, n_species),
                    dtype=jnp.float64,
                )
                if max_bolus_events == 0
                else _as_jax_array(bolus_event_Cin_rows)
            ),
            bolus_event_mask=jnp.asarray(bolus_event_mask_rows, dtype=bool),
            _process_md_by_name=processes_metadata,
        )

    @classmethod
    def from_json(
        cls,
        prepared_json: str | Path,
    ) -> ControlsStore:
        """Load a prepared JSON artifact and construct a `ControlsStore`."""
        collection = load_process_collection(Path(prepared_json))
        return cls.from_collection(collection)

    def get_controls(self, process: str | int) -> PerProcessControls:
        """Return per-process controls by canonical prepared key or index."""
        process_name, process_index = _coerce_index(process, self.process_order)
        process_md = self._process_md_by_name[process_name]

        return PerProcessControls(
            process_name=process_name,
            process_index=process_index,
            name_controlled_FVCs=self.name_controlled_FVCs,
            name_controlled_SVCs=self.name_controlled_SVCs,
            name_controlled_PVs=self.name_controlled_PVs,
            dense_grid=self.dense_grid[process_index],
            control_values=self.control_values[process_index],
            control_derivatives=self.control_derivatives[process_index],
            jump_ts=self.jump_ts[process_index],
            grid_length=int(self.grid_lengths[process_index]),
            jump_ts_length=int(self.jump_ts_lengths[process_index]),
            control_metadata=dict(process_md["control_metadata"]),
            sample_event_times=self.sample_event_times[process_index],
            sample_event_volumes=self.sample_event_volumes[process_index],
            sample_event_mask=self.sample_event_mask[process_index],
            bolus_event_times=self.bolus_event_times[process_index],
            bolus_event_volumes=self.bolus_event_volumes[process_index],
            bolus_event_Cin=self.bolus_event_Cin[process_index],
            bolus_event_mask=self.bolus_event_mask[process_index],
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
            sample_event_times=self.sample_event_times,
            sample_event_volumes=self.sample_event_volumes,
            sample_event_mask=self.sample_event_mask,
            bolus_event_times=self.bolus_event_times,
            bolus_event_volumes=self.bolus_event_volumes,
            bolus_event_Cin=self.bolus_event_Cin,
            bolus_event_mask=self.bolus_event_mask,
        )
