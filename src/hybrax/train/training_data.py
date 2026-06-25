from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from bp_format.dataclasses import BioProcessCollection
from bp_format.mechanistic import build_rhs_ode
from bp_format.serialization import load_process_collection_json

from .controls_store import ControlsStore, PerProcessControls


TARGET_SOURCE_AUTO = "auto"
TARGET_SOURCE_PROCESS_VARIABLES = "process_variables"
TARGET_SOURCE_REACTOR_COMPONENTS = "reactor_components"
# Fit BOTH the modeled RMCs and the modeled (uncontrolled) PVs. Targets are
# ordered to match the integrated state's leading block
# ``[name_modeled_RMCs | name_modeled_PVs]`` so ``target_state_indices`` stays a
# simple ``range(n)``; every modeled RMC and PV must carry a time series.
TARGET_SOURCE_COMBINED = "combined"
TARGET_SOURCES = {
    TARGET_SOURCE_AUTO,
    TARGET_SOURCE_PROCESS_VARIABLES,
    TARGET_SOURCE_REACTOR_COMPONENTS,
    TARGET_SOURCE_COMBINED,
}


def _combined_measured_targets(process, rhs_ode) -> tuple[list[str], list[str]]:
    """Measured (RMC, PV) target names for ``TARGET_SOURCE_COMBINED``, each
    ordered to match the integrated state (``rhs_ode.name_modeled_RMCs`` then
    ``rhs_ode.name_modeled_PVs``). Every modeled RMC and PV must have a time
    series — otherwise the ``range(n)`` state-index mapping would be wrong."""
    components = getattr(process.reactor_medium, "components", {}) or {}
    rmc_targets: list[str] = []
    for name in rhs_ode.name_modeled_RMCs:
        component = components.get(name)
        if component is None or not _is_timeseries_compatible(component.concentration):
            raise ValueError(
                f"{process.metadata.name}: target_source='combined' requires every "
                f"modeled RMC to carry a time series; {name!r} does not."
            )
        rmc_targets.append(name)
    pv_targets: list[str] = []
    for name in rhs_ode.name_modeled_PVs:
        variable = process.process_variables.get(name)
        if variable is None or not _is_timeseries_compatible(variable.values):
            raise ValueError(
                f"{process.metadata.name}: target_source='combined' requires every "
                f"modeled PV to carry a time series; {name!r} does not."
            )
        pv_targets.append(name)
    return rmc_targets, pv_targets


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
                f"{process.metadata.name}: configured target variables missing "
                f"from process variables: {missing}"
            )
        controlled = [
            name
            for name in configured_order
            if bool(process.process_variables[name].is_controlled)
        ]
        if controlled:
            raise ValueError(
                f"{process.metadata.name}: configured target variables must be "
                f"measured (is_controlled=False), got controlled targets: "
                f"{controlled}"
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
    # Measured-target names. Exactly one of ``name_measured_RMCs`` /
    # ``name_measured_PVs`` is non-empty depending on which BioProcess group
    # the prepared collection chose to measure.
    name_measured_RMCs: tuple[str, ...]
    name_measured_PVs: tuple[str, ...]
    # Modeled volume-change names (mirrored from store for JIT-friendly
    # per-process access).
    name_modeled_FVCs: tuple[str, ...]
    name_modeled_SVCs: tuple[str, ...]
    # Number of active measurement rows for this process.
    n_measured: int
    # Padded measurement times for this process.
    t_measured: jax.Array
    # Padded measurement values for this process, columns follow
    # ``name_measured`` (whichever of RMCs/PVs is populated) then the
    # cumulative-volume tail for ``name_modeled_FVCs + name_modeled_SVCs``.
    y_measured: jax.Array
    # Padded per-cell boolean mask `[max_n_meas, n_y_cols]`. True iff the
    # corresponding (timestamp, target) pair is a real measurement; False
    # for rows beyond ``n_measured`` and for cells where a target has no
    # measurement at that timestamp on the union grid.
    mask_measured: jax.Array
    # Initial state `[targets..., V_in_cumulative]`, sourced from measurements at t=0.
    y0_measured: jax.Array
    # Per-process controls view from ControlsStore.
    controls: PerProcessControls

    @property
    def active_t_measured(self) -> jax.Array:
        """Active measurement-time prefix."""
        return self.t_measured[: self.n_measured]

    @property
    def active_y_measured(self) -> jax.Array:
        """Active measurement/value prefix."""
        return self.y_measured[: self.n_measured]

    @property
    def active_mask_measured(self) -> jax.Array:
        """Active measurement-mask prefix."""
        return self.mask_measured[: self.n_measured]

    @property
    def name_measured(self) -> tuple[str, ...]:
        """Combined measured-target names in state order
        ``[name_measured_RMCs | name_measured_PVs]``."""
        return tuple(self.name_measured_RMCs) + tuple(self.name_measured_PVs)


class BatchTrainingData(eqx.Module):
    """Batch view over stacked training tensors for process-index batches."""

    # Process indices used to gather this batch view.
    process_indices: jax.Array
    # Gathered measurement times `[batch_size, max_n_meas]`.
    t_measured: jax.Array
    # Gathered measurement values `[batch_size, max_n_meas, n_targets]`.
    y_measured: jax.Array
    # Gathered per-cell measurement masks `[batch_size, max_n_meas, n_y_cols]`.
    mask_measured: jax.Array
    # Gathered active measurement counts `[batch_size]`.
    n_measured: jax.Array
    # Gathered initial state vectors `[batch_size, n_targets + 1]`, sourced
    # from measurements at t=0.
    y0_measured: jax.Array


class TrainingDataStore(eqx.Module):
    """Collection-level training-data store built from a prepared collection.

    The y_measured columns are ``[species..., B_modeled_cum_per_modeled_feed...]``
    (NOT V_in_cumulative — V_in_cumulative is in the ODE *state* but not in the *loss targets*).

    The y0 vector has layout
    ``[species_0..., V_in_cumulative(0), B_modeled_cum_0(0), ...]`` matching the ODE
    state shape that the wrapper expects.
    """

    # Stable process order across all stacked arrays.
    process_order: list[str]
    # Measured-target names, split by which BioProcess group is being
    # measured. Exactly one is non-empty. ``__post_init__`` enforces the
    # invariant; ``name_measured`` (property) returns whichever one carries
    # the names for code that doesn't care about the kind.
    name_measured_RMCs: tuple[str, ...]
    name_measured_PVs: tuple[str, ...]
    # Ordered modeled-FVC names (shared across processes). Each contributes
    # one cumulative-volume column to y_measured.
    name_modeled_FVCs: tuple[str, ...]
    # Ordered modeled-SVC names. Future-proof placeholder; always ``()`` today
    # (the wrapper rejects continuous modeled SVCs). When populated, each name
    # contributes one cumulative-volume column to y_measured after the
    # FVC block, mirroring ``RhsOde.name_modeled_SVCs`` ordering.
    name_modeled_SVCs: tuple[str, ...]
    # Shared controls store for this prepared artifact.
    controls_store: ControlsStore
    # Padded measurement times `[n_processes, max_n_meas]`.
    t_measured: jax.Array
    # Padded measurement values
    # `[n_processes, max_n_meas, n_species + n_modeled_feeds]`.
    y_measured: jax.Array
    # Padded per-cell measurement mask `[n_processes, max_n_meas, n_y_cols]`.
    # True iff the corresponding (timestamp, target) pair is a real
    # measurement. The modeled-feed cumulative columns are dense by
    # construction (mask=True throughout).
    mask_measured: jax.Array
    # Active measurement counts per process.
    n_measured: jax.Array
    # Initial state matrix `[n_processes, n_species + 1 + n_modeled_feeds]`
    # where layout is `[species_0..., V_in_cumulative(0), B_modeled_cum_0(0), ...]`,
    # sourced from measurements at t=0.
    y0_measured: jax.Array

    def __check_init__(self) -> None:
        # At least one of name_measured_RMCs / name_measured_PVs must be
        # populated; ``combined`` sets both. Fail-fast per CLAUDE.md principle 7.
        if not (self.name_measured_RMCs or self.name_measured_PVs):
            raise ValueError(
                "TrainingDataStore: at least one of name_measured_RMCs / "
                "name_measured_PVs must be non-empty."
            )

    @property
    def name_measured(self) -> tuple[str, ...]:
        """Combined measured-target names in state order
        ``[name_measured_RMCs | name_measured_PVs]`` — the loss/target column
        labels and the leading state block that ``target_state_indices`` maps
        onto."""
        return tuple(self.name_measured_RMCs) + tuple(self.name_measured_PVs)

    @classmethod
    def from_collection(
        cls,
        collection: BioProcessCollection,
        *,
        target_variable_order: list[str] | None = None,
        target_source: str = TARGET_SOURCE_PROCESS_VARIABLES,
    ) -> TrainingDataStore:
        """Build training-data tensors from a prepared process collection."""
        controls_store = ControlsStore.from_collection(collection)
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
        # Source family (reactor_components / process_variables) per target column,
        # aligned with ``reference_targets``. ``combined`` mixes both.
        reference_target_sources: list[str] | None = None

        for process_name in process_order:
            process = collection.processes[process_name]
            if resolved_target_source == TARGET_SOURCE_COMBINED:
                rmc_targets, pv_targets = _combined_measured_targets(
                    process, build_rhs_ode(process)
                )
                current_targets = rmc_targets + pv_targets
                current_sources = [TARGET_SOURCE_REACTOR_COMPONENTS] * len(
                    rmc_targets
                ) + [TARGET_SOURCE_PROCESS_VARIABLES] * len(pv_targets)
            else:
                current_targets = _measurement_targets(
                    process,
                    target_order,
                    resolved_target_source,
                )
                current_sources = [resolved_target_source] * len(current_targets)
            per_process_targets[process_name] = current_targets

            if reference_targets is None:
                reference_targets = list(current_targets)
                reference_target_sources = list(current_sources)
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

        # Split the resolved target names into the RMC / PV tuples. For
        # ``combined`` the RMC columns come first (the source list discriminates);
        # the single-source families populate exactly one tuple.
        name_measured_PVs: tuple[str, ...]
        if resolved_target_source == TARGET_SOURCE_COMBINED:
            n_rmc = reference_target_sources.count(TARGET_SOURCE_REACTOR_COMPONENTS)
            name_measured_RMCs = tuple(reference_targets[:n_rmc])
            name_measured_PVs = tuple(reference_targets[n_rmc:])
        elif resolved_target_source == TARGET_SOURCE_REACTOR_COMPONENTS:
            name_measured_RMCs = tuple(reference_targets)
            name_measured_PVs = ()
        elif resolved_target_source == TARGET_SOURCE_PROCESS_VARIABLES:
            name_measured_RMCs = ()
            name_measured_PVs = tuple(reference_targets)
        else:
            raise ValueError(
                f"resolved target_source must be one of "
                f"{TARGET_SOURCE_REACTOR_COMPONENTS!r}, "
                f"{TARGET_SOURCE_PROCESS_VARIABLES!r}, "
                f"{TARGET_SOURCE_COMBINED!r}, got {resolved_target_source!r}"
            )

        ref_process = collection.processes[process_order[0]]
        ref_rhs_ode = build_rhs_ode(ref_process)
        name_modeled_FVCs = tuple(ref_rhs_ode.name_modeled_FVCs)
        name_modeled_SVCs = tuple(ref_rhs_ode.name_modeled_SVCs)
        for _pn in process_order[1:]:
            _other_rhs_ode = build_rhs_ode(collection.processes[_pn])
            if tuple(_other_rhs_ode.name_modeled_FVCs) != name_modeled_FVCs:
                raise ValueError(
                    f"name_modeled_FVCs differs across processes: "
                    f"{process_order[0]!r} has {name_modeled_FVCs!r} but "
                    f"{_pn!r} has {tuple(_other_rhs_ode.name_modeled_FVCs)!r}"
                )
            if tuple(_other_rhs_ode.name_modeled_SVCs) != name_modeled_SVCs:
                raise ValueError(
                    f"name_modeled_SVCs differs across processes: "
                    f"{process_order[0]!r} has {name_modeled_SVCs!r} but "
                    f"{_pn!r} has {tuple(_other_rhs_ode.name_modeled_SVCs)!r}"
                )
        n_modeled = len(name_modeled_FVCs) + len(name_modeled_SVCs)

        per_process_times: list[np.ndarray] = []
        per_process_values: list[np.ndarray] = []
        per_process_masks: list[np.ndarray] = []
        per_process_y0: list[np.ndarray] = []
        n_meas_list: list[int] = []
        max_n_meas = 0
        n_targets = len(reference_targets)
        # y_measured columns = [species targets..., B_modeled_cum per FVC...,
        # B_modeled_cum per SVC...]. SVC columns are absent today (no example
        # uses modeled SVCs and the wrapper rejects them) but the layout is
        # there for future use.
        n_y_cols = n_targets + n_modeled

        for process_name in process_order:
            process = collection.processes[process_name]
            process_targets = per_process_targets[process_name]

            # Per-target (times, values) — each target may have its own grid.
            # ``reference_target_sources`` resolves each column to its source
            # family (combined mixes reactor components and process variables).
            per_target_times: list[np.ndarray] = []
            per_target_values: list[np.ndarray] = []
            for col_source, target_name in zip(
                reference_target_sources, process_targets, strict=True
            ):
                ts, ys = _timeseries_numpy(
                    process,
                    target_name,
                    col_source,
                )
                per_target_times.append(np.asarray(ts, dtype=np.float32))
                per_target_values.append(np.asarray(ys, dtype=np.float32))

            if not per_target_times:
                raise ValueError(f"{process_name}: no measurement data for targets")

            # Union grid across all per-target measurement times.
            union_ts = np.unique(
                np.concatenate(per_target_times).astype(np.float32)
            )
            t0_union = float(union_ts[0])

            # Strict t[0] requirement: every target must have a measurement
            # at union_ts[0]. Otherwise y0 is undefined.
            for tname, t_arr in zip(process_targets, per_target_times, strict=True):
                if t_arr.size == 0 or not np.any(np.isclose(t_arr, t0_union, atol=1e-9)):
                    raise ValueError(
                        f"Process {process_name!r}: target {tname!r} has no "
                        f"measurement at union_grid t[0] = {t0_union:.6g}. "
                        f"Either supply a t={t0_union:.6g} measurement, mark "
                        f"this variable as a StaticVariable, or remove it from "
                        f"target_variable_order."
                    )

            # Build (n_measured, n_y_cols) value + mask matrices on the union grid.
            n_measured = int(union_ts.size)
            y_matrix = np.zeros((n_measured, n_y_cols), dtype=np.float32)
            mask_matrix = np.zeros((n_measured, n_y_cols), dtype=bool)

            for col_idx, (t_arr, v_arr) in enumerate(
                zip(per_target_times, per_target_values, strict=True)
            ):
                # Map each target measurement onto its row in the union grid.
                # np.searchsorted on a sorted union finds exact-match positions.
                positions = np.searchsorted(union_ts, t_arr)
                # Clamp pathological out-of-range positions defensively.
                positions = np.clip(positions, 0, n_measured - 1)
                y_matrix[positions, col_idx] = v_arr.astype(np.float32)
                mask_matrix[positions, col_idx] = True

            # Modeled-VC cumulative columns: dense by construction, fill the
            # value via linear interpolation of the cumulative volume trace
            # and mark mask True throughout. Layout matches y_measured's
            # column order: FVCs first, then SVCs.
            v0 = float(process.volume.initial_volume)
            for k, fn in enumerate(name_modeled_FVCs + name_modeled_SVCs):
                col_idx = n_targets + k
                vc = process.volume.volume_changes[fn]
                vc_t = np.asarray(vc.values.times, dtype=float)
                vc_v = np.asarray(vc.values.values, dtype=float)
                b_col = np.interp(
                    union_ts.astype(float),
                    vc_t,
                    vc_v,
                    left=float(vc_v[0]),
                    right=float(vc_v[-1]),
                ).astype(np.float32)
                y_matrix[:, col_idx] = b_col
                mask_matrix[:, col_idx] = True

            # y0 = [species(0)..., V_in_cumulative(0)=v0, B_modeled_cum_k(0)=0...]
            # Strict t[0] check above guarantees y_matrix[0, :n_targets] are
            # all real measurements.
            y0_species = y_matrix[0, :n_targets]
            y0 = np.concatenate(
                [
                    y0_species,
                    np.asarray([v0], dtype=np.float32),
                    np.zeros(n_modeled, dtype=np.float32),
                ],
                axis=0,
            )

            max_n_meas = max(max_n_meas, n_measured)
            per_process_times.append(union_ts)
            per_process_values.append(y_matrix)
            per_process_masks.append(mask_matrix)
            per_process_y0.append(y0)
            n_meas_list.append(n_measured)

        n_processes = len(process_order)
        t_measured = np.zeros((n_processes, max_n_meas), dtype=np.float32)
        y_measured = np.zeros((n_processes, max_n_meas, n_y_cols), dtype=np.float32)
        mask_measured = np.zeros((n_processes, max_n_meas, n_y_cols), dtype=bool)

        for index, (ts, ys, mk) in enumerate(
            zip(
                per_process_times,
                per_process_values,
                per_process_masks,
                strict=True,
            )
        ):
            n_measured = n_meas_list[index]
            t_measured[index, :n_measured] = ts.astype(np.float32)
            y_measured[index, :n_measured, :] = ys.astype(np.float32)
            mask_measured[index, :n_measured, :] = mk

        return cls(
            process_order=process_order,
            name_measured_RMCs=name_measured_RMCs,
            name_measured_PVs=name_measured_PVs,
            name_modeled_FVCs=name_modeled_FVCs,
            name_modeled_SVCs=name_modeled_SVCs,
            controls_store=controls_store,
            t_measured=jnp.asarray(t_measured),
            y_measured=jnp.asarray(y_measured),
            mask_measured=jnp.asarray(mask_measured),
            n_measured=jnp.asarray(n_meas_list, dtype=jnp.int32),
            y0_measured=jnp.asarray(np.asarray(per_process_y0, dtype=np.float32)),
        )

    @classmethod
    def from_json(
        cls,
        prepared_json: str | Path,
        *,
        target_variable_order: list[str] | None = None,
        target_source: str = TARGET_SOURCE_PROCESS_VARIABLES,
    ) -> TrainingDataStore:
        """Load a prepared JSON artifact and construct a training-data store."""
        collection = load_process_collection_json(Path(prepared_json))
        return cls.from_collection(
            collection,
            target_variable_order=target_variable_order,
            target_source=target_source,
        )

    def get_process(self, process: str | int) -> PerProcessTrainingData:
        """Return per-process training data by canonical name or integer index."""
        process_name, process_index = _coerce_process_index(process, self.process_order)
        return PerProcessTrainingData(
            process_name=process_name,
            process_index=process_index,
            name_measured_RMCs=self.name_measured_RMCs,
            name_measured_PVs=self.name_measured_PVs,
            name_modeled_FVCs=self.name_modeled_FVCs,
            name_modeled_SVCs=self.name_modeled_SVCs,
            n_measured=int(self.n_measured[process_index]),
            t_measured=self.t_measured[process_index],
            y_measured=self.y_measured[process_index],
            mask_measured=self.mask_measured[process_index],
            y0_measured=self.y0_measured[process_index],
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
            t_measured=self.t_measured[indices],
            y_measured=self.y_measured[indices],
            mask_measured=self.mask_measured[indices],
            n_measured=self.n_measured[indices],
            y0_measured=self.y0_measured[indices],
        )
