from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from bp_format.dataclasses import BioProcessCollection
from bp_format.mechanistic import RhsOde, build_rhs_ode
from bp_format.serialization import load_process_collection
from bp_format.validate import validate_biological_ode_equivalence

from .controls_store import BatchControls, ControlsStore, PerProcessControls
from .wrapper import validate_rhs_ode_compatibility


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


def _target_values(process, target_name: str, target_source: str):
    """Raw variable object for one target, dispatched by source group."""
    if target_source == TARGET_SOURCE_PROCESS_VARIABLES:
        return process.process_variables[target_name].values
    if target_source == TARGET_SOURCE_REACTOR_COMPONENTS:
        return process.reactor_medium.components[target_name].concentration
    raise ValueError(f"unsupported target_source: {target_source!r}")


def _initial_value_numpy(
    process,
    target_name: str,
    target_source: str,
    t0: float,
) -> float:
    values = _target_values(process, target_name, target_source)
    if hasattr(values, "times") and hasattr(values, "values"):
        ts = np.asarray(values.times, dtype=float)
        ys = np.asarray(values.values, dtype=float)
        matches = np.flatnonzero(np.isclose(ts, t0, atol=1e-9))
        if matches.size == 0:
            raise ValueError(
                f"{process.metadata.name}: state {target_name!r} has no "
                f"initial value at t={t0:.6g}"
            )
        return float(ys[int(matches[0])])
    if hasattr(values, "value"):
        return float(values.value)
    raise ValueError(
        f"{process.metadata.name}: state {target_name!r} must be a time-series "
        "or static variable"
    )


def _timeseries_numpy(
    process,
    target_name: str,
    target_source: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract measurement time/value arrays from one target source."""
    values = _target_values(process, target_name, target_source)
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


def replace_rhs_ode_process_matrices(
    rhs_ode: RhsOde,
    Cin_controlled_Inflows: jax.Array,
    Cin_modeled_Inflows: jax.Array,
    retention_controlled_Outflows: jax.Array,
    retention_modeled_Outflows: jax.Array,
) -> RhsOde:
    """Replace all process-specific physical matrices on an ODE template."""
    return eqx.tree_at(
        lambda rhs: (
            rhs.Cin_controlled_Inflows,
            rhs.Cin_modeled_Inflows,
            rhs.retention_controlled_Outflows,
            rhs.retention_modeled_Outflows,
        ),
        rhs_ode,
        (
            Cin_controlled_Inflows,
            Cin_modeled_Inflows,
            retention_controlled_Outflows,
            retention_modeled_Outflows,
        ),
    )


class PerProcessTrainingData(eqx.Module):
    """Per-process active view over padded training-data tensors."""

    # Canonical process key from prepared metadata.
    process_name: str
    # Integer row index into collection-level stacked arrays.
    process_index: int
    # Measured-target names split by BioProcess group. Either or both may be
    # populated depending on target_source.
    name_measured_RMCs: tuple[str, ...]
    name_measured_PVs: tuple[str, ...]
    # Modeled volume-change names (mirrored from store for JIT-friendly
    # per-process access).
    name_modeled_Inflows: tuple[str, ...]
    name_modeled_Outflows: tuple[str, ...]
    # Number of active measurement rows for this process.
    n_measured: int
    # Padded measurement times for this process.
    t_measured: jax.Array
    # Padded measurement values for this process, columns follow
    # ``name_measured`` (RMC targets, PV targets, or both) then the cumulative
    # volume tail for ``name_modeled_Inflows + name_modeled_Outflows``.
    y_measured: jax.Array
    # Padded per-cell boolean mask `[max_n_meas, n_y_cols]`. True iff the
    # corresponding (timestamp, target) pair is a real measurement; False
    # for rows beyond ``n_measured`` and for cells where a target has no
    # measurement at that timestamp on the union grid.
    mask_measured: jax.Array
    # Full physical initial state `[all_RMCs..., all_PVs..., V, modeled_cum...]`.
    y0_measured: jax.Array
    # This process's flow matrices. A wrapper's baked matrices belong to
    # whichever process supplied its template, so every solve substitutes all
    # four alongside ``controls``.
    Cin_controlled_Inflows: jax.Array
    Cin_modeled_Inflows: jax.Array
    retention_controlled_Outflows: jax.Array
    retention_modeled_Outflows: jax.Array
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
    """Aligned measurement and control rows for a process-index batch."""

    # Global process indices used to gather this batch view.
    process_indices: jax.Array
    # Controls gathered in the same row order as the measurements.
    controls: BatchControls
    # Gathered measurement times `[batch_size, max_n_meas]`.
    t_measured: jax.Array
    # Gathered measurement values `[batch_size, max_n_meas, n_y_cols]`.
    y_measured: jax.Array
    # Gathered per-cell measurement masks `[batch_size, max_n_meas, n_y_cols]`.
    mask_measured: jax.Array
    # Gathered active measurement counts `[batch_size]`.
    n_measured: jax.Array
    # Gathered full physical initial state vectors.
    y0_measured: jax.Array
    # Process-aligned Outflow retention matrices.
    retention_controlled_Outflows: jax.Array
    retention_modeled_Outflows: jax.Array


class TrainingDataStore(eqx.Module):
    """Collection-level training-data store built from a prepared collection.

    The y_measured columns are ``[targets..., B_modeled_cum_per_modeled_feed...]``
    where targets may be RMCs, PVs, or both. V is in the ODE state but not in
    the loss targets.

    The y0 vector has layout
    ``[all_RMCs_0..., all_PVs_0..., V(0), B_modeled_cum_0(0), ...]`` matching
    the physical ODE state shape that the wrapper expects.
    """

    # Stable process order across all stacked arrays.
    process_order: list[str]
    # Measured-target names split by BioProcess group. Either or both may be
    # non-empty; ``name_measured`` returns the loss/target column labels.
    name_measured_RMCs: tuple[str, ...]
    name_measured_PVs: tuple[str, ...]
    # Ordered modeled-Inflow names (shared across processes). Each contributes
    # one cumulative-volume column to y_measured.
    name_modeled_Inflows: tuple[str, ...]
    # Ordered modeled-Outflow names. Each contributes one cumulative-volume
    # column to y_measured after the Inflow block, mirroring
    # ``RhsOde.name_modeled_Outflows`` ordering.
    name_modeled_Outflows: tuple[str, ...]
    # Shared controls store for this prepared artifact.
    controls_store: ControlsStore
    # Canonical ODE structure plus process-aligned flow matrices.
    rhs_ode: RhsOde
    Cin_controlled_Inflows: jax.Array
    Cin_modeled_Inflows: jax.Array
    retention_controlled_Outflows: jax.Array
    retention_modeled_Outflows: jax.Array
    # Padded measurement times `[n_processes, max_n_meas]`.
    t_measured: jax.Array
    # Padded measurement values
    # `[n_processes, max_n_meas, n_targets + n_modeled_flows]`.
    y_measured: jax.Array
    # Padded per-cell measurement mask `[n_processes, max_n_meas, n_y_cols]`.
    # True iff the corresponding (timestamp, target) pair is a real
    # measurement. The modeled-flow cumulative columns are dense by
    # construction (mask=True throughout).
    mask_measured: jax.Array
    # Active measurement counts per process.
    n_measured: jax.Array
    # Initial state matrix `[n_processes, n_RMC + n_PV + 1 + n_modeled_flows]`
    # where layout is `[all_RMCs_0..., all_PVs_0..., V(0), B_modeled_cum_0(0), ...]`.
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
        """Loss/target column labels in ``[RMC targets | PV targets]`` order."""
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
        if not process_order:
            raise ValueError("process collection is empty")
        rhs_odes = [build_rhs_ode(collection.processes[name]) for name in process_order]
        ref_rhs_ode = rhs_odes[0]

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

        for process_name, rhs_ode in zip(process_order, rhs_odes, strict=True):
            process = collection.processes[process_name]
            if resolved_target_source == TARGET_SOURCE_COMBINED:
                rmc_targets, pv_targets = _combined_measured_targets(process, rhs_ode)
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

        biological_ode_ok, biological_ode_message = validate_biological_ode_equivalence(
            collection
        )
        if not biological_ode_ok:
            raise ValueError(biological_ode_message)

        for process_name, rhs_ode in zip(process_order[1:], rhs_odes[1:], strict=True):
            validate_rhs_ode_compatibility(
                process_order[0], ref_rhs_ode, process_name, rhs_ode
            )

        if reference_targets is None:
            raise AssertionError("non-empty process order produced no targets")
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

        name_modeled_Inflows = tuple(ref_rhs_ode.name_modeled_Inflows)
        name_modeled_Outflows = tuple(ref_rhs_ode.name_modeled_Outflows)
        n_modeled = len(name_modeled_Inflows) + len(name_modeled_Outflows)

        per_process_times: list[np.ndarray] = []
        per_process_values: list[np.ndarray] = []
        per_process_masks: list[np.ndarray] = []
        per_process_y0: list[np.ndarray] = []
        n_meas_list: list[int] = []
        max_n_meas = 0
        n_targets = len(reference_targets)
        # y_measured columns = [species targets..., B_modeled_cum per Inflow...,
        # B_modeled_cum per Outflow...].
        n_y_cols = n_targets + n_modeled

        for process_name, process_rhs_ode in zip(process_order, rhs_odes, strict=True):
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
                per_target_times.append(np.asarray(ts, dtype=np.float64))
                per_target_values.append(np.asarray(ys, dtype=np.float64))

            if not per_target_times:
                raise ValueError(f"{process_name}: no measurement data for targets")

            # Union grid across all per-target measurement times.
            union_ts = np.unique(np.concatenate(per_target_times).astype(np.float64))
            t0_union = float(union_ts[0])

            # Strict t[0] requirement: every target must have a measurement
            # at union_ts[0]. Otherwise y0 is undefined.
            for tname, t_arr in zip(process_targets, per_target_times, strict=True):
                if t_arr.size == 0 or not np.any(
                    np.isclose(t_arr, t0_union, atol=1e-9)
                ):
                    raise ValueError(
                        f"Process {process_name!r}: target {tname!r} has no "
                        f"measurement at union_grid t[0] = {t0_union:.6g}. "
                        f"Either supply a t={t0_union:.6g} measurement, mark "
                        f"this variable as a StaticVariable, or remove it from "
                        f"target_variable_order."
                    )

            # Build (n_measured, n_y_cols) value + mask matrices on the union grid.
            n_measured = int(union_ts.size)
            y_matrix = np.zeros((n_measured, n_y_cols), dtype=np.float64)
            mask_matrix = np.zeros((n_measured, n_y_cols), dtype=bool)

            for col_idx, (t_arr, v_arr) in enumerate(
                zip(per_target_times, per_target_values, strict=True)
            ):
                # Map each target measurement onto its row in the union grid.
                # np.searchsorted on a sorted union finds exact-match positions.
                positions = np.searchsorted(union_ts, t_arr)
                # Clamp pathological out-of-range positions defensively.
                positions = np.clip(positions, 0, n_measured - 1)
                y_matrix[positions, col_idx] = v_arr.astype(np.float64)
                mask_matrix[positions, col_idx] = True

            # Modeled-VC cumulative columns: dense by construction, fill the
            # value via linear interpolation of the cumulative volume trace
            # and mark mask True throughout. Layout matches y_measured's
            # column order: Inflows first, then Outflows.
            v0 = float(process.volume.initial_volume)
            for k, fn in enumerate(name_modeled_Inflows + name_modeled_Outflows):
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
                ).astype(np.float64)
                y_matrix[:, col_idx] = b_col
                mask_matrix[:, col_idx] = True

            # Full physical initial state:
            # [all modeled RMCs | all modeled PVs | V | modeled_cum...].
            # Loss targets may be a subset (e.g. PV-only), but the solver still
            # needs every modeled physical state axis.
            y0_state = np.asarray(
                [
                    *(
                        _initial_value_numpy(
                            process,
                            name,
                            TARGET_SOURCE_REACTOR_COMPONENTS,
                            t0_union,
                        )
                        for name in process_rhs_ode.name_modeled_RMCs
                    ),
                    *(
                        _initial_value_numpy(
                            process,
                            name,
                            TARGET_SOURCE_PROCESS_VARIABLES,
                            t0_union,
                        )
                        for name in process_rhs_ode.name_modeled_PVs
                    ),
                ],
                dtype=np.float64,
            )
            y0 = np.concatenate(
                [
                    y0_state,
                    np.asarray([v0], dtype=np.float64),
                    np.zeros(n_modeled, dtype=np.float64),
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
        t_measured = np.zeros((n_processes, max_n_meas), dtype=np.float64)
        y_measured = np.zeros((n_processes, max_n_meas, n_y_cols), dtype=np.float64)
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
            t_measured[index, :n_measured] = ts.astype(np.float64)
            y_measured[index, :n_measured, :] = ys.astype(np.float64)
            mask_measured[index, :n_measured, :] = mk

        return cls(
            process_order=process_order,
            name_measured_RMCs=name_measured_RMCs,
            name_measured_PVs=name_measured_PVs,
            name_modeled_Inflows=name_modeled_Inflows,
            name_modeled_Outflows=name_modeled_Outflows,
            controls_store=controls_store,
            rhs_ode=ref_rhs_ode,
            Cin_controlled_Inflows=jnp.stack(
                [rhs_ode.Cin_controlled_Inflows for rhs_ode in rhs_odes]
            ),
            Cin_modeled_Inflows=jnp.stack(
                [rhs_ode.Cin_modeled_Inflows for rhs_ode in rhs_odes]
            ),
            retention_controlled_Outflows=jnp.stack(
                [rhs_ode.retention_controlled_Outflows for rhs_ode in rhs_odes]
            ),
            retention_modeled_Outflows=jnp.stack(
                [rhs_ode.retention_modeled_Outflows for rhs_ode in rhs_odes]
            ),
            t_measured=jnp.asarray(t_measured),
            y_measured=jnp.asarray(y_measured),
            mask_measured=jnp.asarray(mask_measured),
            n_measured=jnp.asarray(n_meas_list, dtype=jnp.int32),
            y0_measured=jnp.asarray(np.asarray(per_process_y0, dtype=np.float64)),
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
        collection = load_process_collection(Path(prepared_json))
        return cls.from_collection(
            collection,
            target_variable_order=target_variable_order,
            target_source=target_source,
        )

    def select_processes(
        self,
        process_names: tuple[str, ...],
        collection: BioProcessCollection,
    ) -> TrainingDataStore:
        """Return a closed parent-aligned row selection in the requested order."""
        if not process_names:
            raise ValueError("selected training store must be non-empty")
        if tuple(collection.processes) != process_names:
            raise ValueError("selected collection order must match process_names")
        try:
            indices = jnp.asarray(
                [self.process_order.index(name) for name in process_names],
                dtype=jnp.int32,
            )
        except ValueError as error:
            raise KeyError(f"unknown selected process: {error.args[0]}") from error

        def rows(array):
            return array[indices]

        controlled_cin = rows(self.Cin_controlled_Inflows)
        modeled_cin = rows(self.Cin_modeled_Inflows)
        controlled_retention = rows(self.retention_controlled_Outflows)
        modeled_retention = rows(self.retention_modeled_Outflows)
        n_measured = rows(self.n_measured)
        max_measurements = int(np.max(np.asarray(n_measured)))
        rhs_ode = replace_rhs_ode_process_matrices(
            self.rhs_ode,
            controlled_cin[0],
            modeled_cin[0],
            controlled_retention[0],
            modeled_retention[0],
        )
        return TrainingDataStore(
            process_order=list(process_names),
            name_measured_RMCs=self.name_measured_RMCs,
            name_measured_PVs=self.name_measured_PVs,
            name_modeled_Inflows=self.name_modeled_Inflows,
            name_modeled_Outflows=self.name_modeled_Outflows,
            rhs_ode=rhs_ode,
            controls_store=self.controls_store.select_processes(
                process_names, collection
            ),
            Cin_controlled_Inflows=controlled_cin,
            Cin_modeled_Inflows=modeled_cin,
            retention_controlled_Outflows=controlled_retention,
            retention_modeled_Outflows=modeled_retention,
            t_measured=rows(self.t_measured)[:, :max_measurements],
            y_measured=rows(self.y_measured)[:, :max_measurements],
            mask_measured=rows(self.mask_measured)[:, :max_measurements],
            n_measured=n_measured,
            y0_measured=rows(self.y0_measured),
        )

    def validate_control_support(self, process_names: tuple[str, ...]) -> None:
        """Validate measured solve spans for the processes about to be solved."""
        spans = {}
        for process_name in process_names:
            _, process_index = _coerce_process_index(process_name, self.process_order)
            n_measured = int(np.asarray(self.n_measured[process_index]))
            active_ts = np.asarray(self.t_measured[process_index, :n_measured])
            spans[process_name] = (float(active_ts[0]), float(active_ts[-1]))
        self.controls_store.validate_supports(spans)

    def get_process(self, process: str | int) -> PerProcessTrainingData:
        """Return per-process training data by canonical name or integer index."""
        process_name, process_index = _coerce_process_index(process, self.process_order)
        return PerProcessTrainingData(
            process_name=process_name,
            process_index=process_index,
            name_measured_RMCs=self.name_measured_RMCs,
            name_measured_PVs=self.name_measured_PVs,
            name_modeled_Inflows=self.name_modeled_Inflows,
            name_modeled_Outflows=self.name_modeled_Outflows,
            n_measured=int(self.n_measured[process_index]),
            t_measured=self.t_measured[process_index],
            y_measured=self.y_measured[process_index],
            mask_measured=self.mask_measured[process_index],
            y0_measured=self.y0_measured[process_index],
            Cin_controlled_Inflows=self.Cin_controlled_Inflows[process_index],
            Cin_modeled_Inflows=self.Cin_modeled_Inflows[process_index],
            retention_controlled_Outflows=self.retention_controlled_Outflows[
                process_index
            ],
            retention_modeled_Outflows=self.retention_modeled_Outflows[process_index],
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
            controls=self.controls_store._gather_batch_rows(indices),
            t_measured=self.t_measured[indices],
            y_measured=self.y_measured[indices],
            mask_measured=self.mask_measured[indices],
            n_measured=self.n_measured[indices],
            y0_measured=self.y0_measured[indices],
            retention_controlled_Outflows=self.retention_controlled_Outflows[indices],
            retention_modeled_Outflows=self.retention_modeled_Outflows[indices],
        )
