from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from bp_format.dataclasses import (
    BioProcess,
    BioProcessCollection,
)
from bp_format.mechanistic import build_rhs_ode
from bp_format.serialization import load_process_collection
from bp_format.time_series.spline_ops import rebase_piece

from .constants import METADATA_NAMESPACE
from .controls import (
    ControlSourceBundle,
    build_linear_payload,
    build_spline_payload,
    collect_discrete_event_metadata,
    select_control_sources,
)


def _as_jax_array(values: Any, *, dtype: Any = jnp.float64) -> jax.Array:
    """Convert JSON-loaded values into a JAX array."""
    return jnp.asarray(np.asarray(values, dtype=dtype))


def _offline_measurement_times(process: BioProcess) -> np.ndarray:
    """Every timestamp an offline measurement could sit on, as a sorted array.

    Exactly ``{reactor components} ∪ {MEASURED process variables}`` — which is the
    complete set of columns any ``target_source`` can select
    (``reactor_components``, ``process_variables``, ``combined``, ``auto``), so it
    bounds the measurement block of the output grid for every run.

    ``is_controlled`` process variables are excluded, and that exclusion is the whole
    point of this helper. They are pH, temperature, gas flow, stirring speed — control
    INPUTS the RHS reads via ``eval_controlled_PVs``, logged online at thousands of
    points. They can never be measurement targets: ``training_data`` filters them out by
    default (``_process_variable_targets``) and raises if one is configured
    explicitly. Counting their timestamps as measurement rows inflated the
    output-window bound by two orders of magnitude (G of 288/317/451 instead of
    1/15/6 on the shipped examples),
    which silently disabled the window on every real dataset.
    """
    times: set[float] = set()

    def _collect(values: Any) -> None:
        # NB ``getattr(..., None) or ()`` would force a truth-value check on a JAX
        # array; compare against None explicitly.
        ts = getattr(values, "times", None)
        if ts is None:
            return
        times.update(np.asarray(ts, dtype=np.float64).reshape(-1).tolist())

    for component in process.reactor_medium.components.values():
        _collect(getattr(component, "concentration", None))
    for variable in (process.process_variables or {}).values():
        if bool(variable.is_controlled):
            continue
        _collect(getattr(variable, "values", None))
    return np.asarray(sorted(times), dtype=np.float64)


def _output_window_bounds(
    collection: BioProcessCollection,
    process_order: list[str],
    *,
    modeled_rmc_names_by_process: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[float, int]:
    """Collection-wide constants that size the solver's per-segment output window.

    The solver saves its trajectory with ``SaveAt(ts=...)`` inside each segment, and
    hands each segment only a fixed-size window of the output grid — otherwise diffrax
    writes every output slot in every segment and the work is ``O(n_segments * n_out)``,
    which on an event-heavy process is far slower than saving only at boundaries
    (measured 10x on the forward pass at 160 events). Per-segment save cost scales with
    the window, so it wants to be as tight as it provably can be. The size must be a
    static Python int, so it has to be bounded here rather than at trace time.

    An output grid is the measurement times plus up to two ``linspace(t0, t1, N)``
    blocks. In a gap of relative width ``f`` an N-point linspace contributes at most
    ``ceil(f * N) + 1`` points, so with two blocks and the measurement contribution:

        window >= ceil(f * (n_dense + n_prediction)) + measurements_per_gap + 2

    Returns ``(max_event_gap_fraction, max_measurements_per_event_gap)``: the two
    collection-wide maxima that ``f`` and ``measurements_per_gap`` need. Both are taken
    across every process because the window is one static size for the whole vmapped
    batch.

    PRECONDITION: the solver's output grid spans the process's measurement window, which
    every production path satisfies -- ``dense.build_union_time_grid`` linspaces
    over ``[t_measured[0], t_measured[n_measured - 1]]``. ``f`` is a RELATIVE gap,
    so a caller that hand-rolls a strictly narrower ``t_eval`` shrinks the
    denominator and can push the true fraction above this bound. No tight bound
    exists that survives an arbitrary sub-window (shrink the window around one gap
    and the fraction tends to 1), so this is
    a precondition rather than something to defend against, and
    ``CallbackSolution.output_overflow`` turns a violation into a loud error instead of
    silently dropped output rows.
    """
    spans: list[tuple[np.ndarray, np.ndarray]] = []
    for process_name in process_order:
        process = collection.processes[process_name]
        measured = _offline_measurement_times(process)
        if measured.size < 2:
            continue
        t0, t1 = float(measured[0]), float(measured[-1])
        species_names = (
            tuple(build_rhs_ode(process).name_modeled_RMCs)
            if modeled_rmc_names_by_process is None
            else modeled_rmc_names_by_process[process_name]
        )
        event_md = collect_discrete_event_metadata(process, species_names)
        events = np.asarray(
            list(event_md["sample_times"]) + list(event_md["bolus_times"]),
            dtype=np.float64,
        )
        events = np.unique(events[(events > t0) & (events <= t1)])
        spans.append((measured, np.unique(np.concatenate([[t0], events, [t1]]))))

    if not spans:
        return 1.0, 0
    # Padding width: ``clamp_padded_time_rows`` parks unused measurement slots at t1, so
    # they all land in the final gap and must be counted there.
    padded_width = max(measured.size for measured, _ in spans)

    gap_fraction = 0.0
    measurements_per_gap = 0
    for measured, boundaries in spans:
        t0, t1 = float(measured[0]), float(measured[-1])
        padded = np.concatenate(
            [measured, np.full(padded_width - measured.size, t1, dtype=np.float64)]
        )
        for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True):
            gap_fraction = max(gap_fraction, float(hi - lo) / (t1 - t0))
            measurements_per_gap = max(
                measurements_per_gap, int(((padded > lo) & (padded <= hi)).sum())
            )
    return gap_fraction, measurements_per_gap


def _discrete_event_jump_ts(process: BioProcess) -> list[float]:
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


_DEFAULT_CONTINUITY_SIDE = "right"


@dataclass(frozen=True)
class ControlPartition:
    """Canonical control layout implied by a collection's own control sources.

    `continuity_side` is `None` when no time-varying control constrains it, so a
    caller comparing against a stored side can tell "undetermined" apart from a
    genuine disagreement.
    """

    name_controlled_Inflows: tuple[str, ...]
    name_controlled_Outflows: tuple[str, ...]
    name_controlled_PVs: tuple[str, ...]
    spline_indices: tuple[int, ...]
    linear_indices: tuple[int, ...]
    continuity_side: str | None


def _control_partition(
    process_order: tuple[str, ...],
    process_bundles: Mapping[str, ControlSourceBundle],
) -> ControlPartition:
    """Categorise controls and split them into spline- and linear-backed columns."""
    reference_categorised: (
        tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None
    ) = None
    for process_name in process_order:
        bundle = process_bundles[process_name]
        categorised = (
            bundle.name_controlled_Inflows,
            bundle.name_controlled_Outflows,
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
    if reference_categorised is None:
        raise ValueError("process collection is empty")

    (
        name_controlled_Inflows,
        name_controlled_Outflows,
        name_controlled_PVs,
    ) = reference_categorised
    canonical_names = list(
        name_controlled_Inflows + name_controlled_Outflows + name_controlled_PVs
    )

    spline_names: list[str] = []
    sides_by_control: dict[str, dict[str, str]] = {}
    for control_name in canonical_names:
        sources = [
            process_bundles[process_name].sources_by_name[control_name]
            for process_name in process_order
        ]
        spline_processes = [
            process_name
            for process_name, source in zip(process_order, sources, strict=True)
            if source.spline_coeffs is not None
        ]
        non_spline_processes = [
            process_name
            for process_name, source in zip(process_order, sources, strict=True)
            if source.spline_coeffs is None
        ]
        if spline_processes and non_spline_processes:
            raise ValueError(
                f"control {control_name!r} must be spline-backed in every "
                "process or no process; spline-backed in "
                f"{spline_processes!r}, but not {non_spline_processes!r}"
            )
        if spline_processes:
            spline_names.append(control_name)
        for process_name, source in zip(process_order, sources, strict=True):
            if not source.is_static:
                assert source.continuity_side is not None
                sides_by_control.setdefault(control_name, {})[process_name] = (
                    source.continuity_side
                )

    all_sides = {
        side
        for sides_by_process in sides_by_control.values()
        for side in sides_by_process.values()
    }
    if len(all_sides) > 1:
        side_summary = {
            name: {
                side: next(
                    process
                    for process, process_side in sides.items()
                    if process_side == side
                )
                for side in sorted(set(sides.values()))
            }
            for name, sides in sides_by_control.items()
        }
        raise ValueError(
            "all time-varying controls must use one continuity side; found "
            f"{side_summary!r}"
        )
    spline_name_set = set(spline_names)
    canonical_index = {name: index for index, name in enumerate(canonical_names)}
    return ControlPartition(
        name_controlled_Inflows=name_controlled_Inflows,
        name_controlled_Outflows=name_controlled_Outflows,
        name_controlled_PVs=name_controlled_PVs,
        spline_indices=tuple(canonical_index[name] for name in spline_names),
        linear_indices=tuple(
            canonical_index[name]
            for name in canonical_names
            if name not in spline_name_set
        ),
        continuity_side=next(iter(all_sides), None),
    )


def derive_control_partition(collection: BioProcessCollection) -> ControlPartition:
    """Derive the canonical control layout from a collection's processes alone.

    Used at the runtime-artifact trust boundary, where control arrays are loaded
    straight from disk and therefore bypass `ControlsStore.__post_init__`.
    """
    process_order = tuple(collection.processes)
    return _control_partition(
        process_order,
        {
            name: select_control_sources(collection.processes[name])
            for name in process_order
        },
    )


def _control_representation(metadata: Mapping[str, Any]) -> str:
    source = str(metadata.get("source", "unknown"))
    return "raw" if source == "timeseries" else source


def _support_violations(
    process_name: str,
    t0: float,
    t1: float,
    control_supports: Mapping[str, tuple[float | None, float | None]],
    control_metadata: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Describe controls whose closed support does not contain ``[t0, t1]``."""
    if t1 < t0:
        raise ValueError(f"solve interval must satisfy t0 <= t1; got [{t0}, {t1}]")
    duration_tolerance = (t1 - t0) * 1.0e-6

    def _outside(value: float, bound: float, *, side: str) -> bool:
        scale = max(1.0, abs(value), abs(bound))
        tolerance = max(
            duration_tolerance,
            8.0 * abs(float(np.spacing(np.float32(scale)))),
        )
        if side == "left":
            return value < bound - tolerance
        return value > bound + tolerance

    violations = []
    for name, stored_support in control_supports.items():
        lower = -np.inf if stored_support[0] is None else float(stored_support[0])
        upper = np.inf if stored_support[1] is None else float(stored_support[1])
        sides = [
            side
            for side, violated in (
                ("left", _outside(t0, lower, side="left")),
                ("right", _outside(t1, upper, side="right")),
            )
            if violated
        ]
        if sides:
            representation = _control_representation(control_metadata[name])
            violations.append(
                f"process={process_name!r}, control={name!r}, "
                f"representation={representation!r}, requested=[{t0}, {t1}], "
                f"support=[{lower}, {upper}], violated_side={'+'.join(sides)!r}"
            )
    return violations


def _raise_support_violations(violations: list[str]) -> None:
    if violations:
        raise ValueError(
            "control support does not cover requested solve intervals:\n- "
            + "\n- ".join(violations)
        )


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
    """Linearly interpolate a `[n_grid, n_controls]` payload at query times."""

    def _interp_column(column: jax.Array) -> jax.Array:
        return jnp.interp(ts, grid, column, left=column[0], right=column[-1])

    return jax.vmap(_interp_column, in_axes=1, out_axes=1)(values)


def _eval_linear_columns(
    ts: jax.Array,
    grid: jax.Array,
    values: jax.Array,
    interval_slopes: jax.Array,
    side: str,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate piecewise-linear values and exact per-interval slopes."""
    indices = jnp.clip(
        jnp.searchsorted(grid, ts, side=side) - 1,
        0,
        grid.shape[0] - 2,
    )
    return _interp_columns(ts, grid, values), interval_slopes[indices]


def _eval_spline_columns(
    ts: jax.Array,
    breaks: jax.Array,
    coeffs: jax.Array,
    side: str,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate shared-grid cubic values and analytic first derivatives."""
    indices = jnp.clip(
        jnp.searchsorted(breaks, ts, side=side) - 1,
        0,
        coeffs.shape[0] - 1,
    )
    dt = ts - breaks[indices]
    pieces = coeffs[indices]
    dt = dt[:, None]
    a, b, c, d = (pieces[..., index] for index in range(4))
    values = a + dt * (b + dt * (c + dt * d))
    derivatives = b + dt * (2.0 * c + dt * 3.0 * d)
    return values, derivatives


def _eval_hybrid_columns(
    ts: jax.Array,
    spline_breaks: jax.Array,
    spline_coeffs: jax.Array,
    linear_grid: jax.Array,
    control_values: jax.Array,
    control_derivatives: jax.Array,
    spline_indices: tuple[int, ...],
    linear_indices: tuple[int, ...],
    continuity_side: str,
    n_controls: int,
) -> tuple[jax.Array, jax.Array]:
    values = jnp.zeros((ts.shape[0], n_controls), dtype=linear_grid.dtype)
    derivatives = jnp.zeros_like(values)
    if spline_indices:
        spline_values, spline_derivatives = _eval_spline_columns(
            ts, spline_breaks, spline_coeffs, continuity_side
        )
        values = values.at[:, spline_indices].set(spline_values)
        derivatives = derivatives.at[:, spline_indices].set(spline_derivatives)
    if linear_indices:
        linear_values, linear_derivatives = _eval_linear_columns(
            ts,
            linear_grid,
            control_values,
            control_derivatives,
            continuity_side,
        )
        values = values.at[:, linear_indices].set(linear_values)
        derivatives = derivatives.at[:, linear_indices].set(linear_derivatives)
    return values, derivatives


class PerProcessControls(eqx.Module):
    """Per-process hybrid runtime view over direct and linear controls.

    Column axis follows
    ``[name_controlled_Inflows | name_controlled_Outflows | name_controlled_PVs]``
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
    name_controlled_Inflows: tuple[str, ...] = eqx.field(static=True)
    name_controlled_Outflows: tuple[str, ...] = eqx.field(static=True)
    name_controlled_PVs: tuple[str, ...] = eqx.field(static=True)
    spline_breaks: jax.Array
    spline_coeffs: jax.Array
    linear_grid: jax.Array
    control_values: jax.Array
    control_derivatives: jax.Array
    spline_indices: tuple[int, ...] = eqx.field(static=True)
    linear_indices: tuple[int, ...] = eqx.field(static=True)
    continuity_side: str = eqx.field(static=True)
    jump_ts: jax.Array
    grid_length: int = eqx.field(static=True)
    jump_ts_length: int = eqx.field(static=True)
    min_V: jax.Array
    control_metadata: Mapping[str, Mapping[str, Any]] = eqx.field(static=True)
    control_supports: Mapping[str, tuple[float, float]] = eqx.field(static=True)
    sample_event_times: jax.Array
    sample_event_volumes: jax.Array
    sample_event_mask: jax.Array
    bolus_event_times: jax.Array
    bolus_event_volumes: jax.Array
    bolus_event_Cin: jax.Array
    bolus_event_mask: jax.Array
    # Collection-wide bounds that size the solver's per-segment output window; see
    # ``_output_window_bounds``. Static so a pytree traversal cannot rewrite them.
    max_event_gap_fraction: float = eqx.field(static=True)
    max_measurements_per_event_gap: int = eqx.field(static=True)

    @property
    def n_u(self) -> int:
        return (
            len(self.name_controlled_Inflows)
            + len(self.name_controlled_Outflows)
            + len(self.name_controlled_PVs)
        )

    @property
    def active_linear_grid(self) -> jax.Array:
        return self.linear_grid[: self.grid_length]

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
        """Interpolate all control values in canonical flow/PV column order.

        Private; public access is through the per-axis ``eval_controlled_*``
        methods.
        """
        query = jnp.asarray(ts, dtype=self.linear_grid.dtype)
        scalar_input = query.ndim == 0
        query_1d = jnp.atleast_1d(query)
        values, _ = _eval_hybrid_columns(
            query_1d,
            self.spline_breaks,
            self.spline_coeffs,
            self.active_linear_grid,
            self.active_control_values,
            self.active_control_derivatives,
            self.spline_indices,
            self.linear_indices,
            self.continuity_side,
            self.n_u,
        )
        return values[0] if scalar_input else values

    def _eval_derivatives(self, ts: float | np.ndarray | jax.Array) -> jax.Array:
        """Interpolate all control DERIVATIVES (flow rates) in canonical order.
        Private — sliced by the per-axis ``eval_controlled_*_rates`` accessors."""
        query = jnp.asarray(ts, dtype=self.linear_grid.dtype)
        scalar_input = query.ndim == 0
        query_1d = jnp.atleast_1d(query)
        _, derivatives = _eval_hybrid_columns(
            query_1d,
            self.spline_breaks,
            self.spline_coeffs,
            self.active_linear_grid,
            self.active_control_values,
            self.active_control_derivatives,
            self.spline_indices,
            self.linear_indices,
            self.continuity_side,
            self.n_u,
        )
        return derivatives[0] if scalar_input else derivatives

    # ------------------------------------------------------------------
    # Semantic, non-overlapping per-axis accessors. Each returns RAW
    # (physical, unscaled) values for a single control axis. ``states`` is a
    # placeholder for future state-dependent controls (e.g. pH feedback) and
    # is currently unused. The wrapper scales each result to SCL space via the
    # module's ``scale_controlled_*`` helpers before building ReactionInputs.
    # ------------------------------------------------------------------
    def validate_support(self, t0: float, t1: float) -> None:
        """Reject a solve interval outside any control's closed support.

        Comparisons allow one part per million of the solve duration, with
        eight float32 steps as a floor because source endpoints may have passed
        through float32 even when validation receives float64 arrays.
        """
        _raise_support_violations(
            _support_violations(
                self.process_name,
                t0,
                t1,
                self.control_supports,
                self.control_metadata,
            )
        )

    def eval_controlled_Inflows_cumulative(self, t_arr, states) -> jax.Array:
        n_inflows = len(self.name_controlled_Inflows)
        return self._eval_values(t_arr)[..., :n_inflows]

    def eval_controlled_Inflows_rates(self, t_arr, states) -> jax.Array:
        n_inflows = len(self.name_controlled_Inflows)
        return self._eval_derivatives(t_arr)[..., :n_inflows]

    def eval_controlled_Outflows_cumulative(self, t_arr, states) -> jax.Array:
        n_inflows = len(self.name_controlled_Inflows)
        n_outflows = len(self.name_controlled_Outflows)
        return self._eval_values(t_arr)[..., n_inflows : n_inflows + n_outflows]

    def eval_controlled_Outflows_rates(self, t_arr, states) -> jax.Array:
        n_inflows = len(self.name_controlled_Inflows)
        n_outflows = len(self.name_controlled_Outflows)
        return self._eval_derivatives(t_arr)[..., n_inflows : n_inflows + n_outflows]

    def eval_controlled_PVs(self, t_arr, states) -> jax.Array:
        n_inflows = len(self.name_controlled_Inflows)
        n_outflows = len(self.name_controlled_Outflows)
        n_pvs = len(self.name_controlled_PVs)
        start = n_inflows + n_outflows
        return self._eval_values(t_arr)[..., start : start + n_pvs]


class BatchControls(eqx.Module):
    """Batch-row controls evaluator with index-based runtime lookup.

    Column axis follows the same canonical
    ``[name_controlled_Inflows | name_controlled_Outflows | name_controlled_PVs]``
    order as :class:`PerProcessControls`.
    """

    spline_breaks: jax.Array
    spline_coeffs: jax.Array
    # Padded linear grids `[batch_size, max_grid_length]` with right-clamped tail.
    linear_grid: jax.Array
    # Padded linear values `[batch_size, max_grid_length, n_linear]`.
    control_values: jax.Array
    # Padded linear derivatives, same shape as control_values.
    control_derivatives: jax.Array
    spline_indices: tuple[int, ...] = eqx.field(static=True)
    linear_indices: tuple[int, ...] = eqx.field(static=True)
    continuity_side: str = eqx.field(static=True)
    name_controlled_Inflows: tuple[str, ...] = eqx.field(static=True)
    name_controlled_Outflows: tuple[str, ...] = eqx.field(static=True)
    name_controlled_PVs: tuple[str, ...] = eqx.field(static=True)
    min_V: jax.Array
    sample_event_times: jax.Array
    sample_event_volumes: jax.Array
    sample_event_mask: jax.Array
    bolus_event_times: jax.Array
    bolus_event_volumes: jax.Array
    bolus_event_Cin: jax.Array
    bolus_event_mask: jax.Array
    # Collection-wide bounds that size the solver's per-segment output window; see
    # ``_output_window_bounds``. Static so a pytree traversal cannot rewrite them.
    max_event_gap_fraction: float = eqx.field(static=True)
    max_measurements_per_event_gap: int = eqx.field(static=True)

    def _eval_values(self, row_idx: int, t: jax.Array) -> jax.Array:
        """Interpolate all control VALUES for one batch row, canonical order.
        Private — sliced by the per-axis ``eval_controlled_*`` accessors."""
        if isinstance(row_idx, (int, np.integer)):
            idx = int(row_idx)
            batch_size = int(self.linear_grid.shape[0])
            if idx < 0 or idx >= batch_size:
                raise IndexError(f"batch row out of range: {idx}")

        grid = self.linear_grid[row_idx]
        query = jnp.asarray(t, dtype=grid.dtype)
        scalar_input = query.ndim == 0
        query_1d = jnp.atleast_1d(query)
        out, _ = _eval_hybrid_columns(
            query_1d,
            self.spline_breaks[row_idx],
            self.spline_coeffs[row_idx],
            grid,
            self.control_values[row_idx],
            self.control_derivatives[row_idx],
            self.spline_indices,
            self.linear_indices,
            self.continuity_side,
            len(self.spline_indices) + len(self.linear_indices),
        )
        return out[0] if scalar_input else out

    def _eval_derivatives(self, row_idx: int, t: jax.Array) -> jax.Array:
        """Interpolate all control DERIVATIVES for one batch row, canonical
        order. Private — sliced by the ``eval_controlled_*_rates`` accessors."""
        grid = self.linear_grid[row_idx]
        query = jnp.asarray(t, dtype=grid.dtype)
        scalar_input = query.ndim == 0
        query_1d = jnp.atleast_1d(query)
        _, out = _eval_hybrid_columns(
            query_1d,
            self.spline_breaks[row_idx],
            self.spline_coeffs[row_idx],
            grid,
            self.control_values[row_idx],
            self.control_derivatives[row_idx],
            self.spline_indices,
            self.linear_indices,
            self.continuity_side,
            len(self.spline_indices) + len(self.linear_indices),
        )
        return out[0] if scalar_input else out

    # Semantic, non-overlapping per-axis accessors (RAW values). ``states`` is a
    # placeholder for future state-dependent controls and is currently unused.
    def eval_controlled_Inflows_cumulative(self, row_idx, t_arr, states) -> jax.Array:
        n_inflows = len(self.name_controlled_Inflows)
        return self._eval_values(row_idx, t_arr)[..., :n_inflows]

    def eval_controlled_Inflows_rates(self, row_idx, t_arr, states) -> jax.Array:
        n_inflows = len(self.name_controlled_Inflows)
        return self._eval_derivatives(row_idx, t_arr)[..., :n_inflows]

    def eval_controlled_Outflows_cumulative(self, row_idx, t_arr, states) -> jax.Array:
        n_inflows = len(self.name_controlled_Inflows)
        n_outflows = len(self.name_controlled_Outflows)
        return self._eval_values(row_idx, t_arr)[
            ..., n_inflows : n_inflows + n_outflows
        ]

    def eval_controlled_Outflows_rates(self, row_idx, t_arr, states) -> jax.Array:
        n_inflows = len(self.name_controlled_Inflows)
        n_outflows = len(self.name_controlled_Outflows)
        return self._eval_derivatives(row_idx, t_arr)[
            ..., n_inflows : n_inflows + n_outflows
        ]

    def eval_controlled_PVs(self, row_idx, t_arr, states) -> jax.Array:
        n_inflows = len(self.name_controlled_Inflows)
        n_outflows = len(self.name_controlled_Outflows)
        n_pvs = len(self.name_controlled_PVs)
        start = n_inflows + n_outflows
        return self._eval_values(row_idx, t_arr)[..., start : start + n_pvs]


class ControlsStore(eqx.Module):
    """Collection-level loader and index for prepared, padded JAX control tensors.

    Column axis follows
    ``[name_controlled_Inflows | name_controlled_Outflows | name_controlled_PVs]``
    consistently across every process; the wrapper consumes the full
    u-block via :meth:`PerProcessControls.eval_u`.
    """

    # Canonical prepared process keys in stable collection order.
    process_order: list[str]
    # Categorised name tuples (must be identical across processes).
    name_controlled_Inflows: tuple[str, ...]
    name_controlled_Outflows: tuple[str, ...]
    name_controlled_PVs: tuple[str, ...]
    # Padded max shapes. No runtime consumer today (only tests assert on it);
    # kept because it may be wanted again. Static so that a pytree traversal of
    # the store cannot silently rewrite the values, as it would for a leaf.
    shape_metadata: dict[str, Any] = eqx.field(static=True)
    spline_indices: tuple[int, ...] = eqx.field(static=True)
    linear_indices: tuple[int, ...] = eqx.field(static=True)
    continuity_side: str = eqx.field(static=True)
    # Shared spline grids and coefficients, padded across processes.
    spline_breaks: jax.Array
    spline_coeffs: jax.Array
    # Stacked right-clamped linear grids, shape
    # `[n_processes, max_grid_length]`.
    linear_grid: jax.Array
    # Stacked right-clamped linear values, shape
    # `[n_processes, max_grid_length, max_linear_controls]`.
    control_values: jax.Array
    # Stacked right-clamped control derivatives, same shape as `control_values`.
    control_derivatives: jax.Array
    # Stacked padded jump-time arrays, shape `[n_processes, max_jump_ts_length]`.
    jump_ts: jax.Array
    # Active linear-grid lengths per process.
    grid_lengths: jax.Array
    # Active `jump_ts` lengths per process.
    jump_ts_lengths: jax.Array
    min_V: jax.Array
    sample_event_times: jax.Array
    sample_event_volumes: jax.Array
    sample_event_mask: jax.Array
    bolus_event_times: jax.Array
    bolus_event_volumes: jax.Array
    bolus_event_Cin: jax.Array
    bolus_event_mask: jax.Array
    # Collection-wide bounds that size the solver's per-segment output window; see
    # ``_output_window_bounds``. Static so a pytree traversal cannot rewrite them.
    max_event_gap_fraction: float = eqx.field(static=True)
    max_measurements_per_event_gap: int = eqx.field(static=True)
    # Runtime-built per-process metadata entries needed to construct thin views.
    _process_md_by_name: dict[str, dict[str, Any]]

    def __post_init__(self) -> None:
        """Structurally validate the dispatch split on explicit construction.

        `spline_indices` and `linear_indices` must be ascending `tuple`s of exact
        `int` partitioning the canonical column range, `continuity_side` must be a
        known value, and each dispatch payload must be as wide as the index tuple
        addressing it. Motivating bug class: a cardinality-preserving error leaves
        every array shape unchanged, and for single-control collections the flow
        and PV axes are both width 1, so nothing downstream notices.

        **Structural only** — it cannot tell that a split is the one
        `prepared.json` implies. A paired swap (moving a spline to another column
        while adjusting the complement) and a legal-but-wrong `continuity_side` both
        pass; either would need the split re-derived from prepared. It also runs on
        explicit construction only: equinox rebuilds through `tree_unflatten`, so
        JAX transforms and leaf-level deserialization bypass it and must invoke
        this validation themselves.
        """
        n_columns = (
            len(self.name_controlled_Inflows)
            + len(self.name_controlled_Outflows)
            + len(self.name_controlled_PVs)
        )
        for name, indices in (
            ("spline_indices", self.spline_indices),
            ("linear_indices", self.linear_indices),
        ):
            # A list is mutable, so its contents could be edited after this check
            # ran, defeating a construction-time invariant. It also changes the
            # treedef, so an otherwise-identical store retraces instead of hitting
            # the jit cache, and mixing a list with a tuple fails obscurely later
            # at concatenation rather than here.
            if not isinstance(indices, tuple):
                raise TypeError(f"{name} must be a tuple; got {type(indices).__name__}")
            # `bool` is an `int` subclass and `0.0 == 0`; both pass a looser test
            # and then fail at JAX indexing, far from the cause.
            if any(type(index) is not int for index in indices):
                raise TypeError(f"{name} must hold exact ints; got {indices!r}")
            if list(indices) != sorted(indices):
                raise ValueError(
                    f"{name} must be ascending to match the column order its "
                    f"arrays were built in; got {indices!r}"
                )
        if sorted(self.spline_indices + self.linear_indices) != list(range(n_columns)):
            raise ValueError(
                f"spline_indices {self.spline_indices!r} and linear_indices "
                f"{self.linear_indices!r} must partition range({n_columns})"
            )
        if self.continuity_side not in ("left", "right"):
            raise ValueError(
                "continuity_side must be 'left' or 'right'; got "
                f"{self.continuity_side!r}"
            )
        n_linear = len(self.linear_indices)
        for name, width, expected in (
            ("control_values", self.control_values.shape[-1], n_linear),
            ("control_derivatives", self.control_derivatives.shape[-1], n_linear),
            ("spline_coeffs", self.spline_coeffs.shape[-2], len(self.spline_indices)),
        ):
            if width != expected:
                raise ValueError(
                    f"{name} is {width} columns wide but its index tuple names "
                    f"{expected}"
                )

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
            ("name_controlled_Inflows", bundle.name_controlled_Inflows),
            ("name_controlled_Outflows", bundle.name_controlled_Outflows),
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
    def _pad_spline_payload(
        payload: dict[str, Any],
        max_breaks: int,
    ) -> tuple[list[float], list[list[list[float]]]]:
        breaks = list(payload["breaks"])
        coeffs = [list(map(list, row)) for row in payload["coeffs"]]
        if not breaks:
            return [], []

        padding = max_breaks - len(breaks)
        if padding:
            endpoint = breaks[-1]
            shift = endpoint - breaks[-2]
            final_row = [
                rebase_piece(np.asarray(source_coeffs), shift).tolist()
                for source_coeffs in coeffs[-1]
            ]
            coeffs.extend([final_row] * padding)
            breaks.extend([float("inf")] * padding)
        return breaks, coeffs

    @classmethod
    def from_collection(
        cls,
        collection: BioProcessCollection,
    ) -> ControlsStore:
        """Build a JAX-backed runtime store from a prepared `BioProcessCollection`."""
        metadata = dict(collection.metadata or {})
        process_order = cls._process_order(collection, metadata, METADATA_NAMESPACE)
        bp_train = dict(metadata.get(METADATA_NAMESPACE, {}))
        prepared_process_md = dict(bp_train.get("processes", {}))

        process_bundles: dict[str, ControlSourceBundle] = {}
        process_control_metadata: dict[str, dict[str, Any]] = {}

        for process_name in process_order:
            process = collection.processes[process_name]
            bundle = select_control_sources(process)
            prepared_md = prepared_process_md.get(process_name)
            cls._validate_bundle_against_metadata(
                process_name=process_name,
                bundle=bundle,
                process_md=prepared_md,
            )
            process_bundles[process_name] = bundle
            process_control_metadata[process_name] = {
                source.name: source.metadata for source in bundle.all_sources
            }

        partition = _control_partition(tuple(process_order), process_bundles)
        name_controlled_Inflows = partition.name_controlled_Inflows
        name_controlled_Outflows = partition.name_controlled_Outflows
        name_controlled_PVs = partition.name_controlled_PVs
        canonical_names: list[str] = list(
            name_controlled_Inflows + name_controlled_Outflows + name_controlled_PVs
        )
        spline_indices = partition.spline_indices
        linear_indices = partition.linear_indices
        continuity_side = partition.continuity_side or _DEFAULT_CONTINUITY_SIDE
        spline_names = [canonical_names[index] for index in spline_indices]
        linear_names = [canonical_names[index] for index in linear_indices]

        reference_species: tuple[str, ...] | None = None
        modeled_rmc_names_by_process: dict[str, tuple[str, ...]] = {}
        max_grid_length = 0
        max_spline_breaks = 0
        max_jump_ts_length = 0
        max_sample_events = 0
        max_bolus_events = 0
        for process_name in process_order:
            process = collection.processes[process_name]
            bundle = process_bundles[process_name]
            species_names = tuple(build_rhs_ode(process).name_modeled_RMCs)
            modeled_rmc_names_by_process[process_name] = species_names
            if reference_species is None:
                reference_species = species_names
            elif species_names != reference_species:
                raise ValueError(
                    "controls store requires identical modeled RMC layout across "
                    f"processes for event metadata; {process_name!r} has "
                    f"{species_names!r} but expected {reference_species!r}"
                )
            event_md = collect_discrete_event_metadata(process, species_names)
            payload = build_linear_payload(
                process=process,
                sources=[bundle.sources_by_name[name] for name in linear_names],
            )
            spline_payload = build_spline_payload(
                [bundle.sources_by_name[name] for name in spline_names]
            )
            max_grid_length = max(max_grid_length, len(payload["grid"]))
            max_spline_breaks = max(max_spline_breaks, len(spline_payload["breaks"]))
            max_jump_ts_length = max(
                max_jump_ts_length, len(_discrete_event_jump_ts(process))
            )
            max_sample_events = max(max_sample_events, len(event_md["sample_times"]))
            max_bolus_events = max(max_bolus_events, len(event_md["bolus_times"]))

        n_processes = len(process_order)
        n_linear = len(linear_names)
        n_splines = len(spline_names)
        n_species = 0 if reference_species is None else len(reference_species)
        spline_break_rows = np.empty((n_processes, max_spline_breaks), dtype=np.float64)
        spline_coeff_rows = np.empty(
            (n_processes, max(0, max_spline_breaks - 1), n_splines, 4),
            dtype=np.float64,
        )
        linear_grid_rows = np.empty((n_processes, max_grid_length), dtype=np.float64)
        control_value_rows = np.empty(
            (n_processes, max_grid_length, n_linear), dtype=np.float64
        )
        control_derivative_rows = np.empty_like(control_value_rows)
        jump_ts_rows = np.zeros((n_processes, max_jump_ts_length), dtype=np.float64)
        grid_lengths = np.empty(n_processes, dtype=np.int32)
        jump_ts_lengths = np.empty(n_processes, dtype=np.int32)
        min_V_rows = np.empty(n_processes, dtype=np.float64)
        sample_event_time_rows = np.zeros(
            (n_processes, max_sample_events), dtype=np.float64
        )
        sample_event_volume_rows = np.zeros_like(sample_event_time_rows)
        sample_event_mask_rows = np.zeros((n_processes, max_sample_events), dtype=bool)
        bolus_event_time_rows = np.zeros(
            (n_processes, max_bolus_events), dtype=np.float64
        )
        bolus_event_volume_rows = np.zeros_like(bolus_event_time_rows)
        bolus_event_Cin_rows = np.zeros(
            (n_processes, max_bolus_events, n_species), dtype=np.float64
        )
        bolus_event_mask_rows = np.zeros((n_processes, max_bolus_events), dtype=bool)
        processes_metadata: dict[str, dict[str, Any]] = {}

        for process_index, process_name in enumerate(process_order):
            process = collection.processes[process_name]
            bundle = process_bundles[process_name]
            payload = build_linear_payload(
                process=process,
                sources=[bundle.sources_by_name[name] for name in linear_names],
            )
            jump_ts = _discrete_event_jump_ts(process)
            spline_breaks, spline_coeffs = cls._pad_spline_payload(
                build_spline_payload(
                    [bundle.sources_by_name[name] for name in spline_names]
                ),
                max_spline_breaks,
            )
            grid_length = len(payload["grid"])
            jump_ts_length = len(jump_ts)
            grid = np.asarray(payload["grid"], dtype=np.float64)
            values = np.asarray(payload["values"], dtype=np.float64)
            derivatives = np.asarray(payload["derivatives"], dtype=np.float64)

            if n_splines:
                spline_break_rows[process_index] = spline_breaks
                spline_coeff_rows[process_index] = spline_coeffs
            linear_grid_rows[process_index, :grid_length] = grid
            linear_grid_rows[process_index, grid_length:] = grid[-1]
            control_value_rows[process_index, :grid_length] = values
            control_value_rows[process_index, grid_length:] = values[-1]
            control_derivative_rows[process_index, :grid_length] = derivatives
            control_derivative_rows[process_index, grid_length:] = derivatives[-1]
            jump_ts_rows[process_index, :jump_ts_length] = jump_ts
            grid_lengths[process_index] = grid_length
            jump_ts_lengths[process_index] = jump_ts_length
            min_V_rows[process_index] = process.volume.initial_volume * 1e-3

            event_md = collect_discrete_event_metadata(process, reference_species)
            n_samples = len(event_md["sample_times"])
            n_bolus = len(event_md["bolus_times"])
            sample_event_time_rows[process_index, :n_samples] = event_md["sample_times"]
            sample_event_volume_rows[process_index, :n_samples] = event_md[
                "sample_volumes"
            ]
            sample_event_mask_rows[process_index, :n_samples] = True
            bolus_event_time_rows[process_index, :n_bolus] = event_md["bolus_times"]
            bolus_event_volume_rows[process_index, :n_bolus] = event_md["bolus_volumes"]
            if n_bolus:
                bolus_event_Cin_rows[process_index, :n_bolus] = event_md["bolus_Cin"]
            bolus_event_mask_rows[process_index, :n_bolus] = True

            processes_metadata[process_name] = {
                "name_controlled_Inflows": list(name_controlled_Inflows),
                "name_controlled_Outflows": list(name_controlled_Outflows),
                "name_controlled_PVs": list(name_controlled_PVs),
                "control_metadata": process_control_metadata[process_name],
                "control_supports": {
                    source.name: tuple(
                        None if not np.isfinite(bound) else bound
                        for bound in source.support
                    )
                    for source in bundle.all_sources
                },
            }

        gap_fraction, measurements_per_gap = _output_window_bounds(
            collection,
            process_order,
            modeled_rmc_names_by_process=modeled_rmc_names_by_process,
        )

        shape_metadata = {
            "n_processes": len(process_order),
            "max_grid_length": max_grid_length,
            "max_spline_breaks": max_spline_breaks,
            "max_controls": len(canonical_names),
            "max_jump_ts_length": max_jump_ts_length,
            "max_sample_events": max_sample_events,
            "max_bolus_events": max_bolus_events,
        }

        return cls(
            process_order=process_order,
            name_controlled_Inflows=name_controlled_Inflows,
            name_controlled_Outflows=name_controlled_Outflows,
            name_controlled_PVs=name_controlled_PVs,
            shape_metadata=shape_metadata,
            spline_indices=spline_indices,
            linear_indices=linear_indices,
            continuity_side=continuity_side,
            spline_breaks=(
                jnp.zeros((len(process_order), 0), dtype=jnp.float64)
                if not spline_names
                else _as_jax_array(spline_break_rows)
            ),
            spline_coeffs=(
                jnp.zeros((len(process_order), 0, 0, 4), dtype=jnp.float64)
                if not spline_names
                else _as_jax_array(spline_coeff_rows)
            ),
            linear_grid=_as_jax_array(linear_grid_rows),
            control_values=_as_jax_array(control_value_rows),
            control_derivatives=_as_jax_array(control_derivative_rows),
            jump_ts=_as_jax_array(jump_ts_rows),
            grid_lengths=jnp.asarray(grid_lengths, dtype=jnp.int32),
            jump_ts_lengths=jnp.asarray(jump_ts_lengths, dtype=jnp.int32),
            min_V=_as_jax_array(min_V_rows),
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
            max_event_gap_fraction=gap_fraction,
            max_measurements_per_event_gap=measurements_per_gap,
            _process_md_by_name=processes_metadata,
        )

    def select_processes(
        self,
        process_names: tuple[str, ...],
        collection: BioProcessCollection,
        *,
        modeled_rmc_names: tuple[str, ...] | None = None,
    ) -> ControlsStore:
        """Return a closed row-selected store without rebuilding controls.

        ``modeled_rmc_names`` reuses a caller's already-validated canonical RHS
        layout. Direct callers may omit it and retain the standalone fallback.
        """
        if not process_names:
            raise ValueError("selected controls store must be non-empty")
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

        spline_breaks = rows(self.spline_breaks)
        grid_lengths = rows(self.grid_lengths)
        jump_ts_lengths = rows(self.jump_ts_lengths)
        sample_event_mask = rows(self.sample_event_mask)
        bolus_event_mask = rows(self.bolus_event_mask)
        max_grid_length = int(np.max(np.asarray(grid_lengths)))
        max_spline_breaks = (
            int(np.max(np.sum(np.isfinite(np.asarray(spline_breaks)), axis=1)))
            if spline_breaks.shape[1]
            else 0
        )
        max_jump_ts_length = int(np.max(np.asarray(jump_ts_lengths)))
        max_sample_events = int(np.max(np.sum(np.asarray(sample_event_mask), axis=1)))
        max_bolus_events = int(np.max(np.sum(np.asarray(bolus_event_mask), axis=1)))

        gap_fraction, measurements_per_gap = _output_window_bounds(
            collection,
            process_names,
            modeled_rmc_names_by_process=(
                None
                if modeled_rmc_names is None
                else dict.fromkeys(process_names, modeled_rmc_names)
            ),
        )
        shape_metadata = {
            **self.shape_metadata,
            "n_processes": len(process_names),
            "max_grid_length": max_grid_length,
            "max_spline_breaks": max_spline_breaks,
            "max_jump_ts_length": max_jump_ts_length,
            "max_sample_events": max_sample_events,
            "max_bolus_events": max_bolus_events,
        }
        return ControlsStore(
            process_order=list(process_names),
            name_controlled_Inflows=self.name_controlled_Inflows,
            name_controlled_Outflows=self.name_controlled_Outflows,
            name_controlled_PVs=self.name_controlled_PVs,
            shape_metadata=shape_metadata,
            spline_indices=self.spline_indices,
            linear_indices=self.linear_indices,
            continuity_side=self.continuity_side,
            spline_breaks=spline_breaks[:, :max_spline_breaks],
            spline_coeffs=rows(self.spline_coeffs)[:, : max(0, max_spline_breaks - 1)],
            linear_grid=rows(self.linear_grid)[:, :max_grid_length],
            control_values=rows(self.control_values)[:, :max_grid_length],
            control_derivatives=rows(self.control_derivatives)[:, :max_grid_length],
            jump_ts=rows(self.jump_ts)[:, :max_jump_ts_length],
            grid_lengths=grid_lengths,
            jump_ts_lengths=jump_ts_lengths,
            min_V=rows(self.min_V),
            sample_event_times=rows(self.sample_event_times)[:, :max_sample_events],
            sample_event_volumes=rows(self.sample_event_volumes)[:, :max_sample_events],
            sample_event_mask=sample_event_mask[:, :max_sample_events],
            bolus_event_times=rows(self.bolus_event_times)[:, :max_bolus_events],
            bolus_event_volumes=rows(self.bolus_event_volumes)[:, :max_bolus_events],
            bolus_event_Cin=rows(self.bolus_event_Cin)[:, :max_bolus_events],
            bolus_event_mask=bolus_event_mask[:, :max_bolus_events],
            max_event_gap_fraction=gap_fraction,
            max_measurements_per_event_gap=measurements_per_gap,
            _process_md_by_name={
                name: self._process_md_by_name[name] for name in process_names
            },
        )

    @classmethod
    def from_json(
        cls,
        prepared_json: str | Path,
    ) -> ControlsStore:
        """Load a prepared JSON artifact and construct a `ControlsStore`."""
        collection = load_process_collection(Path(prepared_json))
        return cls.from_collection(collection)

    def validate_supports(
        self,
        spans: Mapping[str, tuple[float, float]],
    ) -> None:
        """Validate selected process solve spans and report all violations."""
        violations = []
        for process_name, (t0, t1) in spans.items():
            if process_name not in self._process_md_by_name:
                raise KeyError(f"unknown process name: {process_name}")
            process_md = self._process_md_by_name[process_name]
            violations.extend(
                _support_violations(
                    process_name,
                    t0,
                    t1,
                    process_md["control_supports"],
                    process_md["control_metadata"],
                )
            )
        _raise_support_violations(violations)

    def get_controls(self, process: str | int) -> PerProcessControls:
        """Return per-process controls by canonical prepared key or index."""
        process_name, process_index = _coerce_index(process, self.process_order)
        process_md = self._process_md_by_name[process_name]

        return PerProcessControls(
            process_name=process_name,
            process_index=process_index,
            name_controlled_Inflows=self.name_controlled_Inflows,
            name_controlled_Outflows=self.name_controlled_Outflows,
            name_controlled_PVs=self.name_controlled_PVs,
            spline_breaks=self.spline_breaks[process_index],
            spline_coeffs=self.spline_coeffs[process_index],
            linear_grid=self.linear_grid[process_index],
            control_values=self.control_values[process_index],
            control_derivatives=self.control_derivatives[process_index],
            spline_indices=self.spline_indices,
            linear_indices=self.linear_indices,
            continuity_side=self.continuity_side,
            jump_ts=self.jump_ts[process_index],
            grid_length=int(self.grid_lengths[process_index]),
            jump_ts_length=int(self.jump_ts_lengths[process_index]),
            min_V=self.min_V[process_index],
            control_metadata=process_md["control_metadata"],
            control_supports={
                name: (
                    -np.inf if support[0] is None else support[0],
                    np.inf if support[1] is None else support[1],
                )
                for name, support in process_md["control_supports"].items()
            },
            sample_event_times=self.sample_event_times[process_index],
            sample_event_volumes=self.sample_event_volumes[process_index],
            sample_event_mask=self.sample_event_mask[process_index],
            bolus_event_times=self.bolus_event_times[process_index],
            bolus_event_volumes=self.bolus_event_volumes[process_index],
            bolus_event_Cin=self.bolus_event_Cin[process_index],
            bolus_event_mask=self.bolus_event_mask[process_index],
            max_event_gap_fraction=self.max_event_gap_fraction,
            max_measurements_per_event_gap=self.max_measurements_per_event_gap,
        )

    def gather_batch(self, process_indices: jax.Array | np.ndarray) -> BatchControls:
        """Gather control and event rows in the requested order."""
        indices = jnp.asarray(process_indices, dtype=jnp.int32)
        if indices.ndim != 1:
            raise ValueError("process_indices must be a 1D array")
        if indices.size == 0:
            raise ValueError("process_indices must be non-empty")
        if bool(jnp.any(indices < 0)) or bool(
            jnp.any(indices >= len(self.process_order))
        ):
            raise IndexError("process index out of range in process_indices")

        return self._gather_batch_rows(indices)

    def _gather_batch_rows(self, indices: jax.Array) -> BatchControls:
        """Gather already-validated, int32 process-index rows."""
        return BatchControls(
            spline_breaks=self.spline_breaks[indices],
            spline_coeffs=self.spline_coeffs[indices],
            linear_grid=self.linear_grid[indices],
            control_values=self.control_values[indices],
            control_derivatives=self.control_derivatives[indices],
            spline_indices=self.spline_indices,
            linear_indices=self.linear_indices,
            continuity_side=self.continuity_side,
            name_controlled_Inflows=self.name_controlled_Inflows,
            name_controlled_Outflows=self.name_controlled_Outflows,
            name_controlled_PVs=self.name_controlled_PVs,
            min_V=self.min_V[indices],
            sample_event_times=self.sample_event_times[indices],
            sample_event_volumes=self.sample_event_volumes[indices],
            sample_event_mask=self.sample_event_mask[indices],
            bolus_event_times=self.bolus_event_times[indices],
            bolus_event_volumes=self.bolus_event_volumes[indices],
            bolus_event_Cin=self.bolus_event_Cin[indices],
            bolus_event_mask=self.bolus_event_mask[indices],
            max_event_gap_fraction=self.max_event_gap_fraction,
            max_measurements_per_event_gap=self.max_measurements_per_event_gap,
        )
