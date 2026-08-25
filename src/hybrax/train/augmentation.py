"""Synthetic sibling-process generation from
:class:`~hybrax.train.run_config.AugmentationConfig`.

Resamples each real (parent) process's fitted state splines onto a new,
randomly jittered time grid, optionally adds Gaussian noise, and stores the
result as new :class:`~hybrax.format.dataclasses.AugmentedBioProcess` entries
in the collection. See ``AugmentationConfig`` for the full parameter
semantics (seed, grid spacing, noise, initial-value handling).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import warnings
from typing import Any

import numpy as np
from hybrax.format.dataclasses import (
    AugmentedBioProcess,
    BioProcess,
    BioProcessCollection,
    Outflow,
    StaticVariable,
    TimeSeries,
)
from hybrax.format.mechanistic import get_process_ordering

from .run_config import AugmentationConfig, InitialValueSource, RunConfig


_TIME_ATOL = 1e-9
_SPLINE_ROOT_IMAG_ATOL = 1e-12
_MOSTLY_NONNEGATIVE_FRACTION = 0.5
# Grid-coordinate roundoff grows with the requested spacing, so gap validation
# needs relative slack. The requested minimum must still clear the coarsest
# timestamp step by several ULP, or the grid cannot meaningfully honor it.
_MIN_SPACING_REL_TOL = 1e-6
_MIN_SPACING_MIN_ULPS = 4
# Matches hybrax.format.validate_measurement_sampling_alignment's default.
_SAMPLING_NEAR_MISS_REL_THRESHOLD = 1e-4
_MAX_GRID_ATTEMPTS = 100


def _times_match(left, right):
    return np.isclose(left, right, atol=_TIME_ATOL, rtol=0.0)


def _rng(seed: int, *identity: object) -> np.random.Generator:
    text = "\0".join([str(seed), *(str(part) for part in identity)])
    stable_seed = int.from_bytes(sha256(text.encode()).digest()[:8], "little")
    return np.random.default_rng(stable_seed)


def _child_name(parent_name: str, child_index: int) -> str:
    return f"{parent_name}__aug_{child_index:03d}"


def _child_grid(
    augmentation: AugmentationConfig,
    parent_name: str,
    child_index: int,
    t0: float,
    t_end: float,
    sampling_times: tuple[float, ...] = (),
) -> np.ndarray:
    n_intervals = augmentation.n_time_points - 1
    duration = t_end - t0
    min_spacing = augmentation.min_spacing_fraction * duration / n_intervals
    remaining_duration = duration * (1.0 - augmentation.min_spacing_fraction)
    resolution = np.spacing(max(abs(t0), abs(t_end)))
    if min_spacing <= _MIN_SPACING_MIN_ULPS * resolution:
        raise ValueError(
            f"{_child_name(parent_name, child_index)}: cannot represent the "
            "requested minimum child-grid spacing"
        )

    samples = np.asarray(sampling_times)
    for attempt in range(_MAX_GRID_ATTEMPTS):
        identity = (
            (augmentation.seed, parent_name, child_index, "grid")
            if attempt == 0
            else (augmentation.seed, parent_name, child_index, "grid", attempt)
        )
        cuts = np.sort(
            _rng(*identity).uniform(
                0.0,
                remaining_duration,
                augmentation.n_time_points - 2,
            )
        )
        interior = t0 + np.arange(1, n_intervals) * min_spacing + cuts
        grid = np.concatenate(([t0], interior, [t_end]))
        diffs = np.diff(grid)
        if np.any(diffs <= 0.0) or np.any(
            diffs < min_spacing * (1.0 - _MIN_SPACING_REL_TOL)
        ):
            raise ValueError(
                f"{_child_name(parent_name, child_index)}: cannot represent the "
                "requested minimum child-grid spacing"
            )
        if samples.size == 0:
            return grid
        deltas = grid[:, None] - samples[None, :]
        if not np.any(
            (deltas > 0.0) & (deltas <= _SAMPLING_NEAR_MISS_REL_THRESHOLD * duration)
        ):
            return grid

    raise ValueError(
        f"{_child_name(parent_name, child_index)}: could not sample a child grid "
        "away from sampling-event near-misses"
    )


def _state_series(process, state_name: str) -> TimeSeries | StaticVariable:
    if state_name in process.reactor_medium.components:
        return process.reactor_medium.components[state_name].concentration
    return process.process_variables[state_name].values


def _parent_processes(
    collection: BioProcessCollection,
) -> list[tuple[str, BioProcess]]:
    return [
        (name, process)
        for name, process in collection.processes.items()
        if not isinstance(process, AugmentedBioProcess)
    ]


def _sampling_times(process) -> tuple[float, ...]:
    return tuple(
        float(t)
        for change in process.volume.volume_changes.values()
        if isinstance(change, Outflow) and not change.is_continuous
        for t in change.values.times
    )


def _set_state_series(process, state_name: str, series: TimeSeries) -> None:
    if state_name in process.reactor_medium.components:
        process.reactor_medium.components[state_name].concentration = series
    else:
        process.process_variables[state_name].values = series


def _initial_value_source(
    augmentation: AugmentationConfig,
    state_name: str,
) -> InitialValueSource:
    source = augmentation.initial_value_source
    return source[state_name] if isinstance(source, dict) else source


def _measured_initial_value(
    parent_name: str,
    state_name: str,
    series: TimeSeries,
    t0: float,
) -> float:
    times = np.asarray(series.times, dtype=float)
    matches = np.flatnonzero(_times_match(times, t0))
    if matches.size == 0:
        raise ValueError(
            f"{parent_name}: initial_value_source='measured' for {state_name!r} "
            f"requires an observation at process start t={t0:.6g}"
        )
    return float(np.asarray(series.values, dtype=float)[matches[0]])


def _spline_dips_below_zero(series: TimeSeries, t0: float, t_end: float) -> bool:
    breaks = np.asarray(series.breaks, dtype=float)
    coeffs = np.asarray(series.coeffs, dtype=float)
    last_piece = len(coeffs) - 1

    for index, (a, b, c, d) in enumerate(coeffs):
        left = t0 if index == 0 else max(t0, breaks[index])
        right = t_end if index == last_piece else min(t_end, breaks[index + 1])
        if left > right:
            continue

        local_left = left - breaks[index]
        local_right = right - breaks[index]
        candidates = [local_left, local_right]
        for root in np.roots([3.0 * d, 2.0 * c, b]):
            if (
                abs(root.imag) <= _SPLINE_ROOT_IMAG_ATOL
                and local_left <= root.real <= local_right
            ):
                candidates.append(float(root.real))
        values = [a + x * (b + x * (c + x * d)) for x in candidates]
        if min(values) < 0.0:
            return True

    return False


def _validate_requested_states(
    parent_name: str,
    process,
    augmentation: AugmentationConfig,
) -> tuple[str, ...]:
    ordering = get_process_ordering(process)
    modeled_names = ordering.name_modeled_RMCs + ordering.name_modeled_PVs
    for state_name in augmentation.noise_std:
        if state_name in ordering.name_controlled_PVs:
            raise ValueError(
                f"{parent_name}: controlled process variable {state_name!r} "
                "cannot be augmented"
            )
        if state_name not in modeled_names:
            raise ValueError(f"{parent_name}: {state_name!r} is not a modeled state")
        series = _state_series(process, state_name)
        if not isinstance(series, TimeSeries) or series.poly is None:
            raise ValueError(
                f"{parent_name}: modeled state {state_name!r} requires a spline"
            )
    return modeled_names


def augment_process_collection(
    collection: BioProcessCollection,
    run_config: RunConfig,
    augment_state_values: Any = None,
) -> BioProcessCollection:
    """Generate augmented children for every parent process and add them to
    ``collection``.

    No-op (returns ``collection`` unchanged) when ``run_config.prepare.augmentation``
    is unset. Otherwise, for each real (non-augmented) process, generates
    ``augmentation.n_children_per_process`` synthetic children: each gets its
    own randomly jittered time grid, and every modeled state named in
    ``augmentation.noise_std`` is resampled from the parent's fitted spline
    onto that grid with Gaussian noise added (reactor-medium components are
    clipped to stay non-negative). Every other modeled state is copied through
    from the spline unchanged. Mutates ``collection.processes`` in place with
    the new children and also returns ``collection``.

    Args:
        collection: Parent process collection to augment; every non-augmented
            process in it is a candidate parent.
        run_config: Run configuration; only ``run_config.prepare.augmentation``
            is read.
        augment_state_values: Optional ``augment_state_values`` custom hook.
            When given, called once per augmented state per child to override
            the noised values before they are stored; must return an array of
            the same shape with only finite values.

    Returns:
        ``collection``, with the generated children added.

    Raises:
        ValueError: If a generated child name collides with an existing
            process, a requested state cannot be augmented (controlled PV,
            unmodeled, or spline-less), the degenerate case ``t0 == t_end``,
            the requested grid spacing cannot be represented, or a custom
            ``augment_state_values`` hook returns a mismatched shape or
            non-finite values.
    """
    prepare = run_config.prepare
    if prepare is None or prepare.augmentation is None:
        return collection
    augmentation = prepare.augmentation
    parents = _parent_processes(collection)

    child_names = [
        _child_name(parent_name, child_index)
        for parent_name, _ in parents
        for child_index in range(augmentation.n_children_per_process)
    ]
    collisions = [name for name in child_names if name in collection.processes]
    if collisions:
        raise ValueError(f"generated augmented process already exists: {collisions[0]}")

    validated_parents = []
    for parent_name, parent in parents:
        modeled_names = _validate_requested_states(parent_name, parent, augmentation)
        t0 = float(parent.time_axis.start)
        t_end = float(parent.time_axis.end)
        if t0 == t_end:
            raise ValueError(f"{parent_name}: cannot augment a degenerate time range")
        measured_initial_values = {
            state_name: _measured_initial_value(
                parent_name,
                state_name,
                _state_series(parent, state_name),
                t0,
            )
            for state_name in augmentation.noise_std
            if _initial_value_source(augmentation, state_name) == "measured"
        }
        validated_parents.append(
            (
                parent_name,
                parent,
                modeled_names,
                t0,
                t_end,
                measured_initial_values,
            )
        )

    child_times = {
        (parent_name, child_index): _child_grid(
            augmentation,
            parent_name,
            child_index,
            t0,
            t_end,
            _sampling_times(parent),
        )
        for parent_name, parent, _, t0, t_end, _ in validated_parents
        for child_index in range(augmentation.n_children_per_process)
    }
    children = {}

    for (
        parent_name,
        parent,
        modeled_names,
        t0,
        t_end,
        measured_initial_values,
    ) in validated_parents:
        for state_name in modeled_names:
            series = _state_series(parent, state_name)
            if not isinstance(series, TimeSeries) or series.poly is None:
                continue
            source = (
                _initial_value_source(augmentation, state_name)
                if state_name in augmentation.noise_std
                else "spline"
            )
            if source != "measured" and series.times is not None:
                first_observation = float(np.asarray(series.times)[0])
                extrapolates_to_t0 = first_observation > t0 and not _times_match(
                    first_observation, t0
                )
                if extrapolates_to_t0:
                    warnings.warn(
                        f"{parent_name}: spline for {state_name!r} is extrapolated "
                        f"before its first observation to construct the augmented "
                        f"initial value at t={t0:.6g}",
                        stacklevel=2,
                    )
            is_rmc = state_name in parent.reactor_medium.components
            mostly_nonnegative = series.values is not None and (
                np.mean(np.asarray(series.values) >= 0.0) > _MOSTLY_NONNEGATIVE_FRACTION
            )
            if _spline_dips_below_zero(series, t0, t_end) and (
                is_rmc or mostly_nonnegative
            ):
                state_kind = (
                    "reactor-medium component" if is_rmc else "process variable"
                )
                warnings.warn(
                    f"{parent_name}: spline for {state_kind} {state_name!r} "
                    "evaluated below zero during augmentation",
                    stacklevel=2,
                )

        for child_index in range(augmentation.n_children_per_process):
            child_name = _child_name(parent_name, child_index)
            child = AugmentedBioProcess(
                **vars(deepcopy(parent)),
                parent_process=parent_name,
            )
            child.metadata.name = child_name
            if hasattr(child.metadata, "_pre_transform_key"):
                del child.metadata._pre_transform_key
            times = child_times[parent_name, child_index]

            for state_name in modeled_names:
                series = _state_series(child, state_name)
                if not isinstance(series, TimeSeries) or series.poly is None:
                    continue
                is_rmc = state_name in child.reactor_medium.components
                base_values = np.array(
                    series.evaluate_many(times),
                    dtype=float,
                    copy=True,
                )
                if state_name in augmentation.noise_std:
                    noise_std = augmentation.noise_std[state_name]
                    if noise_std == 0.0:
                        values = base_values.copy()
                    else:
                        standard_normal = _rng(
                            augmentation.seed,
                            parent_name,
                            child_index,
                            state_name,
                            "values",
                        ).standard_normal(times.shape)
                        values = base_values + noise_std * standard_normal
                    source = _initial_value_source(augmentation, state_name)
                    if source == "measured":
                        values[0] = measured_initial_values[state_name]
                    elif source == "spline":
                        values[0] = base_values[0]
                else:
                    values = base_values
                if is_rmc:
                    values = np.clip(values, 0.0, None)
                if (
                    state_name in augmentation.noise_std
                    and augment_state_values is not None
                ):
                    custom_values = np.asarray(
                        augment_state_values(
                            parent_name=parent_name,
                            child_name=child_name,
                            state_name=state_name,
                            times=times.copy(),
                            base_values=base_values.copy(),
                            augmented_values=values.copy(),
                            config=run_config,
                        ),
                        dtype=float,
                    )
                    if custom_values.shape != values.shape:
                        raise ValueError(
                            f"{child_name}: augment_state_values returned shape "
                            f"{custom_values.shape} for {state_name!r}; expected "
                            f"{values.shape}"
                        )
                    if not np.all(np.isfinite(custom_values)):
                        raise ValueError(
                            f"{child_name}: augment_state_values returned non-finite "
                            f"values for {state_name!r}"
                        )
                    values = custom_values
                _set_state_series(
                    child,
                    state_name,
                    replace(
                        series,
                        times=times,
                        values=values,
                        breaks=None,
                        coeffs=None,
                        segment_start_piece_idx=None,
                    ),
                )

            children[child_name] = child

    collection.processes.update(children)
    return collection
