from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import warnings
from typing import Any, Callable

import numpy as np
from bp_format.dataclasses import (
    AugmentedBioProcess,
    BioProcessCollection,
    TimeSeries,
)
from bp_format.mechanistic import get_process_ordering

from .run_config import AugmentationConfig, InitialValueSource, RunConfig


AugmentStateValues = Callable[..., Any]
_TIME_ATOL = 1e-9
_SPLINE_ROOT_IMAG_ATOL = 1e-12
_MULTIPLICATIVE_SCALE_FLOOR = 1e-8
_MOSTLY_NONNEGATIVE_FRACTION = 0.5
_MIN_RELATIVE_RESIDUAL_RMS = 1e-6
# Grid-coordinate roundoff grows with the requested spacing, so gap validation
# needs relative slack. The requested minimum must still clear the coarsest
# timestamp step by several ULP, or the grid cannot meaningfully honor it.
_MIN_SPACING_REL_TOL = 1e-6
_MIN_SPACING_MIN_ULPS = 4


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
) -> np.ndarray:
    n_intervals = augmentation.n_time_points - 1
    duration = t_end - t0
    min_spacing = augmentation.min_spacing_fraction * duration / n_intervals
    remaining_duration = duration * (1.0 - augmentation.min_spacing_fraction)
    cuts = np.sort(
        _rng(augmentation.seed, parent_name, child_index, "grid").uniform(
            0.0,
            remaining_duration,
            augmentation.n_time_points - 2,
        )
    )
    interior = t0 + np.arange(1, n_intervals) * min_spacing + cuts
    grid = np.concatenate(([t0], interior, [t_end]))
    diffs = np.diff(grid)
    resolution = np.spacing(max(abs(t0), abs(t_end)))
    if (
        np.any(diffs <= 0.0)
        or min_spacing <= _MIN_SPACING_MIN_ULPS * resolution
        or np.any(diffs < min_spacing * (1.0 - _MIN_SPACING_REL_TOL))
    ):
        raise ValueError(
            f"{_child_name(parent_name, child_index)}: cannot represent the "
            "requested minimum child-grid spacing"
        )
    return grid


def _state_series(process, state_name: str) -> Any:
    if state_name in process.reactor_medium.components:
        return process.reactor_medium.components[state_name].concentration
    return process.process_variables[state_name].values


def _parent_processes(collection: BioProcessCollection) -> list[tuple[str, Any]]:
    return [
        (name, process)
        for name, process in collection.processes.items()
        if not isinstance(process, AugmentedBioProcess)
    ]


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


def _residual_statistics(
    parent_name: str,
    state_name: str,
    series: TimeSeries,
) -> tuple[float, float]:
    if series.times is None or series.values is None:
        raise ValueError(
            f"{parent_name}: modeled state {state_name!r} requires observations "
            "to compute spline residuals"
        )
    observed = np.asarray(series.values, dtype=float)
    fitted = np.asarray(series.evaluate_many(series.times), dtype=float)
    residual_rms = float(np.sqrt(np.mean((observed - fitted) ** 2)))
    observed_rms = float(np.sqrt(np.mean(observed**2)))
    return residual_rms, observed_rms


def _effective_residual_statistics(
    parents: list[tuple[str, Any]],
    augmentation: AugmentationConfig,
) -> dict[tuple[str, str], tuple[float, float, float]]:
    statistics = {}
    for parent_name, parent in parents:
        for state_name in augmentation.variable_names:
            residual_rms, observed_rms = _residual_statistics(
                parent_name,
                state_name,
                _state_series(parent, state_name),
            )
            statistics[parent_name, state_name] = (
                residual_rms,
                observed_rms,
                observed_rms,
            )
    if augmentation.residual_scope == "process":
        return statistics

    for state_name in augmentation.variable_names:
        weighted_squared_residuals = 0.0
        weighted_squared_observations = 0.0
        observation_count = 0
        for parent_name, parent in parents:
            series = _state_series(parent, state_name)
            count = len(series.values)
            residual_rms, observed_rms, _ = statistics[parent_name, state_name]
            if observed_rms == 0.0:
                continue
            weighted_squared_residuals += count * residual_rms**2
            weighted_squared_observations += count * observed_rms**2
            observation_count += count
        pooled_residual_rms = (
            np.sqrt(weighted_squared_residuals / observation_count)
            if observation_count
            else 0.0
        )
        pooled_observed_rms = (
            np.sqrt(weighted_squared_observations / observation_count)
            if observation_count
            else 0.0
        )
        for parent_name, _ in parents:
            _, observed_rms, _ = statistics[parent_name, state_name]
            statistics[parent_name, state_name] = (
                float(pooled_residual_rms),
                observed_rms,
                float(pooled_observed_rms),
            )
    return statistics


def _multiplicative_reference_magnitude(series: TimeSeries) -> float:
    fitted = np.asarray(series.evaluate_many(series.times), dtype=float)
    magnitudes = np.abs(fitted[fitted != 0.0])
    return float(np.mean(magnitudes)) if magnitudes.size else 0.0


def _multiplicative_relative_std(err_std: float, reference_magnitude: float) -> float:
    return err_std / max(reference_magnitude, _MULTIPLICATIVE_SCALE_FLOOR)


def _validate_requested_states(
    parent_name: str,
    process,
    augmentation: AugmentationConfig,
) -> tuple[str, ...]:
    ordering = get_process_ordering(process)
    modeled_names = ordering.name_modeled_RMCs + ordering.name_modeled_PVs
    for state_name in augmentation.variable_names:
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


def _built_in_values(
    *,
    parent_name: str,
    state_name: str,
    base_values: np.ndarray,
    residual_rms: float,
    observed_scale_rms: float,
    multiplicative_reference_magnitude: float,
    standard_normal: np.ndarray,
    augmentation: AugmentationConfig,
) -> np.ndarray:
    relative_residual_rms = (
        residual_rms / observed_scale_rms if observed_scale_rms else 0.0
    )
    if relative_residual_rms <= _MIN_RELATIVE_RESIDUAL_RMS:
        raise ValueError(
            f"{parent_name}: modeled state {state_name!r} has an effectively "
            "zero spline residual; augment_state_values can supply an absolute "
            "noise level"
        )
    if state_name not in augmentation.noise_scale:
        raise ValueError(f"{parent_name}: noise_scale is missing {state_name!r}")

    err_std = augmentation.noise_scale[state_name] * residual_rms
    if augmentation.noise_model == "add":
        return np.clip(base_values + standard_normal * err_std, 0.0, None)

    rel_std = _multiplicative_relative_std(
        err_std,
        multiplicative_reference_magnitude,
    )
    sigma = np.sqrt(np.log1p(rel_std**2))
    return base_values * np.exp(-0.5 * sigma**2 + sigma * standard_normal)


def _augment_values(
    *,
    parent_name: str,
    child_name: str,
    child_index: int,
    state_name: str,
    times: np.ndarray,
    base_values: np.ndarray,
    residual_rms: float,
    observed_rms: float,
    observed_scale_rms: float,
    multiplicative_reference_magnitude: float,
    augmentation: AugmentationConfig,
    run_config: RunConfig,
    augment_state_values: AugmentStateValues | None,
) -> np.ndarray:
    standard_normal = _rng(
        augmentation.seed,
        parent_name,
        child_index,
        state_name,
        "values",
    ).standard_normal(times.shape)

    values = None
    if augment_state_values is not None:
        values = augment_state_values(
            parent_name=parent_name,
            child_name=child_name,
            state_name=state_name,
            times=times.copy(),
            base_values=base_values.copy(),
            residual_rms=residual_rms,
            observed_rms=observed_rms,
            standard_normal=standard_normal.copy(),
            config=run_config,
        )
    if values is None:
        values = _built_in_values(
            parent_name=parent_name,
            state_name=state_name,
            base_values=base_values,
            residual_rms=residual_rms,
            observed_scale_rms=observed_scale_rms,
            multiplicative_reference_magnitude=(multiplicative_reference_magnitude),
            standard_normal=standard_normal,
            augmentation=augmentation,
        )

    values = np.array(values, dtype=float, copy=True)
    if values.shape != times.shape:
        raise ValueError(
            f"{child_name}: augment_state_values returned shape {values.shape} "
            f"for {state_name!r}; expected {times.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{child_name}: augment_state_values returned non-finite values for "
            f"{state_name!r}"
        )
    return values


def augment_process_collection(
    collection: BioProcessCollection,
    run_config: RunConfig,
    augment_state_values: AugmentStateValues | None = None,
) -> BioProcessCollection:
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
            for state_name in augmentation.variable_names
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

    residual_statistics = _effective_residual_statistics(parents, augmentation)
    multiplicative_reference_magnitudes = {
        (parent_name, state_name): _multiplicative_reference_magnitude(
            _state_series(parent, state_name)
        )
        for parent_name, parent in parents
        for state_name in augmentation.variable_names
    }

    child_times = {
        (parent_name, child_index): _child_grid(
            augmentation, parent_name, child_index, t0, t_end
        )
        for parent_name, _, _, t0, t_end, _ in validated_parents
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
                if state_name in augmentation.variable_names
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
                base_values = np.asarray(series.evaluate_many(times), dtype=float)
                if state_name in augmentation.variable_names:
                    residual_rms, observed_rms, observed_scale_rms = (
                        residual_statistics[parent_name, state_name]
                    )
                    multiplicative_reference_magnitude = (
                        multiplicative_reference_magnitudes[parent_name, state_name]
                    )
                    values = _augment_values(
                        parent_name=parent_name,
                        child_name=child_name,
                        child_index=child_index,
                        state_name=state_name,
                        times=times,
                        base_values=base_values,
                        residual_rms=residual_rms,
                        observed_rms=observed_rms,
                        observed_scale_rms=observed_scale_rms,
                        multiplicative_reference_magnitude=(
                            multiplicative_reference_magnitude
                        ),
                        augmentation=augmentation,
                        run_config=run_config,
                        augment_state_values=augment_state_values,
                    )
                    source = _initial_value_source(augmentation, state_name)
                    if source == "measured":
                        values[0] = measured_initial_values[state_name]
                    elif source == "spline":
                        values[0] = base_values[0]
                else:
                    values = base_values
                if is_rmc:
                    values = np.clip(values, 0.0, None)
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
