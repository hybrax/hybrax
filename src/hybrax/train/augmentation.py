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


def _times_match(left, right):
    return np.isclose(left, right, atol=_TIME_ATOL, rtol=0.0)


def _rng(seed: int, *identity: object) -> np.random.Generator:
    text = "\0".join([str(seed), *(str(part) for part in identity)])
    stable_seed = int.from_bytes(sha256(text.encode()).digest()[:8], "little")
    return np.random.default_rng(stable_seed)


def _child_grid(
    config: AugmentationConfig,
    parent_name: str,
    child_index: int,
    t0: float,
    t_end: float,
) -> np.ndarray:
    interior = _rng(config.seed, parent_name, child_index, "grid").uniform(
        t0,
        t_end,
        config.n_time_points - 2,
    )
    return np.concatenate(([t0], np.sort(interior), [t_end]))


def _state_series(process, state_name: str) -> Any:
    if state_name in process.reactor_medium.components:
        return process.reactor_medium.components[state_name].concentration
    return process.process_variables[state_name].values


def _set_state_series(process, state_name: str, series: TimeSeries) -> None:
    if state_name in process.reactor_medium.components:
        process.reactor_medium.components[state_name].concentration = series
    else:
        process.process_variables[state_name].values = series


def _initial_value_source(
    config: AugmentationConfig,
    state_name: str,
) -> InitialValueSource:
    source = config.initial_value_source
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
            if abs(root.imag) <= 1e-12 and local_left <= root.real <= local_right:
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


def _validate_requested_states(
    parent_name: str,
    process,
    config: AugmentationConfig,
) -> tuple[str, ...]:
    ordering = get_process_ordering(process)
    modeled_names = ordering.name_modeled_RMCs + ordering.name_modeled_PVs
    for state_name in config.variable_names:
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
    observed_rms: float,
    standard_normal: np.ndarray,
    config: AugmentationConfig,
) -> np.ndarray:
    relative_residual_rms = residual_rms / observed_rms if observed_rms else 0.0
    if observed_rms == 0.0 or (
        relative_residual_rms <= config.min_relative_residual_rms
    ):
        raise ValueError(
            f"{parent_name}: modeled state {state_name!r} has an effectively "
            "zero spline residual; augment_state_values can supply an absolute "
            "noise level"
        )
    if state_name not in config.noise_scale:
        raise ValueError(f"{parent_name}: noise_scale is missing {state_name!r}")

    err_std = config.noise_scale[state_name] * residual_rms
    if config.noise_model == "add":
        return np.clip(base_values + standard_normal * err_std, 0.0, None)

    positive = base_values[base_values > 0.0]
    mean_positive = float(np.mean(positive)) if positive.size else 0.0
    rel_std = err_std / max(mean_positive, 1e-8)
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
    config: AugmentationConfig,
    run_config: RunConfig,
    augment_state_values: AugmentStateValues | None,
) -> np.ndarray:
    standard_normal = _rng(
        config.seed,
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
            observed_rms=observed_rms,
            standard_normal=standard_normal,
            config=config,
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
    config = prepare.augmentation
    parents = [
        (name, process)
        for name, process in collection.processes.items()
        if not isinstance(process, AugmentedBioProcess)
    ]

    child_names = [
        f"{parent_name}__aug_{child_index:03d}"
        for parent_name, _ in parents
        for child_index in range(config.n_children_per_process)
    ]
    collisions = [name for name in child_names if name in collection.processes]
    if collisions:
        raise ValueError(f"generated augmented process already exists: {collisions[0]}")

    residual_statistics = {}
    validated_parents = []
    for parent_name, parent in parents:
        modeled_names = _validate_requested_states(parent_name, parent, config)
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
            for state_name in config.variable_names
            if _initial_value_source(config, state_name) == "measured"
        }
        for state_name in config.variable_names:
            residual_statistics[parent_name, state_name] = _residual_statistics(
                parent_name,
                state_name,
                _state_series(parent, state_name),
            )
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
                _initial_value_source(config, state_name)
                if state_name in config.variable_names
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
                np.mean(np.asarray(series.values) >= 0.0) > 0.5
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

        for child_index in range(config.n_children_per_process):
            child_name = f"{parent_name}__aug_{child_index:03d}"
            parent_copy = deepcopy(parent)
            child = AugmentedBioProcess(
                **vars(parent_copy),
                parent_process=parent_name,
            )
            child.metadata.name = child_name
            if hasattr(child.metadata, "_pre_transform_key"):
                del child.metadata._pre_transform_key
            times = _child_grid(config, parent_name, child_index, t0, t_end)

            for state_name in modeled_names:
                series = _state_series(child, state_name)
                if not isinstance(series, TimeSeries) or series.poly is None:
                    continue
                is_rmc = state_name in child.reactor_medium.components
                base_values = np.asarray(series.evaluate_many(times), dtype=float)
                if state_name in config.variable_names:
                    residual_rms, observed_rms = residual_statistics[
                        parent_name, state_name
                    ]
                    values = _augment_values(
                        parent_name=parent_name,
                        child_name=child_name,
                        child_index=child_index,
                        state_name=state_name,
                        times=times,
                        base_values=base_values,
                        residual_rms=residual_rms,
                        observed_rms=observed_rms,
                        config=config,
                        run_config=run_config,
                        augment_state_values=augment_state_values,
                    )
                    source = _initial_value_source(config, state_name)
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

            collection.processes[child_name] = child

    return collection
