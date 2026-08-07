from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from bp_format.dataclasses import (
    BioProcessCollection,
    SampleVolumeChange,
    StaticVariable,
)
from bp_format.time_series.timeseries import TimeSeries

from .model_api import EstimatedScales
from .training_data import TrainingDataStore


RawTrace = tuple[np.ndarray, np.ndarray]
BoundDeclaration = tuple[str, str, int, float | None, float | None]
BoundSnapshot = tuple[BoundDeclaration, ...]
BoundRecord = tuple[str, str, int, float, float]


@dataclass(frozen=True)
class RuntimeDataContext:
    """Prepared collection-free numeric data available to runtime hooks."""

    training_data: TrainingDataStore
    augmentation_parents: tuple[str | None, ...]
    process_time_bounds: tuple[tuple[float, float], ...]
    modeled_volume_change_traces: tuple[tuple[RawTrace, ...], ...]
    raw_state_traces: tuple[tuple[RawTrace, ...], ...]
    sample_volume_event_traces: tuple[RawTrace, ...]
    bound_snapshots: tuple[BoundSnapshot, ...]

    @property
    def rhs_ode(self):
        return self.training_data.rhs_ode

    @property
    def controls_store(self):
        return self.training_data.controls_store

    @property
    def process_order(self) -> tuple[str, ...]:
        return tuple(self.training_data.process_order)

    @property
    def parent_indices(self) -> tuple[int, ...]:
        indices = tuple(
            index
            for index, parent in enumerate(self.augmentation_parents)
            if parent is None
        )
        if not indices:
            raise ValueError("runtime data contains no non-augmented processes")
        return indices

    def time_bounds(self, process_index: int) -> tuple[float, float]:
        return self.process_time_bounds[process_index]

    def initial_volume(self, process_index: int) -> float:
        rhs_ode = self.rhs_ode
        volume_index = len(rhs_ode.name_modeled_RMCs) + len(rhs_ode.name_modeled_PVs)
        return float(self.training_data.y0_measured[process_index, volume_index])

    def measured_values(self, process_index: int, name: str) -> np.ndarray:
        """Return one unpadded measured target trace."""
        try:
            column = self.training_data.name_measured.index(name)
        except ValueError as error:
            raise KeyError(f"unknown measured target {name!r}") from error
        mask = np.asarray(self.training_data.mask_measured[process_index, :, column])
        return np.asarray(self.training_data.y_measured[process_index, mask, column])

    def raw_state_trace(self, process_index: int, name: str) -> RawTrace:
        """Return one exact raw modeled state trace."""
        names = self.rhs_ode.name_modeled_RMCs + self.rhs_ode.name_modeled_PVs
        try:
            column = names.index(name)
        except ValueError as error:
            raise KeyError(f"unknown modeled state {name!r}") from error
        return self.raw_state_traces[process_index][column]

    def sample_volume_events(self, process_index: int) -> RawTrace:
        """Return exact sample-volume event times and values."""
        return self.sample_volume_event_traces[process_index]

    def modeled_volume_change_trace(self, process_index: int, name: str) -> RawTrace:
        """Return one exact modeled cumulative volume-change trace."""
        names = (
            self.training_data.name_modeled_FVCs + self.training_data.name_modeled_SVCs
        )
        try:
            column = names.index(name)
        except ValueError as error:
            raise KeyError(f"unknown modeled volume change {name!r}") from error
        return self.modeled_volume_change_traces[process_index][column]

    @classmethod
    def from_collection(
        cls,
        training_data: TrainingDataStore,
        collection: BioProcessCollection,
    ) -> RuntimeDataContext:
        process_order = tuple(training_data.process_order)
        if not process_order:
            raise ValueError("runtime context requires a non-empty collection")
        if tuple(collection.processes) != process_order:
            raise ValueError(
                "runtime context process order differs between collection and training data"
            )

        rhs_ode = training_data.rhs_ode
        state_names = rhs_ode.name_modeled_RMCs + rhs_ode.name_modeled_PVs
        volume_change_names = (
            training_data.name_modeled_FVCs + training_data.name_modeled_SVCs
        )
        parents: list[str | None] = []
        time_bounds: list[tuple[float, float]] = []
        modeled_traces: list[tuple[RawTrace, ...]] = []
        state_traces: list[tuple[RawTrace, ...]] = []
        sample_traces: list[RawTrace] = []
        bound_snapshots: list[BoundSnapshot] = []

        for process_name in process_order:
            process = collection.processes[process_name]
            start = float(process.time_axis.start)
            end = float(process.time_axis.end)
            parents.append(getattr(process, "parent_process", None))
            time_bounds.append((start, end))
            modeled_traces.append(
                tuple(
                    _trace(
                        process.volume.volume_changes[name].values,
                        process_name,
                        f"modeled volume change {name!r}",
                    )
                    for name in volume_change_names
                )
            )
            state_traces.append(
                tuple(
                    _raw_state_trace(process, name, start, end) for name in state_names
                )
            )
            sample_traces.append(_sample_volume_events(process, process_name))
            bound_snapshots.append(_bound_snapshot(process, training_data))

        return cls(
            training_data=training_data,
            augmentation_parents=tuple(parents),
            process_time_bounds=tuple(time_bounds),
            modeled_volume_change_traces=tuple(modeled_traces),
            raw_state_traces=tuple(state_traces),
            sample_volume_event_traces=tuple(sample_traces),
            bound_snapshots=tuple(bound_snapshots),
        )


@dataclass(frozen=True)
class RuntimeContext:
    """Prepared runtime data plus fully resolved semantic-axis scales."""

    data: RuntimeDataContext
    scales: EstimatedScales

    def __post_init__(self) -> None:
        if not isinstance(self.data, RuntimeDataContext):
            raise TypeError("data must be a RuntimeDataContext")
        if not isinstance(self.scales, EstimatedScales):
            raise TypeError("scales must be an EstimatedScales")

    @property
    def training_data(self) -> TrainingDataStore:
        return self.data.training_data


def collect_bound_records(
    snapshots: tuple[BoundSnapshot, ...],
) -> tuple[BoundRecord, ...]:
    """Validate per-process bound declarations when bounds loss is requested."""
    if not snapshots:
        raise ValueError("bounds loss requires a non-empty bounds snapshot")
    records: list[BoundRecord] = []
    reference = snapshots[0]
    for index, declaration in enumerate(reference):
        label, source, axis, lower, upper = declaration
        for process_index, snapshot in enumerate(snapshots[1:], start=1):
            try:
                other = snapshot[index]
            except IndexError as error:
                raise ValueError(
                    f"Bounds source {label!r} is missing from process index {process_index}"
                ) from error
            if other != declaration:
                raise ValueError(
                    f"Bounds for {label!r} differ across processes: {declaration[3:]!r} "
                    f"vs {other[3:]!r}"
                )
        for description, threshold in (("Lower", lower), ("Upper", upper)):
            if threshold is not None and not math.isfinite(threshold):
                raise ValueError(
                    f"{description} bound for {label!r} must be finite or None"
                )
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(
                f"Lower bound for {label!r} must not exceed its upper bound"
            )
        if lower is not None:
            records.append((f"lwr_bnd/{label}", source, axis, 1.0, lower))
        if upper is not None:
            records.append((f"upr_bnd/{label}", source, axis, -1.0, upper))
    return tuple(records)


def _bound_snapshot(process, store: TrainingDataStore) -> BoundSnapshot:
    rhs_ode = store.rhs_ode
    declarations: list[BoundDeclaration] = []
    for index, name in enumerate(rhs_ode.name_modeled_RMCs):
        declarations.append(
            (
                name,
                "state",
                index,
                *_bounds(process.reactor_medium.components[name].bounds),
            )
        )
    pv_offset = len(rhs_ode.name_modeled_RMCs)
    for index, name in enumerate(rhs_ode.name_modeled_PVs, start=pv_offset):
        declarations.append(
            (name, "state", index, *_bounds(process.process_variables[name].bounds))
        )
    state_names = rhs_ode.name_modeled_RMCs + rhs_ode.name_modeled_PVs
    volume_label = "volume/V" if "V" in state_names else "V"
    declarations.append(
        (
            volume_label,
            "volume",
            pv_offset + len(rhs_ode.name_modeled_PVs),
            *_bounds(process.volume.bounds),
        )
    )
    for index, name in enumerate(rhs_ode.name_modeled_rates):
        bounds = (
            (None, None)
            if process.biological_ode is None
            else process.biological_ode.rates[name]
        )
        declarations.append((f"rate/{name}", "rate", index, *_bounds(bounds)))
    return tuple(declarations)


def _bounds(bounds) -> tuple[float | None, float | None]:
    lower, upper = tuple(bounds)
    return (
        None if lower is None else float(lower),
        None if upper is None else float(upper),
    )


def _raw_state_trace(process, name: str, start: float, end: float) -> RawTrace:
    if name in process.reactor_medium.components:
        value = process.reactor_medium.components[name].concentration
    else:
        value = process.process_variables[name].values
    if isinstance(value, StaticVariable):
        return _readonly_trace([start, end], [value.value, value.value])
    return _trace(value, process.metadata.name, f"modeled state {name!r}")


def _sample_volume_events(process, process_name: str) -> RawTrace:
    traces = tuple(
        _trace(change.values, process_name, f"sample volume change {name!r}")
        for name, change in process.volume.volume_changes.items()
        if isinstance(change, SampleVolumeChange)
    )
    if not traces:
        return _readonly_trace([], [])
    return _readonly_trace(
        np.concatenate([times for times, _ in traces]),
        np.concatenate([values for _, values in traces]),
    )


def _trace(value, process_name: str, description: str) -> RawTrace:
    if not isinstance(value, TimeSeries) or value.times is None:
        raise TypeError(
            f"{process_name}: unsupported raw trace for {description}: "
            f"{type(value).__name__}"
        )
    return _readonly_trace(value.times, value.values, process_name, description)


def _readonly_trace(
    times, values, process_name: str = "", description: str = ""
) -> RawTrace:
    times_array = np.array(times, dtype=np.float64, copy=True)
    values_array = np.array(values, dtype=np.float64, copy=True)
    if (
        times_array.ndim != 1
        or values_array.ndim != 1
        or times_array.size != values_array.size
        or (description and not times_array.size)
    ):
        raise ValueError(
            f"{process_name}: {description} has invalid non-empty 1D time/value shape"
        )
    times_array.setflags(write=False)
    values_array.setflags(write=False)
    return times_array, values_array
