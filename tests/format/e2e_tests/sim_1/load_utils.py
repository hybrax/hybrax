"""Load sim 1 simulation CSVs into hybrax.format collections."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import re
from typing import Iterable

os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax.numpy as jnp

import hybrax.format as bp  # noqa: E402

from .simulation import (  # noqa: E402
    FEED_GLUCOSE,
    FEED_GLUTAMINE,
    FULL_STATE_NAMES,
    RATIO_RELAXATION_PER_H,
    RATIO_TARGET,
    REACTOR_MEDIUM_NAMES,
    REACTOR_STATE_UNITS,
    TRACER_FED_FEED_CONCENTRATION,
)

PROCESS_ID_COLUMN = "process_id"
INITIAL_TIME_MATCH_TOL = 1e-12
STATE_UNITS = REACTOR_STATE_UNITS
CONTROL_COLUMNS = ("pH", "temperature", "cum_base_feed", "cum_conti_feed")
EVENT_COLUMNS = (
    PROCESS_ID_COLUMN,
    "time",
    "event_order",
    "event_type",
    "delta_volume",
    "feed_id",
    *(f"feed_{name}" for name in REACTOR_MEDIUM_NAMES),
)


def parse_all_processes(
    *,
    dense_csv: Path,
    events_csv: Path,
    collection_name: str = "sim_1",
    process_ids: Iterable[str] | None = None,
) -> bp.BioProcessCollection:
    dense_columns, dense_rows = _read_csv(dense_csv)
    event_columns, event_rows = _read_csv(events_csv)
    _require_columns(
        dense_columns, [PROCESS_ID_COLUMN, "time", "row_type", *FULL_STATE_NAMES]
    )
    _require_columns(dense_columns, CONTROL_COLUMNS)
    _require_columns(event_columns, EVENT_COLUMNS)
    if not dense_rows:
        raise ValueError("dense CSV has no rows.")

    dense_process_ids = {row[PROCESS_ID_COLUMN] for row in dense_rows}
    if process_ids is None:
        selected_process_ids = sorted(dense_process_ids)
    else:
        selected_process_ids = sorted(set(process_ids))
        missing = sorted(set(selected_process_ids) - dense_process_ids)
        if missing:
            raise ValueError(f"Unknown requested process ids: {missing}")

    event_process_ids = {row[PROCESS_ID_COLUMN] for row in event_rows}
    unknown_event_process_ids = sorted(event_process_ids - dense_process_ids)
    if unknown_event_process_ids:
        raise ValueError(
            "simulation_events.csv has unknown process ids: "
            f"{unknown_event_process_ids}"
        )

    processes = {}
    for process_id in selected_process_ids:
        process = _load_single_process(
            process_id,
            [row for row in dense_rows if row[PROCESS_ID_COLUMN] == process_id],
            [row for row in event_rows if row[PROCESS_ID_COLUMN] == process_id],
        )
        is_valid, results = bp.validate.validate_process(process)
        if not is_valid:
            raise ValueError("\n".join(msg for _, msg in results))
        processes[process_id] = process

    return bp.BioProcessCollection(
        metadata={
            "name": collection_name,
            "source_example": "sim_1_intracellular",
            "description": (
                "Parsed experimental-like view of the sim 1 intracellular "
                "simulation output."
            ),
        },
        processes=processes,
    )


def _load_single_process(
    process_id: str,
    dense_rows: list[dict[str, str]],
    event_rows: list[dict[str, str]],
) -> bp.BioProcess:
    online_rows = _sort_time_rows(
        row for row in dense_rows if row["row_type"] == "online"
    )
    offline_rows = _sort_time_rows(
        row for row in dense_rows if row["row_type"] == "offline"
    )
    if not online_rows:
        raise ValueError(f"No online rows for process {process_id!r}.")
    if not offline_rows:
        raise ValueError(f"No offline rows for process {process_id!r}.")

    initial_row = online_rows[0]
    if (
        abs(float(offline_rows[0]["time"]) - float(initial_row["time"]))
        < INITIAL_TIME_MATCH_TOL
    ):
        measurement_rows = offline_rows
    else:
        # hybrax.format concentration series must start at the process start.
        # If there is no sample at t=0, use the simulator's initial state.
        measurement_rows = [initial_row, *offline_rows]
    t_measure = _times(measurement_rows)
    t_online = _times(online_rows)
    _require_strictly_increasing(t_measure, "offline plus initial times")
    _require_strictly_increasing(t_online, "online times")
    _validate_event_rows(event_rows)
    _require_offline_rows_match_sample_events(offline_rows, event_rows)
    fermentation_end_time = _fermentation_end_time(event_rows)
    if fermentation_end_time != float(t_online[-1]):
        raise ValueError(
            f"fermentation_end event does not match online end time for {process_id!r}."
        )

    reactor_components = {}
    for name in REACTOR_MEDIUM_NAMES:
        reactor_components[name] = bp.ReactorMediumComponent(
            name=name,
            unit=STATE_UNITS[name],
            concentration=bp.TimeSeries(
                times=jnp.asarray(t_measure),
                values=jnp.asarray([float(row[name]) for row in measurement_rows]),
            ),
        )

    process_variables = {
        "pH": bp.ProcessVariable(
            name="pH",
            unit="-",
            is_controlled=True,
            values=bp.TimeSeries(
                times=jnp.asarray(t_online),
                values=jnp.asarray([float(row["pH"]) for row in online_rows]),
            ),
        ),
        "temperature": bp.ProcessVariable(
            name="temperature",
            unit="degC",
            is_controlled=True,
            values=bp.TimeSeries(
                times=jnp.asarray(t_online),
                values=jnp.asarray([float(row["temperature"]) for row in online_rows]),
            ),
        ),
        "intracellular_product_ratio": bp.ProcessVariable(
            name="intracellular_product_ratio",
            unit="-",
            is_controlled=False,
            values=bp.TimeSeries(
                times=jnp.asarray(t_measure),
                values=jnp.asarray(
                    [
                        float(row["intracellular_product_ratio"])
                        for row in measurement_rows
                    ]
                ),
            ),
        ),
    }

    volume_changes = _build_volume_changes(online_rows, event_rows)
    biological_ode = _build_biological_ode()
    return bp.BioProcess(
        metadata=bp.BioProcessMetadata(
            name=process_id,
            process_type="fed_batch",
            notes="sim 1 simulated intracellular-product process",
        ),
        time_axis=bp.TimeAxis(
            unit="h",
            start=float(t_online[0]),
            end=fermentation_end_time,
            time_reference="batch_start",
        ),
        volume=bp.Volume(
            initial_volume=float(initial_row["volume"]),
            unit="L",
            volume_changes=volume_changes,
        ),
        reactor_medium=bp.ReactorMedium(
            name="reactor_medium",
            density=1.0,
            density_unit="kg/L",
            components=reactor_components,
        ),
        process_variables=process_variables,
        biological_ode=biological_ode,
    )


def _build_biological_ode() -> bp.BiologicalOde:
    """Return the sim 1 intracellular-product ODE contract."""
    return bp.BiologicalOde(
        algebraic={"X_active": "biomass - product_intracellular"},
        rates={
            "q_biomass": (None, None),
            "q_product_extracellular": (None, None),
            "q_product_intracellular": (None, None),
            "q_dead_cells": (None, None),
            "q_glucose": (None, None),
            "q_glutamine": (None, None),
            "q_lactate": (None, None),
            "q_ammonia": (None, None),
        },
        derivatives={
            "biomass": "q_biomass * X_active",
            "product_extracellular": "q_product_extracellular * X_active",
            "product_intracellular": "q_product_intracellular * X_active",
            "dead_cells": "q_dead_cells * X_active",
            "glucose": "q_glucose * X_active",
            "glutamine": "q_glutamine * X_active",
            "lactate": "q_lactate * X_active",
            "ammonia": "q_ammonia * X_active",
            "intracellular_product_ratio": (
                f"{RATIO_RELAXATION_PER_H:.12g} * "
                f"({RATIO_TARGET:.12g} - intracellular_product_ratio)"
            ),
            # Inert pseudobatch tracers: reactor-medium components with no biological
            # dynamics (validate_biological_ode requires every component to have a
            # derivative; "0" declares none). Their dilution/feed is handled by the
            # volume-change machinery, like every other species.
            "tracer_unfed": "0",
            "tracer_fed": "0",
        },
    )


def _require_offline_rows_match_sample_events(
    offline_rows: list[dict[str, str]],
    event_rows: list[dict[str, str]],
) -> None:
    offline_times = _times(offline_rows)
    sample_and_end_rows = _sort_time_rows(
        row for row in event_rows if row["event_type"] in {"sample", "fermentation_end"}
    )
    sample_and_end_times = _times(sample_and_end_rows)
    if offline_times != sample_and_end_times:
        raise ValueError(
            "Offline measurement times must match sample and fermentation-end "
            "event times exactly."
        )


def _fermentation_end_time(event_rows: list[dict[str, str]]) -> float:
    fermentation_end_rows = [
        row for row in event_rows if row["event_type"] == "fermentation_end"
    ]
    if len(fermentation_end_rows) != 1:
        raise ValueError("Each process must have exactly one fermentation_end event.")
    return float(fermentation_end_rows[0]["time"])


def _build_volume_changes(
    online_rows: list[dict[str, str]],
    event_rows: list[dict[str, str]],
) -> dict[str, bp.VolumeChange]:
    t_online = _times(online_rows)
    bolus_rows = _sort_time_rows(
        row for row in event_rows if row["event_type"] == "bolus"
    )
    sample_rows = _sort_time_rows(
        row for row in event_rows if row["event_type"] == "sample"
    )
    volume_changes: dict[str, bp.VolumeChange] = {
        "conti_feed": bp.Inflow(
            name="conti_feed",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            feed_medium=_feed_medium(
                "conti_feed_medium",
                {
                    "glucose": FEED_GLUCOSE,
                    "glutamine": FEED_GLUTAMINE,
                    # The continuous stream physically carries the fed inert tracer
                    # (integrated in _rhs_numpy); record it so the JSON feed medium
                    # matches the simulation. tracer_unfed is never fed -> 0.
                    "tracer_fed": TRACER_FED_FEED_CONCENTRATION,
                },
            ),
            values=bp.TimeSeries(
                times=jnp.asarray(t_online),
                values=jnp.asarray(
                    [float(row["cum_conti_feed"]) for row in online_rows]
                ),
            ),
        ),
        "base_feed": bp.Inflow(
            name="base_feed",
            unit="L",
            is_controlled=False,
            is_continuous=True,
            feed_medium=_feed_medium("base_feed_medium", {}),
            values=bp.TimeSeries(
                times=jnp.asarray(t_online),
                values=jnp.asarray(
                    [float(row["cum_base_feed"]) for row in online_rows]
                ),
            ),
        ),
    }
    if sample_rows:
        volume_changes["sampling"] = bp.Outflow(
            name="sampling",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=bp.TimeSeries(
                times=jnp.asarray(_times(sample_rows)),
                values=jnp.asarray([float(row["delta_volume"]) for row in sample_rows]),
            ),
        )
    bolus_groups = _bolus_rows_by_feed_id(bolus_rows)
    for feed_id, rows in bolus_groups.items():
        change_name = _bolus_change_name(feed_id, single_group=len(bolus_groups) == 1)
        volume_changes[change_name] = bp.Inflow(
            name=change_name,
            unit="L",
            is_controlled=True,
            is_continuous=False,
            feed_medium=_feed_medium(feed_id, _bolus_feed_concentrations(rows)),
            values=bp.TimeSeries(
                times=jnp.asarray(_times(rows)),
                values=jnp.asarray([float(row["delta_volume"]) for row in rows]),
            ),
        )
    return volume_changes


def _feed_medium(name: str, concentrations: dict[str, float]) -> bp.FeedMedium:
    components = {}
    for state_name in REACTOR_MEDIUM_NAMES:
        components[state_name] = bp.FeedMediumComponent(
            name=state_name,
            unit=STATE_UNITS[state_name],
            concentration=bp.StaticVariable(
                value=float(concentrations.get(state_name, 0.0))
            ),
            is_controlled=state_name in {"glucose", "glutamine"},
        )
    return bp.FeedMedium(
        name=name,
        density=1.0,
        density_unit="kg/L",
        components=components,
    )


def _bolus_feed_concentrations(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {}
    first = {name: float(rows[0][f"feed_{name}"]) for name in REACTOR_MEDIUM_NAMES}
    for row in rows[1:]:
        current = {name: float(row[f"feed_{name}"]) for name in REACTOR_MEDIUM_NAMES}
        if current != first:
            raise ValueError("bolus feed composition must be constant per process.")
    return first


def _bolus_rows_by_feed_id(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        feed_id = row["feed_id"] or "bolus_feed_medium"
        groups.setdefault(feed_id, []).append(row)
    return groups


def _bolus_change_name(feed_id: str, *, single_group: bool) -> str:
    if single_group:
        return "bolus_feed"
    suffix = re.sub(r"[^0-9A-Za-z_]+", "_", feed_id).strip("_")
    suffix = suffix.removeprefix("bolus_feed_")
    return f"bolus_feed_{suffix or 'medium'}"


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header.")
        return reader.fieldnames, list(reader)


def _require_columns(fieldnames: list[str], columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in fieldnames]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _sort_time_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: float(row["time"]))


def _times(rows: list[dict[str, str]]) -> list[float]:
    return [float(row["time"]) for row in rows]


def _require_strictly_increasing(times: list[float], label: str) -> None:
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError(f"{label} must be strictly increasing.")


def _validate_event_rows(rows: list[dict[str, str]]) -> None:
    allowed = {"sample", "bolus", "fermentation_end"}
    groups: dict[tuple[str, float], list[dict[str, str]]] = {}
    for row in rows:
        if row["event_type"] not in allowed:
            raise ValueError(f"Unexpected event_type: {row['event_type']!r}")
        delta_volume = float(row["delta_volume"])
        if row["event_type"] == "sample" and delta_volume >= 0.0:
            raise ValueError("sample events must have negative delta_volume.")
        if row["event_type"] == "bolus" and delta_volume <= 0.0:
            raise ValueError("bolus events must have positive delta_volume.")
        if row["event_type"] == "fermentation_end" and delta_volume != 0.0:
            raise ValueError("fermentation_end events must have zero delta_volume.")
        key = (row[PROCESS_ID_COLUMN], float(row["time"]))
        groups.setdefault(key, []).append(row)

    for key, group in groups.items():
        ordered = sorted(group, key=lambda row: int(row["event_order"]))
        event_types = [row["event_type"] for row in ordered]
        if event_types not in (
            ["sample"],
            ["bolus"],
            ["sample", "bolus"],
            ["fermentation_end"],
        ):
            raise ValueError(f"Invalid event order at {key}: {event_types}.")
        if [int(row["event_order"]) for row in ordered] != list(range(len(group))):
            raise ValueError(f"Invalid event_order sequence at {key}.")
