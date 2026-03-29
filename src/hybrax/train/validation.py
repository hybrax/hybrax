from __future__ import annotations

from typing import Iterable

import numpy as np
from bpbench import validate_process
from bpbench.dataclasses import (
    BioProcessCollection,
    FeedMediumComponent,
    FeedVolumeChange,
    ReactorMediumComponent,
    StaticVariable,
    TimeSeries,
)


def validate_collection(
    collection: BioProcessCollection,
    *,
    strict: bool = False,
) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}

    for process_name, process in collection.processes.items():
        ok, messages = validate_process(process)
        report[process_name] = {
            "ok": bool(ok),
            "messages": list(messages),
        }

    if strict:
        errors = [
            f"{process_name}: {'; '.join(entry['messages'])}"
            for process_name, entry in report.items()
            if not entry["ok"]
        ]
        if errors:
            raise ValueError("bpbench validation failed:\n" + "\n".join(errors))

    return report


def validate_raw_collection(
    collection: BioProcessCollection,
    *,
    strict: bool = False,
) -> dict[str, dict[str, object]]:
    return validate_collection(collection, strict=strict)


def _serialize_concentration(value: TimeSeries | StaticVariable) -> dict[str, object]:
    if isinstance(value, StaticVariable):
        return {
            "kind": "static",
            "value": float(value.value),
        }
    return {
        "kind": "timeseries",
        "times": np.asarray(value.times, dtype=float).tolist(),
        "values": np.asarray(value.values, dtype=float).tolist(),
    }


def _serialize_reactor_component(
    component: ReactorMediumComponent,
) -> dict[str, object]:
    return {
        "name": component.name,
        "unit": component.unit,
        "is_intracellular": bool(component.is_intracellular),
        "concentration": _serialize_concentration(component.concentration),
    }


def _serialize_feed_component(component: FeedMediumComponent) -> dict[str, object]:
    return {
        "name": component.name,
        "unit": component.unit,
        "is_controlled": bool(component.is_controlled),
        "concentration": _serialize_concentration(component.concentration),
    }


def ensure_required_controls(
    process_name: str,
    available_control_names: Iterable[str],
    required_control_names: Iterable[str],
) -> None:
    available = set(available_control_names)
    missing = [name for name in required_control_names if name not in available]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"{process_name}: config-declared controls are missing: {missing_str}"
        )


def summarize_process_semantics(process) -> dict[str, object]:
    reactor_component_details = {
        name: _serialize_reactor_component(component)
        for name, component in sorted((process.reactor_medium.components or {}).items())
    }
    reactor_components = list(reactor_component_details.keys())
    has_biomass = any(name.strip().lower() == "biomass" for name in reactor_components)

    feed_component_names_by_change: dict[str, list[str]] = {}
    feed_component_details_by_change: dict[str, dict[str, dict[str, object]]] = {}
    feed_medium_present_by_change: dict[str, bool] = {}
    all_feed_changes: list[str] = []
    positive_feed_changes: list[str] = []

    for change_name, volume_change in process.volume.volume_changes.items():
        if not isinstance(volume_change, FeedVolumeChange):
            continue

        all_feed_changes.append(change_name)
        feed_medium_present_by_change[change_name] = (
            volume_change.feed_medium is not None
        )
        if (
            volume_change.feed_medium is None
            or not volume_change.feed_medium.components
        ):
            feed_component_names_by_change[change_name] = []
            feed_component_details_by_change[change_name] = {}
        else:
            component_details = {
                name: _serialize_feed_component(component)
                for name, component in sorted(
                    volume_change.feed_medium.components.items()
                )
            }
            feed_component_details_by_change[change_name] = component_details
            feed_component_names_by_change[change_name] = list(component_details.keys())

        values = np.asarray(volume_change.values.values, dtype=float)
        all_non_negative = bool(np.all(values >= 0.0))
        has_positive = bool(np.any(values > 0.0))
        if all_non_negative and has_positive:
            positive_feed_changes.append(change_name)

    return {
        "reactor_component_names": reactor_components,
        "reactor_component_details": reactor_component_details,
        "has_biomass": has_biomass,
        "feed_component_names_by_change": feed_component_names_by_change,
        "feed_component_details_by_change": feed_component_details_by_change,
        "feed_medium_present_by_change": feed_medium_present_by_change,
        "all_feed_changes": sorted(all_feed_changes),
        "positive_feed_changes": sorted(positive_feed_changes),
    }


def ensure_prepared_training_semantics(
    collection: BioProcessCollection,
) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    errors: list[str] = []

    for process_name, process in collection.processes.items():
        summary = summarize_process_semantics(process)
        process_errors: list[str] = []

        if not summary["reactor_component_names"]:
            process_errors.append("reactor_medium.components is empty after prep")

        if not summary["has_biomass"]:
            process_errors.append(
                "reactor medium does not define a biomass component after prep"
            )

        for change_name in summary["all_feed_changes"]:
            if not summary["feed_medium_present_by_change"].get(change_name, False):
                process_errors.append(
                    f"feed '{change_name}' has no feed_medium after prep"
                )
                continue
            if not summary["feed_component_names_by_change"].get(change_name):
                process_errors.append(
                    f"feed '{change_name}' has no "
                    "feed-medium component metadata after prep"
                )

        report[process_name] = {
            "ok": not process_errors,
            "messages": process_errors,
            "summary": summary,
        }
        if process_errors:
            errors.append(f"{process_name}: {'; '.join(process_errors)}")

    if errors:
        raise ValueError("prepared semantics validation failed:\n" + "\n".join(errors))

    return report
