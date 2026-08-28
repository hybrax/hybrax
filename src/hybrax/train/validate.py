"""hybrax.train's own training/prepare-readiness checks, layered on hybrax.format's."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from hybrax.format import validate_cross_process_consistency, validate_process
from hybrax.format.dataclasses import (
    BioProcessCollection,
    FeedMediumComponent,
    Inflow,
    ReactorMediumComponent,
    StaticVariable,
    TimeSeries,
)


def _check_result(verdict: str, check_name: str, detail: str) -> tuple[bool, str]:
    """Return a result using hybrax.format's validation message convention."""
    ok = verdict != "FAIL"
    return ok, f"{verdict} {check_name}: {detail}"


def validate_for_training(
    collection: BioProcessCollection,
    *,
    strict: bool = False,
    require_reaction_ode: bool = False,
) -> dict[str, dict[str, object]]:
    """hybrax.train's training-readiness validator.

    Distinct from hybrax.format's own ``validate_for_publication`` (a
    storage/publication concern), but composes the same
    ``validate_cross_process_consistency`` structural check rather than
    duplicating it — training data is expected to come from one coherent
    case study by default.

    Args:
        collection: Process collection to validate.
        strict: Raise instead of returning a report containing failures.
        require_reaction_ode: Also fail any process missing
            ``reaction_ode`` (checked after ``transform_process_collection``
            would have added one).

    Returns:
        One report entry per process name (plus ``"__consistency__"`` for the
        cross-process check): ``ok`` and the list of ``(passed, message)``
        check results.

    Raises:
        ValueError: If ``strict`` is set and any process or the
            cross-process check fails.
    """
    report: dict[str, dict[str, object]] = {}

    for process_name, process in collection.processes.items():
        ok, results = validate_process(process)
        process_results = list(results)
        if require_reaction_ode and process.reaction_ode is None:
            ok = False
            process_results.append(
                _check_result(
                    "FAIL",
                    "reaction_ode_required",
                    "reaction_ode is missing after transform_process_collection",
                )
            )
        report[process_name] = {
            "ok": bool(ok),
            "messages": process_results,
        }

    consistency_ok, consistency_results = validate_cross_process_consistency(collection)
    report["__consistency__"] = {
        "ok": consistency_ok,
        "messages": consistency_results,
    }

    if strict:
        errors = [
            f"{process_name}: "
            + "; ".join(message for ok, message in entry["messages"] if not ok)
            for process_name, entry in report.items()
            if not entry["ok"]
        ]
        if errors:
            raise ValueError("hybrax.format validation failed:\n" + "\n".join(errors))

    return report


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
    """Raise if any config-declared required control is missing from a process.

    Args:
        process_name: Process name, used only in the error message.
        available_control_names: Control names the process actually has.
        required_control_names: Control names ``prepare.required_control_names``
            declares as required for this process.

    Raises:
        ValueError: If any name in ``required_control_names`` is not in
            ``available_control_names``.
    """
    available = set(available_control_names)
    missing = [name for name in required_control_names if name not in available]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"{process_name}: config-declared controls are missing: {missing_str}"
        )


def summarize_process_semantics(process) -> dict[str, object]:
    """Snapshot one process's reactor/feed component structure for provenance diffing.

    Used by :func:`~hybrax.train.prepare.prepare_artifact` to compare a
    process's semantics before and after the ``transform_process_collection``
    hook, and by :func:`ensure_prepared_training_semantics` to check the
    prepared result.

    Args:
        process: Process to summarize.

    Returns:
        A dict with ``reactor_component_names``, ``reactor_component_details``,
        ``has_biomass``, ``feed_component_names_by_change``,
        ``feed_component_details_by_change``, ``feed_medium_present_by_change``,
        ``all_feed_changes``, and ``positive_feed_changes``.
    """
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
        if not isinstance(volume_change, Inflow):
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
    """Check every process still has reactor/biomass/feed-medium semantics after prep.

    Runs after ``transform_process_collection`` and augmentation, to catch a
    hook that silently dropped required structure (an empty
    ``reactor_medium.components``, a missing biomass component, or a feed with
    no ``feed_medium``/component metadata).

    Args:
        collection: Prepared collection to check.

    Returns:
        One report entry per process: ``ok``, the list of ``(passed, message)``
        check results, and the :func:`summarize_process_semantics` snapshot.

    Raises:
        ValueError: If any process fails one of these checks.
    """
    report: dict[str, dict[str, object]] = {}
    errors: list[str] = []

    for process_name, process in collection.processes.items():
        summary = summarize_process_semantics(process)
        results: list[tuple[bool, str]] = []

        if summary["reactor_component_names"]:
            results.append(
                _check_result(
                    "PASS",
                    "reactor_medium_components_present",
                    f"{len(summary['reactor_component_names'])} reactor-medium "
                    "component(s) present after prep",
                )
            )
        else:
            results.append(
                _check_result(
                    "FAIL",
                    "reactor_medium_components_present",
                    "reactor_medium.components is empty after prep",
                )
            )

        if summary["has_biomass"]:
            results.append(
                _check_result(
                    "PASS",
                    "biomass_component_present",
                    "reactor medium defines a biomass component after prep",
                )
            )
        else:
            results.append(
                _check_result(
                    "FAIL",
                    "biomass_component_present",
                    "reactor medium does not define a biomass component after prep",
                )
            )

        for change_name in summary["all_feed_changes"]:
            if not summary["feed_medium_present_by_change"].get(change_name, False):
                results.append(
                    _check_result(
                        "FAIL",
                        "feed_medium_populated",
                        f"feed {change_name!r} has no feed_medium after prep",
                    )
                )
                continue
            if not summary["feed_component_names_by_change"].get(change_name):
                results.append(
                    _check_result(
                        "FAIL",
                        "feed_medium_populated",
                        f"feed {change_name!r} has no feed-medium component "
                        "metadata after prep",
                    )
                )
                continue
            results.append(
                _check_result(
                    "PASS",
                    "feed_medium_populated",
                    f"feed {change_name!r} has feed-medium component "
                    "metadata after prep",
                )
            )

        ok = all(check_ok for check_ok, _ in results)
        report[process_name] = {
            "ok": ok,
            "messages": results,
            "summary": summary,
        }
        if not ok:
            errors.append(
                f"{process_name}: "
                + "; ".join(message for check_ok, message in results if not check_ok)
            )

    if errors:
        raise ValueError("prepared semantics validation failed:\n" + "\n".join(errors))

    return report
