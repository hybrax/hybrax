from __future__ import annotations

from typing import Iterable

from bpbench import validate_process
from bpbench.dataclasses import BioProcessCollection


def validate_raw_collection(
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
