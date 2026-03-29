"""Phase 4 checks for canonical internal TimeSeries API usage."""

from __future__ import annotations

import ast
from pathlib import Path

import jax.numpy as jnp

from bpbench import (
    BioProcess,
    BioProcessMetadata,
    ReactorMedium,
    ReactorMediumComponent,
    TimeAxis,
    TimeSeries,
    Volume,
)
from bpbench.inspect import _collect_process_panels
from bpbench.validate import validate_timeseries_shape


def _legacy_timepoints_usages(path: Path) -> list[str]:
    """Find real code-level legacy timepoints usages in one module."""
    tree = ast.parse(path.read_text())
    findings: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "timepoints":
            findings.append(f"attribute access at line {node.lineno}")

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "hasattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "timepoints"
        ):
            findings.append(f"hasattr(..., 'timepoints') at line {node.lineno}")

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "timepoints"
        ):
            findings.append(f"getattr(..., 'timepoints') at line {node.lineno}")

        if isinstance(node, ast.Call) and (
            (isinstance(node.func, ast.Name) and node.func.id == "TimeSeries")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "TimeSeries")
        ):
            for keyword in node.keywords:
                if keyword.arg == "timepoints":
                    findings.append(
                        f"TimeSeries(timepoints=...) keyword at line {node.lineno}"
                    )
                    break

    return findings


def test_phase4_runtime_modules_use_canonical_times_api() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runtime_files = [
        repo_root / "bpbench" / "validate.py",
        repo_root / "bpbench" / "splines.py",
        repo_root / "bpbench" / "mechanistic.py",
        repo_root / "bpbench" / "inspect.py",
    ]

    for path in runtime_files:
        findings = _legacy_timepoints_usages(path)
        assert not findings, (
            f"legacy timepoints usage still present in {path}: {findings}"
        )


def _build_spline_only_series() -> TimeSeries:
    return TimeSeries(
        breaks=jnp.array([0.0, 1.0, 2.0]),
        coeffs=jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
            ]
        ),
        segment_start_piece_idx=jnp.array([0]),
    )


def test_phase4_validate_timeseries_shape_handles_spline_only_gracefully() -> None:
    spline_only = _build_spline_only_series()
    ok, msg = validate_timeseries_shape(spline_only, "spline_only")
    assert ok is False
    assert "missing discrete times/values arrays" in msg


def test_phase4_collect_process_panels_handles_spline_only_series() -> None:
    process = BioProcess(
        metadata=BioProcessMetadata(name="phase4", process_type="batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="t0"),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes={}),
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "x": ReactorMediumComponent(
                    name="x",
                    unit="g/L",
                    concentration=_build_spline_only_series(),
                    is_intracellular=False,
                )
            },
        ),
        process_variables={},
    )

    panels = _collect_process_panels(process)
    assert len(panels) == 1
    assert panels[0]["type"] == "dynamic"
    assert panels[0]["x"].shape[0] == 3
    assert panels[0]["y"].shape[0] == 3
