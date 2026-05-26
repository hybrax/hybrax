"""Stable runtime regression tests for canonical TimeSeries usage."""

from __future__ import annotations

import ast
from pathlib import Path

import jax.numpy as jnp

from bp_format import (
    BioProcess,
    BioProcessMetadata,
    FeedMedium,
    FeedMediumComponent,
    FeedVolumeChange,
    ProcessVariable,
    ReactorMedium,
    ReactorMediumComponent,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)
from bp_format.inspect import print_process_structure
from bp_format.mechanistic import get_control_splines
from bp_format.splines import build_pseudobatch_inputs


def _legacy_timepoints_usages(path: Path) -> list[str]:
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

        if isinstance(node, ast.Call):
            is_timeseries_ctor = (
                isinstance(node.func, ast.Name) and node.func.id == "TimeSeries"
            ) or (
                isinstance(node.func, ast.Attribute) and node.func.attr == "TimeSeries"
            )
            if is_timeseries_ctor:
                for keyword in node.keywords:
                    if keyword.arg == "timepoints":
                        findings.append(
                            f"TimeSeries(timepoints=...) keyword at line {node.lineno}"
                        )
                        break

    return findings


def test_runtime_modules_use_canonical_times_api() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    runtime_files = [
        repo_root / "bp_format" / "validate.py",
        repo_root / "bp_format" / "splines.py",
        repo_root / "bp_format" / "mechanistic.py",
        repo_root / "bp_format" / "inspect.py",
    ]

    for path in runtime_files:
        findings = _legacy_timepoints_usages(path)
        assert not findings, (
            f"legacy timepoints usage still present in {path}: {findings}"
        )


def _linear_cumulative_spline(end_time: float, end_value: float) -> TimeSeries:
    slope = end_value / end_time
    return TimeSeries(
        breaks=jnp.array([0.0, end_time]),
        coeffs=jnp.array([[0.0, slope, 0.0, 0.0]]),
        segment_start_piece_idx=jnp.array([0]),
    )


def _build_process_with_spline_only_feed() -> BioProcess:
    biomass_ts = TimeSeries(
        times=jnp.array([0.0, 5.0, 10.0]),
        values=jnp.array([0.2, 0.8, 1.6]),
    )
    feed_cum_ts = _linear_cumulative_spline(end_time=10.0, end_value=1.0)

    feed_medium = FeedMedium(
        name="feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": FeedMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=StaticVariable(value=0.0),
                is_controlled=True,
            )
        },
    )

    return BioProcess(
        metadata=BioProcessMetadata(name="rt", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=10.0, time_reference="t0"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "feed": FeedVolumeChange(
                    name="feed",
                    unit="L",
                    is_controlled=True,
                    is_continuous=True,
                    values=feed_cum_ts,
                    feed_medium=feed_medium,
                )
            },
        ),
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=biomass_ts,
                )
            },
        ),
        process_variables={
            "pH": ProcessVariable(
                name="pH",
                unit="",
                is_controlled=True,
                values=TimeSeries(
                    times=jnp.array([0.0, 10.0]),
                    values=jnp.array([7.0, 7.0]),
                ),
            )
        },
    )


def _build_process_with_discrete_continuous_feed() -> BioProcess:
    biomass_ts = TimeSeries(
        times=jnp.array([0.0, 5.0, 10.0]),
        values=jnp.array([0.2, 0.8, 1.6]),
    )
    feed_cum_ts = TimeSeries(
        times=jnp.array([0.0, 5.0, 10.0]),
        values=jnp.array([0.0, 0.5, 1.0]),
    )

    feed_medium = FeedMedium(
        name="feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": FeedMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=StaticVariable(value=0.0),
                is_controlled=True,
            )
        },
    )

    return BioProcess(
        metadata=BioProcessMetadata(name="rt2", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=10.0, time_reference="t0"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "feed": FeedVolumeChange(
                    name="feed",
                    unit="L",
                    is_controlled=True,
                    is_continuous=True,
                    values=feed_cum_ts,
                    feed_medium=feed_medium,
                )
            },
        ),
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=biomass_ts,
                )
            },
        ),
        process_variables={},
    )


def test_pseudobatch_uses_spline_evaluation_for_feed_series() -> None:
    process = _build_process_with_spline_only_feed()
    inputs = build_pseudobatch_inputs(process, "biomass")

    dense_times = inputs["dense_times"]
    accumulated_feed = inputs["accumulated_feed_dense"]

    assert dense_times.shape[0] >= 3
    assert accumulated_feed.shape == dense_times.shape
    assert float(accumulated_feed[0]) == 0.0
    assert jnp.isclose(accumulated_feed[-1], 1.0, atol=1e-6)


def test_inspect_prints_integral_for_continuous_spline_series(capsys) -> None:
    process = _build_process_with_spline_only_feed()

    print_process_structure(process, verbosity=3)
    out = capsys.readouterr().out

    assert "Series integral over span" in out


def test_inspect_does_not_integrate_discrete_only_series(capsys) -> None:
    process = _build_process_with_discrete_continuous_feed()

    print_process_structure(process, verbosity=3)
    out = capsys.readouterr().out

    assert "Series integral over span" not in out


def test_mechanistic_control_splines_smoke_for_canonical_timeseries() -> None:
    process = BioProcess(
        metadata=BioProcessMetadata(name="mech-smoke", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=10.0, time_reference="t0"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "feed": FeedVolumeChange(
                    name="feed",
                    unit="L",
                    is_controlled=True,
                    is_continuous=True,
                    values=TimeSeries(
                        times=jnp.array([0.0, 5.0, 10.0]),
                        values=jnp.array([0.0, 0.2, 0.5]),
                    ),
                    feed_medium=FeedMedium(
                        name="feed",
                        density=1.0,
                        density_unit="kg/L",
                        components={
                            "biomass": FeedMediumComponent(
                                name="biomass",
                                unit="g/L",
                                concentration=StaticVariable(value=0.0),
                                is_controlled=True,
                            )
                        },
                    ),
                )
            },
        ),
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.array([0.0, 5.0, 10.0]),
                        values=jnp.array([0.2, 0.8, 1.6]),
                    ),
                )
            },
        ),
        process_variables={
            "pH": ProcessVariable(
                name="pH",
                unit="",
                is_controlled=True,
                values=TimeSeries(
                    times=jnp.array([0.0, 10.0]),
                    values=jnp.array([7.0, 7.0]),
                ),
            )
        },
    )
    control = get_control_splines(process)
    values = control(jnp.array(5.0))
    assert control.name_controlled_FVCs == ("feed",)
    assert control.name_controlled_PVs == ("pH",)
    assert values.shape == (2,)
