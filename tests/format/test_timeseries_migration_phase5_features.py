"""Phase 5 tests for rich TimeSeries feature usage in bpbench workflows."""

from __future__ import annotations

import jax.numpy as jnp

from bpbench import (
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
from bpbench.inspect import print_process_structure
from bpbench.splines import build_pseudobatch_inputs


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
        metadata=BioProcessMetadata(name="phase5", process_type="fed_batch"),
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
                    is_intracellular=False,
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
        metadata=BioProcessMetadata(name="phase5-discrete", process_type="fed_batch"),
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
                    is_intracellular=False,
                )
            },
        ),
        process_variables={},
    )


def test_phase5_pseudobatch_uses_spline_evaluation_for_feed_series() -> None:
    process = _build_process_with_spline_only_feed()
    inputs = build_pseudobatch_inputs(process, "biomass")

    dense_times = inputs["dense_times"]
    accumulated_feed = inputs["accumulated_feed_dense"]

    assert dense_times.shape[0] >= 3
    assert accumulated_feed.shape == dense_times.shape
    assert float(accumulated_feed[0]) == 0.0
    assert jnp.isclose(accumulated_feed[-1], 1.0, atol=1e-6)


def test_phase5_inspect_prints_integral_for_continuous_spline_series(capsys) -> None:
    process = _build_process_with_spline_only_feed()

    print_process_structure(process, verbosity=3)
    out = capsys.readouterr().out

    assert "Series integral over span" in out


def test_phase5_inspect_does_not_integrate_discrete_only_series(capsys) -> None:
    process = _build_process_with_discrete_continuous_feed()

    print_process_structure(process, verbosity=3)
    out = capsys.readouterr().out

    assert "Series integral over span" not in out
