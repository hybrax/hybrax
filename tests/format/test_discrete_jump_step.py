"""
Tests verifying that ADF and feed-term produce correct step behaviour
during the pseudo-batch backtransform via SplineRepresentation.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from bpbench import (
    BioProcess,
    BioProcessMetadata,
    FeedMedium,
    FeedMediumComponent,
    FeedVolumeChange,
    ReactorMedium,
    ReactorMediumComponent,
    SampleVolumeChange,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)
from bpbench.splines import (
    build_pseudobatch_inputs,
    build_splines,
    to_spline_representation,
    build_backtransform_spline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(t, v):
    return TimeSeries(
        timepoints=jnp.array(t, dtype=float),
        values=jnp.array(v, dtype=float),
    )


def _make_bolus_process(feed_time=10.0, delta_v=0.2, c_feed=500.0):
    """Minimal process with a single bolus feed event."""
    feed_medium = FeedMedium(
        name="feed", density=1.0, density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose", unit="g/L",
                concentration=StaticVariable(value=c_feed),
                is_controlled=True,
            ),
        },
    )
    rm = ReactorMedium(
        name="medium", density=1.0, density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose", unit="g/L",
                concentration=_ts(
                    [0.0, 5.0, 10.0, 15.0, 20.0],
                    [10.0, 8.0, 6.0, 5.0, 4.0],
                ),
                is_intracellular=False,
            ),
        },
    )
    vol = Volume(
        initial_volume=1.0, unit="L",
        volume_changes={
            "bolus": FeedVolumeChange(
                name="bolus", unit="L",
                is_controlled=True, is_continuous=False,
                feed_medium=feed_medium,
                values=_ts([feed_time], [delta_v]),
            ),
        },
    )
    return BioProcess(
        metadata=BioProcessMetadata(name="test", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=20.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_step_jump_at_bolus():
    """Backtransformed concentration should have a jump at the bolus feed time."""
    proc = _make_bolus_process(feed_time=10.0, delta_v=0.2, c_feed=500.0)
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_spline_representation(inputs, splines, "glucose")

    bt = build_backtransform_spline(rep)

    # Use eps > _EPS (1e-4) to cross the dense grid's pre-event epsilon point
    eps = 5e-4
    val_before = float(bt(jnp.array(10.0 - eps)))
    val_after = float(bt(jnp.array(10.0 + eps)))

    jump = abs(val_after - val_before)
    assert jump > 0.1, f"Expected discontinuity at bolus time, got jump={jump}"


def test_step_consistent_at_different_distances():
    """Value at t_event + small_eps and t_event + larger_eps should be similar
    (step function, not linear ramp)."""
    proc = _make_bolus_process(feed_time=10.0, delta_v=0.2, c_feed=500.0)
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_spline_representation(inputs, splines, "glucose")

    bt = build_backtransform_spline(rep)
    val_close = float(bt(jnp.array(10.0 + 5e-4)))
    val_far = float(bt(jnp.array(10.0 + 0.1)))

    # Both should be similar (within spline interpolation tolerance).
    # Note: with linear ADF interpolation on the dense grid, the sharp ramp
    # spans ~1e-4 time units, so values at 5e-4 and 0.1 may differ due to
    # spline curvature, but should be in the same ballpark.
    assert abs(val_close - val_far) < 2.0, (
        f"Step function should give roughly consistent values after event: "
        f"close={val_close}, far={val_far}"
    )


def test_no_jump_for_sampling():
    """Sampling events should NOT produce concentration jumps."""
    rm = ReactorMedium(
        name="medium", density=1.0, density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose", unit="g/L",
                concentration=_ts(
                    [0.0, 5.0, 10.0, 15.0, 20.0],
                    [10.0, 8.0, 6.0, 5.0, 4.0],
                ),
                is_intracellular=False,
            ),
        },
    )
    vol = Volume(
        initial_volume=1.0, unit="L",
        volume_changes={
            "sample": SampleVolumeChange(
                name="sample", unit="L",
                is_controlled=True, is_continuous=False,
                values=_ts([10.0], [-0.05]),
            ),
        },
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="test", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=20.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )

    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_spline_representation(inputs, splines, "glucose")

    bt = build_backtransform_spline(rep)

    eps = 1e-6
    val_before = float(bt(jnp.array(10.0 - eps)))
    val_after = float(bt(jnp.array(10.0 + eps)))

    # Should be smooth across sampling (no jump)
    assert abs(val_after - val_before) < 0.5, (
        f"Sampling should not cause a jump: before={val_before}, after={val_after}"
    )
