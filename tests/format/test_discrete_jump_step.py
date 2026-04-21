"""
Tests verifying that ADF and feed-term produce correct step behaviour
during the pseudo-batch backtransform via Interpolator.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from bp_format import (
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
from bp_format.splines import (
    build_pseudobatch_inputs,
    build_splines,
    to_interpolator,
    build_backtransform_spline,
    evaluate_left_continuous_step,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(t, v):
    return TimeSeries(
        times=jnp.array(t, dtype=float),
        values=jnp.array(v, dtype=float),
    )


def _make_bolus_process(feed_time=10.0, delta_v=0.2, c_feed=500.0):
    """Minimal process with a single bolus feed event."""
    feed_medium = FeedMedium(
        name="feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=StaticVariable(value=c_feed),
                is_controlled=True,
            ),
        },
    )
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=_ts(
                    [0.0, 5.0, 10.0, 15.0, 20.0],
                    [10.0, 8.0, 6.0, 5.0, 4.0],
                ),
                is_intracellular=False,
            ),
        },
    )
    vol = Volume(
        initial_volume=1.0,
        unit="L",
        volume_changes={
            "bolus": FeedVolumeChange(
                name="bolus",
                unit="L",
                is_controlled=True,
                is_continuous=False,
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
    """Backtransform has a jump around bolus feed time."""
    proc = _make_bolus_process(feed_time=10.0, delta_v=0.2, c_feed=500.0)
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_interpolator(inputs, splines, "glucose")

    bt = build_backtransform_spline(rep)

    t_b = 10.0
    post_probe = 5e-4
    pre_probe = 5e-4

    val_before = float(bt(jnp.array(t_b - pre_probe)))
    val_at = float(bt(jnp.array(t_b)))
    val_after = float(bt(jnp.array(t_b + post_probe)))

    assert val_at == pytest.approx(val_before, abs=2e-2)

    jump = abs(val_after - val_at)
    assert jump > 0.1, f"Expected discontinuity at bolus time, got jump={jump}"


def test_adf_is_instantaneous_at_bolus():
    """ADF should jump immediately after event (left-continuous at t_b)."""
    proc = _make_bolus_process(feed_time=10.0, delta_v=0.2, c_feed=500.0)
    inputs = build_pseudobatch_inputs(proc, "glucose")
    splines = build_splines(inputs, proc, "glucose")
    rep = to_interpolator(inputs, splines, "glucose")

    t_b = 10.0
    # > float32 resolution near t=10, but still << _EPS (1e-4)
    tiny_delta = 1e-5
    tr = rep.interpolator_metadata["transform"]
    adf_t = jnp.asarray(tr["adf_times"], dtype=float)
    adf_v = jnp.asarray(tr["adf_values"], dtype=float)

    adf_at = float(evaluate_left_continuous_step(jnp.array(t_b), adf_t, adf_v))
    adf_after_tiny = float(
        evaluate_left_continuous_step(jnp.array(t_b + tiny_delta), adf_t, adf_v)
    )
    adf_after_far = float(
        evaluate_left_continuous_step(jnp.array(t_b + 5e-5), adf_t, adf_v)
    )

    assert adf_after_tiny > adf_at
    assert adf_after_tiny == pytest.approx(adf_after_far, abs=1e-10)


def test_no_jump_for_sampling():
    """Sampling events should NOT produce concentration jumps."""
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=_ts(
                    [0.0, 5.0, 10.0, 15.0, 20.0],
                    [10.0, 8.0, 6.0, 5.0, 4.0],
                ),
                is_intracellular=False,
            ),
        },
    )
    vol = Volume(
        initial_volume=1.0,
        unit="L",
        volume_changes={
            "sample": SampleVolumeChange(
                name="sample",
                unit="L",
                is_controlled=True,
                is_continuous=False,
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
    rep = to_interpolator(inputs, splines, "glucose")

    bt = build_backtransform_spline(rep)

    t_s = 10.0
    tiny_delta = 1e-6
    pre_probe = 5e-4  # safely away from event edge
    val_before = float(bt(jnp.array(t_s - pre_probe)))
    val_at = float(bt(jnp.array(t_s)))
    val_after = float(bt(jnp.array(t_s + tiny_delta)))

    assert val_at == pytest.approx(val_before, abs=2e-2)
    assert abs(val_after - val_at) < 2e-2, (
        f"Sampling should be continuous: at={val_at}, after={val_after}"
    )
