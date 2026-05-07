"""
Tests verifying that ADF and feed-term produce correct step behaviour
during the pseudo-batch backtransform via transformed TimeSeries carriers.
"""

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
    build_pseudobatch_transform,
    build_backtransform_spline,
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


def _build_backtransform(proc, species="glucose"):
    transform = build_pseudobatch_transform(proc, [species])
    return build_backtransform_spline(transform, species)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_step_jump_at_bolus():
    """Backtransform has a jump around bolus feed time."""
    proc = _make_bolus_process(feed_time=10.0, delta_v=0.2, c_feed=500.0)
    bt = _build_backtransform(proc)

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
    transform = build_pseudobatch_transform(proc, ["glucose"])

    t_b = 10.0
    post_delta = 1e-6
    adf_ts = transform.adf_ts

    adf_at = float(adf_ts.evaluate(jnp.array(t_b), side="left"))
    adf_after = float(adf_ts.evaluate(jnp.array(t_b + post_delta), side="left"))
    adf_after_far = float(adf_ts.evaluate(jnp.array(t_b + 5e-4), side="left"))

    assert adf_after > adf_at
    assert adf_ts.continuity_side == "left"
    assert adf_after == pytest.approx(adf_after_far, rel=1e-3)


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

    bt = _build_backtransform(proc)

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


def test_start_boundary_same_time_sample_bolus_physical_invariants():
    """Sample applies before bolus in ADF/feed-correction event physics."""
    feed_medium = FeedMedium(
        name="feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose",
                unit="g/L",
                concentration=StaticVariable(value=300.0),
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
                concentration=_ts([0.0, 5.0, 10.0], [10.0, 8.0, 6.0]),
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
                values=_ts([0.0], [-0.2]),
            ),
            "bolus": FeedVolumeChange(
                name="bolus",
                unit="L",
                is_controlled=True,
                is_continuous=False,
                feed_medium=feed_medium,
                values=_ts([0.0], [0.1]),
            ),
        },
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="start_same_time", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=10.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )

    transform = build_pseudobatch_transform(proc, ["glucose"])
    t_post = jnp.array(1e-6)

    assert transform.adf_ts.metadata["boundary_start_value"] == pytest.approx(1.0)
    assert transform.species["glucose"].feed_corr_ts.metadata[
        "boundary_start_value"
    ] == pytest.approx(0.0)

    sample_comp = float(transform.sample_compensation_ts.evaluate(t_post))
    reactor_volume = float(transform.reactor_volume_ts.evaluate(t_post))
    adf = float(transform.adf_ts.evaluate(t_post))
    feed_corr = float(transform.species["glucose"].feed_corr_ts.evaluate(t_post))

    expected_sample_comp = 1.0 / 0.8
    expected_volume = 0.9
    expected_adf = expected_volume * expected_sample_comp / 1.0
    expected_feed_corr = expected_sample_comp * 0.1 * 300.0 / 1.0

    assert sample_comp == pytest.approx(expected_sample_comp)
    assert reactor_volume == pytest.approx(expected_volume)
    assert adf == pytest.approx(expected_adf)
    assert feed_corr == pytest.approx(expected_feed_corr)
