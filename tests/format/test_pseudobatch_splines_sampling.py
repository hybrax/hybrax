"""
Tests for sampling-induced concentration jump fix in pseudo-batch splines.

These tests verify that:
1. Sampling (SampleVolumeChange) does NOT produce concentration jumps.
2. Bolus feeds (FeedVolumeChange, is_continuous=False) DO produce jumps.
3. Mixed scenarios (continuous feed + bolus + sampling) behave correctly:
   jumps only at bolus times, smooth across sampling times.
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


def _build_backtransform(proc, species):
    """Build real-space backtransform from the process-level bundle."""
    transform = build_pseudobatch_transform(proc, [species])
    return build_backtransform_spline(transform, species)


# ---------------------------------------------------------------------------
# Test 1: Sampling-only -> no jumps
# ---------------------------------------------------------------------------


def test_sampling_only_no_concentration_jump():
    """With only sampling (no feed), modeled concentration must be continuous
    across sampling event times - no jumps allowed."""
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=_ts(
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                ),
            ),
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=_ts(
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
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
                values=_ts([2.0, 5.0], [-0.1, -0.1]),
            ),
        },
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="sampling_only", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=6.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )

    bt = _build_backtransform(proc, "glucose")

    tiny_delta = 1e-6
    pre_probe = 5e-4  # safely away from event edge
    # No jump at either sampling time
    for t_s in [2.0, 5.0]:
        val_before = float(bt(jnp.array(t_s - pre_probe)))
        val_at = float(bt(jnp.array(t_s)))
        val_after = float(bt(jnp.array(t_s + tiny_delta)))
        jump = abs(val_after - val_at)
        assert val_at == pytest.approx(val_before, abs=2e-2)
        assert jump < 2e-2, (
            f"Sampling should NOT cause a concentration jump at t={t_s}; "
            f"got val_at={val_at:.6f}, val_after={val_after:.6f}, jump={jump:.6f}"
        )

    # Overall curve should stay approximately constant at ~10
    t_eval = np.linspace(0.0, 6.0, 50)
    c_hat = np.array([float(bt(jnp.array(t))) for t in t_eval])
    assert np.max(c_hat) - np.min(c_hat) < 0.5, (
        f"Concentration should remain approximately constant; "
        f"range = {np.max(c_hat) - np.min(c_hat):.4f}"
    )


# ---------------------------------------------------------------------------
# Test 2: Bolus-only feed -> jumps present
# ---------------------------------------------------------------------------


def test_bolus_only_has_concentration_jump():
    """With a bolus feed of concentrated glucose, there should be a
    concentration discontinuity at the feed time."""
    feed_medium = FeedMedium(
        name="feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=StaticVariable(value=100.0),
                is_controlled=True,
            ),
        },
    )
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=_ts(
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                ),
            ),
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=_ts(
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                ),
            ),
        },
    )
    vol = Volume(
        initial_volume=1.0,
        unit="L",
        volume_changes={
            "feed": FeedVolumeChange(
                name="feed",
                unit="L",
                is_controlled=True,
                is_continuous=False,
                feed_medium=feed_medium,
                values=_ts([2.0], [0.5]),
            ),
        },
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="bolus_only", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=6.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )

    bt = _build_backtransform(proc, "glucose")

    t_b = 2.0
    tiny_delta = 1e-6
    pre_probe = 5e-4
    val_before = float(bt(jnp.array(t_b - pre_probe)))
    val_at = float(bt(jnp.array(t_b)))
    val_after = float(bt(jnp.array(t_b + tiny_delta)))
    jump = abs(val_after - val_at)

    assert val_at == pytest.approx(val_before, abs=2e-2)
    assert jump > 0.1, (
        f"Bolus feed should cause a concentration jump at t=2.0; "
        f"got val_at={val_at:.6f}, val_after={val_after:.6f}, jump={jump:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3: Mixed continuous feed + bolus feed + sampling
# ---------------------------------------------------------------------------


def test_mixed_continuous_bolus_sampling():
    """Concentration should jump at bolus feed times, be smooth at sampling
    times, and not show step discontinuities from continuous feed."""
    feed_medium_cont = FeedMedium(
        name="cont_feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=StaticVariable(value=50.0),
                is_controlled=True,
            ),
        },
    )
    feed_medium_bolus = FeedMedium(
        name="bolus_feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=StaticVariable(value=200.0),
                is_controlled=True,
            ),
        },
    )
    rm = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=_ts(
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                ),
            ),
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=_ts(
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                ),
            ),
        },
    )
    vol = Volume(
        initial_volume=1.0,
        unit="L",
        volume_changes={
            "cont_feed": FeedVolumeChange(
                name="cont_feed",
                unit="L",
                is_controlled=True,
                is_continuous=True,
                feed_medium=feed_medium_cont,
                values=_ts(
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                ),
            ),
            "bolus_feed": FeedVolumeChange(
                name="bolus_feed",
                unit="L",
                is_controlled=True,
                is_continuous=False,
                feed_medium=feed_medium_bolus,
                values=_ts([3.0], [0.5]),
            ),
            "sample": SampleVolumeChange(
                name="sample",
                unit="L",
                is_controlled=True,
                is_continuous=False,
                values=_ts([4.0], [-0.2]),
            ),
        },
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="mixed", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=6.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )

    bt = _build_backtransform(proc, "glucose")

    tiny_delta = 1e-6
    pre_probe = 5e-4

    # 1) No jump at sampling time (t=4)
    val_before_sample = float(bt(jnp.array(4.0 - pre_probe)))
    val_at_sample = float(bt(jnp.array(4.0)))
    val_after_sample = float(bt(jnp.array(4.0 + tiny_delta)))
    sample_jump = abs(val_after_sample - val_at_sample)
    assert val_at_sample == pytest.approx(val_before_sample, abs=2e-2)
    assert sample_jump < 2e-2, (
        f"Sampling should NOT cause a large concentration jump at t=4.0; "
        f"got jump={sample_jump:.6f}"
    )

    # 2) Jump at bolus feed time (t=3)
    val_before_bolus = float(bt(jnp.array(3.0 - pre_probe)))
    val_at_bolus = float(bt(jnp.array(3.0)))
    val_after_bolus = float(bt(jnp.array(3.0 + tiny_delta)))
    bolus_jump = abs(val_after_bolus - val_at_bolus)
    assert val_at_bolus == pytest.approx(val_before_bolus, abs=3e-2)
    assert bolus_jump > 0.1, (
        f"Bolus feed should cause a concentration jump at t=3.0; "
        f"got jump={bolus_jump:.6f}"
    )

    # 3) No step discontinuity at a non-event time (continuous feed is smooth)
    val_before_smooth = float(bt(jnp.array(2.5 - tiny_delta)))
    val_after_smooth = float(bt(jnp.array(2.5 + tiny_delta)))
    smooth_jump = abs(val_after_smooth - val_before_smooth)
    assert smooth_jump < 2e-2, (
        f"Continuous feed should not create a large step at t=2.5; "
        f"got jump={smooth_jump:.6f}"
    )
