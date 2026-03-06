"""
Tests verifying that ADF and feed-term use step (piecewise-constant)
interpolation, not linear interpolation, during the pseudo-batch
backtransform.

A step function must produce identical values regardless of how close
the query time is to the event boundary. Linear interpolation would
produce values that vary with the distance to the boundary.
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
    evaluate_timeseries_spline_at,
    fit_state_timeseries_spline_pseudobatch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(t, v):
    return TimeSeries(
        timepoints=jnp.array(t, dtype=float),
        values=jnp.array(v, dtype=float),
    )


# ---------------------------------------------------------------------------
# Test 1: Bolus feed produces a true discrete jump (no linear ramp)
# ---------------------------------------------------------------------------

def test_bolus_feed_discrete_jump_is_step_not_ramp():
    """
    At a bolus feed event, the backtransformed concentration must jump
    instantaneously. Evaluating at t_event - delta for several small
    delta values must all return the same pre-jump value (constant),
    and evaluating at t_event + delta must all return the same post-jump
    value (constant). If ADF/feed_term were linearly interpolated, the
    values would vary with delta.
    """
    t_feed = 5.0
    V0 = 1.0
    dV = 0.5
    c_feed = 100.0

    # Simple process: constant concentration of 10, then bolus feed at t=5
    # After feed at t=5: c_after = (c_before * V0 + c_feed * dV) / (V0 + dV)
    # = (10 * 1.0 + 100 * 0.5) / 1.5 = 60 / 1.5 = 40
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    values = [10.0, 10.0, 10.0, 10.0, 10.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0]

    glucose_ts = _ts(times, values)

    feed_medium = FeedMedium(
        name="feed",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": FeedMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=StaticVariable(value=c_feed),
                is_controlled=True,
            ),
        },
    )
    vol = Volume(
        initial_volume=V0,
        unit="L",
        volume_changes={
            "bolus": FeedVolumeChange(
                name="bolus",
                unit="L",
                is_continuous=False,
                is_controlled=True,
                values=_ts([t_feed], [dV]),
                feed_medium=feed_medium,
            ),
        },
    )
    rm = ReactorMedium(
        name="reactor",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=glucose_ts,
                is_intracellular=False,
            ),
        },
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="test", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=10.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )

    rep = fit_state_timeseries_spline_pseudobatch(glucose_ts, proc, "glucose")

    # Evaluate at several distances before and after the feed event
    deltas = [1e-4, 1e-6, 1e-8, 1e-10]

    pre_values = [evaluate_timeseries_spline_at(rep, t_feed - d) for d in deltas]
    post_values = [evaluate_timeseries_spline_at(rep, t_feed + d) for d in deltas]

    # All pre-event values should be approximately equal (step = constant)
    for i in range(1, len(pre_values)):
        assert abs(pre_values[i] - pre_values[0]) < 0.1, (
            f"Pre-event values should be constant (step function) but got "
            f"{pre_values[0]:.6f} at delta={deltas[0]} vs "
            f"{pre_values[i]:.6f} at delta={deltas[i]}. "
            f"This suggests linear interpolation is being used instead of step."
        )

    # All post-event values should be approximately equal
    for i in range(1, len(post_values)):
        assert abs(post_values[i] - post_values[0]) < 0.1, (
            f"Post-event values should be constant (step function) but got "
            f"{post_values[0]:.6f} at delta={deltas[0]} vs "
            f"{post_values[i]:.6f} at delta={deltas[i]}. "
            f"This suggests linear interpolation is being used instead of step."
        )

    # There should be a jump between pre and post
    jump = abs(post_values[0] - pre_values[0])
    assert jump > 1.0, (
        f"Expected a significant concentration jump at the feed event, got {jump:.6f}"
    )

    # Pre-event should be close to 10, post-event close to 40
    assert abs(pre_values[0] - 10.0) < 2.0, (
        f"Pre-event concentration should be ~10, got {pre_values[0]:.4f}"
    )
    assert abs(post_values[0] - 40.0) < 2.0, (
        f"Post-event concentration should be ~40, got {post_values[0]:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 2: Sampling-only still produces no jump
# ---------------------------------------------------------------------------

def test_sampling_no_jump_still_works():
    """Sampling events should NOT cause any concentration discontinuity."""
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    values = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

    glucose_ts = _ts(times, values)

    vol = Volume(
        initial_volume=1.0,
        unit="L",
        volume_changes={
            "sampling": SampleVolumeChange(
                name="sampling",
                unit="L",
                is_continuous=False,
                is_controlled=True,
                values=_ts([2.0, 4.0], [-0.01, -0.01]),
            ),
        },
    )
    rm = ReactorMedium(
        name="reactor",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=glucose_ts,
                is_intracellular=False,
            ),
        },
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="test", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=6.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )

    rep = fit_state_timeseries_spline_pseudobatch(glucose_ts, proc, "glucose")

    eps = 1e-6
    for t_s in [2.0, 4.0]:
        val_before = evaluate_timeseries_spline_at(rep, t_s - eps)
        val_after = evaluate_timeseries_spline_at(rep, t_s + eps)
        assert abs(val_after - val_before) < 0.5, (
            f"Sampling at t={t_s} should NOT cause a jump, got "
            f"before={val_before:.6f}, after={val_after:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 3: Metadata tag reflects step interpolation
# ---------------------------------------------------------------------------

def test_metadata_interp_tag_is_step():
    """The spline metadata should indicate step interpolation, not linear."""
    t_feed = 5.0
    glucose_ts = _ts(
        [0.0, 2.0, 5.0, 8.0, 10.0],
        [10.0, 10.0, 40.0, 40.0, 40.0],
    )
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
    vol = Volume(
        initial_volume=1.0,
        unit="L",
        volume_changes={
            "bolus": FeedVolumeChange(
                name="bolus",
                unit="L",
                is_continuous=False,
                is_controlled=True,
                values=_ts([t_feed], [0.5]),
                feed_medium=feed_medium,
            ),
        },
    )
    rm = ReactorMedium(
        name="reactor",
        density=1.0,
        density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose",
                unit="mmol/L",
                concentration=glucose_ts,
                is_intracellular=False,
            ),
        },
    )
    proc = BioProcess(
        metadata=BioProcessMetadata(name="test", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=10.0, time_reference="inoculation"),
        volume=vol,
        reactor_medium=rm,
    )

    rep = fit_state_timeseries_spline_pseudobatch(glucose_ts, proc, "glucose")
    assert rep.spline_metadata["transform"]["interp"] == "step", (
        f"Expected interp='step', got "
        f"'{rep.spline_metadata['transform']['interp']}'"
    )
