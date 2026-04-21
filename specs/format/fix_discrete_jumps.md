# Bug Fix: ADF and Feed-Term Must Use Step (Piecewise-Constant) Interpolation, Not Linear

## Problem Summary

In `bp_format/splines.py`, the function `evaluate_timeseries_spline_at()` uses `np.interp()` (linear interpolation) to look up ADF and feed-term values between grid points during the pseudo-batch backtransform. This is incorrect. ADF and feed-term are **piecewise-constant** (step) functions that must jump discretely at bolus feed event times. Linear interpolation creates a gradual ramp between pre-event and post-event values instead of an instantaneous jump, which is visible as diagonal lines in the spline plot where there should be vertical discontinuities.

The backtransform formula is: `ĉ(t) = (ĉ*(t) + feed_term(t)) / ADF(t)`

When ADF(t) and feed_term(t) are linearly interpolated between `t_event - epsilon` and `t_event`, the backtransformed concentration ramps linearly instead of jumping discretely. The yellow simulation line in the plot shows the correct behavior (vertical jumps), while the blue spline line shows incorrect diagonal ramps at every feed event boundary.

## Root Cause

In `evaluate_timeseries_spline_at()` (around line 823-837), when `"adf_times"` is present in the transform metadata, the code uses:

```python
adf_t = float(np.interp(t, np.array(tr["adf_times"]), np.array(tr["adf_values"]), ...))
feed_t = float(np.interp(t, np.array(tr["feed_term_times"]), np.array(tr["feed_term_values"]), ...))
```

`np.interp` performs **linear** interpolation. Since the grid contains pairs of points like `(t_event - 1e-10, pre_value)` and `(t_event, post_value)`, any query time between these two points gets a blended value. Even though the epsilon gap is tiny (1e-10), floating point queries within that range produce intermediate values, and more importantly, the conceptual model is wrong — these are step functions, not ramps.

The existing `_step_eval()` function in the same file already implements the correct piecewise-constant lookup and is used in the legacy/fallback code path. It should be used for the primary path too.

## Files to Change

### 1. `bp_format/splines.py` — `evaluate_timeseries_spline_at()` function

**What to change:** Replace `np.interp()` calls with `_step_eval()` calls for both ADF and feed-term lookup.

Replace this block (lines ~823-837):

```python
if "adf_times" in tr:
    adf_t = float(np.interp(
        t,
        np.array(tr["adf_times"]),
        np.array(tr["adf_values"]),
        left=tr["adf_values"][0],
        right=tr["adf_values"][-1],
    ))
    feed_t = float(np.interp(
        t,
        np.array(tr["feed_term_times"]),
        np.array(tr["feed_term_values"]),
        left=tr["feed_term_values"][0],
        right=tr["feed_term_values"][-1],
    ))
```

With:

```python
if "adf_times" in tr:
    adf_t = _step_eval(
        np.array(tr["adf_times"]),
        np.array(tr["adf_values"]),
        t,
    )
    feed_t = _step_eval(
        np.array(tr["feed_term_times"]),
        np.array(tr["feed_term_values"]),
        t,
    )
```

This makes both the primary path (with `"adf_times"` key) and the legacy fallback path use `_step_eval()`.

### 2. `bp_format/splines.py` — metadata tag (optional cleanup)

In `fit_state_timeseries_spline_pseudobatch()`, around line 790, change the metadata interpolation tag from `"linear"` to `"step"` to accurately describe what the code does:

```python
# Change this:
"interp": "linear",
# To this:
"interp": "step",
```

### 3. `evaluate_timeseries_spline()` — vectorized version

Also check that the vectorized wrapper `evaluate_timeseries_spline()` at line 863 delegates to `evaluate_timeseries_spline_at()` — it already does, so no change needed there. The fix propagates automatically.

## What NOT to Change

- Do NOT change `pseudo_batch_transform_timeseries()` — the grid construction with epsilon pre-event points is correct
- Do NOT change `_prepare_pseudobatch_inputs()` — the pseudobatch transform computation is correct  
- Do NOT change `_step_eval()` or `_step_eval_array()` — these are already correct
- Do NOT change `fit_timeseries_spline()` or `evaluate_spline_at()` — spline fitting/evaluation in pseudo-batch space is correct
- Do NOT change any test that currently passes — only add new tests or update the `"interp"` metadata assertion if it exists

## Tests to Implement

Create or update tests to verify discrete jumps are actual steps, not linear ramps. The key test strategy: evaluate the spline at `t_event - small_delta` and `t_event + small_delta` for multiple delta values. With correct step behavior, both sides should give constant values (independent of delta). With linear interpolation, the value would change as delta changes.

### Test 1: Bolus feed produces a true discrete jump (no linear ramp)

```python
def test_bolus_feed_discrete_jump_is_step_not_ramp():
    """
    At a bolus feed event, the backtransformed concentration must jump
    instantaneously. Evaluating at t_event - delta for several small
    delta values must all return the same pre-jump value (constant),
    and evaluating at t_event + delta must all return the same post-jump
    value (constant). If ADF/feed_term were linearly interpolated, the
    values would vary with delta.
    """
    import numpy as np
    import jax.numpy as jnp
    from bp_format import (
        BioProcess, BioProcessMetadata, FeedMedium, FeedMediumComponent,
        FeedVolumeChange, ReactorMedium, ReactorMediumComponent,
        StaticVariable, TimeAxis, TimeSeries, Volume,
    )
    from bp_format.splines import (
        evaluate_timeseries_spline_at,
        fit_state_timeseries_spline_pseudobatch,
    )

    t_feed = 5.0
    V0 = 1.0
    dV = 0.5
    c_feed = 100.0

    # Simple process: constant concentration of 10, then bolus feed at t=5
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    # After feed at t=5: c_after = (c_before * V0 + c_feed * dV) / (V0 + dV)
    # = (10 * 1.0 + 100 * 0.5) / 1.5 = 60 / 1.5 = 40
    values = [10.0, 10.0, 10.0, 10.0, 10.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0]

    glucose_ts = TimeSeries(
        timepoints=jnp.array(times, dtype=float),
        values=jnp.array(values, dtype=float),
    )

    feed_medium = FeedMedium(
        name="feed", components={
            "glucose": FeedMediumComponent(
                name="glucose", unit="mmol/L",
                concentration=StaticVariable(value=c_feed, unit="mmol/L"),
            )
        }
    )
    vol = Volume(
        initial_volume=V0, unit="L",
        volume_changes={
            "bolus": FeedVolumeChange(
                name="bolus", unit="L", is_continuous=False,
                values=TimeSeries(
                    timepoints=jnp.array([t_feed]),
                    values=jnp.array([dV]),
                ),
                feed_medium=feed_medium,
            ),
        },
    )
    rm = ReactorMedium(
        name="reactor", density=1.0, density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose", unit="mmol/L",
                concentration=glucose_ts,
            )
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

    # All pre-event values should be approximately equal (step function = constant before jump)
    for i in range(1, len(pre_values)):
        assert abs(pre_values[i] - pre_values[0]) < 0.1, (
            f"Pre-event values should be constant (step function) but got "
            f"{pre_values[0]:.6f} at delta={deltas[0]} vs {pre_values[i]:.6f} at delta={deltas[i]}. "
            f"This suggests linear interpolation is being used instead of step evaluation."
        )

    # All post-event values should be approximately equal
    for i in range(1, len(post_values)):
        assert abs(post_values[i] - post_values[0]) < 0.1, (
            f"Post-event values should be constant (step function) but got "
            f"{post_values[0]:.6f} at delta={deltas[0]} vs {post_values[i]:.6f} at delta={deltas[i]}. "
            f"This suggests linear interpolation is being used instead of step evaluation."
        )

    # There should be a jump between pre and post
    jump = abs(post_values[0] - pre_values[0])
    assert jump > 1.0, (
        f"Expected a significant concentration jump at the feed event, got {jump:.6f}"
    )

    # Pre-event should be close to 10, post-event close to 40
    assert abs(pre_values[0] - 10.0) < 2.0, f"Pre-event concentration should be ~10, got {pre_values[0]:.4f}"
    assert abs(post_values[0] - 40.0) < 2.0, f"Post-event concentration should be ~40, got {post_values[0]:.4f}"
```

### Test 2: Sampling-only still produces no jump

```python
def test_sampling_no_jump_still_works():
    """Sampling events should NOT cause any concentration discontinuity."""
    import numpy as np
    import jax.numpy as jnp
    from bp_format import (
        BioProcess, BioProcessMetadata, ReactorMedium, ReactorMediumComponent,
        SampleVolumeChange, StaticVariable, TimeAxis, TimeSeries, Volume,
    )
    from bp_format.splines import (
        evaluate_timeseries_spline_at,
        fit_state_timeseries_spline_pseudobatch,
    )

    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    values = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

    glucose_ts = TimeSeries(
        timepoints=jnp.array(times, dtype=float),
        values=jnp.array(values, dtype=float),
    )

    vol = Volume(
        initial_volume=1.0, unit="L",
        volume_changes={
            "sampling": SampleVolumeChange(
                name="sampling", unit="L", is_continuous=False,
                values=TimeSeries(
                    timepoints=jnp.array([2.0, 4.0]),
                    values=jnp.array([-0.01, -0.01]),
                ),
            ),
        },
    )
    rm = ReactorMedium(
        name="reactor", density=1.0, density_unit="kg/L",
        components={
            "glucose": ReactorMediumComponent(
                name="glucose", unit="mmol/L",
                concentration=glucose_ts,
            )
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
```

### Test 3: Metadata tag reflects step interpolation

```python
def test_metadata_interp_tag_is_step():
    """The spline metadata should indicate step interpolation, not linear."""
    # (build a simple process with a bolus feed, fit the spline)
    # ...
    rep = fit_state_timeseries_spline_pseudobatch(glucose_ts, proc, "glucose")
    assert rep.interpolator_metadata["transform"]["interp"] == "step", (
        f"Expected interp='step', got '{rep.interpolator_metadata['transform']['interp']}'"
    )
```

## How to Verify the Fix

1. Run existing tests — they should all still pass:
   ```
   pytest tests/test_splines.py tests/test_pseudobatch_splines_sampling.py -v
   ```

2. Run the new discrete jump test — it should now pass:
   ```
   pytest tests/test_discrete_jump_step.py -v
   ```

3. Visual check: re-run the notebook `examples/05_martens_2025_a/03_spline_fits.ipynb` and confirm the blue pseudo-batch spline line now shows vertical jumps at feed events (matching the yellow simulation), not diagonal ramps.

## Summary of Changes

| File | Change | Reason |
|------|--------|--------|
| `bp_format/splines.py` L823-837 | Replace `np.interp()` with `_step_eval()` | ADF and feed-term are step functions, not linear |
| `bp_format/splines.py` L790 | `"interp": "linear"` → `"interp": "step"` | Metadata should reflect actual behavior |
| `tests/test_discrete_jump_step.py` (new) | Add tests for step behavior | Verify jumps are instantaneous, not ramped |
