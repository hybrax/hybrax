"""Tests for TimeSeries float64 dtype and dtype-governed casting behaviour.

The whole pipeline is float64: JAX x64 is enabled in ``bp_format/__init__``.
"""

import warnings

import jax.numpy as jnp
import pytest

from bp_format.time_series.timeseries import TimeSeries
from bp_format.splines import fit_timeseries_spline


# ---------------------------------------------------------------------------
# 1. Arrays are float64
# ---------------------------------------------------------------------------


def test_default_dtype_is_float64():
    ts = TimeSeries(times=[1.0, 2.0, 3.0], values=[1.0, 2.0, 3.0])
    assert ts.dtype == jnp.dtype("float64")
    assert ts.times.dtype == jnp.float64
    assert ts.values.dtype == jnp.float64


def test_default_dtype_float64_spline_fields():
    ts = TimeSeries(
        times=[0.0, 1.0, 2.0],
        values=[0.0, 1.0, 0.0],
        breaks=[0.0, 1.0, 2.0],
        coeffs=jnp.zeros((2, 4)),
        segment_start_piece_idx=[0],
    )
    assert ts.dtype == jnp.dtype("float64")
    assert ts.breaks.dtype == jnp.float64
    assert ts.coeffs.dtype == jnp.float64


# ---------------------------------------------------------------------------
# 2. A narrower-than-float64 float input is rejected (fail-fast)
# ---------------------------------------------------------------------------


def test_float32_input_raises_on_float_mismatch():
    with pytest.raises(TypeError, match="float64"):
        TimeSeries(
            times=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
            values=[1.0, 2.0],
        )


def test_no_casting_warning_for_python_list_input():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TimeSeries(times=[1.0, 2.0], values=[1.0, 2.0])
    casting_warnings = [
        w
        for w in caught
        if issubclass(w.category, UserWarning) and "casting" in str(w.message)
    ]
    assert casting_warnings == [], (
        f"Unexpected casting warnings: {[str(w.message) for w in casting_warnings]}"
    )


# ---------------------------------------------------------------------------
# 3. JSON roundtrip stays float64 (incl. legacy dicts without a dtype key)
# ---------------------------------------------------------------------------


def test_json_roundtrip_preserves_float64():
    ts = TimeSeries(times=[1.0, 2.0, 3.0], values=[4.0, 5.0, 6.0])
    d = ts.to_dict()
    assert "dtype" not in d  # dtype is no longer serialized
    ts2 = TimeSeries.from_dict(d)
    assert ts2.dtype == jnp.dtype("float64")
    assert ts2.times.dtype == jnp.float64
    assert ts2.values.dtype == jnp.float64


def test_legacy_dict_with_dtype_key_ignored():
    # Older serialized payloads carry a "dtype" string; it must be ignored.
    old_dict = {
        "times": [1.0, 2.0, 3.0],
        "values": [1.0, 2.0, 3.0],
        "dtype": "float32",
    }
    ts = TimeSeries.from_dict(old_dict)
    assert ts.dtype == jnp.dtype("float64")
    assert ts.times.dtype == jnp.float64


# ---------------------------------------------------------------------------
# 4. evaluate_many(ts.times) ≈ ts.values at tight rtol with float64
# ---------------------------------------------------------------------------


def test_evaluate_many_roundtrip_float64_tight():
    t = jnp.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    y = jnp.asarray([1.0, 1.5, 2.0, 1.8, 1.2, 0.8, 1.0])
    ts = fit_timeseries_spline(TimeSeries(times=t, values=y))

    assert ts.dtype == jnp.dtype("float64")

    result = ts.evaluate_many(ts.times)
    assert jnp.allclose(result, ts.values, rtol=1e-12, atol=1e-12), (
        f"Max deviation: {jnp.max(jnp.abs(result - ts.values))}"
    )
