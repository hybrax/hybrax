"""Tests for TimeSeries dtype field and dtype-governed casting behaviour.

x64 mode is enabled globally in `tests/conftest.py`.
"""

import warnings
import subprocess
import sys

import jax.numpy as jnp
import pytest

from bp_format.time_series.timeseries import TimeSeries
from bp_format.splines import fit_timeseries_spline


# ---------------------------------------------------------------------------
# 1. Default dtype is float64
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
# 2. Explicit float32: fields and dtype attribute
# ---------------------------------------------------------------------------


def test_explicit_float32_dtype():
    ts = TimeSeries(
        times=jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
        values=jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
        dtype=jnp.float32,
    )
    assert ts.dtype == jnp.dtype("float32")
    assert ts.times.dtype == jnp.float32
    assert ts.values.dtype == jnp.float32


def test_explicit_float32_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # Should not raise; inputs already match the requested dtype.
        TimeSeries(
            times=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
            values=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
            dtype=jnp.float32,
        )


# ---------------------------------------------------------------------------
# 3. Casting warning fires on float-input dtype mismatch
# ---------------------------------------------------------------------------


def test_casting_warning_fires_on_float_mismatch():
    with pytest.warns(UserWarning, match="casting") as record:
        TimeSeries(
            times=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
            values=[1.0, 2.0],
        )
    # At least one warning must mention "times"
    messages = [str(w.message) for w in record]
    assert any("times" in m for m in messages)


# ---------------------------------------------------------------------------
# 4. No casting warning on plain Python-list input
# ---------------------------------------------------------------------------


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
# 5. JSON roundtrip preserves float32 dtype
# ---------------------------------------------------------------------------


def test_json_roundtrip_preserves_float32():
    ts = TimeSeries(
        times=jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
        values=jnp.asarray([4.0, 5.0, 6.0], dtype=jnp.float32),
        dtype=jnp.float32,
    )
    d = ts.to_dict()
    ts2 = TimeSeries.from_dict(d)
    assert ts2.dtype == jnp.dtype("float32")
    assert ts2.times.dtype == jnp.float32
    assert ts2.values.dtype == jnp.float32


# ---------------------------------------------------------------------------
# 6. JSON roundtrip preserves float64 dtype
# ---------------------------------------------------------------------------


def test_json_roundtrip_preserves_float64():
    ts = TimeSeries(times=[1.0, 2.0, 3.0], values=[4.0, 5.0, 6.0])
    d = ts.to_dict()
    ts2 = TimeSeries.from_dict(d)
    assert ts2.dtype == jnp.dtype("float64")
    assert ts2.times.dtype == jnp.float64
    assert ts2.values.dtype == jnp.float64


# ---------------------------------------------------------------------------
# 7. to_dict includes "dtype" string
# ---------------------------------------------------------------------------


def test_to_dict_includes_dtype_string_float64():
    ts = TimeSeries(times=[1.0, 2.0], values=[1.0, 2.0])
    d = ts.to_dict()
    assert "dtype" in d
    assert d["dtype"] == "float64"


def test_to_dict_includes_dtype_string_float32():
    ts = TimeSeries(
        times=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        values=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        dtype=jnp.float32,
    )
    d = ts.to_dict()
    assert d["dtype"] == "float32"


# ---------------------------------------------------------------------------
# 8. Backwards compat: old dict without "dtype" key defaults to float64
# ---------------------------------------------------------------------------


def test_backwards_compat_no_dtype_key():
    old_dict = {"times": [1.0, 2.0, 3.0], "values": [1.0, 2.0, 3.0]}
    ts = TimeSeries.from_dict(old_dict)
    assert ts.dtype == jnp.dtype("float64")
    assert ts.times.dtype == jnp.float64


# ---------------------------------------------------------------------------
# 9. from_dict explicit dtype kwarg overrides saved field
# ---------------------------------------------------------------------------


def test_from_dict_dtype_kwarg_overrides_saved_field():
    ts = TimeSeries(
        times=jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
        values=jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
        dtype=jnp.float32,
    )
    d = ts.to_dict()
    assert d["dtype"] == "float32"  # sanity check

    # Reload with explicit override to float64
    ts2 = TimeSeries.from_dict(d, dtype=jnp.float64)
    assert ts2.dtype == jnp.dtype("float64")
    assert ts2.times.dtype == jnp.float64
    assert ts2.values.dtype == jnp.float64


# ---------------------------------------------------------------------------
# 10. Binary op on dtype-mismatched TimeSeries raises TypeError
# ---------------------------------------------------------------------------


def test_binary_op_dtype_mismatch_raises():
    a = TimeSeries(times=[1.0, 2.0, 3.0], values=[1.0, 2.0, 3.0])
    b = TimeSeries(
        times=jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
        values=jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
        dtype=jnp.float32,
    )
    with pytest.raises(TypeError, match="dtype mismatch"):
        _ = a + b


def test_x64_off_default_dtype_matches_actual_arrays_in_subprocess():
    code = """
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from bp_format.time_series.timeseries import TimeSeries

a = TimeSeries(times=[1.0, 2.0], values=[1.0, 2.0])
b = TimeSeries(times=[1.0, 2.0], values=[3.0, 4.0], dtype=jnp.float32)
c = a + b

assert a.dtype == jnp.dtype("float32")
assert a.times.dtype == jnp.float32
assert b.dtype == jnp.dtype("float32")
assert c.dtype == jnp.dtype("float32")
assert c.values.dtype == jnp.float32
"""
    subprocess.run([sys.executable, "-c", code], check=True)


# ---------------------------------------------------------------------------
# 11. evaluate_many(ts.times) ≈ ts.values at tight rtol with float64
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
