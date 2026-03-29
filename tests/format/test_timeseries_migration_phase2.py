"""Phase 2 tests for hard-break canonical TimeSeries API behavior."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import bpbench
from bpbench.time_series import TimeSeries as VendoredTimeSeries


def test_phase2_legacy_constructor_is_no_longer_supported() -> None:
    with pytest.raises(TypeError):
        bpbench.TimeSeries(
            timepoints=jnp.array([0.0, 1.0, 2.0]),
            values=jnp.array([1.0, 2.0, 3.0]),
        )


def test_phase2_canonical_constructor_works() -> None:
    ts = bpbench.TimeSeries(
        times=jnp.array([0.0, 2.0]),
        values=jnp.array([4.0, 5.0]),
    )
    assert ts.times.shape == (2,)
    assert ts.values.shape == (2,)


def test_phase2_constructor_rejects_timepoints_keyword_even_with_times() -> None:
    with pytest.raises(TypeError):
        bpbench.TimeSeries(
            times=jnp.array([0.0, 1.0]),
            timepoints=jnp.array([0.0, 2.0]),
            values=jnp.array([1.0, 2.0]),
        )


def test_phase2_public_timeseries_is_vendored_type() -> None:
    assert bpbench.TimeSeries is VendoredTimeSeries
    ts = bpbench.TimeSeries(times=jnp.array([0.0, 1.0]), values=jnp.array([2.0, 3.0]))
    assert isinstance(ts, bpbench.TimeSeries)
    assert isinstance(ts, VendoredTimeSeries)


def test_phase2_canonical_times_mode_keeps_strict_invariants() -> None:
    with pytest.raises(ValueError, match="times and values must have the same length"):
        bpbench.TimeSeries(
            times=jnp.array([0.0, 1.0, 2.0]),
            values=jnp.array([1.0, 2.0]),
        )
    with pytest.raises(ValueError, match="times must be strictly increasing"):
        bpbench.TimeSeries(
            times=jnp.array([0.0, 0.0, 1.0]),
            values=jnp.array([1.0, 2.0, 3.0]),
        )
