"""Phase 6 tests for hard-break canonical TimeSeries API."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import bpbench
from bpbench.serialization import _timeseries_to_dict_payload


def test_phase6_legacy_constructor_keyword_is_rejected() -> None:
    with pytest.raises(TypeError):
        bpbench.TimeSeries(
            timepoints=jnp.array([0.0, 1.0]),
            values=jnp.array([1.0, 2.0]),
        )


def test_phase6_timepoints_property_is_removed() -> None:
    ts = bpbench.TimeSeries(
        times=jnp.array([0.0, 1.0]),
        values=jnp.array([1.0, 2.0]),
    )
    with pytest.raises(AttributeError):
        _ = ts.timepoints


def test_phase6_canonical_payload_serialization_uses_times_only() -> None:
    ts = bpbench.TimeSeries(
        times=jnp.array([0.0, 1.0]),
        values=jnp.array([1.0, 2.0]),
    )

    payload = _timeseries_to_dict_payload(ts)

    assert "times" in payload
    assert "timepoints" not in payload
