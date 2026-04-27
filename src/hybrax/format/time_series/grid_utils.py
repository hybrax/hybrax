"""Helpers for time-grid merging and discrete sample synthesis."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np

from .constants import TIME_DEDUP_ATOL, TIME_DEDUP_RTOL


def merge_times_with_tolerance(
    times_a: jnp.ndarray,
    times_b: jnp.ndarray,
    *,
    atol: float = TIME_DEDUP_ATOL,
    rtol: float = TIME_DEDUP_RTOL,
) -> jnp.ndarray:
    """Merge two sorted time arrays with tolerance-based deduplication."""
    a = np.asarray(times_a, dtype=np.float64)
    b = np.asarray(times_b, dtype=np.float64)
    candidate = np.sort(np.concatenate([a, b], axis=0))
    if candidate.size == 0:
        return jnp.asarray([], dtype=jnp.float64)

    scale = float(candidate[-1] - candidate[0]) if candidate.size > 1 else 1.0
    if scale <= 0.0:
        scale = 1.0
    tol = float(atol + rtol * scale)

    merged = [float(candidate[0])]
    for t in candidate[1:]:
        if float(t - merged[-1]) > tol:
            merged.append(float(t))
    return jnp.asarray(np.asarray(merged, dtype=np.float64), dtype=jnp.float64)


def linear_interpolate_samples(
    source_times: jnp.ndarray,
    source_values: jnp.ndarray,
    target_times: jnp.ndarray,
) -> jnp.ndarray:
    """Linearly interpolate source samples onto a target time grid."""
    x = np.asarray(source_times, dtype=np.float64)
    y = np.asarray(source_values, dtype=np.float64)
    t = np.asarray(target_times, dtype=np.float64)
    return jnp.asarray(np.interp(t, x, y).astype(np.float64), dtype=jnp.float64)


def synthesize_binary_samples(
    left_times: jnp.ndarray | None,
    left_values: jnp.ndarray | None,
    right_times: jnp.ndarray | None,
    right_values: jnp.ndarray | None,
    *,
    op: Literal["add", "sub"],
) -> tuple[jnp.ndarray | None, jnp.ndarray | None]:
    """Synthesize discrete samples for simple derived binary operations."""
    if left_times is None or left_values is None:
        return None, None
    if right_times is None or right_values is None:
        return None, None

    out_times = merge_times_with_tolerance(left_times, right_times)
    left_interp = linear_interpolate_samples(left_times, left_values, out_times)
    right_interp = linear_interpolate_samples(right_times, right_values, out_times)
    if op == "add":
        out_values = left_interp + right_interp
    elif op == "sub":
        out_values = left_interp - right_interp
    else:
        raise ValueError("op must be 'add' or 'sub'")
    return out_times, out_values
