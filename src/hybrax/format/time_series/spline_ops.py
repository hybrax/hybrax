"""Spline evaluation helpers for piecewise cubic power-basis splines."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import PPoly


VALID_SIDES = ("left", "right")


def validate_side(side: str, *, name: str = "side") -> None:
    if side not in VALID_SIDES:
        raise ValueError(f"{name} must be 'left' or 'right'")


def piece_index(t: Any, breaks: jnp.ndarray, side: str) -> jnp.ndarray:
    """Return the piece index for time t with continuity-side semantics."""
    validate_side(side)
    raw = jnp.searchsorted(breaks, t, side=side) - 1
    return jnp.clip(raw, 0, breaks.shape[0] - 2)


def evaluate_piece(coeff_row: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
    """Evaluate one cubic piece in Horner form."""
    a, b, c, d = coeff_row
    return a + dt * (b + dt * (c + dt * d))


def evaluate_scalar(
    t: Any,
    breaks: jnp.ndarray,
    coeffs: jnp.ndarray,
    side: str,
) -> jnp.ndarray:
    """Evaluate a piecewise cubic spline at one time."""
    idx = piece_index(t, breaks, side)
    dt = jnp.asarray(t, dtype=breaks.dtype) - breaks[idx]
    return evaluate_piece(coeffs[idx], dt)


def evaluate_many(
    ts: Any,
    breaks: jnp.ndarray,
    coeffs: jnp.ndarray,
    side: str,
) -> jnp.ndarray:
    """Evaluate a piecewise cubic spline at many time points."""
    ts_arr = jnp.asarray(ts, dtype=breaks.dtype)
    if ts_arr.ndim != 1:
        raise ValueError("ts must be a 1D array")
    return jax.vmap(lambda t: evaluate_scalar(t, breaks, coeffs, side))(ts_arr)


def rebase_piece(coeff_row: np.ndarray, shift: float) -> np.ndarray:
    """Rebase cubic coeffs [a,b,c,d] from x0 to x0+shift."""
    a, b, c, d = coeff_row
    s = shift
    return np.asarray(
        [
            a + b * s + c * s * s + d * s * s * s,
            b + 2.0 * c * s + 3.0 * d * s * s,
            c + 3.0 * d * s,
            d,
        ],
        dtype=np.float64,
    )


def rebase_to_breaks(
    breaks: jnp.ndarray,
    coeffs: jnp.ndarray,
    target_breaks: jnp.ndarray,
) -> jnp.ndarray:
    """Express spline coeffs on a refined breakpoint grid."""
    src_b = np.asarray(breaks, dtype=np.float64)
    src_c = np.asarray(coeffs, dtype=np.float64)
    tgt_b = np.asarray(target_breaks, dtype=np.float64)
    n_out = tgt_b.shape[0] - 1
    out = np.zeros((n_out, 4), dtype=np.float64)
    for i in range(n_out):
        u = float(tgt_b[i])
        src_idx = int(np.searchsorted(src_b, u, side="right") - 1)
        src_idx = int(np.clip(src_idx, 0, src_c.shape[0] - 1))
        shift = u - float(src_b[src_idx])
        out[i] = rebase_piece(src_c[src_idx], shift)
    return jnp.asarray(out, dtype=jnp.float64)


def merge_breaks(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    merged = np.unique(np.concatenate([np.asarray(a), np.asarray(b)]))
    return jnp.asarray(merged, dtype=jnp.float64)


def ppoly_to_power_basis(ppoly: PPoly) -> tuple[np.ndarray, np.ndarray]:
    """Convert SciPy PPoly to cubic power-basis arrays (breaks, coeffs)."""
    x = np.asarray(ppoly.x, dtype=np.float64)
    c = np.asarray(ppoly.c, dtype=np.float64)
    if c.shape[0] < 1 or c.shape[0] > 4:
        raise ValueError("unsupported polynomial degree")
    if c.shape[0] < 4:
        full = np.zeros((4, c.shape[1]), dtype=np.float64)
        full[4 - c.shape[0] :, :] = c
        c = full

    widths = np.diff(x)
    keep = widths > 0
    breaks = np.concatenate([x[:-1][keep], x[-1:]], axis=0)
    coeffs = np.stack([c[3], c[2], c[1], c[0]], axis=1)[keep]
    return breaks, coeffs


def _piece_has_near_zero(coeff_row: np.ndarray, width: float, threshold: float) -> bool:
    a, b, c, d = coeff_row

    # p(t) = a + b t + c t^2 + d t^3 on [0, width]
    candidates = [0.0, width]
    poly_coeffs = np.asarray([d, c, b, a], dtype=np.float64)
    poly_roots = np.roots(poly_coeffs)
    for root in poly_roots:
        if abs(root.imag) <= 1e-12:
            t = float(root.real)
            if 0.0 <= t <= width:
                candidates.append(t)

    deriv_coeffs = np.asarray([3.0 * d, 2.0 * c, b], dtype=np.float64)
    deriv_roots = np.roots(deriv_coeffs)
    for root in deriv_roots:
        if abs(root.imag) <= 1e-12:
            t = float(root.real)
            if 0.0 <= t <= width:
                candidates.append(t)

    for t in candidates:
        val = a + b * t + c * t * t + d * t * t * t
        if abs(val) <= threshold:
            return True
    return False


def has_near_zero_piece_value(
    breaks: jnp.ndarray,
    coeffs: jnp.ndarray,
    threshold: float,
) -> bool:
    """Return True if any piece gets within threshold of zero."""
    br = np.asarray(breaks, dtype=np.float64)
    cf = np.asarray(coeffs, dtype=np.float64)
    thr = float(abs(threshold))
    for i in range(cf.shape[0]):
        width = float(br[i + 1] - br[i])
        if _piece_has_near_zero(cf[i], width, thr):
            return True
    return False


def derivative_coeffs(coeffs: jnp.ndarray, order: int = 1) -> jnp.ndarray:
    if order < 0:
        raise ValueError("order must be >= 0")
    out = np.asarray(coeffs, dtype=np.float64)
    for _ in range(order):
        a = out[:, 0]
        b = out[:, 1]
        c = out[:, 2]
        d = out[:, 3]
        out = np.stack([b, 2.0 * c, 3.0 * d, np.zeros_like(a)], axis=1)
    return jnp.asarray(out, dtype=jnp.float64)


def integrate_definite(
    breaks: jnp.ndarray,
    coeffs: jnp.ndarray,
    a: float,
    b: float,
) -> float:
    if a == b:
        return 0.0
    sign = 1.0
    lo = float(a)
    hi = float(b)
    if lo > hi:
        sign = -1.0
        lo, hi = hi, lo

    br = np.asarray(breaks, dtype=np.float64)
    cf = np.asarray(coeffs, dtype=np.float64)

    total = 0.0
    left = max(lo, float(br[0]))
    right = min(hi, float(br[-1]))
    if right <= left:
        return 0.0

    start_idx = int(np.searchsorted(br, left, side="right") - 1)
    end_idx = int(np.searchsorted(br, right, side="left"))
    start_idx = int(np.clip(start_idx, 0, cf.shape[0] - 1))
    end_idx = int(np.clip(end_idx, 1, cf.shape[0]))

    for i in range(start_idx, end_idx):
        seg_l = max(left, float(br[i]))
        seg_r = min(right, float(br[i + 1]))
        if seg_r <= seg_l:
            continue
        a0, b0, c0, d0 = cf[i]
        dl = seg_l - float(br[i])
        dr = seg_r - float(br[i])
        total += (
            a0 * (dr - dl)
            + 0.5 * b0 * (dr * dr - dl * dl)
            + (1.0 / 3.0) * c0 * (dr**3 - dl**3)
            + 0.25 * d0 * (dr**4 - dl**4)
        )

    return sign * float(total)


def merge_segment_starts(
    breaks_a: jnp.ndarray,
    starts_a: jnp.ndarray,
    breaks_b: jnp.ndarray,
    starts_b: jnp.ndarray,
    merged_breaks: jnp.ndarray,
) -> jnp.ndarray:
    start_times = [float(np.asarray(breaks_a)[int(i)]) for i in np.asarray(starts_a)]
    start_times.extend(
        [float(np.asarray(breaks_b)[int(i)]) for i in np.asarray(starts_b)]
    )
    start_times.append(float(np.asarray(merged_breaks)[0]))
    idxs: set[int] = set()
    merged = np.asarray(merged_breaks)
    n_pieces = merged.shape[0] - 1
    for t in start_times:
        idx = int(np.searchsorted(merged, t, side="left"))
        idx = min(max(idx, 0), n_pieces - 1)
        idxs.add(idx)
    out = np.asarray(sorted(idxs), dtype=np.int32)
    if out[0] != 0:
        out = np.concatenate([np.asarray([0], dtype=np.int32), out], axis=0)
    return jnp.asarray(np.unique(out), dtype=jnp.int32)
