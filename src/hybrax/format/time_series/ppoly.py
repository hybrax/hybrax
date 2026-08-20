"""Owned piecewise cubic power-basis polynomial."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import PPoly as SciPyPPoly
from scipy.interpolate import PchipInterpolator

from . import spline_ops


class PPoly(eqx.Module):
    """Piecewise cubic polynomial with coeffs shaped (pieces, 4, *output_shape)."""

    breaks: jnp.ndarray
    coeffs: jnp.ndarray
    continuity_side: str = eqx.field(static=True, default="right")

    def __init__(self, breaks: Any, coeffs: Any, continuity_side: str = "right"):
        spline_ops.validate_side(continuity_side, name="continuity_side")
        breaks_arr = jnp.asarray(breaks)
        coeffs_arr = jnp.asarray(coeffs)
        if breaks_arr.ndim != 1:
            raise ValueError("breaks must be a 1D array")
        if coeffs_arr.ndim < 2:
            raise ValueError("coeffs must have shape (n_pieces, 4, *output_shape)")
        if coeffs_arr.shape[1] != 4:
            raise ValueError("coeffs must have shape (n_pieces, 4, *output_shape)")
        if breaks_arr.shape[0] != coeffs_arr.shape[0] + 1:
            raise ValueError("len(breaks) must be len(coeffs) + 1")
        if coeffs_arr.shape[0] == 0:
            raise ValueError("PPoly must contain at least one piece")
        try:
            breaks_are_increasing = bool(jnp.all(jnp.diff(breaks_arr) > 0))
        except jax.errors.TracerBoolConversionError:
            # TimeSeries.poly may construct a PPoly while an Equinox module is
            # being traced. Shapes have already been checked; defer data-value
            # monotonicity validation for traced arrays.
            breaks_are_increasing = True
        if not breaks_are_increasing:
            raise ValueError("breaks must be strictly increasing")
        dtype = jnp.result_type(breaks_arr, coeffs_arr, jnp.float32)
        object.__setattr__(self, "breaks", jnp.asarray(breaks_arr, dtype=dtype))
        object.__setattr__(self, "coeffs", jnp.asarray(coeffs_arr, dtype=dtype))
        object.__setattr__(self, "continuity_side", continuity_side)

    @classmethod
    def from_scipy_ppoly(
        cls, ppoly: SciPyPPoly, continuity_side: str = "right"
    ) -> "PPoly":
        """Convert a SciPy PPoly to local power-basis coefficients."""
        x = np.asarray(ppoly.x, dtype=np.float64)
        c = np.asarray(ppoly.c, dtype=np.float64)
        if c.shape[0] < 1 or c.shape[0] > 4:
            raise ValueError("unsupported polynomial degree")
        if c.shape[0] < 4:
            full = np.zeros((4, c.shape[1], *c.shape[2:]), dtype=np.float64)
            full[4 - c.shape[0] :, ...] = c
            c = full
        widths = np.diff(x)
        keep = widths > 0
        breaks = np.concatenate([x[:-1][keep], x[-1:]], axis=0)
        coeffs = np.stack([c[3], c[2], c[1], c[0]], axis=1)[keep]
        return cls(breaks, coeffs, continuity_side=continuity_side)

    @classmethod
    def from_samples_pchip(
        cls, t: Any, y: Any, continuity_side: str = "right"
    ) -> "PPoly":
        """Build an owned PPoly from SciPy's PCHIP sample interpolator.

        This constructor is a low-level conversion convenience. The project-wide
        spline fitting policy remains smoothing-first and does not use this as a
        fallback selector.
        """
        scipy_poly = PchipInterpolator(t, y, axis=0)
        return cls.from_scipy_ppoly(scipy_poly, continuity_side=continuity_side)

    def __call__(self, t: Any, nu: int = 0, side: str | None = None) -> jnp.ndarray:
        """Evaluate at ``t``; result shape is ``t.shape + output_shape``."""
        order = int(nu)
        if order < 0:
            raise ValueError("nu must be >= 0")
        effective_side = self.continuity_side if side is None else side
        spline_ops.validate_side(effective_side)
        coeffs = self._derivative_coeffs(order)
        t_arr = jnp.asarray(t, dtype=self.breaks.dtype)
        idx = spline_ops.piece_index(t_arr, self.breaks, effective_side)
        dt = t_arr - self.breaks[idx]
        piece = coeffs[idx]
        coeff_axis = t_arr.ndim
        a = jnp.take(piece, 0, axis=coeff_axis)
        b = jnp.take(piece, 1, axis=coeff_axis)
        c = jnp.take(piece, 2, axis=coeff_axis)
        d = jnp.take(piece, 3, axis=coeff_axis)
        while dt.ndim < a.ndim:
            dt = dt[..., jnp.newaxis]
        return a + dt * (b + dt * (c + dt * d))

    def derivative(self, order: int = 1) -> "PPoly":
        """Return the piecewise-polynomial derivative of the given order."""
        order = int(order)
        if order < 0:
            raise ValueError("order must be >= 0")
        return PPoly(self.breaks, self._derivative_coeffs(order), self.continuity_side)

    def _derivative_coeffs(self, order: int) -> jnp.ndarray:
        out = self.coeffs
        for _ in range(order):
            a = out[:, 0, ...]
            b = out[:, 1, ...]
            c = out[:, 2, ...]
            d = out[:, 3, ...]
            out = jnp.stack([b, 2.0 * c, 3.0 * d, jnp.zeros_like(a)], axis=1)
        return out
