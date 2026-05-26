"""TimeSeries core data model and invariant validation."""

from __future__ import annotations

from typing import Any
import warnings

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.interpolate import PPoly as SciPyPPoly
from scipy.interpolate import make_splrep

from . import grid_utils
from . import spline_ops
from .ppoly import PPoly
from .constants import APPROX_ABS_FLOOR
from .constants import APPROX_INITIAL_S
from .constants import APPROX_MAX_REFIT_ATTEMPTS
from .constants import APPROX_REL_ERR_TARGET
from .constants import APPROX_S_REDUCTION_FACTOR
from .constants import DIVISION_NEAR_ZERO_THRESHOLD


def _as_float_array(name: str, value: Any, dtype: Any, *, ndim: int) -> jnp.ndarray:
    arr_in = jnp.asarray(value)
    if jnp.issubdtype(arr_in.dtype, jnp.floating) and arr_in.dtype != dtype:
        warnings.warn(
            f"TimeSeries: casting {name} from {arr_in.dtype} to {dtype}",
            stacklevel=3,
        )
    arr = jnp.asarray(value, dtype=dtype)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D array")
    return arr


def _as_int_1d(name: str, value: Any) -> jnp.ndarray:
    arr = jnp.asarray(value, dtype=jnp.int32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return arr


def _as_integral_index_1d(name: str, value: Any) -> jnp.ndarray:
    raw = jnp.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if raw.size == 0:
        return jnp.asarray([], dtype=jnp.int32)

    if jnp.issubdtype(raw.dtype, jnp.bool_):
        raise ValueError(f"{name} must contain integer indices")

    if not jnp.issubdtype(raw.dtype, jnp.integer):
        if not bool(jnp.all(jnp.isfinite(raw))):
            raise ValueError(f"{name} must contain finite numeric indices")
        if not bool(jnp.all(raw == jnp.floor(raw))):
            raise ValueError(f"{name} must contain integer-valued indices")
    return jnp.asarray(raw, dtype=jnp.int32)


def _is_strictly_increasing(arr: jnp.ndarray) -> bool:
    if arr.size <= 1:
        return True
    return bool(jnp.all(jnp.diff(arr) > 0))


class TimeSeries(eqx.Module):
    """Scalar-valued time series with optional samples and/or spline state.

    The `dtype` field governs all floating-point array fields (`times`,
    `values`, `jump_times`, `breaks`, `coeffs`). It defaults to requested
    `jnp.float64`.

    Note: if `jax_enable_x64` is off, `jnp.float64` will silently be
    downgraded by JAX to float32. In that case, `dtype` records the actual
    dtype JAX can represent.
    """

    times: jnp.ndarray | None = None
    values: jnp.ndarray | None = None
    jump_times: jnp.ndarray | None = None
    breaks: jnp.ndarray | None = None
    coeffs: jnp.ndarray | None = None
    segment_start_piece_idx: jnp.ndarray | None = None
    derived: bool = eqx.field(static=True, default=False)
    continuity_side: str = eqx.field(static=True, default="right")
    metadata: Any = eqx.field(static=True, default=None)
    dtype: jnp.dtype = eqx.field(static=True, default=jnp.dtype(jnp.float64))

    def __init__(
        self,
        *,
        times: Any | None = None,
        values: Any | None = None,
        derived: bool = False,
        jump_times: Any | None = None,
        poly: PPoly | None = None,
        breaks: Any | None = None,
        coeffs: Any | None = None,
        segment_start_piece_idx: Any | None = None,
        continuity_side: str = "right",
        metadata: Any | None = None,
        dtype: Any | None = None,
    ) -> None:
        # Resolve dtype first so all field setup can use it.
        dtype_requested = (
            jnp.dtype(dtype) if dtype is not None else jnp.dtype(jnp.float64)
        )
        dtype_resolved = jnp.asarray(0.0, dtype=dtype_requested).dtype
        object.__setattr__(self, "dtype", dtype_resolved)

        has_discrete = times is not None or values is not None
        has_spline = (
            poly is not None
            or breaks is not None
            or coeffs is not None
            or segment_start_piece_idx is not None
        )

        if not has_discrete and not has_spline:
            raise ValueError("Provide discrete samples and/or spline representation")

        if has_discrete and (times is None or values is None):
            raise ValueError(
                "times and values must either both be provided or both be None"
            )

        if poly is not None:
            if not isinstance(poly, PPoly):
                raise TypeError("poly must be a bp_format.time_series.PPoly")
            if poly.continuity_side != continuity_side:
                raise ValueError("poly continuity_side must match continuity_side")
            if poly.coeffs.ndim != 2:
                raise ValueError("TimeSeries requires scalar-valued PPoly coeffs")
            if breaks is None:
                breaks = poly.breaks
            elif not np.allclose(np.asarray(breaks), np.asarray(poly.breaks)):
                raise ValueError("breaks must match poly.breaks when both are provided")
            if coeffs is None:
                coeffs = poly.coeffs
            elif not np.allclose(np.asarray(coeffs), np.asarray(poly.coeffs)):
                raise ValueError("coeffs must match poly.coeffs when both are provided")

        if has_spline and (
            breaks is None or coeffs is None or segment_start_piece_idx is None
        ):
            raise ValueError(
                "breaks, coeffs, and segment_start_piece_idx must all be provided "
                "for spline representation"
            )

        spline_ops.validate_side(continuity_side, name="continuity_side")

        object.__setattr__(self, "derived", bool(derived))
        object.__setattr__(self, "continuity_side", continuity_side)
        object.__setattr__(self, "metadata", metadata)

        if jump_times is None:
            object.__setattr__(
                self, "jump_times", jnp.asarray([], dtype=dtype_resolved)
            )
        else:
            object.__setattr__(
                self,
                "jump_times",
                _as_float_array("jump_times", jump_times, dtype_resolved, ndim=1),
            )

        if has_discrete:
            normalized_times = _as_float_array("times", times, dtype_resolved, ndim=1)
            normalized_values = _as_float_array(
                "values", values, dtype_resolved, ndim=1
            )
            object.__setattr__(self, "times", normalized_times)
            object.__setattr__(self, "values", normalized_values)
            if self.times.shape[0] != self.values.shape[0]:
                raise ValueError("times and values must have the same length")
            if not _is_strictly_increasing(self.times):
                raise ValueError("times must be strictly increasing")
        else:
            object.__setattr__(self, "times", None)
            object.__setattr__(self, "values", None)

        if has_spline:
            object.__setattr__(
                self,
                "breaks",
                _as_float_array("breaks", breaks, dtype_resolved, ndim=1),
            )
            object.__setattr__(
                self,
                "coeffs",
                _as_float_array("coeffs", coeffs, dtype_resolved, ndim=2),
            )
            object.__setattr__(
                self,
                "segment_start_piece_idx",
                _as_integral_index_1d(
                    "segment_start_piece_idx",
                    segment_start_piece_idx,
                ),
            )

            if self.coeffs.ndim != 2:
                raise ValueError("coeffs must be a 2D array")
            if self.coeffs.shape[1] != 4:
                raise ValueError("coeffs must have shape (n_pieces, 4)")
            if self.breaks.shape[0] != self.coeffs.shape[0] + 1:
                raise ValueError("len(breaks) must be len(coeffs) + 1")
            if self.coeffs.shape[0] == 0:
                raise ValueError(
                    "spline representation must contain at least one piece"
                )
            if not _is_strictly_increasing(self.breaks):
                raise ValueError("breaks must be strictly increasing")

            if self.segment_start_piece_idx.shape[0] == 0:
                raise ValueError("segment_start_piece_idx must not be empty")
            if int(self.segment_start_piece_idx[0]) != 0:
                raise ValueError("segment_start_piece_idx[0] must be 0")
            if not _is_strictly_increasing(self.segment_start_piece_idx):
                raise ValueError("segment_start_piece_idx must be strictly increasing")

            n_pieces = self.coeffs.shape[0]
            min_idx = int(jnp.min(self.segment_start_piece_idx))
            max_idx = int(jnp.max(self.segment_start_piece_idx))
            if min_idx < 0 or max_idx >= n_pieces:
                raise ValueError(
                    "segment_start_piece_idx entries out of valid piece range"
                )
        else:
            object.__setattr__(self, "breaks", None)
            object.__setattr__(self, "coeffs", None)
            object.__setattr__(self, "segment_start_piece_idx", None)

    @classmethod
    def from_dict(cls, data, *, dtype=None):
        from .io import timeseries_from_dict

        return timeseries_from_dict(cls, data, dtype=dtype)

    @classmethod
    def from_process_state(cls, process_state, variable):
        from .io import timeseries_from_process_state

        return timeseries_from_process_state(cls, process_state, variable)

    @classmethod
    def from_input_dict(cls, input_data, process_key, variable):
        from .io import timeseries_from_input_dict

        return timeseries_from_input_dict(cls, input_data, process_key, variable)

    def to_dict(self):
        from .io import timeseries_to_dict

        return timeseries_to_dict(self)

    def to_pd_series(self):
        if self.times is None or self.values is None:
            raise ValueError("to_pd_series requires discrete samples")
        return pd.Series(data=np.asarray(self.values), index=np.asarray(self.times))

    @property
    def poly(self) -> PPoly | None:
        """Return the owned spline evaluator for the canonical spline state."""
        if self.breaks is None or self.coeffs is None:
            return None
        return PPoly(self.breaks, self.coeffs, continuity_side=self.continuity_side)
    
    def lin_interp(self, t):
        if self.times is None or self.values is None:
            raise ValueError("lin_interp requires discrete samples")
        return grid_utils.linear_interpolate_samples(self.times, self.values, t)

    def evaluate(self, t, *, side=None):
        poly = self.poly
        if poly is None:
            raise ValueError("spline representation required for evaluation")
        return poly(t, side=side)

    def evaluate_many(self, ts, *, side=None):
        poly = self.poly
        if poly is None:
            raise ValueError("spline representation required for evaluation")
        ts_arr = jnp.asarray(ts, dtype=poly.breaks.dtype)
        if ts_arr.ndim != 1:
            raise ValueError("ts must be a 1D array")
        return poly(ts_arr, side=side)

    def deriv(self, order: int = 1):
        order = int(order)
        poly = self.poly
        if poly is None:
            raise ValueError("spline representation required for derivative")
        new_poly = poly.derivative(order=order)
        return TimeSeries(
            derived=True,
            jump_times=self.jump_times,
            breaks=new_poly.breaks,
            coeffs=new_poly.coeffs,
            segment_start_piece_idx=self.segment_start_piece_idx,
            continuity_side=self.continuity_side,
            metadata=self.metadata,
            dtype=self.dtype,
        )

    def integrate(self, a, b):
        if self.breaks is None or self.coeffs is None:
            raise ValueError("spline representation required for integration")
        return spline_ops.integrate_definite(self.breaks, self.coeffs, a, b)

    def _has_spline(self) -> bool:
        return self.breaks is not None and self.coeffs is not None

    def _has_discrete(self) -> bool:
        return self.times is not None and self.values is not None

    def __add__(self, other):
        if not isinstance(other, TimeSeries):
            return NotImplemented
        if not (self._has_spline() and other._has_spline()):
            return self._binary_discrete(other, op="add")
        return self._binary_exact(other, op="add")

    def __sub__(self, other):
        if not isinstance(other, TimeSeries):
            return NotImplemented
        if not (self._has_spline() and other._has_spline()):
            return self._binary_discrete(other, op="sub")
        return self._binary_exact(other, op="sub")

    def __mul__(self, other):
        if isinstance(other, TimeSeries):
            if not (self._has_spline() and other._has_spline()):
                return self._binary_discrete(other, op="mul")
            return self._binary_approx(other, op="mul")
        scalar = float(other)
        if self.breaks is not None and self.coeffs is not None:
            out_times = self.times
            out_values = None if self.values is None else self.values * scalar
            return TimeSeries(
                times=out_times,
                values=out_values,
                derived=True,
                jump_times=self.jump_times,
                breaks=self.breaks,
                coeffs=self.coeffs * scalar,
                segment_start_piece_idx=self.segment_start_piece_idx,
                continuity_side=self.continuity_side,
                metadata=self.metadata,
                dtype=self.dtype,
            )
        if self.times is None or self.values is None:
            raise ValueError(
                "scalar multiply requires spline representation or discrete samples"
            )
        return TimeSeries(
            times=self.times,
            values=self.values * scalar,
            derived=True,
            jump_times=self.jump_times,
            continuity_side=self.continuity_side,
            metadata=self.metadata,
            dtype=self.dtype,
        )

    def __truediv__(self, other):
        if isinstance(other, TimeSeries):
            if not (self._has_spline() and other._has_spline()):
                return self._binary_discrete(other, op="div")
            return self._binary_approx(other, op="div")
        scalar = float(other)
        if scalar == 0.0:
            raise ZeroDivisionError("division by zero scalar")
        if self.breaks is not None and self.coeffs is not None:
            out_times = self.times
            out_values = None if self.values is None else self.values / scalar
            return TimeSeries(
                times=out_times,
                values=out_values,
                derived=True,
                jump_times=self.jump_times,
                breaks=self.breaks,
                coeffs=self.coeffs / scalar,
                segment_start_piece_idx=self.segment_start_piece_idx,
                continuity_side=self.continuity_side,
                metadata=self.metadata,
                dtype=self.dtype,
            )
        if self.times is None or self.values is None:
            raise ValueError(
                "scalar divide requires spline representation or discrete samples"
            )
        return TimeSeries(
            times=self.times,
            values=self.values / scalar,
            derived=True,
            jump_times=self.jump_times,
            continuity_side=self.continuity_side,
            metadata=self.metadata,
            dtype=self.dtype,
        )

    def _binary_discrete(self, other: "TimeSeries", op: str) -> "TimeSeries":
        if self.dtype != other.dtype:
            raise TypeError(f"TimeSeries dtype mismatch: {self.dtype} vs {other.dtype}")
        if not self._has_discrete() or not other._has_discrete():
            raise ValueError(
                "binary operation without spline requires both operands to "
                "have discrete samples"
            )
        self_has_spline = self._has_spline()
        other_has_spline = other._has_spline()
        if self_has_spline != other_has_spline:
            warnings.warn(
                "Mixed spline/non-spline operands; result drops spline representation.",
                UserWarning,
                stacklevel=2,
            )
        out_times = grid_utils.merge_times_with_tolerance(self.times, other.times)
        left_discrete = grid_utils.linear_interpolate_samples(
            self.times, self.values, out_times
        )
        right_discrete = grid_utils.linear_interpolate_samples(
            other.times, other.values, out_times
        )
        if op == "add":
            out_values = left_discrete + right_discrete
        elif op == "sub":
            out_values = left_discrete - right_discrete
        elif op == "mul":
            out_values = left_discrete * right_discrete
        elif op == "div":
            if jnp.any(jnp.abs(right_discrete) <= DIVISION_NEAR_ZERO_THRESHOLD):
                raise ZeroDivisionError("discrete denominator near zero")
            out_values = left_discrete / right_discrete
        else:
            raise ValueError("invalid discrete binary operation")
        out_jumps = np.unique(
            np.concatenate(
                [
                    np.asarray(self.jump_times, dtype=np.dtype(self.dtype)),
                    np.asarray(other.jump_times, dtype=np.dtype(self.dtype)),
                ]
            )
        )
        return TimeSeries(
            times=out_times,
            values=out_values,
            derived=True,
            jump_times=jnp.asarray(out_jumps, dtype=self.dtype),
            continuity_side=self.continuity_side,
            metadata={"source": "discrete_binary_op", "op": op},
            dtype=self.dtype,
        )

    def _binary_exact(self, other: "TimeSeries", op: str) -> "TimeSeries":
        if self.dtype != other.dtype:
            raise TypeError(f"TimeSeries dtype mismatch: {self.dtype} vs {other.dtype}")
        if self.breaks is None or self.coeffs is None:
            raise ValueError("left operand missing spline representation")
        if other.breaks is None or other.coeffs is None:
            raise ValueError("right operand missing spline representation")
        if self.continuity_side != other.continuity_side:
            raise ValueError(
                "binary operations require matching continuity_side on both operands"
            )

        merged_breaks = spline_ops.merge_breaks(self.breaks, other.breaks)
        left_coeffs = spline_ops.rebase_to_breaks(
            self.breaks, self.coeffs, merged_breaks
        )
        right_coeffs = spline_ops.rebase_to_breaks(
            other.breaks, other.coeffs, merged_breaks
        )
        if op == "add":
            out_coeffs = left_coeffs + right_coeffs
        elif op == "sub":
            out_coeffs = left_coeffs - right_coeffs
        else:
            raise ValueError("invalid binary operation")

        out_starts = spline_ops.merge_segment_starts(
            self.breaks,
            self.segment_start_piece_idx,
            other.breaks,
            other.segment_start_piece_idx,
            merged_breaks,
        )
        out_jumps = np.unique(
            np.concatenate(
                [
                    np.asarray(self.jump_times, dtype=np.dtype(self.dtype)),
                    np.asarray(other.jump_times, dtype=np.dtype(self.dtype)),
                ]
            )
        )
        out_times, out_values = grid_utils.synthesize_binary_samples(
            self.times,
            self.values,
            other.times,
            other.values,
            op=op,
        )
        return TimeSeries(
            times=out_times,
            values=out_values,
            derived=True,
            jump_times=jnp.asarray(out_jumps, dtype=self.dtype),
            breaks=merged_breaks,
            coeffs=out_coeffs,
            segment_start_piece_idx=out_starts,
            continuity_side=self.continuity_side,
            metadata={"source": "exact_binary_op", "op": op},
            dtype=self.dtype,
        )

    def _binary_approx(self, other: "TimeSeries", op: str) -> "TimeSeries":
        if self.dtype != other.dtype:
            raise TypeError(f"TimeSeries dtype mismatch: {self.dtype} vs {other.dtype}")
        if self.breaks is None or self.coeffs is None:
            raise ValueError("left operand missing spline representation")
        if other.breaks is None or other.coeffs is None:
            raise ValueError("right operand missing spline representation")
        if self.continuity_side != other.continuity_side:
            raise ValueError(
                "binary operations require matching continuity_side on both operands"
            )

        fitting_grid = spline_ops.merge_breaks(self.breaks, other.breaks)
        if self.times is not None:
            fitting_grid = grid_utils.merge_times_with_tolerance(
                fitting_grid, self.times
            )
        if other.times is not None:
            fitting_grid = grid_utils.merge_times_with_tolerance(
                fitting_grid, other.times
            )

        left_vals = np.asarray(self.evaluate_many(fitting_grid), dtype=np.float64)
        right_vals = np.asarray(other.evaluate_many(fitting_grid), dtype=np.float64)
        if op == "mul":
            target_vals = left_vals * right_vals
        elif op == "div":
            if spline_ops.has_near_zero_piece_value(
                other.breaks,
                other.coeffs,
                threshold=DIVISION_NEAR_ZERO_THRESHOLD,
            ):
                raise ZeroDivisionError("denominator crosses or approaches zero")
            if np.any(np.abs(right_vals) <= DIVISION_NEAR_ZERO_THRESHOLD):
                raise ZeroDivisionError("denominator crosses or approaches zero")
            target_vals = left_vals / right_vals
        else:
            raise ValueError("invalid approx binary operation")

        breaks, coeffs = self._fit_cubic_power_basis(
            np.asarray(fitting_grid, dtype=np.float64),
            target_vals,
        )

        out_times = None
        out_values = None
        if self.times is not None and self.values is not None:
            if other.times is not None and other.values is not None:
                out_times = grid_utils.merge_times_with_tolerance(
                    self.times, other.times
                )
                left_discrete = grid_utils.linear_interpolate_samples(
                    self.times, self.values, out_times
                )
                right_discrete = grid_utils.linear_interpolate_samples(
                    other.times, other.values, out_times
                )
                if op == "mul":
                    out_values = left_discrete * right_discrete
                else:
                    if jnp.any(jnp.abs(right_discrete) <= DIVISION_NEAR_ZERO_THRESHOLD):
                        raise ZeroDivisionError("discrete denominator near zero")
                    out_values = left_discrete / right_discrete

        out_jumps = np.unique(
            np.concatenate(
                [
                    np.asarray(self.jump_times, dtype=np.dtype(self.dtype)),
                    np.asarray(other.jump_times, dtype=np.dtype(self.dtype)),
                ]
            )
        )
        return TimeSeries(
            times=out_times,
            values=out_values,
            derived=True,
            jump_times=jnp.asarray(out_jumps, dtype=self.dtype),
            breaks=jnp.asarray(breaks, dtype=self.dtype),
            coeffs=jnp.asarray(coeffs, dtype=self.dtype),
            segment_start_piece_idx=jnp.asarray([0], dtype=jnp.int32),
            continuity_side=self.continuity_side,
            metadata={"source": "approx_binary_op", "op": op},
            dtype=self.dtype,
        )

    def _fit_cubic_power_basis(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.dtype(self.dtype))
        y_arr = np.asarray(y_true, dtype=np.dtype(self.dtype))
        if x_arr.shape[0] < 2:
            raise ValueError("need at least two grid points for approximate refit")
        scale = max(float(np.ptp(y_arr)), APPROX_ABS_FLOOR)
        s_val = APPROX_INITIAL_S
        degree = min(3, int(x_arr.shape[0] - 1))

        for _ in range(APPROX_MAX_REFIT_ATTEMPTS):
            bspline = make_splrep(x_arr, y_arr, k=degree, s=s_val)
            poly = PPoly.from_scipy_ppoly(
                SciPyPPoly.from_spline(bspline),
                continuity_side=self.continuity_side,
            )
            breaks, coeffs = poly.breaks, poly.coeffs
            probe = TimeSeries(
                breaks=breaks,
                coeffs=coeffs,
                segment_start_piece_idx=[0],
                continuity_side=self.continuity_side,
                dtype=self.dtype,
            )
            y_fit = np.asarray(probe.evaluate_many(x_arr), dtype=np.dtype(self.dtype))
            rel_err = float(np.mean(np.abs(y_fit - y_arr)) / scale)
            if rel_err <= APPROX_REL_ERR_TARGET:
                return np.asarray(breaks, dtype=np.dtype(self.dtype)), np.asarray(
                    coeffs, dtype=np.dtype(self.dtype)
                )
            s_val *= APPROX_S_REDUCTION_FACTOR

        raise ValueError("approximate spline refit did not meet error tolerance")
