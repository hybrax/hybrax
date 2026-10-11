"""Adaptive monotone cubic spline fitting under a root-mean-square error budget.

The fit is a clamped cubic B-spline in normalized coordinates
`x = (t - t0) / T` and `z = (y - y0) / D`, where `D` is the signed net change.
Coefficients are anchored at 0 and 1 and strictly increasing, which makes the
curve monotone in the direction of `D` (a sufficient, not necessary, condition).

Knots are chosen by refinement: for the current knots, a second-order cone
program finds the smallest achievable RMS error, and new knots go where the
squared residuals of that fit concentrate. Once the budget is met, a QP picks
the curve with the least integrated squared second derivative within the
budget. All solves use Clarabel, with variables scaled by the budget so that
solver tolerances are relative to it.
"""

from __future__ import annotations

from typing import Any

import clarabel
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.interpolate import PPoly as SciPyPPoly

from .time_series import TimeSeries, spline_ops

# Refinement stops once the smallest achievable RMS error is below this fraction
# of the budget, which leaves the smoothing QP room inside the full budget.
_ACCEPT_FRACTION = 0.95
# The smoothing budget is at most this far inside the full budget (relative).
_QP_MARGIN = 1e-3
# Minimum normalized coefficient increment. A zero lower bound produces rates of
# about -1e-20 on flat stretches after power-basis conversion, which strict sign
# checks reject.
_INCREMENT_FLOOR = 1e-9
# Knots are inserted in every interval whose residual score is at least this
# fraction of the worst interval's score.
_BATCH_FRACTION = 0.1
_MIN_KNOT_GAP = 1e-7
_MIN_DEFAULT_KNOTS = 16

# Metadata written by spline fits; replaced when a series is refitted.
_FIT_METADATA_KEYS = frozenset(
    {
        "smoothing_s",
        "actual_segments",
        "fit_strategy",
        "fit_strategies",
        "segment_boundaries",
        "smoothing_storage",
        "smoothing_bc_type",
        "interp_segment_bc_type",
        "is_constant",
        "constant_value",
        "monotone_fit",
    }
)


def fit_monotone_timeseries_spline(
    ts: TimeSeries,
    *,
    atol: float,
    max_knots: int | None = None,
) -> TimeSeries:
    """Fit a smooth monotone C2 cubic spline with automatically chosen knots.

    The root-mean-square error at the samples is at most `atol`:
    `sqrt(mean((f(t_i) - y_i) ** 2)) <= atol`, in the units of the values. Both
    endpoints are reproduced exactly up to roundoff, so the derivative
    integrates to the measured net change. The direction (increasing or
    decreasing) follows from the endpoints, and the derivative has that sign
    everywhere in `[t[0], t[-1]]`. Equal endpoints give a constant curve, which
    is accepted only if it meets the budget.

    Knots are added where the residuals concentrate until the budget can be met.
    For the chosen knots, the curve with the smallest integrated squared second
    derivative within the budget is returned, so the fit generally uses most of
    the budget and smooths noise and short reversals of direction. The error at
    individual samples can exceed `atol`. Raw samples are kept unchanged.

    A budget below the noise level of the data makes the fit follow the noise
    with many knots. `max_knots` stops this with an error, so choose `atol` at
    least as large as the measurement noise.

    Args:
        ts: Scalar series with at least two samples and no jumps. Any existing
            spline state is ignored and replaced.
        atol: RMS error budget, `> 0`.
        max_knots: Maximum number of interior knots. Defaults to a tenth of the
            number of samples, but at least 16.

    Returns:
        A new TimeSeries with the same samples, continuity side and unrelated
        metadata, holding the fitted spline. Fit details are stored under the
        `"monotone_fit"` metadata key.

    Raises:
        ValueError: For invalid input, or if the endpoints are equal and the
            constant curve misses the budget.
        RuntimeError: If refinement reaches `max_knots` or cannot place further
            knots before the budget can be met (usually because `atol` is below
            the noise level), or if no solution passes the final checks.
    """
    times = np.asarray(ts.times, dtype=np.float64)
    values = np.asarray(ts.values, dtype=np.float64)
    _validate_input(ts, times, values, atol)
    if max_knots is None:
        max_knots = max(_MIN_DEFAULT_KNOTS, len(times) // 10)
    if max_knots < 0:
        raise ValueError("max_knots must be nonnegative.")

    y0, net = values[0], values[-1] - values[0]
    info: dict[str, Any] = {"strategy": "adaptive_monotone_bspline", "atol": atol}
    if net == 0.0:
        rms = float(np.sqrt(np.mean((values - y0) ** 2)))
        if rms > atol:
            raise ValueError(
                "Endpoints are equal, so the fit is constant, but its RMS error "
                f"{rms:.3g} exceeds atol={atol:.3g}."
            )
        breaks = np.array([times[0], times[-1]])
        coeffs = np.array([[y0, 0.0, 0.0, 0.0]])
        info |= {"direction": "constant", "interior_knots": 0, "refinements": 0}
    else:
        breaks, coeffs, fit_info = _fit_nonconstant(times, values, atol, max_knots)
        info |= {"direction": "increasing" if net > 0 else "decreasing"} | fit_info

    residuals = _evaluate(breaks, coeffs, times) - values
    info["rms_error"] = float(np.sqrt(np.mean(residuals**2)))
    info["max_error"] = float(np.max(np.abs(residuals)))
    metadata = {
        key: value
        for key, value in dict(ts.metadata or {}).items()
        if key not in _FIT_METADATA_KEYS
    }
    metadata["monotone_fit"] = info
    return TimeSeries(
        times=ts.times,
        values=ts.values,
        derived=ts.derived,
        continuity_side=ts.continuity_side,
        breaks=jnp.asarray(breaks, dtype=jnp.float64),
        coeffs=jnp.asarray(coeffs, dtype=jnp.float64),
        segment_start_piece_idx=jnp.array([0], dtype=jnp.int32),
        metadata=metadata,
    )


def _validate_input(
    ts: TimeSeries, times: np.ndarray, values: np.ndarray, atol: float
) -> None:
    if not (np.isfinite(atol) and atol > 0):
        raise ValueError("atol must be finite and positive.")
    if times.ndim != 1 or values.shape != times.shape:
        raise ValueError("Only scalar-valued series are supported.")
    if len(times) < 2:
        raise ValueError("At least two samples are required.")
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(values))):
        raise ValueError("Times and values must be finite.")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Times must be strictly increasing.")
    if ts.jump_times is not None and np.asarray(ts.jump_times).size:
        raise ValueError("Series with jumps are not supported.")


def _fit_nonconstant(
    times: np.ndarray, values: np.ndarray, atol: float, max_knots: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    t0, span = times[0], times[-1] - times[0]
    y0, net = values[0], values[-1] - values[0]
    x = (times - t0) / span
    z = (values - y0) / net
    budget = atol / abs(net)  # Normalized RMS budget.

    interior = np.empty(0)
    refinements = 0
    while True:
        knots = np.r_[[0.0] * 4, interior, [1.0] * 4]
        design = BSpline.design_matrix(x, knots, 3).tocsc()
        c_best = _least_rms(design, z, budget)
        refinements += 1
        residual = design @ c_best - z
        rms = float(np.sqrt(np.mean(residual**2))) / budget
        if rms <= _ACCEPT_FRACTION:
            break
        limit = max_knots - len(interior)
        new = _choose_knots(x, residual, interior, limit) if limit > 0 else []
        if not new:
            if rms <= 1.0:
                break  # The margin is out of reach but the budget is met.
            reason = (
                f"reached max_knots={max_knots}"
                if limit <= 0
                else "cannot place further knots"
            )
            raise RuntimeError(
                f"Monotone fit {reason} with {len(interior)} interior knots; its RMS "
                f"error {rms * atol:.3g} still exceeds atol={atol:.3g}. A budget below "
                "the noise level of the data needs a knot for nearly every sample."
            )
        interior = np.sort(np.r_[interior, new])

    smoothing_budget = 1.0 - min(_QP_MARGIN, (1.0 - rms) / 2.0)
    candidates = []
    c_smooth = _smoothest(design, z, budget, smoothing_budget, knots)
    if c_smooth is not None:
        candidates.append(("smoothed", c_smooth))
    candidates.append(("least_rms", c_best))
    failures = []
    for source, c in candidates:
        breaks, coeffs = _to_power_basis(knots, _project(c), t0, span, y0, net)
        breaks[-1] = times[-1]
        problem = _check_stored(breaks, coeffs, times, values, atol, np.sign(net))
        if problem is None:
            info = {
                "interior_knots": len(interior),
                "refinements": refinements,
                "coefficient_source": source,
                "increment_floor": _INCREMENT_FLOOR,
            }
            return breaks, coeffs, info
        failures.append(f"{source}: {problem}")
    raise RuntimeError("Monotone fit failed final checks: " + "; ".join(failures))


def _solve_conic(
    design: sparse.csc_matrix,
    z: np.ndarray,
    budget: float,
    P: sparse.csc_matrix | None,
    rms_bound: float | None,
) -> np.ndarray | None:
    """Solve a monotone fitting problem in `u = c / budget` with Clarabel.

    With `rms_bound=None`, minimize the RMS error (an auxiliary variable `r`
    bounds the residual norm). Otherwise minimize `u'Pu/2` subject to an RMS
    error of at most `rms_bound` budgets. Returns `c` or `None` on failure.
    """
    m, n = design.shape
    extra = 1 if rms_bound is None else 0
    zs = z / budget
    anchors = sparse.csc_matrix(
        ([1.0, 1.0], ([0, 1], [0, n - 1])), shape=(2, n + extra)
    )
    increments = sparse.diags(
        [np.ones(n - 1), -np.ones(n - 1)], [0, 1], shape=(n - 1, n + extra)
    )
    # Second-order cone: (norm bound, zs - A u) with ||zs - A u|| <= norm bound.
    norm_row = sparse.csc_matrix(
        ([-1.0], ([0], [n])) if extra else ([], ([], [])), shape=(1, n + extra)
    )
    residual_rows = sparse.hstack([design, sparse.csc_matrix((m, extra))])
    A = sparse.vstack([anchors, increments, norm_row, residual_rows], format="csc")
    norm_bound = 0.0 if rms_bound is None else rms_bound * np.sqrt(m)
    floor = np.full(n - 1, -_INCREMENT_FLOOR / budget)
    b = np.r_[0.0, 1.0 / budget, floor, norm_bound, zs]
    cones = [
        clarabel.ZeroConeT(2),
        clarabel.NonnegativeConeT(n - 1),
        clarabel.SecondOrderConeT(m + 1),
    ]
    q = np.zeros(n + extra)
    if rms_bound is None:
        q[-1] = 1.0
        P = sparse.csc_matrix((n + extra, n + extra))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    solution = clarabel.DefaultSolver(P, q, A, b, cones, settings).solve()
    if solution.status != clarabel.SolverStatus.Solved:
        return None
    return np.asarray(solution.x)[:n] * budget


def _least_rms(design: sparse.csc_matrix, z: np.ndarray, budget: float) -> np.ndarray:
    c = _solve_conic(design, z, budget, None, None)
    if c is None:
        raise RuntimeError("Monotone fit: least-RMS problem did not solve.")
    return c


def _curvature_gram(knots: np.ndarray) -> np.ndarray:
    """Return the matrix of integrals of products of basis second derivatives.

    Second derivatives are linear on each knot interval, so two-point Gauss
    quadrature per interval is exact.
    """
    n = len(knots) - 4
    lo, hi = knots[3:-4], knots[4:-3]
    keep = hi > lo
    mid, half = (lo[keep] + hi[keep]) / 2, (hi[keep] - lo[keep]) / 2
    offset = half / np.sqrt(3.0)
    nodes = np.r_[mid - offset, mid + offset]
    weights = np.r_[half, half]
    second = BSpline(knots, np.eye(n), 3).derivative(2)(nodes)
    return second.T @ (weights[:, None] * second)


def _smoothest(
    design: sparse.csc_matrix,
    z: np.ndarray,
    budget: float,
    rms_bound: float,
    knots: np.ndarray,
) -> np.ndarray | None:
    """Minimize curvature subject to the RMS bound and the monotone constraints."""
    gram = _curvature_gram(knots)
    # Rescaling leaves the minimizer unchanged; without it Clarabel can fail on
    # clustered knots (diagonal entries spanning ~1e9).
    gram /= np.max(np.diag(gram))
    P = sparse.triu(sparse.csc_matrix(2.0 * gram), format="csc")
    return _solve_conic(design, z, budget, P, rms_bound)


def _project(c: np.ndarray) -> np.ndarray:
    """Enforce `c[0] = 0`, `c[-1] = 1` and increments `>= floor` exactly."""
    floor = _INCREMENT_FLOOR
    increments = np.maximum(np.diff(c), floor)
    excess = increments - floor
    increments = floor + excess * (1.0 - floor * len(increments)) / excess.sum()
    projected = np.r_[0.0, np.cumsum(increments)]
    projected[-1] = 1.0
    return projected


def _choose_knots(
    x: np.ndarray, residual: np.ndarray, interior: np.ndarray, limit: int
) -> list[float]:
    """Pick new knots in the intervals with the largest sum of squared residuals.

    A new knot goes at the median sample of its interval, or at the midpoint
    when the interval holds two or fewer samples.
    """
    edges = np.r_[0.0, interior, 1.0]
    n_intervals = len(edges) - 1
    index = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_intervals - 1)
    score = np.bincount(index, residual**2, minlength=n_intervals)
    new: list[float] = []
    for index in np.argsort(-score):
        if len(new) == limit or score[index] < _BATCH_FRACTION * score.max():
            break
        a, b = edges[index], edges[index + 1]
        inside = x[(x > a) & (x < b)]
        candidates = [float(np.median(inside))] if len(inside) > 2 else []
        candidates.append((a + b) / 2.0)
        for knot in candidates:
            if min(knot - a, b - knot) > _MIN_KNOT_GAP:
                new.append(knot)
                break
    return new


def _to_power_basis(
    knots: np.ndarray,
    c: np.ndarray,
    t0: float,
    span: float,
    y0: float,
    net: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert to physical local ascending-power coefficients.

    Conversion happens in normalized space; the value offset enters only the
    constant coefficient, so it cannot perturb the derivative.
    """
    ppoly = SciPyPPoly.from_spline(BSpline(knots, c, 3))
    keep = np.diff(ppoly.x) > 0
    breaks = np.r_[ppoly.x[:-1][keep], ppoly.x[-1]]
    a3, a2, a1, a0 = ppoly.c[:, keep]  # SciPy stores descending powers.
    coeffs = np.stack(
        [y0 + net * a0, net * a1 / span, net * a2 / span**2, net * a3 / span**3],
        axis=1,
    )
    return t0 + span * breaks, coeffs


def _evaluate(breaks: np.ndarray, coeffs: np.ndarray, t: np.ndarray) -> np.ndarray:
    index = np.clip(np.searchsorted(breaks, t, side="right") - 1, 0, len(coeffs) - 1)
    dt = t - breaks[index]
    a = coeffs[index]
    return a[:, 0] + dt * (a[:, 1] + dt * (a[:, 2] + dt * a[:, 3]))


def _check_stored(
    breaks: np.ndarray,
    coeffs: np.ndarray,
    times: np.ndarray,
    values: np.ndarray,
    atol: float,
    sign: float,
) -> str | None:
    """Return a problem description if the stored spline breaks a guarantee."""
    if not (np.all(np.isfinite(breaks)) and np.all(np.isfinite(coeffs))):
        return "nonfinite coefficients"
    if np.any(np.diff(breaks) <= 0):
        return "collapsed breakpoints"
    fitted = _evaluate(breaks, coeffs, times)
    rms = np.sqrt(np.mean((fitted - values) ** 2))
    if rms > atol:
        return f"RMS error {rms:.3g} exceeds atol={atol:.3g}"
    # Roundoff at the end scales with the size of the last piece's terms there,
    # plus its slope times the time resolution of the pinned last break.
    a = coeffs[-1]
    width = breaks[-1] - breaks[-2]
    end_rate = abs(a[1] + 2 * a[2] * width + 3 * a[3] * width**2)
    tolerance = (
        64
        * np.finfo(np.float64).eps
        * (max(np.sum(np.abs(a * width ** np.arange(4))), abs(values[0])))
        + 4 * np.spacing(abs(breaks[-1])) * end_rate
    )
    endpoint_error = max(abs(fitted[0] - values[0]), abs(fitted[-1] - values[-1]))
    if endpoint_error > tolerance:
        return f"endpoint error {endpoint_error:.3g}"
    rates = spline_ops.piece_rate_candidates(breaks, coeffs, times[0], times[-1])
    if np.any(sign * rates < 0.0):
        return "derivative changes sign"
    return None
