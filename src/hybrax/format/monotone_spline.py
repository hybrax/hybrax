"""Adaptive monotone cubic spline fitting under a root-mean-square error budget.

The fit is a clamped cubic B-spline in normalized coordinates
`x = (t - t0) / T` and `z = (y - y0) / D`, where `D` is the signed net change.
Coefficients are anchored at 0 and 1 and strictly increasing, which makes the
curve monotone in the direction of `D` (a sufficient, not necessary, condition).

Knots are chosen by refinement: for the current knots, a second-order cone
program finds the smallest achievable RMS error, and new knots go where the
squared residuals of that fit concentrate. An unconstrained pilot guides each
insertion using its curvature variation, with a knot target set by max_knots.
Once the budget is met, a QP
minimizes the integrated squared second derivative among curves with ordered
B-spline coefficients within the budget. This minimization is numerical: with
strongly clustered samples the result can be rougher than necessary. All
solves use Clarabel
in normalized coefficients, with the RMS constraint scaled by the budget so
that its tolerance stays relative to it.
"""

from __future__ import annotations

import warnings
from threading import Timer
from typing import Any

import clarabel
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline, generate_knots, make_lsq_spline, make_splrep
from scipy.interpolate import PPoly as SciPyPPoly
from scipy.optimize import isotonic_regression

from ._logging import get_logger
from .time_series import TimeSeries, spline_ops

_logger = get_logger(__name__)
_SLOW_FIT_SECONDS = 20.0

# Refinement stops once the smallest achievable RMS error is below this fraction
# of the budget, which leaves the smoothing QP room inside the full budget.
_ACCEPT_FRACTION = 0.95
# The smoothing budget is at most this far inside the full budget (relative).
_QP_MARGIN = 1e-3
# Minimum normalized coefficient increment, applied as a solver constraint and
# enforced again by `_project`. It prevents roundoff-level reversed rates on
# flat stretches after power-basis conversion.
_INCREMENT_FLOOR = 1e-9
# Knots are inserted in every interval whose residual score is at least this
# fraction of the worst interval's score.
_BATCH_FRACTION = 0.1
# Minimum gap between a new knot and its interval ends in normalized time.
_MIN_KNOT_GAP = 1e-7
# Minimum default cap, so short series can resolve sharp features even with
# safeguarded splits that need several refinements to reach a narrow event.
_MIN_DEFAULT_KNOTS = 64
# Default cap is the sample count divided by this, with the minimum above.
_DEFAULT_KNOT_DIVISOR = 10
# Use half the remaining RMS slack when it is smaller than the QP margin.
_QP_SLACK_DIVISOR = 2.0
# Roundoff allowance for endpoint and value-continuity polynomial evaluation.
_ROUNDOFF_FACTOR = 64
# Allow endpoint drift from a few ulps in the pinned final time.
_ENDPOINT_TIME_ULPS = 4
# Keep each new knot this fraction of its interval away from either end, so
# a split's two child intervals differ in width by at most a factor of three.
_SPLIT_GUARD = 0.25
# Cubic pilots need at least four observations.
_MIN_PILOT_SAMPLES = 4
# Use the midpoint without guide mass when at most two interior samples remain.
_MEDIAN_SAMPLE_THRESHOLD = 2
# Adjacent pieces may disagree by this fraction of the net change. Rounding of the
# stored physical breakpoints causes small mismatches when times have an offset;
# below this level they are irrelevant.
_CONTINUITY_RTOL = 1e-9
# Times may be at most this many durations away from zero. Larger offsets (e.g.
# epoch timestamps) make breakpoint rounding visible; subtract the start time.
_MAX_TIME_OFFSET_RATIO = 1000
_RMS_TOL_GUIDANCE = (
    "As a rule of thumb, choose rms_tol of at least 1.5 times the "
    "expected measurement-noise standard deviation, without exceeding the "
    "largest RMS deviation acceptable for your application."
)


def fit_monotone_timeseries_spline(
    ts: TimeSeries,
    *,
    rms_tol: float,
    increasing: bool,
    max_knots: int | None = None,
) -> TimeSeries:
    """Fit a smooth monotone C2 cubic spline with automatically chosen knots.

    The root-mean-square error at the samples is at most `rms_tol`:
    `sqrt(mean((f(t_i) - y_i) ** 2)) <= rms_tol`, in the units of the values. Both
    endpoints are reproduced exactly up to roundoff, so the derivative
    integrates to the measured net change. The requested direction must agree
    with the endpoints. For unequal endpoints, the derivative is strictly
    positive or negative everywhere in `[t[0], t[-1]]`. Consecutive B-spline
    coefficients differ by at least 1e-9 of the net change, so flat stretches
    get a negligible nonzero rate, about 1e-9 of the net change per knot interval.
    Equal endpoints give a constant curve for either direction, accepted only if
    it meets the budget.

    Knots are added where the residuals concentrate until the budget can be met.
    An unconstrained pilot guides their locations throughout refinement. It uses
    the same RMS target and max_knots as its interior-knot target, but can miss
    that RMS target when its layout is too small. It only proposes locations;
    the returned monotone curve must meet the requested budget.
    For the chosen knots, the smoothing step numerically minimizes the integrated
    squared second derivative among curves with ordered B-spline coefficients
    within the budget, so the fit generally uses most of the budget and smooths
    noise and short reversals of direction. The error at individual samples can
    exceed `rms_tol`. Raw samples are kept unchanged.

    The budget, monotonicity, continuity and endpoints are checked on the stored
    curve: the function either returns a curve that meets them or raises.
    Smoothness is not checked: with strongly clustered samples, such as bursts of
    fast logging, the smoothing can fail or return a curve rougher than
    necessary. Thinning redundant burst readings helps, but keep enough readings
    to resolve short additions; the budget then covers the remaining samples.

    Larger `rms_tol` gives a smoother curve and rate; it is the smoothing knob.
    Choose it at least about 1.5 times the measurement noise standard deviation.
    At about the noise standard deviation, the fit is unstable: it can switch
    between few knots, the cap and an error. Every sample counts equally, so
    densely logged stretches weigh more towards `rms_tol`.

    Args:
        ts: Scalar series with at least two samples and no jumps. Any existing
            spline state is ignored and replaced. Times must lie within 1000
            durations of zero, so pass time since start rather than absolute
            timestamps.
        rms_tol: RMS error budget, `> 0`.
        increasing: Whether the curve must increase or decrease.
        max_knots: Maximum number of interior knots. Defaults to a tenth of the
            number of samples, but at least 64. Also sets the pilot knot target.

    Returns:
        A new TimeSeries with the same samples and continuity side, holding the
        fitted spline. Input metadata is discarded. Fit details are stored under
        the `"monotone_fit"` metadata key.

    Raises:
        TypeError: If `increasing` is not a bool.
        ValueError: For invalid input, or if the endpoints are equal and the
            constant curve misses the budget, or the endpoints contradict
            `increasing`.
        RuntimeError: If the closest monotone sequence exceeds the RMS budget,
            if refinement reaches `max_knots` or cannot place further
            knots before the budget can be met, or if a solve or
            final check fails.
    """
    times = np.asarray(ts.times, dtype=np.float64)
    values = np.asarray(ts.values, dtype=np.float64)
    _validate_input(ts, times, values, rms_tol)
    if max_knots is None:
        max_knots = max(_MIN_DEFAULT_KNOTS, len(times) // _DEFAULT_KNOT_DIVISOR)
    if (
        isinstance(max_knots, bool)
        or not isinstance(max_knots, (int, np.integer))
        or max_knots < 0
    ):
        raise ValueError("max_knots must be a nonnegative integer.")
    if not isinstance(increasing, (bool, np.bool_)):
        raise TypeError("increasing must be a bool.")

    y0, net = values[0], values[-1] - values[0]
    if (increasing and net < 0) or (not increasing and net > 0):
        raise ValueError(
            f"Endpoint values {values[0]} and {values[-1]} contradict "
            f"increasing={increasing}."
        )
    info: dict[str, Any] = {
        "rms_tol": float(rms_tol),
        "direction": "increasing" if increasing else "decreasing",
    }
    if net == 0.0:
        rms = float(np.sqrt(np.mean(((values - y0) / rms_tol) ** 2)))
        if rms > 1.0:
            raise ValueError(
                "Endpoints are equal, so the fit is constant, but its RMS error "
                f"{rms * rms_tol:.3g} exceeds rms_tol={rms_tol:.3g}."
            )
        breaks = np.array([times[0], times[-1]])
        coeffs = np.array([[y0, 0.0, 0.0, 0.0]])
        info |= {"interior_knots": 0, "refinements": 0}
    else:
        timer = Timer(
            _SLOW_FIT_SECONDS, _log_slow_fit, args=(len(times), rms_tol, max_knots)
        )
        timer.daemon = True
        timer.start()
        try:
            breaks, coeffs, fit_info = _fit_nonconstant(
                times, values, rms_tol, max_knots, ts.continuity_side
            )
        finally:
            timer.cancel()
        info |= fit_info

    residuals = _evaluate(breaks, coeffs, times, ts.continuity_side) - values
    info["rms_error"] = float(np.sqrt(np.mean((residuals / rms_tol) ** 2)) * rms_tol)
    info["max_error"] = float(np.max(np.abs(residuals)))
    return TimeSeries(
        times=ts.times,
        values=ts.values,
        derived=ts.derived,
        continuity_side=ts.continuity_side,
        breaks=jnp.asarray(breaks, dtype=jnp.float64),
        coeffs=jnp.asarray(coeffs, dtype=jnp.float64),
        segment_start_piece_idx=jnp.array([0], dtype=jnp.int32),
        metadata={"monotone_fit": info},
    )


def _log_slow_fit(samples: int, rms_tol: float, max_knots: int) -> None:
    """Emit one progress notice; elapsed time alone does not diagnose conditioning."""
    _logger.info(
        "Monotone spline fit for %d samples is still running after %.0f seconds "
        "(rms_tol=%g, max_knots=%d). Tight tolerances, many samples, or clustered "
        "sampling can make fitting slow. To reduce work, increase rms_tol to allow "
        "a larger RMS error if physically acceptable, or reduce max_knots to limit "
        "refinement (the fit may then fail to meet rms_tol). For clustered data, "
        "consider removing redundant samples while preserving short events; the "
        "RMS budget then applies to the retained samples. Tightening rms_tol "
        "usually increases fitting time. The current fit continues.",
        samples,
        _SLOW_FIT_SECONDS,
        rms_tol,
        max_knots,
    )


def _validate_input(
    ts: TimeSeries, times: np.ndarray, values: np.ndarray, rms_tol: float
) -> None:
    if not (np.isfinite(rms_tol) and rms_tol > 0):
        raise ValueError("rms_tol must be finite and positive.")
    if times.ndim != 1 or values.shape != times.shape:
        raise ValueError("Only scalar-valued series are supported.")
    if len(times) < 2:
        raise ValueError("At least two samples are required.")
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(values))):
        raise ValueError("Times and values must be finite.")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Times must be strictly increasing.")
    if np.max(np.abs(times)) > _MAX_TIME_OFFSET_RATIO * (times[-1] - times[0]):
        raise ValueError(
            "Times are too far from zero relative to the duration; subtract the "
            "start time before fitting."
        )
    if ts.jump_times is not None and np.asarray(ts.jump_times).size:
        raise ValueError("Series with jumps are not supported.")


def _fit_nonconstant(
    times: np.ndarray,
    values: np.ndarray,
    rms_tol: float,
    max_knots: int,
    side: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    t0, span = times[0], times[-1] - times[0]
    y0, net = values[0], values[-1] - values[0]
    x = (times - t0) / span
    z = (values - y0) / net
    budget = rms_tol / abs(net)  # Normalized RMS budget.

    # A lower bound for any monotone curve, without spline or endpoint
    # constraints. Reject impossible budgets before any spline solve or pilot.
    monotone_rms = float(np.sqrt(np.mean((isotonic_regression(z).x - z) ** 2)))
    if monotone_rms > budget:
        raise RuntimeError(
            f"Monotone fit cannot meet rms_tol={rms_tol:.3g}. "
            "Even without spline constraints, the closest monotone sequence "
            f"has RMS error {monotone_rms * abs(net):.3g}, above rms_tol. "
            "Increasing max_knots cannot resolve this; check the data and "
            "requested direction, or increase rms_tol if a larger RMS "
            "deviation is physically acceptable. " + _RMS_TOL_GUIDANCE
        )

    guide = None
    interior = np.empty(0)
    refinements = 0
    while True:
        knots = np.r_[[0.0] * 4, interior, [1.0] * 4]
        design = BSpline.design_matrix(x, knots, 3).tocsc()
        c_best = _solve_conic(design, z, budget, None, None)
        if c_best is None:
            raise RuntimeError(
                f"Monotone fit: the solver did not converge with rms_tol={rms_tol:.3g} "
                f"for a total change of {abs(net):.3g}. This usually means rms_tol is "
                "far below the measurement noise; use at least about 1.5 times the "
                "noise standard deviation."
            )
        # Measure an exactly feasible curve, whatever the solver accuracy.
        c_best = _project(c_best)
        refinements += 1
        residual = design @ c_best - z
        rms = float(np.sqrt(np.mean(residual**2))) / budget
        if rms <= _ACCEPT_FRACTION:
            break
        limit = max_knots - len(interior)
        if limit > 0:
            if guide is None:
                guide = _pilot_guide(x, z, budget, max_knots)
            new = _choose_knots(x, residual, interior, limit, guide)
        else:
            new = []
        if not new:
            if rms <= 1.0:
                break  # The margin is out of reach but the budget is met.
            reason = (
                f"reached max_knots={max_knots}"
                if limit <= 0
                else f"cannot place further knots with {len(interior)} interior knots"
            )
            if limit <= 0:
                guidance = (
                    "Increase max_knots to allow more refinement, or increase "
                    "rms_tol if a larger RMS deviation is physically acceptable. "
                )
            else:
                guidance = (
                    "Refinement cannot resolve finer features at the minimum knot "
                    "spacing. Increase rms_tol if a larger RMS deviation is "
                    "physically acceptable. "
                )
            raise RuntimeError(
                f"Monotone fit {reason}; its RMS "
                f"error {rms * rms_tol:.3g} still exceeds rms_tol={rms_tol:.3g}. "
                + guidance
                + _RMS_TOL_GUIDANCE
            )
        interior = np.sort(np.r_[interior, new])

    smoothing_budget = 1.0 - min(_QP_MARGIN, (1.0 - rms) / _QP_SLACK_DIVISOR)
    c = _smoothest(design, z, budget, smoothing_budget, knots)
    if c is None:
        raise RuntimeError(
            "Monotone fit: smoothing problem did not solve. This can happen with "
            "strongly clustered samples; thinning redundant burst readings helps."
        )
    breaks, coeffs = _to_power_basis(knots, _project(c), t0, span, y0, net)
    breaks[-1] = times[-1]
    problem = _check_stored(breaks, coeffs, times, values, rms_tol, np.sign(net), side)
    if problem is not None:
        raise RuntimeError("Monotone fit failed final checks: " + problem)
    return (
        breaks,
        coeffs,
        {
            "interior_knots": len(interior),
            "refinements": refinements,
        },
    )


def _solve_conic(
    design: sparse.csc_matrix,
    z: np.ndarray,
    budget: float,
    P: sparse.csc_matrix | None,
    rms_bound: float | None,
) -> np.ndarray | None:
    """Solve a monotone fitting problem in normalized coefficients `c` with Clarabel.

    With `rms_bound=None`, minimize the RMS error (an auxiliary variable `r`
    bounds the residual norm). Otherwise minimize `c'Pc/2` subject to an RMS
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
    # Second-order cone: (norm bound, (z - design @ c) / budget).
    norm_row = sparse.csc_matrix(
        ([-1.0], ([0], [n])) if extra else ([], ([], [])), shape=(1, n + extra)
    )
    residual_rows = sparse.hstack([design / budget, sparse.csc_matrix((m, extra))])
    A = sparse.vstack([anchors, increments, norm_row, residual_rows], format="csc")
    norm_bound = 0.0 if rms_bound is None else rms_bound * np.sqrt(m)
    floor = np.full(n - 1, -_INCREMENT_FLOOR)
    b = np.r_[0.0, 1.0, floor, norm_bound, zs]
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
    c = np.asarray(solution.x)[:n]
    if rms_bound is None:
        # Noise-free data put the least-RMS optimum at or next to the tip of the
        # cone, where Clarabel may stop short of full convergence. Any finite
        # iterate is usable: refinement projects it to an exactly feasible curve
        # before measuring it, and smoothing must fully solve.
        return c if c.size == n and np.all(np.isfinite(c)) else None
    if solution.status != clarabel.SolverStatus.Solved:
        return None
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
    # Rescaling leaves the minimizer unchanged. Clustered knots make the diagonal
    # span many orders of magnitude; dividing by its largest entry makes the
    # objective so small that Clarabel stops before it has minimized curvature.
    # The geometric mean keeps the objective at a middle scale.
    diagonal = np.diag(gram)
    gram /= np.exp(np.mean(np.log(diagonal[diagonal > 0])))
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


def _fit_pilot(x: np.ndarray, z: np.ndarray, budget: float, max_knots: int) -> BSpline:
    """Fit a smoothing guide, or its best LSQ approximation at the knot target.

    A bounded layout may not meet the requested pilot RMS. In that case use
    its least-RMS curve rather than asking the smoothing solver to satisfy an
    infeasible constraint. The final monotone fit retains its separate budget.
    SciPy's nest targets the full knot vector, including eight endpoint repeats.
    """
    weights = np.full_like(x, 1.0 / budget)
    nest = min(max_knots + 8, len(x) + 4)
    knots = list(generate_knots(x, z, w=weights, s=float(len(x)), nest=nest))[-1]
    pilot = make_lsq_spline(x, z, knots, w=weights)
    if np.linalg.norm((pilot(x) - z) / budget) > np.sqrt(len(x)):
        return pilot
    # The pilot may be approximate; its smoothing iteration limit must not
    # reject an otherwise valid final fit, including under warnings-as-errors.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"error\. the maximal number of iterations maxit",
            category=RuntimeWarning,
            module=r"scipy\.interpolate\._fitpack_repro",
        )
        return make_splrep(x, z, w=weights, s=float(len(x)), t=knots)


def _pilot_guide(
    x: np.ndarray, z: np.ndarray, budget: float, max_knots: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return pilot breakpoints and curvature-variation density per interval.

    The second derivative of a cubic is piecewise linear. Its total variation
    on an interval is exactly the integral of the absolute third derivative.
    Thus each pilot interval has constant density abs(pilot'''). The smoothing
    condition sum((residual / budget)**2) <= n expresses the sample RMS budget.
    """
    if len(x) < _MIN_PILOT_SAMPLES:
        return np.array([0.0, 1.0]), np.zeros(1)
    pilot = _fit_pilot(x, z, budget, max_knots)
    edges = np.unique(pilot.t)
    midpoints = (edges[:-1] + edges[1:]) / 2.0
    density = np.abs(pilot.derivative(3)(midpoints))
    if not np.all(np.isfinite(density)):
        raise RuntimeError("Unconstrained pilot has nonfinite curvature variation.")
    return edges, density


def _guide_location(
    a: float, b: float, guide: tuple[np.ndarray, np.ndarray]
) -> float | None:
    """Bisect the integral of curvature variation within the chosen interval.

    Intersect the pilot cells with [a, b] before accumulating their masses, so
    features outside this interval cannot attract its knot. Return None when
    the pilot's curvature is constant throughout the interval.
    """
    edges, density = guide
    lo = np.maximum(edges[:-1], a)
    hi = np.minimum(edges[1:], b)
    mass = np.maximum(hi - lo, 0.0) * density
    cumulative = np.cumsum(mass)
    total = cumulative[-1]
    if total == 0.0:
        return None
    half = total / 2.0
    index = int(np.searchsorted(cumulative, half))
    before = cumulative[index - 1] if index else 0.0
    return float(lo[index] + (half - before) / density[index])


def _choose_knots(
    x: np.ndarray,
    residual: np.ndarray,
    interior: np.ndarray,
    limit: int,
    guide: tuple[np.ndarray, np.ndarray],
) -> list[float]:
    """Score intervals by sample RMS error, then use the pilot for locations."""
    edges = np.r_[0.0, interior, 1.0]
    n_intervals = len(edges) - 1
    index = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_intervals - 1)
    score = np.bincount(index, residual**2, minlength=n_intervals)
    new: list[float] = []
    for index in np.argsort(-score):
        if len(new) == limit or score[index] < _BATCH_FRACTION * score.max():
            break
        a, b = edges[index], edges[index + 1]
        knot = _guide_location(a, b, guide)
        if knot is None:
            inside = x[(x > a) & (x < b)]
            knot = (
                float(np.median(inside))
                if len(inside) > _MEDIAN_SAMPLE_THRESHOLD
                else (a + b) / 2.0
            )
        guard = _SPLIT_GUARD * (b - a)
        knot = float(np.clip(knot, a + guard, b - guard))
        # On narrow intervals the clipped proposal can miss the minimum gap
        # while the midpoint still fits. Try it before abandoning the interval.
        if min(knot - a, b - knot) <= _MIN_KNOT_GAP:
            knot = (a + b) / 2.0
        if min(knot - a, b - knot) > _MIN_KNOT_GAP:
            new.append(knot)
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


def _evaluate(
    breaks: np.ndarray, coeffs: np.ndarray, t: np.ndarray, side: str
) -> np.ndarray:
    """Evaluate like `TimeSeries.evaluate`, with `side` choosing the piece at breaks."""
    index = np.clip(np.searchsorted(breaks, t, side=side) - 1, 0, len(coeffs) - 1)
    dt = t - breaks[index]
    a = coeffs[index]
    return a[:, 0] + dt * (a[:, 1] + dt * (a[:, 2] + dt * a[:, 3]))


def _check_stored(
    breaks: np.ndarray,
    coeffs: np.ndarray,
    times: np.ndarray,
    values: np.ndarray,
    rms_tol: float,
    sign: float,
    side: str = "right",
) -> str | None:
    """Return a problem description if the stored spline breaks a guarantee."""
    if not (np.all(np.isfinite(breaks)) and np.all(np.isfinite(coeffs))):
        return "nonfinite coefficients"
    if np.any(np.diff(breaks) <= 0):
        return "collapsed breakpoints"
    fitted = _evaluate(breaks, coeffs, times, side)
    rms = np.sqrt(np.mean(((fitted - values) / rms_tol) ** 2))
    if rms > 1.0:
        return f"RMS error {rms * rms_tol:.3g} exceeds rms_tol={rms_tol:.3g}"
    # Roundoff at the end scales with the size of the last piece's terms there,
    # plus its slope times the time resolution of the pinned last break.
    a = coeffs[-1]
    width = breaks[-1] - breaks[-2]
    end_rate = abs(a[1] + 2 * a[2] * width + 3 * a[3] * width**2)
    tolerance = (
        _ROUNDOFF_FACTOR
        * np.finfo(np.float64).eps
        * (max(np.sum(np.abs(a * width ** np.arange(4))), abs(values[0])))
        + _ENDPOINT_TIME_ULPS * np.spacing(abs(breaks[-1])) * end_rate
    )
    endpoint_error = max(abs(fitted[0] - values[0]), abs(fitted[-1] - values[-1]))
    if not endpoint_error <= tolerance:
        return f"endpoint error {endpoint_error:.3g}"
    # Adjacent pieces must meet, up to roundoff of the values and breakpoints.
    widths = np.diff(breaks)[:-1]
    a = coeffs[:-1]
    left = a[:, 0] + widths * (a[:, 1] + widths * (a[:, 2] + widths * a[:, 3]))
    tolerance = _ROUNDOFF_FACTOR * np.finfo(np.float64).eps * np.max(
        np.abs(values)
    ) + _CONTINUITY_RTOL * abs(values[-1] - values[0])
    if not np.all(np.abs(left - coeffs[1:, 0]) <= tolerance):
        return "value discontinuity at an interior break"
    rates = spline_ops.piece_rate_candidates(breaks, coeffs, times[0], times[-1])
    if np.any(sign * rates < 0.0):
        return "derivative changes sign"
    return None
