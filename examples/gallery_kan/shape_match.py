"""Fit a small library of textbook shapes to a 1-D curve and report the
best-scoring match, or "no clean match" if nothing clears the R2 floor.

Shared by the per-edge check (each of l1's learned curves) and the
equation-recovery check (the whole trained model's rate output, swept over
one real input at a time) in run.py.
"""

import numpy as np
from scipy.optimize import curve_fit


def _flat(x, c):
    return np.full_like(x, c)


def _linear(x, a, b):
    return a * x + b


def _power(x, a, b, c):
    return a * np.power(np.maximum(x, 1e-9), b) + c


def _saturating(x, a, k, c):
    return a * x / (k + x) + c


def _exponential(x, a, b):
    return a * np.exp(np.clip(b * x, -50, 50))


# name -> (function, initial guess, (lower bounds, upper bounds))
# power and saturating carry a constant offset c: a curve that saturates or
# follows a power law rarely happens to pass through zero, and without an
# offset a shifted version of either shape is scored as "no match" even
# when the underlying shape is a clean fit.
CANDIDATES = {
    "flat": (_flat, [0.0], ([-np.inf], [np.inf])),
    "linear": (_linear, [1.0, 0.0], ([-np.inf, -np.inf], [np.inf, np.inf])),
    "power": (
        _power,
        [1.0, 1.0, 0.0],
        ([-np.inf, 0.01, -np.inf], [np.inf, 5.0, np.inf]),
    ),
    "saturating": (
        _saturating,
        [1.0, 1.0, 0.0],
        ([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf]),
    ),
    "exponential": (_exponential, [1.0, 0.1], ([-np.inf, -5.0], [np.inf, 5.0])),
}


def best_match(x, y, r2_floor=0.9):
    """Fit every candidate in CANDIDATES to (x, y), score by BIC, and return
    the best one. If the best candidate's R2 falls below r2_floor, report
    "no clean match" instead of forcing an answer, but keep the numbers so
    the caller can still show what almost matched.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    fits = {}
    for name, (fn, p0, bounds) in CANDIDATES.items():
        try:
            popt, _ = curve_fit(fn, x, y, p0=p0, bounds=bounds, maxfev=10000)
            pred = fn(x, *popt)
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else (1.0 if ss_res < 1e-9 else 0.0)
            k = len(popt)
            bic = n * np.log(max(ss_res / n, 1e-300)) + k * np.log(n)
            fits[name] = {"params": [float(p) for p in popt], "r2": r2, "bic": bic}
        except Exception:
            continue
    if not fits:
        return {"best": "no clean match", "r2": None, "params": None, "fits": fits}
    best_name = min(fits, key=lambda nm: fits[nm]["bic"])
    result = {
        "best": best_name if fits[best_name]["r2"] >= r2_floor else "no clean match",
        "closest": best_name,
        "r2": fits[best_name]["r2"],
        "params": fits[best_name]["params"],
        "fits": fits,
    }
    return result
