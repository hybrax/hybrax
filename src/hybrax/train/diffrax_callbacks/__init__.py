"""Julia-style callback system for Diffrax."""

from ._callbacks import (
    CallbackSet,
    ContinuousCallback,
    DiscreteCallback,
    ManifoldProjection,
    PeriodicCallback,
    PresetTimeCallback,
)
from ._solution import CallbackSolution
from ._solve import diffeqsolve_with_callbacks, evaluate_trajectory

__all__ = [
    "CallbackSet",
    "CallbackSolution",
    "ContinuousCallback",
    "DiscreteCallback",
    "ManifoldProjection",
    "PeriodicCallback",
    "PresetTimeCallback",
    "diffeqsolve_with_callbacks",
    "evaluate_trajectory",
]
