"""
diffrax_callbacks: Julia-style callback system for Diffrax.

Provides ContinuousCallback, DiscreteCallback, PresetTimeCallback,
PeriodicCallback, ManifoldProjection, and CallbackSet for JAX/Diffrax
with full differentiability through discrete events.

Example:
    from diffrax_callbacks import (
        ContinuousCallback, PresetTimeCallback, ManifoldProjection,
        CallbackSet, diffeqsolve_with_callbacks,
    )

    cb = CallbackSet(
        ContinuousCallback(
            condition_fn=lambda y, t, args: y[1] - 1.0,
            affect_fn=lambda y, t, args: apply_feed(y),
            direction="down",
        ),
        PresetTimeCallback(
            times=jnp.array([12.0, 24.0]),
            affect_fn=lambda y, t, args: take_sample(y),
        ),
        ManifoldProjection(
            project_fn=lambda y, t, args: jnp.maximum(y, 0.0),
        ),
    )

    sol = diffeqsolve_with_callbacks(
        terms, solver, t0, t1, dt0, y0, args,
        callbacks=cb, max_events=20,
    )
"""

from ._callbacks import (
    ContinuousCallback,
    DiscreteCallback,
    PresetTimeCallback,
    PeriodicCallback,
    ManifoldProjection,
    CallbackSet,
)
from ._solve import diffeqsolve_with_callbacks, evaluate_trajectory
from ._solution import CallbackSolution
