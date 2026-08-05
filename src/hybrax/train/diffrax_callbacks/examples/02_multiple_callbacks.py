"""
Example 2: Multiple callbacks — realistic bioreactor operation.

Demonstrates:
  - CallbackSet: combining multiple callback types
  - ContinuousCallback: feed on low substrate, bleed on high biomass
  - PresetTimeCallback: scheduled sampling
  - PeriodicCallback: regular monitoring
  - ManifoldProjection: enforce non-negative concentrations
  - repeat_nudge: prevent infinite re-triggering

This models a realistic bioreactor with:
  - Automatic substrate feeding when glucose is low
  - Cell bleeding when biomass concentration gets too high
  - Samples taken at fixed times for quality control
  - Periodic state logging
  - Physical constraint enforcement (no negative concentrations)
"""

import jax
import jax.numpy as jnp
import diffrax

from diffrax_callbacks import (
    ContinuousCallback,
    PresetTimeCallback,
    PeriodicCallback,
    ManifoldProjection,
    CallbackSet,
    diffeqsolve_with_callbacks,
)

jax.config.update("jax_enable_x64", True)


def bioreactor_ode(t, y, args):
    """Monod kinetics with substrate inhibition.
    State: [X (biomass), S (substrate), P (product), V (volume)]
    """
    X, S, P, V = y[0], y[1], y[2], y[3]
    mu = 0.4 * S / (2.0 + S + S**2 / 50.0)
    return jnp.array([mu * X, -mu * X / 0.5, 0.1 * mu * X, 0.0])


y0 = jnp.array([0.5, 20.0, 0.0, 1.0])
t_end = 48.0
solver = diffrax.Tsit5()
controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)


# -- Build the callback set --

callbacks = CallbackSet(
    # 1. Feed when substrate drops below 2.0 g/L
    ContinuousCallback(
        condition_fn=lambda y, t, args: y[1] - 2.0,
        affect_fn=lambda y, t, args: jnp.array(
            [
                y[0] * y[3] / (y[3] + 0.15),
                (y[1] * y[3] + 150.0 * 0.15) / (y[3] + 0.15),
                y[2] * y[3] / (y[3] + 0.15),
                y[3] + 0.15,
            ]
        ),
        direction="down",
    ),
    # 2. Bleed when biomass exceeds 40 g/L (remove 10% of culture volume)
    #    This removes both cells and media, reducing total biomass but not
    #    concentration. Combined with continued growth, X will re-cross 40.
    ContinuousCallback(
        condition_fn=lambda y, t, args: y[0] - 40.0,
        affect_fn=lambda y, t, args: jnp.array(
            [
                y[0] * 0.85,  # dilute biomass (simulate partial harvest)
                y[1],  # substrate unchanged
                y[2] * 0.85,  # dilute product
                y[3] * 0.9,  # remove 10% volume
            ]
        ),
        direction="up",
        repeat_nudge=1.0,  # wait at least 1h before next bleed
    ),
    # 3. Take samples at fixed times (removes 50 mL)
    PresetTimeCallback(
        times=jnp.array([6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0]),
        affect_fn=lambda y, t, args, i: y.at[3].add(-0.05),
    ),
    # 4. Log state every 4 hours (no-op affect, just records in event log)
    PeriodicCallback(
        dt=4.0,
        affect_fn=lambda y, t, args: y,
        t_end=t_end,
    ),
    # 5. Enforce non-negative concentrations
    ManifoldProjection(
        project_fn=lambda y, t, args: jnp.maximum(y, 0.0),
    ),
)


# -- Run simulation --

print("Bioreactor simulation with 5 callback types")
print("=" * 65)
print(
    f"Initial: X={y0[0]:.1f} g/L, S={y0[1]:.1f} g/L, P={y0[2]:.1f} g/L, V={y0[3]:.1f} L"
)
print()

sol = jax.jit(
    lambda: diffeqsolve_with_callbacks(
        diffrax.ODETerm(bioreactor_ode),
        solver,
        0.0,
        t_end,
        0.01,
        y0,
        None,
        callbacks=callbacks,
        max_events=50,
        stepsize_controller=controller,
    )
)()

# -- Print results --

callback_names = {
    0: "feed",
    1: "bleed",
    2: "sample",
    3: "log",
}

n_feed = int(sol.count_by_type(0))
n_bleed = int(sol.count_by_type(1))
n_sample = int(sol.count_by_type(2))
n_log = int(sol.count_by_type(3))

print(f"Total events: {sol.event_count}")
print(f"  Feeds (S < 2.0):     {n_feed}")
print(f"  Bleeds (X > 40.0):   {n_bleed}")
print(f"  Samples (scheduled): {n_sample}")
print(f"  Logs (periodic):     {n_log}")
print()

sol.print_events(["X", "S", "P", "V"], callback_names=callback_names)

print(f"\nFinal state:")
print(f"  Biomass:   X = {sol.y_final[0]:.2f} g/L")
print(f"  Substrate: S = {sol.y_final[1]:.2f} g/L")
print(f"  Product:   P = {sol.y_final[2]:.2f} g/L")
print(f"  Volume:    V = {sol.y_final[3]:.2f} L")
print(f"  Total product: {sol.y_final[2] * sol.y_final[3]:.2f} g")
