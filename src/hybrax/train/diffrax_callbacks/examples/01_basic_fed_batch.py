"""
Example 1: Basic fed-batch bioreactor with state-triggered feeding.

Demonstrates:
  - ContinuousCallback: feed when substrate drops below a threshold
  - Differentiability: compute gradients w.r.t. feed parameters
  - Optimization: find optimal feed volume and concentration

This is the simplest use case: a bioreactor where cells consume substrate,
and a bolus feed is triggered whenever substrate drops below 1.0 g/L.
"""

import jax
import jax.numpy as jnp
import diffrax
import optax

from diffrax_callbacks import ContinuousCallback, diffeqsolve_with_callbacks

jax.config.update("jax_enable_x64", True)


# -- Bioreactor model --
# State: [X (biomass), S (substrate), P (product), V (volume)]
# Kinetics: Haldane (substrate inhibition)


def bioreactor_ode(t, y, args):
    X, S, _, _ = y[0], y[1], y[2], y[3]
    mu = 0.4 * S / (2.0 + S + S**2 / 50.0)  # Haldane
    return jnp.array([mu * X, -mu * X / 0.5, 0.1 * mu * X, 0.0])


y0 = jnp.array([0.5, 20.0, 0.0, 1.0])
t_end = 48.0
solver = diffrax.Tsit5()
controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)


# -- Define the feed callback --


def make_feed_callback(feed_volume, feed_concentration, threshold):
    """Create a callback that feeds when S drops below threshold."""

    def affect_fn(y, t, args):
        X, S, P, V = y[0], y[1], y[2], y[3]
        V_new = V + feed_volume
        return jnp.array(
            [
                X * V / V_new,  # dilute biomass
                (S * V + feed_concentration * feed_volume) / V_new,  # mix substrate
                P * V / V_new,  # dilute product
                V_new,  # new volume
            ]
        )

    return ContinuousCallback(
        condition_fn=lambda y, t, args: y[1] - threshold,
        affect_fn=affect_fn,
        direction="down",
    )


# -- 1. Forward simulation --

print("=" * 55)
print("1. Forward simulation")
print("=" * 55)

cb = make_feed_callback(
    feed_volume=0.1,
    feed_concentration=100.0,
    threshold=1.0,
)

sol = diffeqsolve_with_callbacks(
    diffrax.ODETerm(bioreactor_ode),
    solver,
    0.0,
    t_end,
    0.01,
    y0,
    None,
    callbacks=cb,
    max_events=20,
    stepsize_controller=controller,
)

print(f"Events triggered: {sol.event_count}")
sol.print_events(["X", "S", "P", "V"])
print(
    f"\nFinal: X={sol.y_final[0]:.2f} g/L, S={sol.y_final[1]:.2f} g/L, "
    f"P={sol.y_final[2]:.2f} g/L, V={sol.y_final[3]:.2f} L"
)
print(f"Total product: {sol.y_final[2] * sol.y_final[3]:.2f} g")


# -- 2. Differentiate through events --

print(f"\n{'=' * 55}")
print("2. Gradients through state-triggered events")
print("=" * 55)


def total_product(params):
    """Total product mass as a function of feed parameters."""
    feed_vol, feed_conc, threshold = params
    cb = make_feed_callback(feed_vol, feed_conc, threshold)
    sol = diffeqsolve_with_callbacks(
        diffrax.ODETerm(bioreactor_ode),
        solver,
        0.0,
        t_end,
        0.01,
        y0,
        None,
        callbacks=cb,
        max_events=20,
        stepsize_controller=controller,
    )
    return sol.y_final[2] * sol.y_final[3]


params = jnp.array([0.1, 100.0, 1.0])
value, grads = jax.jit(jax.value_and_grad(total_product))(params)

print(f"Product: {value:.4f} g")
print(f"d(product)/d(feed_volume):  {grads[0]:.4f}")
print(f"d(product)/d(feed_conc):    {grads[1]:.6f}")
print(f"d(product)/d(threshold):    {grads[2]:.6f}")


# -- 3. Optimize feed strategy --

print(f"\n{'=' * 55}")
print("3. Optimize feed parameters")
print("=" * 55)

neg_product = jax.jit(jax.value_and_grad(lambda p: -total_product(p)))
optimizer = optax.adam(1e-2)
opt_state = optimizer.init(params)

for step in range(150):
    loss, grads = neg_product(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    params = jnp.clip(
        params, jnp.array([0.01, 10.0, 0.1]), jnp.array([1.0, 300.0, 5.0])
    )
    if (step + 1) % 50 == 0:
        print(
            f"  Step {step + 1:3d} | Product: {-loss:.2f}g | "
            f"vol={params[0]:.3f}L, conc={params[1]:.0f}g/L, "
            f"threshold={params[2]:.3f}g/L"
        )

print(
    f"\nOptimized: vol={params[0]:.3f}L, conc={params[1]:.0f}g/L, "
    f"threshold={params[2]:.3f}g/L"
)
