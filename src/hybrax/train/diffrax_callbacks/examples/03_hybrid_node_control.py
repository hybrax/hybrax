"""
Example 3: End-to-end — learn dynamics + optimize control through events.

This is the showcase: something nobody has done in JAX before.

Pipeline:
  1. Generate training data from a known bioreactor model (batch cultures)
  2. Train a hybrid Neural ODE: known mass balances + neural growth rate
  3. Optimize a neural feed controller end-to-end, differentiating through
     state-triggered events, the neural controller, and the neural ODE

Gradients flow through:
  loss -> ODE segments (neural dynamics) -> event detection (root-finding)
       -> affect (neural feed controller) -> next ODE segment -> ...
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import optax

from diffrax_callbacks import ContinuousCallback, diffeqsolve_with_callbacks

jax.config.update("jax_enable_x64", True)

SOLVER = diffrax.Tsit5()
CTRL = diffrax.PIDController(rtol=1e-6, atol=1e-8)

# Known stoichiometric constants
YXS = 0.5   # g biomass / g substrate
YPX = 0.1   # g product / g biomass


# ================================================================
# Ground truth (the "unknown" model we'll learn from data)
# ================================================================

def true_growth_rate(S, X):
    """Haldane kinetics — this is what the neural network will learn."""
    return 0.4 * S / (2.0 + S + S**2 / 50.0)


def true_ode(t, y, args):
    X, S, P, V = y[0], y[1], y[2], y[3]
    mu = true_growth_rate(S, X)
    return jnp.array([mu * X, -mu * X / YXS, YPX * mu * X, 0.0])


# ================================================================
# Step 1: Generate training data (batch cultures, no events)
# ================================================================

def generate_data():
    """Multiple batch trajectories with varied initial conditions."""
    initial_conditions = [
        jnp.array([0.5, 5.0, 0.0, 1.0]),
        jnp.array([0.5, 15.0, 0.0, 1.0]),
        jnp.array([0.5, 30.0, 0.0, 1.0]),
        jnp.array([2.0, 10.0, 0.0, 1.0]),
        jnp.array([0.1, 25.0, 0.0, 1.0]),
        jnp.array([1.0, 40.0, 0.0, 1.0]),
    ]

    ts = jnp.linspace(0.0, 20.0, 30)
    all_ts, all_ys = [], []

    for y0 in initial_conditions:
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(true_ode), SOLVER,
            0.0, 20.0, 0.01, y0, None,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=CTRL, max_steps=4096,
        )
        all_ts.append(sol.ts)
        all_ys.append(sol.ys)

    return all_ts, all_ys


# ================================================================
# Step 2: Hybrid Neural ODE
# ================================================================

class NeuralGrowthRate(eqx.Module):
    """Learns mu(S, X) from data. Output is positive via softplus."""
    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=2, out_size=1, width_size=32, depth=2,
            activation=jax.nn.tanh, key=key,
        )

    def __call__(self, S, X):
        return jax.nn.softplus(self.mlp(jnp.array([S, X]))[0]) * 0.1


def hybrid_ode(t, y, args):
    """Known mass balances + neural growth rate (passed via args)."""
    neural_mu = args
    X, S, P, V = y[0], y[1], y[2], y[3]
    mu = neural_mu(jnp.maximum(S, 0.0), jnp.maximum(X, 0.0))
    return jnp.array([mu * X, -mu * X / YXS, YPX * mu * X, 0.0])


def train_dynamics(all_ts, all_ys, n_epochs=500):
    """Train neural growth rate via multiple shooting."""
    neural_mu = NeuralGrowthRate(jax.random.PRNGKey(42))
    optimizer = optax.adam(3e-3)
    opt_state = optimizer.init(eqx.filter(neural_mu, eqx.is_array))

    window = 3

    @eqx.filter_value_and_grad
    def loss_fn(neural_mu):
        total_loss, count = 0.0, 0.0
        for ts, ys in zip(all_ts, all_ys):
            n = ts.shape[0]
            for i in range(0, n - window, 2):
                seg_ts = ts[i:i + window + 1]
                sol = diffrax.diffeqsolve(
                    diffrax.ODETerm(hybrid_ode), SOLVER,
                    seg_ts[0], seg_ts[-1], 0.01, ys[i], neural_mu,
                    saveat=diffrax.SaveAt(ts=seg_ts),
                    stepsize_controller=CTRL, max_steps=2048,
                )
                total_loss += jnp.mean((sol.ys - ys[i:i + window + 1]) ** 2)
                count += 1.0
        return total_loss / count

    @eqx.filter_jit
    def step(neural_mu, opt_state):
        loss, grads = loss_fn(neural_mu)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(neural_mu, eqx.is_array)
        )
        return eqx.apply_updates(neural_mu, updates), opt_state, loss

    print("Training neural growth rate...")
    for epoch in range(1, n_epochs + 1):
        neural_mu, opt_state, loss = step(neural_mu, opt_state)
        if epoch % 100 == 0 or epoch == 1:
            mu_l = float(neural_mu(5.0, 1.0))
            mu_t = float(true_growth_rate(5.0, 1.0))
            print(f"  Epoch {epoch:3d} | Loss: {float(loss):.6f} | "
                  f"mu(5,1): {mu_l:.4f} (true: {mu_t:.4f})")

    print("\nGrowth rate comparison:")
    for S, X in [(1.0, 1.0), (5.0, 5.0), (10.0, 10.0), (20.0, 5.0)]:
        mu_l, mu_t = float(neural_mu(S, X)), float(true_growth_rate(S, X))
        print(f"  S={S:5.1f}, X={X:5.1f}: learned={mu_l:.4f}  true={mu_t:.4f}  "
              f"err={abs(mu_l - mu_t) / (mu_t + 1e-10):.1%}")

    return neural_mu


# ================================================================
# Step 3: Neural feed controller
# ================================================================

class FeedController(eqx.Module):
    """State -> (feed_volume, feed_concentration). Outputs bounded via sigmoid."""
    mlp: eqx.nn.MLP
    vol_min: float = eqx.field(static=True, default=0.05)
    vol_max: float = eqx.field(static=True, default=0.5)
    conc_min: float = eqx.field(static=True, default=50.0)
    conc_max: float = eqx.field(static=True, default=300.0)

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=4, out_size=2, width_size=16, depth=2,
            activation=jax.nn.tanh, key=key,
        )

    def __call__(self, y):
        sig = jax.nn.sigmoid(self.mlp(y))
        vol = self.vol_min + sig[0] * (self.vol_max - self.vol_min)
        conc = self.conc_min + sig[1] * (self.conc_max - self.conc_min)
        return vol, conc


def apply_feed(y, feed_vol, feed_conc):
    X, S, P, V = y[0], y[1], y[2], y[3]
    V_new = V + feed_vol
    return jnp.array([
        X * V / V_new,
        (S * V + feed_conc * feed_vol) / V_new,
        P * V / V_new,
        V_new,
    ])


def optimize_control(neural_mu, y0, t_end, n_epochs=200):
    """Optimize feed controller + threshold end-to-end."""
    feed_controller = FeedController(jax.random.PRNGKey(123))
    threshold = jnp.array(2.0)
    control_params = (feed_controller, threshold)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(control_params, eqx.is_array))

    def objective(control_params):
        feed_ctrl, thresh = control_params
        args = (neural_mu, feed_ctrl, thresh)

        def ode_fn(t, y, args):
            mu_net = args[0]
            X, S, P, V = y[0], y[1], y[2], y[3]
            mu = mu_net(jnp.maximum(S, 0.0), jnp.maximum(X, 0.0))
            return jnp.array([mu * X, -mu * X / YXS, YPX * mu * X, 0.0])

        cb = ContinuousCallback(
            condition_fn=lambda y, t, args: y[1] - args[2],
            affect_fn=lambda y, t, args: apply_feed(y, *args[1](y)),
            direction="down",
        )

        sol = diffeqsolve_with_callbacks(
            diffrax.ODETerm(ode_fn), SOLVER,
            0.0, t_end, 0.01, y0, args,
            callbacks=cb, max_events=15,
            stepsize_controller=CTRL,
        )

        product = sol.y_final[2] * sol.y_final[3]
        penalty = 0.5 * jnp.maximum(sol.y_final[3] - 3.0, 0.0) ** 2
        return -(product - penalty)

    @eqx.filter_jit
    def opt_step(control_params, opt_state):
        loss, grads = eqx.filter_value_and_grad(objective)(control_params)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(control_params, eqx.is_array)
        )
        return eqx.apply_updates(control_params, updates), opt_state, loss

    print("\nOptimizing neural feed controller...")
    print("  Gradients flow: events -> neural controller -> neural ODE -> loss")
    print()

    for epoch in range(1, n_epochs + 1):
        control_params, opt_state, loss = opt_step(control_params, opt_state)

        if epoch % 25 == 0 or epoch == 1:
            feed_ctrl, thresh = control_params
            fv, fc = feed_ctrl(jnp.array([15.0, 2.0, 1.0, 1.5]))
            print(f"  Epoch {epoch:3d} | Product: {-float(loss):7.2f}g | "
                  f"threshold={float(thresh):.3f} | "
                  f"ctrl@test: vol={float(fv):.3f}L conc={float(fc):.0f}g/L")

    # Final evaluation
    feed_ctrl, thresh = control_params
    args_final = (neural_mu, feed_ctrl, thresh)

    def ode_eval(t, y, args):
        mu_net = args[0]
        X, S, P, V = y[0], y[1], y[2], y[3]
        mu = mu_net(jnp.maximum(S, 0.0), jnp.maximum(X, 0.0))
        return jnp.array([mu * X, -mu * X / YXS, YPX * mu * X, 0.0])

    cb_final = ContinuousCallback(
        condition_fn=lambda y, t, args: y[1] - args[2],
        affect_fn=lambda y, t, args: apply_feed(y, *args[1](y)),
        direction="down",
    )
    sol = diffeqsolve_with_callbacks(
        diffrax.ODETerm(ode_eval), SOLVER,
        0.0, t_end, 0.01, y0, args_final,
        callbacks=cb_final, max_events=15,
        stepsize_controller=CTRL,
    )

    print(f"\nOptimized strategy:")
    print(f"  Feed threshold: S < {float(thresh):.3f} g/L")
    sol.print_events(["X", "S", "P", "V"])
    print(f"\n  Final: X={sol.y_final[0]:.2f}, S={sol.y_final[1]:.2f}, "
          f"P={sol.y_final[2]:.2f}, V={sol.y_final[3]:.2f}")
    print(f"  Total product: {sol.y_final[2] * sol.y_final[3]:.2f} g")

    return control_params


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    y0 = jnp.array([0.5, 20.0, 0.0, 1.0])
    t_end = 48.0

    print("=" * 60)
    print("Step 1: Generate training data")
    print("=" * 60)
    all_ts, all_ys = generate_data()
    print(f"Generated {len(all_ts)} batch trajectories\n")

    print("=" * 60)
    print("Step 2: Learn growth kinetics (hybrid Neural ODE)")
    print("=" * 60)
    neural_mu = train_dynamics(all_ts, all_ys, n_epochs=500)

    print("\n" + "=" * 60)
    print("Step 3: Optimize neural feed controller")
    print("=" * 60)
    optimize_control(neural_mu, y0, t_end, n_epochs=200)

    print("\n" + "=" * 60)
    print("Done. Data -> learned dynamics -> optimal control with")
    print("discrete events, all differentiable, all in JAX.")
    print("=" * 60)
