"""
Tests for diffrax_callbacks.

Run with: pytest tests/ -v
"""

import jax
import jax.numpy as jnp
import diffrax
import pytest

from diffrax_callbacks import (
    ContinuousCallback,
    DiscreteCallback,
    PresetTimeCallback,
    PeriodicCallback,
    ManifoldProjection,
    CallbackSet,
    diffeqsolve_with_callbacks,
    evaluate_trajectory,
)

jax.config.update("jax_enable_x64", True)


# ---- Shared fixtures ----

SOLVER = diffrax.Tsit5()
CTRL = diffrax.PIDController(rtol=1e-6, atol=1e-8)


def bioreactor_ode(t, y, args):
    """Monod + substrate inhibition. State: [X, S, P, V]."""
    X, S = y[0], y[1]
    mu = 0.4 * S / (2.0 + S + S**2 / 50.0)
    return jnp.array([mu * X, -mu * X / 0.5, 0.1 * mu * X, 0.0])


TERMS = diffrax.ODETerm(bioreactor_ode)
Y0 = jnp.array([0.5, 20.0, 0.0, 1.0])
T_END = 48.0


def run(cb, max_events=20):
    return diffeqsolve_with_callbacks(
        TERMS,
        SOLVER,
        0.0,
        T_END,
        0.01,
        Y0,
        None,
        callbacks=cb,
        max_events=max_events,
        stepsize_controller=CTRL,
    )


def feed_affect(y, t, args):
    X, S, P, V = y[0], y[1], y[2], y[3]
    V_new = V + 0.1
    return jnp.array(
        [
            X * V / V_new,
            (S * V + 100.0 * 0.1) / V_new,
            P * V / V_new,
            V_new,
        ]
    )


# ---- Tests ----


def test_float32_state_with_float64_times_keeps_state_dtype():
    """Regression: float64 times must not promote a float32 state solve."""

    def ode_uses_time(t, y, args):
        del y, args
        return jnp.asarray([t, t])

    def noop_affect(y, t, args):
        del t, args
        return y

    sol = diffeqsolve_with_callbacks(
        diffrax.ODETerm(ode_uses_time),
        SOLVER,
        t0=jnp.asarray(0.0, dtype=jnp.float64),
        t1=jnp.asarray(1.0, dtype=jnp.float64),
        dt0=jnp.asarray(0.1, dtype=jnp.float64),
        y0=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        callbacks=PresetTimeCallback(
            times=jnp.asarray([0.5], dtype=jnp.float64),
            affect_fn=noop_affect,
        ),
        max_events=3,
        stepsize_controller=CTRL,
    )

    assert sol.y_final.dtype == jnp.float32
    assert jnp.allclose(sol.y_final, jnp.asarray([0.5, 0.5], dtype=jnp.float32))


class TestContinuousCallback:
    def test_basic_triggering(self):
        """Feed when S < 1.0. Should trigger multiple times."""
        cb = ContinuousCallback(
            condition_fn=lambda y, t, args: y[1] - 1.0,
            affect_fn=feed_affect,
            direction="down",
        )
        sol = run(cb)
        assert int(sol.event_count) > 0
        assert sol.y_final[3] > 1.0  # volume increased from feeds

    def test_event_times_are_ordered(self):
        cb = ContinuousCallback(
            condition_fn=lambda y, t, args: y[1] - 1.0,
            affect_fn=feed_affect,
            direction="down",
        )
        sol = run(cb)
        n = int(sol.event_count)
        for i in range(1, n):
            assert sol.event_times[i] >= sol.event_times[i - 1]

    def test_condition_holds_at_event(self):
        """S should be approximately 1.0 at each event."""
        cb = ContinuousCallback(
            condition_fn=lambda y, t, args: y[1] - 1.0,
            affect_fn=feed_affect,
            direction="down",
        )
        sol = run(cb)
        n = int(sol.event_count)
        for i in range(n):
            assert abs(sol.event_states_before[i, 1] - 1.0) < 0.01

    def test_direction_up(self):
        """Trigger on upcrossing: X > 5.0."""
        cb = ContinuousCallback(
            condition_fn=lambda y, t, args: y[0] - 5.0,
            affect_fn=lambda y, t, args: y,  # no-op
            direction="up",
        )
        sol = run(cb, max_events=3)
        assert int(sol.event_count) >= 1
        assert sol.event_states_before[0, 0] == pytest.approx(5.0, abs=0.01)


class TestRepeatNudge:
    def test_prevents_infinite_retriggering(self):
        """Bleed that doesn't change X concentration. Without nudge, loops forever."""
        cb = CallbackSet(
            ContinuousCallback(
                condition_fn=lambda y, t, args: y[1] - 1.0,
                affect_fn=feed_affect,
                direction="down",
            ),
            ContinuousCallback(
                condition_fn=lambda y, t, args: y[0] - 30.0,
                affect_fn=lambda y, t, args: y.at[3].set(y[3] * 0.9),
                direction="up",
                repeat_nudge=0.5,
            ),
        )
        sol = run(cb, max_events=30)

        # Find bleed events and check time gaps
        bleed_times = [
            float(sol.event_times[i])
            for i in range(int(sol.event_count))
            if int(sol.event_types[i]) == 1
        ]
        if len(bleed_times) >= 2:
            min_gap = min(
                bleed_times[i + 1] - bleed_times[i] for i in range(len(bleed_times) - 1)
            )
            assert min_gap >= 0.49  # nudge = 0.5


class TestDiscreteCallback:
    def test_clamping(self):
        """Discrete callback clamps states to non-negative."""
        cb = CallbackSet(
            ContinuousCallback(
                condition_fn=lambda y, t, args: y[1] - 1.0,
                affect_fn=feed_affect,
                direction="down",
            ),
            DiscreteCallback(
                condition_fn=lambda y, t, args: jnp.any(y < 0),
                affect_fn=lambda y, t, args: jnp.maximum(y, 0.0),
            ),
        )
        sol = run(cb)
        assert jnp.all(sol.y_final >= 0)


class TestPresetTimeCallback:
    def test_triggers_at_exact_times(self):
        preset_times = jnp.array([8.0, 16.0, 24.0, 32.0, 40.0])
        cb = PresetTimeCallback(
            times=preset_times,
            affect_fn=lambda y, t, args: y.at[3].add(-0.05),
        )
        sol = run(cb, max_events=10)
        n = int(sol.event_count)
        assert n == 5
        for i in range(n):
            assert sol.event_times[i] == pytest.approx(float(preset_times[i]), abs=1e-8)

    def test_volume_change(self):
        cb = PresetTimeCallback(
            times=jnp.array([10.0, 20.0]),
            affect_fn=lambda y, t, args: y.at[3].add(-0.1),
        )
        sol = run(cb, max_events=5)
        assert sol.y_final[3] == pytest.approx(0.8, abs=0.01)


class TestPeriodicCallback:
    def test_correct_count(self):
        cb = PeriodicCallback(
            dt=8.0,
            affect_fn=lambda y, t, args: y.at[3].add(-0.02),
            t_end=T_END,
        )
        sol = run(cb, max_events=10)
        expected = int(T_END // 8.0)
        assert int(sol.event_count) == expected

    def test_cumulative_effect(self):
        dt = 12.0
        cb = PeriodicCallback(
            dt=dt,
            affect_fn=lambda y, t, args: y.at[3].add(-0.05),
            t_end=T_END,
        )
        sol = run(cb, max_events=10)
        n_events = int(T_END // dt)
        expected_vol = 1.0 - n_events * 0.05
        assert sol.y_final[3] == pytest.approx(expected_vol, abs=0.01)


class TestManifoldProjection:
    def test_non_negative(self):
        cb = CallbackSet(
            ContinuousCallback(
                condition_fn=lambda y, t, args: y[1] - 1.0,
                affect_fn=feed_affect,
                direction="down",
            ),
            ManifoldProjection(
                project_fn=lambda y, t, args: jnp.maximum(y, 0.0),
            ),
        )
        sol = run(cb)
        assert jnp.all(sol.event_states_after >= -1e-10)


class TestCallbackSet:
    def test_mixed_callbacks(self):
        """Continuous + preset + periodic + discrete all together."""
        cb = CallbackSet(
            ContinuousCallback(
                condition_fn=lambda y, t, args: y[1] - 1.0,
                affect_fn=feed_affect,
                direction="down",
                repeat_nudge=0.01,
            ),
            PresetTimeCallback(
                times=jnp.array([12.0, 24.0, 36.0]),
                affect_fn=lambda y, t, args: y.at[3].add(-0.05),
            ),
            PeriodicCallback(
                dt=6.0,
                affect_fn=lambda y, t, args: y,  # no-op log
                t_end=T_END,
            ),
            ManifoldProjection(
                project_fn=lambda y, t, args: jnp.maximum(y, 0.0),
            ),
        )
        sol = run(cb, max_events=40)
        assert int(sol.event_count) > 0
        assert jnp.all(sol.y_final >= 0)

    def test_priority_continuous_over_preset(self):
        """When continuous event fires before preset time, it takes priority."""
        cb = CallbackSet(
            ContinuousCallback(
                condition_fn=lambda y, t, args: y[1] - 1.0,
                affect_fn=feed_affect,
                direction="down",
            ),
            PresetTimeCallback(
                times=jnp.array([12.0]),
                affect_fn=lambda y, t, args: y.at[3].add(-0.05),
            ),
        )
        sol = run(cb, max_events=5)
        # First event should be the continuous one (S hits 1 at ~11.1h, before 12h)
        assert int(sol.event_types[0]) == 0  # continuous callback index


class TestGradients:
    def test_gradient_through_continuous(self):
        """Gradient of final product w.r.t. feed parameters."""

        def objective(params):
            feed_vol, feed_conc = params

            def affect(y, t, args):
                X, S, P, V = y[0], y[1], y[2], y[3]
                V_new = V + feed_vol
                return jnp.array(
                    [
                        X * V / V_new,
                        (S * V + feed_conc * feed_vol) / V_new,
                        P * V / V_new,
                        V_new,
                    ]
                )

            cb = ContinuousCallback(
                condition_fn=lambda y, t, args: y[1] - 1.0,
                affect_fn=affect,
                direction="down",
            )
            sol = diffeqsolve_with_callbacks(
                TERMS,
                SOLVER,
                0.0,
                T_END,
                0.01,
                Y0,
                None,
                callbacks=cb,
                max_events=20,
                stepsize_controller=CTRL,
            )
            return sol.y_final[2] * sol.y_final[3]

        params = jnp.array([0.1, 100.0])
        value, grads = jax.value_and_grad(objective)(params)

        # Verify with finite differences
        eps = 1e-5
        for i in range(2):
            p_plus = params.at[i].set(params[i] + eps)
            p_minus = params.at[i].set(params[i] - eps)
            fd = (objective(p_plus) - objective(p_minus)) / (2 * eps)
            rel_err = abs(grads[i] - fd) / (abs(fd) + 1e-10)
            assert rel_err < 1e-3, (
                f"Gradient {i}: ad={grads[i]}, fd={fd}, err={rel_err}"
            )

    def test_gradient_through_threshold(self):
        """Gradient w.r.t. the event threshold itself."""

        def objective(threshold):
            cb = ContinuousCallback(
                condition_fn=lambda y, t, args: y[1] - threshold,
                affect_fn=feed_affect,
                direction="down",
            )
            sol = diffeqsolve_with_callbacks(
                TERMS,
                SOLVER,
                0.0,
                T_END,
                0.01,
                Y0,
                None,
                callbacks=cb,
                max_events=20,
                stepsize_controller=CTRL,
            )
            return sol.y_final[2] * sol.y_final[3]

        threshold = jnp.array(1.0)
        value, grad = jax.value_and_grad(objective)(threshold)

        eps = 1e-5
        fd = (objective(threshold + eps) - objective(threshold - eps)) / (2 * eps)
        rel_err = abs(grad - fd) / (abs(fd) + 1e-10)
        assert rel_err < 1e-2

    def test_gradient_through_manifold(self):
        """Gradient flows through ManifoldProjection."""

        def objective(feed_vol):
            cb = CallbackSet(
                ContinuousCallback(
                    condition_fn=lambda y, t, args: y[1] - 1.0,
                    affect_fn=lambda y, t, args: jnp.array(
                        [
                            y[0] * y[3] / (y[3] + feed_vol),
                            (y[1] * y[3] + 100.0 * feed_vol) / (y[3] + feed_vol),
                            y[2] * y[3] / (y[3] + feed_vol),
                            y[3] + feed_vol,
                        ]
                    ),
                    direction="down",
                ),
                ManifoldProjection(
                    project_fn=lambda y, t, args: jnp.maximum(y, 0.0),
                ),
            )
            sol = diffeqsolve_with_callbacks(
                TERMS,
                SOLVER,
                0.0,
                T_END,
                0.01,
                Y0,
                None,
                callbacks=cb,
                max_events=20,
                stepsize_controller=CTRL,
            )
            return sol.y_final[2] * sol.y_final[3]

        feed_vol = jnp.array(0.1)
        value, grad = jax.value_and_grad(objective)(feed_vol)
        assert jnp.isfinite(grad)


class TestSolution:
    def test_get_events_filtered(self):
        cb = CallbackSet(
            ContinuousCallback(
                condition_fn=lambda y, t, args: y[1] - 1.0,
                affect_fn=feed_affect,
                direction="down",
            ),
            PresetTimeCallback(
                times=jnp.array([12.0, 24.0]),
                affect_fn=lambda y, t, args: y.at[3].add(-0.05),
            ),
        )
        sol = run(cb, max_events=25)

        # Filter by type
        times_cont, _, _ = sol.get_events(callback_index=0)
        times_preset, _, _ = sol.get_events(callback_index=1)

        n_cont = jnp.sum(~jnp.isnan(times_cont)).astype(int)
        n_preset = jnp.sum(~jnp.isnan(times_preset)).astype(int)
        assert n_cont > 0
        assert n_preset > 0
        assert n_cont + n_preset == int(sol.event_count)

    def test_count_by_type(self):
        cb = CallbackSet(
            ContinuousCallback(
                condition_fn=lambda y, t, args: y[1] - 1.0,
                affect_fn=feed_affect,
                direction="down",
            ),
            PresetTimeCallback(
                times=jnp.array([12.0]),
                affect_fn=lambda y, t, args: y,
            ),
        )
        sol = run(cb, max_events=25)
        n_cont = int(sol.count_by_type(0))
        n_preset = int(sol.count_by_type(1))
        assert n_cont > 0
        assert n_cont + n_preset == int(sol.event_count)

    def test_evaluate_trajectory(self):
        """Reconstruct full trajectory at arbitrary time points."""
        cb = ContinuousCallback(
            condition_fn=lambda y, t, args: y[1] - 1.0,
            affect_fn=feed_affect,
            direction="down",
        )
        sol = run(cb)

        ts_eval = jnp.linspace(0.0, T_END, 100)
        ts_out, ys_out = evaluate_trajectory(
            sol,
            TERMS,
            SOLVER,
            0.0,
            T_END,
            0.01,
            Y0,
            None,
            ts=ts_eval,
            stepsize_controller=CTRL,
        )
        # Should have ~100 points
        assert len(ts_out) >= 95
        # Times should be monotonically increasing
        assert jnp.all(jnp.diff(ts_out) > 0)
        # First point should match y0
        assert jnp.allclose(ys_out[0], Y0, atol=1e-4)
        # All states should be finite
        assert jnp.all(jnp.isfinite(ys_out))
        # Biomass should be positive throughout
        assert jnp.all(ys_out[:, 0] > 0)
        # Volume should match (no events change it after last event)
        assert ys_out[-1, 3] == pytest.approx(float(sol.y_final[3]), abs=1e-6)
