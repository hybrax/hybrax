"""
CallbackSolution: rich solution object from diffeqsolve_with_callbacks.
"""

from __future__ import annotations

import jax.numpy as jnp
import equinox as eqx


class CallbackSolution(eqx.Module):
    """Solution from diffeqsolve_with_callbacks.

    Attributes:
        y_final: Final state after all events and integration.
        t_final: Final time reached.
        fail_time: Time of the first segment failure (max_steps / dt_min / etc.);
            ``inf`` if the lane never failed. Measurements at ``t > fail_time`` are
            past a failed solve and should be masked out of the loss.

        event_times: (max_events,) times when events triggered.
            Padded with t1 for unused slots.
        event_types: (max_events,) int. Which callback triggered:
            -1 = no event (unused slot)
             0..n_continuous-1 = ContinuousCallback index
             n_continuous..n_continuous+n_preset-1 = PresetTimeCallback index
        event_states_before: (max_events, state_dim) states just before each event.
        event_states_after: (max_events, state_dim) states just after each affect.
        event_count: Total number of events that triggered.
    """

    y_final: jnp.ndarray
    t_final: jnp.ndarray
    fail_time: jnp.ndarray

    event_times: jnp.ndarray
    event_types: jnp.ndarray
    event_states_before: jnp.ndarray
    event_states_after: jnp.ndarray
    event_count: jnp.ndarray

    def get_events(self, callback_index: int = None):
        """Get event times and states, optionally filtered by callback index.

        Returns:
            times, states_before, states_after — NaN for non-matching slots.
        """
        mask = self.event_types >= 0
        if callback_index is not None:
            mask = mask & (self.event_types == callback_index)
        return (
            jnp.where(mask, self.event_times, jnp.nan),
            jnp.where(mask[:, None], self.event_states_before, jnp.nan),
            jnp.where(mask[:, None], self.event_states_after, jnp.nan),
        )

    def count_by_type(self, callback_index: int) -> jnp.ndarray:
        """Count events triggered by a specific callback."""
        return jnp.sum((self.event_types == callback_index).astype(jnp.int32))

    def print_events(self, state_names: list[str] = None, callback_names: dict = None):
        """Pretty-print the event log.

        Args:
            state_names: Names for state variables, e.g. ["X", "S", "P", "V"].
            callback_names: Dict mapping type index to name,
                e.g. {0: "feed", 1: "sample"}.
        """
        n = int(self.event_count)
        if n == 0:
            print("No events triggered.")
            return
        print(f"{'#':>3}  {'Time':>10}  {'Type':>8}  State before → after")
        print("-" * 65)
        for i in range(n):
            t = float(self.event_times[i])
            typ = int(self.event_types[i])
            before = self.event_states_before[i]
            after = self.event_states_after[i]

            if callback_names and typ in callback_names:
                type_str = callback_names[typ]
            else:
                type_str = f"cb[{typ}]"

            if state_names:
                before_str = ", ".join(
                    f"{n}={v:.3f}" for n, v in zip(state_names, before)
                )
                after_str = ", ".join(
                    f"{n}={v:.3f}" for n, v in zip(state_names, after)
                )
            else:
                before_str = ", ".join(f"{v:.3f}" for v in before)
                after_str = ", ".join(f"{v:.3f}" for v in after)
            print(
                f"{i + 1:3d}  {t:10.4f}  {type_str:>8}  [{before_str}] → [{after_str}]"
            )
