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
        terminated_by_event: Whether a StopConditionCallback ended the solve early.

        event_times: (max_events,) times when callbacks triggered.
            Padded with t1 for unused slots.
        event_types: (max_events,) int. Which callback triggered:
            -1 = no event (unused slot)
             0..n_continuous-1 = ContinuousCallback index
             n_continuous..n_continuous+n_preset-1 = PresetTimeCallback index
        event_states_before: (max_events, state_dim) states just before each event.
        event_states_after: (max_events, state_dim) states just after each affect.
        event_count: Total number of events that triggered.
        segment_num_steps: (max_events,) solver steps taken by each segment. Collapsed
            post-failure and post-``done`` segments take zero steps.

        output_states: (n_output, state_dim) state at each requested ``output_times``
            entry, or ``None`` when ``output_times`` was not passed. This is the
            TRAJECTORY readout: unlike the event log it does not require an output time
            to be a segment boundary, because each segment saves its own points with
            ``SaveAt(ts=...)`` (pure interpolation, no extra solver steps). A time
            no later than the numerically located event is owned by the segment that
            ENDS there, so it reports the PRE-affect state -- the same convention as
            ``event_states_before``. A nominal event time just above the located root
            remains ``inf`` rather than reporting the post-affect state. Rows the solve
            never reached after a failure or stop condition are also ``inf``; use
            ``fail_time`` to classify failures.
        output_overflow: scalar bool, or ``None`` when ``output_times`` was not passed.
            True if any segment owned more output points than ``output_window`` could
            carry, which would silently drop the excess. Callers must treat this as a
            hard error.
    """

    y_final: jnp.ndarray
    t_final: jnp.ndarray
    fail_time: jnp.ndarray
    terminated_by_event: jnp.ndarray

    event_times: jnp.ndarray
    event_types: jnp.ndarray
    event_states_before: jnp.ndarray
    event_states_after: jnp.ndarray
    event_count: jnp.ndarray
    segment_num_steps: jnp.ndarray

    output_states: jnp.ndarray | None = None
    output_overflow: jnp.ndarray | None = None

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
