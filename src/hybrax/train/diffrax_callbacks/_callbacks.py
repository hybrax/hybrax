"""
Callback types for diffrax_callbacks.

Julia DifferentialEquations.jl equivalents:
  ContinuousCallback  -> ContinuousCallback (zero-crossing with root-finding)
  DiscreteCallback     -> DiscreteCallback (evaluated at segment boundaries)
  PresetTimeCallback   -> PresetTimeCallback (events at known times)
  PeriodicCallback     -> PeriodicCallback (events every Δt)
  VectorContinuousCallback -> multiple ContinuousCallbacks in a CallbackSet
  CallbackSet          -> CallbackSet (combines multiple callbacks)
  ManifoldProjection   -> ManifoldProjection (project state after each event)

Design principles:
  - Pure functional: affect(y, t, args) -> new_y (no mutation)
  - Equinox modules: proper pytree nodes, JIT/grad compatible
  - Composable: CallbackSet combines arbitrary callbacks
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import jax.numpy as jnp
import equinox as eqx
import optimistix as optx


# ================================================================
# ContinuousCallback
# ================================================================


class ContinuousCallback(eqx.Module):
    """Triggers when a continuous condition function crosses zero.

    Uses root-finding to locate the exact crossing time within a solver step.

    Args:
        condition_fn: (y, t, args) -> scalar. Event triggers on zero-crossing.
        affect_fn: (y, t, args) -> new_y. Applied when event triggers.
        direction: "up" (neg->pos), "down" (pos->neg), or "both".
        root_finder: Optimistix root finder for exact event time.
        repeat_nudge: Minimum time advance after an event before the same
            condition can re-trigger. Prevents infinite loops when the affect
            doesn't change the condition sign (e.g., bleed that doesn't reduce
            biomass concentration). Default 1e-6.

    Example:
        cb = ContinuousCallback(
            condition_fn=lambda y, t, args: y[1] - 1.0,
            affect_fn=lambda y, t, args: apply_feed(y),
            direction="down",
        )
    """

    condition_fn: Callable
    affect_fn: Callable
    direction: str = "both"
    root_finder: optx.AbstractRootFinder = optx.Newton(rtol=1e-8, atol=1e-10)
    repeat_nudge: float = 1e-6

    def __check_init__(self):
        if self.direction not in ("up", "down", "both"):
            raise ValueError(
                f"direction must be 'up', 'down', or 'both', got {self.direction!r}"
            )

    @property
    def _diffrax_direction(self) -> Optional[bool]:
        if self.direction == "up":
            return True
        elif self.direction == "down":
            return False
        return None


# ================================================================
# DiscreteCallback
# ================================================================


class DiscreteCallback(eqx.Module):
    """Evaluated at each segment boundary (between events).

    Unlike Julia's DiscreteCallback which runs at every solver step, this
    runs at each event boundary in the scan loop. For most practical purposes
    (checking bounds, applying corrections), this is equivalent.

    Args:
        condition_fn: (y, t, args) -> bool. Checked at segment boundaries.
        affect_fn: (y, t, args) -> new_y. Applied when condition is True.

    Example:
        # Clamp biomass to maximum
        cb = DiscreteCallback(
            condition_fn=lambda y, t, args: y[0] > 100.0,
            affect_fn=lambda y, t, args: y.at[0].set(100.0),
        )
    """

    condition_fn: Callable
    affect_fn: Callable


# ================================================================
# PresetTimeCallback
# ================================================================


class PresetTimeCallback(eqx.Module):
    """Triggers at predetermined times.

    The solver is forced to step exactly to each preset time.

    Args:
        times: Array of times at which to trigger.
        affect_fn: ``(y, t, args, preset_index) -> new_y``. Applied at each preset
            time. ``preset_index`` is the index into **this callback's own**
            ``times`` (not into the merged/sorted array the solver scans), so an
            affect can identify *which* preset fired without comparing floats.
            During speculative batched evaluation, ``preset_index`` may be ``-1``;
            the returned state is then discarded. Affects must therefore be pure and
            must mask validation or other JAX effects when ``preset_index < 0``.
            Under ``vmap``, JAX may evaluate every callback branch, not only the one
            selected for a lane.

            Prefer the index over ``t`` when ``preset_index >= 0``: ``t`` is the
            solver's *realised* stop time
            and the merged preset array is cast to the solve's working dtype, so
            neither is guaranteed bit-identical to the value in ``times``. Matching
            on ``times[preset_index]`` is exact by construction. Matching on ``t``
            with a tolerance instead makes any other node within that tolerance
            re-trigger this preset — see ``hybrax.train.physical_solve.affect_fn``,
            which used to need a parking guard for exactly that reason.

    Example:
        cb = PresetTimeCallback(
            times=jnp.array([12.0, 24.0, 36.0]),
            affect_fn=lambda y, t, args, i: take_sample(y),
        )
    """

    times: jnp.ndarray
    affect_fn: Callable


# ================================================================
# PeriodicCallback
# ================================================================


class PeriodicCallback(eqx.Module):
    """Triggers every Δt time units.

    Convenience wrapper around PresetTimeCallback for uniform spacing.

    Args:
        dt: Time interval between triggers.
        affect_fn: ``(y, t, args, preset_index) -> new_y``. Applied at each trigger.
            ``to_preset`` hands this straight to a :class:`PresetTimeCallback`, so it
            takes the same 4-argument contract; ``preset_index`` is the slot in the
            generated ``times``, or ``-1`` during speculative batched evaluation.
        t_start: First trigger time. Default: dt (skip t=0).
        t_end: Last possible trigger time. Must be set for array pre-allocation.

    Example:
        # Log state every 2 hours over a 48h fermentation
        cb = PeriodicCallback(dt=2.0, affect_fn=log_fn, t_end=48.0)
    """

    dt: float
    affect_fn: Callable
    t_start: float = 0.0
    t_end: float = 100.0

    def to_preset(self) -> PresetTimeCallback:
        """Convert to a PresetTimeCallback with pre-computed times."""
        start = self.t_start if self.t_start > 0 else self.dt
        times = jnp.arange(start, self.t_end + 1e-10, self.dt)
        return PresetTimeCallback(times=times, affect_fn=self.affect_fn)


# ================================================================
# ManifoldProjection
# ================================================================


class ManifoldProjection(eqx.Module):
    """Projects the state onto a manifold after each event.

    Common use: enforce physical constraints like non-negative concentrations
    or mass conservation.

    Applied as a DiscreteCallback that always triggers.

    Args:
        project_fn: (y, t, args) -> new_y. Projects state onto the manifold.

    Example:
        # Ensure all concentrations are non-negative
        cb = ManifoldProjection(
            project_fn=lambda y, t, args: jnp.maximum(y, 0.0),
        )

        # Enforce mass conservation: X + S/Yxs = const
        def conserve_mass(y, t, args):
            total = y[0] + y[1] / 0.5  # X + S/Yxs
            target = 20.0  # known conserved quantity
            correction = (target - total) * 0.5 / (1 + 0.5)
            return y.at[0].add(correction).at[1].add(correction / 0.5)
        cb = ManifoldProjection(project_fn=conserve_mass)
    """

    project_fn: Callable

    def to_discrete(self) -> DiscreteCallback:
        """Convert to a DiscreteCallback that always triggers."""
        return DiscreteCallback(
            condition_fn=lambda y, t, args: True,
            affect_fn=self.project_fn,
        )


# ================================================================
# CallbackSet
# ================================================================


class CallbackSet(eqx.Module):
    """Combines multiple callbacks with priority handling.

    Priority order (matching Julia):
      1. ContinuousCallbacks — earliest event wins
      2. PresetTimeCallbacks / PeriodicCallbacks — at exact times
      3. DiscreteCallbacks / ManifoldProjection — at segment boundaries

    DiscreteCallbacks run AFTER every segment (including after continuous
    and preset events).

    Args:
        *callbacks: Any mix of callback types or nested CallbackSets.

    Example:
        cb = CallbackSet(
            ContinuousCallback(...),   # feed on low substrate
            PresetTimeCallback(...),   # sample at fixed times
            ManifoldProjection(...),   # enforce non-negative concentrations
        )
    """

    continuous_callbacks: tuple[ContinuousCallback, ...]
    preset_callbacks: tuple[PresetTimeCallback, ...]
    discrete_callbacks: tuple[DiscreteCallback, ...]

    def __init__(
        self,
        *callbacks: Union[
            ContinuousCallback,
            DiscreteCallback,
            PresetTimeCallback,
            PeriodicCallback,
            ManifoldProjection,
            "CallbackSet",
        ],
    ):
        continuous = []
        preset = []
        discrete = []
        for cb in callbacks:
            if isinstance(cb, ContinuousCallback):
                continuous.append(cb)
            elif isinstance(cb, PresetTimeCallback):
                preset.append(cb)
            elif isinstance(cb, PeriodicCallback):
                preset.append(cb.to_preset())
            elif isinstance(cb, DiscreteCallback):
                discrete.append(cb)
            elif isinstance(cb, ManifoldProjection):
                discrete.append(cb.to_discrete())
            elif isinstance(cb, CallbackSet):
                continuous.extend(cb.continuous_callbacks)
                preset.extend(cb.preset_callbacks)
                discrete.extend(cb.discrete_callbacks)
            else:
                raise TypeError(f"Unknown callback type: {type(cb)}")
        object.__setattr__(self, "continuous_callbacks", tuple(continuous))
        object.__setattr__(self, "preset_callbacks", tuple(preset))
        object.__setattr__(self, "discrete_callbacks", tuple(discrete))

    @property
    def n_continuous(self) -> int:
        """Number of continuous callbacks in the set."""
        return len(self.continuous_callbacks)

    @property
    def n_preset(self) -> int:
        """Number of preset-time callbacks in the set."""
        return len(self.preset_callbacks)

    @property
    def n_discrete(self) -> int:
        """Number of discrete callbacks in the set."""
        return len(self.discrete_callbacks)

    def _preset_sort_order(self) -> jnp.ndarray:
        """Permutation that sorts the concatenated preset times.

        Single source of the ordering for ``get_all_preset_times`` /
        ``get_preset_affect_indices`` / ``get_preset_local_indices``, so the three
        derived arrays cannot drift apart. (They previously used ``jnp.sort`` and
        ``jnp.argsort`` independently, which agree only while both stay stable.)
        """
        all_times = jnp.concatenate([cb.times for cb in self.preset_callbacks])
        return jnp.argsort(all_times)

    def get_all_preset_times(self) -> jnp.ndarray:
        """Every preset callback's times, concatenated and sorted."""
        if not self.preset_callbacks:
            return jnp.array([])
        all_times = jnp.concatenate([cb.times for cb in self.preset_callbacks])
        return all_times[self._preset_sort_order()]

    def get_preset_affect_indices(self) -> jnp.ndarray:
        """For each sorted preset slot: which preset callback owns it."""
        if not self.preset_callbacks:
            return jnp.array([], dtype=jnp.int32)
        all_indices = jnp.concatenate(
            [
                jnp.full(cb.times.shape[0], i, dtype=jnp.int32)
                for i, cb in enumerate(self.preset_callbacks)
            ]
        )
        return all_indices[self._preset_sort_order()]

    def get_preset_local_indices(self) -> jnp.ndarray:
        """For each sorted preset slot: its index within its OWN callback's ``times``.

        A firing preset's local index is handed to
        ``PresetTimeCallback.affect_fn`` as ``preset_index``, letting an affect look
        the time up in its own array exactly, with no float comparison or dtype
        round-trip. The solver may instead pass ``-1`` during speculative batched
        evaluation; see ``PresetTimeCallback``'s contract.
        """
        if not self.preset_callbacks:
            return jnp.array([], dtype=jnp.int32)
        local = jnp.concatenate(
            [
                jnp.arange(cb.times.shape[0], dtype=jnp.int32)
                for cb in self.preset_callbacks
            ]
        )
        return local[self._preset_sort_order()]

    def get_max_repeat_nudge(self) -> float:
        """Get the maximum repeat_nudge across all continuous callbacks."""
        if not self.continuous_callbacks:
            return 0.0
        return max(cc.repeat_nudge for cc in self.continuous_callbacks)
