"""Dense-grid helpers for loss modules.

When a :class:`UserLossModule` opts in via the ``dense_grid_n`` property, the
trainer solves the ODE on the union of the measurement times and a uniform
linspace, then exposes both views on :class:`LossInputs`. The helpers here
build that union grid and provide mask utilities for skipping dense points
near controls-discontinuities (where finite differences are unreliable).

Lifted verbatim from the Martens structured example's internals so any
consumer can use them without re-inventing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def build_union_time_grid(
    t_measured: jax.Array,
    n_measured: jax.Array,
    n_dense: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Splice ``n_dense`` evenly-spaced points into the measurement grid.

    Returns ``(t_eval_sorted, sample_indices, dense_t, dense_indices)``:
    - ``t_eval_sorted`` — the sorted union grid; pass to the solver as ``t_eval``.
    - ``sample_indices`` — for each measurement row, which output row in the
      sorted union it corresponds to (so ``save_outputs[sample_indices]`` gives
      the measurement-grid view).
    - ``dense_t`` — the original (unsorted) dense linspace, useful for
      finite-difference computations on it.
    - ``dense_indices`` — analogous to ``sample_indices`` for the dense block.
    """
    t0 = t_measured[0]
    t1 = t_measured[jnp.maximum(jnp.asarray(n_measured, dtype=jnp.int32) - 1, 0)]
    dense_t = jnp.linspace(t0, t1, n_dense)
    t_unsorted = jnp.concatenate([t_measured, dense_t])
    order = jnp.argsort(t_unsorted)
    inverse_order = jnp.empty_like(order)
    inverse_order = inverse_order.at[order].set(
        jnp.arange(order.shape[0], dtype=order.dtype)
    )
    n_sample = t_measured.shape[0]
    sample_indices = inverse_order[:n_sample]
    dense_indices = inverse_order[n_sample:]
    return t_unsorted[order], sample_indices, dense_t, dense_indices


def dense_point_mask_away_from_jumps(
    dense_t: jax.Array,
    jump_ts: jax.Array | None,
    jump_epsilon_h: float,
) -> jax.Array:
    """Per-point mask: True iff ``dense_t[i]`` is farther than ``jump_epsilon_h``
    from every jump time. When ``jump_ts is None`` (no discontinuities), every
    point is kept.
    """
    if jump_ts is None:
        return jnp.ones(dense_t.shape, dtype=bool)
    jump_eps = jnp.asarray(jump_epsilon_h, dtype=dense_t.dtype)
    valid_jump = jnp.isfinite(jump_ts)
    near_jump = jnp.any(
        valid_jump[None, :]
        & (jnp.abs(dense_t[:, None] - jump_ts[None, :]) <= jump_eps),
        axis=1,
    )
    return ~near_jump


def dense_triple_mask_away_from_jumps(
    dense_t: jax.Array,
    jump_ts: jax.Array | None,
    jump_epsilon_h: float,
) -> jax.Array:
    """Per-triple mask: True for triples ``(i-1, i, i+1)`` whose time span
    does not straddle any jump (with ``jump_epsilon_h`` padding on both sides).
    Length ``dense_t.shape[0] - 2``. Use for finite-difference curvature
    penalties so curvature is never measured across a discontinuity.
    """
    if jump_ts is None:
        return jnp.ones((dense_t.shape[0] - 2,), dtype=bool)
    jump_eps = jnp.asarray(jump_epsilon_h, dtype=dense_t.dtype)
    valid_jump = jnp.isfinite(jump_ts)
    left = dense_t[:-2, None] - jump_eps
    right = dense_t[2:, None] + jump_eps
    crosses_jump = jnp.any(
        valid_jump[None, :] & (jump_ts[None, :] > left) & (jump_ts[None, :] < right),
        axis=1,
    )
    return ~crosses_jump
