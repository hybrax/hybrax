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
    n_dense: int | None = None,
    n_prediction: int | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
]:
    """Splice up to two evenly-spaced grids into the measurement grid.

    ``n_dense`` is the loss module's dense grid (consumed by its loss);
    ``n_prediction`` is an independent output grid (consumed by forward
    evaluation for ``predictions.csv``/plots). Either may be ``None``; both
    spliced into the **same** sorted union so a single solve serves both. Both
    counts are static.

    Returns ``(t_eval_sorted, sample_indices, dense_t, dense_indices,
    prediction_t, prediction_indices)``:
    - ``t_eval_sorted`` — the sorted union grid; pass to the solver as ``t_eval``.
    - ``sample_indices`` — for each measurement row, which output row in the
      sorted union it corresponds to (so ``save_outputs[sample_indices]`` gives
      the measurement-grid view).
    - ``dense_t`` / ``dense_indices`` — the loss dense linspace and its sorted
      positions (``None`` when ``n_dense is None``).
    - ``prediction_t`` / ``prediction_indices`` — analogous for the prediction
      grid (``None`` when ``n_prediction is None``).
    """
    t0 = t_measured[0]
    t1 = t_measured[jnp.maximum(jnp.asarray(n_measured, dtype=jnp.int32) - 1, 0)]
    n_sample = t_measured.shape[0]

    blocks = [t_measured]
    dense_t = None
    prediction_t = None
    if n_dense is not None:
        dense_t = jnp.linspace(t0, t1, n_dense)
        blocks.append(dense_t)
    if n_prediction is not None:
        prediction_t = jnp.linspace(t0, t1, n_prediction)
        blocks.append(prediction_t)

    t_unsorted = jnp.concatenate(blocks)
    order = jnp.argsort(t_unsorted)
    inverse_order = jnp.empty_like(order)
    inverse_order = inverse_order.at[order].set(
        jnp.arange(order.shape[0], dtype=order.dtype)
    )
    sample_indices = inverse_order[:n_sample]
    offset = n_sample
    dense_indices = None
    prediction_indices = None
    if n_dense is not None:
        dense_indices = inverse_order[offset : offset + n_dense]
        offset += n_dense
    if n_prediction is not None:
        prediction_indices = inverse_order[offset : offset + n_prediction]
        offset += n_prediction
    return (
        t_unsorted[order],
        sample_indices,
        dense_t,
        dense_indices,
        prediction_t,
        prediction_indices,
    )


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
