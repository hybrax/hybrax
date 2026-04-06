from __future__ import annotations

import equinox as eqx
import jax
import jax.tree_util as jtu


def _default_partition_trainable(module: eqx.Module) -> tuple[eqx.Module, eqx.Module]:
    """Default partitioning: trainable leaves are inexact arrays under `.model`."""
    if not hasattr(module, "model"):
        raise ValueError(
            "default partition_trainable requires a `.model` attribute; "
            "override `partition_trainable()` for custom behavior"
        )
    model_subtree = getattr(module, "model")
    if model_subtree is None:
        raise ValueError(
            "default partition_trainable requires `.model` to be non-None; "
            "override `partition_trainable()` for custom behavior"
        )

    filter_all_false = jtu.tree_map(lambda _leaf: False, module)
    model_filter = jtu.tree_map(eqx.is_inexact_array, model_subtree)
    filter_spec = eqx.tree_at(lambda m: m.model, filter_all_false, model_filter)
    return eqx.partition(module, filter_spec)


def _validate_partition_pair(
    module: eqx.Module,
    trainable: eqx.Module,
    static: eqx.Module,
) -> tuple[eqx.Module, eqx.Module]:
    """Validate trainable/static pytrees can reconstruct the original module."""
    module_leaves, module_treedef = jtu.tree_flatten(
        module,
        is_leaf=lambda value: value is None,
    )
    trainable_leaves, trainable_treedef = jtu.tree_flatten(
        trainable,
        is_leaf=lambda value: value is None,
    )
    static_leaves, static_treedef = jtu.tree_flatten(
        static,
        is_leaf=lambda value: value is None,
    )

    if trainable_treedef != module_treedef or static_treedef != module_treedef:
        raise ValueError("partition_trainable outputs must match module structure")

    for module_leaf, trainable_leaf, static_leaf in zip(
        module_leaves,
        trainable_leaves,
        static_leaves,
        strict=False,
    ):
        trainable_is_none = trainable_leaf is None
        static_is_none = static_leaf is None
        if trainable_is_none and static_is_none:
            if module_leaf is None:
                continue
            raise ValueError(
                "partition_trainable leaves must appear in exactly one partition"
            )
        if not trainable_is_none and not static_is_none:
            raise ValueError(
                "partition_trainable leaves must appear in exactly one partition"
            )

    try:
        combined = eqx.combine(trainable, static)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError(
            "partition_trainable must return compatible trainable/static pytrees"
        ) from exc
    if not eqx.tree_equal(combined, module):
        raise ValueError(
            "partition_trainable outputs must reconstruct the original module"
        )
    return trainable, static


def partition_trainable(module: eqx.Module) -> tuple[eqx.Module, eqx.Module]:
    """Return trainable/static pytrees for a user reaction module.

    If the module exposes `partition_trainable()`, that method is used.
    Otherwise the default contract is applied: inexact-array parameters under
    `.model` are trainable and all other leaves are static.
    """
    method = getattr(module, "partition_trainable", None)
    if callable(method):
        trainable, static = method()
    else:
        trainable, static = _default_partition_trainable(module)
    return _validate_partition_pair(module, trainable, static)


class ReactionOutputs(eqx.Module):
    """Structured return value for user reaction modules.

    Attributes
    ----------
    specific_rates:
        Specific rates ``q_i`` for each species, aligned with the species state
        vector.  These are multiplied by ``X_active`` inside the mechanistic
        ODE (``dc_i/dt = q_i * X_active + transport``).
    modeled_feed_rates:
        Volumetric flow rates for uncontrolled (modeled) feed streams, aligned
        with the modeled-flow ordering from the mechanistic ODE module.
        Use a zero-length array when there are no modeled flows.
    """

    specific_rates: jax.Array
    modeled_feed_rates: jax.Array


class UserReactionModule(eqx.Module):
    """Base abstraction for user-defined reaction modules."""

    def __call__(
        self,
        t: jax.Array,
        c_species: jax.Array,
        controls_vector: jax.Array,
    ) -> ReactionOutputs:
        raise NotImplementedError

    def observe(self, states: jax.Array) -> jax.Array:
        """Optional observation map; default identity."""
        return states

    def partition_trainable(self) -> tuple[eqx.Module, eqx.Module]:
        """Default trainable/static partitioning contract."""
        trainable, static = _default_partition_trainable(self)
        return _validate_partition_pair(self, trainable, static)
