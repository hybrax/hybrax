from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from .controls import build_sample_acc_source_default
from .model_api import ReactionOutputs, UserReactionModule


def default_transform_process_collection(collection, config: dict[str, Any]):
    """Default prep hook for process-collection transformation.

    Supports optional process key renaming via:
    `config["process_rename_map"] = {old_name: new_name}`.
    """
    rename_map = config.get("process_rename_map")
    if rename_map is None:
        return collection
    if not isinstance(rename_map, dict):
        raise TypeError("process_rename_map must be a dict from old name to new name")

    renamed_processes: dict[str, Any] = {}
    for process_name, process in collection.processes.items():
        new_name = process_name
        if process_name in rename_map:
            new_name = str(rename_map[process_name])
            process.metadata.name = new_name
        if new_name in renamed_processes:
            raise ValueError(f"duplicate renamed process key: {new_name}")
        renamed_processes[new_name] = process

    collection.processes = renamed_processes
    return collection


def default_build_sample_acc_series(
    process,
    process_name,
    collection_metadata,
    config,
):
    """Default prep hook for sampled-volume control construction."""
    del process_name, collection_metadata, config
    return build_sample_acc_source_default(process)


class DefaultReactionModule(UserReactionModule):
    """Minimal default reaction model for harness runs.

    This model predicts concentration-space reaction terms from species states.
    It ignores controls and emits no modeled feed rates.
    """

    model: eqx.nn.MLP

    def __init__(self, *, n_species: int, key: jax.Array):
        self.model = eqx.nn.MLP(
            in_size=n_species,
            out_size=n_species,
            width_size=max(8, 2 * n_species),
            depth=2,
            key=key,
        )

    def __call__(
        self,
        t: jax.Array,
        c_species: jax.Array,
        controls_vector: jax.Array,
    ) -> ReactionOutputs:
        del t, controls_vector
        reaction_terms = jnp.asarray(self.model(c_species), dtype=c_species.dtype)
        return ReactionOutputs(
            reaction_terms=reaction_terms,
            modeled_feed_rates=jnp.zeros((0,), dtype=c_species.dtype),
        )


def default_build_reaction_module(
    *,
    target_names: list[str],
    process_names: list[str],
    config: dict[str, Any],
    seed: int,
) -> UserReactionModule:
    """Default train hook for reaction-module construction."""
    del process_names, config
    return DefaultReactionModule(
        n_species=len(target_names),
        key=jax.random.key(int(seed)),
    )


def default_build_modeled_feeds(*, target_names: list[str], config: dict[str, Any]):
    """Default train hook for modeled-feed declarations."""
    del target_names, config
    return ()
