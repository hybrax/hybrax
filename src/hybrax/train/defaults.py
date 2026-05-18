from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from bp_format.mechanistic import build_rhs_ode

from .controls import build_sample_acc_source_default, run_min_dt_from_config
from .model_api import ReactionOutputs, UserReactionModule, trainable_field


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
    del process_name, collection_metadata
    return build_sample_acc_source_default(
        process,
        run_min_dt=run_min_dt_from_config(config),
    )


class DefaultReactionModule(UserReactionModule):
    """Minimal default reaction model for harness runs.

    Predicts a flat ``specific_rates`` vector aligned with
    ``rhs_ode.name_modeled_rates``. Ignores controls; emits no modeled feed
    rates.
    """

    model: eqx.nn.MLP = trainable_field()

    def __init__(self, *, n_species: int, n_rates: int, key: jax.Array):
        self.model = eqx.nn.MLP(
            in_size=n_species,
            out_size=n_rates,
            width_size=max(8, 2 * max(n_species, n_rates)),
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
        specific_rates = jnp.asarray(self.model(c_species), dtype=c_species.dtype)
        return ReactionOutputs(
            specific_rates=specific_rates,
            modeled_feed_rates=jnp.zeros((0,), dtype=c_species.dtype),
        )


def default_build_reaction_module(
    *,
    target_names: list[str],
    process_names: list[str],
    config: dict[str, Any],
    seed: int,
    collection: Any,
) -> UserReactionModule:
    """Default train hook for reaction-module construction.

    Derives the rates head size from the first process's BiologicalOde via
    ``rhs_ode.name_modeled_rates`` so user-defined ODEs with rate counts that
    differ from the species count are supported out of the box.
    """
    del config
    if not process_names:
        raise ValueError("default_build_reaction_module requires at least one process")
    first_process = collection.processes[process_names[0]]
    rhs_ode = build_rhs_ode(first_process)
    n_rates = len(rhs_ode.name_modeled_rates)
    return DefaultReactionModule(
        n_species=len(target_names),
        n_rates=n_rates,
        key=jax.random.key(int(seed)),
    )
