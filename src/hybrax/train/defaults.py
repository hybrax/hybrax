from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from bp_format.mechanistic import build_rhs_ode

from .controls import build_sample_acc_source_default, run_min_dt_from_config
from .model_api import (
    LossInputs,
    LossOutputs,
    ReactionInputs,
    ReactionOutputs,
    UserLossModule,
    UserReactionModule,
    trainable_field,
)


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

    Predicts ``SCL_modeled_BiologicalOde_rates`` from the SCL species slice.
    Ignores controls; emits zero-length modeled VC rates.
    """

    model: eqx.nn.MLP = trainable_field()

    def __init__(self, *, key: jax.Array, **scale_kwargs):
        super().__init__(**scale_kwargs)
        n_in = self.n_modeled_RMCs
        n_out = self.n_modeled_BiologicalOde_rates
        self.model = eqx.nn.MLP(
            in_size=n_in,
            out_size=n_out,
            width_size=max(8, 2 * max(n_in, n_out)),
            depth=2,
            key=key,
        )

    def __call__(self, t: jax.Array, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        SCL_modeled_RMCs = inputs.SCL_modeled_RMCs
        SCL_modeled_BiologicalOde_rates = jnp.asarray(
            self.model(SCL_modeled_RMCs), dtype=SCL_modeled_RMCs.dtype
        )
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=SCL_modeled_BiologicalOde_rates,
            SCL_modeled_FVCs_rates=jnp.zeros(
                (self.n_modeled_FVCs,), dtype=SCL_modeled_RMCs.dtype
            ),
        )


def default_build_reaction_module(
    *,
    target_names: list[str],
    process_names: list[str],
    config: dict[str, Any],
    seed: int,
    collection: Any,
    **scale_kwargs: Any,
) -> UserReactionModule:
    """Default train hook for reaction-module construction.

    Derives the rates head size from the first process's BiologicalOde via
    ``rhs_ode.name_modeled_rates`` so user-defined ODEs with rate counts that
    differ from the species count are supported out of the box.

    If the optional ``estimate_all_scales`` hook supplied SCALE_* values, they
    arrive via ``scale_kwargs`` and are stored on the module. Otherwise the
    13 axes default to ones (no scaling).
    """
    del config
    if not process_names:
        raise ValueError("default_build_reaction_module requires at least one process")
    first_process = collection.processes[process_names[0]]
    rhs_ode = build_rhs_ode(first_process)
    n_species = len(target_names)
    n_rates = len(rhs_ode.name_modeled_rates)
    n_modeled_FVCs = len(rhs_ode.name_modeled_FVCs)
    n_controlled_FVCs = len(rhs_ode.name_controlled_FVCs)

    # If no scales provided, fall back to all-ones for every axis so the wrapper
    # constructor (which validates shapes) still accepts the module.
    if not scale_kwargs:
        _proc = collection.processes[process_names[0]]
        scale_kwargs = _default_scale_kwargs(
            n_species=n_species,
            n_rates=n_rates,
            n_modeled_FVCs=n_modeled_FVCs,
            n_controlled_FVCs=n_controlled_FVCs,
            rhs_ode=rhs_ode,
        )

    return DefaultReactionModule(
        key=jax.random.key(int(seed)),
        **scale_kwargs,
    )


class DefaultLossModule(UserLossModule):
    """Per-target SCL-space measurement loss — the default when no loss hook.

    Emits one named term per measured target (named after the target). Override
    ``residual_reduction`` to swap the per-target reduction (MSE → MAE / Huber).
    """

    target_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, *, target_names):
        self.target_names = tuple(target_names)

    @property
    def loss_names(self) -> tuple[str, ...]:
        return self.target_names

    def residual_reduction(self, residual, mask):
        """Per-column reduction of the masked residual; default mean-squared.

        ``residual`` / ``mask`` are ``(n_meas, n_target)``; returns
        ``(n_target,)``. Each column is normalised by its own active-cell count
        so sparsely-measured targets are not diluted by padding rows.
        """
        sq = jnp.square(residual)
        masked = jnp.where(mask, sq, 0.0)
        n_active = jnp.maximum(jnp.sum(mask, axis=0), 1)
        return jnp.sum(masked, axis=0) / n_active

    def __call__(self, inputs: LossInputs) -> LossOutputs:
        residual = inputs.SCL_target_pred - jnp.where(
            inputs.mask_measured, inputs.SCL_target_measured, 0.0
        )
        per_target = self.residual_reduction(residual, inputs.mask_measured)
        return LossOutputs(
            named_losses={
                name: per_target[i] for i, name in enumerate(self.target_names)
            }
        )


def default_build_loss_module(
    *,
    target_names: list[str],
    process_names: list[str],
    config: dict[str, Any],
    seed: int,
    collection: Any,
) -> UserLossModule:
    """Default train hook for loss-module construction (per-target MSE)."""
    del process_names, config, seed, collection
    return DefaultLossModule(target_names=list(target_names))


def _default_scale_kwargs(
    *,
    n_species: int,
    n_rates: int,
    n_modeled_FVCs: int,
    n_controlled_FVCs: int,
    rhs_ode: Any,
) -> dict[str, jnp.ndarray]:
    """All-ones defaults for every SCALE_* axis. Used when no estimate hook is supplied.

    Bolus FVCs (n_controlled_FVCs_bolus) aren't reachable from rhs_ode here, so the
    placeholder size 0 is used; the wrapper's shape validation will surface a clean
    error if the actual layout has bolus events and no estimate_all_scales hook is
    configured.
    """
    one = jnp.float32(1.0)
    return {
        "SCALE_modeled_RMCs": jnp.ones(n_species, dtype=jnp.float32),
        "SCALE_V_in_cumulative": one,
        "SCALE_modeled_FVCs_cumulative": jnp.ones(n_modeled_FVCs, dtype=jnp.float32),
        "SCALE_controlled_FVCs_cumulative": jnp.ones(
            n_controlled_FVCs, dtype=jnp.float32
        ),
        "SCALE_controlled_FVCs_rates": jnp.ones(n_controlled_FVCs, dtype=jnp.float32),
        "SCALE_controlled_FVCs_Cin": jnp.ones(
            (n_controlled_FVCs, n_species), dtype=jnp.float32
        ),
        "SCALE_controlled_FVCs_bolus_rates": jnp.ones(0, dtype=jnp.float32),
        "SCALE_controlled_PVs": jnp.ones(
            len(rhs_ode.name_controlled_PVs), dtype=jnp.float32
        ),
        "SCALE_modeled_FVCs_Cin": jnp.ones(
            (n_modeled_FVCs, n_species), dtype=jnp.float32
        ),
        "SCALE_modeled_BiologicalOde_rates": jnp.ones(n_rates, dtype=jnp.float32),
        "SCALE_modeled_FVCs_rates": jnp.ones(n_modeled_FVCs, dtype=jnp.float32),
    }
