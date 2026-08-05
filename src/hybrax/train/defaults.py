from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from bp_format.mechanistic import build_rhs_ode

from .model_api import (
    LinearScaler,
    LossInputs,
    LossOutputs,
    ReactionInputs,
    ReactionOutputs,
    Scaler,
    UserLossModule,
    UserReactionModule,
    trainable_field,
)
from .run_config import RunConfig


def default_transform_process_collection(collection, config: RunConfig):
    """Default prep hook for process-collection transformation."""
    if config.prepare is None:
        raise ValueError("prepare config section is required")
    rename_map = config.prepare.process_rename_map
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


class DefaultStatefulReactionModule(UserReactionModule):
    """Standard-GRU latent-ODE reaction model with calibrated output heads.

    The GRU consumes physical and control inputs, with ``h`` passed only as its
    hidden state. Input kernels use per-gate Glorot initialization, recurrent
    kernels are per-gate orthogonal, and internal biases start at zero.
    """

    gru_cell: eqx.nn.GRUCell = trainable_field()
    rate_head: eqx.nn.Linear = trainable_field()
    feed_head: eqx.nn.Linear | None = trainable_field()

    def __init__(self, *, key: jax.Array, n_latent: int, **scale_kwargs):
        if n_latent <= 0:
            raise ValueError("DefaultStatefulReactionModule requires n_latent > 0")
        if "SCALE_latent" in scale_kwargs:
            raise ValueError(
                "DefaultStatefulReactionModule sizes SCALE_latent from n_latent"
            )
        scale_kwargs = {
            **scale_kwargs,
            "SCALE_latent": jnp.ones(n_latent, dtype=jnp.float64),
        }
        super().__init__(**scale_kwargs)
        key_gru, key_rate, key_feed = jax.random.split(key, 3)
        gru_key, gru_init_key = jax.random.split(key_gru)
        rate_key, rate_init_key = jax.random.split(key_rate)
        feed_key, feed_init_key = jax.random.split(key_feed)
        n_input = (
            self.n_modeled_RMCs
            + self.n_modeled_PVs
            + 1  # V
            + self.n_modeled_FVCs
            + self.n_controlled_FVCs  # controlled-FVC cumulatives
            + self.n_controlled_FVCs  # controlled-FVC rates
            + self.n_controlled_PVs
        )
        self.gru_cell = eqx.nn.GRUCell(
            input_size=n_input,
            hidden_size=self.n_latent,
            key=gru_key,
        )
        gru_keys = jax.random.split(gru_init_key, 6)
        glorot_init = jax.nn.initializers.glorot_uniform(in_axis=1, out_axis=0)
        orthogonal_init = jax.nn.initializers.orthogonal()
        input_blocks = jnp.split(self.gru_cell.weight_ih, 3)
        recurrent_blocks = jnp.split(self.gru_cell.weight_hh, 3)
        weight_ih = jnp.concatenate(
            [
                glorot_init(gru_keys[i], block.shape, block.dtype)
                for i, block in enumerate(input_blocks)
            ]
        )
        weight_hh = jnp.concatenate(
            [
                orthogonal_init(gru_keys[i + 3], block.shape, block.dtype)
                for i, block in enumerate(recurrent_blocks)
            ]
        )
        self.gru_cell = eqx.tree_at(
            lambda cell: (cell.weight_ih, cell.weight_hh, cell.bias, cell.bias_n),
            self.gru_cell,
            (
                weight_ih,
                weight_hh,
                jnp.zeros_like(self.gru_cell.bias),
                jnp.zeros_like(self.gru_cell.bias_n),
            ),
        )
        n_readout = self.n_latent + self.n_modeled_RMCs + self.n_modeled_PVs
        self.rate_head = eqx.nn.Linear(
            in_features=n_readout,
            out_features=self.n_modeled_BiologicalOde_rates,
            key=rate_key,
        )
        rate_weight = self.rate_head.weight
        if rate_weight.size:
            rate_weight = 0.01 * glorot_init(
                rate_init_key, rate_weight.shape, rate_weight.dtype
            )
        self.rate_head = eqx.tree_at(
            lambda head: (head.weight, head.bias),
            self.rate_head,
            (rate_weight, jnp.zeros_like(self.rate_head.bias)),
        )
        self.feed_head = (
            eqx.nn.Linear(
                in_features=n_readout,
                out_features=self.n_modeled_FVCs,
                key=feed_key,
            )
            if self.n_modeled_FVCs
            else None
        )
        if self.feed_head is not None:
            feed_weight = 0.01 * glorot_init(
                feed_init_key,
                self.feed_head.weight.shape,
                self.feed_head.weight.dtype,
            )
            feed_bias = jnp.zeros_like(self.feed_head.bias) + jnp.log(
                jnp.expm1(jnp.asarray(0.01, dtype=self.feed_head.bias.dtype))
            )
            self.feed_head = eqx.tree_at(
                lambda head: (head.weight, head.bias),
                self.feed_head,
                (feed_weight, feed_bias),
            )

    def __call__(self, t: jax.Array, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        h = inputs.SCL_latent
        cell_input = jnp.concatenate(
            [
                inputs.SCL_modeled_RMCs,
                inputs.SCL_modeled_PVs,
                jnp.atleast_1d(inputs.SCL_modeled_V),
                inputs.SCL_modeled_FVCs_cumulative,
                inputs.SCL_controlled_FVCs_cumulative,
                inputs.SCL_controlled_FVCs_rates,
                inputs.SCL_controlled_PVs,
            ]
        )
        dh_dt = self.gru_cell(cell_input, h) - h
        readout = jnp.concatenate([h, inputs.SCL_modeled_RMCs, inputs.SCL_modeled_PVs])
        bio_rates = jnp.asarray(self.rate_head(readout), dtype=h.dtype)
        if self.feed_head is None:
            feed_rates = jnp.zeros((0,), dtype=h.dtype)
        else:
            feed_rates = jax.nn.softplus(self.feed_head(readout)).astype(h.dtype)
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=bio_rates,
            SCL_modeled_FVCs_rates=feed_rates,
            SCL_latent_derivative=dh_dt,
        )


class DefaultReactionModule(UserReactionModule):
    """Minimal default reaction model for harness runs.

    Predicts ``SCL_modeled_BiologicalOde_rates`` (which includes any ``r_<pv>``
    PV rates) from the SCL species + modeled-PV slices. Ignores controls; emits
    zero-valued modeled VC rates. Uses tanh/Glorot through three hidden layers,
    or SiLU/He for deeper networks. The rate head starts near zero.
    """

    model: eqx.nn.MLP = trainable_field()

    def __init__(
        self,
        *,
        key: jax.Array,
        depth: int = 2,
        width_size: int | None = None,
        **scale_kwargs,
    ):
        super().__init__(**scale_kwargs)
        n_in = self.n_modeled_RMCs + self.n_modeled_PVs
        n_out = self.n_modeled_BiologicalOde_rates
        if depth < 0:
            raise ValueError("depth must be non-negative")
        if width_size is None:
            width_size = max(8, 2 * max(n_in, n_out))
        if width_size <= 0:
            raise ValueError("width_size must be positive")
        model_key, init_key = jax.random.split(key)
        self.model = eqx.nn.MLP(
            in_size=n_in,
            out_size=n_out,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.tanh if depth <= 3 else jax.nn.silu,
            key=model_key,
        )

        layer_keys = jax.random.split(init_key, depth + 1)
        glorot_init = jax.nn.initializers.glorot_uniform(in_axis=1, out_axis=0)
        hidden_init = (
            glorot_init
            if depth <= 3
            else jax.nn.initializers.he_uniform(in_axis=1, out_axis=0)
        )
        layers = []
        for i, (layer, layer_key) in enumerate(zip(self.model.layers, layer_keys)):
            init = glorot_init if i == depth else hidden_init
            weight = init(layer_key, layer.weight.shape, layer.weight.dtype)
            if i == depth:
                weight *= 0.01
            layer = eqx.tree_at(lambda linear: linear.weight, layer, weight)
            if layer.bias is not None:
                layer = eqx.tree_at(
                    lambda linear: linear.bias, layer, jnp.zeros_like(layer.bias)
                )
            layers.append(layer)
        self.model = eqx.tree_at(lambda mlp: mlp.layers, self.model, tuple(layers))

    def __call__(self, t: jax.Array, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        dtype = inputs.SCL_modeled_RMCs.dtype
        SCL_features = jnp.concatenate(
            [inputs.SCL_modeled_RMCs, inputs.SCL_modeled_PVs]
        )
        SCL_modeled_BiologicalOde_rates = jnp.asarray(
            self.model(SCL_features), dtype=dtype
        )
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=SCL_modeled_BiologicalOde_rates,
            SCL_modeled_FVCs_rates=jnp.zeros((self.n_modeled_FVCs,), dtype=dtype),
        )


def default_build_reaction_module(
    *,
    target_names: list[str],
    process_names: list[str],
    config: RunConfig,
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
    scale axes default to unit scales (no scaling).
    """
    del config, target_names
    if not process_names:
        raise ValueError("default_build_reaction_module requires at least one process")
    first_process = collection.processes[process_names[0]]
    rhs_ode = build_rhs_ode(first_process)
    # Scales are sized by the modeled RMC state slice, not by measured targets:
    # combined/PV target sets have their own SCALE_modeled_PVs axis.
    n_RMCs = len(rhs_ode.name_modeled_RMCs)
    n_rates = len(rhs_ode.name_modeled_rates)
    n_modeled_FVCs = len(rhs_ode.name_modeled_FVCs)
    n_controlled_FVCs = len(rhs_ode.name_controlled_FVCs)

    # If no scales provided, fall back to unit scales so the wrapper constructor
    # (which validates shapes) still accepts the module.
    if not scale_kwargs:
        scale_kwargs = _default_scale_kwargs(
            n_RMCs=n_RMCs,
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
    config: RunConfig,
    seed: int,
    collection: Any,
) -> UserLossModule:
    """Default train hook for loss-module construction (per-target MSE)."""
    del process_names, config, seed, collection
    return DefaultLossModule(target_names=list(target_names))


def _default_scale_kwargs(
    *,
    n_RMCs: int,
    n_rates: int,
    n_modeled_FVCs: int,
    n_controlled_FVCs: int,
    rhs_ode: Any,
) -> dict[str, Scaler]:
    """All-ones defaults for every SCALE_* axis, as ``LinearScaler``.

    Used when no estimate hook is supplied. Returns scalers (not bare arrays)
    so the no-hook path matches the hook path's promotion.
    """
    one = jnp.float64(1.0)
    return {
        "SCALE_modeled_RMCs": LinearScaler(jnp.ones(n_RMCs, dtype=jnp.float64)),
        "SCALE_modeled_PVs": LinearScaler(
            jnp.ones(len(rhs_ode.name_modeled_PVs), dtype=jnp.float64)
        ),
        "SCALE_V_in_cumulative": LinearScaler(one),
        "SCALE_modeled_FVCs_cumulative": LinearScaler(
            jnp.ones(n_modeled_FVCs, dtype=jnp.float64)
        ),
        "SCALE_controlled_FVCs_cumulative": LinearScaler(
            jnp.ones(n_controlled_FVCs, dtype=jnp.float64)
        ),
        "SCALE_controlled_FVCs_rates": LinearScaler(
            jnp.ones(n_controlled_FVCs, dtype=jnp.float64)
        ),
        "SCALE_controlled_FVCs_Cin": LinearScaler(
            jnp.ones((n_controlled_FVCs, n_RMCs), dtype=jnp.float64)
        ),
        "SCALE_controlled_PVs": LinearScaler(
            jnp.ones(len(rhs_ode.name_controlled_PVs), dtype=jnp.float64)
        ),
        "SCALE_modeled_FVCs_Cin": LinearScaler(
            jnp.ones((n_modeled_FVCs, n_RMCs), dtype=jnp.float64)
        ),
        "SCALE_modeled_BiologicalOde_rates": LinearScaler(
            jnp.ones(n_rates, dtype=jnp.float64)
        ),
        "SCALE_modeled_FVCs_rates": LinearScaler(
            jnp.ones(n_modeled_FVCs, dtype=jnp.float64)
        ),
        "SCALE_latent": LinearScaler(jnp.zeros(0, dtype=jnp.float64)),
    }
