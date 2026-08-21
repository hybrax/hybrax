from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from hybrax.format.dataclasses import BioProcess, BioProcessCollection
from hybrax.format.mechanistic import RhsOde

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
from .runtime_context import rhs_ode_from_training_parents


def default_transform_process_collection(collection, config: RunConfig):
    """Default prep hook for process-collection transformation."""
    if config.prepare is None:
        raise ValueError("prepare config section is required")
    rename_map = config.prepare.process_rename_map
    if rename_map is None:
        return collection
    if not isinstance(rename_map, dict):
        raise TypeError("process_rename_map must be a dict from old name to new name")

    renamed_processes: dict[str, BioProcess] = {}
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
    inflow_head: eqx.nn.Linear | None = trainable_field()
    outflow_head: eqx.nn.Linear | None = trainable_field()

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
        key_gru, key_rate, key_inflow, key_outflow = jax.random.split(key, 4)
        gru_key, gru_init_key = jax.random.split(key_gru)
        rate_key, rate_init_key = jax.random.split(key_rate)
        inflow_key, inflow_init_key = jax.random.split(key_inflow)
        outflow_key, outflow_init_key = jax.random.split(key_outflow)
        n_input = (
            self.n_modeled_RMCs
            + self.n_modeled_PVs
            + 1  # V
            + self.n_modeled_Inflows
            + self.n_modeled_Outflows
            + self.n_controlled_Inflows  # controlled-Inflow cumulatives
            + self.n_controlled_Inflows  # controlled-Inflow rates
            + self.n_controlled_Outflows  # controlled-Outflow cumulatives
            + self.n_controlled_Outflows  # controlled-Outflow rates
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
        self.inflow_head = (
            eqx.nn.Linear(
                in_features=n_readout,
                out_features=self.n_modeled_Inflows,
                key=inflow_key,
            )
            if self.n_modeled_Inflows
            else None
        )
        if self.inflow_head is not None:
            inflow_weight = 0.01 * glorot_init(
                inflow_init_key,
                self.inflow_head.weight.shape,
                self.inflow_head.weight.dtype,
            )
            inflow_bias = jnp.zeros_like(self.inflow_head.bias) + jnp.log(
                jnp.expm1(jnp.asarray(0.01, dtype=self.inflow_head.bias.dtype))
            )
            self.inflow_head = eqx.tree_at(
                lambda head: (head.weight, head.bias),
                self.inflow_head,
                (inflow_weight, inflow_bias),
            )
        self.outflow_head = (
            eqx.nn.Linear(
                in_features=n_readout,
                out_features=self.n_modeled_Outflows,
                key=outflow_key,
            )
            if self.n_modeled_Outflows
            else None
        )
        if self.outflow_head is not None:
            outflow_weight = 0.01 * glorot_init(
                outflow_init_key,
                self.outflow_head.weight.shape,
                self.outflow_head.weight.dtype,
            )
            outflow_bias = jnp.zeros_like(self.outflow_head.bias) + jnp.log(
                jnp.expm1(jnp.asarray(0.01, dtype=self.outflow_head.bias.dtype))
            )
            self.outflow_head = eqx.tree_at(
                lambda head: (head.weight, head.bias),
                self.outflow_head,
                (outflow_weight, outflow_bias),
            )

    def __call__(self, t: jax.Array, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        h = inputs.SCL_latent
        cell_input = jnp.concatenate(
            [
                inputs.SCL_modeled_RMCs,
                inputs.SCL_modeled_PVs,
                jnp.atleast_1d(inputs.SCL_modeled_V),
                inputs.SCL_modeled_Inflows_cumulative,
                inputs.SCL_modeled_Outflows_cumulative,
                inputs.SCL_controlled_Inflows_cumulative,
                inputs.SCL_controlled_Inflows_rates,
                inputs.SCL_controlled_Outflows_cumulative,
                inputs.SCL_controlled_Outflows_rates,
                inputs.SCL_controlled_PVs,
            ]
        )
        dh_dt = self.gru_cell(cell_input, h) - h
        readout = jnp.concatenate([h, inputs.SCL_modeled_RMCs, inputs.SCL_modeled_PVs])
        bio_rates = jnp.asarray(self.rate_head(readout), dtype=h.dtype)
        if self.inflow_head is None:
            inflow_rates = jnp.zeros((0,), dtype=h.dtype)
        else:
            inflow_rates = jax.nn.softplus(self.inflow_head(readout)).astype(h.dtype)
        if self.outflow_head is None:
            outflow_rates = jnp.zeros((0,), dtype=h.dtype)
        else:
            outflow_rates = -jax.nn.softplus(self.outflow_head(readout)).astype(h.dtype)
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=bio_rates,
            SCL_modeled_Inflows_rates=inflow_rates,
            SCL_modeled_Outflows_rates=outflow_rates,
            SCL_latent_derivative=dh_dt,
        )


class DefaultReactionModule(UserReactionModule):
    """Minimal default reaction model for harness runs.

    Predicts ``SCL_modeled_BiologicalOde_rates`` (which includes any ``r_<pv>``
    PV rates) from the SCL species + modeled-PV slices. Ignores controls; emits
    zero-valued modeled Inflow and Outflow rates. Uses tanh/Glorot for shallow
    networks and SiLU/He for deeper networks. The rate head starts near zero.
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
            weight = layer.weight
            if weight.size:
                weight = init(layer_key, weight.shape, weight.dtype)
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
            SCL_modeled_Inflows_rates=jnp.zeros((self.n_modeled_Inflows,), dtype=dtype),
            SCL_modeled_Outflows_rates=jnp.zeros(
                (self.n_modeled_Outflows,), dtype=dtype
            ),
        )


def default_build_reaction_module(
    *,
    target_names: list[str],
    process_names: list[str],
    config: RunConfig,
    seed: int,
    training_parent_collection: BioProcessCollection,
    **scale_kwargs: Scaler,
) -> UserReactionModule:
    """Default train hook for reaction-module construction.

    Derives the rates head size from the prepared canonical
    ``rhs_ode.name_modeled_rates`` so user-defined ODEs with rate counts that
    differ from the species count are supported out of the box.

    If the optional ``estimate_all_scales`` hook supplied SCALE_* values, they
    arrive via ``scale_kwargs`` and are stored on the module. Otherwise the
    scale axes default to unit scales (no scaling).
    """
    del config, target_names
    if not process_names:
        raise ValueError("default_build_reaction_module requires at least one process")
    rhs_ode = rhs_ode_from_training_parents(
        training_parent_collection,
        empty_message=("default_build_reaction_module requires a training parent"),
    )
    # Scales are sized by the modeled RMC state slice, not by measured targets:
    # combined/PV target sets have their own SCALE_modeled_PVs axis.
    n_RMCs = len(rhs_ode.name_modeled_RMCs)
    n_rates = len(rhs_ode.name_modeled_rates)
    n_modeled_Inflows = len(rhs_ode.name_modeled_Inflows)
    n_modeled_Outflows = len(rhs_ode.name_modeled_Outflows)
    n_controlled_Inflows = len(rhs_ode.name_controlled_Inflows)
    n_controlled_Outflows = len(rhs_ode.name_controlled_Outflows)

    # If no scales provided, fall back to unit scales so the wrapper constructor
    # (which validates shapes) still accepts the module.
    if not scale_kwargs:
        scale_kwargs = _default_scale_kwargs(
            n_RMCs=n_RMCs,
            n_rates=n_rates,
            n_modeled_Inflows=n_modeled_Inflows,
            n_modeled_Outflows=n_modeled_Outflows,
            n_controlled_Inflows=n_controlled_Inflows,
            n_controlled_Outflows=n_controlled_Outflows,
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
    training_parent_collection: BioProcessCollection,
) -> UserLossModule:
    """Default train hook for loss-module construction (per-target MSE)."""
    del process_names, config, seed, training_parent_collection
    return DefaultLossModule(target_names=list(target_names))


def _default_scale_kwargs(
    *,
    n_RMCs: int,
    n_rates: int,
    n_modeled_Inflows: int,
    n_modeled_Outflows: int,
    n_controlled_Inflows: int,
    n_controlled_Outflows: int,
    rhs_ode: RhsOde,
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
        "SCALE_modeled_Inflows_cumulative": LinearScaler(
            jnp.ones(n_modeled_Inflows, dtype=jnp.float64)
        ),
        "SCALE_modeled_Outflows_cumulative": LinearScaler(
            jnp.ones(n_modeled_Outflows, dtype=jnp.float64)
        ),
        "SCALE_controlled_Inflows_cumulative": LinearScaler(
            jnp.ones(n_controlled_Inflows, dtype=jnp.float64)
        ),
        "SCALE_controlled_Inflows_rates": LinearScaler(
            jnp.ones(n_controlled_Inflows, dtype=jnp.float64)
        ),
        "SCALE_controlled_Inflows_Cin": LinearScaler(
            jnp.ones((n_controlled_Inflows, n_RMCs), dtype=jnp.float64)
        ),
        "SCALE_controlled_Outflows_cumulative": LinearScaler(
            jnp.ones(n_controlled_Outflows, dtype=jnp.float64)
        ),
        "SCALE_controlled_Outflows_rates": LinearScaler(
            jnp.ones(n_controlled_Outflows, dtype=jnp.float64)
        ),
        "SCALE_controlled_PVs": LinearScaler(
            jnp.ones(len(rhs_ode.name_controlled_PVs), dtype=jnp.float64)
        ),
        "SCALE_modeled_Inflows_Cin": LinearScaler(
            jnp.ones((n_modeled_Inflows, n_RMCs), dtype=jnp.float64)
        ),
        "SCALE_modeled_BiologicalOde_rates": LinearScaler(
            jnp.ones(n_rates, dtype=jnp.float64)
        ),
        "SCALE_modeled_Inflows_rates": LinearScaler(
            jnp.ones(n_modeled_Inflows, dtype=jnp.float64)
        ),
        "SCALE_modeled_Outflows_rates": LinearScaler(
            jnp.ones(n_modeled_Outflows, dtype=jnp.float64)
        ),
        "SCALE_latent": LinearScaler(jnp.zeros(0, dtype=jnp.float64)),
    }
