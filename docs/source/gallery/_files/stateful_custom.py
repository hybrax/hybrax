"""A stateful reaction module: a continuous-time LSTM.

Every reaction module so far has been memoryless: its rates depend only on
the CURRENT state. A stateful module (``n_latent > 0``) adds its own hidden
state, integrated as extra ODE dimensions alongside the physical ones, so the
rates can depend on the process's recent history too.

bp-train's own default stateful module uses a GRU cell (see
``DefaultStatefulReactionModule`` in ``bp_train/defaults.py``); this one uses
an LSTM cell instead, to show the same trick applies to any recurrent cell
with a fixed-size hidden state.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bp_train import (
    EstimatedScales,
    ReactionInputs,
    ReactionOutputs,
    UserReactionModule,
    trainable_field,
)


class LSTMReactionModule(UserReactionModule):
    """Cell state and hidden state are both integrated ODE latents, each
    pulled toward the LSTM cell's discrete target at every solver step."""

    lstm_cell: eqx.nn.LSTMCell = trainable_field()
    rate_head: eqx.nn.Linear = trainable_field()
    n_hidden: int = eqx.field(static=True)

    def __init__(self, *, key, n_hidden=4, **scale_kwargs):
        # SCL_latent holds [hidden | cell], so its width is 2 * n_hidden.
        scale_kwargs = {**scale_kwargs, "SCALE_latent": jnp.ones(2 * n_hidden)}
        super().__init__(**scale_kwargs)
        self.n_hidden = n_hidden
        key_cell, key_head = jax.random.split(key)
        self.lstm_cell = eqx.nn.LSTMCell(
            input_size=self.n_modeled_RMCs, hidden_size=n_hidden, key=key_cell)
        self.rate_head = eqx.nn.Linear(
            in_features=n_hidden + self.n_modeled_RMCs,
            out_features=self.n_modeled_BiologicalOde_rates,
            key=key_head,
        )

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        h, c = jnp.split(inputs.SCL_latent, 2)
        h_new, c_new = self.lstm_cell(inputs.SCL_modeled_RMCs, (h, c))
        # The continuous-time-RNN trick: an ODE derivative that pulls the
        # latent toward the cell's one-step target, rather than the discrete
        # jump an LSTM normally takes. At convergence h tracks h_new.
        latent_derivative = jnp.concatenate([h_new, c_new]) - inputs.SCL_latent
        # Read out from h (not h_new): the CURRENT hidden state, consistent
        # with every other input the reaction module receives at time t.
        readout = jnp.concatenate([h, inputs.SCL_modeled_RMCs])
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=self.rate_head(readout),
            SCL_modeled_FVCs_rates=jnp.zeros(0),
            SCL_latent_derivative=latent_derivative,
        )


def build_reaction_module(*, seed, **kwargs):
    scale_kwargs = {k: v for k, v in kwargs.items() if k.startswith("SCALE_")}
    return LSTMReactionModule(key=jax.random.key(seed), **scale_kwargs)


def estimate_all_scales(runtime_data, target_names, config):
    del target_names, config
    rhs = runtime_data.rhs_ode
    n_processes = len(runtime_data.process_order)

    def max_abs_state(name):
        best = 0.0
        for i in range(n_processes):
            _, values = runtime_data.raw_state_trace(i, name)
            if values.size:
                best = max(best, float(np.max(np.abs(values))))
        return max(best, 1e-6)

    rmc_scale = {name: max_abs_state(name) for name in rhs.name_modeled_RMCs}

    def rate_scale_for(species):
        per_process = []
        for i in range(n_processes):
            c_times, c_values = runtime_data.raw_state_trace(i, species)
            x_times, x_values = runtime_data.raw_state_trace(i, "biomass")
            exposure = np.trapezoid(x_values, x_times)
            per_process.append(abs(c_values[-1] - c_values[0]) / max(exposure, 1e-9))
        return max(max(per_process), 1e-9)

    rate_scale = [rate_scale_for(name[2:]) for name in rhs.name_modeled_rates]

    # Note: no SCALE_latent here. LSTMReactionModule.__init__ sizes and sets
    # it itself from n_hidden, since the latent has no physical counterpart
    # in the data for this hook to estimate a magnitude from.
    empty = jnp.zeros(0)
    return EstimatedScales(
        SCALE_modeled_RMCs=jnp.asarray([rmc_scale[n] for n in rhs.name_modeled_RMCs]),
        SCALE_modeled_BiologicalOde_rates=jnp.asarray(rate_scale),
        SCALE_V_in_cumulative=jnp.asarray(
            max(runtime_data.initial_volume(i) for i in range(n_processes))),
        SCALE_modeled_FVCs_cumulative=empty,
        SCALE_modeled_FVCs_rates=empty,
        SCALE_controlled_FVCs_cumulative=empty,
        SCALE_controlled_FVCs_rates=empty,
        SCALE_controlled_PVs=empty,
        SCALE_controlled_FVCs_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_controlled_FVCs)), 1.0),
        SCALE_modeled_FVCs_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_modeled_FVCs)), 1.0),
    )
