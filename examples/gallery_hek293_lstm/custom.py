"""Use a native LSTM with z-scored inputs and MATLAB's max-normalized loss.

Hybrax discovers the three hooks below through ``custom_py`` in the config.
At model setup they construct fixed scalers, the reaction module and the loss.
Concentrations and volume are z-scored by the framework; time is z-scored in
input_features. All means and sample SDs use all nine cultures, matching MATLAB's
choice of cultures, but with the correct (value - mean) / SD calculation.

Loss normalization is separate: physical residuals / all-nine concentration
maxima. The converter also independently max-normalizes reacted amounts for PCA.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from hybrax.train import (
    AffineScaler,
    DefaultLossModule,
    DefaultLstmReactionModule,
    EstimatedScales,
    ReactionOutputs,
    frozen_field,
)


class InputLstm(DefaultLstmReactionModule):
    """Native four-unit cell with 25 concentrations, volume and time as inputs."""

    time_center: jax.Array = frozen_field()
    time_spread: jax.Array = frozen_field()

    def __init__(self, *, key, config, **scales):
        super().__init__(key=key, hidden_width=4, input_size=27, **scales)
        # Retain the default head's hidden-state columns and initial biases,
        # but omit its direct concentration, volume and time inputs.
        self.rate_head = eqx.tree_at(
            lambda head: (head.weight, head.bias),
            eqx.nn.Linear(4, 4, key=key),
            (self.rate_head.weight[:, :4], self.rate_head.bias),
        )
        self.time_center = jnp.asarray(config.custom.feature_center[-1])
        self.time_spread = jnp.asarray(config.custom.feature_spread[-1])

    def input_features(self, t, inputs):
        """Concentrations and volume arrive z-scored; standardize time here."""
        return jnp.concatenate(
            [
                inputs.SCL_modeled_RMCs,
                jnp.atleast_1d(inputs.SCL_modeled_V),
                jnp.atleast_1d((t - self.time_center) / self.time_spread),
            ]
        )

    def __call__(self, t, inputs):
        features = self.input_features(t, inputs)
        h, c = jnp.split(inputs.SCL_latent, 2)
        h_new, c_new = self.lstm_cell(features, (h, c))
        h_diff = h_new - h
        c_diff = c_new - c
        rates = self.rate_head(h)
        return ReactionOutputs(
            SCL_modeled_ReactionOde_rates=rates,
            SCL_modeled_Inflows_rates=jnp.zeros(0),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            SCL_latent_derivative=jnp.concatenate([h_diff, c_diff]),
        )


def build_reaction_module(*, seed, config, **kwargs):
    """Construct the model, then shrink its three weight matrices by 0.1.

    Biases and preprocessing statistics remain unchanged. The LSTM cell uses
    Hybrax's native initialization before this adjustment.
    """
    scales = {k: v for k, v in kwargs.items() if k.startswith("SCALE_")}
    module = InputLstm(key=jax.random.key(seed), config=config, **scales)
    return eqx.tree_at(
        lambda m: (
            m.lstm_cell.weight_ih,
            m.lstm_cell.weight_hh,
            m.rate_head.weight,
        ),
        module,
        replace_fn=lambda weight: 0.1 * weight,
    )


def estimate_all_scales(runtime_data, target_names, config):
    """Supply fixed physical-to-solver transformations during model setup.

    runtime_data contains training parents, but this benchmark deliberately
    uses the all-nine statistics saved by the converter, including held-out
    cultures. Best practice would be to fit preprocessing on training cultures only.

    AffineScaler computes SCL = (physical - mean) / sample_SD. Hybrax also
    uses these coordinates for ODE integration and restores physical values
    before evaluating the balances. Derivatives are divided by SD without
    subtracting the mean. Predictions are converted back to physical units.

    Bare arrays mean division. Ones retain physical units; empty arrays mean
    that no variables exist on that axis. Their shapes define module widths.
    Scales remain fixed throughout training and prediction.
    """
    del target_names
    rhs = runtime_data.rhs_ode
    order = [config.custom.state_names.index(n) for n in rhs.name_modeled_RMCs]
    center = jnp.asarray(config.custom.feature_center)
    spread = jnp.asarray(config.custom.feature_spread)
    empty = jnp.zeros(0)
    return EstimatedScales(
        # Reorder concentration statistics to the ODE's species order.
        SCALE_modeled_RMCs=AffineScaler(
            spread[jnp.array(order)], center[jnp.array(order)]
        ),
        # This axis scales total reactor volume.
        SCALE_V_in_cumulative=AffineScaler(spread[25], center[25]),
        # Four learned PCA-mode rates. Unit scales mean the rate head's
        # outputs need no further unit conversion before entering X * M * r.
        SCALE_modeled_ReactionOde_rates=jnp.ones(4),
        # No ANN-predicted inflows: neither cumulative volumes nor flow rates.
        SCALE_modeled_Inflows_cumulative=empty,
        SCALE_modeled_Inflows_rates=empty,
        # No ANN-predicted outflows either.
        SCALE_modeled_Outflows_cumulative=empty,
        SCALE_modeled_Outflows_rates=empty,
        # One known (controlled) feed. Keep cumulative volume in L and feed
        # rate in L/h. These signals drive the physical feed/dilution terms;
        # this example's 27 ANN features do not include them explicitly.
        SCALE_controlled_Inflows_cumulative=jnp.ones(1),
        SCALE_controlled_Inflows_rates=jnp.ones(1),
        # No controlled outflow signals.
        SCALE_controlled_Outflows_cumulative=empty,
        SCALE_controlled_Outflows_rates=empty,
        # No extra controlled process variables, such as pH or temperature.
        SCALE_controlled_PVs=empty,
        # Known feed-composition matrix: retain each component's concentration
        # units. This is distinct from scaling reactor concentrations above.
        SCALE_controlled_Inflows_Cin=jnp.ones_like(rhs.Cin_controlled_Inflows),
        # No predicted inflows, so their composition matrix has no feed entries.
        SCALE_modeled_Inflows_Cin=jnp.ones_like(rhs.Cin_modeled_Inflows),
    )


class MatlabLoss(DefaultLossModule):
    """Convert z-score residuals to MATLAB's max-normalized residuals.

    The mean cancels in predicted minus measured z-scores. Multiplying by
    sample_SD / all_nine_max therefore gives the physical residual divided by
    all_nine_max. The default reduction squares and averages these residuals.
    """

    ratio: jax.Array = frozen_field()

    def __init__(self, target_names, ratio):
        super().__init__(target_names=target_names)
        self.ratio = ratio

    def residual_reduction(self, residual, mask):
        # t=0 prescribes the initial condition. Exclude it from the loss while
        # retaining later measurements that happen to have concentration zero.
        return super().residual_reduction(residual * self.ratio, mask.at[0].set(False))


def build_loss_module(*, target_names, config, **kwargs):
    """Loss-construction hook, called by Hybrax during model setup."""
    # Ratios are stored in source-species order, so check the target order.
    assert list(target_names) == config.custom.state_names
    ratio = jnp.asarray(config.custom.feature_spread[:25]) / jnp.asarray(
        config.custom.loss_maxima
    )
    return MatlabLoss(target_names, ratio)
