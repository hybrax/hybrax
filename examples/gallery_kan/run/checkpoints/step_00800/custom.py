"""A Kolmogorov-Arnold Network (KAN) reaction module.

Every edge between an input and a hidden or output node carries its own
learnable univariate function (a SiLU base term plus a small Gaussian
radial-basis expansion); a node's output sums over its incoming edges. Two
stacked layers, each genuinely KAN-shaped, not a relabeled MLP. Each of the
three output rates also gets one small multiplicative term, prod_a(h0) *
prod_b(h1), added on top of its sum: see the page for the architecture and
what it does and does not reproduce from SR-KAN.

``estimate_all_scales`` below is unchanged from Tutorial 4.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hybrax.train import (
    EstimatedScales,
    ReactionInputs,
    ReactionOutputs,
    UserReactionModule,
    frozen_field,
    trainable_field,
)


class KANLayer(eqx.Module):
    base_w: jax.Array = trainable_field()  # (out, in)
    spline_c: jax.Array = trainable_field()  # (out, in, grid)
    centers: jax.Array = frozen_field()  # (grid,)
    inv_h2: float = eqx.field(static=True)

    def __init__(self, in_dim, out_dim, grid, key, out_scale):
        kb, ks = jax.random.split(key)
        self.base_w = (
            out_scale * jax.random.normal(kb, (out_dim, in_dim)) / max(in_dim, 1) ** 0.5
        )
        self.spline_c = (
            out_scale
            * jax.random.normal(ks, (out_dim, in_dim, grid))
            / max(in_dim, 1) ** 0.5
        )
        self.centers = jnp.linspace(-2.0, 2.0, grid)
        spacing = 4.0 / max(grid - 1, 1)
        self.inv_h2 = 1.0 / (spacing * spacing)

    def __call__(self, x):
        xb = jnp.tanh(x)  # bound inputs onto the RBF grid
        rbf = jnp.exp(
            -self.inv_h2 * (xb[:, None] - self.centers[None, :]) ** 2
        )  # (in, grid)
        spline = jnp.einsum("oig,ig->o", self.spline_c, rbf)
        base = self.base_w @ jax.nn.silu(xb)
        return spline + base


class KANReactionModule(UserReactionModule):
    l1: KANLayer = trainable_field()
    l2: KANLayer = trainable_field()
    prod_a: KANLayer = trainable_field()
    prod_b: KANLayer = trainable_field()

    def __init__(self, *, key, hidden=8, grid=6, **scale_kwargs):
        super().__init__(**scale_kwargs)
        n_in = self.n_modeled_RMCs
        n_out = self.n_modeled_BiologicalOde_rates
        k1, k2, ka, kb = jax.random.split(key, 4)
        # Hidden layer full-scale; output layer near-zero so the ODE starts flat
        # yet both layers receive gradient at step 0 (avoids an all-zero cold start).
        self.l1 = KANLayer(n_in, hidden, grid, k1, out_scale=1.0)
        self.l2 = KANLayer(hidden, n_out, grid, k2, out_scale=0.0)
        # One small multiplicative term per rate, prod_a(h0) * prod_b(h1), added
        # on top of l2's sum. h0/h1 are the first two of l1's hidden units,
        # picked arbitrarily since none of l1's hidden units carry individual
        # meaning. Ordinary non-zero start (unlike l2): a product started at
        # zero can never move away from zero under gradient descent.
        self.prod_a = KANLayer(1, n_out, grid, ka, out_scale=1.0)
        self.prod_b = KANLayer(1, n_out, grid, kb, out_scale=1.0)

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        h = self.l1(inputs.SCL_modeled_RMCs)
        out = self.l2(h) + self.prod_a(h[0:1]) * self.prod_b(h[1:2])
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=out,
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            SCL_modeled_Inflows_rates=jnp.zeros(0),
        )


def build_reaction_module(*, seed, **kwargs):
    scale_kwargs = {k: v for k, v in kwargs.items() if k.startswith("SCALE_")}
    return KANReactionModule(key=jax.random.key(seed), **scale_kwargs)


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

    empty = jnp.zeros(0)
    return EstimatedScales(
        SCALE_modeled_RMCs=jnp.asarray([rmc_scale[n] for n in rhs.name_modeled_RMCs]),
        SCALE_modeled_BiologicalOde_rates=jnp.asarray(rate_scale),
        SCALE_V_in_cumulative=jnp.asarray(
            max(runtime_data.initial_volume(i) for i in range(n_processes))
        ),
        SCALE_modeled_Inflows_cumulative=empty,
        SCALE_modeled_Inflows_rates=empty,
        SCALE_modeled_Outflows_cumulative=empty,
        SCALE_modeled_Outflows_rates=empty,
        SCALE_controlled_Inflows_cumulative=empty,
        SCALE_controlled_Inflows_rates=empty,
        SCALE_controlled_Outflows_cumulative=empty,
        SCALE_controlled_Outflows_rates=empty,
        SCALE_controlled_PVs=empty,
        SCALE_controlled_Inflows_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_controlled_Inflows)), 1.0
        ),
        SCALE_modeled_Inflows_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_modeled_Inflows)), 1.0
        ),
    )
