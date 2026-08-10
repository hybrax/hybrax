"""Fed-batch: a reaction module that reads the feed and a controlled PV.

Unlike the batch tutorials, this process has a continuous feed, two boluses, a
controlled process variable (dissolved oxygen), and sampling events. None of
that changes how you write the reaction module in principle, but the module
now has real inputs beyond the state, and the scale hook has real controlled
axes to estimate.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bp_format.mechanistic import build_rhs_ode
from bp_train import (
    EstimatedScales,
    ReactionInputs,
    ReactionOutputs,
    UserReactionModule,
    trainable_field,
)


class FedBatchModule(UserReactionModule):
    """MLP over [modeled state | controlled feed rate | controlled PVs]."""

    mlp: eqx.nn.MLP = trainable_field()

    def __init__(self, *, key, **scale_kwargs):
        super().__init__(**scale_kwargs)
        n_in = self.n_modeled_RMCs + self.n_controlled_FVCs + self.n_controlled_PVs
        self.mlp = eqx.nn.MLP(
            in_size=n_in,
            out_size=self.n_modeled_BiologicalOde_rates,
            width_size=32, depth=3, activation=jax.nn.tanh, key=key,
        )

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        # The feed rate and DO are real biological inputs here, not just
        # transport bookkeeping: the model is allowed to respond to them.
        features = jnp.concatenate([
            inputs.SCL_modeled_RMCs,
            inputs.SCL_controlled_FVCs_rates,
            inputs.SCL_controlled_PVs,
        ])
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=self.mlp(features),
            SCL_modeled_FVCs_rates=jnp.zeros(0),   # no MODELED feeds here
        )


def build_reaction_module(*, seed, **kwargs):
    scale_kwargs = {k: v for k, v in kwargs.items() if k.startswith("SCALE_")}
    return FedBatchModule(key=jax.random.key(seed), **scale_kwargs)


def estimate_all_scales(collection, target_names, config, *, controls_store):
    """Note the 4th argument: declaring `controls_store` is what makes
    bp-train pass it. Needed here because (unlike the batch tutorials)
    there is a real controlled feed and a real controlled PV to scale."""
    del target_names, config
    processes = list(collection.processes.values())
    rhs = build_rhs_ode(processes[0])

    rmc_scale = {
        name: max(
            max(float(np.max(np.abs(np.asarray(
                p.reactor_medium.components[name].concentration.values, float))))
                for p in processes),
            1e-6,
        )
        for name in rhs.name_modeled_RMCs
    }

    def rate_scale_for(species):
        per_process = []
        for p in processes:
            c = p.reactor_medium.components[species].concentration
            X = p.reactor_medium.components["biomass"].concentration
            values = np.asarray(c.values, float)
            exposure = np.trapezoid(np.asarray(X.values, float),
                                    np.asarray(c.times, float))
            per_process.append(abs(values[-1] - values[0]) / max(exposure, 1e-9))
        return max(max(per_process), 1e-9)

    rate_scale = [rate_scale_for(name[2:]) for name in rhs.name_modeled_rates]

    # Controlled axes: sample the fitted control splines over each run and
    # take the per-axis max-abs. This is the same recipe as the state scales,
    # just evaluated through the controls store instead of read off the data.
    n_FVC = len(controls_store.name_controlled_FVCs)
    n_PV = len(controls_store.name_controlled_PVs)
    fvc_rate_samples, pv_samples = [], []
    for process_name, process in collection.processes.items():
        per_process = controls_store.get_controls(process_name)
        t_start, t_end = float(process.time_axis.start), float(process.time_axis.end)
        for t in np.linspace(t_start + 1e-3, t_end - 1e-3, 50):
            fvc_rate_samples.append(
                np.asarray(per_process.eval_controlled_FVCs_rates(float(t), None)))
            pv_samples.append(
                np.asarray(per_process.eval_controlled_PVs(float(t), None)))

    def axis_scale(samples, n_axis):
        arr = np.stack(samples, axis=0) if samples else np.ones((1, n_axis))
        return np.maximum(np.max(np.abs(arr), axis=0), 1e-2)

    empty = jnp.zeros(0)
    return EstimatedScales(
        SCALE_modeled_RMCs=jnp.asarray([rmc_scale[n] for n in rhs.name_modeled_RMCs]),
        SCALE_modeled_BiologicalOde_rates=jnp.asarray(rate_scale),
        SCALE_V_in_cumulative=jnp.asarray(
            max(float(p.volume.initial_volume) for p in processes)),
        SCALE_modeled_FVCs_cumulative=empty,
        SCALE_modeled_FVCs_rates=empty,
        SCALE_controlled_FVCs_cumulative=jnp.ones(n_FVC),
        SCALE_controlled_FVCs_rates=jnp.asarray(axis_scale(fvc_rate_samples, n_FVC)),
        SCALE_controlled_PVs=jnp.asarray(axis_scale(pv_samples, n_PV)),
        SCALE_controlled_FVCs_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_controlled_FVCs)), 1.0),
        SCALE_modeled_FVCs_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_modeled_FVCs)), 1.0),
    )
