"""PLS-dFBA-inspired reaction module on e_coli_core.

Builds on fba_hyb.md, but replaces its MLP with an actual PLS-shaped
component: a linear, low-rank latent-variable regression (predictors -> a
handful of latent components -> outputs), which is PLS's real structural
form: no nonlinearity anywhere in the regression itself.
Trained here by ordinary gradient descent through the whole ODE trajectory
rather than NIPALS (the actual algorithm real PLS is fit with): the one
disclosed algorithmic difference from a textbook PLS fit.

Also adds a controlled process variable, `media_blend_fraction`, attached via
`transform_process_collection`. The PLS component takes the blend fraction as
an extra predictor alongside state, so the predicted FBA objective weights,
and hence the predicted rates, become a function of both physiology AND
recipe: Negahban et al. 2026's real structural idea (a kinetic corridor that
depends on media composition).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import hybrax.format as hxf
from hybrax.format.time_series import TimeSeries
from hybrax.train import (
    EstimatedScales,
    ReactionInputs,
    ReactionOutputs,
    UserReactionModule,
    trainable_field,
)

AVG_QG = 10.250012796042299
AVG_N = jnp.array([1.00000008, 1.00000023, 1.00000072, 0.99999969])
MW = jnp.array([180.156, 60.05, 118.09])  # glucose, acetate, succinate (g/mol)

BLEND_BY_PROCESS = {
    "blend_00": 0.0,
    "blend_33": 0.33,
    "blend_67": 0.67,
    "blend_100": 1.0,
}


def transform_process_collection(collection, config):
    del config
    for name, process in collection.processes.items():
        blend = BLEND_BY_PROCESS[name]
        times = process.reactor_medium.components["biomass"].concentration.times
        process.process_variables["media_blend_fraction"] = hxf.ProcessVariable(
            name="media_blend_fraction",
            unit="-",
            is_controlled=True,
            values=TimeSeries(times=times, values=np.full(times.shape, blend)),
            bounds=(0.0, 1.0),
        )
    return collection


def _pos(B):
    return 0.5 * (B + jnp.sqrt(B * B + 1.5))


def surrogate_fba(x):
    """SCALED [qG, n_X, n_M, n_A, n_S] -> RAW [q_glc, qX, qM, qA, qS],
    mmol/(gX.h). Identical fit to fba_hyb.md's surrogate; this page uses all
    five outputs (n_S/qS included), where fba_hyb.md drops them.
    """
    qG = x[0] / AVG_QG
    n = x[1:5] / AVG_N
    n_X, n_M, n_A, n_S = n[0], n[1], n[2], n[3]
    q_glc = -AVG_QG * qG
    qX = (
        qG
        * (
            -40.05086 * n_X
            - 0.011743758 * n_M
            + 0.014346741 * n_A
            - 0.0052064408 * n_S
            + 0.0082783369
        )
        * (
            -49.319124 * n_X
            - 0.41473128 * n_M
            + 2.1157595 * n_A
            - 1.6528906 * n_S
            - 3.5412292
        )
        / (
            (
                _pos(
                    24.688972 * n_X
                    + 0.20074873 * n_M
                    - 1.0261536 * n_A
                    + 0.79800013 * n_S
                    + 1.7869275
                )
                + 0.05
            )
            * (
                _pos(
                    85.331771 * n_X
                    + 0.48121828 * n_M
                    + 1.7234505 * n_A
                    + 4.5642331 * n_S
                    - 0.46385535
                )
                + 0.05
            )
        )
    )
    qM = (
        qG
        * (
            -30.549273 * n_X
            - 0.26807454 * n_M
            + 0.60650343 * n_A
            - 1.143952 * n_S
            - 0.92775662
        )
        * (
            -0.037824693 * n_X
            - 24.825578 * n_M
            + 0.027984563 * n_A
            + 0.0020666315 * n_S
            - 0.007740588
        )
        / (
            (
                _pos(
                    17.512874 * n_X
                    + 0.16739751 * n_M
                    - 0.27899872 * n_A
                    + 0.67477612 * n_S
                    + 0.76306517
                )
                + 0.05
            )
            * (
                _pos(
                    46.415016 * n_X
                    + 0.22525384 * n_M
                    + 0.68795142 * n_A
                    + 2.3243431 * n_S
                    - 0.91656303
                )
                + 0.05
            )
        )
    )
    qA = (
        qG
        * (
            0.11560359 * n_X
            + 0.025412058 * n_M
            + 24.333516 * n_A
            + 0.037186754 * n_S
            - 0.094942148
        )
        * (
            20.777323 * n_X
            - 0.40676165 * n_M
            + 3.1964066 * n_A
            + 0.9569548 * n_S
            - 0.19868886
        )
        / (
            (
                _pos(
                    41.246274 * n_X
                    - 0.34994074 * n_M
                    + 4.3166512 * n_A
                    + 2.1503221 * n_S
                    - 1.9194144
                )
                + 0.05
            )
            * (
                _pos(
                    13.35614 * n_X
                    - 0.041404912 * n_M
                    + 0.72777261 * n_A
                    + 0.64795735 * n_S
                    + 0.29150121
                )
                + 0.05
            )
        )
    )
    qS = (
        qG
        * (
            -0.028994836 * n_X
            + 0.00034001442 * n_M
            + 0.018303022 * n_A
            - 17.239703 * n_S
            - 0.0048146897
        )
        * (
            -55.150953 * n_X
            - 0.50212877 * n_M
            + 1.331242 * n_A
            - 1.9245035 * n_S
            - 1.8896264
        )
        / (
            (
                _pos(
                    44.312184 * n_X
                    + 0.20545498 * n_M
                    + 0.56363208 * n_A
                    + 2.2376853 * n_S
                    - 0.70094854
                )
                + 0.05
            )
            * (
                _pos(
                    22.984164 * n_X
                    + 0.22825551 * n_M
                    - 0.41448157 * n_A
                    + 0.81603398 * n_S
                    + 1.0517311
                )
                + 0.05
            )
        )
    )
    return jnp.array([q_glc, qX, qM, qA, qS])


def _bounded_softplus(x, alpha):
    return jnp.tanh(jax.nn.softplus(x) / alpha) * alpha


class PLSComponent(eqx.Module):
    """Linear, low-rank latent-variable regression: predictors -> a handful
    of latent components -> outputs. This is PLS's actual structural form
    (no nonlinearity anywhere), standing in for Negahban et al. 2026's own
    piecewise-nonlinear PLS regression, whose piecewise refinement is the
    one part not reproduced here. `n_components` << `n_in` is the whole
    point of PLS: compress collinear predictors (state + media composition)
    into a handful of components before regressing onto the outputs.
    """

    W: jax.Array = trainable_field()  # (n_in, n_components) predictor loadings
    Q: jax.Array = trainable_field()  # (n_components, n_out) response loadings
    b: jax.Array = trainable_field()  # (n_out,) intercept

    def __init__(self, *, n_in, n_out, n_components, key):
        k1, k2 = jax.random.split(key)
        scale = 1.0 / jnp.sqrt(n_in)
        self.W = jax.random.normal(k1, (n_in, n_components)) * scale
        self.Q = jax.random.normal(k2, (n_components, n_out)) * scale
        self.b = jnp.zeros(n_out)

    def scores(self, x):
        return x @ self.W

    def __call__(self, x):
        return self.scores(x) @ self.Q + self.b


class PLSdFBAReactionModule(UserReactionModule):
    pls: PLSComponent = trainable_field()

    def __init__(self, *, key, n_components=3, **scale_kwargs):
        super().__init__(**scale_kwargs)
        n_in = self.n_modeled_RMCs + self.n_controlled_PVs
        self.pls = PLSComponent(n_in=n_in, n_out=5, n_components=n_components, key=key)

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        SCL_features = jnp.concatenate(
            [inputs.SCL_modeled_RMCs, inputs.SCL_controlled_PVs]
        )

        raw = self.pls(SCL_features)
        qG = _bounded_softplus(raw[0], 18.0)
        n_X = _bounded_softplus(raw[1], 1.8)
        n_M = _bounded_softplus(raw[2], 1.8)
        n_A = _bounded_softplus(raw[3], 1.8)
        n_S = _bounded_softplus(raw[4], 1.8)

        fba_out = surrogate_fba(jnp.array([qG, n_X, n_M, n_A, n_S]))
        q_glc, qX, qM, qA, qS = (
            fba_out[0],
            fba_out[1],
            fba_out[2],
            fba_out[3],
            fba_out[4],
        )

        RAW_glc = q_glc * MW[0] / 1000.0
        RAW_ace = qA * MW[1] / 1000.0
        RAW_suc = qS * MW[2] / 1000.0
        RAW_rates = jnp.array([qX, RAW_glc, RAW_ace, RAW_suc])

        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=self.scale_modeled_BiologicalOde_rates(
                RAW_rates
            ),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            SCL_modeled_Inflows_rates=jnp.zeros(0),
            auxiliary={
                "n_weights": jnp.array([n_X, n_M, n_A, n_S]),
                "latent_scores": self.pls.scores(SCL_features),
            },
        )


def build_reaction_module(*, seed, **kwargs):
    scale_kwargs = {k: v for k, v in kwargs.items() if k.startswith("SCALE_")}
    return PLSdFBAReactionModule(key=jax.random.key(seed), **scale_kwargs)


def estimate_all_scales(runtime_data, target_names, config):
    del target_names, config
    rhs = runtime_data.rhs_ode
    controls_store = runtime_data.controls_store
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

    n_PV = len(controls_store.name_controlled_PVs)
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
        SCALE_controlled_PVs=jnp.ones(n_PV),
        SCALE_controlled_Inflows_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_controlled_Inflows)), 1.0
        ),
        SCALE_modeled_Inflows_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_modeled_Inflows)), 1.0
        ),
    )
