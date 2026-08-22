"""Minimal FBA-Hyb reaction module on e_coli_core.

Two small MLPs predict (qG, n_X, n_M, n_A) from state; a frozen, pole-free
rational surrogate (fit offline against 10,000 real pFBA solves on
e_coli_core.xml, Orth/Fleming/Palsson 2010, via the method from Gotsmy &
Guillen-Gosalbez's FBA-Hyb, bioRxiv:10.64898/2026.04.22.720062v1, see
01_generate_fba_data.py and 02_fit_surrogate.py) converts these into
biomass/glucose/acetate specific rates. n_S (succinate weight) is fixed at 0:
this page has no deliberate product, only real E. coli overflow metabolism
(growth vs. maintenance vs. acetate secretion). See knowledge_transfer.md's
sibling page, pls_dfba.md, for a product-forming, media-blend-aware version.
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
    trainable_field,
)

AVG_QG = 10.250012796042299
AVG_N = jnp.array([1.00000008, 1.00000023, 1.00000072, 0.99999969])
MW = jnp.array([180.156, 60.05])  # glucose, acetate (g/mol)


def _pos(B):
    return 0.5 * (B + jnp.sqrt(B * B + 1.5))


def surrogate_fba(x):
    """SCALED [qG, n_X, n_M, n_A, n_S] -> RAW [q_glc, qX, qM, qA], mmol/(gX.h).

    Validation R2 >= 0.999 on all four fitted fluxes; boundedness certificate
    passed (max overshoot 1.7x over the sampling box, min denominator
    0.199 > 0, pole-free).
    """
    qG = x[0] / AVG_QG
    n = x[1:5] / AVG_N
    n_X, n_M, n_A, n_S = n[0], n[1], n[2], n[3]
    q_glc = -AVG_QG * qG
    qX = qG * (-40.05086*n_X -0.011743758*n_M +0.014346741*n_A -0.0052064408*n_S +0.0082783369) * (-49.319124*n_X -0.41473128*n_M +2.1157595*n_A -1.6528906*n_S -3.5412292) / ((_pos(24.688972*n_X +0.20074873*n_M -1.0261536*n_A +0.79800013*n_S +1.7869275) + 0.05) * (_pos(85.331771*n_X +0.48121828*n_M +1.7234505*n_A +4.5642331*n_S -0.46385535) + 0.05))
    qM = qG * (-30.549273*n_X -0.26807454*n_M +0.60650343*n_A -1.143952*n_S -0.92775662) * (-0.037824693*n_X -24.825578*n_M +0.027984563*n_A +0.0020666315*n_S -0.007740588) / ((_pos(17.512874*n_X +0.16739751*n_M -0.27899872*n_A +0.67477612*n_S +0.76306517) + 0.05) * (_pos(46.415016*n_X +0.22525384*n_M +0.68795142*n_A +2.3243431*n_S -0.91656303) + 0.05))
    qA = qG * (0.11560359*n_X +0.025412058*n_M +24.333516*n_A +0.037186754*n_S -0.094942148) * (20.777323*n_X -0.40676165*n_M +3.1964066*n_A +0.9569548*n_S -0.19868886) / ((_pos(41.246274*n_X -0.34994074*n_M +4.3166512*n_A +2.1503221*n_S -1.9194144) + 0.05) * (_pos(13.35614*n_X -0.041404912*n_M +0.72777261*n_A +0.64795735*n_S +0.29150121) + 0.05))
    return jnp.array([q_glc, qX, qM, qA])


def _bounded_softplus(x, alpha):
    return jnp.tanh(jax.nn.softplus(x) / alpha) * alpha


def _init_xavier_zero_bias(model, key):
    def init_layer(layer):
        if isinstance(layer, eqx.nn.Linear):
            wkey, _ = jax.random.split(key)
            new_w = jax.nn.initializers.glorot_normal()(wkey, layer.weight.shape)
            layer = eqx.tree_at(lambda l: l.weight, layer, new_w)
            if layer.bias is not None:
                layer = eqx.tree_at(lambda l: l.bias, layer, jnp.zeros_like(layer.bias))
        return layer
    return jax.tree_util.tree_map(init_layer, model, is_leaf=lambda x: isinstance(x, eqx.nn.Linear))


class FBAHybReactionModule(UserReactionModule):
    ann_qG: eqx.nn.MLP = trainable_field()
    ann_obj: eqx.nn.MLP = trainable_field()

    def __init__(self, *, key, **scale_kwargs):
        super().__init__(**scale_kwargs)
        n_in = self.n_modeled_RMCs
        k_qG, k_obj = jax.random.split(key, 2)
        self.ann_qG = _init_xavier_zero_bias(
            eqx.nn.MLP(in_size=n_in, out_size=1, width_size=8, depth=2,
                       activation=jax.nn.softplus, key=k_qG), k_qG)
        self.ann_obj = _init_xavier_zero_bias(
            eqx.nn.MLP(in_size=n_in, out_size=3, width_size=16, depth=2,
                       activation=jax.nn.softplus, key=k_obj), k_obj)

    def __call__(self, t, inputs: ReactionInputs) -> ReactionOutputs:
        del t
        SCL_features = inputs.SCL_modeled_RMCs

        qG = _bounded_softplus(self.ann_qG(SCL_features)[0], 18.0)
        obj = self.ann_obj(SCL_features)
        n_X = _bounded_softplus(obj[0], 1.8)
        n_M = _bounded_softplus(obj[1], 1.8)
        n_A = _bounded_softplus(obj[2], 1.8)

        fba_out = surrogate_fba(jnp.array([qG, n_X, n_M, n_A, 0.0]))
        q_glc, qX, qM, qA = fba_out[0], fba_out[1], fba_out[2], fba_out[3]

        RAW_glc = q_glc * MW[0] / 1000.0
        RAW_ace = qA * MW[1] / 1000.0
        RAW_rates = jnp.array([qX, RAW_glc, RAW_ace])

        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=self.scale_modeled_BiologicalOde_rates(RAW_rates),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            SCL_modeled_Inflows_rates=jnp.zeros(0),
            auxiliary={"n_weights": jnp.array([n_X, n_M, n_A])},
        )


def build_reaction_module(*, seed, **kwargs):
    scale_kwargs = {k: v for k, v in kwargs.items() if k.startswith("SCALE_")}
    return FBAHybReactionModule(key=jax.random.key(seed), **scale_kwargs)


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
            max(runtime_data.initial_volume(i) for i in range(n_processes))),
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
            jnp.abs(jnp.asarray(rhs.Cin_controlled_Inflows)), 1.0),
        SCALE_modeled_Inflows_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_modeled_Inflows)), 1.0),
    )
