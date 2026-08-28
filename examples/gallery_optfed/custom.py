"""OptFed rate law: non-competitive-inhibition Michaelis-Menten kinetics with
an Eyring-equation temperature dependence.

Reduced to a smaller inhibition/activation variable set than the paper's own
Eq. 4a-4c: see the gallery page for exactly what is reproduced and what is
disclosed as reduced.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hybrax.format.mechanistic import build_rhs_ode
from hybrax.train import (
    EstimatedScales,
    ReactionOutputs,
    RateModule,
    frozen_field,
    trainable_field,
)


def _eyring_rate_ceiling(T_K, log_A, log_Ea_R, raw_Teq, log_dHeq_R):
    """Algebraically reparameterized Eq. 4e, vectorized over parameter arrays."""
    A = jnp.exp(log_A)
    Ea_R = jnp.exp(log_Ea_R)
    Teq = 290.0 + 40.0 * jax.nn.sigmoid(raw_Teq)
    dHeq_R = jnp.exp(log_dHeq_R)
    return (
        A
        * T_K
        * jnp.exp(-Ea_R / T_K)
        / (1.0 + jnp.exp(dHeq_R * (1.0 / Teq - 1.0 / T_K)))
    )


def _inhibition_product(values, log_K):
    K = jnp.exp(log_K)
    return jnp.prod(1.0 / (1.0 + jnp.clip(values, 0.0, None) / K))


def _activation_product(values, log_K):
    K = jnp.exp(log_K)
    return jnp.prod(1.0 + jnp.clip(values, 0.0, None) / K)


class OptFedModule(RateModule):
    """Eyring axis order: [0]=gamma_deg (uptake), [1]=gamma_pi (production),
    [2]=gamma_alpha (maintenance)."""

    eyring_log_A: jax.Array = trainable_field()
    eyring_log_Ea_R: jax.Array = trainable_field()
    eyring_raw_Teq: jax.Array = trainable_field()
    eyring_log_dHeq_R: jax.Array = trainable_field()
    log_Km: jax.Array = trainable_field()  # (2,) [Km_deg, Km_pi]
    log_K_inhib: jax.Array = (
        trainable_field()
    )  # (2, 2) [[deg_px, deg_x], [pi_px, pi_x]]
    log_K_activ: jax.Array = trainable_field()  # (2,) [a_deg, a_x]
    Y_XrG: jax.Array = frozen_field()
    Y_PG: jax.Array = frozen_field()

    i_biomass: int = eqx.field(static=True)
    i_glucose: int = eqx.field(static=True)
    i_product: int = eqx.field(static=True)

    def __init__(self, *, i_biomass, i_glucose, i_product, **scale_kwargs):
        super().__init__(**scale_kwargs)
        self.i_biomass, self.i_glucose, self.i_product = i_biomass, i_glucose, i_product
        # Deliberately mediocre starting guesses: the point is that they move.
        self.eyring_log_A = jnp.log(jnp.array([0.5, 0.5, 0.5]))
        self.eyring_log_Ea_R = jnp.log(jnp.array([1000.0, 1000.0, 1000.0]))
        self.eyring_raw_Teq = jnp.zeros(3)  # sigmoid(0)=0.5 -> Teq=310 K
        self.eyring_log_dHeq_R = jnp.log(jnp.array([10000.0, 10000.0, 10000.0]))
        self.log_Km = jnp.log(jnp.array([1.0, 1.0]))
        self.log_K_inhib = jnp.log(jnp.array([[1.0, 100.0], [1.0, 100.0]]))
        self.log_K_activ = jnp.log(jnp.array([1.0, 100.0]))
        self.Y_XrG = jnp.array(0.45)
        self.Y_PG = jnp.array(0.25)

    def __call__(self, t, inputs) -> ReactionOutputs:
        del t
        RAW = self.unscale_modeled_RMCs(inputs.SCL_modeled_RMCs)
        X = jnp.clip(RAW[self.i_biomass], 1e-6, None)
        G = jnp.clip(RAW[self.i_glucose], 0.0, None)
        P = jnp.clip(RAW[self.i_product], 0.0, None)
        T_K = self.unscale_controlled_PVs(inputs.SCL_controlled_PVs)[0] + 273.15
        px = P / X

        gdeg_max, gpi_max, galpha_min = _eyring_rate_ceiling(
            T_K,
            self.eyring_log_A,
            self.eyring_log_Ea_R,
            self.eyring_raw_Teq,
            self.eyring_log_dHeq_R,
        )

        Km_deg, Km_pi = jnp.exp(self.log_Km)
        gdeg = (
            gdeg_max
            * (G / (Km_deg + G))
            * _inhibition_product(jnp.array([px, X]), self.log_K_inhib[0])
        )

        galpha = galpha_min * _activation_product(
            jnp.array([gdeg, X]), self.log_K_activ
        )

        driver = jnp.clip(gdeg - galpha, 0.0, None)
        gpi = (
            gpi_max
            * (driver / (Km_pi + driver))
            * _inhibition_product(jnp.array([px, X]), self.log_K_inhib[1])
        )

        gmu = gdeg - gpi - galpha

        q_biomass = gmu * self.Y_XrG
        q_product = gpi * self.Y_PG
        q_glucose = gdeg

        RAW_rates = jnp.array([q_biomass, q_glucose, q_product])
        return ReactionOutputs(
            SCL_modeled_ReactionOde_rates=self.scale_modeled_ReactionOde_rates(
                RAW_rates
            ),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            SCL_modeled_Inflows_rates=jnp.zeros(0),
        )


def build_reaction_module(*, training_parent_collection, **kwargs):
    process = next(iter(training_parent_collection.processes.values()))
    rhs = build_rhs_ode(process)
    names = list(rhs.name_modeled_RMCs)
    return OptFedModule(
        i_biomass=names.index("biomass"),
        i_glucose=names.index("glucose"),
        i_product=names.index("product"),
        **{k: v for k, v in kwargs.items() if k.startswith("SCALE_")},
    )


def estimate_all_scales(runtime_data, target_names, config):
    del target_names, config
    rhs = runtime_data.rhs_ode
    store = runtime_data.controls_store
    n_processes = len(runtime_data.process_order)
    span = max(
        end - start
        for start, end in (runtime_data.time_bounds(i) for i in range(n_processes))
    )

    def max_abs_state(name):
        best = 0.0
        for i in range(n_processes):
            _, values = runtime_data.raw_state_trace(i, name)
            if values.size:
                best = max(best, float(np.max(np.abs(values))))
        return max(best, 1e-6)

    rmc_scale = {name: max_abs_state(name) for name in rhs.name_modeled_RMCs}
    biomass = rmc_scale["biomass"]

    def controlled_axis_scale(eval_name, n_axis):
        """maxabs over a dense time grid of every process, per controlled axis."""
        best = np.zeros(n_axis)
        for i in range(n_processes):
            per_process = store.get_controls(runtime_data.process_order[i])
            t0, t1 = runtime_data.time_bounds(i)
            evalfn = getattr(per_process, eval_name)
            for tt in np.linspace(t0 + 1e-3, t1 - 1e-3, 20):
                v = np.abs(np.asarray(evalfn(float(tt), None)))
                best = np.maximum(best, v)
        return np.maximum(best, 1e-6)

    n_fvc, n_pv = len(store.name_controlled_Inflows), len(store.name_controlled_PVs)
    empty = jnp.zeros(0)
    return EstimatedScales(
        SCALE_modeled_RMCs=jnp.asarray([rmc_scale[n] for n in rhs.name_modeled_RMCs]),
        SCALE_modeled_ReactionOde_rates=jnp.asarray(
            [rmc_scale[n[2:]] / (biomass * span) for n in rhs.name_modeled_rates]
        ),
        SCALE_V_in_cumulative=jnp.asarray(
            max(runtime_data.initial_volume(i) for i in range(n_processes))
        ),
        SCALE_modeled_Inflows_cumulative=empty,
        SCALE_modeled_Inflows_rates=empty,
        SCALE_modeled_Outflows_cumulative=empty,
        SCALE_modeled_Outflows_rates=empty,
        SCALE_controlled_Inflows_cumulative=jnp.asarray(
            controlled_axis_scale("eval_controlled_Inflows_cumulative", n_fvc)
        )
        if n_fvc
        else empty,
        SCALE_controlled_Inflows_rates=jnp.asarray(
            controlled_axis_scale("eval_controlled_Inflows_rates", n_fvc)
        )
        if n_fvc
        else empty,
        SCALE_controlled_Outflows_cumulative=empty,
        SCALE_controlled_Outflows_rates=empty,
        SCALE_controlled_PVs=jnp.asarray(
            controlled_axis_scale("eval_controlled_PVs", n_pv)
        )
        if n_pv
        else empty,
        SCALE_controlled_Inflows_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_controlled_Inflows)), 1.0
        ),
        SCALE_modeled_Inflows_Cin=jnp.maximum(
            jnp.abs(jnp.asarray(rhs.Cin_modeled_Inflows)), 1.0
        ),
    )
