"""Self-contained bp-train fixture (martens_2025_f, single process `run_1`).

Lives under tests/ so the end-to-end CLI round-trip test does not depend on
anything in examples/ (those are throwaway real-world applications). The
hooks are deliberately generic — no dataset-specific name validation — so the
fixture keeps working if the data evolves:

- a small plain-MLP reaction module (output scaled down so initial rates stay
  near zero and the integrator survives step 0),
- RMS / max-abs scale estimation across the collection,
- a ``DefaultLossModule`` subclass that adds one ``nonneg/<target>`` hinge per
  target, so the build_loss_module hook + named-loss plumbing are exercised
  through train -> checkpoint -> forward -> losses.csv.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bp_format.mechanistic import build_rhs_ode
from bp_train import (
    DefaultLossModule,
    EstimatedScales,
    LossInputs,
    LossOutputs,
    ReactionInputs,
    ReactionOutputs,
    UserReactionModule,
    trainable_field,
)
from bp_train.controls_store import ControlsStore


CONFIG = {
    # Bolus events become triangle ramps of this width (hours).
    "bolus_run_min_dt": 0.01,
}


# ---------------------------------------------------------------------------
# Reaction module: plain MLP -> all BiologicalOde + modeled-feed rates
# ---------------------------------------------------------------------------


class FixtureReactionModule(UserReactionModule):
    model: eqx.nn.MLP = trainable_field()
    output_scale: float = eqx.field(static=True)

    def __init__(self, *, key, output_scale=0.1, **scale_kwargs):
        super().__init__(**scale_kwargs)
        self.output_scale = float(output_scale)
        n_in = (
            self.n_modeled_RMCs
            + self.n_controlled_FVCs
            + self.n_controlled_FVCs_bolus
            + self.n_controlled_PVs
        )
        n_out = self.n_modeled_BiologicalOde_rates + self.n_modeled_FVCs
        self.model = eqx.nn.MLP(
            in_size=max(n_in, 1),
            out_size=n_out,
            width_size=16,
            depth=2,
            key=key,
        )

    def __call__(self, t, inputs: ReactionInputs):
        del t
        SCL_features = jnp.concatenate(
            [
                inputs.SCL_modeled_RMCs,
                inputs.SCL_controlled_FVCs_cumulative,
                inputs.SCL_controlled_FVCs_bolus_rates,
                inputs.SCL_controlled_PVs,
            ]
        )
        raw = self.model(SCL_features) * self.output_scale
        n_rates = self.n_modeled_BiologicalOde_rates
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=self.scale_modeled_BiologicalOde_rates(
                raw[:n_rates]
            ),
            SCL_modeled_FVCs_rates=self.scale_modeled_FVCs_rates(
                jax.nn.softplus(raw[n_rates:])
            ),
        )


def build_reaction_module(
    *, target_names, process_names, config, seed, collection, **scale_kwargs
):
    del target_names, process_names, config, collection
    return FixtureReactionModule(key=jax.random.key(seed), **scale_kwargs)


# ---------------------------------------------------------------------------
# estimate_all_scales (generic RMS / max-abs across the collection)
# ---------------------------------------------------------------------------


def estimate_all_scales(collection, target_names, config):
    del config
    n_species = len(target_names)
    ref_rhs_ode = build_rhs_ode(list(collection.processes.values())[0])
    controls_store = ControlsStore.from_collection(collection)

    n_FVC = len(controls_store.name_controlled_FVCs)
    n_SVC = len(controls_store.name_controlled_SVCs)
    n_PV = len(controls_store.name_controlled_PVs)
    n_extras = len(controls_store.name_extras)
    n_modeled_VCs = len(ref_rhs_ode.name_modeled_FVCs)
    n_rates = len(ref_rhs_ode.name_modeled_rates)

    # Modeled RMCs: RMS |c| pooled across processes per species.
    RAW_species_rms = np.zeros(n_species, dtype=float)
    for i, sp_name in enumerate(target_names):
        pooled = np.concatenate(
            [
                np.asarray(
                    p.reactor_medium.components[sp_name].concentration.values,
                    dtype=float,
                )
                for p in collection.processes.values()
            ]
        )
        RAW_species_rms[i] = float(np.sqrt(np.mean(pooled**2)))
    SCALE_modeled_RMCs = np.maximum(RAW_species_rms, 1e-6)

    RAW_volumes = [float(p.volume.initial_volume) for p in collection.processes.values()]
    SCALE_V_in_cumulative = float(max(np.max(RAW_volumes), 1e-6))

    SCALE_modeled_FVCs_cumulative = np.zeros(n_modeled_VCs, dtype=float)
    for k, flow_name in enumerate(ref_rhs_ode.name_modeled_FVCs):
        max_cum = 0.0
        for p in collection.processes.values():
            vc = p.volume.volume_changes[flow_name]
            max_cum = max(
                max_cum, float(np.max(np.abs(np.asarray(vc.values.values, dtype=float))))
            )
        SCALE_modeled_FVCs_cumulative[k] = max(max_cum, 1.0)

    RAW_u_canonical_samples: list[np.ndarray] = []
    RAW_u_rhs_samples: list[np.ndarray] = []
    for process_name, process in collection.processes.items():
        per_process = controls_store.get_controls(process_name)
        t_start = float(process.time_axis.start)
        t_end = float(process.time_axis.end)
        for t in np.linspace(t_start + 1e-3, t_end - 1e-3, 50):
            RAW_u_canonical_samples.append(np.asarray(per_process.eval(float(t))))
            RAW_u_rhs_samples.append(np.asarray(per_process.eval_u(float(t))))

    RAW_u_canonical_arr = (
        np.stack(RAW_u_canonical_samples, axis=0)
        if RAW_u_canonical_samples
        else np.ones((1, n_FVC + n_SVC + n_PV + n_extras))
    )
    RAW_u_rhs_arr = (
        np.stack(RAW_u_rhs_samples, axis=0)
        if RAW_u_rhs_samples
        else np.ones((1, n_FVC + n_SVC + n_PV))
    )
    canonical_max = np.maximum(np.max(np.abs(RAW_u_canonical_arr), axis=0), 1e-2)
    rhs_max = np.maximum(np.max(np.abs(RAW_u_rhs_arr), axis=0), 1e-2)

    n_bolus = n_extras - 1
    bolus_block_start = n_FVC + n_SVC + n_PV
    SCALE_controlled_FVCs_cumulative = canonical_max[:n_FVC]
    SCALE_controlled_FVCs_rates = rhs_max[:n_FVC]
    SCALE_controlled_PVs = canonical_max[n_FVC + n_SVC : n_FVC + n_SVC + n_PV]
    SCALE_controlled_FVCs_bolus_rates = canonical_max[
        bolus_block_start : bolus_block_start + n_bolus
    ]

    RAW_controlled_FVCs_Cin = np.asarray(ref_rhs_ode.Cin_controlled_FVCs, dtype=float)
    RAW_modeled_FVCs_Cin = np.asarray(ref_rhs_ode.Cin_modeled_FVCs, dtype=float)
    SCALE_controlled_FVCs_Cin = np.maximum(np.abs(RAW_controlled_FVCs_Cin), 1.0)
    SCALE_modeled_FVCs_Cin = np.maximum(np.abs(RAW_modeled_FVCs_Cin), 1.0)

    return EstimatedScales(
        SCALE_modeled_RMCs=jnp.asarray(SCALE_modeled_RMCs, dtype=jnp.float32),
        SCALE_V_in_cumulative=jnp.asarray(SCALE_V_in_cumulative, dtype=jnp.float32),
        SCALE_modeled_FVCs_cumulative=jnp.asarray(SCALE_modeled_FVCs_cumulative, dtype=jnp.float32),
        SCALE_controlled_FVCs_cumulative=jnp.asarray(SCALE_controlled_FVCs_cumulative, dtype=jnp.float32),
        SCALE_controlled_FVCs_rates=jnp.asarray(SCALE_controlled_FVCs_rates, dtype=jnp.float32),
        SCALE_controlled_FVCs_Cin=jnp.asarray(SCALE_controlled_FVCs_Cin, dtype=jnp.float32),
        SCALE_controlled_FVCs_bolus_rates=jnp.asarray(SCALE_controlled_FVCs_bolus_rates, dtype=jnp.float32),
        SCALE_controlled_PVs=jnp.asarray(SCALE_controlled_PVs, dtype=jnp.float32),
        SCALE_modeled_FVCs_Cin=jnp.asarray(SCALE_modeled_FVCs_Cin, dtype=jnp.float32),
        SCALE_modeled_BiologicalOde_rates=jnp.ones(n_rates, dtype=jnp.float32),
        SCALE_modeled_FVCs_rates=jnp.ones(n_modeled_VCs, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Loss: default per-target MSE + one non-negativity hinge per target
# ---------------------------------------------------------------------------


class NonNegLossModule(DefaultLossModule):
    """DefaultLossModule + a ``nonneg/<target>`` hinge penalizing negative
    predicted concentrations. Exercises the build_loss_module hook + custom
    named-term plumbing without needing bp-format bound metadata."""

    weight: float = eqx.field(static=True)

    def __init__(self, *, target_names, weight=0.1):
        super().__init__(target_names=target_names)
        self.weight = float(weight)

    @property
    def loss_names(self):
        return tuple(self.target_names) + tuple(
            f"nonneg/{name}" for name in self.target_names
        )

    def __call__(self, inputs: LossInputs) -> LossOutputs:
        base = super().__call__(inputs).named_losses
        mask = inputs.mask_measured_any
        denom = jnp.maximum(jnp.sum(mask), 1.0)
        penalties = {}
        for i, name in enumerate(self.target_names):
            violation = jax.nn.relu(-inputs.SCL_target_pred[:, i])
            penalties[f"nonneg/{name}"] = self.weight * (
                jnp.sum(jnp.square(violation) * mask) / denom
            )
        return LossOutputs(named_losses={**base, **penalties})


def build_loss_module(*, target_names, process_names, config, seed, collection):
    del process_names, config, seed, collection
    return NonNegLossModule(target_names=target_names)
