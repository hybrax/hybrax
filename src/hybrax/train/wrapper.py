from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .controls import BP_TRAIN_SAMPLE_ACC_NAME
from .controls_store import PerProcessControls
from .model_api import ReactionOutputs


def _component_series_from_serialized(
    component_payload: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse one serialized feed-medium component concentration into `(xp, fp)`."""
    if component_payload is None:
        return np.asarray([0.0, 1.0], dtype=np.float32), np.asarray(
            [0.0, 0.0], dtype=np.float32
        )

    concentration = component_payload.get("concentration", {})
    kind = concentration.get("kind")
    if kind == "static":
        value = float(concentration["value"])
        return np.asarray([0.0, 1.0], dtype=np.float32), np.asarray(
            [value, value], dtype=np.float32
        )
    if kind == "timeseries":
        times_raw = concentration.get("times")
        legacy_times_raw = concentration.get("timepoints")
        if times_raw is None and legacy_times_raw is None:
            raise ValueError("timeseries concentration payload must include 'times'")

        if times_raw is not None and legacy_times_raw is not None:
            times = np.asarray(times_raw, dtype=np.float32)
            legacy_times = np.asarray(legacy_times_raw, dtype=np.float32)
            if times.shape != legacy_times.shape or not np.array_equal(
                times,
                legacy_times,
            ):
                raise ValueError(
                    "timeseries concentration payload has conflicting "
                    "'times' and 'timepoints'"
                )
            xp = times
        else:
            xp = np.asarray(
                times_raw if times_raw is not None else legacy_times_raw,
                dtype=np.float32,
            )

        fp = np.asarray(concentration["values"], dtype=np.float32)
        if xp.ndim != 1 or fp.ndim != 1 or xp.size != fp.size or xp.size == 0:
            raise ValueError(
                "invalid timeseries concentration payload in feed metadata"
            )
        if xp.size == 1:
            return np.asarray([xp[0], xp[0] + 1.0], dtype=np.float32), np.asarray(
                [fp[0], fp[0]], dtype=np.float32
            )
        return xp, fp
    raise ValueError(f"unsupported feed-medium concentration kind: {kind!r}")


def _pad_series_bank(
    bank: list[list[tuple[np.ndarray, np.ndarray]]],
    *,
    n_species: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad `[n_streams][n_species]` concentration series into dense 3D arrays."""
    if not bank:
        return (
            np.zeros((0, n_species, 2), dtype=np.float32),
            np.zeros((0, n_species, 2), dtype=np.float32),
        )

    max_points = max(xp.size for stream in bank for xp, _ in stream)
    max_points = max(max_points, 2)
    n_streams = len(bank)
    xp_out = np.zeros((n_streams, n_species, max_points), dtype=np.float32)
    fp_out = np.zeros((n_streams, n_species, max_points), dtype=np.float32)

    for stream_idx, stream in enumerate(bank):
        for species_idx, (xp, fp) in enumerate(stream):
            n = xp.size
            xp_out[stream_idx, species_idx, :n] = xp
            fp_out[stream_idx, species_idx, :n] = fp
            xp_out[stream_idx, species_idx, n:] = xp[n - 1]
            fp_out[stream_idx, species_idx, n:] = fp[n - 1]

    return xp_out, fp_out


def _interp_series_1d(t: jax.Array, xp: jax.Array, fp: jax.Array) -> jax.Array:
    return jnp.interp(t, xp, fp, left=fp[0], right=fp[-1])


def _evaluate_cin(t: jax.Array, xp: jax.Array, fp: jax.Array) -> jax.Array:
    """Evaluate concentration matrix `Cin` for all streams/species at time `t`."""
    eval_species = jax.vmap(_interp_series_1d, in_axes=(None, 0, 0), out_axes=0)
    eval_streams = jax.vmap(eval_species, in_axes=(None, 0, 0), out_axes=0)
    return eval_streams(t, xp, fp)


def _transport_term(
    rates: jax.Array,
    cin: jax.Array,
    c_species: jax.Array,
    v_real: jax.Array,
) -> jax.Array:
    """Compute sum_k `f_k * (Cin_k - c) / V_real`."""
    if cin.shape[0] == 0:
        return jnp.zeros_like(c_species)
    contrib = rates[:, None] * (cin - c_species[None, :])
    return jnp.sum(contrib, axis=0) / v_real


@dataclass(frozen=True)
class ModeledFeedSpec:
    """Static metadata for one modeled feed stream."""

    name: str
    component_concentrations: dict[str, float]


class LibraryRhsWrapper(eqx.Module):
    """Library-owned RHS wrapper with dilution/feed transport ownership."""

    reaction_module: Any
    controls: PerProcessControls
    species_names: tuple[str, ...] = eqx.field(static=True)
    min_real_volume: float = eqx.field(static=True)

    controlled_feed_names: tuple[str, ...] = eqx.field(static=True)
    controlled_feed_control_indices: jax.Array
    controlled_feed_cin_xp: jax.Array
    controlled_feed_cin_fp: jax.Array

    modeled_feed_names: tuple[str, ...] = eqx.field(static=True)
    modeled_feed_cin_xp: jax.Array
    modeled_feed_cin_fp: jax.Array

    sample_acc_control_index: int = eqx.field(static=True)

    @classmethod
    def from_process_controls(
        cls,
        *,
        reaction_module: Any,
        controls: PerProcessControls,
        species_names: list[str] | tuple[str, ...],
        modeled_feeds: list[ModeledFeedSpec] | None = None,
        min_real_volume: float = 1e-8,
    ) -> "LibraryRhsWrapper":
        """Build wrapper from per-process controls plus explicit species ordering."""
        if not isinstance(reaction_module, eqx.Module):
            raise TypeError("reaction_module must be an `eqx.Module` instance")

        ordered_species = tuple(species_names)
        n_species = len(ordered_species)

        controlled_feed_names: list[str] = []
        controlled_feed_indices: list[int] = []
        controlled_series_bank: list[list[tuple[np.ndarray, np.ndarray]]] = []

        for control_name in controls.control_names:
            if control_name == BP_TRAIN_SAMPLE_ACC_NAME:
                continue
            metadata = controls.control_metadata.get(control_name, {})
            if metadata.get("signal_family") != "feed":
                continue
            if metadata.get("source_kind") != "control":
                continue

            inlet = metadata.get("inlet_feed_medium") or {}
            components = inlet.get("components", {})

            per_species_series: list[tuple[np.ndarray, np.ndarray]] = []
            for species_name in ordered_species:
                component_payload = components.get(species_name)
                per_species_series.append(
                    _component_series_from_serialized(component_payload)
                )

            controlled_feed_names.append(control_name)
            controlled_feed_indices.append(controls.control_name_to_index[control_name])
            controlled_series_bank.append(per_species_series)

        ctrl_xp, ctrl_fp = _pad_series_bank(controlled_series_bank, n_species=n_species)

        modeled_feed_specs = modeled_feeds or []
        modeled_names = [spec.name for spec in modeled_feed_specs]
        if len(set(modeled_names)) != len(modeled_names):
            raise ValueError("modeled feed names must be unique")
        modeled_feed_names: list[str] = []
        modeled_series_bank: list[list[tuple[np.ndarray, np.ndarray]]] = []

        for feed_spec in modeled_feed_specs:
            per_species_series = []
            for species_name in ordered_species:
                value = float(feed_spec.component_concentrations.get(species_name, 0.0))
                per_species_series.append(
                    (
                        np.asarray([0.0, 1.0], dtype=np.float32),
                        np.asarray([value, value], dtype=np.float32),
                    )
                )
            modeled_feed_names.append(feed_spec.name)
            modeled_series_bank.append(per_species_series)

        mdl_xp, mdl_fp = _pad_series_bank(modeled_series_bank, n_species=n_species)

        if BP_TRAIN_SAMPLE_ACC_NAME not in controls.control_name_to_index:
            raise ValueError(
                "controls payload missing required sample control"
                f" {BP_TRAIN_SAMPLE_ACC_NAME}"
            )

        return cls(
            reaction_module=reaction_module,
            controls=controls,
            species_names=ordered_species,
            min_real_volume=float(min_real_volume),
            controlled_feed_names=tuple(controlled_feed_names),
            controlled_feed_control_indices=jnp.asarray(
                controlled_feed_indices, dtype=jnp.int32
            ),
            controlled_feed_cin_xp=jnp.asarray(ctrl_xp, dtype=jnp.float32),
            controlled_feed_cin_fp=jnp.asarray(ctrl_fp, dtype=jnp.float32),
            modeled_feed_names=tuple(modeled_feed_names),
            modeled_feed_cin_xp=jnp.asarray(mdl_xp, dtype=jnp.float32),
            modeled_feed_cin_fp=jnp.asarray(mdl_fp, dtype=jnp.float32),
            sample_acc_control_index=int(controls.sample_acc_global_index),
        )

    def _call_reaction_module(
        self,
        t: jax.Array,
        c_species: jax.Array,
        controls_vector: jax.Array,
    ) -> ReactionOutputs:
        outputs = self.reaction_module(t, c_species, controls_vector)
        if not hasattr(outputs, "reaction_terms") or not hasattr(
            outputs, "modeled_feed_rates"
        ):
            raise TypeError(
                "reaction_module output must expose `reaction_terms` and "
                "`modeled_feed_rates`"
            )
        return ReactionOutputs(
            reaction_terms=jnp.asarray(outputs.reaction_terms, dtype=c_species.dtype),
            modeled_feed_rates=jnp.asarray(
                outputs.modeled_feed_rates,
                dtype=c_species.dtype,
            ),
        )

    def __call__(self, t: float | jax.Array, y: jax.Array) -> jax.Array:
        """Compute full state derivative `[dc_species/dt..., dV_cont/dt]`.

        Notes
        -----
        The wrapped reaction module is called as
        `reaction_module(t, c_species, controls_vector)` and must return
        `ReactionOutputs`-compatible fields.
        """
        if y.ndim != 1:
            raise ValueError("state vector y must be rank-1")
        expected_state_size = len(self.species_names) + 1
        if y.shape[0] != expected_state_size:
            raise ValueError(
                "state vector y must have shape "
                f"({expected_state_size},), got {tuple(y.shape)}"
            )

        t_arr = jnp.asarray(t, dtype=y.dtype)
        c_species = y[:-1]
        v_cont = y[-1]
        controls_vector = self.controls.eval(t_arr)

        sample_acc = controls_vector[self.sample_acc_control_index]
        v_real = jnp.maximum(v_cont - sample_acc, jnp.asarray(self.min_real_volume))

        reaction_outputs = self._call_reaction_module(t_arr, c_species, controls_vector)

        if reaction_outputs.reaction_terms.ndim != 1:
            raise ValueError("reaction_terms must be a rank-1 vector")
        if reaction_outputs.modeled_feed_rates.ndim != 1:
            raise ValueError("modeled_feed_rates must be a rank-1 vector")

        if reaction_outputs.reaction_terms.shape != c_species.shape:
            raise ValueError(
                "reaction_terms must match species shape "
                f"{tuple(c_species.shape)},"
                f" got {tuple(reaction_outputs.reaction_terms.shape)}"
            )

        expected_modeled_shape = (len(self.modeled_feed_names),)
        if reaction_outputs.modeled_feed_rates.shape != expected_modeled_shape:
            raise ValueError(
                "modeled_feed_rates must match modeled feed metadata shape "
                f"{expected_modeled_shape},"
                f" got {tuple(reaction_outputs.modeled_feed_rates.shape)}"
            )

        controlled_rates = controls_vector[self.controlled_feed_control_indices]
        controlled_cin = _evaluate_cin(
            t_arr,
            self.controlled_feed_cin_xp,
            self.controlled_feed_cin_fp,
        )
        controlled_term = _transport_term(
            controlled_rates,
            controlled_cin,
            c_species,
            v_real,
        )

        modeled_cin = _evaluate_cin(
            t_arr,
            self.modeled_feed_cin_xp,
            self.modeled_feed_cin_fp,
        )
        modeled_term = _transport_term(
            reaction_outputs.modeled_feed_rates,
            modeled_cin,
            c_species,
            v_real,
        )

        dc_species = reaction_outputs.reaction_terms + controlled_term + modeled_term
        d_v_cont = jnp.sum(controlled_rates) + jnp.sum(
            reaction_outputs.modeled_feed_rates
        )
        return jnp.concatenate([dc_species, jnp.asarray([d_v_cont], dtype=y.dtype)])
