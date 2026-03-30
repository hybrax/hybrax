from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bpbench.dataclasses import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    ProcessVariable,
    ReactorMedium,
    SampleVolumeChange,
    TimeAxis,
    TimeSeries,
    Volume,
)

from bp_train.model_api import ReactionOutputs, UserReactionModule
import bp_train.trainer as trainer_module
from bp_train.trainer import single_process_measurement_loss, single_process_train_step
from bp_train.training_data import TrainingDataStore
from bp_train.wrapper import LibraryRhsWrapper


class _LinearReactionModule(UserReactionModule):
    model: eqx.nn.Linear
    non_model_bias: jax.Array

    def __init__(self):
        self.model = eqx.nn.Linear(1, 1, key=jax.random.key(123))
        self.non_model_bias = jnp.asarray([0.05], dtype=jnp.float32)

    def __call__(self, t, c_species, controls_vector):
        del t, controls_vector
        reaction = self.model(c_species)[0] + self.non_model_bias[0]
        return ReactionOutputs(
            reaction_terms=jnp.asarray([reaction], dtype=c_species.dtype),
            modeled_feed_rates=jnp.zeros((0,), dtype=c_species.dtype),
        )


class _CustomPartitionReactionModule(_LinearReactionModule):
    def partition_trainable(self):
        filter_spec = eqx.tree_at(
            lambda module: module.non_model_bias,
            jax.tree_util.tree_map(lambda _leaf: False, self),
            True,
        )
        return eqx.partition(self, filter_spec)


def _make_two_process_collection() -> BioProcessCollection:
    p1 = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "sample_1": SampleVolumeChange(
                    name="sample_1",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([1.0]),
                        values=jnp.asarray([-0.1]),
                    ),
                )
            },
        ),
        reactor_medium=ReactorMedium(name="rm", density=1.0, density_unit="kg/L"),
        process_variables={
            "X": ProcessVariable(
                name="X",
                unit="g/L",
                is_controlled=False,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 1.0, 2.0]),
                    values=jnp.asarray([1.0, 0.8, 0.64]),
                ),
            )
        },
    )
    p2 = BioProcess(
        metadata=BioProcessMetadata(name="p2", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.1,
            unit="L",
            volume_changes={
                "sample_1": SampleVolumeChange(
                    name="sample_1",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([1.0]),
                        values=jnp.asarray([-0.1]),
                    ),
                )
            },
        ),
        reactor_medium=ReactorMedium(name="rm", density=1.0, density_unit="kg/L"),
        process_variables={
            "X": ProcessVariable(
                name="X",
                unit="g/L",
                is_controlled=False,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 1.0]),
                    values=jnp.asarray([0.9, 0.72]),
                ),
            )
        },
    )
    return BioProcessCollection(processes={"p1": p1, "p2": p2}, metadata={})


def _build_wrapper_and_process():
    store = TrainingDataStore.from_collection(
        _make_two_process_collection(),
        target_variable_order=["X"],
    )
    process_data = store.get_process("p2")
    wrapper = LibraryRhsWrapper.from_process_controls(
        reaction_module=_LinearReactionModule(),
        controls=process_data.controls,
        species_names=process_data.target_names,
    )
    return wrapper, process_data


def _build_wrapper_and_process_with_custom_partition():
    store = TrainingDataStore.from_collection(
        _make_two_process_collection(),
        target_variable_order=["X"],
    )
    process_data = store.get_process("p2")
    wrapper = LibraryRhsWrapper.from_process_controls(
        reaction_module=_CustomPartitionReactionModule(),
        controls=process_data.controls,
        species_names=process_data.target_names,
    )
    return wrapper, process_data


def test_single_process_train_step_produces_gradients_and_keeps_frozen_params_static():
    wrapper, process_data = _build_wrapper_and_process()

    weight_before = np.asarray(wrapper.reaction_module.model.weight)
    frozen_before = np.asarray(wrapper.reaction_module.non_model_bias)

    wrapper_updated, loss, grads = single_process_train_step(
        wrapper,
        process_data,
        learning_rate=5e-2,
    )

    assert jnp.isfinite(loss)
    assert grads.model.weight is not None
    assert jnp.any(jnp.abs(grads.model.weight) > 0.0)
    assert grads.non_model_bias is None

    weight_after = np.asarray(wrapper_updated.reaction_module.model.weight)
    frozen_after = np.asarray(wrapper_updated.reaction_module.non_model_bias)
    assert not np.allclose(weight_before, weight_after)
    assert np.allclose(frozen_before, frozen_after)


def test_single_process_train_step_respects_custom_partition_trainable_override():
    wrapper, process_data = _build_wrapper_and_process_with_custom_partition()

    weight_before = np.asarray(wrapper.reaction_module.model.weight)
    bias_before = np.asarray(wrapper.reaction_module.non_model_bias)

    wrapper_updated, loss, grads = single_process_train_step(
        wrapper,
        process_data,
        learning_rate=5e-2,
    )

    assert jnp.isfinite(loss)
    assert grads.model.weight is None
    assert grads.non_model_bias is not None
    assert jnp.any(jnp.abs(grads.non_model_bias) > 0.0)

    weight_after = np.asarray(wrapper_updated.reaction_module.model.weight)
    bias_after = np.asarray(wrapper_updated.reaction_module.non_model_bias)
    assert np.allclose(weight_before, weight_after)
    assert not np.allclose(bias_before, bias_after)


def test_measurement_loss_ignores_padded_rows_via_mask():
    wrapper, process_data = _build_wrapper_and_process()
    # p2 has two active measurements but one padded row because p1 has three.
    assert process_data.n_meas == 2
    assert bool(process_data.meas_mask[2]) is False

    base_loss = single_process_measurement_loss(wrapper, process_data)

    poisoned_y = process_data.y_meas.at[2, 0].set(1e6)
    process_poisoned = eqx.tree_at(
        lambda pdata: pdata.y_meas,
        process_data,
        poisoned_y,
    )
    poisoned_loss = single_process_measurement_loss(wrapper, process_poisoned)
    assert poisoned_loss == pytest.approx(base_loss, rel=1e-6, abs=1e-6)


def test_single_process_train_step_accepts_nondefault_solver_settings():
    wrapper, process_data = _build_wrapper_and_process()

    wrapper_updated, loss, grads = single_process_train_step(
        wrapper,
        process_data,
        learning_rate=1e-2,
        max_solver_steps=500_000,
        solver_rtol=1e-4,
        solver_atol=1e-6,
        solver_use_jump_ts=False,
    )

    assert jnp.isfinite(loss)
    assert grads.model.weight is not None
    assert wrapper_updated.reaction_module.model.weight.shape == (1, 1)


def test_measurement_loss_forwards_nondefault_solver_options(monkeypatch):
    wrapper, process_data = _build_wrapper_and_process()
    captured: dict[str, object] = {}

    def _fake_simulate_measurement_states(
        wrapper_arg,
        process_data_arg,
        *,
        max_steps,
        rtol,
        atol,
        use_jump_ts,
    ):
        captured["wrapper"] = wrapper_arg
        captured["process_data"] = process_data_arg
        captured["max_steps"] = max_steps
        captured["rtol"] = rtol
        captured["atol"] = atol
        captured["use_jump_ts"] = use_jump_ts
        n_meas = int(process_data_arg.n_meas)
        return jnp.stack([process_data_arg.y0] * n_meas, axis=0)

    monkeypatch.setattr(
        trainer_module,
        "simulate_measurement_states",
        _fake_simulate_measurement_states,
    )

    loss = single_process_measurement_loss(
        wrapper,
        process_data,
        max_solver_steps=321_000,
        solver_rtol=1e-4,
        solver_atol=1e-6,
        solver_use_jump_ts=False,
    )

    assert jnp.isfinite(loss)
    assert captured["wrapper"] is wrapper
    assert captured["process_data"] is process_data
    assert captured["max_steps"] == 321_000
    assert captured["rtol"] == pytest.approx(1e-4)
    assert captured["atol"] == pytest.approx(1e-6)
    assert captured["use_jump_ts"] is False
