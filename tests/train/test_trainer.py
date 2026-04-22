from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from bp_format.dataclasses import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    ReactorMedium,
    ReactorMediumComponent,
    SampleVolumeChange,
    TimeAxis,
    TimeSeries,
    Volume,
)
from bp_format.mechanistic import get_rhs_ode

import bp_train.trainer as trainer_module
from bp_train.model_api import ReactionOutputs, UserReactionModule
from bp_train.harness import summarize_train_step_input_signature
from bp_train.trainer import (
    build_batched_loss_fn_from_sample_loss,
    clamp_padded_time_rows,
    measurement_loss_from_arrays,
)
from bp_train.training_data import TrainingDataStore
from bp_train.wrapper import HybridOdeWrapper


class _LinearReactionModule(UserReactionModule):
    model: eqx.nn.Linear
    non_model_bias: jax.Array

    def __init__(self):
        self.model = eqx.nn.Linear(1, 1, key=jax.random.key(123))
        self.non_model_bias = jnp.asarray([0.05], dtype=jnp.float32)

    def __call__(self, t, c_species, controls_vector):
        del t, controls_vector
        rate = self.model(c_species)[0] + self.non_model_bias[0]
        return ReactionOutputs(
            specific_rates=jnp.asarray([rate], dtype=c_species.dtype),
            modeled_feed_rates=jnp.zeros((0,), dtype=c_species.dtype),
        )


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
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray([0.0, 1.0, 2.0]),
                        values=jnp.asarray([1.0, 0.8, 0.64]),
                    ),
                    is_intracellular=False,
                ),
            },
        ),
        process_variables={},
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
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray([0.0, 1.0]),
                        values=jnp.asarray([0.9, 0.72]),
                    ),
                    is_intracellular=False,
                ),
            },
        ),
        process_variables={},
    )
    return BioProcessCollection(processes={"p1": p1, "p2": p2}, metadata={})


def _build_wrapper_and_process():
    collection = _make_two_process_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    process_data = store.get_process("p2")
    wrapper = HybridOdeWrapper.from_process(
        reaction_module=_LinearReactionModule(),
        process=collection.processes["p2"],
        controls=process_data.controls,
    )
    return wrapper, process_data


def test_train_step_input_signature_summary_tracks_hashable_scalar_values():
    sig_a = summarize_train_step_input_signature("adam", 1e-3, 42)
    sig_b = summarize_train_step_input_signature("adam", 2e-3, 42)
    assert sig_a != sig_b


def test_train_step_input_signature_summary_handles_none():
    sig = summarize_train_step_input_signature(None)
    assert sig == (("none",),)


def test_clamp_padded_time_rows_repeats_last_active_timestamp():
    times = jnp.asarray(
        [
            [0.0, 1.0, 2.0, 999.0],
            [0.0, 5.0, 6.0, 7.0],
        ],
        dtype=jnp.float32,
    )
    lengths = jnp.asarray([3, 1], dtype=jnp.int32)

    clamped = clamp_padded_time_rows(times, lengths)

    assert jnp.allclose(clamped[0], jnp.asarray([0.0, 1.0, 2.0, 2.0]))
    assert jnp.allclose(clamped[1], jnp.asarray([0.0, 0.0, 0.0, 0.0]))


def test_measurement_loss_from_arrays_ignores_padded_rows_via_mask():
    wrapper, process_data = _build_wrapper_and_process()
    assert process_data.n_meas == 2
    assert bool(process_data.meas_mask[2]) is False
    t_meas = clamp_padded_time_rows(
        process_data.t_meas[None, :],
        jnp.asarray([process_data.n_meas], dtype=jnp.int32),
    )[0]

    base_total, _ = measurement_loss_from_arrays(
        wrapper,
        t_meas=t_meas,
        y_meas=process_data.y_meas,
        meas_mask=process_data.meas_mask,
        n_meas=process_data.n_meas,
        y0=process_data.y0,
        jump_ts=process_data.controls.active_step_ts,
        max_solver_steps=100_000,
        solver_rtol=1e-5,
        solver_atol=1e-7,
    )

    poisoned_y = process_data.y_meas.at[2, 0].set(1e6)
    poisoned_total, _ = measurement_loss_from_arrays(
        wrapper,
        t_meas=t_meas,
        y_meas=poisoned_y,
        meas_mask=process_data.meas_mask,
        n_meas=process_data.n_meas,
        y0=process_data.y0,
        jump_ts=process_data.controls.active_step_ts,
        max_solver_steps=100_000,
        solver_rtol=1e-5,
        solver_atol=1e-7,
    )

    assert poisoned_total == pytest.approx(base_total, rel=1e-6, abs=1e-6)


def test_measurement_loss_from_arrays_forwards_nondefault_solver_options(monkeypatch):
    wrapper, process_data = _build_wrapper_and_process()
    captured: dict[str, object] = {}

    def _fake_simulate_measurement_states_on_grid(
        wrapper_arg,
        *,
        t_eval,
        n_meas,
        y0,
        max_steps,
        rtol,
        atol,
        jump_ts,
    ):
        captured["wrapper"] = wrapper_arg
        captured["t_eval"] = t_eval
        captured["n_meas"] = n_meas
        captured["y0"] = y0
        captured["max_steps"] = max_steps
        captured["rtol"] = rtol
        captured["atol"] = atol
        captured["jump_ts"] = jump_ts
        n_rows = t_eval.shape[0]
        return jnp.repeat(y0[None, :], repeats=n_rows, axis=0)

    monkeypatch.setattr(
        trainer_module,
        "_simulate_measurement_states_on_grid",
        _fake_simulate_measurement_states_on_grid,
    )

    total_loss, _ = measurement_loss_from_arrays(
        wrapper,
        t_meas=process_data.t_meas,
        y_meas=process_data.y_meas,
        meas_mask=process_data.meas_mask,
        n_meas=process_data.n_meas,
        y0=process_data.y0,
        jump_ts=None,
        max_solver_steps=321_000,
        solver_rtol=1e-4,
        solver_atol=1e-6,
    )

    assert jnp.isfinite(total_loss)
    assert captured["wrapper"] is wrapper
    assert captured["max_steps"] == 321_000
    assert captured["rtol"] == pytest.approx(1e-4)
    assert captured["atol"] == pytest.approx(1e-6)
    assert captured["jump_ts"] is None


def test_batched_loss_builder_preserves_none_jump_ts_branch():
    collection = _make_two_process_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    process_data = store.get_process("p2")
    wrapper = HybridOdeWrapper.from_process(
        reaction_module=_LinearReactionModule(),
        process=collection.processes["p2"],
        controls=process_data.controls,
    )

    batch = store.gather_batch(jnp.asarray([1], dtype=jnp.int32))
    batch_controls = store.controls_store.as_batch_controls()

    rhs_by_process = [
        get_rhs_ode(collection.processes[name]) for name in store.process_order
    ]
    batched_cin = jnp.stack([rhs.Cin for rhs in rhs_by_process], axis=0)
    batched_cin_modeled = jnp.stack([rhs.Cin_modeled for rhs in rhs_by_process], axis=0)

    def _sample_loss_fn(
        _wrapper,
        *,
        t_meas,
        y_meas,
        meas_mask,
        n_meas,
        y0,
        jump_ts,
        max_solver_steps,
        solver_rtol,
        solver_atol,
    ):
        del (
            t_meas,
            y_meas,
            meas_mask,
            n_meas,
            y0,
            max_solver_steps,
            solver_rtol,
            solver_atol,
        )
        score = 1.0 if jump_ts is None else 2.0
        return jnp.asarray(score), jnp.asarray([score], dtype=jnp.float32)

    batched_loss_fn = build_batched_loss_fn_from_sample_loss(_sample_loss_fn)
    mean_total_none, _, _ = batched_loss_fn(
        wrapper,
        batch,
        batch_controls,
        batched_cin,
        batched_cin_modeled,
        None,
        max_solver_steps=10,
        solver_rtol=1e-5,
        solver_atol=1e-7,
    )
    jump_ts_rows = jnp.zeros((1, 1), dtype=jnp.float32)
    mean_total_present, _, _ = batched_loss_fn(
        wrapper,
        batch,
        batch_controls,
        batched_cin,
        batched_cin_modeled,
        jump_ts_rows,
        max_solver_steps=10,
        solver_rtol=1e-5,
        solver_atol=1e-7,
    )

    assert float(mean_total_none) == pytest.approx(1.0)
    assert float(mean_total_present) == pytest.approx(2.0)
