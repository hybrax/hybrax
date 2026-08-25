from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hybrax.format.dataclasses import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    FeedMedium,
    FeedMediumComponent,
    Inflow,
    ReactorMedium,
    ReactorMediumComponent,
    Outflow,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)
from hybrax.format.mechanistic import build_rhs_ode

import hybrax.train.trainer as trainer_module
from hybrax.train.physical_solve import solve_physical_states, within_fail_time
from hybrax.train.model_api import (
    AffineScaler,
    LossInputs,
    LossOutputs,
    ReactionOutputs,
    UserReactionModule,
    frozen_field,
    partition_trainable,
    trainable_field,
)
from hybrax.train.defaults import DefaultLossModule
from hybrax.train.harness import summarize_train_step_input_signature
from hybrax.train.trainer import (
    build_batched_loss_fn,
    clamp_padded_time_rows,
    evaluate_one_sample_loss,
    evaluate_sample_with_loss_module,
    simulate_measurement_states,
)
from hybrax.train.training_data import TrainingDataStore
from hybrax.train.wrapper import HybridOdeWrapper, SaveOutputs


def _measurement_loss(wrapper, **kwargs):
    """Thin shim: solve + DefaultLossModule, return (total, per_target)."""
    result = evaluate_sample_with_loss_module(wrapper, **kwargs)
    return result.total_loss, result.per_target_loss


class _LinearReactionModule(UserReactionModule):
    model: eqx.nn.Linear = trainable_field()
    non_model_bias: jax.Array = frozen_field()

    def __init__(self, **scale_kwargs):
        super().__init__(**scale_kwargs)
        self.model = eqx.nn.Linear(1, 1, key=jax.random.key(123))
        self.non_model_bias = jnp.asarray([0.05])

    def __call__(self, t, inputs):
        del t
        SCL_modeled_RMCs = inputs.SCL_modeled_RMCs
        rate = self.model(SCL_modeled_RMCs)[0] + self.non_model_bias[0]
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray(
                [rate], dtype=SCL_modeled_RMCs.dtype
            ),
            SCL_modeled_Inflows_rates=jnp.zeros((0,), dtype=SCL_modeled_RMCs.dtype),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
        )


class _RawDependentReactionModule(UserReactionModule):
    def __init__(self, **scale_kwargs):
        super().__init__(**scale_kwargs)

    def __call__(self, t, inputs):
        del t
        RAW_modeled_RMCs = self.unscale_modeled_RMCs(inputs.SCL_modeled_RMCs)
        rate = -0.2 * RAW_modeled_RMCs[0]
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray([rate]),
            SCL_modeled_Inflows_rates=jnp.zeros((0,), dtype=rate.dtype),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
        )


class _AuxReactionModule(UserReactionModule):
    def __init__(self, **scale_kwargs):
        super().__init__(**scale_kwargs)

    def __call__(self, t, inputs):
        SCL_modeled_RMCs = inputs.SCL_modeled_RMCs
        rate = jnp.asarray([0.0], dtype=SCL_modeled_RMCs.dtype)
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=rate,
            SCL_modeled_Inflows_rates=jnp.zeros((0,), dtype=SCL_modeled_RMCs.dtype),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
            auxiliary={
                "mu_raw": t,
                "latent_pair": jnp.asarray(
                    [t, SCL_modeled_RMCs[0]], dtype=SCL_modeled_RMCs.dtype
                ),
            },
        )


class _BlowUpReactionModule(UserReactionModule):
    """Biomass rate is 0 until t=1, then explosively stiff. Over measurement nodes
    [0, 1, 2] the segment [0, 1] succeeds and [1, 2] bails -> a genuine MID-trajectory
    failure (fail_time == 1.0), unlike ``max_steps=1`` which bails on the very first
    (t0) segment."""

    def __init__(self, **scale_kwargs):
        super().__init__(**scale_kwargs)

    def __call__(self, t, inputs):
        SCL_modeled_RMCs = inputs.SCL_modeled_RMCs
        rate = jnp.where(t > 1.0, 1.0e4 * SCL_modeled_RMCs[0], 0.0)
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray(
                [rate], dtype=SCL_modeled_RMCs.dtype
            ),
            SCL_modeled_Inflows_rates=jnp.zeros((0,), dtype=SCL_modeled_RMCs.dtype),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
        )


def _make_two_process_collection() -> BioProcessCollection:
    p1 = BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "sample_1": Outflow(
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
                "sample_1": Outflow(
                    name="sample_1",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=TimeSeries(
                        times=jnp.asarray([1.0]),
                        values=jnp.asarray([-0.2]),
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
                ),
            },
        ),
        process_variables={},
    )
    return BioProcessCollection(processes={"p1": p1, "p2": p2}, metadata={})


def _unit_scale_kwargs_for(rhs_ode, controls) -> dict[str, jnp.ndarray]:
    n_RMCs = len(rhs_ode.name_modeled_RMCs)
    n_Inflows = len(rhs_ode.name_modeled_Inflows)
    n_Outflows = len(rhs_ode.name_modeled_Outflows)
    n_rates = len(rhs_ode.name_modeled_rates)
    n_Inflow = len(controls.name_controlled_Inflows)
    n_Outflow = len(controls.name_controlled_Outflows)
    n_PV = len(controls.name_controlled_PVs)
    return {
        "SCALE_modeled_RMCs": jnp.ones(n_RMCs),
        "SCALE_V_in_cumulative": jnp.asarray(1.0),
        "SCALE_modeled_Inflows_cumulative": jnp.ones(n_Inflows),
        "SCALE_modeled_Outflows_cumulative": jnp.ones(n_Outflows),
        "SCALE_controlled_Inflows_cumulative": jnp.ones(n_Inflow),
        "SCALE_controlled_Inflows_rates": jnp.ones(n_Inflow),
        "SCALE_controlled_Outflows_cumulative": jnp.ones(n_Outflow),
        "SCALE_controlled_Outflows_rates": jnp.ones(n_Outflow),
        "SCALE_controlled_Inflows_Cin": jnp.ones((n_Inflow, n_RMCs)),
        "SCALE_controlled_PVs": jnp.ones(n_PV),
        "SCALE_modeled_Inflows_Cin": jnp.ones((n_Inflows, n_RMCs)),
        "SCALE_modeled_BiologicalOde_rates": jnp.ones(n_rates),
        "SCALE_modeled_Inflows_rates": jnp.ones(n_Inflows),
        "SCALE_modeled_Outflows_rates": jnp.ones(n_Outflows),
    }


def _build_wrapper_and_process(module_cls=_LinearReactionModule, process_name="p2"):
    from hybrax.format.mechanistic import build_rhs_ode as _build_rhs_ode

    collection = _make_two_process_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    process_data = store.get_process(process_name)
    rhs_ode = _build_rhs_ode(collection.processes[process_name])
    scale_kwargs = _unit_scale_kwargs_for(rhs_ode, process_data.controls)
    wrapper = HybridOdeWrapper.from_process(
        reaction_module=module_cls(**scale_kwargs),
        process=collection.processes[process_name],
        controls=process_data.controls,
        loss_module=DefaultLossModule(target_names=["biomass"]),
    )
    return wrapper, process_data


def _build_aux_wrapper_and_process():
    return _build_wrapper_and_process(_AuxReactionModule)


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
    )
    lengths = jnp.asarray([3, 1], dtype=jnp.int32)

    clamped = clamp_padded_time_rows(times, lengths)

    assert jnp.allclose(clamped[0], jnp.asarray([0.0, 1.0, 2.0, 2.0]))
    assert jnp.allclose(clamped[1], jnp.asarray([0.0, 0.0, 0.0, 0.0]))


def test_measurement_loss_from_arrays_ignores_padded_rows_via_mask():
    wrapper, process_data = _build_wrapper_and_process()
    assert process_data.n_measured == 2
    # Padded row: mask is False on every column.
    assert bool(jnp.any(process_data.mask_measured[2])) is False
    t_measured = clamp_padded_time_rows(
        process_data.t_measured[None, :],
        jnp.asarray([process_data.n_measured], dtype=jnp.int32),
    )[0]

    base_total, _ = _measurement_loss(
        wrapper,
        t_measured=t_measured,
        SCL_target_measured=process_data.y_measured,
        mask_measured=process_data.mask_measured,
        n_measured=process_data.n_measured,
        RAW_y0_measured=process_data.y0_measured,
        jump_ts=process_data.controls.active_jump_ts,
        max_solver_steps=100_000,
        solver_rtol=1e-5,
        solver_atol=1e-7,
    )

    poisoned_y = process_data.y_measured.at[2, 0].set(1e6)
    poisoned_total, _ = _measurement_loss(
        wrapper,
        t_measured=t_measured,
        SCL_target_measured=poisoned_y,
        mask_measured=process_data.mask_measured,
        n_measured=process_data.n_measured,
        RAW_y0_measured=process_data.y0_measured,
        jump_ts=process_data.controls.active_jump_ts,
        max_solver_steps=100_000,
        solver_rtol=1e-5,
        solver_atol=1e-7,
    )

    assert poisoned_total == pytest.approx(base_total, rel=1e-6, abs=1e-6)


def test_measurement_loss_from_arrays_forwards_nondefault_solver_options(monkeypatch):
    wrapper, process_data = _build_wrapper_and_process()
    captured: dict[str, object] = {}

    def _fake_solve_measurement_save_outputs_on_grid(
        wrapper_arg,
        *,
        t_eval,
        n_measured,
        RAW_y0,
        max_steps,
        rtol,
        atol,
        jump_ts,
    ):
        captured["wrapper"] = wrapper_arg
        captured["t_eval"] = t_eval
        captured["n_measured"] = n_measured
        captured["y0"] = RAW_y0
        captured["max_steps"] = max_steps
        captured["rtol"] = rtol
        captured["atol"] = atol
        captured["jump_ts"] = jump_ts
        n_rows = t_eval.shape[0]
        states = jnp.repeat(RAW_y0[None, :], repeats=n_rows, axis=0)
        save_outputs = SaveOutputs(
            SCL_states=states,
            RAW_V_export=states[:, len(wrapper_arg.modeled_RMC_names)],
            RAW_V=states[:, len(wrapper_arg.modeled_RMC_names)],
            RAW_modeled_BiologicalOde_rates=jnp.zeros(
                (n_rows, len(wrapper_arg.modeled_RMC_names)),
                dtype=states.dtype,
            ),
            RAW_modeled_Inflows_rates=jnp.zeros(
                (n_rows, len(wrapper_arg.modeled_Inflow_names)),
                dtype=states.dtype,
            ),
            RAW_modeled_Outflows_rates=jnp.zeros(
                (n_rows, len(wrapper_arg.modeled_Outflow_names)),
                dtype=states.dtype,
            ),
            auxiliary=None,
        )
        return save_outputs, jnp.asarray(jnp.inf, states.dtype)

    monkeypatch.setattr(
        trainer_module,
        "_solve_measurement_save_outputs_on_grid",
        _fake_solve_measurement_save_outputs_on_grid,
    )

    total_loss, _ = _measurement_loss(
        wrapper,
        t_measured=process_data.t_measured,
        SCL_target_measured=process_data.y_measured,
        mask_measured=process_data.mask_measured,
        n_measured=process_data.n_measured,
        RAW_y0_measured=process_data.y0_measured,
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


def test_evaluate_sample_from_arrays_matches_manual_loss_and_state_solve():
    wrapper, process_data = _build_wrapper_and_process()

    result = evaluate_sample_with_loss_module(
        wrapper,
        t_measured=process_data.t_measured,
        SCL_target_measured=process_data.y_measured,
        mask_measured=process_data.mask_measured,
        n_measured=process_data.n_measured,
        RAW_y0_measured=process_data.y0_measured,
        jump_ts=process_data.controls.active_jump_ts,
        max_solver_steps=100_000,
        solver_rtol=1e-5,
        solver_atol=1e-7,
    )
    states = solve_physical_states(
        wrapper,
        t_eval=process_data.t_measured,
        n_measured=process_data.n_measured,
        RAW_y0=process_data.y0_measured,
        max_steps=100_000,
        rtol=1e-5,
        atol=1e-7,
    )
    # Manually replicate the SCL-space loss kernel. ``states`` here is RAW
    # physical state from the callbacks solve. With unit SCALE_* SCL == RAW.
    y_pred = states[:, wrapper.target_state_indices]
    y_meas_safe = jnp.where(process_data.mask_measured, process_data.y_measured, 0.0)
    sq_err = jnp.square(y_pred - y_meas_safe)
    masked_sq_err = jnp.where(process_data.mask_measured, sq_err, 0.0)
    n_active_per_target = jnp.maximum(jnp.sum(process_data.mask_measured, axis=0), 1)
    per_target_loss = jnp.sum(masked_sq_err, axis=0) / n_active_per_target
    total_loss = jnp.mean(per_target_loss)

    assert jnp.isclose(result.total_loss, total_loss)
    assert jnp.allclose(result.per_target_loss, per_target_loss)
    # ``result.states`` is in SCL space; convert back for the comparison. Compare
    # only the active measurement rows: padded slots are masked out of the loss and
    # the two paths handle them differently — the evaluate path clamps padded times to
    # the last valid time (a sample time here, so its post-sample V drops), while the
    # direct solve above leaves them unclamped — so they need not agree.
    n = int(process_data.n_measured)
    assert jnp.allclose(
        jax.vmap(wrapper.reaction_module.unscale_state)(result.states)[:n],
        states[:n],
    )
    assert jnp.allclose(result.states, result.save_outputs.SCL_states)


def test_evaluate_sample_from_arrays_clamps_poisoned_padded_times():
    wrapper, process_data = _build_wrapper_and_process()
    t_measured = process_data.t_measured.at[2].set(-123.0)

    result = evaluate_sample_with_loss_module(
        wrapper,
        t_measured=t_measured,
        SCL_target_measured=process_data.y_measured,
        mask_measured=process_data.mask_measured,
        n_measured=process_data.n_measured,
        RAW_y0_measured=process_data.y0_measured,
        jump_ts=process_data.controls.active_jump_ts,
        max_solver_steps=100_000,
        solver_rtol=1e-5,
        solver_atol=1e-7,
    )
    clamped = clamp_padded_time_rows(
        t_measured[None, :],
        jnp.asarray([process_data.n_measured], dtype=jnp.int32),
    )[0]
    expected_states = solve_physical_states(
        wrapper,
        t_eval=clamped,
        n_measured=process_data.n_measured,
        RAW_y0=process_data.y0_measured,
        max_steps=100_000,
        rtol=1e-5,
        atol=1e-7,
    )

    assert jnp.allclose(result.states, expected_states)


def test_solve_physical_states_gradient_is_finite():
    """The diffrax_callbacks discrete-jump solve must be differentiable: the
    reverse-mode adjoint through the per-event segments yields a finite, non-zero
    gradient w.r.t. the reaction module. This is the property the pseudobatch
    formulation broke (its unbounded accumulator corrupted the adjoint), so it is
    the core regression guard for the callbacks migration."""
    wrapper, process_data = _build_wrapper_and_process()
    n_meas = int(process_data.n_measured)

    def loss(w):
        states = solve_physical_states(
            w,
            t_eval=process_data.t_measured,
            n_measured=process_data.n_measured,
            RAW_y0=process_data.y0_measured,
            max_steps=4096,
            rtol=1e-4,
            atol=1e-5,
        )
        return jnp.sum(states[:n_meas] ** 2)

    value, grad = eqx.filter_value_and_grad(loss)(wrapper)
    assert bool(jnp.isfinite(value))
    ann_leaves = [
        g
        for g in jax.tree_util.tree_leaves(grad.reaction_module)
        if eqx.is_inexact_array(g)
    ]
    assert ann_leaves, "expected differentiable reaction-module leaves"
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in ann_leaves)
    assert any(bool(jnp.any(g != 0.0)) for g in ann_leaves), "adjoint did not propagate"


def test_evaluate_sample_from_arrays_single_point_repeats_auxiliary_outputs():
    wrapper, process_data = _build_aux_wrapper_and_process()
    t_measured = process_data.t_measured.at[1:].set(jnp.asarray([999.0, -999.0]))
    SCL_target_measured = process_data.y_measured.at[1:, :].set(0.0)
    mask_measured = process_data.mask_measured.at[1:].set(False)

    result = evaluate_sample_with_loss_module(
        wrapper,
        t_measured=t_measured,
        SCL_target_measured=SCL_target_measured,
        mask_measured=mask_measured,
        n_measured=1,
        RAW_y0_measured=process_data.y0_measured,
        jump_ts=process_data.controls.active_jump_ts,
        max_solver_steps=100_000,
        solver_rtol=1e-5,
        solver_atol=1e-7,
    )

    assert result.save_outputs.auxiliary is not None
    assert set(result.save_outputs.auxiliary) == {"latent_pair", "mu_raw"}
    assert result.states.shape[0] == t_measured.shape[0]
    assert jnp.allclose(
        result.states,
        jnp.repeat(
            process_data.y0_measured[None, :], repeats=t_measured.shape[0], axis=0
        ),
    )
    assert jnp.allclose(result.save_outputs.auxiliary["mu_raw"], 0.0)
    assert result.save_outputs.auxiliary["latent_pair"].shape == (
        t_measured.shape[0],
        2,
    )
    assert jnp.allclose(
        result.save_outputs.auxiliary["latent_pair"][:, 0],
        jnp.zeros((t_measured.shape[0],), dtype=process_data.y0_measured.dtype),
    )


def test_evaluate_sample_from_arrays_forwards_step_to_result():
    wrapper, process_data = _build_wrapper_and_process()
    result = evaluate_sample_with_loss_module(
        wrapper,
        t_measured=process_data.t_measured,
        SCL_target_measured=process_data.y_measured,
        mask_measured=process_data.mask_measured,
        n_measured=process_data.n_measured,
        RAW_y0_measured=process_data.y0_measured,
        jump_ts=process_data.controls.active_jump_ts,
        max_solver_steps=100_000,
        solver_rtol=1e-5,
        solver_atol=1e-7,
        step=42,
    )

    assert int(result.step) == 42
    assert jnp.issubdtype(result.step.dtype, jnp.integer)


def _build_batched_setup(
    module_cls=_LinearReactionModule,
    process_name="p2",
    batch_process_names=None,
):
    collection = _make_two_process_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    process_data = store.get_process(process_name)
    rhs = build_rhs_ode(collection.processes[process_name])
    wrapper = HybridOdeWrapper.from_process(
        reaction_module=module_cls(
            **_unit_scale_kwargs_for(rhs, process_data.controls)
        ),
        process=collection.processes[process_name],
        controls=process_data.controls,
        loss_module=DefaultLossModule(target_names=["biomass"]),
    )
    process_names = batch_process_names or (process_name,)
    process_indices = [list(store.process_order).index(name) for name in process_names]
    indices = jnp.asarray(process_indices, dtype=jnp.int32)
    batch = store.gather_batch(indices)
    batch_controls = batch.controls
    rhs_by_process = [
        build_rhs_ode(collection.processes[name]) for name in store.process_order
    ]
    batched_cin = jnp.stack(
        [rhs.Cin_controlled_Inflows for rhs in rhs_by_process], axis=0
    )
    batched_cin_modeled = jnp.stack(
        [rhs.Cin_modeled_Inflows for rhs in rhs_by_process], axis=0
    )
    return wrapper, batch, batch_controls, batched_cin, batched_cin_modeled


def test_batched_loss_fn_runs_with_step_and_loss_module():
    wrapper, batch, batch_controls, batched_cin, batched_cin_modeled = (
        _build_batched_setup()
    )
    batched_loss_fn = build_batched_loss_fn()
    mean_total, per_sample_per_target, per_sample, *_ = batched_loss_fn(
        wrapper,
        batch,
        batched_cin,
        batched_cin_modeled,
        None,
        max_solver_steps=100_000,
        solver_rtol=1e-5,
        solver_atol=1e-7,
        step=7,
    )
    assert jnp.isfinite(mean_total)
    # One sample in the batch, one named loss term ("biomass"): the 2nd element
    # is per-sample per-target (n_proc, n_targets); per_sample is (n_proc,).
    assert per_sample_per_target.shape == (1, 1)
    assert per_sample.shape == (1,)


def test_batched_loss_fn_preserves_none_jump_ts_branch():
    wrapper, batch, batch_controls, batched_cin, batched_cin_modeled = (
        _build_batched_setup()
    )
    batched_loss_fn = build_batched_loss_fn()
    mean_total_none, *_ = batched_loss_fn(
        wrapper,
        batch,
        batched_cin,
        batched_cin_modeled,
        None,
        max_solver_steps=100_000,
        solver_rtol=1e-5,
        solver_atol=1e-7,
    )
    jump_ts_rows = jnp.zeros((1, 1))
    mean_total_present, *_ = batched_loss_fn(
        wrapper,
        batch,
        batched_cin,
        batched_cin_modeled,
        jump_ts_rows,
        max_solver_steps=100_000,
        solver_rtol=1e-5,
        solver_atol=1e-7,
    )
    assert jnp.isfinite(mean_total_none)
    assert jnp.isfinite(mean_total_present)


def _run_batched(wrapper, batch, batch_controls, batched_cin, batched_cin_modeled):
    batched_loss_fn = build_batched_loss_fn()
    return batched_loss_fn(
        wrapper,
        batch,
        batched_cin,
        batched_cin_modeled,
        None,
        max_solver_steps=512,
        solver_rtol=1e-5,
        solver_atol=1e-7,
        step=1,
    )


def test_jitted_batched_loss_uses_dynamic_local_controls_and_global_cin(monkeypatch):
    wrapper, reversed_batch, _controls, _cin, cin_modeled = _build_batched_setup(
        batch_process_names=("p2", "p1")
    )
    _, forward_batch, *_ = _build_batched_setup(batch_process_names=("p1", "p2"))
    cin = jnp.asarray([[[10.0]], [[20.0]]])

    def _fake_evaluate(sample_wrapper, **_kwargs):
        signal = (
            sample_wrapper.controls.sample_event_volumes[0]
            + sample_wrapper.rhs_ode.Cin_controlled_Inflows.reshape(-1)[0]
        )
        return type(
            "Result",
            (),
            {
                "total_loss": signal,
                "per_target_loss": jnp.asarray([signal]),
                "prediction_t": None,
                "prediction_save_outputs": None,
                "prediction_valid": None,
                "measurement_save_outputs": None,
                "measurement_prediction_valid": jnp.ones(1, dtype=bool),
                "fail_time": jnp.asarray(jnp.inf),
            },
        )()

    monkeypatch.setattr(
        trainer_module, "evaluate_sample_with_loss_module", _fake_evaluate
    )
    loss_fn = eqx.filter_jit(build_batched_loss_fn())

    def run(batch):
        return loss_fn(
            wrapper,
            batch,
            cin,
            cin_modeled,
            None,
            max_solver_steps=1,
            solver_rtol=1e-5,
            solver_atol=1e-7,
        )[2]

    np.testing.assert_allclose(run(forward_batch), jnp.asarray([10.1, 20.2]))
    np.testing.assert_allclose(run(reversed_batch), jnp.asarray([20.2, 10.1]))


def test_batched_loss_fn_surfaces_per_sample_fail_time():
    """The batched loss fn appends a per-sample fail_time as its LAST element: ``+inf``
    for a clean solve, finite when a sample's ODE segment bailed. This is what the
    harness reduces per step to report how often segments fail."""
    healthy = _build_batched_setup()  # p2 + linear rate: never bails
    *_, healthy_fail = _run_batched(*healthy)
    assert healthy_fail.shape == (1,)
    assert bool(jnp.all(jnp.isinf(healthy_fail))), "clean solve -> fail_time == inf"

    # p1 + blow-up rate: segment [1, 2] bails -> finite fail_time == 1.0.
    blown = _build_batched_setup(_BlowUpReactionModule, process_name="p1")
    *_, blown_fail = _run_batched(*blown)
    assert blown_fail.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(blown_fail))), "bailing sample -> finite fail_time"
    assert float(blown_fail[0]) == pytest.approx(1.0, abs=1e-3)


def test_batched_loss_fn_keeps_fail_times_independent_per_lane():
    setup = _build_batched_setup(
        _BlowUpReactionModule,
        process_name="p1",
        batch_process_names=("p1", "p2"),
    )
    *_, fail_times = _run_batched(*setup)

    assert fail_times.shape == (2,)
    assert float(fail_times[0]) == pytest.approx(1.0, abs=1e-3)
    assert bool(jnp.isinf(fail_times[1])), "healthy p2 lane"


def test_evaluate_one_sample_loss_returns_fail_time():
    """The pmap/gspmd shared per-sample entry point returns fail_time as its 3rd
    value (``+inf`` clean, finite on a bail) so the sharded paths can gather it.
    Inputs mirror the pmap ``_one`` preprocessing (clamp times, pre-scale targets)."""
    wrapper, batch, batch_controls, batched_cin, batched_cin_modeled = (
        _build_batched_setup(_BlowUpReactionModule, process_name="p1")
    )
    scale_targets = wrapper.reaction_module.SCALE_state[wrapper.target_state_indices]
    pidx = batch.process_indices[0]
    t_row = clamp_padded_time_rows(batch.t_measured, batch.n_measured)[0]
    out = evaluate_one_sample_loss(
        wrapper,
        batch_controls,
        pidx,
        t_row,
        batch.y_measured[0] / scale_targets[None, :],
        batch.mask_measured[0],
        batch.n_measured[0],
        batch.y0_measured[0],
        batched_cin[pidx],
        batched_cin_modeled[pidx],
        batch.retention_controlled_Outflows[0],
        batch.retention_modeled_Outflows[0],
        None,
        max_solver_steps=512,
        solver_rtol=1e-5,
        solver_atol=1e-7,
    )
    assert len(out) == 3, "returns (total_loss, per_target_loss, fail_time)"
    total_loss, _per_target, fail_time = out
    assert bool(jnp.isfinite(total_loss)), "loss stays finite despite the bail"
    assert float(fail_time) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Dense-grid helpers (hybrax/train/dense.py)
# ---------------------------------------------------------------------------


def test_build_union_time_grid_sorts_and_indexes_correctly():
    from hybrax.train.dense import build_union_time_grid

    t_meas = jnp.asarray([0.0, 1.0, 4.0])
    t_eval, sample_idx, dense_t, dense_idx, _pred_t, _pred_idx = build_union_time_grid(
        t_meas, n_measured=3, n_dense=3
    )
    # dense linspace covers the (active) measurement span.
    assert jnp.allclose(dense_t, jnp.asarray([0.0, 2.0, 4.0]))
    # t_eval is the sorted concat.
    assert jnp.all(jnp.diff(t_eval) >= 0)
    assert t_eval.shape[0] == 6
    # Round-trip: gathering t_eval by the index arrays returns the originals.
    assert jnp.allclose(t_eval[sample_idx], t_meas)
    assert jnp.allclose(t_eval[dense_idx], dense_t)


def test_dense_point_mask_handles_jump_ts_and_none():
    from hybrax.train.dense import dense_point_mask_away_from_jumps

    dense_t = jnp.linspace(0.0, 10.0, 11)
    # No jumps: every point kept.
    assert bool(jnp.all(dense_point_mask_away_from_jumps(dense_t, None, 0.6)))
    # Jump at t=3.5 with eps=0.6 masks dense_t[3]=3.0 and dense_t[4]=4.0.
    mask = dense_point_mask_away_from_jumps(dense_t, jnp.asarray([3.5]), 0.6)
    expected = jnp.asarray(
        [True, True, True, False, False, True, True, True, True, True, True]
    )
    assert bool(jnp.all(mask == expected))


def test_dense_triple_mask_excludes_triples_straddling_a_jump():
    from hybrax.train.dense import dense_triple_mask_away_from_jumps

    dense_t = jnp.linspace(0.0, 10.0, 11)
    # No jumps: every triple kept.
    triple = dense_triple_mask_away_from_jumps(dense_t, None, 0.6)
    assert triple.shape == (9,)
    assert bool(jnp.all(triple))
    # Jump at t=3.5: triples whose (left-eps, right+eps) span contains 3.5
    # are rejected. Triples are (i-1, i, i+1) over indices 0..8.
    triple = dense_triple_mask_away_from_jumps(dense_t, jnp.asarray([3.5]), 0.6)
    # Triples (1,2,3), (2,3,4), (3,4,5) all have spans covering 3.5; the rest don't.
    # (0,1,2) → [0-0.6, 2+0.6]=[-0.6, 2.6] excludes 3.5 → True.
    # (4,5,6) → [4-0.6, 6+0.6]=[3.4, 6.6] contains 3.5 → False
    # (sits within eps of jump).
    # so the False region is indices 1..4 inclusive.
    expected = jnp.asarray([True, False, False, False, False, True, True, True, True])
    assert bool(jnp.all(triple == expected))


def test_all_triple_reduces_point_mask_to_triple_stencil():
    from hybrax.train.dense import all_triple

    # Length n -> n-2; a triple (i-1,i,i+1) is True iff all three points are True.
    point = jnp.asarray([True, True, False, True, True])
    out = all_triple(point)
    assert out.shape == (3,)
    # (0,1,2) has a False -> F; (1,2,3) has a False -> F; (2,3,4) has a False -> F
    assert bool(jnp.all(out == jnp.asarray([False, False, False])))
    assert bool(jnp.all(all_triple(jnp.ones(5, dtype=bool))))


# ---------------------------------------------------------------------------
# fail_time per-point masking (piece 2): the loss must drop post-failure points
# and stay finite/differentiable even for the ``value * mask`` idiom that every
# repo custom bounds/nonneg/hinge module uses (``0 * inf = nan`` otherwise).
# ``max_solver_steps=1`` forces the first segment (from t0) to bail, so
# ``fail_time == t0`` and every point after t0 is post-failure.
# ---------------------------------------------------------------------------


class _MeasHingeLoss(DefaultLossModule):
    """DefaultLossModule + a measurement-grid nonneg hinge built with the
    ``penalty * mask_measured_any`` multiply idiom (the same shape the tub/kittler/
    fixture custom modules use). If a post-failure predicted state were left as inf,
    ``0 * inf`` would make this nan even though the row's mask is 0."""

    @property
    def loss_names(self):
        return tuple(self.target_names) + ("hinge",)

    def __call__(self, inputs: LossInputs) -> LossOutputs:
        base = super().__call__(inputs).named_losses
        mask = inputs.mask_measured_any
        violation = jax.nn.relu(inputs.SCL_target_pred[:, 0])
        hinge = jnp.sum(jnp.square(violation) * mask) / jnp.maximum(jnp.sum(mask), 1.0)
        return LossOutputs(named_losses={**base, "hinge": hinge})


class _DenseVolumeProbeLoss(DefaultLossModule):
    _dense_grid_n: int = eqx.field(static=True, default=3)

    @property
    def dense_grid_n(self):
        return self._dense_grid_n

    @property
    def loss_names(self):
        return tuple(self.target_names) + ("dense_raw_v", "dense_raw_v_unclamped")

    def __call__(self, inputs: LossInputs) -> LossOutputs:
        base = super().__call__(inputs).named_losses
        return LossOutputs(
            named_losses={
                **base,
                "dense_raw_v": jnp.mean(inputs.dense_RAW_V),
                "dense_raw_v_unclamped": jnp.mean(inputs.dense_RAW_V_unclamped),
            }
        )


class _DenseFailLoss(DefaultLossModule):
    """DefaultLossModule + dense terms that consume ``dense_valid_time``: a validity
    count (to pin that the mask is populated with the right cutoff) and a
    ``value * dense_valid_time`` dense hinge (to pin dense states finite)."""

    _dense_grid_n: int = eqx.field(static=True)

    def __init__(self, *, target_names, dense_grid_n=16):
        super().__init__(target_names=target_names)
        self._dense_grid_n = int(dense_grid_n)

    @property
    def dense_grid_n(self):
        return self._dense_grid_n

    @property
    def loss_names(self):
        return tuple(self.target_names) + ("dense_valid_count", "dense_hinge")

    def __call__(self, inputs: LossInputs) -> LossOutputs:
        base = super().__call__(inputs).named_losses
        valid = inputs.dense_valid_time.astype(inputs.dense_SCL_states.dtype)
        count = jnp.sum(valid)
        hinge = jnp.sum(jnp.square(inputs.dense_SCL_states[:, 0]) * valid)
        return LossOutputs(
            named_losses={**base, "dense_valid_count": count, "dense_hinge": hinge}
        )


def test_trainer_wires_dense_volume_from_export():
    wrapper, process_data = _build_wrapper_and_process()
    wrapper = eqx.tree_at(
        lambda w: w.loss_module,
        wrapper,
        _DenseVolumeProbeLoss(target_names=["biomass"]),
    )

    result = evaluate_sample_with_loss_module(
        wrapper,
        t_measured=process_data.t_measured,
        SCL_target_measured=process_data.y_measured,
        mask_measured=process_data.mask_measured,
        n_measured=process_data.n_measured,
        RAW_y0_measured=process_data.y0_measured,
        jump_ts=process_data.controls.active_jump_ts,
        max_solver_steps=100_000,
        solver_rtol=1e-5,
        solver_atol=1e-7,
    )

    assert result.per_target_loss[1] == pytest.approx(31 / 30)
    assert result.per_target_loss[2] == pytest.approx(31.0 / 30.0)


def _fail_time_kwargs(process_data, **overrides):
    kw = dict(
        t_measured=process_data.t_measured,
        SCL_target_measured=process_data.y_measured,
        mask_measured=process_data.mask_measured,
        n_measured=process_data.n_measured,
        RAW_y0_measured=process_data.y0_measured,
        jump_ts=process_data.controls.active_jump_ts,
        solver_rtol=1e-5,
        solver_atol=1e-7,
    )
    kw.update(overrides)
    return kw


def test_fail_time_measurement_mask_excludes_post_failure_points():
    """Post-failure measurement targets must not affect the loss (the cutoff mask has
    teeth): perturbing every target after the (early) failure time leaves the loss
    unchanged, because those rows are masked out."""
    wrapper, pd = _build_wrapper_and_process()
    kw = _fail_time_kwargs(pd, max_solver_steps=1)  # forces an early bail

    y = pd.y_measured
    del kw["SCL_target_measured"]
    L1 = evaluate_sample_with_loss_module(
        wrapper, SCL_target_measured=y, **kw
    ).total_loss
    # Perturb every target strictly after t0 (all post-failure for an early bail).
    post = (pd.t_measured > pd.t_measured[0] + 1e-6)[:, None]
    y2 = jnp.where(post, y + 1000.0, y)
    L2 = evaluate_sample_with_loss_module(
        wrapper, SCL_target_measured=y2, **kw
    ).total_loss

    assert bool(jnp.isfinite(L1)) and bool(jnp.isfinite(L2))
    assert float(L1) == pytest.approx(float(L2), rel=1e-6, abs=1e-6), (
        "post-failure measurement targets must not affect the loss"
    )


def test_fail_time_custom_value_times_mask_loss_is_finite_on_failure():
    """A ``penalty * mask`` custom loss stays finite AND differentiable on a failed
    solve — proving loss-facing states are finite (no ``0 * inf = nan``)."""
    wrapper, pd = _build_wrapper_and_process()
    wrapper = eqx.tree_at(
        lambda w: w.loss_module, wrapper, _MeasHingeLoss(target_names=["biomass"])
    )
    kw = _fail_time_kwargs(pd, max_solver_steps=1)
    trainable, static = partition_trainable(wrapper)

    def loss(m):
        return evaluate_sample_with_loss_module(eqx.combine(m, static), **kw).total_loss

    val, grad = eqx.filter_value_and_grad(loss)(trainable)
    assert bool(jnp.isfinite(val)), "value*mask custom loss must be finite on a bail"
    leaves = [g for g in jax.tree_util.tree_leaves(grad) if eqx.is_inexact_array(g)]
    assert leaves and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves), (
        "gradient must be finite (no 0*inf=nan from post-failure states)"
    )


def test_fail_time_dense_valid_time_populated_and_finite_on_failure():
    """``dense_valid_time`` is populated with the correct cutoff (all dense rows valid
    on a healthy solve; only pre-failure rows on a bail), and a dense ``value * mask``
    term stays finite + differentiable on a bail."""
    wrapper, pd = _build_wrapper_and_process()
    wrapper = eqx.tree_at(
        lambda w: w.loss_module,
        wrapper,
        _DenseFailLoss(target_names=["biomass"], dense_grid_n=16),
    )
    names = wrapper.loss_module.loss_names
    ci = names.index("dense_valid_count")

    healthy = evaluate_sample_with_loss_module(
        wrapper, **_fail_time_kwargs(pd, max_solver_steps=100_000)
    )
    failed = evaluate_sample_with_loss_module(
        wrapper, **_fail_time_kwargs(pd, max_solver_steps=1)
    )
    healthy_count = float(healthy.per_target_loss[ci])
    failed_count = float(failed.per_target_loss[ci])
    assert healthy_count == pytest.approx(16.0), "healthy solve: every dense row valid"
    assert failed_count < healthy_count, (
        "an early failure must invalidate the post-failure dense rows"
    )

    trainable, static = partition_trainable(wrapper)

    def loss(m):
        return evaluate_sample_with_loss_module(
            eqx.combine(m, static), **_fail_time_kwargs(pd, max_solver_steps=1)
        ).total_loss

    val, grad = eqx.filter_value_and_grad(loss)(trainable)
    assert bool(jnp.isfinite(val)), "dense value*mask term must be finite on a bail"
    leaves = [g for g in jax.tree_util.tree_leaves(grad) if eqx.is_inexact_array(g)]
    assert leaves and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)


def test_fail_time_prediction_export_marks_invalid_rows():
    """A failed forward exposes valid rows without changing loss placeholders."""
    wrapper, pd = _build_wrapper_and_process()
    result = evaluate_sample_with_loss_module(
        wrapper, **_fail_time_kwargs(pd, max_solver_steps=1, prediction_grid_n=8)
    )
    assert bool(jnp.isfinite(result.total_loss)), "loss stays finite on a bail"
    np.testing.assert_array_equal(
        result.measurement_prediction_valid,
        within_fail_time(pd.t_measured, result.fail_time),
    )
    assert not bool(jnp.all(result.measurement_prediction_valid))
    assert bool(jnp.all(jnp.isfinite(result.measurement_save_outputs.SCL_states)))
    assert bool(jnp.all(jnp.isfinite(result.states))), (
        "loss-facing placeholders must remain finite"
    )

    pred_t = result.prediction_t
    np.testing.assert_array_equal(
        result.prediction_valid,
        within_fail_time(pred_t, result.fail_time),
    )
    assert not bool(jnp.all(result.prediction_valid))
    assert bool(jnp.all(jnp.isfinite(result.prediction_save_outputs.SCL_states)))


def test_simulate_measurement_states_validates_control_support(monkeypatch):
    wrapper, pd = _build_wrapper_and_process()
    checked = []

    def validate_support(_controls, t0, t1):
        checked.append((t0, t1))
        raise ValueError("support sentinel")

    monkeypatch.setattr(type(pd.controls), "validate_support", validate_support)

    with pytest.raises(ValueError, match="support sentinel"):
        simulate_measurement_states(wrapper, pd)

    assert checked == [
        (float(pd.active_t_measured[0]), float(pd.active_t_measured[-1]))
    ]


def test_simulate_measurement_states_preserves_failure_sentinel_on_bail():
    """Forward/export callers (which do not request ``fail_time``) keep the diagnostic
    non-finite sentinel on a failed solve, rather than silently reading back as ``y0``
    (the finiteness fallback is gated to the loss path)."""
    wrapper, pd = _build_wrapper_and_process()
    states = simulate_measurement_states(wrapper, pd, max_steps=1)
    assert bool(jnp.any(jnp.isinf(states))), (
        "a failed forward solve must leave a detectable non-finite sentinel"
    )


def test_fail_time_export_marks_mid_trajectory_failure_nonfinite():
    """A MID-trajectory bail (not a t0 bail) must leave a detectable non-finite marker
    on the forward/export path. The raw argmin gather returns a STALE FINITE value for
    post-failure nodes (it falls back to the last reached node), so the marker must be
    written from ``fail_time`` — trusting the gather (as a first-segment-only test would
    never catch) silently presents a stale value as a real prediction."""
    wrapper, pd = _build_wrapper_and_process(_BlowUpReactionModule, process_name="p1")
    t = pd.active_t_measured  # nodes [0, 1, 2]; rate explodes for t>1 -> fail_time == 1
    pre = t <= 1.0 + 1e-4

    # Forward/export path: post-failure rows non-finite, pre-failure rows finite.
    export_states = simulate_measurement_states(wrapper, pd, max_steps=512)
    assert bool(jnp.all(jnp.isfinite(export_states[pre]))), "pre-failure rows finite"
    assert bool(jnp.all(~jnp.isfinite(export_states[~pre]))), (
        "post-failure export rows must be non-finite, not a stale finite value"
    )

    # Loss-facing path: fail_time is the last good node, and states stay finite (y0).
    loss_states, fail_time = solve_physical_states(
        wrapper,
        t_eval=t,
        n_measured=pd.n_measured,
        RAW_y0=pd.y0_measured,
        max_steps=512,
        rtol=1e-5,
        atol=1e-7,
        return_fail_time=True,
    )
    assert float(fail_time) == pytest.approx(1.0, abs=1e-3), (
        "fail_time is the last successfully-reached node"
    )
    assert bool(jnp.all(jnp.isfinite(loss_states))), "loss-facing states stay finite"


def test_affine_offset_preserves_moving_raw_trajectory_through_sample_event():
    linear, process_data = _build_wrapper_and_process(
        _RawDependentReactionModule, process_name="p1"
    )
    RMC_scaler = linear.reaction_module.SCALE_modeled_RMCs
    affine = eqx.tree_at(
        lambda w: w.reaction_module.SCALE_modeled_RMCs,
        linear,
        AffineScaler(
            scale=RMC_scaler.scale,
            offset=jnp.asarray([10.0], dtype=RMC_scaler.scale.dtype),
        ),
    )

    def solve(wrapper):
        return solve_physical_states(
            wrapper,
            t_eval=process_data.t_measured,
            n_measured=process_data.n_measured,
            RAW_y0=process_data.y0_measured,
            max_steps=100_000,
            rtol=1e-8,
            atol=1e-10,
            jump_ts=process_data.controls.active_jump_ts,
        )

    linear_states = solve(linear)
    affine_states = solve(affine)

    assert linear_states[-1, 0] < linear_states[0, 0]
    assert linear_states[1, 1] == pytest.approx(0.9)
    assert jnp.allclose(affine_states, linear_states, rtol=1e-6, atol=1e-8)


def test_affine_offset_cancels_from_measurement_loss():
    # Test 3: same physical prediction/measurement residual with b=0 vs b!=0.
    # _AuxReactionModule emits q=0 independent of its input, so both wrappers
    # have the same physical trajectory; only the state coordinate origin differs.
    linear, process_data = _build_wrapper_and_process(_AuxReactionModule)
    affine = eqx.tree_at(
        lambda w: w.reaction_module.SCALE_modeled_RMCs,
        linear,
        AffineScaler(
            scale=linear.reaction_module.SCALE_modeled_RMCs.scale,
            offset=jnp.asarray([10.0]),
        ),
    )
    t_measured = clamp_padded_time_rows(
        process_data.t_measured[None, :],
        jnp.asarray([process_data.n_measured], dtype=jnp.int32),
    )[0]

    def loss(wrapper):
        target_scaler = wrapper.reaction_module.SCALE_state[
            wrapper.target_state_indices
        ]
        total, _ = _measurement_loss(
            wrapper,
            t_measured=t_measured,
            SCL_target_measured=process_data.y_measured / target_scaler,
            mask_measured=process_data.mask_measured,
            n_measured=process_data.n_measured,
            RAW_y0_measured=process_data.y0_measured,
            jump_ts=process_data.controls.active_jump_ts,
            max_solver_steps=100_000,
            solver_rtol=1e-7,
            solver_atol=1e-9,
        )
        return total

    linear_loss = loss(linear)
    affine_loss = loss(affine)
    assert float(linear_loss) > 0.0  # nontrivial: prediction != measurements
    assert jnp.allclose(affine_loss, linear_loss, rtol=1e-5, atol=1e-7)


def test_zero_offset_affine_matches_linear_through_differentiated_solve():
    """A zero-offset `AffineScaler` is a drop-in for `LinearScaler` on a state axis,
    through the differentiated solve and not merely for values.

    This is the only test covering the traced vector field: `physical_solve.py` closes
    `SCALE = ...SCALE_integrated_state.astype(dtype)` over the `ODETerm`, and because
    the wrapper is the argument being differentiated, that offset is a *tracer*. The
    Python branch in the value unscale is therefore unavailable and `yy * SCALE` selects
    at runtime, whose reverse mode is a masked sum. Equality is asserted to a tight
    tolerance rather than exactly: a zero-offset `AffineScaler` carries an extra array
    leaf, so the two wrappers are different pytrees and fusion may legitimately reorder.

    Know what this test does NOT do. With `b == 0` the affine and linear paths are
    mathematically identical, so it cannot detect offset-semantics faults. An
    injected offset leak into the composed scaler's `scale_derivative` passes here;
    `test_affine_offset_cancels_from_gradient_through_solve` catches it instead. This
    one exists for coverage of the traced path and as a regression net against gross
    breakage of the drop-in claim. Not the guard for offset correctness.
    """
    linear, process_data = _build_wrapper_and_process()
    RMC_scaler = linear.reaction_module.SCALE_modeled_RMCs
    assert RMC_scaler.scale.shape == (1,), (
        "must patch a non-zero-width axis, else the comparison is vacuous"
    )
    affine = eqx.tree_at(
        lambda w: w.reaction_module.SCALE_modeled_RMCs,
        linear,
        AffineScaler(
            scale=RMC_scaler.scale,
            offset=jnp.zeros_like(RMC_scaler.scale),
        ),
    )
    t_measured = clamp_padded_time_rows(
        process_data.t_measured[None, :],
        jnp.asarray([process_data.n_measured], dtype=jnp.int32),
    )[0]

    def loss(wrapper):
        target_scaler = wrapper.reaction_module.SCALE_state[
            wrapper.target_state_indices
        ]
        total, _ = _measurement_loss(
            wrapper,
            t_measured=t_measured,
            SCL_target_measured=process_data.y_measured / target_scaler,
            mask_measured=process_data.mask_measured,
            n_measured=process_data.n_measured,
            RAW_y0_measured=process_data.y0_measured,
            jump_ts=process_data.controls.active_jump_ts,
            max_solver_steps=100_000,
            solver_rtol=1e-7,
            solver_atol=1e-9,
        )
        return total

    linear_loss, linear_grad = eqx.filter_value_and_grad(loss)(linear)
    affine_loss, affine_grad = eqx.filter_value_and_grad(loss)(affine)

    assert float(linear_loss) > 0.0  # nontrivial: prediction != measurements
    assert jnp.allclose(affine_loss, linear_loss, rtol=1e-6, atol=1e-8)

    # Compare the trainable leaves the two wrappers share. Gradients must be non-zero,
    # or a masked-sum regression on the select would pass unnoticed.
    for name in ("weight", "bias"):
        linear_leaf = getattr(linear_grad.reaction_module.model, name)
        affine_leaf = getattr(affine_grad.reaction_module.model, name)
        assert bool(jnp.any(linear_leaf != 0.0)), (
            f"gradient w.r.t. model.{name} is all zero -- comparison would be vacuous"
        )
        assert jnp.allclose(affine_leaf, linear_leaf, rtol=1e-6, atol=1e-8), (
            f"zero-offset affine changed the gradient w.r.t. model.{name}"
        )


def test_affine_offset_cancels_from_gradient_through_solve():
    """A non-zero offset shifts the state coordinate origin but must not reach any
    derivative: `d((RAW - b)/s)/dt == (dRAW/dt)/s`, independent of `b`.

    This is the load-bearing half of the pair. Unlike the zero-offset drop-in test, a
    non-zero `b` makes the invariant substantive: any leak of the offset into the
    derivative ops, or into the reverse-mode residual of the traced vector field, moves
    this gradient. `_AuxReactionModule` emits `q=0` regardless of its input, so the
    physical trajectory is genuinely offset-independent and the gradient w.r.t. the raw
    initial condition must match exactly the same way the forward loss does.
    """
    linear, process_data = _build_wrapper_and_process(_AuxReactionModule)
    RMC_scaler = linear.reaction_module.SCALE_modeled_RMCs
    affine = eqx.tree_at(
        lambda w: w.reaction_module.SCALE_modeled_RMCs,
        linear,
        AffineScaler(
            scale=RMC_scaler.scale,
            offset=jnp.asarray([10.0], dtype=RMC_scaler.scale.dtype),
        ),
    )
    t_measured = clamp_padded_time_rows(
        process_data.t_measured[None, :],
        jnp.asarray([process_data.n_measured], dtype=jnp.int32),
    )[0]

    def loss(wrapper, RAW_y0_measured):
        target_scaler = wrapper.reaction_module.SCALE_state[
            wrapper.target_state_indices
        ]
        total, _ = _measurement_loss(
            wrapper,
            t_measured=t_measured,
            SCL_target_measured=process_data.y_measured / target_scaler,
            mask_measured=process_data.mask_measured,
            n_measured=process_data.n_measured,
            RAW_y0_measured=RAW_y0_measured,
            jump_ts=process_data.controls.active_jump_ts,
            max_solver_steps=100_000,
            solver_rtol=1e-7,
            solver_atol=1e-9,
        )
        return total

    # Differentiate w.r.t. the raw initial condition: a leaf both wrappers share
    # identically, so the two gradients are directly comparable pytrees.
    grad_fn = jax.grad(loss, argnums=1)
    linear_grad = grad_fn(linear, process_data.y0_measured)
    affine_grad = grad_fn(affine, process_data.y0_measured)

    assert bool(jnp.any(linear_grad != 0.0)), (
        "gradient w.r.t. RAW y0 is all zero -- comparison would be vacuous"
    )
    assert jnp.allclose(affine_grad, linear_grad, rtol=1e-6, atol=1e-8), (
        "a non-zero affine offset leaked into a derivative"
    )


def _make_feed_cin_collection(*, cin_lo: float, cin_hi: float) -> BioProcessCollection:
    """Two processes identical except for their controlled feed's composition."""

    def _process(name: str, feed_biomass: float) -> BioProcess:
        return BioProcess(
            metadata=BioProcessMetadata(name=name, process_type="fed_batch"),
            time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
            volume=Volume(
                initial_volume=1.0,
                unit="L",
                volume_changes={
                    "feed_A": Inflow(
                        name="feed_A",
                        unit="L",
                        is_controlled=True,
                        is_continuous=True,
                        values=TimeSeries(
                            times=jnp.asarray([0.0, 1.0, 2.0]),
                            values=jnp.asarray([0.0, 0.2, 0.4]),
                        ),
                        feed_medium=FeedMedium(
                            name="feed",
                            density=1.0,
                            density_unit="kg/L",
                            components={
                                "biomass": FeedMediumComponent(
                                    name="biomass",
                                    unit="g/L",
                                    concentration=StaticVariable(feed_biomass),
                                    is_controlled=False,
                                )
                            },
                        ),
                    ),
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
                    ),
                },
            ),
            process_variables={},
        )

    return BioProcessCollection(
        processes={"p_lo": _process("p_lo", cin_lo), "p_hi": _process("p_hi", cin_hi)},
        metadata={},
    )


def test_simulate_measurement_states_uses_the_simulated_processs_own_Cin():
    """The template's baked ``Cin`` belongs to its reference process, not to the
    process being simulated. ``simulate_measurement_states`` must substitute the
    per-process feed composition alongside the per-process controls, or it
    silently solves ``p_hi`` with ``p_lo``'s feed."""
    from hybrax.train.harness import _build_template_wrapper

    collection = _make_feed_cin_collection(cin_lo=2.0, cin_hi=4.0)
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    # The two processes differ only in their feed concentration.
    np.testing.assert_allclose(
        np.asarray(store.Cin_controlled_Inflows), [[[2.0]], [[4.0]]]
    )

    def template(reference: str) -> HybridOdeWrapper:
        rhs_ode = build_rhs_ode(collection.processes[reference])
        controls = store.get_process(reference).controls
        return _build_template_wrapper(
            store,
            reaction_module=_LinearReactionModule(
                **_unit_scale_kwargs_for(rhs_ode, controls)
            ),
            selected_processes=(reference,),
            loss_module=DefaultLossModule(target_names=["biomass"]),
        )

    p_hi = store.get_process("p_hi")
    from_lo_template = np.asarray(simulate_measurement_states(template("p_lo"), p_hi))
    from_hi_template = np.asarray(simulate_measurement_states(template("p_hi"), p_hi))

    # Simulating p_hi must not depend on which process supplied the template.
    np.testing.assert_allclose(from_lo_template, from_hi_template, rtol=1e-9, atol=1e-9)

    # Teeth: with everything but Cin held equal, the feed composition really does
    # move the biomass trajectory, so the equality above is not vacuous.
    p_lo = store.get_process("p_lo")
    lo_states = np.asarray(simulate_measurement_states(template("p_lo"), p_lo))
    biomass_hi = from_hi_template[-1, 0]
    biomass_lo = lo_states[-1, 0]
    assert abs(biomass_hi - biomass_lo) / abs(biomass_lo) > 0.01
