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
from bp_format.mechanistic import build_rhs_ode

import bp_train.trainer as trainer_module
from bp_train.physical_solve import solve_physical_states
from bp_train.model_api import (
    ReactionOutputs,
    UserReactionModule,
    frozen_field,
    trainable_field,
)
from bp_train.defaults import DefaultLossModule
from bp_train.harness import summarize_train_step_input_signature
from bp_train.trainer import (
    build_batched_loss_fn,
    clamp_padded_time_rows,
    evaluate_sample_with_loss_module,
)
from bp_train.training_data import TrainingDataStore
from bp_train.wrapper import HybridOdeWrapper, SaveOutputs


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
        self.non_model_bias = jnp.asarray([0.05], dtype=jnp.float32)

    def __call__(self, t, inputs):
        del t
        SCL_modeled_RMCs = inputs.SCL_modeled_RMCs
        rate = self.model(SCL_modeled_RMCs)[0] + self.non_model_bias[0]
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=jnp.asarray(
                [rate], dtype=SCL_modeled_RMCs.dtype
            ),
            SCL_modeled_FVCs_rates=jnp.zeros((0,), dtype=SCL_modeled_RMCs.dtype),
        )


class _AuxReactionModule(UserReactionModule):
    def __init__(self, **scale_kwargs):
        super().__init__(**scale_kwargs)

    def __call__(self, t, inputs):
        SCL_modeled_RMCs = inputs.SCL_modeled_RMCs
        rate = jnp.asarray([0.0], dtype=SCL_modeled_RMCs.dtype)
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=rate,
            SCL_modeled_FVCs_rates=jnp.zeros((0,), dtype=SCL_modeled_RMCs.dtype),
            auxiliary={
                "mu_raw": t,
                "latent_pair": jnp.asarray(
                    [t, SCL_modeled_RMCs[0]], dtype=SCL_modeled_RMCs.dtype
                ),
            },
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
                ),
            },
        ),
        process_variables={},
    )
    return BioProcessCollection(processes={"p1": p1, "p2": p2}, metadata={})


def _unit_scale_kwargs_for(rhs_ode, controls) -> dict[str, jnp.ndarray]:
    f32 = jnp.float32
    n_RMCs = len(rhs_ode.name_modeled_RMCs)
    n_FVCs = len(rhs_ode.name_modeled_FVCs)
    n_rates = len(rhs_ode.name_modeled_rates)
    n_FVC = len(controls.name_controlled_FVCs)
    n_PV = len(controls.name_controlled_PVs)
    n_bolus = len(controls.name_extras) - 1
    return {
        "SCALE_modeled_RMCs": jnp.ones(n_RMCs, dtype=f32),
        "SCALE_V_in_cumulative": jnp.asarray(1.0, dtype=f32),
        "SCALE_modeled_FVCs_cumulative": jnp.ones(n_FVCs, dtype=f32),
        "SCALE_controlled_FVCs_cumulative": jnp.ones(n_FVC, dtype=f32),
        "SCALE_controlled_FVCs_rates": jnp.ones(n_FVC, dtype=f32),
        "SCALE_controlled_FVCs_Cin": jnp.ones((n_FVC, n_RMCs), dtype=f32),
        "SCALE_controlled_FVCs_bolus_rates": jnp.ones(n_bolus, dtype=f32),
        "SCALE_controlled_PVs": jnp.ones(n_PV, dtype=f32),
        "SCALE_modeled_FVCs_Cin": jnp.ones((n_FVCs, n_RMCs), dtype=f32),
        "SCALE_modeled_BiologicalOde_rates": jnp.ones(n_rates, dtype=f32),
        "SCALE_modeled_FVCs_rates": jnp.ones(n_FVCs, dtype=f32),
    }


def _build_wrapper_and_process(module_cls=_LinearReactionModule):
    from bp_format.mechanistic import build_rhs_ode as _build_rhs_ode

    collection = _make_two_process_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    process_data = store.get_process("p2")
    rhs_ode = _build_rhs_ode(collection.processes["p2"])
    scale_kwargs = _unit_scale_kwargs_for(rhs_ode, process_data.controls)
    wrapper = HybridOdeWrapper.from_process(
        reaction_module=module_cls(**scale_kwargs),
        process=collection.processes["p2"],
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
        dtype=jnp.float32,
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
        jump_ts=process_data.controls.active_step_ts,
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
        jump_ts=process_data.controls.active_step_ts,
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
        return SaveOutputs(
            SCL_states=states,
            RAW_V_export=states[:, len(wrapper_arg.modeled_RMC_names)],
            RAW_V=states[:, len(wrapper_arg.modeled_RMC_names)],
            RAW_modeled_BiologicalOde_rates=jnp.zeros(
                (n_rows, len(wrapper_arg.modeled_RMC_names)),
                dtype=states.dtype,
            ),
            RAW_modeled_FVCs_rates=jnp.zeros(
                (n_rows, len(wrapper_arg.modeled_FVC_names)),
                dtype=states.dtype,
            ),
            auxiliary=None,
        )

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
        jump_ts=process_data.controls.active_step_ts,
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
        jump_ts=process_data.controls.active_step_ts,
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
        jump_ts=process_data.controls.active_step_ts,
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
        jump_ts=process_data.controls.active_step_ts,
        max_solver_steps=100_000,
        solver_rtol=1e-5,
        solver_atol=1e-7,
        step=42,
    )

    assert int(result.step) == 42
    assert jnp.issubdtype(result.step.dtype, jnp.integer)


def _build_batched_setup():
    collection = _make_two_process_collection()
    store = TrainingDataStore.from_collection(
        collection,
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )
    process_data = store.get_process("p2")
    rhs_p2 = build_rhs_ode(collection.processes["p2"])
    wrapper = HybridOdeWrapper.from_process(
        reaction_module=_LinearReactionModule(
            **_unit_scale_kwargs_for(rhs_p2, process_data.controls)
        ),
        process=collection.processes["p2"],
        controls=process_data.controls,
        loss_module=DefaultLossModule(target_names=["biomass"]),
    )
    batch = store.gather_batch(jnp.asarray([1], dtype=jnp.int32))
    batch_controls = store.controls_store.as_batch_controls()
    rhs_by_process = [
        build_rhs_ode(collection.processes[name]) for name in store.process_order
    ]
    batched_cin = jnp.stack([rhs.Cin_controlled_FVCs for rhs in rhs_by_process], axis=0)
    batched_cin_modeled = jnp.stack(
        [rhs.Cin_modeled_FVCs for rhs in rhs_by_process], axis=0
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
        batch_controls,
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
        batch_controls,
        batched_cin,
        batched_cin_modeled,
        None,
        max_solver_steps=100_000,
        solver_rtol=1e-5,
        solver_atol=1e-7,
    )
    jump_ts_rows = jnp.zeros((1, 1), dtype=jnp.float32)
    mean_total_present, *_ = batched_loss_fn(
        wrapper,
        batch,
        batch_controls,
        batched_cin,
        batched_cin_modeled,
        jump_ts_rows,
        max_solver_steps=100_000,
        solver_rtol=1e-5,
        solver_atol=1e-7,
    )
    assert jnp.isfinite(mean_total_none)
    assert jnp.isfinite(mean_total_present)


# ---------------------------------------------------------------------------
# Dense-grid helpers (bp_train/dense.py)
# ---------------------------------------------------------------------------


def test_build_union_time_grid_sorts_and_indexes_correctly():
    from bp_train.dense import build_union_time_grid

    t_meas = jnp.asarray([0.0, 1.0, 4.0], dtype=jnp.float32)
    t_eval, sample_idx, dense_t, dense_idx, _pred_t, _pred_idx = build_union_time_grid(
        t_meas, n_measured=3, n_dense=3
    )
    # dense linspace covers the (active) measurement span.
    assert jnp.allclose(dense_t, jnp.asarray([0.0, 2.0, 4.0], dtype=jnp.float32))
    # t_eval is the sorted concat.
    assert jnp.all(jnp.diff(t_eval) >= 0)
    assert t_eval.shape[0] == 6
    # Round-trip: gathering t_eval by the index arrays returns the originals.
    assert jnp.allclose(t_eval[sample_idx], t_meas)
    assert jnp.allclose(t_eval[dense_idx], dense_t)


def test_dense_point_mask_handles_jump_ts_and_none():
    from bp_train.dense import dense_point_mask_away_from_jumps

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
    from bp_train.dense import dense_triple_mask_away_from_jumps

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
