"""Continuous-flow preflight against analytic increments and missed feed pulses."""

from __future__ import annotations

from dataclasses import replace

import diffrax
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from hybrax.format.dataclasses import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    DiscreteEvents,
    FeedMedium,
    Inflow,
    Outflow,
    ProcessVariable,
    ReactorMedium,
    ReactorMediumComponent,
    TimeAxis,
    TimeSeries,
    Volume,
)
from hybrax.train import preflight
from hybrax.train.defaults import default_build_reaction_module
from hybrax.train.harness import TrainHarnessConfig, train_collection
from hybrax.train.training_data import TrainingDataStore

SETTINGS = {"max_steps": 2048, "rtol": 1e-6, "atol": 1e-8, "use_jump_ts": True}


def _flow(name, series, flow_type=Inflow):
    extra = {"feed_medium": FeedMedium(name="medium")} if flow_type is Inflow else {}
    return flow_type(
        name=name,
        unit="L",
        is_controlled=True,
        is_continuous=True,
        values=series,
        **extra,
    )


def _process(name, flows, *, measured=(0.0, 2.0), jumps=()):
    return BioProcess(
        metadata=BioProcessMetadata(name=name, process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=2.0, time_reference="start"),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes=flows),
        reactor_medium=ReactorMedium(
            name="medium",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=measured, values=np.ones(len(measured))
                    ),
                )
            },
        ),
        process_variables={},
        discrete_events=DiscreteEvents(times=jnp.asarray(jumps)) if jumps else None,
    )


def _store(*processes):
    return TrainingDataStore.from_collection(
        BioProcessCollection(
            metadata={}, processes={p.metadata.name: p for p in processes}
        ),
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )


def _pulse(sign=1.0):
    return TimeSeries(
        times=[0.0, 0.4, 0.402, 2.0],
        values=sign * np.asarray([0.0, 0.0, 0.01, 0.01]),
    )


@pytest.mark.parametrize("flow_type,sign", [(Inflow, 1.0), (Outflow, -1.0)])
def test_missed_pulse_is_rejected(flow_type, sign):
    store = _store(_process("pulse", {"flow": _flow("flow", _pulse(sign), flow_type)}))
    with pytest.raises(ValueError, match="process 'pulse'.*expected=.*0.01"):
        preflight.check_continuous_flow_integration(store, ("pulse",), **SETTINGS)


def test_opposite_missed_pulses_cannot_cancel():
    store = _store(
        _process(
            "pulse",
            {
                "feed": _flow("feed", _pulse()),
                "bleed": _flow("bleed", _pulse(-1), Outflow),
            },
        )
    )
    with pytest.raises(ValueError, match="process 'pulse'.*expected=.*0.01"):
        preflight.check_continuous_flow_integration(store, ("pulse",), **SETTINGS)


def test_explicit_jump_times_are_used_only_when_enabled():
    store = _store(
        _process("pulse", {"feed": _flow("feed", _pulse())}, jumps=(0.4, 0.402))
    )
    preflight.check_continuous_flow_integration(store, ("pulse",), **SETTINGS)
    with pytest.raises(ValueError, match="flow 'feed'.*integrated=0"):
        preflight.check_continuous_flow_integration(
            store, ("pulse",), **{**SETTINGS, "use_jump_ts": False}
        )


def test_scalar_solver_uses_configured_tolerances(monkeypatch):
    store = _store(
        _process(
            "linear",
            {"feed": _flow("feed", TimeSeries(times=[0.0, 2.0], values=[0.0, 0.1]))},
        )
    )
    observed = []
    original = diffrax.PIDController

    def record_controller(*args, **kwargs):
        observed.append((kwargs["rtol"], kwargs["atol"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(diffrax, "PIDController", record_controller)
    monkeypatch.setattr(
        preflight,
        "_integrate_flow",
        eqx.filter_jit(preflight._integrate_flow.__wrapped__),
    )
    configured = [(2e-5, 3e-7), (4e-8, 5e-10)]
    for rtol, atol in configured:
        preflight.check_continuous_flow_integration(
            store, ("linear",), **{**SETTINGS, "rtol": rtol, "atol": atol}
        )
    assert observed == configured


def test_selected_process_uses_its_own_jump_row():
    first = _process("first", {"feed": _flow("feed", _pulse())})
    second = _process("second", {"feed": _flow("feed", _pulse())}, jumps=(0.4, 0.402))
    store = _store(first, second)
    preflight.check_continuous_flow_integration(store, ("second",), **SETTINGS)
    with pytest.raises(ValueError, match="process 'first'.*integrated=0"):
        preflight.check_continuous_flow_integration(store, ("first",), **SETTINGS)


def test_different_jump_counts_reuse_solver_compilation(monkeypatch):
    processes = [
        _process(
            f"linear_{count}",
            {"feed": _flow("feed", TimeSeries(times=[0.0, 2.0], values=[0.0, 0.1]))},
            jumps=(0.4, 0.8, 1.2)[:count],
        )
        for count in range(4)
    ]
    store = _store(*processes)
    traces = []
    original = diffrax.diffeqsolve

    def record_trace(*args, **kwargs):
        traces.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(diffrax, "diffeqsolve", record_trace)
    # A fresh JIT wrapper makes the compilation counter independent of other tests.
    monkeypatch.setattr(
        preflight,
        "_integrate_flow",
        eqx.filter_jit(preflight._integrate_flow.__wrapped__),
    )
    preflight.check_continuous_flow_integration(
        store, tuple(p.metadata.name for p in processes), **SETTINGS
    )
    assert len(traces) == 1


def test_smooth_spline_pulse_is_also_checked():
    # Cubic smoothstep: continuous feed rate, zero outside the narrow interval.
    width = 1 / 512
    amount = 1 / 64
    times = [0.0, 0.5, 0.5 + width, 2.0]
    series = TimeSeries(
        times=times,
        values=[0.0, 0.0, amount, amount],
        breaks=times,
        coeffs=[
            [0, 0, 0, 0],
            [0, 0, 3 * amount / width**2, -2 * amount / width**3],
            [amount, 0, 0, 0],
        ],
        segment_start_piece_idx=[0],
    )
    store = _store(_process("spline", {"feed": _flow("feed", series)}))
    with pytest.raises(ValueError, match="process 'spline'.*expected=0.015625"):
        preflight.check_continuous_flow_integration(store, ("spline",), **SETTINGS)


def test_mixed_controls_offsets_subspan_and_padded_process(monkeypatch):
    # Inflow F=3+t^3; signed outflow O=-5-2t. Over [.5,1.5], increments are
    # +3.25 and -2, regardless of offsets or the other process's padded arrays.
    inflow = TimeSeries(
        times=[0.0, 2.0],
        values=[3.0, 11.0],
        breaks=[0.0, 2.0],
        coeffs=[[3.0, 0.0, 0.0, 1.0]],
        segment_start_piece_idx=[0],
    )
    outflow = TimeSeries(times=[0.0, 2.0], values=[-5.0, -9.0])
    short = _process(
        "short",
        {"feed": _flow("feed", inflow), "bleed": _flow("bleed", outflow, Outflow)},
        measured=(0.5, 1.5),
    )
    long = _process(
        "long",
        {
            "feed": _flow("feed", inflow),
            "bleed": _flow(
                "bleed",
                TimeSeries(times=[0.0, 1.0, 2.0], values=[-5.0, -7.0, -9.0]),
                Outflow,
            ),
        },
        measured=(0.0, 1.0, 2.0),
    )
    for process in (short, long):
        process.process_variables["temperature"] = ProcessVariable(
            name="temperature",
            unit="C",
            is_controlled=True,
            values=TimeSeries(times=[0.0, 2.0], values=[30.0, 32.0]),
        )
    store = _store(short, long)
    observed = []
    original = preflight._integrate_flow

    def record(*args, **kwargs):
        integrated, expected, result = original(*args, **kwargs)
        observed.append((float(integrated), float(expected)))
        return integrated, expected, result

    monkeypatch.setattr(preflight, "_integrate_flow", record)
    preflight.check_continuous_flow_integration(store, ("short",), **SETTINGS)
    np.testing.assert_allclose(observed, [[3.25, 3.25], [-2.0, -2.0]], atol=1e-10)


def test_step_budget_failure_rejected_even_when_endpoint_matches():
    store = _store(
        _process(
            "zero",
            {"feed": _flow("feed", TimeSeries(times=[0.0, 2.0], values=[4.0, 4.0]))},
        )
    )
    with pytest.raises(
        ValueError, match="process 'zero'.*integrated=0, expected=0.*maximum"
    ):
        preflight.check_continuous_flow_integration(
            store, ("zero",), **{**SETTINGS, "max_steps": 1}
        )


def test_zero_feed_passes_without_absolute_allowance():
    store = _store(
        _process(
            "zero",
            {"feed": _flow("feed", TimeSeries(times=[0.0, 2.0], values=[4.0, 4.0]))},
        )
    )
    preflight.check_continuous_flow_integration(store, ("zero",), **SETTINGS)


@pytest.mark.parametrize("expected", [1.0, -1.0, 2**-30])
def test_relative_allowance_is_configurable_and_independent_of_solver(
    monkeypatch, expected
):
    store = _store(_process("flow", {"feed": _flow("feed", _pulse())}))
    monkeypatch.setattr(
        preflight,
        "_integrate_flow",
        lambda *args, **kwargs: (
            expected * 1.0005,
            expected,
            diffrax.RESULTS.successful,
        ),
    )
    # 0.05% error: inside default 0.1%, outside custom 0.01%, at any flow scale.
    for solver_rtol, solver_atol in [(1e-2, 1.0), (1e-10, 1e-12)]:
        settings = {**SETTINGS, "rtol": solver_rtol, "atol": solver_atol}
        preflight.check_continuous_flow_integration(store, ("flow",), **settings)
        with pytest.raises(ValueError, match="Continuous-flow preflight failed"):
            preflight.check_continuous_flow_integration(
                store,
                ("flow",),
                **settings,
                flow_integral_relative_tolerance=1e-4,
            )


@pytest.mark.parametrize("integrated", [0.0, 1e-12])
def test_zero_expected_increment_requires_exact_agreement(monkeypatch, integrated):
    store = _store(_process("zero", {"feed": _flow("feed", _pulse())}))
    monkeypatch.setattr(
        preflight,
        "_integrate_flow",
        lambda *args, **kwargs: (integrated, 0.0, diffrax.RESULTS.successful),
    )
    if integrated == 0.0:
        preflight.check_continuous_flow_integration(store, ("zero",), **SETTINGS)
    else:
        with pytest.raises(ValueError, match="tolerance=0"):
            preflight.check_continuous_flow_integration(store, ("zero",), **SETTINGS)


def test_only_selected_processes_are_checked():
    good = _process(
        "good", {"feed": _flow("feed", TimeSeries(times=[0.0, 2.0], values=[0.0, 0.1]))}
    )
    bad = _process("bad", {"feed": _flow("feed", _pulse())})
    store = _store(good, bad)
    preflight.check_continuous_flow_integration(store, ("good",), **SETTINGS)
    with pytest.raises(ValueError, match="process 'bad'"):
        preflight.check_continuous_flow_integration(store, ("bad",), **SETTINGS)


def test_single_measurement_has_no_flow_interval(monkeypatch):
    store = _store(
        _process("single", {"feed": _flow("feed", _pulse())}, measured=(1.0,))
    )

    def unexpected_solve(*args, **kwargs):
        raise AssertionError("A zero-span interval must not run the solver")

    monkeypatch.setattr(preflight, "_integrate_flow", unexpected_solve)
    preflight.check_continuous_flow_integration(store, ("single",), **SETTINGS)


def test_training_gate_precedes_warmup_and_uses_configured_jump_flag(monkeypatch):
    process = _process("pulse", {"feed": _flow("feed", _pulse())}, jumps=(0.4, 0.402))
    store = _store(process)
    module = default_build_reaction_module(
        target_names=["biomass"],
        process_names=["pulse"],
        seed=0,
        config=TrainHarnessConfig(),
        training_parent_collection=BioProcessCollection(
            metadata={}, processes={"pulse": process}
        ),
    )

    def warmup_sentinel(*args, **kwargs):
        raise AssertionError("Reached training warmup")

    monkeypatch.setattr(TrainingDataStore, "gather_batch", warmup_sentinel)
    config = TrainHarnessConfig(
        process_names=("pulse",),
        epochs=1,
        batch_size=1,
        solver_rtol=SETTINGS["rtol"],
        solver_atol=SETTINGS["atol"],
        solver_use_jump_ts=False,
    )
    with pytest.raises(ValueError, match="Continuous-flow preflight failed"):
        train_collection(store, reaction_module=module, config=config)
    with pytest.raises(ValueError, match="solver_result=.*maximum"):
        train_collection(
            store,
            reaction_module=module,
            config=replace(config, solver_use_jump_ts=True, solver_max_steps=1),
        )
    with pytest.raises(AssertionError, match="Reached training warmup"):
        train_collection(
            store,
            reaction_module=module,
            config=replace(config, solver_use_jump_ts=True),
        )
    monkeypatch.setattr(
        preflight,
        "_integrate_flow",
        lambda *args, **kwargs: (1.005, 1.0, diffrax.RESULTS.successful),
    )
    with pytest.raises(ValueError, match="Continuous-flow preflight failed"):
        train_collection(store, reaction_module=module, config=config)
    with pytest.raises(AssertionError, match="Reached training warmup"):
        train_collection(
            store,
            reaction_module=module,
            config=replace(config, flow_integral_relative_tolerance=1e-2),
        )
    for invalid in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="flow_integral_relative_tolerance"):
            train_collection(
                store,
                reaction_module=module,
                config=replace(config, flow_integral_relative_tolerance=invalid),
            )


@pytest.mark.parametrize(
    "integrated,expected",
    [
        (np.nan, 0.1),
        (np.inf, 0.1),
        (-np.inf, 0.1),
        (0.1, np.nan),
        (np.inf, np.inf),
        (0.1, np.inf),
    ],
)
def test_nonfinite_integral_cannot_pass(monkeypatch, integrated, expected):
    store = _store(
        _process(
            "finite",
            {"feed": _flow("feed", TimeSeries(times=[0.0, 2.0], values=[0.0, 0.1]))},
        )
    )
    monkeypatch.setattr(
        preflight,
        "_integrate_flow",
        lambda *args, **kwargs: (integrated, expected, diffrax.RESULTS.successful),
    )
    with pytest.raises(ValueError, match="process 'finite'"):
        preflight.check_continuous_flow_integration(store, ("finite",), **SETTINGS)
