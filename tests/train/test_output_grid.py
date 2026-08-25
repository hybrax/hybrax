"""Tier-B output grid: values are read with ``SaveAt(ts=...)`` inside a segment.

Output times (measurements, and any dense/prediction grid spliced in by
``dense.build_union_time_grid``) are NOT segment boundaries — only bolus and sample
events are, because only they jump the state. These tests pin the guarantees that makes
possible, and the per-segment output WINDOW that keeps it fast:

  1. the window changes no value — windowed and whole-grid readouts are bitwise equal;
  2. the window bound derived at prepare time really does cover every inter-event gap;
  3. an undersized window is a loud ``output_overflow``, never a silently dropped row;
  4. a healthy solve returns no non-finite output row;
  5. asking for a finer output grid does not change the answer at the measurement times
     (it used to: every extra point was another segment boundary);
  6. a process with zero bolus AND zero sample events solves (the tier-A preset array is
     then empty, which used to be a hard ``argmin`` error).
"""

from __future__ import annotations

import math

import diffrax
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
    ProcessVariable,
    ReactorMediumComponent,
    Outflow,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)
from hybrax.train.diffrax_callbacks import (
    PresetTimeCallback,
    diffeqsolve_with_callbacks,
)

import hybrax.train.controls_store as controls_store_module
from hybrax.train.controls_store import ControlsStore, _output_window_bounds
from hybrax.train.dense import build_union_time_grid
from hybrax.train.physical_solve import _output_window, solve_physical_states
from hybrax.train.training_data import TrainingDataStore

from stateful_helpers import build_stateful_wrapper, default_stateful_scale_kwargs

T_END = 240.0


def _process(name: str, *, n_sample: int, n_bolus: int, n_extra_meas: int = 0):
    """Fed-batch process with a controllable number of sample/bolus events."""
    sample_ts = np.linspace(8.0, T_END - 8.0, n_sample) if n_sample else np.empty(0)
    bolus_ts = (
        np.linspace(12.0, T_END - 12.0, n_bolus) + 0.37 if n_bolus else np.empty(0)
    )
    volume_changes = {}
    if n_sample:
        volume_changes["sampling"] = Outflow(
            name="sampling",
            unit="L",
            is_controlled=False,
            is_continuous=False,
            values=TimeSeries(
                times=jnp.asarray(sample_ts),
                values=jnp.asarray(-0.002 * np.ones(n_sample)),
            ),
        )
    if n_bolus:
        volume_changes["bolus"] = Inflow(
            name="bolus",
            unit="L",
            is_controlled=True,
            is_continuous=False,
            values=TimeSeries(
                times=jnp.asarray(bolus_ts),
                values=jnp.asarray(0.004 * np.ones(n_bolus)),
            ),
            feed_medium=FeedMedium(
                name="feed",
                density=1.0,
                density_unit="kg/L",
                components={
                    "biomass": FeedMediumComponent(
                        name="biomass",
                        unit="g/L",
                        concentration=StaticVariable(0.0),
                        is_controlled=False,
                    )
                },
            ),
        )
    extra = np.linspace(1.0, T_END - 1.0, n_extra_meas) if n_extra_meas else np.empty(0)
    meas_ts = np.unique(np.concatenate([[0.0], sample_ts, extra, [T_END]]))
    return BioProcess(
        metadata=BioProcessMetadata(name=name, process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=T_END, time_reference="start"),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes=volume_changes),
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=TimeSeries(
                        times=jnp.asarray(meas_ts),
                        values=jnp.asarray(0.5 * np.exp(0.004 * meas_ts)),
                    ),
                )
            },
        ),
        process_variables={},
    ), jnp.asarray(meas_ts, dtype=jnp.float64)


def _wrapper(process):
    from hybrax.train.defaults import DefaultStatefulReactionModule

    module = DefaultStatefulReactionModule(
        key=jax.random.key(0),
        n_latent=1,
        **default_stateful_scale_kwargs(n_controlled_inflows=0),
    )
    return build_stateful_wrapper(process, module)


def _solve(wrapper, t_eval, n_linspace=0):
    return solve_physical_states(
        wrapper,
        t_eval=t_eval,
        n_measured=t_eval.shape[0],
        RAW_y0=jnp.asarray([0.5, 1.0], dtype=jnp.float64),
        max_steps=100_000,
        rtol=1e-8,
        atol=1e-10,
        n_linspace=n_linspace,
    )


# --------------------------------------------------------------------------------
# 1 + 3. The window changes no value, and an undersized one is loud.
# --------------------------------------------------------------------------------


def _windowed_readout(window, *, n_out=61):
    """Bare solver: identity affects at 3 events, one decaying state."""
    grid = jnp.linspace(0.0, 10.0, n_out, dtype=jnp.float64)
    return diffeqsolve_with_callbacks(
        diffrax.ODETerm(lambda t, y, args: -0.3 * y),
        diffrax.Tsit5(),
        t0=0.0,
        t1=10.0,
        dt0=1e-2,
        y0=jnp.ones(2, dtype=jnp.float64),
        callbacks=PresetTimeCallback(
            times=jnp.asarray([2.0, 4.0, 6.0], dtype=jnp.float64),
            affect_fn=lambda y, t, args, i: y * 0.5,
        ),
        max_events=4,
        output_times=grid,
        output_window=window,
        stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-12),
        max_steps=100_000,
    )


def _widest_gap_points(n_out=61):
    """True maximum number of grid points in any inter-event gap, by brute force."""
    grid = np.linspace(0.0, 10.0, n_out)
    boundaries = np.asarray([0.0, 2.0, 4.0, 6.0, 10.0])
    return max(
        int(((grid > lo) & (grid <= hi)).sum())
        for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True)
    )


def test_windowed_matches_unwindowed():
    """The strongest statement: the window is a pure work reduction. Whatever it is set
    to, as long as it is big enough, the readout is BITWISE the whole-grid readout."""
    reference = _windowed_readout(None)
    assert not bool(reference.output_overflow)
    needed = _widest_gap_points()
    for window in (needed, needed + 8, 61):
        sol = _windowed_readout(window)
        assert not bool(sol.output_overflow), f"window={window} should be sufficient"
        assert bool(jnp.array_equal(sol.output_states, reference.output_states)), (
            f"window={window} changed the readout"
        )


def test_output_overflow_fires_when_the_window_is_one_too_small():
    """An undersized window would silently leave the excess rows at ``inf``; the flag is
    what turns that into a hard error instead."""
    needed = _widest_gap_points()
    assert not bool(_windowed_readout(needed).output_overflow)
    sol = _windowed_readout(needed - 1)
    assert bool(sol.output_overflow), "an undersized window must be reported"
    assert bool(jnp.any(~jnp.isfinite(sol.output_states))), (
        "and the dropped rows are exactly what the flag is protecting against"
    )


def test_solve_raises_when_the_grid_violates_the_window_precondition():
    """``solve_physical_states`` promotes the flag to a raise, so an undersized window
    can never reach a prediction file.

    The window is sized from a RELATIVE inter-event gap fraction measured over the
    process's measurement window (``_output_window_bounds``). A caller that hand-rolls a
    strictly narrower ``t_eval`` shrinks the denominator, inflating the true fraction
    past the bound. No tight bound survives an arbitrary sub-window, so this is a
    documented precondition — and this test pins that violating it is LOUD rather
    than silently dropping output rows."""
    # 8 evenly spaced samples over [0, 240] -> widest gap is 1/7.5 of the horizon,
    # so the bound is sized for a small fraction. Solving over [0, 45] leaves one 32 h
    # gap spanning ~70% of that window, far past the bound.
    process, _ = _process("p", n_sample=8, n_bolus=0)
    wrapper = _wrapper(process)
    narrow = jnp.linspace(0.0, 45.0, 400, dtype=jnp.float64)
    with pytest.raises(Exception, match="output window too small"):
        jax.block_until_ready(_solve(wrapper, narrow, n_linspace=400))


# --------------------------------------------------------------------------------
# 2. The prepare-time bound really covers every gap.
# --------------------------------------------------------------------------------


def test_controls_store_reuses_rhs_names_for_output_bounds(monkeypatch):
    processes = {
        "a": _process("a", n_sample=2, n_bolus=1)[0],
        "b": _process("b", n_sample=3, n_bolus=2, n_extra_meas=4)[0],
    }
    collection = BioProcessCollection(processes=processes, metadata={})
    expected_bounds = _output_window_bounds(collection, list(processes))
    build_calls = []
    original_build_rhs_ode = controls_store_module.build_rhs_ode

    def counted_build_rhs_ode(process):
        build_calls.append(process.metadata.name)
        return original_build_rhs_ode(process)

    monkeypatch.setattr(controls_store_module, "build_rhs_ode", counted_build_rhs_ode)
    store = ControlsStore.from_collection(collection)

    assert build_calls == list(processes)
    assert (
        store.max_event_gap_fraction,
        store.max_measurements_per_event_gap,
    ) == expected_bounds


def test_training_selection_reuses_rhs_names_for_output_bounds(monkeypatch):
    processes = {
        "a": _process("a", n_sample=2, n_bolus=1)[0],
        "b": _process("b", n_sample=3, n_bolus=2, n_extra_meas=4)[0],
    }
    collection = BioProcessCollection(processes=processes, metadata={})
    store = TrainingDataStore.from_collection(
        collection, target_source="reactor_components"
    )
    selected_collection = BioProcessCollection(
        processes={"b": processes["b"]}, metadata={}
    )
    expected_bounds = _output_window_bounds(selected_collection, ["b"])
    build_calls = []
    original_build_rhs_ode = controls_store_module.build_rhs_ode

    def counted_build_rhs_ode(process):
        build_calls.append(process.metadata.name)
        return original_build_rhs_ode(process)

    monkeypatch.setattr(controls_store_module, "build_rhs_ode", counted_build_rhs_ode)

    standalone = store.controls_store.select_processes(("b",), selected_collection)
    assert build_calls == ["b"]
    assert (
        standalone.max_event_gap_fraction,
        standalone.max_measurements_per_event_gap,
    ) == expected_bounds

    build_calls.clear()
    selected = store.select_processes(("b",), selected_collection)
    assert build_calls == []
    assert (
        selected.controls_store.max_event_gap_fraction,
        selected.controls_store.max_measurements_per_event_gap,
    ) == expected_bounds


@pytest.mark.parametrize("n_sample", [0, 1, 2, 5, 25])
@pytest.mark.parametrize("n_bolus", [0, 3])
@pytest.mark.parametrize(
    "n_dense,n_prediction", [(0, 30), (0, 200), (30, 0), (30, 200)]
)
def test_output_window_bound_covers_every_gap(n_sample, n_bolus, n_dense, n_prediction):
    """Brute-force the true per-gap point count and check the derived window covers it.

    This is the property the whole windowing rests on: get it wrong and output rows are
    silently dropped. Heterogeneous measurement counts are included so the padded slots
    (parked at ``t1`` by ``clamp_padded_time_rows``) land in the final gap.
    """
    processes = {
        "a": _process("a", n_sample=n_sample, n_bolus=n_bolus)[0],
        "b": _process("b", n_sample=n_sample, n_bolus=n_bolus, n_extra_meas=7)[0],
    }
    collection = BioProcessCollection(processes=processes, metadata={})
    fraction, per_gap = _output_window_bounds(collection, list(processes))

    padded_width = 0
    layouts = []
    for name in processes:
        _, meas = _process(
            name,
            n_sample=n_sample,
            n_bolus=n_bolus,
            n_extra_meas=7 if name == "b" else 0,
        )
        layouts.append(np.asarray(meas))
        padded_width = max(padded_width, meas.shape[0])

    store = ControlsStore.from_collection(collection)
    collection_true_max = 0
    for name, meas in zip(processes, layouts, strict=True):
        controls = store.get_controls(name)
        t0, t1 = float(meas[0]), float(meas[-1])
        padded = np.concatenate([meas, np.full(padded_width - meas.size, t1)])
        blocks = [padded]
        if n_dense:
            blocks.append(np.linspace(t0, t1, n_dense))
        if n_prediction:
            blocks.append(np.linspace(t0, t1, n_prediction))
        grid = np.sort(np.concatenate(blocks))

        events = np.concatenate(
            [
                np.asarray(controls.sample_event_times)[
                    np.asarray(controls.sample_event_mask)
                ],
                np.asarray(controls.bolus_event_times)[
                    np.asarray(controls.bolus_event_mask)
                ],
            ]
        )
        events = np.unique(events[(events > t0) & (events <= t1)])
        boundaries = np.unique(np.concatenate([[t0], events, [t1]]))
        true_max = max(
            int(((grid > lo) & (grid <= hi)).sum())
            for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True)
        )
        collection_true_max = max(collection_true_max, true_max)
        window = _output_window(controls, n_dense + n_prediction)
        assert window >= true_max, (
            f"{name}: window {window} < true max points per gap {true_max} "
            f"(fraction={fraction}, per_gap={per_gap})"
        )
        assert window == math.ceil(fraction * (n_dense + n_prediction)) + per_gap + 2, (
            "window must be exactly the documented formula"
        )

    # ...and TIGHT. Some slack is unavoidable: ``f`` and ``G`` are collection-wide
    # maxima and need not be attained in the same gap, or even the same process. What
    # must never come back is gross looseness — counting online control signals as
    # measurements put G at 451 instead of 6, which pushed the window past the grid so
    # it clamped and did
    # nothing. Per-segment save cost scales with the window, so this is a real guard.
    assert window <= 2 * collection_true_max + 8, (
        f"window {window} is loose against the collection's true max "
        f"{collection_true_max}"
    )


def _with_process_variable(process, name, n_points, *, is_controlled):
    """Attach a process variable sampled at ``n_points`` over the horizon."""
    ts = np.linspace(0.0, T_END, n_points)
    process.process_variables[name] = ProcessVariable(
        name=name,
        unit="-",
        is_controlled=is_controlled,
        values=TimeSeries(times=jnp.asarray(ts), values=jnp.asarray(np.ones(n_points))),
    )
    return process


def test_controlled_process_variables_do_not_inflate_the_window():
    """A CONTROLLED process variable is a model input (pH, temperature, gas flow),
    logged online at thousands of points. It can never be a measurement target, so its
    timestamps must not enter the window bound.

    Counting them was a real bug: G came out at 288/317/451 instead of 1/15/6 on the
    shipped examples, so the window clamped to the whole grid and did nothing at all.
    """
    process, meas = _process("p", n_sample=4, n_bolus=0)
    dense_pv = _with_process_variable(process, "pH", 1441, is_controlled=True)
    collection = BioProcessCollection(processes={"p": dense_pv}, metadata={})
    controls = ControlsStore.from_collection(collection).get_controls("p")

    n_prediction = 200
    grid, *_ = build_union_time_grid(meas, meas.shape[0], n_prediction=n_prediction)
    window = _output_window(controls, n_prediction)
    assert window < grid.shape[0], (
        f"window {window} must stay well under the grid ({grid.shape[0]}); "
        "a controlled "
        "PV's 1441 online samples are not measurements"
    )
    # Same collection without the controlled PV must give the identical window.
    bare, _ = _process("p", n_sample=4, n_bolus=0)
    bare_controls = ControlsStore.from_collection(
        BioProcessCollection(processes={"p": bare}, metadata={})
    ).get_controls("p")
    assert window == _output_window(bare_controls, n_prediction), (
        "a controlled process variable must not change the window at all"
    )


def test_measured_process_variables_do_count_toward_the_window():
    """The mirror: an ``is_controlled=False`` process variable IS selectable as a target
    (``target_source='process_variables'`` / ``'combined'``), so its timestamps must be
    in the bound. Otherwise a combined-target run would overflow."""
    n_pv_points = 61
    process, _ = _process("p", n_sample=4, n_bolus=0)
    measured_pv = _with_process_variable(
        process, "product_ratio", n_pv_points, is_controlled=False
    )
    collection = BioProcessCollection(processes={"p": measured_pv}, metadata={})
    fraction, per_gap = _output_window_bounds(collection, ["p"])

    # The PV grid is far denser than the 6-point offline grid, so it must dominate G.
    boundaries = np.unique(
        np.concatenate([[0.0], np.linspace(8.0, T_END - 8.0, 4), [T_END]])
    )
    pv_ts = np.linspace(0.0, T_END, n_pv_points)
    true_per_gap = max(
        int(((pv_ts > lo) & (pv_ts <= hi)).sum())
        for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True)
    )
    assert per_gap >= true_per_gap, (
        f"a MEASURED process variable must be counted: per_gap={per_gap} < "
        f"{true_per_gap}"
    )
    assert fraction > 0.0


# --------------------------------------------------------------------------------
# 4 + 5 + 6. End-to-end guarantees through ``solve_physical_states``.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("n_sample,n_bolus", [(0, 0), (3, 0), (0, 3), (6, 4)])
def test_healthy_solve_has_no_nonfinite_output_row(n_sample, n_bolus):
    """The observable symptom of an undersized window or a mis-assigned output time is a
    row nothing ever wrote, which comes back ``inf``."""
    process, meas = _process("p", n_sample=n_sample, n_bolus=n_bolus)
    grid, *_ = build_union_time_grid(meas, meas.shape[0], n_prediction=200)
    states = _solve(_wrapper(process), grid.astype(jnp.float64), n_linspace=200)
    assert bool(jnp.all(jnp.isfinite(states))), "healthy solve must be all-finite"


def test_zero_event_process_solves():
    """Tier A leaves the preset array EMPTY when a process has no bolus and no sample
    slots (both padded widths are collection-wide maxima and legitimately 0). That used
    to be a hard ``jnp.argmin`` error inside ``_find_next_preset_time``."""
    process, meas = _process("p", n_sample=0, n_bolus=0, n_extra_meas=5)
    wrapper = _wrapper(process)
    assert int(wrapper.controls.sample_event_mask.shape[0]) == 0
    assert int(wrapper.controls.bolus_event_mask.shape[0]) == 0
    states = _solve(wrapper, meas)
    assert bool(jnp.all(jnp.isfinite(states)))


@pytest.mark.parametrize("n_sample,n_bolus", [(0, 0), (4, 3)])
def test_answer_is_independent_of_the_output_grid_size(n_sample, n_bolus):
    """THE point of the change. Extra output points are interpolation, not extra segment
    boundaries, so they must not move the values at the measurement times. Before, every
    output point was a boundary and a denser grid changed the discretisation."""
    process, meas = _process("p", n_sample=n_sample, n_bolus=n_bolus, n_extra_meas=4)
    wrapper = _wrapper(process)
    reference = _solve(wrapper, meas)

    for n_prediction in (30, 200, 400):
        grid, sample_idx, *_ = build_union_time_grid(
            meas, meas.shape[0], n_prediction=n_prediction
        )
        states = _solve(wrapper, grid.astype(jnp.float64), n_linspace=n_prediction)
        assert jnp.allclose(states[sample_idx], reference, rtol=1e-6, atol=1e-9), (
            f"n_prediction={n_prediction} moved the measurement-time values"
        )
