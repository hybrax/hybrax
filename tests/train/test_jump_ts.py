"""``jump_ts`` wiring: discrete-event-sourced solver jump times.

``jump_ts`` are genuine vector-field discontinuity times (from
``BioProcess.discrete_events``) wired to ``diffrax.PIDController(jump_ts=...)``.
They are NOT the bolus/sample STATE-jump events (those are handled by the
callbacks solve). With no discrete events the whole path resolves to empty /
``None`` so the solve is a plain ``PIDController(rtol, atol)``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from hybrax.format.dataclasses import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    DiscreteEvents,
    ProcessVariable,
    ReactorMedium,
    ReactorMediumComponent,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)

from hybrax.train.controls_store import ControlsStore, _discrete_event_jump_ts
from hybrax.train.trainer import clamp_padded_time_rows


def _process(discrete_events: DiscreteEvents | None) -> BioProcess:
    return BioProcess(
        metadata=BioProcessMetadata(name="p1", process_type="fed_batch"),
        time_axis=TimeAxis(unit="h", start=0.0, end=1.0, time_reference="start"),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes={}),
        reactor_medium=ReactorMedium(
            name="rm",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass", unit="g/L", concentration=StaticVariable(0.1)
                )
            },
        ),
        process_variables={
            "CF": ProcessVariable(
                name="CF",
                unit="g/L",
                is_controlled=True,
                values=TimeSeries(
                    times=jnp.asarray([0.0, 1.0]), values=jnp.asarray([1.0, 1.1])
                ),
            )
        },
        discrete_events=discrete_events,
    )


def _collection(discrete_events: DiscreteEvents | None) -> BioProcessCollection:
    return BioProcessCollection(
        metadata={}, processes={"p1": _process(discrete_events)}
    )


def test_discrete_event_jump_ts_none_is_empty():
    assert _discrete_event_jump_ts(_process(None)) == []


def test_discrete_event_jump_ts_sorted_and_deduped():
    de = DiscreteEvents(times=jnp.asarray([0.7, 0.2, 0.7, 0.4]))
    got = _discrete_event_jump_ts(_process(de))
    assert got == pytest.approx([0.2, 0.4, 0.7], abs=1e-6)


def test_controls_store_sources_jump_ts_from_discrete_events():
    store = ControlsStore.from_collection(
        _collection(DiscreteEvents(times=jnp.asarray([0.25, 0.75])))
    )
    assert store.shape_metadata["max_jump_ts_length"] == 2
    per_process = store.get_controls("p1")
    assert int(per_process.jump_ts_length) == 2
    np.testing.assert_allclose(
        np.asarray(per_process.active_jump_ts), [0.25, 0.75], atol=1e-6
    )


def test_controls_store_empty_jump_ts_without_discrete_events():
    store = ControlsStore.from_collection(_collection(None))
    assert store.shape_metadata["max_jump_ts_length"] == 0
    # width-0 jump_ts ⇒ the solver receives ``jump_ts=None`` (plain solve).
    assert store.jump_ts.shape[1] == 0


def test_clamp_padded_time_rows_handles_zero_width():
    # The no-discrete-events case: a width-0 jump_ts array must not crash the
    # gather inside clamp_padded_time_rows; it passes through unchanged.
    rows = jnp.zeros((3, 0))
    out = clamp_padded_time_rows(rows, jnp.zeros((3,), dtype=jnp.int32))
    assert out.shape == (3, 0)
