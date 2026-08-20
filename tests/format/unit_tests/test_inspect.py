"""Tests for bp_format.inspect display helpers."""

import re

import jax.numpy as jnp

from bp_format import (
    BioProcess,
    BioProcessMetadata,
    FeedMedium,
    FeedMediumComponent,
    Inflow,
    Outflow,
    ReactorMedium,
    ReactorMediumComponent,
    StaticVariable,
    TimeAxis,
    TimeSeries,
    Volume,
)
from bp_format.inspect import _format_rmc_flow
from bp_format.mechanistic import _build_retention, get_process_ordering


def _ts(t, v):
    return TimeSeries(times=jnp.array(t, dtype=float), values=jnp.array(v, dtype=float))


def _make_process():
    """Minimal continuous process: one Inflow, one unretained Outflow, and
    one Outflow that partially retains biomass (e.g. perfusion)."""
    rm = ReactorMedium(
        name="medium",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass", unit="g/L", concentration=_ts([0.0, 1.0], [1.0, 2.0])
            ),
        },
    )
    feed = FeedMedium(
        name="feed_medium",
        components={
            "biomass": FeedMediumComponent(
                name="biomass", unit="g/L", concentration=StaticVariable(value=0.0)
            ),
        },
    )
    volume_changes = {
        "feed": Inflow(
            name="feed",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            feed_medium=feed,
            values=_ts([0.0, 1.0], [0.0, 0.1]),
        ),
        "harvest": Outflow(
            name="harvest",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            values=_ts([0.0, 1.0], [0.0, -0.05]),
        ),
        "perfusion": Outflow(
            name="perfusion",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            values=_ts([0.0, 1.0], [0.0, -0.05]),
            retention={"biomass": 0.7},
        ),
    }
    return BioProcess(
        metadata=BioProcessMetadata(name="p", process_type="fed_batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=1.0, time_reference="x"),
        volume=Volume(initial_volume=1.0, unit="L", volume_changes=volume_changes),
        reactor_medium=rm,
    )


def test_format_rmc_flow_dilution_matches_inflow_presence():
    """_format_rmc_flow shows a dilution(...) term iff there is at least one
    Inflow — the same condition under which _apply_feed_dilution's dilution
    term is driven by a nonzero total_in (mechanistic.py)."""
    process = _make_process()
    ordering = get_process_ordering(process)
    inflow_all = list(ordering.name_controlled_Inflows) + list(
        ordering.name_modeled_Inflows
    )
    outflow_all = list(ordering.name_controlled_Outflows) + list(
        ordering.name_modeled_Outflows
    )
    flow = _format_rmc_flow("biomass", inflow_all, outflow_all, process)
    assert bool(inflow_all) == ("dilution" in flow)


def test_format_rmc_flow_retention_matches_build_retention():
    """_format_rmc_flow hand-reads Outflow.retention for its display table;
    guard against it drifting from _build_retention (mechanistic.py), which
    computes the same retention matrix numerically for the actual RHS ODE.
    """
    process = _make_process()
    ordering = get_process_ordering(process)
    inflow_all = list(ordering.name_controlled_Inflows) + list(
        ordering.name_modeled_Inflows
    )
    outflow_all = list(ordering.name_controlled_Outflows) + list(
        ordering.name_modeled_Outflows
    )
    rmc_idx = ordering.name_modeled_RMCs.index("biomass")

    retention_matrix = _build_retention(
        process, tuple(outflow_all), ordering.name_modeled_RMCs
    )
    numerically_retained = {
        name
        for name, row in zip(outflow_all, retention_matrix)
        if float(row[rmc_idx]) != 0.0
    }

    flow = _format_rmc_flow("biomass", inflow_all, outflow_all, process)
    displayed_retained = set(re.findall(r"(\w+)=[\d.eE+-]+", flow))

    assert displayed_retained == numerically_retained
    assert displayed_retained == {"perfusion"}
