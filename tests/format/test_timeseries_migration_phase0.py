"""Phase 0 baseline tests for TimeSeries migration."""

import json
import tempfile
from pathlib import Path

import jax.numpy as jnp

from bpbench import (
    BioProcess,
    BioProcessMetadata,
    ProcessVariable,
    ReactorMedium,
    ReactorMediumComponent,
    SampleVolumeChange,
    TimeAxis,
    TimeSeries,
    Volume,
)
from bpbench.mechanistic import get_control_splines
from bpbench.serialization import load_process_collection, save_process_collection
from bpbench.splines import split_timeseries


def _ts(times, values):
    return TimeSeries(
        times=jnp.array(times, dtype=float),
        values=jnp.array(values, dtype=float),
    )


def _build_process():
    return BioProcess(
        metadata=BioProcessMetadata(name="phase0", process_type="fed_batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=10.0, time_reference="t0"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "feed": SampleVolumeChange(
                    name="feed",
                    unit="L",
                    is_controlled=True,
                    is_continuous=True,
                    values=_ts([0.0, 5.0, 10.0], [0.0, 0.2, 0.5]),
                )
            },
        ),
        reactor_medium=ReactorMedium(
            name="medium",
            density=1.0,
            density_unit="kg/L",
            components={
                "biomass": ReactorMediumComponent(
                    name="biomass",
                    unit="g/L",
                    concentration=_ts([0.0, 5.0, 10.0], [0.2, 1.1, 2.4]),
                    is_intracellular=False,
                ),
            },
        ),
        process_variables={
            "pH": ProcessVariable(
                name="pH",
                unit="",
                is_controlled=True,
                values=_ts([0.0, 5.0, 10.0], [7.0, 7.0, 7.1]),
            )
        },
    )


def test_phase0_serialization_roundtrip_keeps_canonical_timeseries_shape():
    from bpbench import BioProcessCollection

    process = _build_process()
    collection = BioProcessCollection(metadata=None, processes={"phase0": process})

    with tempfile.TemporaryDirectory() as tmpdir:
        collection_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, collection_dir)
        with open(collection_dir / "data.json", "r") as fh:
            payload = json.load(fh)
        loaded = load_process_collection(collection_dir)

    stored_ts = payload["processes"]["phase0"]["reactor_medium"]["components"][
        "biomass"
    ]["concentration"]
    assert "times" in stored_ts
    assert "values" in stored_ts

    biomass = loaded.processes["phase0"].reactor_medium.components["biomass"]
    feed = loaded.processes["phase0"].volume.volume_changes["feed"]
    assert biomass.concentration.times.shape == (3,)
    assert feed.values.times.shape == (3,)
    assert jnp.allclose(
        biomass.concentration.values,
        jnp.array([0.2, 1.1, 2.4]),
    )


def test_phase0_spline_split_accepts_canonical_timeseries_constructor():
    ts = _ts([0.0, 2.0, 4.0, 6.0], [1.0, 2.0, 3.0, 4.0])
    segments = split_timeseries(ts, jnp.array([0.0, 3.0, 6.0]))
    assert len(segments) == 2
    assert segments[0].times.shape[0] == 2
    assert segments[1].times.shape[0] == 2


def test_phase0_mechanistic_control_splines_with_canonical_timeseries():
    process = _build_process()
    control = get_control_splines(process)
    values = control(jnp.array(5.0))
    assert control.control_names == ("feed", "pH")
    assert values.shape == (2,)
