"""Stable serialization tests for canonical TimeSeries schema."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import jax.numpy as jnp
import pytest

from bpbench import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    ProcessVariable,
    ReactorMedium,
    ReactorMediumComponent,
    SampleVolumeChange,
    TimeAxis,
    TimeSeries,
    Volume,
)
from bpbench.serialization import (
    _timeseries_to_dict_payload,
    load_process_collection,
    save_process_collection,
)


def _ts(times, values):
    return TimeSeries(
        times=jnp.array(times, dtype=float),
        values=jnp.array(values, dtype=float),
    )


def _build_process() -> BioProcess:
    return BioProcess(
        metadata=BioProcessMetadata(name="ts-ser", process_type="fed_batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=10.0, time_reference="t0"),
        volume=Volume(
            initial_volume=1.0,
            unit="L",
            volume_changes={
                "sample": SampleVolumeChange(
                    name="sample",
                    unit="L",
                    is_controlled=False,
                    is_continuous=False,
                    values=_ts([3.0, 8.0], [-0.02, -0.03]),
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
                        times=jnp.array([0.0, 5.0, 10.0]),
                        values=jnp.array([0.2, 1.0, 2.1]),
                        breaks=jnp.array([0.0, 10.0]),
                        coeffs=jnp.array([[0.2, 0.19, 0.0, 0.0]]),
                        segment_start_piece_idx=jnp.array([0]),
                    ),
                    is_intracellular=False,
                ),
            },
        ),
        process_variables={
            "pH": ProcessVariable(
                name="pH",
                unit="",
                is_controlled=True,
                values=_ts([0.0, 10.0], [7.0, 7.1]),
            )
        },
    )


def _read_payload(path: Path) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def test_roundtrip_uses_canonical_timeseries_shape() -> None:
    process = _build_process()
    collection = BioProcessCollection(metadata=None, processes={"p": process})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, out_dir)
        payload = _read_payload(out_dir / "data.json")
        loaded = load_process_collection(out_dir)

    stored_ts = payload["processes"]["p"]["reactor_medium"]["components"]["biomass"][
        "concentration"
    ]
    assert "times" in stored_ts
    assert "values" in stored_ts
    assert "timepoints" not in stored_ts

    biomass = loaded.processes["p"].reactor_medium.components["biomass"]
    sample = loaded.processes["p"].volume.volume_changes["sample"]
    assert biomass.concentration.times.shape == (3,)
    assert sample.values.times.shape == (2,)


def test_load_rejects_legacy_only_timeseries_keys() -> None:
    collection = BioProcessCollection(metadata=None, processes={"p": _build_process()})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, out_dir)
        payload = _read_payload(out_dir / "data.json")

        biomass = payload["processes"]["p"]["reactor_medium"]["components"]["biomass"][
            "concentration"
        ]
        biomass["timepoints"] = biomass["times"]
        biomass.pop("times", None)

        sample_values = payload["processes"]["p"]["volume"]["volume_changes"]["sample"][
            "values"
        ]
        sample_values["timepoints"] = sample_values["times"]
        sample_values.pop("times", None)

        with open(out_dir / "data.json", "w") as fh:
            json.dump(payload, fh)

        with pytest.raises(
            ValueError, match="payload with discrete values must include 'times'"
        ):
            load_process_collection(out_dir)


def test_timeseries_payload_uses_times_only() -> None:
    # Intentional internal-contract check: canonical serializer payload shape.
    payload = _timeseries_to_dict_payload(_ts([0.0, 1.0], [1.0, 2.0]))
    assert "times" in payload
    assert "timepoints" not in payload
