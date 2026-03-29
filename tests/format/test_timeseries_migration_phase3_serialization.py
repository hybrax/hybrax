"""Phase 3 tests for canonical TimeSeries serialization schema."""

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
from bpbench.serialization import load_process_collection, save_process_collection


def _build_process() -> BioProcess:
    return BioProcess(
        metadata=BioProcessMetadata(name="phase3", process_type="fed_batch"),
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
                    values=TimeSeries(
                        times=jnp.array([3.0, 8.0]),
                        values=jnp.array([-0.02, -0.03]),
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
                values=TimeSeries(
                    times=jnp.array([0.0, 10.0]),
                    values=jnp.array([7.0, 7.1]),
                ),
            )
        },
    )


def _read_payload(path: Path) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def test_phase3_save_writes_canonical_timeseries_keys() -> None:
    collection = BioProcessCollection(metadata=None, processes={"p": _build_process()})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, out_dir)
        payload = _read_payload(out_dir / "data.json")

    biomass = payload["processes"]["p"]["reactor_medium"]["components"]["biomass"][
        "concentration"
    ]
    assert biomass["type"] == "TimeSeries"
    assert "times" in biomass
    assert "breaks" in biomass
    assert "coeffs" in biomass

    sample_values = payload["processes"]["p"]["volume"]["volume_changes"]["sample"][
        "values"
    ]
    assert "times" in sample_values


def test_phase3_load_rejects_legacy_only_timeseries_keys() -> None:
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


def test_phase3_load_accepts_canonical_only_timeseries_keys() -> None:
    collection = BioProcessCollection(metadata=None, processes={"p": _build_process()})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, out_dir)
        payload = _read_payload(out_dir / "data.json")

        biomass = payload["processes"]["p"]["reactor_medium"]["components"]["biomass"][
            "concentration"
        ]
        with open(out_dir / "data.json", "w") as fh:
            json.dump(payload, fh)

        loaded = load_process_collection(out_dir)

    biomass_loaded = loaded.processes["p"].reactor_medium.components["biomass"]
    assert biomass_loaded.concentration.times.shape == (3,)
    assert biomass_loaded.concentration.breaks.shape == (2,)
    sample_loaded = loaded.processes["p"].volume.volume_changes["sample"]
    assert sample_loaded.values.times.shape == (2,)
