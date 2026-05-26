"""Stable serialization tests for canonical TimeSeries schema."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import jax.numpy as jnp
import pytest

from bp_format import (
    BioProcess,
    BioProcessCollection,
    BioProcessMetadata,
    ProcessVariable,
    PseudobatchTransform,
    ReactorMedium,
    ReactorMediumComponent,
    SampleVolumeChange,
    TimeAxis,
    TimeSeries,
    Volume,
)
from bp_format.serialization import (
    _timeseries_to_dict_payload,
    load_process_collection,
    save_process_collection,
)


OLD_PSEUDOBATCH_KEYS = {
    "adf_ts",
    "reactor_volume_ts",
    "sample_compensation_ts",
    "accumulated_feed_ts",
    "species",
    "c_star_ts",
    "feed_corr_ts",
    "cstar_fit_strategy",
}


def _ts(times, values, **kwargs):
    return TimeSeries(
        times=jnp.array(times, dtype=float),
        values=jnp.array(values, dtype=float),
        **kwargs,
    )


def _c_star_ts():
    return TimeSeries(
        times=jnp.array([0.0, 5.0, 10.0]),
        values=jnp.array([0.2, 1.2, 2.5]),
        breaks=jnp.array([0.0, 10.0]),
        coeffs=jnp.array([[0.2, 0.23, 0.0, 0.0]]),
        segment_start_piece_idx=jnp.array([0]),
        derived=True,
        metadata={
            "fit_strategy": "smoothing_bspline",
            "transform": {
                "name": "pseudo_batch",
                "component": "biomass",
                "is_constant": False,
                "constant_value": None,
            },
        },
    )


def _contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(v, key) for v in value.values())
    if isinstance(value, list):
        return any(_contains_key(v, key) for v in value)
    return False


def _json_ts(times, values) -> dict:
    return {"times": list(times), "values": list(values), "type": "TimeSeries"}


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
            total_volume=_ts([0.0, 5.0, 10.0], [1.0, 1.05, 1.1]),
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
                        metadata={"fit_strategy": "smoothing_bspline"},
                    ),
                    c_star_concentration=_c_star_ts(),
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

    component_payload = payload["processes"]["p"]["reactor_medium"]["components"][
        "biomass"
    ]
    stored_ts = component_payload["concentration"]
    assert "times" in stored_ts
    assert "values" in stored_ts
    assert "timepoints" not in stored_ts
    assert "c_star_concentration" in component_payload

    volume_payload = payload["processes"]["p"]["volume"]
    assert "total_volume" in volume_payload
    assert "times" in volume_payload["total_volume"]
    assert "timepoints" not in volume_payload["total_volume"]

    biomass = loaded.processes["p"].reactor_medium.components["biomass"]
    sample = loaded.processes["p"].volume.volume_changes["sample"]
    total_volume = loaded.processes["p"].volume.total_volume
    assert biomass.concentration.times.shape == (3,)
    assert biomass.c_star_concentration is not None
    assert biomass.c_star_concentration.values.shape == (3,)
    assert biomass.c_star_concentration.metadata["fit_strategy"] == "smoothing_bspline"
    assert sample.values.times.shape == (2,)
    assert total_volume is not None
    assert total_volume.times.shape == (3,)
    assert loaded.processes["p"].pseudobatch_transform is None
    assert "pseudobatch_transform" not in payload["processes"]["p"]


@pytest.mark.parametrize("field", ["c_star_concentration", "total_volume"])
def test_optional_schema_fields_may_be_absent(field: str) -> None:
    process = _build_process()
    if field == "c_star_concentration":
        process.reactor_medium.components["biomass"].c_star_concentration = None
    else:
        process.volume.total_volume = None
    collection = BioProcessCollection(metadata=None, processes={"p": process})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, out_dir)
        payload = _read_payload(out_dir / "data.json")
        loaded = load_process_collection(out_dir)

    if field == "c_star_concentration":
        component_payload = payload["processes"]["p"]["reactor_medium"]["components"][
            "biomass"
        ]
        assert "c_star_concentration" not in component_payload
        assert (
            loaded.processes["p"]
            .reactor_medium.components["biomass"]
            .c_star_concentration
            is None
        )
    else:
        volume_payload = payload["processes"]["p"]["volume"]
        assert "total_volume" not in volume_payload
        assert loaded.processes["p"].volume.total_volume is None


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


def test_load_rejects_nested_executable_pseudobatch_metadata() -> None:
    process = _build_process()
    collection = BioProcessCollection(metadata=None, processes={"p": process})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, out_dir)
        payload = _read_payload(out_dir / "data.json")
        c_star = payload["processes"]["p"]["reactor_medium"]["components"]["biomass"][
            "c_star_concentration"
        ]
        c_star["metadata"]["transform"]["series"] = {
            "adf": _ts([0.0, 1.0], [1.0, 1.1]).to_dict(),
            "feed_correction": _ts([0.0, 1.0], [0.0, 0.1]).to_dict(),
        }

        with open(out_dir / "data.json", "w") as fh:
            json.dump(payload, fh)

        with pytest.raises(ValueError, match="nested executable pseudobatch"):
            load_process_collection(out_dir)


def test_pseudobatch_transform_roundtrips_process_level_bundle() -> None:
    process = _build_process()
    process.pseudobatch_transform = PseudobatchTransform(
        adf=_ts([0.0, 5.0, 10.0], [1.0, 1.1, 1.2]),
        feed_corrections={"biomass": _ts([0.0, 5.0, 10.0], [0.0, 0.1, 0.2])},
        sample_compensation=_ts([0.0, 5.0, 10.0], [1.0, 1.02, 1.04]),
        accumulated_feeds={"feed": _ts([0.0, 10.0], [0.0, 0.3])},
    )
    collection = BioProcessCollection(metadata=None, processes={"p": process})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, out_dir)
        payload = _read_payload(out_dir / "data.json")
        loaded = load_process_collection(out_dir)

    stored = payload["processes"]["p"]["pseudobatch_transform"]
    assert set(stored) == {
        "adf",
        "feed_corrections",
        "sample_compensation",
        "accumulated_feeds",
    }
    assert "times" in stored["adf"]
    assert "values" in stored["accumulated_feeds"]["feed"]
    assert "values" in stored["feed_corrections"]["biomass"]
    for old_key in OLD_PSEUDOBATCH_KEYS:
        assert not _contains_key(stored, old_key)

    transform = loaded.processes["p"].pseudobatch_transform
    assert transform is not None
    assert transform.adf.values.shape == (3,)
    assert transform.sample_compensation is not None
    assert transform.sample_compensation.values.shape == (3,)
    assert transform.accumulated_feeds["feed"].values.shape == (2,)
    assert transform.feed_corrections["biomass"].values.shape == (3,)


def test_pseudobatch_transform_roundtrips_minimal_bundle_defaults() -> None:
    process = _build_process()
    process.pseudobatch_transform = PseudobatchTransform(
        adf=_ts([0.0, 1.0], [1.0, 1.1]),
        feed_corrections={},
    )
    collection = BioProcessCollection(metadata=None, processes={"p": process})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, out_dir)
        payload = _read_payload(out_dir / "data.json")
        del payload["processes"]["p"]["pseudobatch_transform"]["accumulated_feeds"]
        with open(out_dir / "data.json", "w") as fh:
            json.dump(payload, fh)
        loaded = load_process_collection(out_dir)

    stored = payload["processes"]["p"]["pseudobatch_transform"]
    assert "sample_compensation" not in stored
    assert loaded.processes["p"].pseudobatch_transform.sample_compensation is None
    assert loaded.processes["p"].pseudobatch_transform.accumulated_feeds == {}
    assert loaded.processes["p"].pseudobatch_transform.feed_corrections == {}


def test_old_pseudobatch_transform_payload_is_not_accepted() -> None:
    process = _build_process()
    process.pseudobatch_transform = PseudobatchTransform(
        adf=_ts([0.0, 1.0], [1.0, 1.1]),
        feed_corrections={"biomass": _ts([0.0, 1.0], [0.0, 0.0])},
    )
    collection = BioProcessCollection(metadata=None, processes={"p": process})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, out_dir)
        payload = _read_payload(out_dir / "data.json")
        payload["processes"]["p"]["pseudobatch_transform"] = {
            "adf_ts": _json_ts([0.0, 1.0], [1.0, 1.1]),
            "reactor_volume_ts": _json_ts([0.0, 1.0], [1.0, 1.0]),
            "sample_compensation_ts": _json_ts([0.0, 1.0], [1.0, 1.0]),
            "accumulated_feed_ts": {},
            "species": {
                "biomass": {
                    "species": "biomass",
                    "c_star_ts": _json_ts([0.0, 1.0], [0.2, 0.3]),
                    "feed_corr_ts": _json_ts([0.0, 1.0], [0.0, 0.0]),
                    "is_constant": False,
                    "constant_value": None,
                    "cstar_fit_strategy": "smoothing_bspline",
                }
            },
        }
        with open(out_dir / "data.json", "w") as fh:
            json.dump(payload, fh)

        with pytest.raises(ValueError, match=r"missing required key\(s\): adf"):
            load_process_collection(out_dir)


def test_pseudobatch_transform_loader_rejects_missing_required_keys() -> None:
    process = _build_process()
    process.pseudobatch_transform = PseudobatchTransform(
        adf=_ts([0.0, 1.0], [1.0, 1.1]),
        feed_corrections={"biomass": _ts([0.0, 1.0], [0.0, 0.0])},
    )
    collection = BioProcessCollection(metadata=None, processes={"p": process})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, out_dir)
        payload = _read_payload(out_dir / "data.json")
        del payload["processes"]["p"]["pseudobatch_transform"]["adf"]

        with open(out_dir / "data.json", "w") as fh:
            json.dump(payload, fh)

        with pytest.raises(ValueError, match=r"missing required key\(s\): adf"):
            load_process_collection(out_dir)


def test_pseudobatch_transform_loader_rejects_malformed_feed_corrections() -> None:
    process = _build_process()
    process.pseudobatch_transform = PseudobatchTransform(
        adf=_ts([0.0, 1.0], [1.0, 1.1]),
        feed_corrections={"biomass": _ts([0.0, 1.0], [0.0, 0.0])},
    )
    collection = BioProcessCollection(metadata=None, processes={"p": process})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, out_dir)
        payload = _read_payload(out_dir / "data.json")
        payload["processes"]["p"]["pseudobatch_transform"]["feed_corrections"] = []

        with open(out_dir / "data.json", "w") as fh:
            json.dump(payload, fh)

        with pytest.raises(ValueError, match="feed_corrections must be a dictionary"):
            load_process_collection(out_dir)
