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
    PseudobatchSpeciesTransform,
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


def _ts(times, values):
    return TimeSeries(
        times=jnp.array(times, dtype=float),
        values=jnp.array(values, dtype=float),
    )


def _lightweight_transform_metadata() -> dict:
    return {
        "transform": {
            "name": "pseudo_batch",
            "species": "biomass",
            "cstar_fit_strategy": "cubic_interp",
            "is_constant": False,
            "constant_value": None,
        }
    }


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
                        metadata=_lightweight_transform_metadata(),
                    ),
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
    assert loaded.processes["p"].pseudobatch_transform is None
    assert "pseudobatch_transform" not in payload["processes"]["p"]


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
        concentration = payload["processes"]["p"]["reactor_medium"]["components"][
            "biomass"
        ]["concentration"]
        concentration["metadata"]["transform"]["series"] = {
            "adf_ts": _ts([0.0, 1.0], [1.0, 1.1]).to_dict(),
            "feed_corr_ts": _ts([0.0, 1.0], [0.0, 0.1]).to_dict(),
        }

        with open(out_dir / "data.json", "w") as fh:
            json.dump(payload, fh)

        with pytest.raises(ValueError, match="nested executable pseudobatch"):
            load_process_collection(out_dir)


def test_pseudobatch_transform_roundtrips_process_level_bundle() -> None:
    process = _build_process()
    process.pseudobatch_transform = PseudobatchTransform(
        adf_ts=_ts([0.0, 5.0, 10.0], [1.0, 1.1, 1.2]),
        reactor_volume_ts=_ts([0.0, 5.0, 10.0], [1.0, 1.05, 1.1]),
        sample_compensation_ts=_ts([0.0, 5.0, 10.0], [1.0, 1.02, 1.04]),
        accumulated_feed_ts={"feed": _ts([0.0, 10.0], [0.0, 0.3])},
        species={
            "biomass": PseudobatchSpeciesTransform(
                species="biomass",
                c_star_ts=_ts([0.0, 5.0, 10.0], [0.2, 1.2, 2.5]),
                feed_corr_ts=_ts([0.0, 5.0, 10.0], [0.0, 0.1, 0.2]),
                is_constant=False,
                constant_value=None,
                cstar_fit_strategy="smoothing_bspline",
            )
        },
    )
    collection = BioProcessCollection(metadata=None, processes={"p": process})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, out_dir)
        payload = _read_payload(out_dir / "data.json")
        loaded = load_process_collection(out_dir)

    stored = payload["processes"]["p"]["pseudobatch_transform"]
    assert "times" in stored["adf_ts"]
    assert "values" in stored["accumulated_feed_ts"]["feed"]
    assert stored["species"]["biomass"]["cstar_fit_strategy"] == "smoothing_bspline"

    transform = loaded.processes["p"].pseudobatch_transform
    assert transform is not None
    assert transform.adf_ts.values.shape == (3,)
    assert transform.accumulated_feed_ts["feed"].values.shape == (2,)
    assert transform.species["biomass"].c_star_ts.values.shape == (3,)
    assert transform.species["biomass"].feed_corr_ts.values.shape == (3,)


@pytest.mark.parametrize("keep_new_key", [False, True])
def test_load_rejects_old_pseudobatch_cstar_interp_key(keep_new_key: bool) -> None:
    process = _build_process()
    process.pseudobatch_transform = PseudobatchTransform(
        adf_ts=_ts([0.0, 10.0], [1.0, 1.2]),
        reactor_volume_ts=_ts([0.0, 10.0], [1.0, 1.1]),
        sample_compensation_ts=_ts([0.0, 10.0], [1.0, 1.0]),
        accumulated_feed_ts={},
        species={
            "biomass": PseudobatchSpeciesTransform(
                species="biomass",
                c_star_ts=_ts([0.0, 10.0], [0.2, 2.5]),
                feed_corr_ts=_ts([0.0, 10.0], [0.0, 0.2]),
                cstar_fit_strategy="smoothing_bspline",
            )
        },
    )
    collection = BioProcessCollection(metadata=None, processes={"p": process})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, out_dir)
        payload = _read_payload(out_dir / "data.json")
        species = payload["processes"]["p"]["pseudobatch_transform"]["species"][
            "biomass"
        ]
        if keep_new_key:
            species["cstar_interp"] = "pchip"
        else:
            species["cstar_interp"] = species.pop("cstar_fit_strategy")
        with open(out_dir / "data.json", "w") as fh:
            json.dump(payload, fh)

        with pytest.raises(ValueError, match="cstar_interp"):
            load_process_collection(out_dir)


def test_load_rejects_unknown_pseudobatch_cstar_fit_strategy() -> None:
    process = _build_process()
    process.pseudobatch_transform = PseudobatchTransform(
        adf_ts=_ts([0.0, 10.0], [1.0, 1.2]),
        reactor_volume_ts=_ts([0.0, 10.0], [1.0, 1.1]),
        sample_compensation_ts=_ts([0.0, 10.0], [1.0, 1.0]),
        accumulated_feed_ts={},
        species={
            "biomass": PseudobatchSpeciesTransform(
                species="biomass",
                c_star_ts=_ts([0.0, 10.0], [0.2, 2.5]),
                feed_corr_ts=_ts([0.0, 10.0], [0.0, 0.2]),
                cstar_fit_strategy="smoothing_bspline",
            )
        },
    )
    collection = BioProcessCollection(metadata=None, processes={"p": process})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, out_dir)
        payload = _read_payload(out_dir / "data.json")
        species = payload["processes"]["p"]["pseudobatch_transform"]["species"][
            "biomass"
        ]
        species["cstar_fit_strategy"] = "legacy_pchip"
        with open(out_dir / "data.json", "w") as fh:
            json.dump(payload, fh)

        with pytest.raises(ValueError, match="cstar_fit_strategy"):
            load_process_collection(out_dir)


def test_pseudobatch_transform_roundtrips_empty_species_dict() -> None:
    process = _build_process()
    process.pseudobatch_transform = PseudobatchTransform(
        adf_ts=_ts([0.0, 1.0], [1.0, 1.1]),
        reactor_volume_ts=_ts([0.0, 1.0], [1.0, 1.0]),
        sample_compensation_ts=_ts([0.0, 1.0], [1.0, 1.0]),
        accumulated_feed_ts={},
        species={},
    )
    collection = BioProcessCollection(metadata=None, processes={"p": process})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, out_dir)
        payload = _read_payload(out_dir / "data.json")
        loaded = load_process_collection(out_dir)

    assert payload["processes"]["p"]["pseudobatch_transform"]["species"] == {}
    assert loaded.processes["p"].pseudobatch_transform.species == {}


def test_pseudobatch_transform_loader_rejects_missing_required_keys() -> None:
    process = _build_process()
    process.pseudobatch_transform = PseudobatchTransform(
        adf_ts=_ts([0.0, 1.0], [1.0, 1.1]),
        reactor_volume_ts=_ts([0.0, 1.0], [1.0, 1.0]),
        sample_compensation_ts=_ts([0.0, 1.0], [1.0, 1.0]),
        accumulated_feed_ts={},
        species={
            "biomass": PseudobatchSpeciesTransform(
                species="biomass",
                c_star_ts=_ts([0.0, 1.0], [0.2, 0.3]),
                feed_corr_ts=_ts([0.0, 1.0], [0.0, 0.0]),
                cstar_fit_strategy="smoothing_bspline",
            )
        },
    )
    collection = BioProcessCollection(metadata=None, processes={"p": process})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, out_dir)
        payload = _read_payload(out_dir / "data.json")
        del payload["processes"]["p"]["pseudobatch_transform"]["adf_ts"]

        with open(out_dir / "data.json", "w") as fh:
            json.dump(payload, fh)

        with pytest.raises(ValueError, match="missing required key\\(s\\): adf_ts"):
            load_process_collection(out_dir)


def test_pseudobatch_transform_loader_rejects_malformed_species_entries() -> None:
    process = _build_process()
    process.pseudobatch_transform = PseudobatchTransform(
        adf_ts=_ts([0.0, 1.0], [1.0, 1.1]),
        reactor_volume_ts=_ts([0.0, 1.0], [1.0, 1.0]),
        sample_compensation_ts=_ts([0.0, 1.0], [1.0, 1.0]),
        accumulated_feed_ts={},
        species={
            "biomass": PseudobatchSpeciesTransform(
                species="biomass",
                c_star_ts=_ts([0.0, 1.0], [0.2, 0.3]),
                feed_corr_ts=_ts([0.0, 1.0], [0.0, 0.0]),
                cstar_fit_strategy="smoothing_bspline",
            )
        },
    )
    collection = BioProcessCollection(metadata=None, processes={"p": process})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, out_dir)
        payload = _read_payload(out_dir / "data.json")
        del payload["processes"]["p"]["pseudobatch_transform"]["species"]["biomass"][
            "feed_corr_ts"
        ]

        with open(out_dir / "data.json", "w") as fh:
            json.dump(payload, fh)

        with pytest.raises(
            ValueError, match="missing required key\\(s\\): feed_corr_ts"
        ):
            load_process_collection(out_dir)


def test_pseudobatch_transform_loader_rejects_species_key_mismatch() -> None:
    process = _build_process()
    process.pseudobatch_transform = PseudobatchTransform(
        adf_ts=_ts([0.0, 1.0], [1.0, 1.1]),
        reactor_volume_ts=_ts([0.0, 1.0], [1.0, 1.0]),
        sample_compensation_ts=_ts([0.0, 1.0], [1.0, 1.0]),
        accumulated_feed_ts={},
        species={
            "biomass": PseudobatchSpeciesTransform(
                species="biomass",
                c_star_ts=_ts([0.0, 1.0], [0.2, 0.3]),
                feed_corr_ts=_ts([0.0, 1.0], [0.0, 0.0]),
                cstar_fit_strategy="smoothing_bspline",
            )
        },
    )
    collection = BioProcessCollection(metadata=None, processes={"p": process})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, out_dir)
        payload = _read_payload(out_dir / "data.json")
        payload["processes"]["p"]["pseudobatch_transform"]["species"]["biomass"][
            "species"
        ] = "glucose"

        with open(out_dir / "data.json", "w") as fh:
            json.dump(payload, fh)

        with pytest.raises(ValueError, match="species key must match"):
            load_process_collection(out_dir)


def test_pseudobatch_transform_loader_rejects_invalid_is_constant_type() -> None:
    process = _build_process()
    process.pseudobatch_transform = PseudobatchTransform(
        adf_ts=_ts([0.0, 1.0], [1.0, 1.1]),
        reactor_volume_ts=_ts([0.0, 1.0], [1.0, 1.0]),
        sample_compensation_ts=_ts([0.0, 1.0], [1.0, 1.0]),
        accumulated_feed_ts={},
        species={
            "biomass": PseudobatchSpeciesTransform(
                species="biomass",
                c_star_ts=_ts([0.0, 1.0], [0.2, 0.3]),
                feed_corr_ts=_ts([0.0, 1.0], [0.0, 0.0]),
                cstar_fit_strategy="smoothing_bspline",
            )
        },
    )
    collection = BioProcessCollection(metadata=None, processes={"p": process})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "collection"
        save_process_collection(collection, out_dir)
        payload = _read_payload(out_dir / "data.json")
        payload["processes"]["p"]["pseudobatch_transform"]["species"]["biomass"][
            "is_constant"
        ] = "False"

        with open(out_dir / "data.json", "w") as fh:
            json.dump(payload, fh)

        with pytest.raises(ValueError, match="is_constant must be bool"):
            load_process_collection(out_dir)
