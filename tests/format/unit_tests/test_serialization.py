"""Tests for bp_format.serialization functionality."""

import gzip
import json
from pathlib import Path
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

import bp_format.serialization as serialization
from bp_format.json_io import JSONParseError

from bp_format import (
    BiologicalOde,
    BioProcessCollection,
    BioProcess,
    BioProcessMetadata,
    TimeAxis,
    TimeSeries,
    StaticVariable,
    ProcessVariable,
    ReactorMediumComponent,
    ReactorMedium,
    FeedMediumComponent,
    FeedMedium,
    Inflow,
    Outflow,
    Volume,
)
from bp_format.serialization import (
    save_process_collection,
    load_process_collection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_process():
    """Build a minimal but realistic BioProcess for serialization tests."""
    biomass_ts = TimeSeries(
        times=jnp.array([0.0, 12.0, 24.0, 36.0, 48.0]),
        values=jnp.array([0.1, 1.2, 3.5, 5.8, 6.0]),
    )
    glucose_ts = TimeSeries(
        times=jnp.array([0.0, 12.0, 24.0, 36.0, 48.0]),
        values=jnp.array([20.0, 15.0, 8.0, 2.0, 0.5]),
    )
    biomass_rc = ReactorMediumComponent(
        name="biomass",
        unit="g/L",
        concentration=biomass_ts,
    )
    glucose_rc = ReactorMediumComponent(
        name="glucose",
        unit="g/L",
        concentration=glucose_ts,
    )
    reactor_medium = ReactorMedium(
        name="medium",
        density=1.0,
        density_unit="kg/L",
        components={"biomass": biomass_rc, "glucose": glucose_rc},
    )

    feed_comp = FeedMediumComponent(
        name="glucose",
        unit="g/L",
        concentration=StaticVariable(value=500.0),
        is_controlled=True,
    )
    feed_medium = FeedMedium(
        name="glucose_feed",
        density=1.1,
        density_unit="kg/L",
        components={"glucose": feed_comp},
    )
    feed_ts = TimeSeries(
        times=jnp.array([0.0, 12.0, 24.0, 36.0, 48.0]),
        values=jnp.array([0.0, 0.05, 0.10, 0.15, 0.20]),
    )
    volume_change = Inflow(
        name="glucose_feed",
        unit="L",
        is_controlled=True,
        is_continuous=True,
        feed_medium=feed_medium,
        values=feed_ts,
    )
    volume = Volume(
        initial_volume=1.0,
        unit="L",
        volume_changes={"glucose_feed": volume_change},
    )

    temp_ts = TimeSeries(
        times=jnp.array([0.0, 12.0, 24.0]),
        values=jnp.array([37.0, 37.0, 37.0]),
    )
    pv_temp = ProcessVariable(
        name="temperature", unit="°C", is_controlled=True, values=temp_ts
    )
    pv_ph = ProcessVariable(
        name="pH", unit="", is_controlled=False, values=StaticVariable(value=7.0)
    )

    return BioProcess(
        metadata=BioProcessMetadata(name="fed_batch_001", process_type="fed_batch"),
        time_axis=TimeAxis(
            unit="hours", start=0.0, end=48.0, time_reference="inoculation"
        ),
        volume=volume,
        reactor_medium=reactor_medium,
        process_variables={"temperature": pv_temp, "pH": pv_ph},
    )


@pytest.fixture
def sample_collection(sample_process):
    """Build a minimal but realistic BioProcessCollection for serialization tests."""
    return BioProcessCollection(
        metadata=None,
        processes={"fed_batch_001": sample_process},
    )


@pytest.fixture
def sample_collection_with_metadata(sample_process):
    """Build a BioProcessCollection that carries top-level collection metadata."""
    return BioProcessCollection(
        metadata={"source": "raw_lab_export", "instrument": "ambr250"},
        processes={"fed_batch_001": sample_process},
    )


@pytest.fixture
def sample_case_study(sample_process):
    """Build a minimal but realistic case-study BioProcessCollection (case_id/
    organism/citation set) for serialization tests."""
    return BioProcessCollection(
        case_id="ecoli_study",
        organism="Escherichia coli",
        citation="Doe et al. 2024",
        processes={"fed_batch_001": sample_process},
    )


# ---------------------------------------------------------------------------
# Default JSON serialization
# ---------------------------------------------------------------------------


def test_save_process_collection_creates_data_json_in_directory(sample_collection):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_collection"
        save_process_collection(sample_collection, save_path)
        assert (save_path / "data.json").exists()


def test_save_load_process_collection_roundtrip(sample_collection):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_collection"
        save_process_collection(sample_collection, save_path)
        loaded = load_process_collection(save_path)

        assert loaded.metadata is None
        assert "fed_batch_001" in loaded.processes
        assert loaded.processes["fed_batch_001"].metadata.name == "fed_batch_001"


def test_case_study_fields_roundtrip_and_omitted_from_json_when_none(
    sample_process,
):
    """case_id/organism/citation survive a save/load roundtrip when set, and
    are omitted from the JSON entirely (not written as null) when unset."""
    with_fields = BioProcessCollection(
        case_id="ecoli_study",
        organism="Escherichia coli",
        citation="Doe et al. 2024",
        processes={"fed_batch_001": sample_process},
    )
    without_fields = BioProcessCollection(processes={"fed_batch_001": sample_process})

    with tempfile.TemporaryDirectory() as tmpdir:
        with_path = Path(tmpdir) / "with_fields.json"
        without_path = Path(tmpdir) / "without_fields.json"
        save_process_collection(with_fields, with_path)
        save_process_collection(without_fields, without_path)

        loaded_with = load_process_collection(with_path)
        assert loaded_with.case_id == "ecoli_study"
        assert loaded_with.organism == "Escherichia coli"
        assert loaded_with.citation == "Doe et al. 2024"

        loaded_without = load_process_collection(without_path)
        assert loaded_without.case_id is None
        assert loaded_without.organism is None
        assert loaded_without.citation is None

        without_payload = json.loads(without_path.read_text())
        assert "case_id" not in without_payload
        assert "organism" not in without_payload
        assert "citation" not in without_payload


def test_load_process_collection_streams_processes_with_legacy_values(
    sample_collection,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "collection.json"
        sample_collection.metadata = {
            "source": "raw_lab_export",
            "run_ids": jnp.array([1, 2]),
        }
        sample_collection.processes = {
            "first": sample_collection.processes["fed_batch_001"],
            "second": sample_collection.processes["fed_batch_001"],
        }
        save_process_collection(sample_collection, path)

        legacy = serialization._dict_to_process_collection(
            serialization._load_json(path)
        )
        loaded = load_process_collection(path)

        assert list(loaded.processes) == list(legacy.processes) == ["first", "second"]
        assert loaded.metadata["source"] == legacy.metadata["source"]
        np.testing.assert_array_equal(
            loaded.metadata["run_ids"], legacy.metadata["run_ids"]
        )
        for process_id in loaded.processes:
            expected = legacy.processes[process_id]
            actual = loaded.processes[process_id]
            np.testing.assert_array_equal(
                actual.process_variables["temperature"].values.values,
                expected.process_variables["temperature"].values.values,
            )
            np.testing.assert_array_equal(
                actual.reactor_medium.components["biomass"].concentration.times,
                expected.reactor_medium.components["biomass"].concentration.times,
            )
            assert (
                actual.process_variables["pH"].values.value
                == expected.process_variables["pH"].values.value
            )


def test_load_process_collection_accepts_whole_line_comments(sample_collection):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "collection.json"
        separator_value = "before\u2028//must-stay\u2028after"
        sample_collection.metadata = {
            "source": "https://example.com/data",
            "note": separator_value,
        }
        save_process_collection(sample_collection, path)
        serialized = path.read_text(encoding="utf-8").replace("\\u2028", "\u2028")
        path.write_text(
            "// collection\n  // source URL follows\n"
            + serialized
            + "\n// final comment",
            encoding="utf-8",
        )

        loaded = load_process_collection(path)

        assert loaded.metadata == {
            "source": "https://example.com/data",
            "note": separator_value,
        }
        assert "fed_batch_001" in loaded.processes


def test_load_process_collection_accepts_yajl_inline_and_block_comments(
    sample_collection,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "collection.json"
        save_process_collection(sample_collection, path)
        serialized = path.read_text(encoding="utf-8")
        serialized = serialized.replace("{", "{/* block comment */", 1)
        serialized = serialized.replace(
            '"metadata": null,', '"metadata": null, // inline comment', 1
        )
        path.write_text(serialized, encoding="utf-8")

        loaded = load_process_collection(path)

        assert "fed_batch_001" in loaded.processes


def test_save_process_collection_normalizes_nonfinite_values(sample_collection):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "collection.json"
        sample_collection.metadata = {
            "nan": float("nan"),
            "positive": float("inf"),
            "negative": np.float32("-inf"),
        }
        component = sample_collection.processes[
            "fed_batch_001"
        ].reactor_medium.components["biomass"]
        component.concentration = TimeSeries(
            times=jnp.array([0.0, 1.0]),
            values=jnp.array([0.1, jnp.inf]),
        )
        save_process_collection(sample_collection, path)

        text = path.read_text(encoding="utf-8")
        assert "NaN" not in text
        assert "Infinity" not in text
        loaded = load_process_collection(path)

        assert loaded.metadata == {
            "nan": None,
            "positive": None,
            "negative": None,
        }
        values = (
            loaded.processes["fed_batch_001"]
            .reactor_medium.components["biomass"]
            .concentration.values
        )
        assert values[0] == pytest.approx(0.1)
        assert np.isnan(values[1])


def test_restore_arrays_rejects_null_in_integer_and_bool_payloads():
    for dtype in ("int32", "bool"):
        with pytest.raises(ValueError, match="null is invalid"):
            serialization._restore_arrays({"__ndarray__": [1, None], "dtype": dtype})


def test_load_process_collection_converts_each_process_before_reading_next(
    monkeypatch, sample_process
):
    process_data = serialization._process_to_dict(sample_process)
    converted = []
    original = serialization._dict_to_process

    def record_conversion(data):
        converted.append(data["metadata"]["name"])
        return original(data)

    def stream_processes(_file, prefix, *, source):
        assert prefix == "processes"
        assert source.name == "collection.json"
        yield "first", process_data
        assert converted == ["fed_batch_001"]
        yield "second", process_data

    monkeypatch.setattr(serialization, "_dict_to_process", record_conversion)
    monkeypatch.setattr(serialization, "_kvitems", stream_processes)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "collection.json"
        path.write_text('{"metadata": null, "processes": {}}', encoding="utf-8")
        loaded = load_process_collection(path)

    assert list(loaded.processes) == ["first", "second"]
    assert converted == ["fed_batch_001", "fed_batch_001"]


def test_load_process_collection_rejects_nonfinite_tokens_without_fallback(
    monkeypatch, tmp_path
):
    path = tmp_path / "collection.json"
    path.write_text('{"metadata": {"loss": NaN}, "processes": {}}', encoding="utf-8")
    monkeypatch.setattr(
        serialization,
        "_load_json",
        lambda _path: pytest.fail("full-document fallback must not run"),
    )

    with pytest.raises(ValueError, match=str(path)):
        load_process_collection(path)


@pytest.mark.parametrize(
    "document, message",
    [
        ("42", "root must be an object"),
        ('{"metadata": null}', "must contain a processes object"),
        ('{"metadata": null, "processes": []}', "processes must be an object"),
        ('{"metadata": null, "processes": null}', "processes must be an object"),
        ('{"metadata": [], "processes": {}}', "metadata must be an object or null"),
    ],
)
def test_load_process_collection_rejects_invalid_structure(tmp_path, document, message):
    path = tmp_path / "collection.json"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_process_collection(path)


def test_load_process_collection_validates_suffix_and_top_level_key_order(
    sample_collection, tmp_path
):
    path = tmp_path / "collection.json"
    sample_collection.metadata = {"source": "reordered"}
    save_process_collection(sample_collection, path)
    original = json.loads(path.read_text(encoding="utf-8"))
    reordered = {
        "processes": original["processes"],
        "metadata": original["metadata"],
    }
    path.write_text(json.dumps(reordered), encoding="utf-8")

    loaded = load_process_collection(path)
    assert "fed_batch_001" in loaded.processes
    assert loaded.metadata == {"source": "reordered"}

    path.write_text(path.read_text(encoding="utf-8") + " trailing", encoding="utf-8")
    with pytest.raises(ValueError, match=str(path)):
        load_process_collection(path)


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("suffix", [b" /* unterminated", b" 2"])
def test_load_process_collection_rejects_malformed_suffix(tmp_path, compressed, suffix):
    directory = tmp_path / "comments"
    directory.mkdir()
    path = directory / ("collection.json.gz" if compressed else "collection.json")
    payload = b'{"metadata": null, "processes": {}}' + suffix
    if compressed:
        with gzip.open(path, "wb") as stream:
            stream.write(payload)
    else:
        path.write_bytes(payload)

    with pytest.raises(JSONParseError, match=str(path)):
        load_process_collection(path)


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("duplicate_member", ["metadata", "processes"])
def test_load_process_collection_rejects_duplicate_top_level_members(
    tmp_path, sample_process, compressed, duplicate_member
):
    process_json = json.dumps(
        serialization._process_to_dict(sample_process), cls=serialization.NumpyEncoder
    )
    if duplicate_member == "metadata":
        document = '{"metadata":{"version":1},"processes":{},"metadata":{"version":2}}'
    else:
        document = (
            '{"metadata":null,"processes":{"stale":'
            f"{process_json}" + '},"processes":{}}'
        )
    path = tmp_path / ("collection.json.gz" if compressed else "collection.json")
    if compressed:
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(document)
    else:
        path.write_text(document, encoding="utf-8")

    with pytest.raises(JSONParseError, match="duplicate top-level key"):
        load_process_collection(path)


def test_default_api_accepts_explicit_process_collection_json_gz_paths(
    sample_collection,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_collection.json.gz"
        save_process_collection(sample_collection, save_path)
        loaded = load_process_collection(save_path)

        assert save_path.exists()
        assert save_path.is_file()
        assert "fed_batch_001" in loaded.processes


def test_save_load_process_collection_metadata_roundtrip(
    sample_collection_with_metadata,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_collection"
        save_process_collection(sample_collection_with_metadata, save_path)
        loaded = load_process_collection(save_path)

        assert loaded.metadata == {"source": "raw_lab_export", "instrument": "ambr250"}


def test_save_creates_data_json_in_directory(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study"
        save_process_collection(sample_case_study, save_path)
        assert (save_path / "data.json").exists()


def test_save_load_roundtrip_identity(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study"
        save_process_collection(sample_case_study, save_path)
        loaded = load_process_collection(save_path)

        assert loaded.case_id == "ecoli_study"
        assert loaded.organism == "Escherichia coli"
        assert loaded.citation == "Doe et al. 2024"


def test_save_load_roundtrip_structure(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study"
        save_process_collection(sample_case_study, save_path)
        loaded = load_process_collection(save_path)

        assert loaded.case_id == "ecoli_study"
        assert loaded.organism == "Escherichia coli"
        assert "fed_batch_001" in loaded.processes


def test_save_load_roundtrip_process_metadata(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study"
        save_process_collection(sample_case_study, save_path)
        loaded = load_process_collection(save_path)

        proc = loaded.processes["fed_batch_001"]
        assert proc.metadata.name == "fed_batch_001"
        assert proc.metadata.process_type == "fed_batch"


def test_save_load_roundtrip_timeseries(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study"
        save_process_collection(sample_case_study, save_path)
        loaded = load_process_collection(save_path)

        proc = loaded.processes["fed_batch_001"]
        biomass = proc.reactor_medium.components["biomass"]
        assert hasattr(biomass.concentration, "times")
        assert biomass.concentration.times.shape == (5,)
        assert jnp.allclose(
            biomass.concentration.values,
            jnp.array([0.1, 1.2, 3.5, 5.8, 6.0]),
        )


def test_save_load_roundtrip_static_variable(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study"
        save_process_collection(sample_case_study, save_path)
        loaded = load_process_collection(save_path)

        proc = loaded.processes["fed_batch_001"]
        ph = proc.process_variables["pH"]
        assert isinstance(ph.values, StaticVariable)
        assert ph.values.value == pytest.approx(7.0)


def test_save_load_roundtrip_volume(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study"
        save_process_collection(sample_case_study, save_path)
        loaded = load_process_collection(save_path)

        proc = loaded.processes["fed_batch_001"]
        assert proc.volume.initial_volume == pytest.approx(1.0)
        assert "glucose_feed" in proc.volume.volume_changes
        vc = proc.volume.volume_changes["glucose_feed"]
        assert vc.is_continuous is True
        assert vc.values.times.shape == (5,)


def test_save_load_roundtrip_feed_medium(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study"
        save_process_collection(sample_case_study, save_path)
        loaded = load_process_collection(save_path)

        proc = loaded.processes["fed_batch_001"]
        vc = proc.volume.volume_changes["glucose_feed"]
        assert vc.feed_medium is not None
        assert vc.feed_medium.name == "glucose_feed"
        assert "glucose" in vc.feed_medium.components
        assert vc.feed_medium.components[
            "glucose"
        ].concentration.value == pytest.approx(500.0)


def test_save_load_roundtrip_outflow_retention():
    rm = ReactorMedium(
        name="medium",
        components={
            "biomass": ReactorMediumComponent(
                name="biomass",
                unit="g/L",
                concentration=TimeSeries(
                    times=jnp.array([0.0, 1.0]), values=jnp.array([1.0, 2.0])
                ),
            ),
        },
    )
    outflow = Outflow(
        name="sample",
        unit="L",
        is_controlled=True,
        is_continuous=True,
        values=TimeSeries(
            times=jnp.array([0.0, 1.0]), values=jnp.array([-0.1, -0.2])
        ),
        retention={"biomass": 0.95},
    )
    process = BioProcess(
        metadata=BioProcessMetadata(name="p", process_type="fed_batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=1.0, time_reference="x"),
        volume=Volume(
            initial_volume=1.0, unit="L", volume_changes={"sample": outflow}
        ),
        reactor_medium=rm,
    )
    collection = BioProcessCollection(processes={"p": process})

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "collection.json"
        save_process_collection(collection, save_path)
        loaded = load_process_collection(save_path)

    loaded_outflow = loaded.processes["p"].volume.volume_changes["sample"]
    assert loaded_outflow.retention == {"biomass": 0.95}


def test_load_old_outflow_json_without_retention_key_defaults_empty():
    """Old serialized files predating this field must still load, with the
    field defaulting to an empty dict."""
    outflow = serialization._volume_change_to_dict(
        Outflow(
            name="sample",
            unit="L",
            is_controlled=True,
            is_continuous=True,
            values=TimeSeries(
                times=jnp.array([0.0, 1.0]), values=jnp.array([-0.1, -0.2])
            ),
        )
    )
    del outflow["retention"]  # simulate a pre-existing on-disk file
    reconstructed = serialization._dict_to_volume_change(outflow)
    assert reconstructed.retention == {}


# ---------------------------------------------------------------------------
# Explicit JSON path serialization
# ---------------------------------------------------------------------------


def test_json_save_process_collection_creates_file(sample_collection):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_collection.json"
        save_process_collection(sample_collection, save_path)
        assert save_path.exists()


def test_json_process_collection_roundtrip(sample_collection):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_collection.json"
        save_process_collection(sample_collection, save_path)
        loaded = load_process_collection(save_path)

        assert loaded.metadata is None
        assert "fed_batch_001" in loaded.processes
        assert loaded.processes["fed_batch_001"].metadata.process_type == "fed_batch"


def test_json_process_with_optional_metadata_roundtrip(sample_process):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_collection.json"
        process = BioProcess(
            metadata=None,
            time_axis=sample_process.time_axis,
            volume=sample_process.volume,
            reactor_medium=sample_process.reactor_medium,
            process_variables=sample_process.process_variables,
        )
        collection = BioProcessCollection(processes={"fed_batch_001": process})
        save_process_collection(collection, save_path)
        loaded = load_process_collection(save_path)

        assert loaded.processes["fed_batch_001"].metadata is None


def test_json_save_creates_file(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study.json"
        save_process_collection(sample_case_study, save_path)
        assert save_path.exists()


def test_json_roundtrip_identity(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study.json"
        save_process_collection(sample_case_study, save_path)
        loaded = load_process_collection(save_path)

        assert loaded.case_id == "ecoli_study"
        assert loaded.organism == "Escherichia coli"


def test_json_roundtrip_timeseries(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study.json"
        save_process_collection(sample_case_study, save_path)
        loaded = load_process_collection(save_path)

        proc = loaded.processes["fed_batch_001"]
        biomass = proc.reactor_medium.components["biomass"]
        assert biomass.concentration.times.shape == (5,)
        assert jnp.allclose(
            biomass.concentration.values,
            jnp.array([0.1, 1.2, 3.5, 5.8, 6.0]),
        )


def test_default_api_accepts_explicit_json_paths(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study.json"
        save_process_collection(sample_case_study, save_path)
        loaded = load_process_collection(save_path)

        assert save_path.exists()
        assert loaded.case_id == "ecoli_study"


def test_default_api_accepts_explicit_json_gz_paths(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study.json.gz"
        save_process_collection(sample_case_study, save_path)
        loaded = load_process_collection(save_path)

        assert save_path.exists()
        assert save_path.is_file()
        assert loaded.case_id == "ecoli_study"


def test_json_gz_roundtrip_case_study(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study.json.gz"
        save_process_collection(sample_case_study, save_path)
        loaded = load_process_collection(save_path)

        assert loaded.case_id == "ecoli_study"
        with gzip.open(save_path, "rt", encoding="utf-8") as f:
            assert '"case_id"' in f.read()


def test_json_gz_roundtrip_process_collection(sample_collection):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_collection.json.gz"
        save_process_collection(sample_collection, save_path)
        loaded = load_process_collection(save_path)

        assert save_path.is_file()
        assert "fed_batch_001" in loaded.processes
        assert loaded.processes["fed_batch_001"].metadata.process_type == "fed_batch"


def test_default_load_from_directory_accepts_data_json_gz(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study"
        save_process_collection(sample_case_study, save_path / "data.json.gz")

        loaded = load_process_collection(save_path)

        assert loaded.case_id == "ecoli_study"


def test_default_load_process_collection_from_directory_accepts_data_json_gz(
    sample_collection,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_collection"
        save_process_collection(sample_collection, save_path / "data.json.gz")

        loaded = load_process_collection(save_path)

        assert "fed_batch_001" in loaded.processes


def test_default_load_rejects_non_json_file_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "dataset.yaml"
        save_path.write_text("metadata: {}\n")

        with pytest.raises(
            FileNotFoundError, match="Only JSON serialization is supported"
        ):
            load_process_collection(save_path)


# ---------------------------------------------------------------------------
# bounds and biological_ode round-trip
# ---------------------------------------------------------------------------


def _make_process_with_biological_ode_and_bounds(sample_process):
    """Augment the sample process with bounds on every relevant slot and a
    minimal but realistic ``biological_ode`` block for round-trip testing."""
    p = sample_process
    p.volume.bounds = (0.0, 5.0)
    p.reactor_medium.components["biomass"].bounds = (0.0, None)
    p.reactor_medium.components["glucose"].bounds = (0.0, 500.0)
    # Both controlled and uncontrolled PVs get bounds
    for pv in p.process_variables.values():
        if pv.is_controlled:
            pv.bounds = (0.0, 14.0)
        else:
            pv.bounds = (None, 100.0)
    p.biological_ode = BiologicalOde(
        algebraic={"X_active": "biomass"},
        rates={
            "q_X": (0.0, None),
            "q_S": (None, 0.0),
            "q_unused": (None, None),
        },
        derivatives={"biomass": "q_X * X_active", "glucose": "q_S * X_active"},
    )
    return p


def test_json_roundtrip_bounds_on_every_slot(sample_process):
    """Bounds on reactor components, PVs, volume, and rates round-trip
    losslessly. The unbounded default ``(None, None)`` is omitted from JSON."""
    _make_process_with_biological_ode_and_bounds(sample_process)
    cs = BioProcessCollection(
        case_id="b",
        organism="o",
        citation="c",
        processes={"fed_batch_001": sample_process},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_process_collection(cs, Path(tmpdir) / "d.json")
        loaded = load_process_collection(Path(tmpdir) / "d.json")

    p2 = loaded.processes["fed_batch_001"]
    assert p2.volume.bounds == (0.0, 5.0)
    assert p2.reactor_medium.components["biomass"].bounds == (0.0, None)
    assert p2.reactor_medium.components["glucose"].bounds == (0.0, 500.0)
    for pv in p2.process_variables.values():
        if pv.is_controlled:
            assert pv.bounds == (0.0, 14.0)
        else:
            assert pv.bounds == (None, 100.0)


def test_json_roundtrip_biological_ode(sample_process):
    """biological_ode block round-trips losslessly: derived / derivatives /
    rates (with per-rate bounds)."""
    _make_process_with_biological_ode_and_bounds(sample_process)
    cs = BioProcessCollection(
        case_id="b",
        organism="o",
        citation="c",
        processes={"fed_batch_001": sample_process},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_process_collection(cs, Path(tmpdir) / "d.json")
        loaded = load_process_collection(Path(tmpdir) / "d.json")

    p2 = loaded.processes["fed_batch_001"]
    assert p2.biological_ode is not None
    assert p2.biological_ode.algebraic == {"X_active": "biomass"}
    assert p2.biological_ode.derivatives == {
        "biomass": "q_X * X_active",
        "glucose": "q_S * X_active",
    }
    assert set(p2.biological_ode.rates.keys()) == {"q_X", "q_S", "q_unused"}
    assert p2.biological_ode.rates["q_X"] == (0.0, None)
    assert p2.biological_ode.rates["q_S"] == (None, 0.0)
    assert p2.biological_ode.rates["q_unused"] == (None, None)


def test_rmc_bounds_default_missing_key_and_explicit_unbounded_roundtrip(
    sample_process,
):
    """RMC bounds distinguish the nonnegative default from explicit unbounded."""
    sample_process.reactor_medium.components["glucose"].bounds = (None, None)
    cs = BioProcessCollection(
        case_id="b",
        organism="o",
        citation="c",
        processes={"fed_batch_001": sample_process},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "d.json"
        save_process_collection(cs, path)
        payload = json.loads(path.read_text())
        components = payload["processes"]["fed_batch_001"]["reactor_medium"][
            "components"
        ]
        assert "bounds" not in components["biomass"]
        assert components["glucose"]["bounds"] is None
        assert "bounds" not in payload["processes"]["fed_batch_001"]["volume"]
        assert all(
            "bounds" not in variable
            for variable in payload["processes"]["fed_batch_001"][
                "process_variables"
            ].values()
        )
        loaded_components = (
            load_process_collection(path).processes["fed_batch_001"].reactor_medium.components
        )
        assert loaded_components["glucose"].bounds == (None, None)
        assert loaded_components["biomass"].bounds == (0.0, None)


def test_auto_generated_biological_ode_roundtrips(sample_process):
    """Processes without a user-supplied block get one auto-populated in
    ``BioProcess.__post_init__``; the auto block round-trips losslessly."""
    cs = BioProcessCollection(
        case_id="b",
        organism="o",
        citation="c",
        processes={"fed_batch_001": sample_process},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_process_collection(cs, Path(tmpdir) / "d.json")
        loaded = load_process_collection(Path(tmpdir) / "d.json")

    p2 = loaded.processes["fed_batch_001"]
    assert p2.biological_ode is not None
    assert sample_process.biological_ode is not None
    assert list(p2.biological_ode.rates.keys()) == list(
        sample_process.biological_ode.rates.keys()
    )
    assert p2.biological_ode.derivatives == sample_process.biological_ode.derivatives


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
