"""Tests for bp_format.serialization functionality."""

import gzip
import pytest
import jax.numpy as jnp
from pathlib import Path
import tempfile

from bp_format import (
    BenchmarkDataset,
    BioProcessCollection,
    CaseStudy,
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
    FeedVolumeChange,
    Volume,
)
from bp_format.serialization import (
    save_dataset,
    save_process_collection,
    load_dataset,
    load_process_collection,
    save_dataset_json,
    save_process_collection_json,
    load_dataset_json,
    load_process_collection_json,
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
    volume_change = FeedVolumeChange(
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
def sample_dataset(sample_process):
    """Build a minimal but realistic BenchmarkDataset for serialization tests."""
    case_study = CaseStudy(
        case_id="ecoli_study",
        organism="Escherichia coli",
        citation="Doe et al. 2024",
        processes={"fed_batch_001": sample_process},
    )

    return BenchmarkDataset(
        metadata={
            "name": "Test Dataset",
            "version": "0.1.0",
            "description": "Test dataset for serialization",
        },
        case_studies={"ecoli": case_study},
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


def test_save_creates_data_json_in_directory(sample_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset"
        save_dataset(sample_dataset, save_path)
        assert (save_path / "data.json").exists()


def test_save_load_roundtrip_metadata(sample_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset"
        save_dataset(sample_dataset, save_path)
        loaded = load_dataset(save_path)

        assert loaded.metadata["name"] == "Test Dataset"
        assert loaded.metadata["version"] == "0.1.0"


def test_save_load_roundtrip_structure(sample_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset"
        save_dataset(sample_dataset, save_path)
        loaded = load_dataset(save_path)

        assert "ecoli" in loaded.case_studies
        cs = loaded.case_studies["ecoli"]
        assert cs.case_id == "ecoli_study"
        assert cs.organism == "Escherichia coli"
        assert "fed_batch_001" in cs.processes


def test_save_load_roundtrip_process_metadata(sample_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset"
        save_dataset(sample_dataset, save_path)
        loaded = load_dataset(save_path)

        proc = loaded.case_studies["ecoli"].processes["fed_batch_001"]
        assert proc.metadata.name == "fed_batch_001"
        assert proc.metadata.process_type == "fed_batch"


def test_save_load_roundtrip_timeseries(sample_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset"
        save_dataset(sample_dataset, save_path)
        loaded = load_dataset(save_path)

        proc = loaded.case_studies["ecoli"].processes["fed_batch_001"]
        biomass = proc.reactor_medium.components["biomass"]
        assert hasattr(biomass.concentration, "times")
        assert biomass.concentration.times.shape == (5,)
        assert jnp.allclose(
            biomass.concentration.values,
            jnp.array([0.1, 1.2, 3.5, 5.8, 6.0]),
        )


def test_save_load_roundtrip_static_variable(sample_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset"
        save_dataset(sample_dataset, save_path)
        loaded = load_dataset(save_path)

        proc = loaded.case_studies["ecoli"].processes["fed_batch_001"]
        ph = proc.process_variables["pH"]
        assert isinstance(ph.values, StaticVariable)
        assert ph.values.value == pytest.approx(7.0)


def test_save_load_roundtrip_volume(sample_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset"
        save_dataset(sample_dataset, save_path)
        loaded = load_dataset(save_path)

        proc = loaded.case_studies["ecoli"].processes["fed_batch_001"]
        assert proc.volume.initial_volume == pytest.approx(1.0)
        assert "glucose_feed" in proc.volume.volume_changes
        vc = proc.volume.volume_changes["glucose_feed"]
        assert vc.is_continuous is True
        assert vc.values.times.shape == (5,)


def test_save_load_roundtrip_feed_medium(sample_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset"
        save_dataset(sample_dataset, save_path)
        loaded = load_dataset(save_path)

        proc = loaded.case_studies["ecoli"].processes["fed_batch_001"]
        vc = proc.volume.volume_changes["glucose_feed"]
        assert vc.feed_medium is not None
        assert vc.feed_medium.name == "glucose_feed"
        assert "glucose" in vc.feed_medium.components
        assert vc.feed_medium.components[
            "glucose"
        ].concentration.value == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Explicit JSON path serialization
# ---------------------------------------------------------------------------


def test_json_save_process_collection_creates_file(sample_collection):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_collection.json"
        save_process_collection_json(sample_collection, save_path)
        assert save_path.exists()


def test_json_process_collection_roundtrip(sample_collection):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_collection.json"
        save_process_collection_json(sample_collection, save_path)
        loaded = load_process_collection_json(save_path)

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
        save_process_collection_json(collection, save_path)
        loaded = load_process_collection_json(save_path)

        assert loaded.processes["fed_batch_001"].metadata is None


def test_json_save_creates_file(sample_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset.json"
        save_dataset_json(sample_dataset, save_path)
        assert save_path.exists()


def test_json_roundtrip_metadata(sample_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset.json"
        save_dataset_json(sample_dataset, save_path)
        loaded = load_dataset_json(save_path)

        assert loaded.metadata["name"] == "Test Dataset"
        assert "ecoli" in loaded.case_studies
        assert loaded.case_studies["ecoli"].organism == "Escherichia coli"


def test_json_roundtrip_timeseries(sample_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset.json"
        save_dataset_json(sample_dataset, save_path)
        loaded = load_dataset_json(save_path)

        proc = loaded.case_studies["ecoli"].processes["fed_batch_001"]
        biomass = proc.reactor_medium.components["biomass"]
        assert biomass.concentration.times.shape == (5,)
        assert jnp.allclose(
            biomass.concentration.values,
            jnp.array([0.1, 1.2, 3.5, 5.8, 6.0]),
        )


def test_default_api_accepts_explicit_json_paths(sample_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset.json"
        save_dataset(sample_dataset, save_path)
        loaded = load_dataset(save_path)

        assert save_path.exists()
        assert loaded.metadata["name"] == "Test Dataset"


def test_default_api_accepts_explicit_json_gz_paths(sample_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset.json.gz"
        save_dataset(sample_dataset, save_path)
        loaded = load_dataset(save_path)

        assert save_path.exists()
        assert save_path.is_file()
        assert loaded.metadata["name"] == "Test Dataset"


def test_json_gz_roundtrip_dataset(sample_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset.json.gz"
        save_dataset_json(sample_dataset, save_path)
        loaded = load_dataset_json(save_path)

        assert loaded.metadata["name"] == "Test Dataset"
        with gzip.open(save_path, "rt", encoding="utf-8") as f:
            assert '"case_studies"' in f.read()


def test_json_gz_roundtrip_process_collection(sample_collection):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_collection.json.gz"
        save_process_collection_json(sample_collection, save_path)
        loaded = load_process_collection_json(save_path)

        assert save_path.is_file()
        assert "fed_batch_001" in loaded.processes
        assert loaded.processes["fed_batch_001"].metadata.process_type == "fed_batch"


def test_default_load_from_directory_accepts_data_json_gz(sample_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset"
        save_dataset_json(sample_dataset, save_path / "data.json.gz")

        loaded = load_dataset(save_path)

        assert loaded.metadata["name"] == "Test Dataset"


def test_default_load_process_collection_from_directory_accepts_data_json_gz(
    sample_collection,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_collection"
        save_process_collection_json(sample_collection, save_path / "data.json.gz")

        loaded = load_process_collection(save_path)

        assert "fed_batch_001" in loaded.processes


def test_default_load_rejects_non_json_file_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "dataset.yaml"
        save_path.write_text("metadata: {}\n")

        with pytest.raises(
            FileNotFoundError, match="Only JSON serialization is supported"
        ):
            load_dataset(save_path)


# ---------------------------------------------------------------------------
# bounds and biological_ode round-trip
# ---------------------------------------------------------------------------


from bp_format import BiologicalOde, RateDecl


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
            "q_X": RateDecl(bounds=(0.0, None)),
            "q_S": RateDecl(bounds=(None, 0.0)),
            "q_unused": RateDecl(),
        },
        derivatives={"biomass": "q_X * X_active", "glucose": "q_S * X_active"},
    )
    return p


def test_json_roundtrip_bounds_on_every_slot(sample_process):
    """Bounds on reactor components, PVs, volume, and rates round-trip
    losslessly. The unbounded default ``(None, None)`` is omitted from JSON."""
    _make_process_with_biological_ode_and_bounds(sample_process)
    cs = CaseStudy(
        case_id="b", organism="o", citation="c", processes={"fed_batch_001": sample_process}
    )
    ds = BenchmarkDataset(metadata={"name": "B"}, case_studies={"b": cs})

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dataset_json(ds, Path(tmpdir) / "d.json")
        loaded = load_dataset_json(Path(tmpdir) / "d.json")

    p2 = loaded.case_studies["b"].processes["fed_batch_001"]
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
    cs = CaseStudy(
        case_id="b", organism="o", citation="c", processes={"fed_batch_001": sample_process}
    )
    ds = BenchmarkDataset(metadata={"name": "B"}, case_studies={"b": cs})

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dataset_json(ds, Path(tmpdir) / "d.json")
        loaded = load_dataset_json(Path(tmpdir) / "d.json")

    p2 = loaded.case_studies["b"].processes["fed_batch_001"]
    assert p2.biological_ode is not None
    assert p2.biological_ode.algebraic == {"X_active": "biomass"}
    assert p2.biological_ode.derivatives == {
        "biomass": "q_X * X_active",
        "glucose": "q_S * X_active",
    }
    assert set(p2.biological_ode.rates.keys()) == {"q_X", "q_S", "q_unused"}
    assert p2.biological_ode.rates["q_X"].bounds == (0.0, None)
    assert p2.biological_ode.rates["q_S"].bounds == (None, 0.0)
    assert p2.biological_ode.rates["q_unused"].bounds == (None, None)


def test_default_unbounded_is_omitted_from_json(sample_process):
    """A freshly built process without bounds writes no bounds keys to JSON."""
    cs = CaseStudy(
        case_id="b", organism="o", citation="c", processes={"fed_batch_001": sample_process}
    )
    ds = BenchmarkDataset(metadata={"name": "B"}, case_studies={"b": cs})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "d.json"
        save_dataset_json(ds, path)
        text = path.read_text()
        assert '"bounds"' not in text


def test_auto_generated_biological_ode_roundtrips(sample_process):
    """Processes without a user-supplied block get one auto-populated in
    ``BioProcess.__post_init__``; the auto block round-trips losslessly."""
    cs = CaseStudy(
        case_id="b", organism="o", citation="c", processes={"fed_batch_001": sample_process}
    )
    ds = BenchmarkDataset(metadata={"name": "B"}, case_studies={"b": cs})

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dataset_json(ds, Path(tmpdir) / "d.json")
        loaded = load_dataset_json(Path(tmpdir) / "d.json")

    p2 = loaded.case_studies["b"].processes["fed_batch_001"]
    assert p2.biological_ode is not None
    assert sample_process.biological_ode is not None
    assert (
        list(p2.biological_ode.rates.keys())
        == list(sample_process.biological_ode.rates.keys())
    )
    assert p2.biological_ode.derivatives == sample_process.biological_ode.derivatives


def test_serialized_reactor_component_omits_is_intracellular(sample_dataset):
    """The legacy ``is_intracellular`` flag was purged from the data
    model and must not appear in newly written JSON."""
    import json
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "data.json"
        save_dataset_json(sample_dataset, save_path)
        with open(save_path) as f:
            raw = f.read()
    assert "is_intracellular" not in raw


def test_load_tolerates_legacy_is_intracellular_field(sample_dataset):
    """Pre-purge JSON files contain ``is_intracellular`` on every reactor
    component. The deserializer ignores the field instead of crashing,
    so existing on-disk artefacts continue to load."""
    import json
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "data.json"
        save_dataset_json(sample_dataset, save_path)
        with open(save_path) as f:
            payload = json.load(f)

        # Inject the legacy field into every reactor component before reload.
        for cs in payload["case_studies"].values():
            for proc in cs["processes"].values():
                for comp in proc["reactor_medium"]["components"].values():
                    comp["is_intracellular"] = True

        with open(save_path, "w") as f:
            json.dump(payload, f)

        loaded = load_dataset_json(save_path)

    proc = loaded.case_studies["ecoli"].processes["fed_batch_001"]
    # Field is gone from the dataclass entirely; the legacy JSON entry
    # was just dropped on the floor.
    assert not hasattr(proc.reactor_medium.components["biomass"], "is_intracellular")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
