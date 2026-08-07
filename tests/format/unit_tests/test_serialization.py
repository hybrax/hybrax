"""Tests for bp_format.serialization functionality."""

import gzip
import json
import pytest
import jax.numpy as jnp
from pathlib import Path
import tempfile

from bp_format import (
    BiologicalOde,
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
    save_case_study,
    save_process_collection,
    load_case_study,
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
def sample_case_study(sample_process):
    """Build a minimal but realistic CaseStudy for serialization tests."""
    return CaseStudy(
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
        save_case_study(sample_case_study, save_path)
        assert (save_path / "data.json").exists()


def test_save_load_roundtrip_identity(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study"
        save_case_study(sample_case_study, save_path)
        loaded = load_case_study(save_path)

        assert loaded.case_id == "ecoli_study"
        assert loaded.organism == "Escherichia coli"
        assert loaded.citation == "Doe et al. 2024"


def test_save_load_roundtrip_structure(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study"
        save_case_study(sample_case_study, save_path)
        loaded = load_case_study(save_path)

        assert loaded.case_id == "ecoli_study"
        assert loaded.organism == "Escherichia coli"
        assert "fed_batch_001" in loaded.processes


def test_save_load_roundtrip_process_metadata(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study"
        save_case_study(sample_case_study, save_path)
        loaded = load_case_study(save_path)

        proc = loaded.processes["fed_batch_001"]
        assert proc.metadata.name == "fed_batch_001"
        assert proc.metadata.process_type == "fed_batch"


def test_save_load_roundtrip_timeseries(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study"
        save_case_study(sample_case_study, save_path)
        loaded = load_case_study(save_path)

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
        save_case_study(sample_case_study, save_path)
        loaded = load_case_study(save_path)

        proc = loaded.processes["fed_batch_001"]
        ph = proc.process_variables["pH"]
        assert isinstance(ph.values, StaticVariable)
        assert ph.values.value == pytest.approx(7.0)


def test_save_load_roundtrip_volume(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study"
        save_case_study(sample_case_study, save_path)
        loaded = load_case_study(save_path)

        proc = loaded.processes["fed_batch_001"]
        assert proc.volume.initial_volume == pytest.approx(1.0)
        assert "glucose_feed" in proc.volume.volume_changes
        vc = proc.volume.volume_changes["glucose_feed"]
        assert vc.is_continuous is True
        assert vc.values.times.shape == (5,)


def test_save_load_roundtrip_feed_medium(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study"
        save_case_study(sample_case_study, save_path)
        loaded = load_case_study(save_path)

        proc = loaded.processes["fed_batch_001"]
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
        save_case_study(sample_case_study, save_path)
        assert save_path.exists()


def test_json_roundtrip_identity(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study.json"
        save_case_study(sample_case_study, save_path)
        loaded = load_case_study(save_path)

        assert loaded.case_id == "ecoli_study"
        assert loaded.organism == "Escherichia coli"


def test_json_roundtrip_timeseries(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study.json"
        save_case_study(sample_case_study, save_path)
        loaded = load_case_study(save_path)

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
        save_case_study(sample_case_study, save_path)
        loaded = load_case_study(save_path)

        assert save_path.exists()
        assert loaded.case_id == "ecoli_study"


def test_default_api_accepts_explicit_json_gz_paths(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study.json.gz"
        save_case_study(sample_case_study, save_path)
        loaded = load_case_study(save_path)

        assert save_path.exists()
        assert save_path.is_file()
        assert loaded.case_id == "ecoli_study"


def test_json_gz_roundtrip_case_study(sample_case_study):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_case_study.json.gz"
        save_case_study(sample_case_study, save_path)
        loaded = load_case_study(save_path)

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
        save_case_study(sample_case_study, save_path / "data.json.gz")

        loaded = load_case_study(save_path)

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
            load_case_study(save_path)


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
    cs = CaseStudy(
        case_id="b",
        organism="o",
        citation="c",
        processes={"fed_batch_001": sample_process},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_case_study(cs, Path(tmpdir) / "d.json")
        loaded = load_case_study(Path(tmpdir) / "d.json")

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
    cs = CaseStudy(
        case_id="b",
        organism="o",
        citation="c",
        processes={"fed_batch_001": sample_process},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_case_study(cs, Path(tmpdir) / "d.json")
        loaded = load_case_study(Path(tmpdir) / "d.json")

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
    cs = CaseStudy(
        case_id="b",
        organism="o",
        citation="c",
        processes={"fed_batch_001": sample_process},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "d.json"
        save_case_study(cs, path)
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
            load_case_study(path).processes["fed_batch_001"].reactor_medium.components
        )
        assert loaded_components["glucose"].bounds == (None, None)
        assert loaded_components["biomass"].bounds == (0.0, None)


def test_auto_generated_biological_ode_roundtrips(sample_process):
    """Processes without a user-supplied block get one auto-populated in
    ``BioProcess.__post_init__``; the auto block round-trips losslessly."""
    cs = CaseStudy(
        case_id="b",
        organism="o",
        citation="c",
        processes={"fed_batch_001": sample_process},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_case_study(cs, Path(tmpdir) / "d.json")
        loaded = load_case_study(Path(tmpdir) / "d.json")

    p2 = loaded.processes["fed_batch_001"]
    assert p2.biological_ode is not None
    assert sample_process.biological_ode is not None
    assert list(p2.biological_ode.rates.keys()) == list(
        sample_process.biological_ode.rates.keys()
    )
    assert p2.biological_ode.derivatives == sample_process.biological_ode.derivatives


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
