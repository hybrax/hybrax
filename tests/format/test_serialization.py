"""
Tests for bpbench.serialization functionality (current architecture).
"""

import pytest
import jax.numpy as jnp
from pathlib import Path
import tempfile

from bpbench import (
    BenchmarkDataset,
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
    VolumeChange,
    Volume,
)
from bpbench.serialization import (
    save_dataset,
    load_dataset,
    save_dataset_json,
    load_dataset_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_dataset():
    """Build a minimal but realistic BenchmarkDataset for serialization tests."""
    biomass_ts = TimeSeries(
        timepoints=jnp.array([0., 12., 24., 36., 48.]),
        values=jnp.array([0.1, 1.2, 3.5, 5.8, 6.0]),
    )
    glucose_ts = TimeSeries(
        timepoints=jnp.array([0., 12., 24., 36., 48.]),
        values=jnp.array([20.0, 15.0, 8.0, 2.0, 0.5]),
    )
    biomass_rc = ReactorMediumComponent(
        name="biomass", unit="g/L",
        concentration=biomass_ts,
        is_intracellular=False,
    )
    glucose_rc = ReactorMediumComponent(
        name="glucose", unit="g/L",
        concentration=glucose_ts,
        is_intracellular=False,
    )
    reactor_medium = ReactorMedium(
        name="medium", density=1.0, density_unit="kg/L",
        components={"biomass": biomass_rc, "glucose": glucose_rc},
    )

    feed_comp = FeedMediumComponent(
        name="glucose", unit="g/L",
        concentration=StaticVariable(value=500.0),
        is_controlled=True,
    )
    feed_medium = FeedMedium(
        name="glucose_feed", density=1.1, density_unit="kg/L",
        components={"glucose": feed_comp},
    )
    feed_ts = TimeSeries(
        timepoints=jnp.array([0., 12., 24., 36., 48.]),
        values=jnp.array([0.0, 0.05, 0.10, 0.15, 0.20]),
    )
    volume_change = VolumeChange(
        name="glucose_feed", unit="L",
        is_controlled=True, is_continuous=True,
        feed_medium=feed_medium,
        values=feed_ts,
    )
    volume = Volume(
        initial_volume=1.0, unit="L",
        volume_changes={"glucose_feed": volume_change},
    )

    temp_ts = TimeSeries(
        timepoints=jnp.array([0., 12., 24.]),
        values=jnp.array([37.0, 37.0, 37.0]),
    )
    pv_temp = ProcessVariable(
        name="temperature", unit="°C", is_controlled=True, values=temp_ts
    )
    pv_ph = ProcessVariable(
        name="pH", unit="", is_controlled=False, values=StaticVariable(value=7.0)
    )

    process = BioProcess(
        metadata=BioProcessMetadata(name="fed_batch_001", process_type="fed_batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=48.0, time_reference="inoculation"),
        volume=volume,
        reactor_medium=reactor_medium,
        process_variables={"temperature": pv_temp, "pH": pv_ph},
    )

    case_study = CaseStudy(
        case_id="ecoli_study",
        organism="Escherichia coli",
        citation="Doe et al. 2024",
        processes={"fed_batch_001": process},
    )

    return BenchmarkDataset(
        metadata={"name": "Test Dataset", "version": "0.1.0",
                  "description": "Test dataset for serialization"},
        case_studies={"ecoli": case_study},
    )


# ---------------------------------------------------------------------------
# YAML + HDF5 serialization
# ---------------------------------------------------------------------------

def test_save_creates_files(sample_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset"
        save_dataset(sample_dataset, save_path)
        assert (save_path / "metadata.yaml").exists()
        assert (save_path / "arrays.h5").exists()


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
        assert hasattr(biomass.concentration, "timepoints")
        assert biomass.concentration.timepoints.shape == (5,)
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
        assert vc.values.timepoints.shape == (5,)


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
        assert vc.feed_medium.components["glucose"].concentration.value == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

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
        assert biomass.concentration.timepoints.shape == (5,)
        assert jnp.allclose(
            biomass.concentration.values,
            jnp.array([0.1, 1.2, 3.5, 5.8, 6.0]),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
