"""
Tests for serialization functionality
"""

import pytest
import jax.numpy as jnp
from pathlib import Path
import tempfile

from bpbench import (
    BenchmarkDataset, CaseStudy, Process,
    TimeSeries, RawTimeSeries, TimeAxis,
    ReactorProperties, Feed, FeedComponent,
    StaticVariable,
    save_dataset, load_dataset,
    save_dataset_json, load_dataset_json
)


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing"""
    time_axis = TimeAxis(
        unit="hours",
        start=0.0,
        end=48.0,
        time_reference="inoculation"
    )
    
    biomass = TimeSeries(
        name="Biomass",
        canonical_name="biomass",
        unit="g/L",
        role="state",
        raw=RawTimeSeries(
            timepoints=jnp.array([0., 12., 24., 36., 48.]),
            values=jnp.array([0.1, 1.2, 3.5, 5.8, 6.0])
        )
    )
    
    feed = Feed(
        name="Glucose feed",
        density=1.1,
        density_unit="kg/L",
        components={
            "glucose": FeedComponent(concentration=500.0, unit="g/L")
        }
    )
    
    process = Process(
        process_id="batch_001",
        process_type="batch",
        time=time_axis,
        dynamic_variables={"biomass": biomass},
        feeds={"feed1": feed},
        static_variables={"mu_max": StaticVariable(value=0.5, unit="1/h")},
        reactor=ReactorProperties(
            working_volume=1.0,
            volume_unit="L"
        )
    )
    
    case_study = CaseStudy(
        case_id="ecoli_study",
        organism="Escherichia coli",
        citation="Doe et al. 2024",
        processes={"batch_001": process}
    )
    
    dataset = BenchmarkDataset(
        metadata={
            "name": "Test Dataset",
            "version": "0.1.0",
            "description": "Test dataset for serialization"
        },
        case_studies={"ecoli": case_study}
    )
    
    return dataset


def test_save_load_yaml_hdf5(sample_dataset):
    """Test YAML + HDF5 serialization"""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset"
        
        # Save
        save_dataset(sample_dataset, save_path)
        
        # Check files exist
        assert (save_path / "metadata.yaml").exists()
        assert (save_path / "arrays.h5").exists()
        
        # Load
        loaded_dataset = load_dataset(save_path)
        
        # Verify metadata
        assert loaded_dataset.metadata["name"] == "Test Dataset"
        assert loaded_dataset.metadata["version"] == "0.1.0"
        
        # Verify structure
        assert "ecoli" in loaded_dataset.case_studies
        case_study = loaded_dataset.case_studies["ecoli"]
        assert case_study.case_id == "ecoli_study"
        assert case_study.organism == "Escherichia coli"
        
        # Verify process
        assert "batch_001" in case_study.processes
        process = case_study.processes["batch_001"]
        assert process.process_type == "batch"
        
        # Verify timeseries data
        assert "biomass" in process.dynamic_variables
        biomass = process.dynamic_variables["biomass"]
        assert biomass.name == "Biomass"
        assert biomass.raw is not None
        assert biomass.raw.timepoints.shape == (5,)


def test_save_load_json(sample_dataset):
    """Test JSON serialization"""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset.json"
        
        # Save
        save_dataset_json(sample_dataset, save_path)
        
        # Check file exists
        assert save_path.exists()
        
        # Load
        loaded_dataset = load_dataset_json(save_path)
        
        # Verify metadata
        assert loaded_dataset.metadata["name"] == "Test Dataset"
        
        # Verify structure
        assert "ecoli" in loaded_dataset.case_studies
        case_study = loaded_dataset.case_studies["ecoli"]
        assert case_study.organism == "Escherichia coli"


def test_feed_serialization(sample_dataset):
    """Test that feed data is properly serialized and loaded"""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset"
        
        save_dataset(sample_dataset, save_path)
        loaded_dataset = load_dataset(save_path)
        
        process = loaded_dataset.case_studies["ecoli"].processes["batch_001"]
        
        # Verify feed
        assert "feed1" in process.feeds
        feed = process.feeds["feed1"]
        assert feed.name == "Glucose feed"
        assert feed.density == 1.1
        assert "glucose" in feed.components
        assert feed.components["glucose"].concentration == 500.0


def test_static_variables_serialization(sample_dataset):
    """Test that static variables are properly serialized and loaded"""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dataset"
        
        save_dataset(sample_dataset, save_path)
        loaded_dataset = load_dataset(save_path)
        
        process = loaded_dataset.case_studies["ecoli"].processes["batch_001"]
        
        # Verify static variable
        assert "mu_max" in process.static_variables
        mu_max = process.static_variables["mu_max"]
        assert mu_max.value == 0.5
        assert mu_max.unit == "1/h"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
