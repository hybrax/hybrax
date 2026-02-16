"""
Serialization utilities for bioprocess benchmarking dataset
Hybrid approach: YAML for metadata + HDF5 for arrays

NOTE: This module needs to be updated to work with the new data structure.
For now, serialization functions are disabled.
"""

import yaml
import h5py
import json
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
from .dataclasses import (
    BenchmarkDataset, CaseStudy, BioProcess, TimeSeries, TimeAxis,
    SplineRepresentation, FeedMedium, FeedMediumComponent,
    StaticVariable, BioProcessMetadata, Volume, VolumeChange,
    ReactorMedium, ReactorMediumComponent, ProcessVariable
)


# ============================================================
# Serialization to YAML + HDF5
# ============================================================

def save_dataset(dataset: BenchmarkDataset, base_path: Path) -> None:
    """
    Save dataset using hybrid approach - NOT YET UPDATED FOR NEW STRUCTURE
    
    TODO: Update this function to work with the new data structure:
    - ProcessVariable instead of TimeSeries with name/unit/controlled
    - ReactorMedium and ReactorMediumComponent
    - FeedMediumComponent with concentration as TimeSeries | StaticVariable
    - time_axis instead of time
    - reactor_medium and process_variables instead of dynamic_variables/static_variables
    """
    raise NotImplementedError("save_dataset needs to be updated for the new data structure")


def load_dataset(base_path: Path) -> BenchmarkDataset:
    """
    Load dataset from YAML + HDF5 - NOT YET UPDATED FOR NEW STRUCTURE
    
    TODO: Update this function to work with the new data structure
    """
    raise NotImplementedError("load_dataset needs to be updated for the new data structure")


def save_dataset_json(dataset: BenchmarkDataset, json_path: Path) -> None:
    """
    Save dataset to JSON - NOT YET UPDATED FOR NEW STRUCTURE
    
    TODO: Update this function to work with the new data structure
    """
    raise NotImplementedError("save_dataset_json needs to be updated for the new data structure")


def load_dataset_json(json_path: Path) -> BenchmarkDataset:
    """
    Load dataset from JSON - NOT YET UPDATED FOR NEW STRUCTURE
    
    TODO: Update this function to work with the new data structure
    """
    raise NotImplementedError("load_dataset_json needs to be updated for the new data structure")
