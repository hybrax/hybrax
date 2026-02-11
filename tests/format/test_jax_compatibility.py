"""
Test JAX compatibility and PyTree functionality
"""

import pytest
import jax
import jax.numpy as jnp
from jax import tree_util

from bpbench import (
    TimeAxis, RawTimeSeries, TimeSeries,
    StaticVariable, ReactorProperties, Process
)


def test_timeaxis_pytree():
    """Test TimeAxis is a valid PyTree"""
    time_axis = TimeAxis(
        unit="hours",
        start=0.0,
        end=48.0,
        time_reference="inoculation"
    )
    
    # Check it can be flattened and unflattened
    leaves, treedef = tree_util.tree_flatten(time_axis)
    reconstructed = tree_util.tree_unflatten(treedef, leaves)
    
    assert reconstructed.unit == time_axis.unit
    assert reconstructed.start == time_axis.start
    assert reconstructed.end == time_axis.end
    assert reconstructed.time_reference == time_axis.time_reference


def test_raw_timeseries_pytree():
    """Test RawTimeSeries is a valid PyTree"""
    raw = RawTimeSeries(
        timepoints=jnp.array([0., 12., 24.]),
        values=jnp.array([0.1, 1.2, 3.5])
    )
    
    # Check it can be flattened and unflattened
    leaves, treedef = tree_util.tree_flatten(raw)
    reconstructed = tree_util.tree_unflatten(treedef, leaves)
    
    assert jnp.allclose(reconstructed.timepoints, raw.timepoints)
    assert jnp.allclose(reconstructed.values, raw.values)


def test_timeseries_pytree():
    """Test TimeSeries is a valid PyTree"""
    raw = RawTimeSeries(
        timepoints=jnp.array([0., 12., 24.]),
        values=jnp.array([0.1, 1.2, 3.5])
    )
    
    timeseries = TimeSeries(
        name="Biomass",
        unit="g/L",
        role="state",
        raw=raw
    )
    
    # Check it can be flattened and unflattened
    leaves, treedef = tree_util.tree_flatten(timeseries)
    reconstructed = tree_util.tree_unflatten(treedef, leaves)
    
    assert reconstructed.name == timeseries.name
    assert reconstructed.unit == timeseries.unit
    assert jnp.allclose(reconstructed.raw.values, timeseries.raw.values)


def test_jax_transformations():
    """Test that PyTrees work with JAX transformations"""
    raw = RawTimeSeries(
        timepoints=jnp.array([0., 12., 24.]),
        values=jnp.array([0.1, 1.2, 3.5])
    )
    
    # Define a simple function
    def compute_mean(raw_ts):
        return jnp.mean(raw_ts.values)
    
    # Test basic computation
    result = compute_mean(raw)
    assert isinstance(result, jnp.ndarray)
    
    # Test with jit
    jit_compute = jax.jit(compute_mean)
    result_jit = jit_compute(raw)
    assert jnp.allclose(result, result_jit)


def test_process_pytree():
    """Test Process is a valid PyTree"""
    time_axis = TimeAxis(
        unit="hours",
        start=0.0,
        end=48.0,
        time_reference="inoculation"
    )
    
    biomass = TimeSeries(
        name="Biomass",
        unit="g/L",
        role="state",
        raw=RawTimeSeries(
            timepoints=jnp.array([0., 12., 24.]),
            values=jnp.array([0.1, 1.2, 3.5])
        )
    )
    
    process = Process(
        process_id="batch_001",
        process_type="batch",
        time=time_axis,
        dynamic_states={"biomass": biomass}
    )
    
    # Check it can be flattened and unflattened
    leaves, treedef = tree_util.tree_flatten(process)
    reconstructed = tree_util.tree_unflatten(treedef, leaves)
    
    assert reconstructed.process_id == process.process_id
    assert reconstructed.process_type == process.process_type
    assert "biomass" in reconstructed.dynamic_states


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
