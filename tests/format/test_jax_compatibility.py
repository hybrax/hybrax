"""
Test JAX compatibility with BPbench dataclasses.

Regular Python dataclasses are NOT registered as JAX PyTrees by default.
These tests verify that JAX arrays embedded in our dataclasses work correctly
with JAX operations (jit, grad, vmap, etc.) when extracted from the structures.
"""

import pytest
import jax
import jax.numpy as jnp

from bpbench import (
    TimeAxis,
    TimeSeries,
    StaticVariable,
    BioProcessMetadata,
    ReactorMediumComponent,
    ReactorMedium,
    Volume,
    BioProcess,
    ProcessVariable,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(timepoints, values):
    return TimeSeries(timepoints=jnp.array(timepoints), values=jnp.array(values))


def _make_process():
    ts = _ts([0., 1., 2.], [0.1, 0.5, 1.0])
    rc = ReactorMediumComponent(name="biomass", unit="g/L", concentration=ts, is_intracellular=False)
    return BioProcess(
        metadata=BioProcessMetadata(name="test", process_type="batch"),
        time_axis=TimeAxis(unit="hours", start=0.0, end=2.0, time_reference="inoculation"),
        volume=Volume(initial_volume=1.0, unit="L"),
        reactor_medium=ReactorMedium(name="medium", density=1.0, density_unit="kg/L",
                                     components={"biomass": rc}),
    )


# ---------------------------------------------------------------------------
# JAX array storage in dataclasses
# ---------------------------------------------------------------------------

def test_timeseries_stores_jax_arrays():
    ts = _ts([0., 1., 2.], [0.1, 0.5, 1.0])
    assert isinstance(ts.timepoints, jnp.ndarray)
    assert isinstance(ts.values, jnp.ndarray)
    assert ts.timepoints.shape == (3,)
    assert ts.values.shape == (3,)


def test_reactor_component_stores_timeseries():
    ts = _ts([0., 1., 2.], [0.1, 0.5, 1.0])
    rc = ReactorMediumComponent(name="biomass", unit="g/L", concentration=ts, is_intracellular=False)
    assert isinstance(rc.concentration.values, jnp.ndarray)


def test_jax_operations_on_timeseries_values():
    ts = _ts([0., 1., 2.], [0.1, 0.5, 1.0])
    mean_val = jnp.mean(ts.values)
    assert float(mean_val) == pytest.approx((0.1 + 0.5 + 1.0) / 3, rel=1e-5)


def test_jax_operations_on_timepoints():
    ts = _ts([0., 1., 2.], [0.1, 0.5, 1.0])
    diffs = jnp.diff(ts.timepoints)
    assert jnp.all(diffs > 0)


# ---------------------------------------------------------------------------
# JAX transformations on arrays extracted from dataclasses
# ---------------------------------------------------------------------------

def test_jit_on_timeseries_values():
    ts = _ts([0., 1., 2.], [0.1, 0.5, 1.0])

    @jax.jit
    def compute_mean(values):
        return jnp.mean(values)

    result = compute_mean(ts.values)
    assert float(result) == pytest.approx((0.1 + 0.5 + 1.0) / 3, rel=1e-5)


def test_jit_on_multiple_arrays():
    ts = _ts([0., 1., 2., 3.], [0.0, 0.5, 1.0, 1.5])

    @jax.jit
    def trapezoid_integral(timepoints, values):
        return jnp.trapezoid(values, timepoints)

    result = trapezoid_integral(ts.timepoints, ts.values)
    assert float(result) == pytest.approx(2.25, rel=1e-4)


def test_grad_on_timeseries_values():
    ts = _ts([0., 1., 2.], [0.1, 0.5, 1.0])

    def sum_fn(values):
        return jnp.sum(values ** 2)

    grad_fn = jax.grad(sum_fn)
    grad = grad_fn(ts.values)
    expected = 2 * ts.values
    assert jnp.allclose(grad, expected)


def test_vmap_on_timeseries():
    # Create a batch of values and use vmap
    batch_values = jnp.stack([
        jnp.array([0.1, 0.5, 1.0]),
        jnp.array([0.2, 0.6, 1.2]),
    ])

    @jax.vmap
    def compute_max(values):
        return jnp.max(values)

    results = compute_max(batch_values)
    assert results.shape == (2,)
    assert float(results[0]) == pytest.approx(1.0, rel=1e-5)
    assert float(results[1]) == pytest.approx(1.2, rel=1e-5)


# ---------------------------------------------------------------------------
# JAX with BioProcess data extraction
# ---------------------------------------------------------------------------

def test_extract_and_operate_on_bioprocess_data():
    process = _make_process()

    # Extract the biomass TimeSeries from the process
    biomass_ts = process.reactor_medium.components["biomass"].concentration

    # Apply JAX operations
    mean_biomass = jnp.mean(biomass_ts.values)
    assert float(mean_biomass) == pytest.approx((0.1 + 0.5 + 1.0) / 3, rel=1e-5)


def test_jit_with_extracted_arrays():
    process = _make_process()
    biomass_ts = process.reactor_medium.components["biomass"].concentration

    @jax.jit
    def growth_rate_approx(values, timepoints):
        return jnp.diff(values) / jnp.diff(timepoints)

    rates = growth_rate_approx(biomass_ts.values, biomass_ts.timepoints)
    assert rates.shape == (2,)
    assert jnp.all(rates > 0)


def test_static_variable_is_plain_float():
    sv = StaticVariable(value=3.14)
    # StaticVariable stores a plain Python float, not a JAX array
    assert sv.value == pytest.approx(3.14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
