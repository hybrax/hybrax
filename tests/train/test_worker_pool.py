from __future__ import annotations

from concurrent.futures.process import BrokenProcessPool

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest
from test_harness import _biomass_loss, _make_multi_process_collection

from hybrax.train.harness import TrainHarnessConfig, train_collection
from hybrax.train.training_data import TrainingDataStore
from hybrax.train.utils import load_custom_module
from hybrax.train.worker_pool import GradientWorkerPool, _flatten, _restore

_CUSTOM_PY = '''
import equinox as eqx
import jax
import jax.numpy as jnp

from hybrax.train.model_api import (
    RateModule,
    ReactionOutputs,
    frozen_field,
    trainable_field,
)


class CustomModule(RateModule):
    """Holds a lambda and a JAX function (inside the MLP) plus a frozen array."""

    model: eqx.nn.MLP = trainable_field()
    offset: jax.Array = frozen_field()

    def __init__(self, **scales):
        super().__init__(**scales)
        self.model = eqx.nn.MLP(1, 1, 4, 1, jax.nn.tanh, key=jax.random.key(0))
        self.offset = jnp.asarray([0.05])

    def __call__(self, t, inputs):
        rate = self.model(inputs.SCL_modeled_RMCs)[0] + self.offset[0]
        return ReactionOutputs(
            SCL_modeled_ReactionOde_rates=jnp.asarray([rate]),
            SCL_modeled_Inflows_rates=jnp.zeros(0),
            SCL_modeled_Outflows_rates=jnp.zeros(0),
        )
'''

_SCALES = {
    "SCALE_modeled_RMCs": jnp.ones(1),
    "SCALE_V_in_cumulative": jnp.asarray(1.0),
    "SCALE_modeled_Inflows_cumulative": jnp.ones(0),
    "SCALE_modeled_Outflows_cumulative": jnp.ones(0),
    "SCALE_controlled_Inflows_cumulative": jnp.ones(0),
    "SCALE_controlled_Inflows_rates": jnp.ones(0),
    "SCALE_controlled_Outflows_cumulative": jnp.ones(0),
    "SCALE_controlled_Outflows_rates": jnp.ones(0),
    "SCALE_controlled_Inflows_Cin": jnp.ones((0, 1)),
    "SCALE_controlled_PVs": jnp.ones(0),
    "SCALE_modeled_Inflows_Cin": jnp.ones((0, 1)),
    "SCALE_modeled_ReactionOde_rates": jnp.ones(1),
    "SCALE_modeled_Inflows_rates": jnp.ones(0),
    "SCALE_modeled_Outflows_rates": jnp.ones(0),
}


def test_flat_transport_restores_dtypes_and_shapes():
    tree = {
        "a": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
        "b": (jnp.asarray(1.5, dtype=jnp.float64), jnp.asarray([7.0, 8.0])),
    }

    restored = _restore(_flatten(tree), tree)

    for expected, actual in zip(
        jtu.tree_leaves(tree), jtu.tree_leaves(restored), strict=True
    ):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="length differs"):
        _restore(np.zeros(8), tree)


def test_workers_match_in_process_training(tmp_path):
    """Two workers reproduce in-process training for a custom.py model with an
    implicit solver, partial shuffled batches, and a frozen field."""
    custom_py = tmp_path / "custom.py"
    custom_py.write_text(_CUSTOM_PY)
    module = load_custom_module(custom_py).CustomModule(**_SCALES)
    store = TrainingDataStore.from_collection(
        _make_multi_process_collection(4),
        target_variable_order=["biomass"],
        target_source="reactor_components",
    )

    def train(workers):
        return train_collection(
            store,
            reaction_module=module,
            loss_module=_biomass_loss(),
            config=TrainHarnessConfig(
                epochs=3,
                batch_size=3,
                workers=workers,
                learning_rate=1e-2,
                ode_solver=diffrax.Kvaerno5(),
            ),
        )

    in_process, pooled = train(1), train(2)

    assert pooled.batch_process_names_by_step == in_process.batch_process_names_by_step
    np.testing.assert_allclose(
        pooled.mean_loss_by_step, in_process.mean_loss_by_step, rtol=1e-9
    )
    np.testing.assert_allclose(
        pooled.per_process_loss_by_step, in_process.per_process_loss_by_step, rtol=1e-9
    )
    for expected, actual in zip(
        jtu.tree_leaves(eqx.filter(in_process.trained_wrapper, eqx.is_array)),
        jtu.tree_leaves(eqx.filter(pooled.trained_wrapper, eqx.is_array)),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)
    trained = pooled.trained_wrapper.reaction_module
    np.testing.assert_array_equal(trained.offset, module.offset)
    assert not np.array_equal(
        trained.model.layers[0].weight, module.model.layers[0].weight
    )


def _build_scaled_identity(scale):
    def gradient(params, batch_indices, step):
        if int(batch_indices[0]) == 13:
            raise RuntimeError("unlucky process")
        index = batch_indices[0].astype(params.dtype)
        return (index, step), scale * index * params

    return gradient


def _build_failing(scale):
    raise RuntimeError("cannot build")


def test_pool_averages_in_batch_order_and_propagates_worker_errors():
    params = jnp.asarray([1.0, 2.0])
    with GradientWorkerPool(2, _build_scaled_identity, (10.0,), params) as pool:
        outputs, grads = pool.mean_gradient(params, jnp.asarray([3, 1, 2]), 7)

        assert [(int(i), int(step)) for i, step in outputs] == [(3, 7), (1, 7), (2, 7)]
        np.testing.assert_allclose(grads, 10.0 * 2.0 * params)
        with pytest.raises(RuntimeError, match="unlucky process"):
            pool.mean_gradient(params, jnp.asarray([1, 13, 2]), 8)
        # The pool survives a failed job.
        outputs, _ = pool.mean_gradient(params, jnp.asarray([5]), 9)
        assert int(outputs[0][0]) == 5


def test_pool_start_up_failure_raises_instead_of_hanging():
    params = jnp.asarray([1.0])
    with (
        GradientWorkerPool(2, _build_failing, (1.0,), params) as pool,
        pytest.raises(BrokenProcessPool),
    ):
        pool.mean_gradient(params, jnp.asarray([0, 1]), 0)
