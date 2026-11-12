"""CPU worker processes that compute per-process gradients for one training run.

``train.workers > 1`` spreads the ODE solves of one optimizer update over
spawned computer processes. The parent keeps the optimizer, checkpoints, and
holdout evaluation. Per update it sends every worker the same parameters,
collects one loss and gradient per batch process, averages them in batch
order, and applies a single optimizer update.

Each worker receives the parent's prepared training context once, as a pickle,
and compiles the gradient of a one-process batch once. ``cloudpickle`` sends
lambdas and interactively defined classes by value; classes from ``custom.py``
are re-imported in the worker.
"""

from __future__ import annotations

import ctypes
import gc
import io
import multiprocessing
import pickle
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Self

import cloudpickle
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from diffrax._root_finder._with_tols import use_stepsize_tol

from .utils import CUSTOM_MODULE_NAME, load_custom_module

# Set by ``_initialize`` in each worker: the compiled gradient, the parameter
# template for the flat transport, and whether the first job has run.
_WORKER: dict[str, Any] = {}


class _Pickler(cloudpickle.Pickler):
    """Diffrax tests its default-tolerance marker by identity, so send a
    reference to that singleton instead of a copy."""

    def persistent_id(self, obj):
        return "use_stepsize_tol" if obj is use_stepsize_tol else None


class _Unpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        assert pid == "use_stepsize_tol"
        return use_stepsize_tol


def _dumps(obj) -> bytes:
    buffer = io.BytesIO()
    _Pickler(buffer).dump(obj)
    return buffer.getvalue()


def _flatten(tree) -> np.ndarray:
    """Concatenate all array leaves into one host vector."""
    leaves = [np.asarray(leaf).ravel() for leaf in jtu.tree_leaves(tree)]
    # A fully frozen model has no trainable leaves.
    return np.concatenate(leaves) if leaves else np.zeros(0)


def _restore(flat: np.ndarray, template):
    """Inverse of :func:`_flatten`, with the dtypes and shapes of ``template``."""
    leaves, treedef = jtu.tree_flatten(template)
    if flat.size != sum(leaf.size for leaf in leaves):
        raise ValueError("flat vector length differs from the parameter template")
    restored, offset = [], 0
    for leaf in leaves:
        chunk = flat[offset : offset + leaf.size]
        restored.append(jnp.asarray(chunk, dtype=leaf.dtype).reshape(leaf.shape))
        offset += leaf.size
    return jtu.tree_unflatten(treedef, restored)


def _trim_memory() -> None:
    """Return free allocator pages to the OS (glibc only; a no-op elsewhere)."""
    gc.collect()
    if sys.platform == "linux":
        trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if trim is not None:
            trim(0)


def _initialize(custom_py: str | None, payload: bytes) -> None:
    """Worker start-up: rebuild the parent's context and the gradient function."""
    if custom_py is not None:
        # The payload refers to custom classes by module name.
        load_custom_module(custom_py)
    build_gradient, context, params = _Unpickler(io.BytesIO(payload)).load()
    _WORKER.update(
        gradient=build_gradient(*context), params=params, first_job_done=False
    )
    jax.clear_caches()
    _trim_memory()


def _gradient_job(index: int, flat_params: np.ndarray, step: int):
    """Loss outputs and flat gradient for the one-process batch ``[index]``."""
    outputs, grads = _WORKER["gradient"](
        _restore(flat_params, _WORKER["params"]),
        jnp.asarray([index], dtype=jnp.int32),
        jnp.asarray(step, dtype=jnp.int32),
    )
    result = jax.device_get(outputs), _flatten(grads)
    if not _WORKER["first_job_done"]:
        # Compilation leaves a lot of free scratch memory behind. Release it
        # once; the compiled function itself stays cached.
        _WORKER["first_job_done"] = True
        _trim_memory()
    return result


class GradientWorkerPool:
    """Persistent worker processes that evaluate one-process gradients.

    Use as a context manager around the training loop::

        with GradientWorkerPool(4, build_gradient, context, params) as pool:
            outputs, grads = pool.mean_gradient(params, batch_indices, step)

    ``build_gradient(*context)`` runs once in every worker and must return a
    function ``(params, batch_indices, step) -> (outputs, grads)``, where
    ``grads`` has the structure of ``params``. It has to be a module-level
    function, and ``context`` and ``params`` have to be picklable.
    """

    def __init__(
        self, workers: int, build_gradient: Callable, context: tuple, params
    ) -> None:
        if jax.default_backend() != "cpu":
            raise ValueError("train.workers > 1 is only supported on CPU")
        custom_module = sys.modules.get(CUSTOM_MODULE_NAME)
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
            initargs=(
                None if custom_module is None else custom_module.__file__,
                _dumps((build_gradient, context, params)),
            ),
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, *_exc) -> None:
        # After an error, do not hold it back until the running solves finish.
        self._executor.shutdown(wait=exc_type is None, cancel_futures=True)

    def mean_gradient(self, params, batch_indices, step) -> tuple[list, Any]:
        """Evaluate every batch process at the same ``params`` and ``step``.

        Returns the per-process ``outputs`` in batch order and the mean of the
        per-process gradients. A failing or dying worker raises here.
        """
        flat_params = _flatten(params)
        futures = [
            self._executor.submit(_gradient_job, int(index), flat_params, int(step))
            for index in np.asarray(batch_indices)
        ]
        # Collect in batch order, whatever order the workers finish in.
        results = [future.result() for future in futures]
        mean_grads = np.mean([grads for _, grads in results], axis=0)
        return [outputs for outputs, _ in results], _restore(mean_grads, params)
