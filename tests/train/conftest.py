import gc

import jax
import pytest


@pytest.fixture(scope="module", autouse=True)
def clear_jax_caches():
    yield
    jax.clear_caches()
    gc.collect()
