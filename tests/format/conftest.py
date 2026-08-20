"""Enable JAX x64 mode for the test suite.

The TimeSeries dtype migration sets `jnp.float64` as the default; without x64
mode JAX silently downgrades to float32, defeating the migration's intent.
Enabling x64 globally for tests keeps the tested numerics aligned with what
users get when they enable x64 in their own environment.
"""

import jax

jax.config.update("jax_enable_x64", True)
