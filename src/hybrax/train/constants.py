"""Shared constants with no natural home in a more specific module."""

import diffrax

METADATA_NAMESPACE = "hybrax.train"
DEFAULT_SOLVER = diffrax.Tsit5()
DEFAULT_FLOW_INTEGRAL_RELATIVE_TOLERANCE = 1e-3

# Initial proposed step as a fraction of the integration span.
INITIAL_STEP_FRACTION = 1e-3
