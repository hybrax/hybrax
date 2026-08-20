"""Shared logging setup for hybrax.format."""

import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``hybrax.format`` namespace, configured on first use."""
    logging.basicConfig(format=_FORMAT, datefmt=_DATEFMT)  # no-op if the host app already configured a root handler
    logging.getLogger("hybrax.format").setLevel(logging.INFO)  # only hybrax.format's own messages become visible by default
    return logging.getLogger(name)
