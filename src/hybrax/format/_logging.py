"""Shared logging setup for hybrax.format."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``hybrax.format`` namespace.

    Library code must never call ``logging.basicConfig()`` or set a shared
    logger's level — that decision belongs solely to the hosting application.
    A ``NullHandler`` is attached once so that, absent app-level logging
    configuration, records are silently discarded instead of falling through
    to ``logging.lastResort``.
    """
    top = logging.getLogger("hybrax.format")
    if not top.handlers:
        top.addHandler(logging.NullHandler())
    return logging.getLogger(name)
