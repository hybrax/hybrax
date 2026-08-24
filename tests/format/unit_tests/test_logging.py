"""Tests for hybrax.format._logging."""

import logging

from hybrax.format._logging import get_logger


def test_get_logger_never_configures_root():
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    try:
        root.handlers, root.level = [], logging.WARNING
        get_logger("hybrax.format.probe")
        assert root.handlers == []
        assert root.level == logging.WARNING
    finally:
        root.handlers, root.level = saved_handlers, saved_level


def test_get_logger_is_silent_by_default(capsys):
    root = logging.getLogger()
    fmt = logging.getLogger("hybrax.format")
    saved_root, saved_level, saved_fmt = (
        list(root.handlers),
        root.level,
        list(fmt.handlers),
    )
    try:
        root.handlers, root.level, fmt.handlers = [], logging.WARNING, []
        get_logger("hybrax.format.probe").warning("should not appear")
        out = capsys.readouterr()
        assert out.out == ""
        assert out.err == ""
    finally:
        root.handlers, root.level, fmt.handlers = saved_root, saved_level, saved_fmt
