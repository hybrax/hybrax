import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(format=_FORMAT, datefmt=_DATEFMT)  # no-op if the host app already configured a root handler
    logging.getLogger("bp_format").setLevel(logging.INFO)  # only bp-format's own messages become visible by default
    return logging.getLogger(name)
