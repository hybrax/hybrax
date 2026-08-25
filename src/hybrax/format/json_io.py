"""JSON input helpers."""

import gzip
import io
from pathlib import Path
from typing import Any, BinaryIO, Iterator, TextIO

import ijson

_PARSER_OPTIONS = {
    "use_float": True,
    "allow_comments": True,
    "multiple_values": True,
}
_EOF_SENTINEL_VALUE = "__hybrax_format_eof_sentinel__"
_EOF_SENTINEL = f'\n"{_EOF_SENTINEL_VALUE}"'
_ROOT_END_EVENTS = frozenset(
    {"null", "boolean", "number", "string", "end_map", "end_array"}
)


class JSONParseError(ValueError):
    """Invalid hybrax.format JSON with source context."""


class _EOFCheckingStream:
    """Make YAJL prove that it left comment mode before reaching EOF."""

    def __init__(self, stream: BinaryIO | TextIO):
        self._stream = stream
        self.sentinel_sent = False

    def read(self, size: int = -1) -> bytes | str:
        """Read from the wrapped stream, appending the EOF sentinel once it runs dry."""
        chunk = self._stream.read(size)
        if size == 0 or chunk or self.sentinel_sent:
            return chunk
        self.sentinel_sent = True
        if isinstance(chunk, bytes):
            return _EOF_SENTINEL.encode()
        return _EOF_SENTINEL


def _unsupported_comments() -> RuntimeError:
    """Build the error raised when the active ijson backend can't parse comments."""
    return RuntimeError("The active ijson backend does not support allow_comments=True")


def _validated_events(
    iterator: Iterator[tuple[str, str, Any]], source: str | Path
) -> Iterator[tuple[str, str, Any]]:
    """Pass through parser events while enforcing single-value,
    no-duplicate-key input.
    """
    root_count = 0
    top_level_keys = set()
    pending_sentinel = False
    sentinel_event = ("", "string", _EOF_SENTINEL_VALUE)
    try:
        for item in iterator:
            if pending_sentinel:
                root_count += 1
                if root_count > 1:
                    raise JSONParseError(f"{source}: trailing JSON value")
                yield sentinel_event
                pending_sentinel = False
            if item == sentinel_event:
                pending_sentinel = True
                continue
            yield item
            prefix, event, value = item
            if prefix == "" and event == "map_key":
                if value in top_level_keys:
                    raise JSONParseError(f"{source}: duplicate top-level key {value!r}")
                top_level_keys.add(value)
            if prefix == "" and event in _ROOT_END_EVENTS:
                root_count += 1
                if root_count > 1:
                    raise JSONParseError(f"{source}: trailing JSON value")
    except JSONParseError:
        raise
    except ijson.JSONError as exc:
        raise JSONParseError(f"{source}: {exc}") from exc
    except (TypeError, ValueError) as exc:
        if "comment" not in str(exc).lower():
            raise
        raise _unsupported_comments() from exc

    if not pending_sentinel:
        raise JSONParseError(f"{source}: unterminated block comment")
    if root_count == 0:
        raise JSONParseError(f"{source}: expected a JSON value")


def _parse(
    stream: BinaryIO | TextIO, *, source: str | Path = "<stream>"
) -> Iterator[tuple[str, str, Any]]:
    """Yield validated parser events using the shared parser policy."""
    checked_stream = _EOFCheckingStream(stream)
    try:
        iterator = ijson.parse(checked_stream, **_PARSER_OPTIONS)
    except TypeError as exc:
        if "comment" not in str(exc).lower():
            raise
        raise _unsupported_comments() from exc
    return _validated_events(iterator, source)


def _items(
    stream: BinaryIO | TextIO, prefix: str, *, source: str | Path = "<stream>"
) -> Iterator[Any]:
    """Yield values at an ijson prefix using the shared parser policy."""
    return ijson.items(_parse(stream, source=source), prefix)


def _kvitems(
    stream: BinaryIO | TextIO, prefix: str, *, source: str | Path = "<stream>"
) -> Iterator[tuple[str, Any]]:
    """Yield mapping entries at an ijson prefix using the shared parser policy."""
    return ijson.kvitems(_parse(stream, source=source), prefix)


def _load_stream(stream: BinaryIO | TextIO, *, source: str | Path = "<stream>") -> Any:
    """Decode the single top-level JSON value from ``stream``."""
    values = list(_items(stream, "", source=source))
    if not values:
        raise JSONParseError(f"{source}: expected a JSON value")
    return values[0]


def loads_json(text: str) -> Any:
    """Decode one JSON value, accepting YAJL-style comments."""
    return _load_stream(io.BytesIO(text.encode()), source="<string>")


def load_json(path: str | Path) -> Any:
    """Incrementally read and decode one UTF-8 JSON or JSON.GZ input."""
    path = Path(path)
    opener = gzip.open if path.suffixes[-2:] == [".json", ".gz"] else open
    with opener(path, "rb") as stream:
        return _load_stream(stream, source=path)


def has_top_level_key(path: str | Path, key: str) -> bool:
    """Return whether a JSON object's top level contains ``key``."""
    path = Path(path)
    opener = gzip.open if path.suffixes[-2:] == [".json", ".gz"] else open
    found = False
    with opener(path, "rb") as stream:
        for prefix, event, value in _parse(stream, source=path):
            if prefix == "" and event == "map_key" and value == key:
                found = True
    return found
