import gzip
import io

import pytest

import bp_format.json_io as json_io
from bp_format.json_io import (
    JSONParseError,
    has_top_level_key,
    load_json,
    loads_json,
)


def test_loads_json_materializes_full_root_and_accepts_comments():
    text = (
        "// heading\n"
        "{/* block */\n"
        '  "url": "https://example.com/data", // inline\n'
        '  "values": [1, 2]\n'
        "}\n"
        "// footer"
    )

    assert loads_json(text) == {
        "url": "https://example.com/data",
        "values": [1, 2],
    }


@pytest.mark.parametrize("compressed", [False, True])
def test_load_json_streams_plain_and_gzip(tmp_path, compressed):
    path = tmp_path / ("input.json.gz" if compressed else "input.json")
    payload = b'{"value": 1}'
    if compressed:
        with gzip.open(path, "wb") as stream:
            stream.write(payload)
    else:
        path.write_bytes(payload)

    assert load_json(path) == {"value": 1}


@pytest.mark.parametrize(
    "text",
    [
        '{"value": 1} trailing',
        '{"value": 1',
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"value": -Infinity}',
        '{"value": 1} /* unterminated block comment',
        '{"value": 1} 2',
        '{"value": 1} 2e',
        '{"value": 1} 2.',
        "1 2",
        "2 /* comment */ 3",
        '{"value": 1, "value": 2}',
    ],
)
def test_loads_json_rejects_malformed_and_nonfinite_json(text):
    with pytest.raises(JSONParseError):
        loads_json(text)


@pytest.mark.parametrize("compressed", [False, True])
def test_has_top_level_key_handles_order_comments_and_gzip(tmp_path, compressed):
    path = tmp_path / ("input.json.gz" if compressed else "input.json")
    payload = (
        b'{/* header */ "metadata": {"nested_only": true}, '
        b'"processes": {}, // top-level key comes last\n "case_id": "top"}'
    )
    if compressed:
        with gzip.open(path, "wb") as stream:
            stream.write(payload)
    else:
        path.write_bytes(payload)

    assert has_top_level_key(path, "case_id")
    assert not has_top_level_key(path, "nested_only")
    assert not has_top_level_key(path, "missing")


@pytest.mark.parametrize(
    "text",
    [
        '{"processes": {',
        '{"case_id": "matched-before-error"} trailing',
    ],
)
def test_has_top_level_key_rejects_malformed_json_with_path(tmp_path, text):
    path = tmp_path / "broken.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(JSONParseError, match=str(path)):
        has_top_level_key(path, "case_id")


@pytest.mark.parametrize("compressed", [False, True])
def test_path_readers_reject_trailing_numeric_value(tmp_path, compressed):
    directory = tmp_path / "comments"
    directory.mkdir()
    path = directory / ("input.json.gz" if compressed else "input.json")
    payload = b'{"metadata": null, "processes": {}} 2'
    if compressed:
        with gzip.open(path, "wb") as stream:
            stream.write(payload)
    else:
        path.write_bytes(payload)

    with pytest.raises(JSONParseError, match=str(path)):
        load_json(path)
    with pytest.raises(JSONParseError, match=str(path)):
        has_top_level_key(path, "processes")


def test_unsupported_comment_keyword_fails_clearly(monkeypatch):
    def fail(*_args, **_kwargs):
        raise TypeError("'allow_comments' is an invalid keyword argument")

    monkeypatch.setattr(json_io.ijson, "parse", fail)

    with pytest.raises(RuntimeError, match="does not support allow_comments"):
        json_io._load_stream(io.BytesIO(b"{}"))


def test_unsupported_comment_backend_fails_clearly_while_parsing(monkeypatch):
    def fail(*_args, **_kwargs):
        raise ValueError("Comments are not supported by the python backend")
        yield

    monkeypatch.setattr(json_io.ijson, "parse", fail)

    with pytest.raises(RuntimeError, match="does not support allow_comments"):
        json_io._load_stream(io.BytesIO(b"{}"))


def test_load_json_error_includes_source_path_and_parser_context(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"value": Infinity}', encoding="utf-8")

    with pytest.raises(JSONParseError) as exc_info:
        load_json(path)

    message = str(exc_info.value)
    assert str(path) in message
    assert "lexical error" in message
