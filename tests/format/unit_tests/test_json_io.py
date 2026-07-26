import json

import pytest

from bp_format.json_io import load_json, loads_json


def test_load_json_accepts_only_whole_line_comments(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(
        "// heading\n"
        "{\n"
        "  // field note\n"
        '  "url": "https://example.com/data",\n'
        '  "note": "before\u2028//must-stay\u2028after"\n'
        "}\n"
        "// footer",
        encoding="utf-8",
    )

    assert load_json(path) == {
        "url": "https://example.com/data",
        "note": "before\u2028//must-stay\u2028after",
    }


@pytest.mark.parametrize(
    "text",
    [
        '{"value": 1 // inline\n}',
        '{/* block */ "value": 1}',
        '{"value": 1,}',
    ],
)
def test_loads_json_rejects_non_whole_line_comment_syntax(text):
    with pytest.raises(json.JSONDecodeError):
        loads_json(text)
