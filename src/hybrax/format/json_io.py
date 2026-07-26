"""JSON input helpers."""

import json
from pathlib import Path
from typing import Any


def loads_json(text: str) -> Any:
    """Decode JSON after removing whole-line ``//`` comments."""
    uncommented = "\n".join(
        "" if line.lstrip().startswith("//") else line for line in text.split("\n")
    )
    return json.loads(uncommented)


def load_json(path: str | Path) -> Any:
    """Read and decode a UTF-8 JSON input."""
    return loads_json(Path(path).read_text(encoding="utf-8"))
