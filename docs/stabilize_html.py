"""Normalize MyST-NB's clean versus incremental CSS cache suffix."""

import re
import sys
import zlib
from pathlib import Path

CSS_LINK = re.compile(r"(mystnb\.[0-9a-f]{64}\.css)(?:\?v=[0-9a-f]{8})?")


def main(output_dir: str) -> None:
    output = Path(output_dir)
    checksums = {
        path.name: f"{zlib.crc32(path.read_bytes().translate(None, b'\r')):08x}"
        for path in (output / "_static").glob("mystnb.*.css")
    }

    def versioned(match: re.Match) -> str:
        filename = match.group(1)
        return f"{filename}?v={checksums[filename]}"

    for path in output.rglob("*.html"):
        original = path.read_text()
        stable = CSS_LINK.sub(versioned, original)
        if stable != original:
            path.write_text(stable)


if __name__ == "__main__":
    main(sys.argv[1])
