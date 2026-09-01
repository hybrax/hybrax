"""Make MyST-NB's CSS suffix match Sphinx's asset checksum.

MyST-NB writes its CSS after clean pages render, so clean builds omit the
suffix that incremental builds include. This mirrors
`sphinx.builders.html._assets._file_checksum` to canonicalize both.
"""

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
        original = path.read_text(encoding="utf-8")
        stable = CSS_LINK.sub(versioned, original)
        if stable != original:
            path.write_text(stable, encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
