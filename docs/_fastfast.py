"""Build a local HTML preview with saved outputs, never executing notebooks."""

import sys
from pathlib import Path

from jupyter_cache import get_cache
from myst_nb import sphinx_ as myst_sphinx
from myst_nb.core.execute.base import NotebookClientBase
from sphinx.cmd.build import main
from sphinx.errors import ExtensionError


class CachedPreviewClient(NotebookClientBase):
    """Attach the latest outputs by page path, ignoring code-source changes."""

    def start_client(self):
        cells = [cell for cell in self.notebook.cells if cell.cell_type == "code"]
        if not cells:
            return
        page = self.path.resolve().relative_to(Path(__file__).resolve().parent)
        cache = get_cache(self.nb_config.execution_cache_path)
        # Cache URIs retain the original absolute mount path. Match only the
        # page's path relative to this build script, e.g. source/gallery/optfed.md.
        records = [
            record
            for record in cache.list_cache_records()
            if Path(record.uri).parts[-len(page.parts) :] == page.parts
        ]
        if not records:
            raise ExtensionError(
                f"No cached outputs for {self.path}. "
                "Run docs_rebuild_fast.sh first. Preview never executes code."
            )
        record = max(records, key=lambda item: item.created)
        saved = cache.get_cache_bundle(record.pk).nb
        if len(cells) != len(saved.cells):
            raise ExtensionError(
                f"Code-cell count changed in {self.path}; cached outputs cannot "
                "be aligned. Run docs_rebuild_fast.sh first."
            )
        for cell, old in zip(cells, saved.cells, strict=True):
            cell.outputs = old.outputs
            cell.execution_count = old.execution_count
        for key in ("kernelspec", "language_info", "widgets"):
            if key in saved.metadata:
                self.notebook.metadata[key] = saved.metadata[key]
        self.logger.info(f"Preview using saved outputs: ID={record.pk}")
        self.exec_metadata = {
            "mtime": record.created.timestamp(),
            "runtime": record.data.get("execution_seconds"),
            "method": "cached-preview",
            "succeeded": True,
            "error": None,
            "traceback": None,
        }


def preview_client(notebook, source, nb_config, logger, read_fmt=None):
    """Replace MyST-NB's execution dispatch, including per-page overrides."""
    return CachedPreviewClient(notebook, Path(source), nb_config, logger)


if __name__ == "__main__":
    myst_sphinx.create_client = preview_client
    sys.exit(main(sys.argv[1:]))
