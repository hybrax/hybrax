# hybrax docs

The Sphinx project for hybrax's documentation site, under `source/`.

## Building

One-time setup, from the `hybrax` repo root:

```bash
uv sync --extra docs
```

or, via conda:

```bash
conda env create -f docs/environment.yml
```

Then, from `docs/`:

- `bash docs_rebuild.sh` does a full rebuild into `html/` (regenerated from
  scratch and committed for publishing). Open `html/index.html` to preview.
- `bash docs_rebuild_fast.sh` does an incremental rebuild for local iteration.

## Notes

- The API reference is generated with **sphinx-autoapi**, which parses
  `hybrax`'s source statically and never imports it. `hybrax.train`'s
  import-time device bootstrap and the heavy JAX stack never run during a
  build.
- `source/autoapi/`, `source/_data/out/`, and `_scratch/` are generated and
  gitignored. `html/` is generated too, but committed for publishing.
