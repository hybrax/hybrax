# bp-docs

Private documentation for **hybrax**. This repo holds the Sphinx project under
`source/` and the rendered site under `html/` (committed), so the docs can be shared
without publishing them publicly.

## Reading the docs (colleagues)

```bash
git clone https://github.com/Gotsmy/bp-docs
```

Then open `html/index.html` in a browser. It works from `file://`, no server needed,
and search works too. To get the latest version later, run `git pull`.

## Rebuilding / publishing (maintainer)

The build documents the **sibling** package, so it expects this layout:

```
bpbench/
├── hybrax/       # github.com/hybrax/hybrax
└── bp-docs/      # this repo
```

One-time setup: install `hybrax` and the doc tooling into the project env.

```bash
PY=/home/mgotsmy/anaconda3/envs/bench13/bin/python
"$PY" -m pip install -e ../hybrax
"$PY" -m pip install -r requirements.txt
```

Then:

- `bash docs_rebuild.sh` rebuilds `html/` locally, with no git changes. Open
  `html/index.html` to preview.
- `bash docs_publish.sh` rebuilds, then commits `html/` and pushes so colleagues can
  pull.

## Notes

- The API reference is generated with **sphinx-autoapi**, which parses `hybrax`'s
  source statically and never imports it. `hybrax.train`'s import-time device
  bootstrap and the heavy JAX stack never run during a build.
- `source/autoapi/`, `source/narrative/`, and `_scratch/` are generated each build and
  are gitignored. `html/` is intentionally tracked: it's the shared artifact.
- `hybrax` is a **read-only input**; nothing here writes into it.
- To make the docs public later, enable GitHub Pages on this repo, serving `html/`.
  The committed site becomes live with no other change.
