# bp-docs

Private, combined documentation for **bp-train** and **bp-format**. This repo holds the
Sphinx project under `source/` **and** the rendered site under `html/` (committed), so
the docs can be shared without publishing them publicly.

## Reading the docs (colleagues)

```bash
git clone https://github.com/Gotsmy/bp-docs
```

Then open `html/index.html` in a browser (it works from `file://` — no server needed;
search works too). To get the latest version later: `git pull`.

## Rebuilding / publishing (maintainer)

The build documents the **sibling** packages, so it expects this layout:

```
bpbench/
├── bp-train/     # github.com/julibeg/bp-train
├── bp-format/    # github.com/Gotsmy/bp-format
└── bp-docs/      # this repo
```

One-time setup — install the doc tooling into the project env:

```bash
/home/mgotsmy/anaconda3/envs/bench13/bin/python -m pip install -r requirements.txt
```

Then:

- `bash docs_rebuild.sh` — rebuild `html/` locally (no git changes). Open
  `html/index.html` to preview.
- `bash docs_publish.sh` — rebuild, then commit `html/` and push so colleagues can pull.

## Notes

- The API reference is generated with **sphinx-autoapi**, which parses the packages'
  source statically (never imports them) — so `bp_train`'s import-time device bootstrap
  and the heavy JAX stack never run during a build.
- `source/autoapi/`, `source/narrative/`, and `_scratch/` are generated each build and
  are gitignored. `html/` is intentionally tracked — it's the shared artifact.
- bp-train and bp-format are **read-only inputs**; nothing here writes into them.
- To make the docs public later, enable GitHub Pages on this repo (serving `html/`);
  the committed site becomes live with no other change.
