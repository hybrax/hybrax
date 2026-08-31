# Installation

> Clone the repo, install it editable into a Python ≥ 3.12 environment, and check that
> `hybrax --help` runs.

## Requirements

- **Python ≥ 3.12**.
- A working JAX install. `hybrax` pulls in `jax` and `jaxlib`; CPU is fine and is what
  these docs are built against.

## Layout

```
hybrax/
├── src/hybrax/
│   ├── format/     the data model, mechanistic RHS
│   └── train/      hybrid ODE training
├── tests/
├── specs/          numbered technical design docs
├── examples/       runnable tutorial and gallery walkthroughs
├── docs/           this Sphinx site
└── pyproject.toml
```

## Install

```bash
python -m pip install -e ./hybrax
```

Editable installs (`-e`) are the norm here: the package is under active development and
deliberately ships breaking changes, so you want your checkout and your imports to be
the same thing.

### Optional extras

| Extra | Command | Gives you |
|---|---|---|
| Development | `pip install -e "./hybrax[dev]"` | pytest, ruff, black, flake8, mypy, ipdb, jupyter, openpyxl |
| Docs | `pip install -e "./hybrax[docs]"` | Sphinx, furo, myst-nb, autoapi, and the rest of the toolchain that builds this site |

Plotting (`plot_process`, `plot_collection`) needs no extra: matplotlib is a base
dependency.

## Verify

```bash
python -c "import hybrax; print(hybrax.__version__)"
hybrax --help
```

`hybrax` is installed as a console script. Everywhere in these docs you can substitute
`python -m hybrax.train.cli` for `hybrax` if you prefer not to rely on the script being
on your `PATH`.

## Things worth knowing before your first run

:::{admonition} float64 is forced on, globally
:class: important

Importing `hybrax.format` sets `JAX_ENABLE_X64=true` **before JAX loads**. Bioprocess
mass balances span many orders of magnitude and single precision loses them, so this is
deliberate, but it is global. If you import `jax` first and configure it yourself,
importing `hybrax.format` afterwards will flip x64 on underneath you, and float32 arrays
handed to `TimeSeries` will raise rather than silently upcast.
:::

:::{admonition} `hybrax.train` decides its device count at import time
:class: important

JAX fixes the number of CPU devices when it initializes, so `hybrax.train` resolves the
device count *before* that: by scanning the command line and config. The default is **1
device**, so it never quietly takes over your machine. `HYBRAX_TRAIN_DEVICES=N` in the
environment always wins over the config file. See [Training](../train/train.md).
:::

## Building these docs

This site lives under `docs/` in the `hybrax` repo. From the repo root:

```bash
pip install -e ".[docs]"
```

or, via conda:

```bash
conda env create -f docs/environment.yml
```

Then, from `docs/`:

```bash
bash docs_rebuild.sh
```

builds the committed `html/` output from scratch. Open `docs/html/index.html`
to preview. `docs_rebuild_fast.sh` does an incremental rebuild for local
iteration.

## See also

- [Quickstart](quickstart.md): the first thing to run once this works.
- [Troubleshooting](../troubleshooting/errors.md): if the verify step failed.
