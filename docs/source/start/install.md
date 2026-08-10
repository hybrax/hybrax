# Installation

> **In one sentence.** Clone both repos side by side, install them editable into one
> Python ≥ 3.12 environment, and check that `bp-train --help` runs.

## Requirements

- **Python ≥ 3.12** (bp-train's floor; bp-format alone would accept 3.10).
- A working JAX install. Both packages pull in `jax` / `jaxlib`; CPU is fine and is what
  these docs are built against.

## Layout

The two packages are separate repositories and are normally checked out as siblings:

```
bpbench/
├── bp-format/     # the data model + mechanistic RHS
├── bp-train/      # training on top of it
└── bp-docs/       # these docs
```

## Install

```bash
python -m pip install -e ./bp-format
python -m pip install -e ./bp-train
```

Editable installs (`-e`) are the norm here: both packages are under active development
and deliberately ship breaking changes, so you want your checkout and your imports to be
the same thing.

### Optional extras

| Extra | Command | Gives you |
|---|---|---|
| Plotting | `pip install -e "./bp-format[plotting]"` | `plot_process` / `plot_case_study` (matplotlib, plotly) |
| Development | `pip install -e "./bp-format[dev]"` | pytest, ruff, black, jupyter, openpyxl |
| Development | `pip install -e "./bp-train[dev]"` | pytest, ipdb, plotly |

`bp-train` already depends on matplotlib, so its own plots work without an extra.

## Verify

```bash
python -c "import bp_format, bp_train; print(bp_format.__version__, bp_train.__version__)"
bp-train --help
```

`bp-train` is installed as a console script. Everywhere in these docs you can substitute
`python -m bp_train.cli` for `bp-train` if you prefer not to rely on the script being on
your `PATH`.

## Things worth knowing before your first run

:::{admonition} float64 is forced on, globally
:class: important

Importing `bp_format` sets `JAX_ENABLE_X64=true` **before JAX loads**. Bioprocess mass
balances span many orders of magnitude and single precision loses them, so this is
deliberate, but it is global. If you import `jax` first and configure it yourself,
importing `bp_format` afterwards will flip x64 on underneath you, and float32 arrays
handed to `TimeSeries` will raise rather than silently upcast.
:::

:::{admonition} bp-train decides its device count at import time
:class: important

JAX fixes the number of CPU devices when it initializes, so bp-train resolves the device
count *before* that: by scanning the command line and config. The default is **1
device**, so it never quietly takes over your machine. `BP_TRAIN_DEVICES=N` in the
environment always wins over the config file. See
[Training](../train/train.md).
:::

## Building these docs

The docs live in `bp-docs` and are built against the *installed* packages:

```bash
python -m pip install -r bp-docs/requirements.txt
bash bp-docs/docs_rebuild.sh
```

That regenerates the demo datasets, executes every tutorial and gallery page for real,
and writes the site to `bp-docs/html/`. Open `html/index.html` directly: it works from
`file://`, search included. A cold build takes a few minutes; afterwards only pages whose
source changed are re-executed.

## See also

- [Quickstart](quickstart.md): the first thing to run once this works.
- [Troubleshooting](../troubleshooting/errors.md): if the verify step failed.
