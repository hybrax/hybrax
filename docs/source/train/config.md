# Configuration

> JSON files that are strict about typos, relative to themselves, and mostly optional.

## The commands

```bash
hybrax prepare --config prepare-config.json --output-dir prepared [--overwrite]
hybrax train   --config train-config.json [--output-dir DIR] [--overwrite]
                 [--epochs N] [--log-level LEVEL]
hybrax forward --config forward-config.json [--output-dir DIR] [--overwrite]
hybrax loo     --config loo-config.json [--output-dir DIR] [--overwrite]
hybrax loo     --resume RUN_DIR        # mutually exclusive with --config
```

`hybrax` is a console script; `python -m hybrax.train.cli` is equivalent.

Command-line flags override the config file. `--epochs` is there because it is the one
you change constantly while iterating.

## The minimum

Exactly one key is mandatory per command.

```json
// prepare-config.json: the whole file
{ "prepare": { "raw_input": "data.json" } }
```

```json
// train-config.json: the whole file
{ "data": { "prepared": "prepared" } }
```

```json
// forward-config.json: the whole file
{ "models": ["run"] }
```

Everything else (including `custom_py`) has a default. What you get from that minimal
train config: 5 epochs, Adam at 1e-3, gradient clipping at norm 1000, full batch, one
device, the default MLP and MSE modules, **no scaling**, and output in `./output`.

## Three rules

**1. Paths resolve relative to the config file, not your shell.** A config in
`experiments/run3/train.json` naming `"prepared": "prepared"` means
`experiments/run3/prepared`. This is what makes a config directory portable.

**2. Unknown keys are rejected.** Every section forbids extras and the top level is
checked explicitly, so `"epocs": 300` is a hard error rather than a silently ignored
setting. This is the single most useful piece of strictness in the package.

**3. Each command reads only its own sections.** A `prepare` block in a train config is
*ignored*, not rejected, which is what lets you keep one config file for the whole
pipeline if you want to.

## The sections

| Section | Used by | Holds |
|---|---|---|
| `prepare` | prepare | `raw_input`, validation strictness, required controls, `augmentation` |
| `data` | train, loo | `prepared`, `processes`, `targets`, `target_source` |
| `train` | train, loo | `epochs`, `seed`, `optimizer`, `learning_rate`, `grad_clip_norm`, `batch_size`, `shuffle`, `devices`, `allow_stateful_models` |
| `solver` | train, forward, loo | `max_steps`, `rtol`, `atol` |
| `checkpoint` | train, loo | `every` |
| `output` | all | `dir`, plotting |
| `logging` | all | `decimals`: rounding precision for logged numbers |
| `custom_py` | all | path to your hooks file |
| `custom` | all | free-form, handed to your hooks |
| `loo` | loo | `per_fold_holdout_sets`, `parallel_folds`, `devices_per_fold` |
| `models` | forward | list of run or checkpoint directories |

Exact fields, types and defaults are in the
[API reference](../autoapi/hybrax/train/run_config/index): not repeated here, because they
change and this page would be wrong first.

## A realistic config

```json
{
  // whole-line comments like this are allowed
  "data": {
    "prepared": "prepared",
    "target_source": "combined"
  },
  "custom_py": "custom.py",
  "train": {
    "epochs": 2000,
    "seed": 0,
    "learning_rate": 3e-4,
    "grad_clip_norm": 10.0,
    "devices": "max"
  },
  "solver": { "max_steps": 4096, "rtol": 1e-5, "atol": 1e-7 },
  "checkpoint": { "every": 100 },
  "custom": { "hidden_width": 32 }
}
```

Only **whole-line** `//` comments are stripped. A trailing comment after data is still a
JSON error, and so are trailing commas.

## `target_source`

Which measurements the loss is computed against.

| Value | Targets |
|---|---|
| `reactor_components` | Reactor medium components only. |
| `process_variables` | Process variables only. |
| `combined` | Both. |
| `auto` (default) | Decide from what the dataset actually has. |

Set it explicitly as soon as you have modeled process variables: `auto` is a
convenience, not a decision you want made implicitly on a dataset you care about.

## The `custom` block

Anything under `custom` is passed through to your hooks. By default it is wrapped in a
permissive container that accepts any fields:

```json
{ "custom": { "hidden_width": 32, "n_layers": 3, "smoothing": 0.1 } }
```

```python
def build_reaction_module(*, config, seed, **kwargs):
    width = config.custom.hidden_width
    ...
```

If you want it validated instead of permissive, define `get_custom_config` in your
`custom.py`. It runs **before** every other hook, and whatever it returns becomes
`config.custom` everywhere:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class MyConfig:
    hidden_width: int = 32
    n_layers: int = 2

def get_custom_config(raw_custom, config):
    return MyConfig(**raw_custom)
```

Now a typo in the `custom` block is a `TypeError` at startup rather than an
`AttributeError` three minutes into training.

There is also a module-level fallback: a `CONFIG` dict or a `get_config()` function in
`custom.py` is merged into the config, with `get_config()` winning if both exist. Useful
for defaults you do not want to repeat in every config file.

## Gotchas

- **`--overwrite` is required to reuse an output directory.** You will meet this on your
  second run. It is deliberate.
- **`--config` and `--resume` are mutually exclusive** on `loo`.
- **`prepare` fails rather than clobbering** an existing `prepared.json` without
  `--overwrite`.
- **A missing `custom_py` path is a `FileNotFoundError`**, but a missing *hook inside* a
  present file is silent. Different failure modes, and only one of them is loud.
- **`hybrax loo` has a hidden `--fold` flag.** It is internal worker dispatch. Do not
  use it.

## See also

- [Customization](hooks_cheatsheet.md): every hook in one table.
- [Prepare](prepare.md) · [Training](train.md) · [Forward](forward.md): per-stage detail.
- [Errors](../troubleshooting/errors.md): config errors and their fixes.
