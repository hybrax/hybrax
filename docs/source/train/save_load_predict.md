# The Python API

> Every other stage runs from the command line. This one runs from a script: loading a
> trained model, predicting with it, and reading its provenance, all in Python. Only the
> trainable parameters are saved; everything else is rebuilt, which is what makes run
> directories portable, and what makes one loading function dangerous.

## What is on disk

`save_model` writes **only the trainable partition** to `params.eqx`. The static half (the controls store, the assembled `RhsOde`, the index tables, every `SCALE_*`) is
**always rebuilt** at load time from the `prepared.json.gz` and `custom.py` bundled
alongside.

Three consequences worth knowing:

- Checkpoints are small.
- A change in the controls store cannot cause a shape mismatch on load.
- **The `custom.py` is part of the model.** Reconstruction runs your hooks again, so a run
  directory without it cannot be loaded.

Every run *and* checkpoint directory is self-contained:

```
run/
├── config.json          exactly what was run
├── custom.py            your hooks, frozen
├── prepared.json.gz     the data, frozen
├── metrics.csv          per-step loss and timing
├── losses.csv           final per-process, per-target loss
├── predictions.csv      dense trajectories, if output.predictions selects any
├── loss_curve.png
├── grad_norm_curve.png
├── model/
│   ├── params.eqx       trainable parameters only
│   └── opt_state.eqx    optimizer state, for resuming
└── checkpoints/step_NNNNN/    the same again, per checkpoint, plus train_state.json
```

You can copy a run directory to another machine and load it, provided hybrax is
installed.

## Loading

```python
import hybrax.train as hxt

wrapper, config = hxt.model_load("run")
```

`model_load` rebuilds everything from the directory's own bundled data. It is the one you
want, essentially always.

:::{admonition} `model_reload` keeps the static half you hand it
:class: danger

There is a second entry point, `model_reload`, which reuses an existing static half rather
than rebuilding from a directory. Point it at a *different* dataset and it will load the
trained weights into a **different scaled space**: the model was fitted with one set of
`SCALE_*` values and is now evaluated under another.

There is no exception, no shape error and no `NaN`. The predictions are simply wrong, and
they look plausible. Use `model_load` unless you specifically know why you need the other.
:::

## Predicting

```python
import hybrax.format as hxf

collection = hxf.serialization.load_process_collection("data.json")

predictions = hxt.model_predict(wrapper, config, collection, grid_n=200)
export = predictions["run_1"]
export.t, export.c_species, export.q_rates, export.v_real, export.auxiliary
```

**Neither `model_predict` nor the CLI `forward` path re-estimates scales against your
evaluation data.** Both use exactly the scales the model was trained under. What differs
is whether `custom.py` is needed at all: see
[Forward](forward.md) and [Silent failures](../troubleshooting/silent_failures.md).

## Resuming training

`opt_state.eqx` sits next to `params.eqx` precisely so a run can continue with the
optimizer's momentum intact rather than restarting cold. Checkpoint frequency:

```json
{ "checkpoint": { "every": 100 } }
```

Because each checkpoint is self-contained, it re-exports predictions and re-writes the
bundled data. On a fast run that can dominate the wall clock: set `every` coarse enough
that checkpointing is not the bottleneck.

For LOO, resuming is a first-class command: `hybrax loo --resume RUN_DIR` re-runs only
the folds that never finished. See [Cross-Validation](loo.md).

## Provenance

Runs record content hashes and environment versions alongside the config. Given a result,
you can tell which data and which code produced it: worth checking before you conclude
that two runs disagree, because most of the time they were not run on the same thing.

## Inspection

```python
hxt.print_trainable_structure(wrapper)   # what is optimized, what is frozen
hxt.print_reaction_schema(wrapper)       # which array index is which species
```

Run the first before any long training: an untagged field is silently frozen and will
never move. Run the second whenever you are about to index into a state or rate vector.

## Gotchas

- **A run directory without its `custom.py` cannot be loaded.** Reconstruction needs it.
- **Checkpoint directories work anywhere a run directory does**: including as `models`
  entries for `forward`.
- **Loading resolves paths in a fixed order**, preferring the directory's own bundled
  data. That is what makes a copied directory work.
- **`params.eqx` alone is not a model.** It is the trainable half; the rest is rebuilt.

## See also

- [Tutorial 5](../tutorials/05_predict.md): the walkthrough.
- [Forward](forward.md): the CLI path.
- [Silent failures](../troubleshooting/silent_failures.md): the `model_reload` hazard in
  context.
- [Design rationale](../under_the_hood/design_rationale.md): why only the trainable
  partition is saved.
