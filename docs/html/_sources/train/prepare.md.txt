# Prepare

> The step where a dataset stops being data and becomes a training problem: done once,
> reused by every model you fit against it.

```bash
hybrax prepare --config prepare-config.json [--output-dir DIR] [--overwrite]
```

## Why it is a separate command

Because it is expensive, deterministic, and shared. Prepare resolves everything that is a
property of *the dataset* rather than *the model*:

- which measured quantities are the fit **targets**;
- the canonical **state and control layout** (from hybrax.format's `ProcessOrdering`);
- **control splines**: the continuous inputs, fitted once so the solver can evaluate them
  at arbitrary `t`;
- **discrete events**: bolus and sample times and their jumps;
- **validation** of both hybrax.format structure and prepared-artifact semantics.

Once written, training reads only `prepared/prepared.json`. Twenty models fitted against
the same prepared artifact start from a byte-identical problem, which is what makes
comparisons between them meaningful.

## What it writes

```
prepared/
├── prepared.json                     the artifact: the only thing training reads
├── prepare_config.json               exactly what produced it
├── prepare_diagnostics/
│   └── <process>_controls.png        how each control was interpreted
└── augmented-data.png                only if augmentation is configured
```

**Look at the diagnostics the first time you prepare your own data.** They show the
fitted control traces against the raw points: a feed spline that overshoots, or a
smoothing setting that flattened a real step change, is obvious there and invisible
later.

## Configuration

```json
{
  "prepare": {
    "raw_input": "data.json",
    "strict_format_validation": false
  },
  "custom_py": "custom.py"
}
```

`raw_input` accepts a `BioProcessCollection`, as a file or a directory.
`strict_format_validation` decides whether hybrax.format validation failures
stop the run or are reported and tolerated: set it `true` for a dataset you intend to
publish.

Splines are whatever `hybrax.format` fits: an exact interpolating fit by default
(`smoothing_s=0`). To smooth a noisy control before prepare uses it, fit your own
spline in `transform_process_collection` and pass a nonzero `smoothing_s`: see
[Time series and splines](../format/time_series_and_splines.md#fitting-one).

## Hook: `transform_process_collection`

**Fires:** during prepare, on the loaded collection, before anything is derived.
**Signature:** `(collection, config) -> collection`
**Default:** applies `prepare.process_rename_map`.

This is the hook for "my dataset needs one adjustment before it is usable". It is the
last point at which you can change the *data*; everything after it is derived.

```python
def transform_process_collection(collection, config):
    """Drop the first hour of every run: the inoculation transient."""
    del config
    for process in collection.processes.values():
        for component in process.reactor_medium.components.values():
            ts = component.concentration
            if ts.times is None:
                continue
            keep = np.asarray(ts.times) >= 1.0
            component.concentration = TimeSeries(
                times=np.asarray(ts.times)[keep],
                values=np.asarray(ts.values)[keep],
            )
        process.time_axis.start = 1.0
    return collection
```

Three things it is genuinely used for:

**Making a fixed derivative learnable.** If your `reaction_ode` drives a process
variable with a hard-coded relaxation and you would rather learn it, swap the derivative
for a named rate and declare it: the reaction module then predicts it:

```python
def transform_process_collection(collection, config):
    del config
    for process in collection.processes.values():
        ode = process.reaction_ode
        if ode is None or "product_ratio" not in ode.derivatives:
            continue
        ode.derivatives["product_ratio"] = "r_product_ratio"
        ode.rates["r_product_ratio"] = (None, None)
    return collection
```

**Smoothing controlled signals.** A noisy online trace is about to be differentiated to
give a flow rate, and noise differentiates badly. Fitting a smoothing spline here
(`fit_timeseries_spline(series, smoothing_s=0.1)`) is usually worth it, and it is also
the fix for the pseudobatch performance cliff described in
[Time series and splines](../format/time_series_and_splines.md).

**Trimming events outside the time axis.** An event recorded at or after `time_axis.end`
has nowhere to be applied.

## Targets

A target is a measured quantity the loss is computed against. Which measurements qualify
comes from `data.target_source` ([Configuration](config.md)), filtered by
`data.targets` if you name them explicitly.

:::{admonition} Every target needs a measurement at t₀
:class: warning

This is the most common first-run failure on real data. If a target has no measured value
at the first time of the union grid, training stops with:

```
Process 'run_1': target 'glucose' has no measurement at union_grid t[0] = 0.0
```

The initial condition has to come from somewhere. Three fixes, in the message: supply a
t₀ measurement, represent the quantity as a `StaticVariable` if it genuinely does not
vary, or drop it from the targets.

Real datasets often start offline sampling at t = 1 h. Either backfill t = 0 from the
medium recipe (which you know) or move `time_axis.start`.
:::

Targets must also be **consistent across processes**. A model trained on runs that
measure different things has no single output layout; this is rejected rather than
silently intersected.

## Augmentation

`prepare.augmentation` generates synthetic sibling processes (resampled in time, with
noise) to enlarge a small dataset. Children are named `{parent}__aug_{NNN}` and carry a
`parent_process` reference so cross-validation can keep a parent and its synthetic
children in the same fold. Without that grouping, an augmented sibling in the training
set leaks the held-out parent.

The `augment_state_values` hook lets you control the generated values per state. This is
advanced; see [Gallery](../gallery/index.md).

## Gotchas

- **Prepare does not overwrite without `--overwrite`.** It exits non-zero with an error.
- **`--overwrite` deletes everything already in `--output-dir`**, regardless of what put it
  there, before writing fresh output. There is no partial or selective overwrite.
- **`--output-dir` is optional.** It overrides `output.dir` from the config, which itself
  falls back to the literal `output/` if neither is set.
- **`transform_process_collection` must return the collection.** Mutating in place and
  returning `None` gives you a `None` collection downstream.
- **Prepare is where bad volume descriptions become visible**: via the diagnostic plots,
  not via an error. Look at them.
- **Programmatic prepare takes a loaded config, not a path**: call `load_prepare_config`
  first, then `prepare_artifact`.

## See also

- [Configuration](config.md): the `prepare` section in context.
- [Validating and inspecting](../format/validate_and_inspect.md): do this before preparing.
- [Training](train.md): what consumes the artifact.
- [Customization](hooks_cheatsheet.md).
