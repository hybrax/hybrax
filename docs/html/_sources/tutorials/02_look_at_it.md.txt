---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

<!-- LOCK -->
# 2. View your Dataset
<!-- UNLOCK -->

> Before modeling anything, make the package tell you what it
> understood from your data.

Most bad models come from data that was described slightly wrong, not from a bad
network. This tutorial is four function calls that catch that early.

```{code-cell} ipython3
:tags: [remove-cell]

%matplotlib inline
```

```{code-cell} ipython3
import hybrax.format as hxf

collection = hxf.serialization.load_process_collection("../../../examples/tutorial_02_look_at_it/data.json")
process = collection.processes["run_1"]
```

## 2.1 Validate

For a `BioProcess`, `validate_process` returns `(ok, messages)` and collects the
checks' results instead of raising on the first failed check. Passing the wrong
object type to either validator raises `TypeError`.

```{code-cell} ipython3
ok, messages = hxf.validate_process(process)
print("ok:", ok)
for line in messages:
    print(" ", line)
```

Across a whole collection, `validate_for_publication` runs the same checks on every run
and adds cross-process checks that all runs share one structure: the same components,
process variables and units, and the same time-axis unit and reference (start and end
may differ). Training checks more on top of this, for example
that every run declares the same `reaction_ode`.

```{code-cell} ipython3
ok, report = hxf.validate_for_publication(collection)
print("ok:", ok)
for section, results in report.items():
    for check_ok, message in results:
        # Print every failure, plus the cross-process results even when they pass.
        if not check_ok or section.startswith("__"):
            print(f"  {section}: {message}")
```

The report has one section per run and two extra ones: `__consistency__` holds the
cross-process checks, and `__augmented__` checks that augmented runs point to existing
parents.

The checks that catch real bugs most often:

| Check | The bug it catches |
|---|---|
| volume change sign | A feed recorded as negative, or a sample as positive. |
| feed medium units | A feed with no feed medium, or a feed component whose unit differs from the reactor's. |
| measurement / sampling alignment | An offline measurement timestamped just *after* its own sample draw, which corrupts every dilution correction built on it. |
| biomass present | Auto-generated dynamics need a biomass component. |
| additive unit consistency | `biomass - product` where one is `g/L` and the other `mg/L`. |

## 2.2 Print the structure

```{code-cell} ipython3
hxf.print_process_structure(process, verbosity=2)
```

Raise `verbosity` to 3 to add units, bounds and the time and value range of each series;
drop to 1 for just the variable names.

## 2.3 Plot a process

```{code-cell} ipython3
fig = hxf.plot_process(process)
```

Look for the things that are hard to see in a table: a species that never moves, a trace
that jumps where nothing happened, a run that starts before inoculation.

## 2.4 Print the ODE

This is the one most people do not know exists, and it is the fastest way to check that
your description means what you think.

```{code-cell} ipython3
hxf.print_rhs_ode(process)
```

Read it as two halves. The **biological** half is what a model will predict: here the
three `q_*` rates. The **physical** half is what hybrax.format already wrote for you: feed
inflow, dilution, sample outflow, volume dynamics. In a batch run the physical half is
nearly empty, which is exactly why batch is the right place to start.

When you move to fed-batch, come back to this call. Everything that appears in the
physical half is a term you did not have to derive, and a term that can go wrong if the
feed metadata is wrong.

## What you learned

- Validation collects data-quality failures in a report. Passing the wrong object
  type raises `TypeError`.
- `print_rhs_ode` shows the boundary between "what you must model" and "what is already
  handled".
- Four calls, before any training: validate, print, plot, print the ODE.

## What's next

This tutorial has no output of its own: it reads the shared demo dataset directly from
`./examples/tutorial_02_look_at_it/`.

- **[Tutorial 3](03_train.md)**: train a model on this dataset.
- A check failed? [Errors](../troubleshooting/errors.md).
- More on these tools: [Validating and inspecting](../format/validate_and_inspect.md).
