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

# Tutorial 2: look at it

> **In one sentence.** Before modeling anything, make the package tell you what it
> understood from your data.
>
> **You need this if** you just built a dataset. **You can skip it if** you enjoy
> debugging a training run instead.

Most bad models come from data that was described slightly wrong, not from a bad
network. This tutorial is four function calls that catch that early.

```{code-cell} ipython3
:tags: [remove-cell]

%matplotlib inline
```

```{code-cell} ipython3
import bp_format as bp

case_study = bp.serialization.load_case_study("../_data/out/demo_batch/data.json")
process = case_study.processes["run_1"]
```

## 1. Validate

`validate_process` returns `(ok, messages)` and — importantly — **collects every issue
in one pass** rather than raising on the first one. You get a full report, not a
whack-a-mole session.

```{code-cell} ipython3
ok, messages = bp.validate_process(process)
print("ok:", ok)
for line in messages:
    print(" ", line)
```

Across a whole case study, `validate_case_study` adds cross-process checks — that every
run has the same structure, so a model trained on one can be applied to another:

```{code-cell} ipython3
ok, per_process = bp.validate_case_study(case_study)
print("ok:", ok)
print("checked:", list(per_process))
```

The checks that catch real bugs most often:

| Check | The bug it catches |
|---|---|
| volume change sign | A feed recorded as negative, or a sample as positive. |
| feed medium covers all species | A feed whose composition omits a reactor species — "absent" confused with "unrecorded". |
| measurement / sampling alignment | An offline measurement timestamped just *after* its own sample draw, which corrupts every dilution correction built on it. |
| biomass present | Auto-generated dynamics need a biomass component. |
| additive unit consistency | `biomass - product` where one is `g/L` and the other `mg/L`. |

## 2. Print the structure

```{code-cell} ipython3
bp.print_process_structure(process, verbosity=2)
```

Raise `verbosity` to 3 for every value; drop to 1 for a one-line-per-object summary.

## 3. Plot it

```{code-cell} ipython3
fig = bp.plot_process(process)
```

This needs the plotting extra (`pip install -e "./bp-format[plotting]"`). Look for the
things that are hard to see in a table: a species that never moves, a trace that jumps
where nothing happened, a run that starts before inoculation.

## 4. See the ODE that was assembled

This is the one most people do not know exists, and it is the fastest way to check that
your description means what you think.

```{code-cell} ipython3
bp.print_rhs_ode(process)
```

Read it as two halves. The **biological** half is what a model will predict — here the
three `q_*` rates. The **physical** half is what bp-format already wrote for you: feed
inflow, dilution, sample outflow, volume dynamics. In a batch run the physical half is
nearly empty, which is exactly why batch is the right place to start.

When you move to fed-batch, come back to this call. Everything that appears in the
physical half is a term you did not have to derive, and a term that can go wrong if the
feed metadata is wrong.

## What you learned

- Validation is non-raising and exhaustive — run it, read all of it.
- `print_rhs_ode` shows the boundary between "what you must model" and "what is already
  handled".
- Four calls, before any training: validate, print, plot, print the ODE.

## What's next

- **[Tutorial 3](03_train.md)** — train a model on this dataset.
- A check failed? [Errors](../troubleshooting/errors.md).
- More on these tools: [Validating and inspecting](../format/validate_and_inspect.md).
