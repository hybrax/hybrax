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

# Validating and inspecting

> **In one sentence.** Make the package report what it understood, before you spend a day
> training against a misdescribed dataset.
>
> **You need this if** you have data. **You can skip it if**: you can't. This is the
> cheapest page in the guide.

## Validation is non-raising and exhaustive

Every validator returns `(ok, message)` rather than throwing. That is deliberate: it lets
the aggregate collect **all** problems in one pass, so you fix a dataset once instead of
discovering issues one failed run at a time.

```{code-cell} ipython3
import bp_format as bp

case_study = bp.serialization.load_case_study("../_data/out/demo_fedbatch/data.json")
process = case_study.processes["fedbatch_1"]

ok, messages = bp.validate_process(process)
print("ok:", ok)
for line in messages:
    print(" ", line)
```

`validate_case_study` runs the same per-process checks and adds cross-process ones: 
principally that every run has the same structure, so a model trained on one can be
applied to another.

```{code-cell} ipython3
ok, per_process = bp.validate_case_study(case_study)
print("ok:", ok, "| sections:", list(per_process))
```

Note the two synthetic sections: `__consistency__` holds the cross-process results and
`__augmented__` checks that augmented children reference real parents.

## What each check is actually protecting you from

| Check | The real-world bug |
|---|---|
| `validate_volume_change_sign` | A feed imported as negative or a sample as positive. The single most common import bug. |
| `validate_volume_change_states` | A feed medium that omits a reactor species: "not present" silently confused with "not recorded". |
| `validate_measurement_sampling_alignment` | An offline measurement timestamped just *after* its own sample draw. Corrupts the dilution factor and every spline built on it. |
| `validate_biomass_in_reactor_medium` | Auto-generated dynamics with nothing to be specific *to*. |
| `validate_timeseries_shape` | Mismatched `times`/`values` lengths. |
| `validate_biological_ode` | A dynamic state with no derivative entry. Omission is rejected; write `"0"` if you mean no dynamics. |
| `validate_bounds` | An inverted or impossible `(lo, hi)`. |

:::{admonition} Measurements exactly *at* a sampling time are correct
:class: note
The alignment check is not complaining that a measurement coincides with a sample, that
is the normal case, since the sample is where the measurement came from. It flags
measurements timestamped *just after*, which implies the offline value describes
post-removal broth. Usually that is a timestamp-rounding artifact in the export.
:::

### The one that is not in the aggregate

`validate_volume_consistency` needs a number only you know (the measured final volume) 
so it is not part of `validate_process`:

```{code-cell} ipython3
import json
truth = json.loads(open("../_data/out/demo_fedbatch/ground_truth.json").read())

ok, message, total_change = bp.validate_volume_consistency(
    process, final_volume=truth["final_volume"])
print(f"ok={ok}   total volume change = {total_change:+.4f} L\n")
print(message)
```

Initial volume plus every signed change should land on the volume you actually measured.
When it does not, the discrepancy tells you which stream is mis-scaled.

## Inspecting

### As text

```{code-cell} ipython3
bp.print_process_structure(process, verbosity=1)
```

`verbosity` runs 1 to 3: 1 is one line per object, 3 prints values. `print_case_study_structure`
does the same across a whole study.

### As plots

```{code-cell} ipython3
:tags: [remove-cell]

%matplotlib inline
```

```{code-cell} ipython3
fig = bp.plot_process(process)
```

Needs the plotting extra (`pip install -e "./bp-format[plotting]"`). One panel per
variable, shared x-axis. What you are looking for is the stuff a table hides: a trace
that never moves, a jump where no event happened, a run that starts before inoculation.

### As equations

The most under-used call in the package:

```{code-cell} ipython3
bp.print_rhs_ode(process)
```

This is the assembled right-hand side: the biological half you must supply, and the
physical half bp-format already wrote. Compare it against the fed-batch structure you
described: every feed should appear, every sample should appear, and nothing should
appear twice.

## Gotchas

- **`bp.inspect` is not a module handle.** `bp.inspect.plot_process` raises
  `AttributeError` on a fresh import (it starts working only after some other access has
  pulled the submodule in). Always use `bp.plot_process(...)` on the root.
- **`plot_timeseries` is not root-exported**: import it as
  `from bp_format.inspect import plot_timeseries`.
- **`validate_process` raises `TypeError`** if handed something that is not a
  `BioProcess`. That one *is* a hard error, because it is a programming mistake rather
  than a data-quality issue.
- **Validation is not run on save or load.** It is yours to call.

## See also

- [Tutorial 2](../tutorials/02_look_at_it.md): the same four calls as a walkthrough.
- [Errors](../troubleshooting/errors.md), when a check fails and you need the fix.
- [The Bioprocess ODE](bioprocess_ode.md): reading `print_rhs_ode` output properly.
