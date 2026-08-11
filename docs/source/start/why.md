# Why this exists

> **In one sentence.** There is no shared format for bioprocess data and no shared
> implementation of bioprocess physics, so every project pays the same tax twice: 
> once reading the data, once re-deriving the mass balance.

## The problem

Bioprocess datasets arrive in whatever shape the person who ran the experiment
happened to use. One file has offline samples in a wide CSV and the feed profile in a
separate export from the control software. Another has cumulative base addition but no
feed composition. A third records sampling times but not sampling volumes. None of them
agree on units, on what `t = 0` means, or on whether a column is a rate or a cumulative
total.

So the first two weeks of every modeling project look the same:

1. Write a bespoke parser for this dataset.
2. Re-derive the mass balance: dilution from feeds, mass added by each feed stream,
   volume removed by sampling, discontinuities at boluses.
3. Discover a sign convention error somewhere in step 2.
4. Only then start on the actual modeling question.

Steps 2 and 3 are the expensive part, and they are *the same every time*. The dilution
term `(f_k / V) · (C_in[k,i] − c_i)` does not depend on your organism or your
hypothesis. Neither does "a sample is a well-mixed removal: concentrations unchanged,
volume drops". Everybody re-implements them, and a meaningful fraction of published
bioprocess models contain a quiet transport bug as a result.

## The idea

Pay the cost once, at the boundary.

**Get the data into bp-format once.** That means answering the questions the physics
needs answered anyway: what was in the reactor, what went in and at what concentration,
what came out and when, on what clock. It is real work, and [Tutorial
1](../tutorials/01_your_first_dataset.md) is exactly that work.

**After that, everything else is available.** Because the description is complete,
bp-format can assemble the mechanistic right-hand side for you (feed inflow, dilution,
sample outflow, volume dynamics, discrete event jumps) and hand bp-train a
differentiable ODE where the only thing left undetermined is the biology. You supply
the part that is actually your research question: how fast do the cells do things, and
how do you want to be scored on it.

The same dataset then feeds a mechanistic Monod fit, a neural ODE, a hybrid of the two,
and leave-one-process-out cross-validation, with no further data work.

## What this buys you concretely

- **Transport is implemented once and tested once.** You write rates; the physics is
  already there.
- **Datasets become comparable.** Batch and fed-batch runs from different labs end up
  in the same object, so the same model can be trained across them.
- **Errors surface at load time, not in your results.** A feed with no composition, a
  sample with a positive volume, a measurement timestamped just after a sampling
  event: these are caught by validators instead of quietly biasing a fit.
- **Swapping the model is cheap.** The reaction module and the loss module are two
  small, replaceable objects. Everything else stays.

## What this is not

- Not a parameter database and not a protocol source. It stores *your* data.
- Not a unit engine. Units are recorded as strings and checked for consistency, never
  converted. See [Limits and gotchas](../format/limits_and_gotchas.md).
- Not a general reactor simulator. The model is a well-mixed vessel. Perfusion with
  cell retention and evaporation with solute retention are not implemented.
- Not automatic. Nothing here decides *which* model is right for your process. It
  removes the plumbing so you can spend your attention on that.

## See also

- [Install](install.md): get bp-format and bp-train set up.
- [Quickstart](quickstart.md): see the whole loop run in ten minutes.
- [Concepts and vocabulary](concepts.md): the terms used throughout these docs.
- [Design rationale](../under_the_hood/design_rationale.md): the architectural
  decisions and why they were made that way.
