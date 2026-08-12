# Why this exists

<!-- LOCK -->
> There is no shared format for bioprocess data and no shared implementation of
> bioprocess physics. Thus, every new bioprocess modeling project has to
> reinvent the wheel in several places:
> - parse, clean, and validate input data
> - derive and enforce mass balance
> - implement reactor dynamics as segmented ODEs: biological rates, dilution, sampling, boluses, ...
> - set up loss function and training loop
> - train a model ensemble with cross-validation
>
> This is a lot of code you have to write before you can even ask your modeling question.
> The goal of this project is to abstract all of that away so that you can focus on your modeling idea, implemented as a single `equinox.Module`.
<!-- UNLOCK -->

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

Step 1 will always require at least some bespoke work, but step 2 is the same
every time. The term `(f_k / V) · (C_in[k,i] − c_i)` does not depend on your
organism or your hypothesis. Neither do the mechanics of sampling
(concentrations stay the same, volume drops) or of a bolus event. Yet, everybody
has to implement them from scratch, and every implementation is a potential
source of bugs.

## The idea

Pay the cost once, at the boundary.

**Get the data into bp-format once.** That means answering the questions the
physics needs answered anyway: what was in the reactor at `t=0` (and what does
`t=0` actually mean: batch start, batch end, induction?), what went in later
(and at what concentration), what came out? It is a little tedius, but when it's
done, it's done. See [Tutorial 1](../tutorials/01_your_first_dataset.md) for
details.

**After that, this package provides everything else.** Because the description is complete,
bp-format can assemble the mechanistic right-hand side for you (feed inflow, dilution,
sample outflow, volume dynamics, discrete event jumps) and hand bp-train a
differentiable ODE where the only thing left undetermined are the biological dynamics.

Next, you supply the part that is actually your research question: what do the
cells do and how do you want to model it? As long as you can write down the
right-hand side of your ODEs as JAX code you're good to go.

This means that, once a dataset is `bp-format`, you can easily swap out one
model for another. It doesn't matter whether it's a [mechanistic Monod
fit](../gallery/mechanistic_rates.md), a neural ODE, a [hybrid of the
two](../gallery/fed_batch.md) (most popular), or a [hybrid models with
GPs](../gallery/gaussian_process.md).

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
  cell retention and evaporation with solute retention are not yet implemented.
- Not automatic. Nothing here decides *which* model is right for your process. It
  removes the plumbing so you can spend your attention on that.

## See also

- [Install](install.md): get bp-format and bp-train set up.
- [Quickstart](quickstart.md): see the whole loop run in ten minutes.
- [Concepts and vocabulary](concepts.md): the terms used throughout these docs.
- [Design rationale](../under_the_hood/design_rationale.md): the architectural
  decisions and why they were made that way.
