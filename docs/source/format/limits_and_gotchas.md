# Limits and gotchas

> What bp-format deliberately does not do, and the sharp edges in what it does. Worth
> reading the first section before you design a dataset around an assumption that does
> not hold.

## Not implemented

These are not bugs and not oversights; they are boundaries. If your process needs one of
them, you need to know now rather than halfway through.

| Limit | Detail |
|---|---|
| **Time-varying feed composition** | A `FeedMediumComponent` concentration may be a `TimeSeries` in the schema, but `build_rhs_ode` raises `NotImplementedError`. Use `StaticVariable`; split genuinely changing feeds into separate streams. |
| **Perfusion / cell retention** | The vessel model is a well-mixed CSTR. There is no mechanism for retaining cells while removing liquid. A perfusion process cannot be described correctly. |
| **Evaporation with solute retention** | Same reason: removing solvent while keeping solutes is not representable. |
| **Pseudobatch on a continuous `Outflow`** | `build_pseudobatch_transform` raises `NotImplementedError` for any continuous `Outflow` (perfusion, continuous harvest), retention or not. Its closed-form ADF assumes volume only grows from `Inflow`s. Only `Inflow` and discrete `Outflow` (sampling) are supported. See [The pseudobatch transform](pseudobatch_transform.md). |
| **Rate inversion** | There is no facility for computing rates analytically from measured concentrations. Rates come from a model. |
| **Unit conversion** | Units are free-form strings. Nothing is parsed and nothing is converted. |

The last one deserves emphasis: units are used for exactly two checks, that quantities
you *add* in a `biological_ode` expression share a unit, and that processes in a case
study agree. `"g/L"` and `"g/l"` are different strings. Pick one spelling.

## Things that raise

Loud failures, which is the design intent. Each of these is better than the silent
alternative.

**At construction**

- **No `biomass` component** and no explicit `biological_ode` → `BioProcess(...)` raises.
  Auto-generated rates are specific, so they need a biomass.
- **`TimeSeries` with neither samples nor spline** → raises. So does `times` without
  `values`, non-increasing `times`, or a `coeffs`/`breaks` shape mismatch.
- **float32 input** → raises rather than upcasting.

**When the process ordering is built**

- A **`StaticVariable` process variable with `is_controlled=False`**: a state with no
  time axis cannot be integrated.
- An **`Inflow` with no `feed_medium`**.
- A **feed naming a species not in `reactor_medium.components`**.
- **Name collisions across groups**: the same name used as both a state and a rate.
- **Cyclic `algebraic` dependencies.**

**During a solve**

- **Reactor volume at or below `1e-10`**: dilution divides by `V`, so this aborts rather
  than producing infinities.
- **`|ADF| ≤ 1e-12`** in the pseudobatch machinery, for the same reason.

## Things that do not raise

The dangerous category. None of these produce an error; all of them produce wrong
numbers.

- **`build_pseudobatch_transform` does not set `process.pseudobatch_transform`.** It
  writes `pseudobatch_concentration` onto each component in place and *returns* the
  bundle. If
  you ignore the return value, the components look transformed but the process has no
  transform attached.
- **Feed composition omitting a species.** Only a validator catches it, and only if you
  run the validator. A missing species means a missing dilution term.
- **Sample volume recorded as `0.0`.** That is an assertion that sampling removed nothing,
  which is different from "unknown". If it is unknown, say so; do not write a zero.
- **Bounds are never enforced.** They are metadata for downstream consumers.
- **Saving does not validate.** You can write a file with a sign-flipped feed.

## Performance cliffs

- **Pseudobatch on unfitted control traces.** If continuous-feed `TimeSeries` carry no
  spline, the transform ends up with roughly one polynomial piece per raw sample. On a
  densely logged online trace that is tens of thousands of pieces and seconds per
  species. Fit the control splines first: it is about a hundredfold difference.

## API surprises

- **`bp.inspect` is not a module handle.** `bp.inspect.plot_process` raises
  `AttributeError` on a fresh import and only starts working after something else has
  pulled the submodule in. Use `bp.plot_process(...)`. Same for `bp.simulation`.
- **Save/load are not on the root.** `bp.serialization.save_process_collection`, not
  `bp.save_process_collection`.
- **`PPoly` is not root-exported.** `from bp_format.time_series import PPoly`.
- **`plot_timeseries` is not root-exported.** `from bp_format.inspect import plot_timeseries`.
- **Importing `bp_format` sets `JAX_ENABLE_X64=true` globally**, before JAX loads. If you
  configured JAX yourself first, this changes it underneath you.
- **`AugmentedBioProcess` is a shape with no producer** in bp-format. It exists so
  bp-train's augmentation can rely on it.
- **`DiscreteEvents` is a mirror, not the source of truth.** Events live in
  `volume.volume_changes` with `is_continuous=False`.

## See also

- [Errors](../troubleshooting/errors.md): message-to-fix index.
- [Silent failures](../troubleshooting/silent_failures.md): the bp-train equivalent of
  this page's middle section.
- [Further reading](further_reading.md), where the exhaustive reference lives.
