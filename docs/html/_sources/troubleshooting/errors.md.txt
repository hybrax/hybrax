# Errors

> Error message → what it means → what to do. If nothing raised but the answer is wrong,
> see [Silent failures](silent_failures.md) instead.

hybrax prefers failing loudly to falling back quietly, so most of what you meet
here is the design working. Search this page for a fragment of your message.

## Getting data in

### `auto-generated BiologicalOde requires a 'biomass' component`

Raised when you construct a `BioProcess`.

**Why.** Auto-generated dynamics use *specific* rates (per unit biomass) so there has to
be a biomass to be specific to. The lookup is case-insensitive on the component name.

**Fix.** Either name your biomass component `biomass`, or write
[`biological_ode`](../format/bioprocess_ode.md#writing-your-own) yourself. A dataset with
no biomass at all (a chemical process, an abiotic control) needs the explicit form.

### `Provide discrete samples and/or spline representation`

**Why.** A `TimeSeries` with neither samples nor a fitted spline holds nothing.

**Fix.** Pass `times` **and** `values` together. Passing only one is also an error. If the
quantity does not vary, use `hxf.StaticVariable(value)` instead.

Related shape errors from the same constructor: `times` must be strictly increasing;
`coeffs` must be `(n_pieces, 4)` with `len(breaks) == n_pieces + 1`; a spline needs all
three of `breaks`, `coeffs` and `segment_start_piece_idx`.

### A float32 array raises instead of being accepted

**Why.** Importing `hybrax.format` turns on JAX's x64 mode, and the package refuses to silently
upcast. Bioprocess mass balances span orders of magnitude, and single precision loses them.

**Fix.** `np.asarray(x, dtype=float)`.

### `A state with no time axis cannot be integrated`

**Why.** A `ProcessVariable` with `values=StaticVariable(...)` and `is_controlled=False`
claims to be a dynamic state with nothing to be dynamic over.

**Fix.** Mark it `is_controlled=True` if it is a known input, or give it a real
`TimeSeries`.

### An `Inflow` has no `feed_medium` / names an unknown species

**Why.** A feed is litres *of something*; and hybrax.format will not invent a reactor state
for a species that only appears in a feed.

**Fix.** Attach a `FeedMedium`, and make sure every species it names exists in
`reactor_medium.components`. See [Volume, feeds and events](../format/volume_feeds_events.md).

### `NotImplementedError` from `build_rhs_ode` on a feed concentration

**Why.** Time-varying feed composition is allowed by the schema but not implemented.

**Fix.** Use `hxf.StaticVariable`. A feed whose composition genuinely changed must be split
into separate streams. See [Limits and gotchas](../format/limits_and_gotchas.md).

### Name collisions, or a cyclic `algebraic` graph

**Why.** The same name used as both a state and a rate has no unambiguous slot in the
layout; a cycle in `algebraic` cannot be evaluated.

**Fix.** Rename. See [The Bioprocess ODE](../format/bioprocess_ode.md).

## Validation reports (these do not raise)

`validate_process` returns `(ok, messages)` rather than throwing, so read the whole list.

| Report | Meaning | Fix |
|---|---|---|
| volume change sign | A feed has negative values, or a sample positive | Flip the sign, or the type |
| feed medium missing species | A reactor species has no declared feed concentration | Declare it, including `StaticVariable(0.0)` |
| measurement / sampling alignment | A measurement is timestamped just *after* its own sample draw | Usually rounding in the export; move it to the sample time |
| missing derivative | A dynamic state has no `derivatives` entry | Add one; write `"0"` if it has no biological dynamics |
| additive unit mismatch | `biomass - product` with different units | Make units consistent: they are strings, never converted |

`validate_process` **does** raise `TypeError` if handed something that is not a
`BioProcess`. That is a programming error, not a data-quality one.

## Training

### `target '<name>' has no measurement at union_grid t[0]`

The most common first-run failure on real data.

**Why.** Every target needs an initial condition, and it has to come from a measurement at
the first time point.

**Fix**, in the message itself:

- supply a t₀ measurement: often you *know* it, from the medium recipe;
- represent the quantity as a `StaticVariable` if it genuinely does not vary;
- drop it from the targets.

Real datasets frequently start offline sampling at t = 1 h. Either backfill t = 0, or move
`time_axis.start` to where measurement actually begins.

### `unknown top-level config key(s): …`

**Why.** Config sections forbid extra fields and the top level is checked explicitly, so a
typo is fatal rather than ignored.

**Fix.** Check spelling against [Configuration](../train/config.md). Note that a section
belonging to *another command* (a `prepare` block in a train config) is ignored rather
than rejected; that is not this error.

### `batch_size` greater than the number of processes

**Why.** Not clamped, deliberately: it usually means the process selection is not what you
thought.

**Fix.** Lower `train.batch_size`, or check `data.processes`.

### `Stateful reaction modules (n_latent > 0) require explicit opt-in`

**Fix.** `{"train": {"allow_stateful_models": true}}`. The opt-in exists because a latent
state changes what the model *is*, not just its size.

### `TypeError` from a hook return value

**Why.** `estimate_all_scales`, `build_reaction_module` and `build_loss_module` are
type-checked.

**Fix.** Return `EstimatedScales`, a `UserReactionModule` and a `UserLossModule`
respectively. The other four hooks are not checked.

### `ReactionOutputs.__init__() missing 1 required positional argument: 'SCL_modeled_Inflows_rates'`

**Why.** `SCL_modeled_Inflows_rates` and `SCL_modeled_Outflows_rates` are both required,
even when the process has no modeled feeds or outflows. There is no default for either.

**Fix.** `SCL_modeled_Inflows_rates=jnp.zeros(self.n_modeled_Inflows)`,
`SCL_modeled_Outflows_rates=jnp.zeros(self.n_modeled_Outflows)`.

### `loss_names` does not match the returned keys

**Why.** The names label plot panels and metric columns, so they are fixed for the run.

**Fix.** Make `loss_names` exactly equal the `named_losses` keys, in the same order.

### An offset was rejected on a rate axis

**Why.** Rate scaling must be a pure multiplication, that is what makes one factor work
for both a value and its derivative.

**Fix.** Use a `LinearScaler` (or a bare array) for `SCALE_*_rates`. Affine scaling is for
value axes. See [Scaling](../train/scaling.md).

### `FileNotFoundError` on `custom.py`

**Why.** `custom_py` points at a file that is not there: remembering that config paths
resolve relative to *the config file*.

**Fix.** Check the path. Note the asymmetry: a missing *file* raises; a missing *hook
inside* the file is silent.

### Solves keep failing / the loss plateaus high

Not one error, but the common cluster.

1. Raise `solver.max_steps` (default 2048).
2. Check `grad_norm_curve.png`, if it is pinned at `grad_clip_norm`, your step size is
   not what you think.
3. **Suspect scaling before anything else.** Most apparent stiffness on these problems is
   an axis four orders of magnitude away from the others. See [Scaling](../train/scaling.md).

### Something was OOM-killed

**Why.** Several JAX processes each claiming cores. `parallel_folds` in LOO already runs
multiple processes.

**Fix.** Run one training at a time; shard within a run using `train.devices`. Do not
launch parallel `hybrax` commands from a shell loop.

## Runtime aborts

### Reactor volume at or below `1e-10`

**Why.** Dilution divides by volume. Rather than producing infinities, the solve aborts.

**Fix.** Usually a volume description that removes more than it adds: check
`validate_volume_consistency`.

### `|ADF| ≤ 1e-12`

Same reasoning, inside the pseudobatch machinery.

## Import-time surprises

### `AttributeError: module 'hybrax.format' has no attribute 'inspect'`

**Why.** `hxf.inspect` is not a module handle. It appears to work only after something else
has pulled the submodule in, which makes it worse than a consistent failure.

**Fix.** Use `hxf.plot_process(...)` on the root. Same for `hxf.simulation`. And save/load
are on `hxf.serialization`, not the root.

## See also

- [Silent failures](silent_failures.md): no exception, wrong answer.
- [Limits and gotchas](../format/limits_and_gotchas.md): what is simply not implemented.
