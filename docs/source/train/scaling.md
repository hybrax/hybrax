# Scaling

> One number per semantic axis that makes every quantity in the solve roughly 1, which
> is the difference between a model that trains and one that thrashes. This hook is
> optional and silent, and it is the most consequential thing you can add.

## The problem

A bioprocess state vector might hold biomass at 5 g/L, glucose at 20 g/L, product at
0.4 g/L, volume at 1 L and cumulative feed at 0.001 L. Spread: four orders of magnitude.

Gradients through an ODE solve are dominated by whichever axis is largest. The solver's
own error control (`rtol`, `atol`) also applies across a shared state vector, so an
absolute tolerance appropriate for glucose is meaningless for a trace species. Everything
downstream inherits the conditioning of the worst axis.

So bp-train **integrates in scaled (SCL) space**, where every axis is O(1), and converts
to physical (RAW) units only where the chemistry needs them.

## Why one linear factor per axis

Because scaling is linear, the *same* factor converts a value and its time derivative:

```
d(x/k)/dt  =  (dx/dt)/k
```

which means one number per axis handles both states and rates, and the `scale_*` /
`unscale_*` helper pairs work for both. That is the whole design.

## The hook

**Fires:** at training setup, before the reaction module is built.
**Signature:** `(runtime_data, target_names, config) -> EstimatedScales`
**Default:** none, and that is the problem below.
**Type-checked:** returning something other than `EstimatedScales` raises `TypeError`.

:::{admonition} No hook means every scale is 1.0
:class: danger

Omitting `estimate_all_scales` does not raise, does not warn, and does not disable
anything. It leaves SCL identical to RAW: the exact ill-conditioning the architecture
exists to prevent.

Training still runs. The loss still goes down. It is just much worse than it should be,
and nothing tells you. [Tutorial 4](../tutorials/04_your_first_custom_py.md#did-it-help)
measures the gap on a small, well-behaved dataset: the initial loss alone differs by more
than two orders of magnitude, before a single optimizer step.
:::

## Writing one

The whole job is: for each axis, what is a characteristic magnitude?

**State axes** are easy: how big does this species get, anywhere in the data. The hook
receives a `RuntimeDataContext`, collection-free numeric traces rather than raw
bp-format objects: `raw_state_trace(process_index, name)` returns that state's exact
`(times, values)`.

```python
def max_abs_state(name):
    best = 0.0
    for i in range(len(runtime_data.process_order)):
        _, values = runtime_data.raw_state_trace(i, name)
        if values.size:
            best = max(best, float(np.max(np.abs(values))))
    return max(best, 1e-6)                      # floor: never divide by zero

rmc_scale = {name: max_abs_state(name) for name in rhs.name_modeled_RMCs}
```

**Rate axes** need one step of thought, and this is where a first attempt usually goes
wrong. Your module emits an O(1) number, which is multiplied by the rate scale to become
a physical rate. You want the resulting *state change over the run* to be about the size
of the state.

For a specific rate, `d(c)/dt = q_c · biomass`, so:

```
scale(q_c)  ~  scale(c) / (scale(biomass) · duration)
```

For a volumetric rate, `d(p)/dt = r_p` directly, so:

```
scale(r_p)  ~  scale(p) / duration
```

Get this wrong in the optimistic direction and the concentrations blow up on the very
first solve, before anything is learned.

**Controlled axes** come from the controls store, always reachable as
`runtime_data.controls_store`, sampled over the run:

```python
def estimate_all_scales(runtime_data, target_names, config):
    controls_store = runtime_data.controls_store
    ...
```

A complete, runnable minimal version is in
[Tutorial 4](../tutorials/04_your_first_custom_py.md); [Fed-batch](../gallery/fed_batch.md)
shows the controlled-axis case for real.

## The axes

`EstimatedScales` has one field per semantic axis: modeled RMCs, modeled PVs, cumulative
inflow volume, modeled and controlled feed cumulative volumes and rates, feed composition
matrices, controlled PVs, and the biological rate vector. `SCALE_modeled_PVs` defaults to
empty; the rest are required.

Exact field names and shapes are in the
[API reference](../autoapi/bp_train/model_api/index), and `print_reaction_schema` will
show you the shapes for *your* dataset, which is faster than reading either.

For a process with no feeds and no process variables, most axes are `jnp.zeros(0)`. That
is normal, not a sign you did something wrong.

## Linear or affine

By default a bare array becomes a `LinearScaler`: plain division, bit-identical to doing
it by hand. Returning an `AffineScaler(scale, offset)` for one axis opts that axis into
affine scaling, useful when a quantity varies over a narrow band far from zero
(temperature around 37 °C, pH around 7).

:::{admonition} Offsets are rejected on rate axes
:class: warning
Rate scaling must be offset-free, because the value/derivative equivalence above only
holds for a pure multiplication. Supplying an `AffineScaler` with a non-zero offset for
`SCALE_modeled_BiologicalOde_rates`, `SCALE_controlled_FVCs_rates` or
`SCALE_modeled_FVCs_rates` is rejected.
:::

## How do I know it worked?

Three signals, in order of usefulness:

1. **Initial loss.** Before any training, a well-scaled model should already be within an
   order of magnitude or two of the data. A first-epoch loss of 10⁴ means the scales are
   wrong, not that the problem is hard.
2. **Gradient norm.** `grad_norm_curve.png` plots the **raw** norm, before clipping. If it
   sits permanently pinned at `grad_clip_norm`, your effective step size is not what the
   learning rate says it is.
3. **The rate panels.** Rates that saturate immediately or oscillate wildly usually mean a
   rate scale that is too large.

## Gotchas

- **Order matters at setup:** `build_learning_rate` → `build_optimizer` →
  `estimate_all_scales` → `build_reaction_module` → `build_loss_module`. Scales exist
  before the module, which is why they arrive as `scale_kwargs`.
- **Scales are frozen, never trained.** They are a property of the dataset.
- **Do not duplicate scales onto your inputs.** They live on the reaction module; read
  them via `inputs.reaction_module.SCALE_*` or its helpers.
- **`forward_from_collection` re-runs this hook** on whatever collection you hand it;
  `model_predict` does not. Two paths, two behaviours: see
  [Silent failures](../troubleshooting/silent_failures.md).

## See also

- [Tutorial 4](../tutorials/04_your_first_custom_py.md): a working hook, with the
  before/after numbers.
- [The reaction module](reaction_module.md): the consumer of these scales.
- [Silent failures](../troubleshooting/silent_failures.md).
