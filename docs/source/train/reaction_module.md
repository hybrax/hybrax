# The reaction module

> The object that predicts biological rates inside the ODE solve: the half of the model
> that is actually yours. On real data, the default MLP is rarely what you want.

## What it is

An `eqx.Module` with one required method:

```python
def __call__(self, t: jax.Array, inputs: ReactionInputs) -> ReactionOutputs
```

It is called *inside* the solve, at every step the solver takes. It reads the current
state and controls, and returns the rates. Everything else (the mass balance, the
dilution, the events) is hybrax.format's.

```python
from hybrax.train import (
    UserReactionModule, ReactionInputs, ReactionOutputs, trainable_field,
)

class MyReactionModule(UserReactionModule):
    mlp: eqx.nn.MLP = trainable_field()

    def __init__(self, *, key, **scale_kwargs):
        super().__init__(**scale_kwargs)       # always; the base owns the scales
        self.mlp = eqx.nn.MLP(
            in_size=self.n_modeled_RMCs,
            out_size=self.n_modeled_BiologicalOde_rates,
            width_size=32, depth=2, key=key,
        )

    def __call__(self, t, inputs):
        del t
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=self.mlp(inputs.SCL_modeled_RMCs),
            SCL_modeled_Inflows_rates=jnp.zeros(self.n_modeled_Inflows),
            SCL_modeled_Outflows_rates=jnp.zeros(self.n_modeled_Outflows),
        )
```

A complete, runnable version is in [Tutorial 4](../tutorials/04_your_first_custom_py.md).

## The hook

**Fires:** at training setup, after `estimate_all_scales`.
**Signature:** `(*, target_names, process_names, config, seed, training_parent_collection, **scale_kwargs) -> UserReactionModule`
**Default:** `DefaultReactionModule`, a 2-layer MLP.
**Type-checked:** returning something that is not a `UserReactionModule` raises `TypeError`.

```python
def build_reaction_module(*, seed, **kwargs):
    scale_kwargs = {k: v for k, v in kwargs.items() if k.startswith("SCALE_")}
    return MyReactionModule(key=jax.random.key(seed), **scale_kwargs)
```

`scale_kwargs` carries the `SCALE_*` axes produced by
[`estimate_all_scales`](scaling.md). Forward them to `super().__init__`: the reaction
module is the **single source of truth** for every scale in hybrax.train, and the wrapper,
trainer and loss module all read them off it.

## What you get in `ReactionInputs`

Everything is in SCL space. Stop guessing at shapes, ask:

```python
hxt.print_reaction_schema(wrapper)
```

which prints each axis with its shape *and its biological names*, so you know that index
0 of `SCL_modeled_RMCs` is biomass. There is a worked example in
[Tutorial 4](../tutorials/04_your_first_custom_py.md#check-what-is-actually-being-trained).

| Field | What it is |
|---|---|
| `SCL_modeled_RMCs` | The modeled concentrations: usually your main features. |
| `SCL_modeled_PVs` | Modeled process variables. |
| `SCL_modeled_V` | Reactor volume (scalar). |
| `SCL_controlled_PVs` | Controlled process variables at this `t`: pH, temperature, DO. Real inputs to the biology. |
| `SCL_controlled_Inflows_rates` | Current flow rate of each controlled feed. |
| `SCL_controlled_Inflows_cumulative` | Cumulative volume delivered so far. |
| `SCL_controlled_Inflows_Cin` | Feed composition matrix (feeds × species). |
| `SCL_modeled_Inflows_*` | The same, for feeds the model itself predicts. |

`t` is passed separately. Most models ignore it: an explicit time dependence is a model
that knows what hour it is, which is rarely what you mean.

## What you must return

```python
ReactionOutputs(
    SCL_modeled_BiologicalOde_rates=...,   # (n_modeled_BiologicalOde_rates,)
    SCL_modeled_Inflows_rates=...,            # (n_modeled_Inflows,)
    SCL_modeled_Outflows_rates=...,           # (n_modeled_Outflows,)
)
```

:::{admonition} All three fields are required, even when empty
:class: warning
There is no default for `SCL_modeled_Inflows_rates` or `SCL_modeled_Outflows_rates`. A
process with no modeled feeds or outflows still needs `jnp.zeros(0)` for each. Omitting
either is a `TypeError` at the first solve: not at import, so it surfaces a few seconds
into a run.
:::

The rate vector is **flat and positional**, aligned with `rhs_ode.name_modeled_rates`: 
not a dict, not a `(q, r)` tuple. Its order is the insertion order of
`BiologicalOde.rates`.

:::{admonition} Modeled feed rates must be non-negative
:class: warning
Nothing enforces it. A feed pump cannot run backwards, so apply your own
`jax.nn.softplus` and do not rely on the optimizer to discover the constraint.
:::

## The SCL/RAW convention

The one thing to get right.

**Your network reads `SCL_*` inputs, so what it emits is already in SCL space. Return it
directly.**

The wrapper unscales your output by `SCALE_modeled_*_rates` on the way back to physical
units. If you *also* call a `scale_*` helper on the output, the two cancel and your rates
are off by the scale factor: with no error, just a model that will not fit.

When you *do* need physical units (a Monod term with a real `K_s` in g/L, say) unscale
the inputs, compute in RAW, then scale the result back:

```python
def __call__(self, t, inputs):
    del t
    RAW = self.unscale_modeled_RMCs(inputs.SCL_modeled_RMCs)
    S = RAW[1]                                   # glucose, g/L
    mu = self.mu_max * S / (self.K_s + S)        # honest physical units
    RAW_rates = jnp.array([mu, -mu / self.Y_xs, self.alpha * mu])
    return ReactionOutputs(
        SCL_modeled_BiologicalOde_rates=self.scale_modeled_BiologicalOde_rates(RAW_rates),
        SCL_modeled_Inflows_rates=jnp.zeros(0),
        SCL_modeled_Outflows_rates=jnp.zeros(0),
    )
```

Both conventions are correct; what is not correct is mixing them. Ask "what space is this
number in?" at every line. See [Gallery: mechanistic models](../gallery/mechanistic_rates.md)
for the full mechanistic version.

## Trainable and frozen

Parameters are declared with field tags, not a partition function:

```python
mu_max: jax.Array = trainable_field()    # optimized
K_s:    jax.Array = frozen_field()       # held fixed
n_in:   int = eqx.field(static=True)     # not an array at all
```

**Untagged array leaves default to frozen.** This is the rule that catches people: a
field you forgot to tag is silently never optimized. The resolution rule is *first
explicit tag on the path wins*, and it applies to the whole wrapper: including the loss
module, so trainable loss parameters are optimized alongside the model.

Check before committing to a long run:

```python
hxt.print_trainable_structure(wrapper)
```

For "freeze these layers, train those" (e.g. a fixed encoder feeding a trainable head),
split the MLP into separate fields with their own tags rather than one field mixing
both. See [Freezing parameters](../gallery/freezing.md) for a full worked example.

## Passing auxiliary values to the loss

`ReactionOutputs` can carry an `auxiliary` dict, which reappears on `LossInputs`. Use it
when the loss needs an internal quantity the model computed anyway (a latent, an
intermediate flux) rather than recomputing it.

## Stateful (latent-ODE) modules

A module with `n_latent > 0` adds its own integrated latent state. That is a genuinely
different model class, and it requires explicit opt-in:

```json
{ "train": { "allow_stateful_models": true } }
```

Without it you get a clear `ValueError`. See [Gallery](../gallery/index.md).

## Gotchas

- **Forgetting `super().__init__(**scale_kwargs)`** leaves the module without scales.
- **A misspelled `build_reaction_module`** silently uses the default MLP.
- **`eqx.Module` is frozen.** Use `eqx.tree_at` for functional updates; you cannot assign
  to fields after construction.
- **The module runs inside `jit` and `grad`.** No Python branching on traced values, no
  `.item()`, no printing.

## See also

- [Scaling](scaling.md), where `scale_kwargs` comes from. Read it next.
- [The loss module](loss_module.md): the other half.
- [Gallery: mechanistic models](../gallery/mechanistic_rates.md): real kinetics.
- [API reference](../autoapi/hybrax/train/model_api/index).
