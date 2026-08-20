# `custom.py` at a glance

> Every hook bp-train looks for, what it does, when it fires, and what happens if you
> omit it.

## How discovery works

`custom.py` is a plain Python module. bp-train looks up each hook **by name**, with an
ordinary attribute lookup. There is no registration, no decorator and no base class for
the file itself.

:::{admonition} A misspelled name is silent
:class: danger
`build_reaction_modul` is not an error. It is a hook that does not exist, which means the
default is used and your code never runs. Nothing is logged.

If an edit to `custom.py` appears to have had no effect, check the spelling before
anything else. This is the single most common cause of "my custom module isn't doing
anything".
:::

## The seven hooks

| Hook | Stage | Returns | If omitted |
|---|---|---|---|
| [`transform_process_collection`](prepare.md#hook-transform_process_collection) | prepare | the collection | applies `prepare.process_rename_map` |
| `augment_state_values` | prepare | an array of values | no-op |
| [`estimate_all_scales`](scaling.md#the-hook) | train setup | `EstimatedScales` | **every scale is 1.0** |
| [`build_reaction_module`](reaction_module.md#the-hook) | train setup | `UserReactionModule` | `DefaultReactionModule` (2-layer MLP) |
| [`build_loss_module`](loss_module.md#the-hook) | train setup | `UserLossModule` | `DefaultLossModule` (per-target MSE) |
| [`build_learning_rate`](train.md#hook-build_learning_rate) | train setup | `float` or `optax.Schedule` | constant `train.learning_rate` |
| [`build_optimizer`](train.md#hook-build_optimizer) | train setup | `optax.GradientTransformation` | clip + adam/sgd |

**Setup order:** `build_learning_rate` → `build_optimizer` → `estimate_all_scales` →
`build_reaction_module` → `build_loss_module`.

That ordering is why scales arrive at the reaction module as `**scale_kwargs`: they
already exist by the time it is built.

**Type-checked:** `estimate_all_scales`, `build_reaction_module` and `build_loss_module`
raise `TypeError` if they return the wrong type. The other four do not.

## Signatures

```python
def transform_process_collection(collection, config): ...
    # -> collection.  Must RETURN it; mutating in place and returning None
    #    hands None downstream.

def augment_state_values(*, parent_name, child_name, state_name,
                         times, base_values, augmented_values, config): ...
    # -> ndarray of values for this synthetic child's state.

def estimate_all_scales(runtime_data, target_names, config): ...
    # -> EstimatedScales.  runtime_data is a RuntimeDataContext: collection-free
    #    numeric traces (raw_state_trace, initial_volume, ...) plus
    #    runtime_data.controls_store, always available, no separate argument needed.

def build_reaction_module(*, target_names, process_names, config, seed,
                          runtime_context, **scale_kwargs): ...
    # -> UserReactionModule.  runtime_context wraps the same RuntimeDataContext
    #    plus the resolved EstimatedScales (also unpacked into **scale_kwargs).

def build_loss_module(*, target_names, process_names, config, seed,
                      runtime_context): ...
    # -> UserLossModule

def build_learning_rate(custom_cfg, train_cfg, total_updates): ...
    # -> float | optax.Schedule

def build_optimizer(custom_cfg, train_cfg): ...
    # -> optax.GradientTransformation
```

## Three things that are not `get_hook` hooks

**`get_custom_config(raw_custom, config)`** runs *before* every hook. Whatever it returns
becomes `config.custom` everywhere. Define it to get a validated config object instead of
the permissive default:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class MyConfig:
    hidden_width: int = 32

def get_custom_config(raw_custom, config):
    return MyConfig(**raw_custom)
```

**`CONFIG` / `get_config()`**: a module-level dict, or a function returning one, merged
into the config. `get_config()` wins if both exist.

**`dense_grid_n`** is a *property on your loss module*, not a hook, but it behaves like
an extension point: declaring it makes the trainer populate every `dense_*` field on
`LossInputs`. See [The loss module](loss_module.md#the-dense-grid).

## A minimal complete `custom.py`

The two hooks that matter most, for a batch process with no feeds:

```python
import equinox as eqx, jax, jax.numpy as jnp, numpy as np
from bp_train import (EstimatedScales, ReactionOutputs,
                      UserReactionModule, trainable_field)

class MyModule(UserReactionModule):
    mlp: eqx.nn.MLP = trainable_field()

    def __init__(self, *, key, **scale_kwargs):
        super().__init__(**scale_kwargs)
        self.mlp = eqx.nn.MLP(in_size=self.n_modeled_RMCs,
                              out_size=self.n_modeled_BiologicalOde_rates,
                              width_size=16, depth=2, key=key)

    def __call__(self, t, inputs):
        del t
        return ReactionOutputs(
            SCL_modeled_BiologicalOde_rates=self.mlp(inputs.SCL_modeled_RMCs),
            SCL_modeled_FVCs_rates=jnp.zeros(self.n_modeled_FVCs),
        )

def build_reaction_module(*, seed, **kwargs):
    return MyModule(key=jax.random.key(seed),
                    **{k: v for k, v in kwargs.items() if k.startswith("SCALE_")})

def estimate_all_scales(runtime_data, target_names, config):
    ...   # see the Scaling page: this one is worth writing properly
```

A runnable version, with the scale hook filled in, is in
[Tutorial 4](../tutorials/04_your_first_custom_py.md).

## Gotchas

- **`custom.py` is copied into the run directory** and is part of the model. Reconstruction
  re-runs your hooks, so a run directory without it cannot be loaded.
- **A missing `custom_py` *path*** is a `FileNotFoundError`; a missing *hook inside* it is
  silent. Two very different failure modes.

## See also

- [Configuration](config.md): how `custom_py` and `custom` are wired in.
- [The reaction module](reaction_module.md) · [Scaling](scaling.md) ·
  [The loss module](loss_module.md): the hooks that do the work.
- [Silent failures](../troubleshooting/silent_failures.md).
