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

# The Bioprocess ODE

> What bp-format assembles from your description, how to read it, and how to replace
> the biological half with your own equations. Read the first two sections even if the
> auto-generated ODE is already right for your process.

## The split

```
d(state)/dt  =  BIOLOGICAL          ← yours: BiologicalOde, in terms of named rates
              + PHYSICAL            ← generated: feed inflow, dilution, sample outflow, dV/dt
```

Everything bp-format does here is in service of that line. You never write the physical
half, and you cannot get it wrong by forgetting a term, but you *can* get it wrong by
describing the volume badly, which is why [Volume, feeds and
events](volume_feeds_events.md) comes first.

## Seeing it

```{code-cell} ipython3
import bp_format as bp

cs = bp.serialization.load_process_collection("../_data/out/demo_fedbatch/data.json")
process = cs.processes["fedbatch_1"]

bp.print_rhs_ode(process)
```

Read it in two halves. Every `q_*` is something a model must supply. Everything else is
already written.

## The default biology

If you do not provide `biological_ode`, bp-format generates one when the `BioProcess` is
constructed:

```{code-cell} ipython3
print("rates      :", list(process.biological_ode.rates))
print("derivatives:", process.biological_ode.derivatives)
```

The rule is:

- every reactor medium component `c` gets a specific rate `q_c`, with
  `d(c)/dt = q_c * biomass`;
- every **dynamic** (uncontrolled, non-static) process variable `p` gets a volumetric
  rate `r_p`, with `d(p)/dt = r_p`;
- rate order is biomass-first reactor components, then dynamic process variables.

:::{admonition} Auto-generation requires a component named `biomass`
:class: warning
Every generated rate is *specific* (per unit biomass) so there has to be a biomass to
be specific to. Without one (case-insensitive match), constructing the `BioProcess`
raises immediately, and the message tells you to supply your own `biological_ode`.
:::

## Writing your own

`BiologicalOde` has three fields, all plain dictionaries of strings.

```{code-cell} ipython3
process.biological_ode = bp.BiologicalOde(
    algebraic={"X_active": "biomass - product"},
    rates={"q_biomass": (None, None), "q_glucose": (None, None),
           "q_product": (None, None)},
    derivatives={
        "biomass": "q_biomass * X_active + q_product * X_active",
        "glucose": "q_glucose * X_active",
        "product": "q_product * X_active",
    },
)
ok, message = bp.validate_biological_ode(process)     # validates the whole process
print(ok, "|", message)
```

| Field | Meaning |
|---|---|
| `algebraic` | `name -> expression`. Recomputed every RHS call, never integrated. Must be acyclic. |
| `rates` | `name -> (lower, upper)`. Declares the rate vector: its length *is* the rate dimension. Bounds are metadata. |
| `derivatives` | `state -> expression` for the **biological** contribution only. |

That example is the standard intracellular-product pattern: measured biomass includes the
product accumulating inside the cells, so growth is driven by the *active* fraction, and
the measured biomass derivative picks up both growth and product accumulation. There is
no `is_intracellular` flag: you write what you mean.

:::{admonition} Every dynamic state needs a derivative entry
:class: warning
Omitting one is rejected. If a state genuinely has no biological dynamics, write `"0"`
explicitly. Silence and "zero" must not look the same.
:::

Expressions are parsed with sympy, so ordinary arithmetic works, and names must refer to
states, algebraic quantities, or declared rates. Two or more states added together
*bare*, with no rate scaling either one, must share a unit: `biomass - product` with
`g/L` against `mg/L` is rejected. A state scaled by its own declared rate is exempt
from that check: `-q_a * a - r_b * b` is fine even when `a` and `b` differ, since each
rate is trusted to carry whatever unit bridges its own term, the same trust already
extended to a lone `rate * state` product. See [Gallery: glutamine
decay](../gallery/glutamine_decay.md) for a worked example: one rate feeding two
derivatives across a `g/L` state and a `mol/L` state.

## Layout: `ProcessOrdering`

Everything downstream needs to agree on which array index is which species.
`ProcessOrdering` is the single place that decides.

```{code-cell} ipython3
from bp_format.mechanistic import get_process_ordering

ordering = get_process_ordering(process)
print("modeled RMCs     :", ordering.name_modeled_RMCs)
print("modeled rates    :", ordering.name_modeled_rates)
print("controlled Inflows :", ordering.name_controlled_Inflows)
print("controlled Outflows:", ordering.name_controlled_Outflows)
print("controlled PVs   :", ordering.name_controlled_PVs)
```

```
state   c = [ modeled RMCs | modeled PVs | V ]
control u = [ controlled Inflows | controlled Outflows | controlled PVs ]
```

The first `len(Inflows) + len(Outflows)` entries of `u` are **flow rates** (spline derivatives);
the rest are direct values.

Ordering rules: rates keep your insertion order, so a rate vector you build matches the
order you declared. Algebraic names are topologically sorted by dependency. Everything
else is alphabetical.

bp-train consumes this object and never re-derives layout. If you are writing anything
that indexes into a state vector, get the names from here rather than assuming.

## Building the callable

```{code-cell} ipython3
from bp_format.mechanistic import build_rhs_ode

rhs = build_rhs_ode(process)
print(type(rhs).__name__)
print("rate names:", rhs.name_modeled_rates)
print("feed composition matrix:", rhs.Cin_controlled_Inflows.shape,
      "(feeds x species)")
```

`RhsOde` is a JAX-compatible callable: given time, state, controls and a rate vector it
returns `d(state)/dt`. It does **not** integrate: bp-format has no solver. Handing it to
a solver is [bp-train](../train/index.md)'s job.

Related helpers, for when you are building your own integrator:
`build_algebraic_func` (evaluate the algebraic block alone), `extract_discrete_events`
(the event times and jumps), `build_state_splines`.

## Gotchas

- **Names must be unique across groups.** Using the same name for a state and a rate
  raises when the ordering is built.
- **Cyclic `algebraic` entries** are rejected.
- **The rate vector is flat and positional.** One `jnp.ndarray` aligned with
  `name_modeled_rates`: not a dict, not a `(q, r)` tuple.
- **Reactor volume must stay above `1e-10`.** Dilution divides by `V`, so the solve
  aborts loudly rather than producing infinities.
- **Bounds in `rates` are metadata.** `RhsOde` will not clip a rate to them.

## See also

- [Volume, feeds and events](volume_feeds_events.md), where the physical half comes from.
- [The reaction module](../train/reaction_module.md): what supplies the rates.
- [Gallery: mechanistic models](../gallery/mechanistic_rates.md): real kinetics in place
  of a bare network.
- [Gallery: glutamine decay](../gallery/glutamine_decay.md): one declared rate feeding
  two coupled derivatives at once.
- [API reference](../autoapi/bp_format/mechanistic/index).
