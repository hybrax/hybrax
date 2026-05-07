# Preliminary spec and roadmap

## Overview

There will be 3-6 packages:
- `bp-form`:
    - contains:
        - data classes with hierarchical ontology to fully describe processes, variables, etc.
        - contains metadata to describe the process, measured variables, splines (optional), etc.
        - I/O: (de)serialization to/from multiple formats
        - simple / non-optimized implementation of process simulation / model evaluation
        - calculate ADF (from V splines or raw trace) & pseudobatch transform from parsed data
    - will be used by all other packages for parsing data and process APIs
- `bp-bench`:
    - "database" of case studies that were already transformed into `bp-form` format
- `bp-prep`:
    - web app for pre-processing raw experimental data; outputs `bp-form`-compliant files
- `bp-train`:
    - utilities for training hybrid models on process data in `bp-form` format
    - integration routines optimized for training (compile once, padded arrays, warm-start, etc.)
    - data augmentation
    - LOO-CV, checkpointing, hyperparameter sweep, model ensemble utils
- `bp-sim`:
    - simulate data and save in `bp-form` format
    - handles discontinuities etc.
    - user can select feed profile (continuous based on function or bolus with schedule) and sampling schedule 
    - parameters determined by user-defined functions
    - simple DoE utils (Latin hypercube sampling, etc.)
- `bp-opt`:
    - once a model is trained, we need to be able to optimize it

## APIs
- interface with process data:      part of `bp-form`
- interface with train utils:       part of `bp-train`

## User story -- researcher implements a BP model
- user uses their own data (import to `bp-form` format first; e.g. with `bp-prep`) or from `bp-bench` database
- user only defines `HybMod`(representing the ODE RHS in an `eqx.Module`)
- `bp-train` has `Trainer` class with two `HybMod` PyTrees as attributes: once just trainable params, once Returns
    - `Trainer` prepares optimizer, handles checkpointing, etc.
    - the `Trainer` combines the PyTrees inside the jit boundary (i.e. inside `batched_loss()`) for predictions

## Open questions
- Where should data augmentation live?
- Mandatory metadata today: case-study fields `case_id`, `organism`, and `citation`; process fields `metadata.name` and `metadata.process_type`.
- Non-mandatory metadata today: top-level dataset metadata (`name`, `version`, `description`, etc.); process `notes`.


## Example snippets

```python
diffrax.diffeqsolve(
    terms = ode_wrapper(hybmod),
    ...
)

def ode_wrapper(hybmod, *args, **kwargs):
    q, f_pred = hybmod(*args, **kwargs)
    return self.rhs(q, f_pred)


class Hybmod (eqx.Module):
    model: eqx.Module = CheckField()  # recursively checks each leaf in `model` if trainable
    mu_max: Array[Float, "1"] = TrainableField()
    self.rhs = StaticField()  # should not be changed by user

    def __init__(self, RhsODE, **kwargs)

    def __call__(self, t, c, u):

    q, f_pred = <austoben wie man q berechnet>


    return self.rhs(q, f_pred)

    def partition(self) -> (HybMod, HybMod):
        for every branch check static_field()

        return (<PyTree of model.static>, <PyTree without model.params>)


@jax.value_and_grad
def batched_loss(
    hybmod_params: HybdMod,
    hybmod_static: HybdMod,
):
    hybmod = eqx.combine(hybmod_params, hybmod_static)
    
    pass


# <loss calc> 
# <loo cv>



class RhsOde(q, f):
    intracellular_product: bool


    def __call__()
        total_feed = f + u_f
        X = Xr - P
        dc_dt = q * Xr - f/V 
        .
        .
        .
        return dc_dt

    self.q_in_size
    self.f_pred_in_size
    self.out_size
    self.ctrl = <spline generator>




class RhsOde(eqx.Module):
    """JAX/Equinox module implementing the generalized fed-batch ODE RHS.

    Created by :func:`get_rhs_ode`; do not instantiate directly.

    The state vector is ``c = [reactor_components..., pv_states..., V]`` where
    the last element is reactor volume. Biomass is always index 0 in the
    reactor-component block.

    Attributes
    ----------
    c_size : int
        ``n_non_volume_states + 1``.
    q_size : int
        ``n_reactor_states`` — number of specific rates (reactor block only).
    u_flow_size : int
        Number of continuous controlled flow streams.
    f_modeled_size : int
        Number of continuous uncontrolled (modeled) flow streams.
    output_size : int
        Same as :attr:`c_size`.
    reactor_component_state_names : tuple[str, ...]
        Ordering of reactor-component states in *c* and *q*. Biomass is
        always first.
    process_variable_state_names : tuple[str, ...]
        Ordering of process-variable states in *c*.
    flow_names : tuple[str, ...]
        Ordering of continuous controlled flow streams in *u_flow*.
    modeled_flow_names : tuple[str, ...]
        Ordering of continuous uncontrolled (modeled) flow streams in
        *f_modeled*.
    biomass_idx : int
        Index of ``"biomass"`` in reactor-component ordering (always 0).
    Cin : jnp.ndarray, shape (n_flows, n_reactor_states)
        Feed composition matrix for controlled flows: ``Cin[k, i]`` is the
        concentration of species *i* in controlled feed stream *k*.
    Cin_modeled : jnp.ndarray, shape (n_modeled_flows, n_reactor_states)
        Feed composition matrix for modeled (uncontrolled) flows.

    Notes
    -----
    JIT usage::

        import equinox as eqx
        mb    = get_rhs_ode(process)
        dc_dt = eqx.filter_jit(mb)(c, q, u_flow)
        # With modeled flows:
        dc_dt = eqx.filter_jit(mb)(c, q, u_flow, f_modeled)
    """
```
