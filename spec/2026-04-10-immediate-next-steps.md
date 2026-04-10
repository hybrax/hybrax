# Next steps: support process-variable (PV) states in HybridOdeWrapper

## Background

bpbench's `RhsOde` now distinguishes two kinds of non-volume states:

- **Reactor-component states** (`reactor_component_state_names`): biomass,
  substrates, products. These participate in the reaction term `q * X_active`
  and in feed/dilution dynamics. The MLP predicts specific rates `q` for these.
- **Process-variable states** (`process_variable_state_names`): pH, DOT,
  temperature, etc. These are driven only by the additive physical-rate vector
  `r` (`dc_pv/dt = r_pv`). The MLP does not predict rates for them (`q_size =
  n_reactor_states` only). They do not participate in feed/dilution dynamics.

`HybridOdeWrapper` currently blocks processes with PV states via a
`NotImplementedError`. Kittler 2022 has none, so this is fine for now. This
spec describes what it would take to lift that restriction.

---

## Key design decision: are PV states training targets?

This question splits the implementation into two paths:

**Option A — carried state only (simpler)**
PV states are included in the wrapper's ODE state vector and integrated
forward, but they are not included in the loss. Their initial values come
from the data. The MLP does not predict anything about them; they evolve
purely through `r = 0` (constant, since we pass zeros for `r`). This is only
useful if PV states affect reactor dynamics through the `r` vector — which they
don't with our current zero-r approach. Effectively, PV states would be frozen
at their initial values and irrelevant to training.

**Option B — full training targets (useful)**
PV states are included in `y_meas` alongside reactor components. The loss
penalises deviations of the integrated PV trajectory from measured data. This
requires the MLP to produce additive rates `r` for PV states in addition to
specific rates `q` for reactor components, or alternatively for PV states to
be modelled as a separate learned component.

**Recommendation: start with Option A** (carried state, not targets). It
unlocks the ability to run processes that happen to have PV state fields
without crashing, even if those states don't contribute to the loss. Option B
can follow once the state-vector plumbing is settled.

---

## Changes required (Option A)

### `bp_train/wrapper.py`

**`from_process` (constructor)**

```
n_reactor = len(rhs_ode.reactor_component_state_names)  # was n_species
n_pv     = len(rhs_ode.process_variable_state_names)   # new
n_modeled = rhs_ode.f_modeled_size
full_state_size = n_reactor + n_pv + 1 + n_modeled     # unchanged formula, new meaning
```

- Remove the `NotImplementedError` guard.
- `q_scale` stays size `n_reactor` (correct already — `q_size = n_reactor`).
- `state_scale` must now have length `n_reactor + n_pv + 1 + n_modeled`. The
  PV block sits between the reactor-component block and `V_cont`.
- Default `target_state_indices`: exclude the V_cont slot at index
  `n_reactor + n_pv`; include reactor component slots `0..n_reactor-1` and
  B_modeled_cum slots `n_reactor+n_pv+1..end`. PV slots are excluded from the
  loss in Option A.

**Fields on `HybridOdeWrapper`**

The existing `species_names` field currently holds reactor-component names.
With PV states added to the state vector, callers (postprocessing, custom
hooks) need to know where each block lives. Two options:

- Keep `species_names` for reactor components and add `pv_state_names`
  alongside it. Low churn in existing callers.
- Rename `species_names` → `reactor_component_state_names` everywhere for
  consistency with bpbench. More churn but cleaner long-term.

Keeping `species_names` + adding `pv_state_names` is the lower-risk path for
now.

**`__call__` (ODE RHS)**

State layout changes from:

```
y = [c_reactor..., V_cont, B_modeled_cum...] / state_scale
```

to:

```
y = [c_reactor..., c_pv..., V_cont, B_modeled_cum...] / state_scale
```

Index arithmetic updates:

```python
n_reactor = len(self.species_names)          # unchanged name, same value
n_pv      = len(self.pv_state_names)         # new
v_idx     = n_reactor + n_pv                 # was n_reactor

C_species = jnp.clip(Y[:n_reactor], 0.0)
C_pv      = Y[n_reactor:v_idx]              # new slice (may be empty)
V_cont    = jnp.maximum(Y[v_idx], 0.0)

C_rhs = jnp.concatenate([C_species, C_pv, jnp.asarray([V_real], dtype=y.dtype)])
```

`dY_rhs` from `self.rhs_ode(C_rhs, Q, U_flow, F_modeled, r)` now has length
`n_reactor + n_pv + 1`. The concatenation with `F_modeled` is unchanged:

```python
dY_full = jnp.concatenate([dY_rhs, F_modeled])
```

`state_size` check needs updating:

```python
expected_state_size = n_reactor + n_pv + 1 + n_modeled
```

### `bp_train/training_data.py`

In Option A, PV states are not training targets, so `y_meas` is unchanged.
However, the **initial state `y0`** must be populated correctly. Currently y0
is `[c_reactor_at_t0..., V0, 0...0_for_B_modeled]`. With PV states:

```
y0 = [c_reactor_at_t0..., c_pv_at_t0..., V0, 0...0_for_B_modeled]
```

The PV initial values should be read from `process.process_variables[name]`
at `t = t_start`. This requires a small change in `TrainingDataStore.from_collection`
where y0 is assembled.

### `bp_train/harness.py`

The harness reads `n_species` from the reference RhsOde and uses it to build
Cin stacks. With PV states, `n_reactor` (not `n_species`) is the right count
for Cin, but `Cin` is already shaped `(n_flows, n_reactor_states)` by bpbench,
so this is likely already correct. Review the y0 assembly and any direct
indexing into the state vector.

### `examples/01_kittler_2022/custom.py` — `estimate_all_scales`

The function manually constructs `state_scale` as:

```python
state_scale = species_scale + [volume_scale] + b_modeled_cum_scale
```

For PV states this becomes:

```python
state_scale = reactor_scale + pv_scale + [volume_scale] + b_modeled_cum_scale
```

PV scale values can be estimated from `process.process_variables[name]` time
series (max observed absolute value, or 1.0 if the variable is static).

### `bp_train/postprocessing.py`

Currently iterates over `species_names` to produce per-species plots. With a
new `pv_state_names` field, a separate plotting section for PV trajectories
would be added if they are targets (Option B). In Option A they can be
skipped in plotting since they are not loss targets.

---

## Rough size estimate

| File | Lines changed (approx) |
|---|---|
| `bp_train/wrapper.py` | 40-60 |
| `bp_train/training_data.py` | 20-30 |
| `bp_train/harness.py` | 10-20 |
| `examples/01_kittler_2022/custom.py` | 15-20 |
| `bp_train/postprocessing.py` | 5-10 (Option A) / 30-50 (Option B) |
| tests | 20-40 |
| **Total** | **~110-180 (Option A)** |

Option B (PV states as loss targets, MLP produces `r`) adds roughly another
100-150 lines, primarily in the reaction module interface, `training_data.py`,
and the MLP output head in the example custom hook.
