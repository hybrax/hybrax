# Spec: Perfusion (with bleed) and Evaporation Support

> Status: **design only**. Not implemented.

## 1. Motivation

Today the auto-RHS at [`_apply_feed_dilution`](../bp_format/mechanistic.py) hardcodes well-mixed CSTR semantics:

```
dilution = -(total_in + total_out) / V * c_RMCs   # uniform across all RMCs
```

That makes two real bioprocess regimes inexpressible:

- **Perfusion with bleed** — biomass is retained on a membrane while supernatant is removed. Today every species (biomass included) washes out at the same rate.
- **Evaporation** — water leaves, all solutes stay; their concentrations rise. Today every solute is *diluted* through the loss SVC, not concentrated. Example `15_biogas` silently accepts this error on its `volume_loss` channel.

`biological_ode` cannot fix either case from the user side: SVC/FVC flow rates are not in its symbol namespace, so any compensation term has no flow value to multiply.

Both regimes reduce to one missing primitive: **per-component retention on each `SampleVolumeChange`**.

## 2. Design (minimal)

Add **one optional field** to `SampleVolumeChange`:

```python
@dataclass
class SampleVolumeChange(VolumeChange):
    component_retention: Dict[str, float] = field(default_factory=dict)
    # σ ∈ [0, 1] per RMC name. Missing key ⇒ σ=0 (current behavior).
    # σ=0 → species leaves with the supernatant (sampling/harvest)
    # σ=1 → species fully retained (cells in pure perfusion; solutes in evaporation)
```

`FeedVolumeChange` is untouched. `Volume`, `BiologicalOde`, `ProcessOrdering`, and the public RhsOde call signature are untouched.

## 3. Mathematical change

One line in [`_apply_feed_dilution`](../bp_format/mechanistic.py#L169-L211).

Before:

```python
dilution = -(total_in + total_out) / V * c_RMCs
```

After:

```python
# eff_out_per_rmc[i] = sum over SVCs of (1 - σ_{svc,i}) * |q_svc|
eff_out_per_rmc = (
    jnp.sum((1.0 - retention_controlled_SVCs) * (-u_controlled_SVCs)[:, None], axis=0)
  + jnp.sum((1.0 - retention_modeled_SVCs)    * (-f_modeled_SVCs)[:, None],    axis=0)
)  # shape (n_RMCs,)
dilution = -(total_in * c_RMCs + eff_out_per_rmc * c_RMCs) / V
```

`dV/dt = total_in - total_out` is unchanged: liquid volume drops by the full SVC magnitude regardless of solute retention. The `addition` term (from FVC `Cin`) is unchanged.

Sanity checks:

- All SVCs have empty `component_retention` ⇒ `(1-σ)=1` everywhere ⇒ `eff_out_per_rmc = total_out * 1` ⇒ identical to current numerics.
- Pure perfusion, σ_biomass=0.95, σ_other=0 ⇒ biomass washout reduced 20×, other species washout unchanged.
- Pure evaporation, σ=1 for every non-water RMC ⇒ those species' outflow term is 0 ⇒ mass conserved while V drops ⇒ concentrations rise as expected.

## 4. Files modified

1. [`bp-format/bp_format/dataclasses.py`](../bp_format/dataclasses.py) — add `component_retention` field on `SampleVolumeChange`.
2. [`bp-format/bp_format/mechanistic.py`](../bp_format/mechanistic.py) —
   - `RhsOde` gains two static fields: `retention_controlled_SVCs`, `retention_modeled_SVCs` (shape `(n_svc, n_RMCs)`), built symmetrically to `Cin_*_FVCs`.
   - `_apply_feed_dilution` takes them and applies the per-species formula above.
   - `build_rhs_ode` constructs the matrices from `process.volume.volume_changes[svc_name].component_retention`. Missing entries default to 0.0.
3. [`bp-format/bp_format/serialization.py`](../bp_format/serialization.py) — extend `_volume_change_to_dict` (line ~496) and `_dict_to_volume_change` (line ~886) to round-trip the dict. Default to `{}` when absent so existing JSON loads unchanged.
4. [`bp-format/bp_format/validate.py`](../bp_format/validate.py) — single new check inside `validate_process`: for each SVC, every key in `component_retention` must be a declared RMC name and every value must be in `[0, 1]`.

## 5. Validation rules

- `0.0 ≤ σ ≤ 1.0` (inclusive). Reject otherwise.
- Keys must be subset of `process.reactor_medium.components`. Unknown keys are a hard error (typo protection — silent default would mask mistakes).
- Missing keys are allowed and mean σ=0.
- No constraint that retention values sum to anything; they are independent per-species.

## 6. Verification

End-to-end checks:

- Existing example regression: rerun all `examples/*` with no `component_retention` set; trajectories must be bit-identical to current behavior (unit test).
- New unit test: synthetic perfusion. Constant V (FVC = SVC), constant `q_perf`, σ_biomass=0.95, no biology. Assert biomass decays at rate `0.05 * q_perf / V` (analytical).
- New unit test: synthetic evaporation. Single non-water RMC with σ=1, no biology, linear evaporative SVC. Assert `c * V` is conserved across the integration.
- Round-trip: serialize a process with `component_retention={"biomass": 0.95}`, deserialize, compare equal.
- Validator: σ=1.5 rejected; key not in RMCs rejected.

## 7. Out of scope

- Discrete `apply_events` retention (discrete bleed pulses). Continuous-only per current request.
- Updating `15_biogas` data to set σ=1 on `volume_loss` — separate data PR; impact on existing fits should be reviewed.
- Exposing flow values in the `biological_ode` symbol namespace. Not needed once retention is native.
- New process_type string ("perfusion"). `process_type` is documentation; the SVC retention field is what makes it perfusion.
- A `FeedVolumeChange.component_retention` (no physical meaning — feeds add, never retain).
