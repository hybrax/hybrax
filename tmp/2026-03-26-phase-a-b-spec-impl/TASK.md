## Task

Implement phases A and B of the V1 spec:

- Phase A: raw artifact loading through `bp_format` deserialization
- Phase B: preparation pipeline that produces `prepared.json`

## Strategy

Start by verifying the mandated runtime environment in `AGENTS.md` because
phase A depends directly on `bp_format`.

## Status

Environment issue was resolved by the user. `bp_format`, `equinox`, and `jax`
now import successfully inside `micromamba run -n jax2 python`.

## Plan

1. Implement phase A raw loading against `bp_format.serialization.load_process_collection_json`.
2. Implement phase B preparation with:
   - first-pass `bp_format.validate_process(...)`,
   - optional `custom.py` hooks,
   - default `V_sample_acc` construction,
   - deterministic control ordering,
   - dense linearized control payload generation,
   - global padding and metadata persistence.
3. Add focused tests around loading, preparation, metadata, and `V_sample_acc`.
4. Run the relevant test subset.

## Outcome

Completed:

- `bp_train.prepare.load_raw_collection(...)` for phase A
- `bp_train.prepare.prepare_artifact(...)` for phase B
- custom hook loading from `custom.py`
- first-pass validation report capture with optional strict mode
- default `V_sample_acc` construction from upstream `SampleVolumeChange` data
- deterministic control ordering plus optional explicit `control_order`
- dense control payload generation with adaptive midpoint refinement
- global padding and persistence under `metadata["bp_train"]`
- focused tests for raw loading, prepared metadata writing, and custom control
  ordering

Relevant test command:

```bash
micromamba run -n jax2 python -m pytest tests/test_prepare.py -q
```

Result:

- `3 passed`
