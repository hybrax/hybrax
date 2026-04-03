# Validation

Source: `bpbench/validate.py`

## Purpose

Provides data integrity checks that catch common errors in bioprocess data before they propagate into modeling. All validation functions return `(bool, str)` tuples for composability, enabling callers to collect all issues in one pass rather than failing on the first error.

## Design Rationale

- **Why validate early?** Bioprocess data comes from diverse sources with inconsistent conventions. Wrong signs on volume changes, missing biomass components, or mismatched array lengths can cause silent numerical errors during ODE integration. Catching these at data loading time saves hours of debugging.
- **Why `(bool, str)` return type?** Allows both programmatic checking (`if not ok: ...`) and human-readable reporting. `validate_process()` aggregates all individual checks into a single pass. This is preferred over raising exceptions because it provides a comprehensive report rather than stopping at the first issue.
- **Why cross-process consistency checks?** All processes in a case study should share the same variable structure for fair benchmarking. `validate_case_study()` checks this.

## Public API

### Individual Validators

Each returns `(bool, str)` where `True` means valid and the string is either empty or describes the issue.

#### `validate_timeseries_shape(ts, name="")`
Checks that a TimeSeries has:
- 1D arrays for both `times` and `values`
- Matching lengths
- Strictly increasing time points

#### `validate_volume_change_sign(volume_change)`
Checks sign conventions:
- `FeedVolumeChange`: all values >= 0 (inflow)
- `SampleVolumeChange`: all values <= 0 (outflow)

#### `validate_volume_change_states(process)`
For every positive volume change (feed), checks that the feed medium defines concentrations for all dynamic state variables in the reactor medium. This ensures the ODE RHS can compute dilution terms for every species.

#### `validate_biomass_in_reactor_medium(process)`
Checks that the reactor medium contains a component named `"biomass"` (case-insensitive). Biomass is required at index 0 in the ODE state vector.

#### `validate_measurement_sampling_alignment(process)`
Checks that measurement times for reactor medium components do not coincide with sampling events. Measurements taken exactly at a sampling time may have corrupted concentrations due to the volume change.

#### `validate_intracellular_units(process)`
Checks that components marked as `is_intracellular=True` have units consistent with the biomass component. Intracellular products are subtracted from measured biomass to compute active biomass, so units must match.

#### `validate_volume_consistency(process)`
Checks that the volume balance is internally consistent: initial volume plus cumulative volume changes should match the expected final volume (within a tolerance).

### Aggregate Validators

#### `validate_process(process) -> (bool, List[str])`
Runs all individual validators on a single process. Returns a list of all error messages (empty if valid).

#### `validate_case_study(case_study) -> (bool, Dict[str, List[str]])`
Runs `validate_process()` on each process in the case study, plus cross-process consistency checks:
- All processes should define the same set of reactor medium components.
- All processes should define the same set of process variables.

Returns a dict mapping process IDs to their error message lists.

## Examples

### Validating a Single Process

```python
import bpbench as bp

dataset = bp.serialization.load_dataset("data.json")
process = dataset.case_studies["kittler_2022"].processes["batch_001"]

is_valid, messages = bp.validate_process(process)
if not is_valid:
    print("Validation errors:")
    for msg in messages:
        print(f"  - {msg}")
else:
    print("Process is valid.")
```

### Validating an Entire Case Study

```python
import bpbench as bp

dataset = bp.serialization.load_dataset("data.json")
case_study = dataset.case_studies["kittler_2022"]

is_valid, report = bp.validate_case_study(case_study)
if not is_valid:
    for process_id, messages in report.items():
        if messages:
            print(f"\n{process_id}:")
            for msg in messages:
                print(f"  - {msg}")
```

### Validating All Processes in a Dataset

```python
import bpbench as bp

dataset = bp.serialization.load_dataset("data.json")

all_valid = True
for case_id, cs in dataset.case_studies.items():
    is_valid, report = bp.validate_case_study(cs)
    if not is_valid:
        all_valid = False
        print(f"\nCase study '{case_id}' has issues:")
        for pid, msgs in report.items():
            for msg in msgs:
                print(f"  [{pid}] {msg}")

if all_valid:
    print("All processes valid.")
```

### Common Failure Modes

| Error | Cause | Fix |
|-------|-------|-----|
| "values must be >= 0" on a FeedVolumeChange | Negative values in feed stream | Check sign convention; feeds should be non-negative |
| "biomass not found in reactor medium" | Missing or misspelled biomass component | Add a component named `"biomass"` to `reactor_medium.components` |
| "times and values must have the same length" | Array length mismatch | Ensure measurement arrays are aligned |
| "feed medium missing component X" | Feed doesn't define all reactor species | Add the missing species to the `FeedMedium` (concentration can be 0) |

## See Also

- [Data Model](02_data_model.md) -- the structures being validated
- [Serialization](03_serialization.md) -- validate after loading
- [Design Rationale](01_design_rationale.md#6-validation-first-approach) -- why validation-first
