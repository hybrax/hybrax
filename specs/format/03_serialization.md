# Serialization

Source: `bp_format/serialization.py`

## Purpose

Provides JSON-based save/load for the full bp-format data hierarchy. Handles
bidirectional conversion between Python dataclass objects and
JSON-serializable dicts, including special handling for JAX arrays,
`NaN`/`Inf` values, and `TimeSeries` with optional spline state. Supports
both plain `.json` and gzipped `.json.gz` files.

## Design Rationale

- **Why JSON?** Human-readable, language-agnostic, git-diffable, and requires no binary format lock-in.
- **Why two top-level types?** `CaseStudy` is the strict, publication-linked container (it requires `case_id`, `organism`, `citation`). `BioProcessCollection` is the loose wrapper for raw or intermediate work (a dict of processes plus optional free-form metadata). Each gets its own file.
- **One function per type:** A single `save_*` / `load_*` pair per type handles both explicit file paths and directories — there are no separate `*_json` variants.
- **Smart path resolution:** Functions accept either a `.json`/`.json.gz` file path or a directory path. When given a directory, they write `data.json` inside it and load `data.json` or `data.json.gz`.

## Public API

| Function | Description |
|----------|-------------|
| `save_case_study(case_study, path)` | Save a `CaseStudy` to JSON. |
| `load_case_study(path) -> CaseStudy` | Load a `CaseStudy` from JSON. |
| `save_process_collection(collection, path)` | Save a `BioProcessCollection` to JSON. |
| `load_process_collection(path) -> BioProcessCollection` | Load a `BioProcessCollection` from JSON. |

### Path Resolution

```python
# These are equivalent:
bp.serialization.save_case_study(case_study, Path("output/"))          # writes output/data.json
bp.serialization.save_case_study(case_study, Path("output/data.json")) # writes output/data.json
bp.serialization.save_case_study(case_study, Path("output/custom.json"))  # writes output/custom.json
bp.serialization.save_case_study(case_study, Path("output/data.json.gz")) # writes gzipped JSON
```

### JSON Structure Overview

A `CaseStudy` file follows the dataclass hierarchy directly: the strict
identity fields sit at the top level, with `processes` nested beneath. (A
`BioProcessCollection` file is the same shape but with `metadata` /
`processes` at the top level instead of the `case_id`/`organism`/`citation`
fields.)

```json
{
  "case_id": "kittler_2022",
  "organism": "S. cerevisiae",
  "citation": "Kittler et al., 2022",
  "processes": {
    "batch_001": {
      "metadata": {"name": "batch_001", "process_type": "batch"},
      "time_axis": {"unit": "h", "start": 0.0, "end": 24.0, "time_reference": "inoculation"},
      "reactor_medium": {
        "name": "...",
        "density": 1.0,
        "density_unit": "kg/L",
        "components": {
          "biomass": {
            "name": "biomass",
            "unit": "g/L",
            "concentration": {
              "type": "TimeSeries",
              "times": [0.0, 6.0, 12.0],
              "values": [0.5, 1.2, 3.1]
            },
            "c_star_concentration": {
              "type": "TimeSeries",
              "times": [0.0, 6.0, 12.0],
              "values": [0.5, 1.1, 2.8],
              "metadata": {
                "transform": {"name": "pseudo_batch", "component": "biomass"}
              }
            }
          }
        }
      },
      "volume": {
        "initial_volume": 1.0,
        "unit": "L",
        "volume_changes": {},
        "total_volume": {"times": [0.0, 24.0], "values": [1.0, 1.2]}
      },
      "process_variables": {},
      "pseudobatch_transform": {
        "adf": {"times": [0.0, 24.0], "values": [1.0, 1.2]},
        "feed_corrections": {
          "biomass": {"times": [0.0, 24.0], "values": [0.0, 0.0]}
        },
        "sample_compensation": {"times": [0.0, 24.0], "values": [1.0, 1.0]},
        "accumulated_feeds": {}
      }
    }
  }
}
```

**TimeSeries payloads** are tagged with `"type": "TimeSeries"` at the top level and carry `times`, `values`, `derived` (bool), `jump_times`, `continuity_side` (`"left"` or `"right"`), `metadata`, and `dtype`. If spline state is present, `breaks`, `coeffs`, and `segment_start_piece_idx` are also included. Raw real concentration stays in `concentration`; optional pseudobatch c* lives in `c_star_concentration`. The process-level pseudobatch bundle uses `adf`, `feed_corrections`, `sample_compensation`, and `accumulated_feeds`.

**StaticVariable payloads** are represented as `{"type": "StaticVariable", "value": 500.0}`.

**VolumeChange payloads** carry a `"type"` discriminator of either `"FeedVolumeChange"` or `"SampleVolumeChange"`, plus the common fields `name`, `unit`, `is_controlled`, `is_continuous`, `values` (a nested `TimeSeries` payload). `FeedVolumeChange` adds a nested `feed_medium` object.

**AugmentedBioProcess payloads** carry every field a `BioProcess` does, plus `"__type__": "AugmentedBioProcess"` and a `"parent_process": "<parent-key>"` entry. Loaders inspect `__type__` to reconstruct the correct subclass; entries without that tag are loaded as plain `BioProcess`.

**Bounds** are serialized as a small object `{"lower": <number-or-null>, "upper": <number-or-null>}` on `ReactorMediumComponent`, `ProcessVariable`, `Volume`, and per-rate inside a `BiologicalOde.rates` entry. The default `(None, None)` (unbounded on both sides) is **omitted** from JSON to keep payloads clean; loaders treat a missing `bounds` key as unbounded.

**`biological_ode` payload** appears at the process level when the user has defined the optional block:

```json
"biological_ode": {
  "algebraic": {
    "X_active": "biomass - product"
  },
  "rates": {
    "q_X_active": {"bounds": {"lower": 0.0, "upper": null}},
    "q_P":        {"bounds": null},
    "q_S":        {"bounds": {"lower": null, "upper": 0.0}}
  },
  "derivatives": {
    "biomass": "q_X_active * X_active + q_P * X_active",
    "product": "q_P * X_active",
    "glucose": "q_S * X_active",
    "pH":      "0"
  }
}
```

`derivatives` keys are dynamic-state names (reactor components or uncontrolled process variables); a value of `"0"` is the canonical way to declare *no biological dynamics for this state*. Omitting an entry for any dynamic state is rejected by `validate_biological_ode` so every choice is deliberate.

## Examples

### Saving and Loading a Case Study

```python
import bp_format as bp
from pathlib import Path

# Save
bp.serialization.save_case_study(case_study, Path("output/"))

# Load
case_study = bp.serialization.load_case_study(Path("output/"))

# Access data
print(f"{case_study.case_id}: {len(case_study.processes)} processes, "
      f"organism={case_study.organism}")
```

### Saving and Loading a Process Collection

```python
import bp_format as bp
from pathlib import Path

# Useful for raw or intermediate work that is not yet a full case study
collection = bp.BioProcessCollection(
    metadata={"source": "preprocessing"},
    processes={"run_1": process_1, "run_2": process_2},
)

bp.serialization.save_process_collection(collection, Path("intermediate/"))
restored = bp.serialization.load_process_collection(Path("intermediate/"))
```

### Round-Trip Verification

```python
import bp_format as bp
from pathlib import Path
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp)
    bp.serialization.save_case_study(case_study, path)
    restored = bp.serialization.load_case_study(path)

    # Verify structure preserved
    assert restored.case_id == case_study.case_id
    assert set(restored.processes.keys()) == set(case_study.processes.keys())
```

## See Also

- [Data Model](02_data_model.md) -- the dataclass structures being serialized
- [TimeSeries](06_time_series.md) -- serialization of discrete samples and spline state
