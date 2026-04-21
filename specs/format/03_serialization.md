# Serialization

Source: `bp_format/serialization.py`

## Purpose

Provides JSON-based save/load for the full bp-format data hierarchy. Handles bidirectional conversion between Python dataclass objects and JSON-serializable dicts, including special handling for JAX arrays, `NaN`/`Inf` values, `Interpolator` objects, and `TimeSeries` with optional spline state. Supports both plain `.json` and gzipped `.json.gz` files.

## Design Rationale

- **Why JSON?** Human-readable, language-agnostic, git-diffable, and requires no binary format lock-in. An earlier YAML + HDF5 approach was abandoned in favor of single-file JSON for simplicity and portability.
- **Why separate dataset and collection APIs?** `BenchmarkDataset` includes the full `CaseStudy` hierarchy with metadata (organism, citation). `BioProcessCollection` is a simpler wrapper for intermediate work (e.g., loading a single case study's processes before assembling into a dataset).
- **Smart path resolution:** Functions accept either a `.json`/`.json.gz` file path or a directory path. When given a directory, they write `data.json` inside it and load `data.json` or `data.json.gz`.

## Public API

### High-Level Functions

| Function | Description |
|----------|-------------|
| `save_dataset(dataset, path)` | Save a `BenchmarkDataset` to JSON. |
| `load_dataset(path) -> BenchmarkDataset` | Load a `BenchmarkDataset` from JSON. |
| `save_process_collection(collection, path)` | Save a `BioProcessCollection` to JSON. |
| `load_process_collection(path) -> BioProcessCollection` | Load a `BioProcessCollection` from JSON. |

### Lower-Level Functions

| Function | Description |
|----------|-------------|
| `save_dataset_json(dataset, json_path)` | Save directly to a specific `.json` or `.json.gz` file path. |
| `load_dataset_json(json_path) -> BenchmarkDataset` | Load directly from a specific `.json` or `.json.gz` file path. |
| `save_process_collection_json(collection, json_path)` | Save collection to a specific `.json` or `.json.gz` file. |
| `load_process_collection_json(json_path) -> BioProcessCollection` | Load collection from a specific `.json` or `.json.gz` file. |

### Path Resolution

```python
# These are equivalent:
bp.serialization.save_dataset(dataset, Path("output/"))          # writes output/data.json
bp.serialization.save_dataset(dataset, Path("output/data.json")) # writes output/data.json
bp.serialization.save_dataset(dataset, Path("output/custom.json"))  # writes output/custom.json
bp.serialization.save_dataset(dataset, Path("output/data.json.gz")) # writes gzipped JSON
```

### JSON Structure Overview

The JSON file follows the dataclass hierarchy directly:

```json
{
  "metadata": {"name": "bp-format v1", "version": "1.0"},
  "case_studies": {
    "kittler_2022": {
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
                "is_intracellular": false
              }
            }
          },
          "volume": {
            "initial_volume": 1.0,
            "unit": "L",
            "volume_changes": {}
          },
          "process_variables": {}
        }
      }
    }
  }
}
```

**TimeSeries payloads** include `times` and `values` arrays. If spline state is present, `breaks`, `coeffs`, and `segment_start_piece_idx` are also included.

**StaticVariable payloads** are represented as `{"type": "StaticVariable", "value": 500.0}`.

**Interpolator objects** are serialized with all their fields (kind, x, y, coefficients, metadata, etc.), with JAX arrays converted to plain lists.

## Examples

### Saving and Loading a Dataset

```python
import bp_format as bp
from pathlib import Path

# Save
bp.serialization.save_dataset(dataset, Path("output/"))

# Load
dataset = bp.serialization.load_dataset(Path("output/"))

# Access data
for case_id, cs in dataset.case_studies.items():
    print(f"{case_id}: {len(cs.processes)} processes, organism={cs.organism}")
```

### Saving and Loading a Process Collection

```python
import bp_format as bp
from pathlib import Path

# Useful for intermediate work before assembling a full dataset
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
    bp.serialization.save_dataset(dataset, path)
    restored = bp.serialization.load_dataset(path)

    # Verify structure preserved
    assert set(restored.case_studies.keys()) == set(dataset.case_studies.keys())
    for case_id in dataset.case_studies:
        orig = dataset.case_studies[case_id]
        rest = restored.case_studies[case_id]
        assert set(orig.processes.keys()) == set(rest.processes.keys())
```

## See Also

- [Data Model](02_data_model.md) -- the dataclass structures being serialized
- [TimeSeries](06_time_series.md) -- serialization of discrete samples and spline state
