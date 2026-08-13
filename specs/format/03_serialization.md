# Serialization

Source: `bp_format/serialization.py`, `bp_format/json_io.py`

## Purpose

Save and load the whole bp-format hierarchy as JSON, including JAX arrays and
fitted spline state. Plain `.json` and gzipped `.json.gz` are both supported.

## Why JSON

Human-readable, diffable in git, language-agnostic, and no binary format to lock
the data in. Datasets are small enough (thousands of samples, not gigabytes)
that a text format costs nothing meaningful; `.json.gz` handles the dense online
traces.

## Public API

| Function | Description |
|----------|-------------|
| `save_process_collection(collection, path)` | Write a `BioProcessCollection`. |
| `load_process_collection(path) -> BioProcessCollection` | Read a `BioProcessCollection`. |

One save/load pair for the top-level type — there are no `*_json` variants.

### Path resolution

`path` may be a file or a directory.

```python
save_process_collection(cs, Path("output/"))              # -> output/data.json
save_process_collection(cs, Path("output/data.json"))     # -> output/data.json
save_process_collection(cs, Path("output/custom.json"))   # -> output/custom.json
save_process_collection(cs, Path("output/data.json.gz"))  # -> gzipped
```

When loading from a directory, `data.json` is tried first, then `data.json.gz`.
Anything that is neither `.json` nor `.json.gz` raises `FileNotFoundError` with
an explicit message.

### JSON parsing and comments

All reads use ijson's YAJL backend with `allow_comments=True`, so whole-line and
inline `//` comments and `/* ... */` block comments are accepted directly. No
comment-stripping preprocessing is used. `load_process_collection` streams the
top-level scalar/metadata fields first, then restores and constructs each
process one at a time, to limit peak memory. The complete document is
consumed, so malformed suffixes and trailing garbage fail.

Bare `NaN`, `Infinity`, and `-Infinity` tokens are invalid and rejected. Writers
always emit valid JSON, replacing non-finite floating values with `null`.

## JSON structure

The file mirrors the dataclass hierarchy directly.

```json
{
  "case_id": "kittler_2022",
  "organism": "E. coli",
  "citation": "Kittler et al., 2022",
  "processes": {
    "run_1": {
      "metadata":  {"name": "run_1", "process_type": "fed_batch", "notes": null},
      "time_axis": {"unit": "h", "start": 0.0, "end": 24.0,
                    "time_reference": "inoculation"},
      "reactor_medium": {
        "name": "medium", "density": 1.0, "density_unit": "kg/L",
        "components": {
          "biomass": {
            "name": "biomass",
            "unit": "g/L",
            "concentration": { "type": "TimeSeries", "...": "see below" }
          }
        }
      },
      "volume": {
        "initial_volume": 1.0,
        "unit": "L",
        "volume_changes": { "feed": { "type": "Inflow", "...": "..." } }
      },
      "process_variables": {},
      "biological_ode": { "...": "see below" }
    }
  }
}
```

`case_id`/`organism`/`citation` are omitted from the JSON entirely when unset
(rather than written as `null`) — a loose, non-case-study collection's file
has only `metadata` and `processes` at the top level.

### Arrays

Every numeric array is written by `NumpyEncoder` as a tagged object, **not** a
bare list:

```json
{"__ndarray__": [0.0, 6.0, 12.0], "dtype": "float64"}
```

On load, `_restore_arrays` walks the tree and rebuilds `jnp` arrays. Floating
dtypes are always restored as float64, so an older float32 payload loads
correctly and is upcast once. A `null` inside a typed floating array reconstructs
as NaN; `null` is invalid inside typed integer and boolean arrays. Outside typed
floating arrays, `null` remains Python `None`.

### `TimeSeries` payloads

```json
"concentration": {
  "type": "TimeSeries",
  "times":  {"__ndarray__": [0.0, 6.0, 12.0], "dtype": "float64"},
  "values": {"__ndarray__": [0.5, 1.2, 3.1],  "dtype": "float64"},
  "derived": false,
  "jump_times": {"__ndarray__": [], "dtype": "float64"},
  "continuity_side": "right",
  "breaks": {"__ndarray__": "...", "dtype": "float64"},
  "coeffs": {"__ndarray__": "...", "dtype": "float64"},
  "segment_start_piece_idx": {"__ndarray__": [0], "dtype": "int32"},
  "metadata": {"fit_strategy": "smoothing_bspline"}
}
```

- `breaks` / `coeffs` / `segment_start_piece_idx` appear only when a spline has
  been fitted. All three go together.
- `times` and `values` may be absent for a spline-only series.
- The `"type": "TimeSeries"` tag is present only where the field could also hold
  a `StaticVariable` (component concentrations, process-variable values).
  Fields that are always a `TimeSeries` — volume-change `values` and
  `total_volume` — omit it.

### `StaticVariable` payloads

```json
{"type": "StaticVariable", "value": 500.0}
```

### `VolumeChange` payloads

Discriminated by `"type"`:

```json
"feed": {
  "type": "Inflow",
  "name": "feed", "unit": "L",
  "is_controlled": true, "is_continuous": true,
  "values": {"times": "...", "values": "..."},
  "feed_medium": {"name": "...", "density": 1.0, "density_unit": "kg/L",
                  "components": {"glucose": {"name": "glucose", "unit": "g/L",
                                             "is_controlled": false,
                                             "concentration": {"type": "StaticVariable",
                                                               "value": 500.0}}}}
}
```

`Outflow` is the same minus `feed_medium`. A payload with no `"type"`
key is rejected as an old schema.

### `bounds` payloads

```json
"bounds": {"lower": 0.0, "upper": null}
```

Present on `ReactorMediumComponent`, `ProcessVariable`, and `Volume`. The RMC
default `(0.0, None)` is omitted; a missing RMC `bounds` key loads with that
default. An explicitly unbounded RMC is written as `"bounds": null` and reloads
as `(None, None)`. The `(None, None)` default for process variables and volume is
also omitted; their missing `bounds` keys load as unbounded.

### `biological_ode` payload

```json
"biological_ode": {
  "algebraic": {
    "X_active": "biomass - product"
  },
  "rates": {
    "q_growth":  {"lower": 0.0,  "upper": null},
    "q_product": null,
    "q_glucose": {"lower": null, "upper": 0.0}
  },
  "derivatives": {
    "biomass": "(q_growth + q_product) * X_active",
    "product": "q_product * X_active",
    "glucose": "q_glucose * X_active",
    "pH":      "0"
  }
}
```

Each `rates` value is a bounds object, or `null` for unbounded. Key order in
`rates` is the rate-vector layout and is preserved by JSON object ordering.

`derivatives` keys are dynamic-state names. `"0"` is the canonical way to say
"no biological dynamics"; omitting a state is rejected by
`validate_biological_ode`.

The block is always written, because `BioProcess.__post_init__` auto-fills it.

### `AugmentedBioProcess` payload

Everything a `BioProcess` has, plus:

```json
"__type__": "AugmentedBioProcess",
"parent_process": "run_1"
```

Loaders switch on `__type__`; entries without it become plain `BioProcess`. A
payload tagged augmented but missing a non-empty `parent_process` string raises.

## Rejected legacy payloads

Loading fails loudly rather than guessing, for:

| Payload | Message |
|---------|---------|
| Sibling `"interpolator"` object on a component, PV, or volume change | Regenerate with TimeSeries-only spline storage |
| `VolumeChange` with no `"type"` key | Old schema; regenerate the dataset |

## Examples

### Round trip

```python
import bp_format as bp
from pathlib import Path

bp.serialization.save_process_collection(case_study, Path("output/"))
restored = bp.serialization.load_process_collection(Path("output/"))

assert restored.case_id == case_study.case_id
assert set(restored.processes) == set(case_study.processes)
```

### Intermediate data

```python
collection = bp.BioProcessCollection(
    metadata={"source": "preprocessing", "date": "2026-07-01"},
    processes={"run_1": process_1, "run_2": process_2},
)
bp.serialization.save_process_collection(collection, Path("intermediate/"))
restored = bp.serialization.load_process_collection(Path("intermediate/"))
```

## See also

- [Data Model](02_data_model.md) — the structures being serialized
- [TimeSeries](06_time_series.md) — sample and spline storage
- [Validation](04_validation.md) — run after loading
