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

# Loading and saving

> Two functions, one JSON format that mirrors the dataclasses exactly.

## The two functions

They live on `hxf.serialization`, not on the package root.

```{code-cell} ipython3
import hybrax.format as hxf

collection = hxf.serialization.load_process_collection("../_data/out/demo_batch/data.json")
print(collection.case_id, "-", list(collection.processes))
```

`save_process_collection(coll, path)` / `load_process_collection(path)` are the whole
API: one container, `BioProcessCollection`. `case_id` / `organism` / `citation` are
optional fields on it, not a separate stricter type. Set all three (non-empty) and
it's a full case study; leave them `None` (the default) for raw or intermediate data.
Both shapes round-trip through the same two functions.

## Paths

A path may be a **file or a directory**. Given a directory, the loader looks for
`data.json`, then `data.json.gz`.

```python
hxf.serialization.load_process_collection("datasets/ecoli_study")        # → .../data.json
hxf.serialization.load_process_collection("datasets/ecoli_study/data.json.gz")
```

Gzip is decided by the `.gz` suffix. For anything above a few megabytes it is worth it: 
these files are mostly numbers and compress by an order of magnitude.

## The on-disk shape

The JSON mirrors the dataclass tree one-to-one. Three conventions make it round-trip:

**Arrays are tagged objects**, so dtype survives:

```{code-cell} ipython3
:tags: [remove-input]

import json
raw = json.loads(open("../_data/out/demo_batch/data.json").read())
ts = raw["processes"]["run_1"]["reactor_medium"]["components"]["biomass"]["concentration"]
print(json.dumps({k: (v if k != "values" else "...") for k, v in ts.items()},
                 indent=2)[:400])
```

**Unions carry a `type` discriminator.** `"type": "TimeSeries"` vs `"StaticVariable"`,
and `"Inflow"` vs `"Outflow"`:

```{code-cell} ipython3
:tags: [remove-input]

fb = json.loads(open("../_data/out/demo_fedbatch/data.json").read())
vc = fb["processes"]["fedbatch_1"]["volume"]["volume_changes"]["glucose_bolus"]
print(json.dumps({k: ("..." if k in ("values", "feed_medium") else v)
                  for k, v in vc.items()}, indent=2))
```

**Whole-line `//` comments are stripped on load**, which makes hand-written fixtures
bearable:

```json
{
  // this line is fine
  "case_id": "demo"        // this trailing one is NOT stripped
}
```

Only comments occupying a whole line are removed. Trailing comments after data, and
trailing commas, are still JSON errors.

## What the loader is strict about

The loader is deliberately lenient about **absent** keys and strict about **malformed
present** ones.

Absent and defaulted: `process_variables`, `volume_changes`, `reaction_ode`,
`discrete_events`, `bounds`. A minimal file can omit all of them.

Rejected outright:

- a `volume_changes` entry with no `"type"`: feed and sample cannot be guessed;
- a partially written `pseudobatch_transform`;
- legacy payloads that no longer exist, such as a stray `interpolator` sibling;
- a component whose concentration is already the transformed `pseudobatch_concentration`:
  the loader will not accept a transformed carrier where a raw one belongs.

That last one is the fail-fast design principle in action: silently accepting
`pseudobatch_concentration` as if it were a measured concentration would corrupt every
downstream fit with no error.

## The smallest valid file

```json
{
  "case_id": "demo",
  "organism": "E. coli",
  "citation": "unpublished",
  "processes": {
    "run_1": {
      "metadata": {"name": "run_1", "process_type": "batch"},
      "time_axis": {"unit": "h", "start": 0.0, "end": 24.0,
                    "time_reference": "inoculation"},
      "volume": {"initial_volume": 1.0, "unit": "L"},
      "reactor_medium": {
        "name": "medium", "density": 1.0, "density_unit": "kg/L",
        "components": {
          "biomass": {
            "name": "biomass", "unit": "g/L",
            "concentration": {
              "type": "TimeSeries",
              "times":  {"__ndarray__": [0.0, 12.0, 24.0], "dtype": "float64"},
              "values": {"__ndarray__": [0.5, 3.0, 9.0],   "dtype": "float64"}
            }
          }
        }
      }
    }
  }
}
```

That loads, auto-generates `q_biomass` with `d(biomass)/dt = q_biomass * biomass`, and
passes validation.

## Round-tripping

```{code-cell} ipython3
import contextlib
import io
from pathlib import Path
import numpy as np

out = Path("../_data/out/runs/roundtrip").resolve()
out.mkdir(parents=True, exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):
    hxf.serialization.save_process_collection(collection, out / "data.json")
print(f"./{(out / 'data.json').relative_to(out.parents[4])}")

again = hxf.serialization.load_process_collection(out / "data.json")
before = collection.processes["run_1"].reactor_medium.components["biomass"].concentration
after  = again.processes["run_1"].reactor_medium.components["biomass"].concentration
print("values identical:", np.array_equal(np.asarray(before.values),
                                          np.asarray(after.values)))
```

## Gotchas

- **`save_*` / `load_*` are not on the package root.** `hxf.save_process_collection` is
  an `AttributeError`; use `hxf.serialization.save_process_collection`.
- **Saving does not validate.** Call [`validate_process`](validate_and_inspect.md)
  yourself; nothing stops you writing a file with a sign-flipped feed.
- **Fitted splines are saved too.** A `TimeSeries` that carries spline coefficients
  round-trips with them, which is what makes a prepared dataset reproducible, and also
  what makes files bigger than you expect.

## See also

- [The data model](data_model.md): what the JSON is a picture of.
- [Validating and inspecting](validate_and_inspect.md): do this after loading.
- [API reference](../autoapi/hybrax/format/serialization/index).
