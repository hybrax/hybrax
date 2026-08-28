# Inspection and Visualization

Source: `src/hybrax/format/inspect.py`

## Purpose

Look at what you have. These functions print and plot; they are for notebooks
and quick checks, not for production pipelines. Plotting needs matplotlib
(`pip install -e ".[plotting]"`).

## Text output

### `print_process_structure(process, verbosity=3)`

A hierarchical view of one `BioProcess`.

| Verbosity | Shows |
|-----------|-------|
| 1 | Which variables exist — component, process-variable, and volume-change names |
| 2 | Adds data types and sizes |
| 3 | Adds units, value ranges, and spline status (default) |

### `print_collection_structure(collection, verbosity=3)`

`case_id`, organism, citation, and a per-process summary with data-point counts
when `case_id` is set (a case study); otherwise a loose-collection summary
(process count and `metadata` keys).

### `print_rhs_ode(target, ordering=None)`

The most useful one for modeling. Renders the mechanistic ODE as a single ASCII
box, with sub-tables for:

- **Algebraic** — name and expression, in evaluation (topological) order
- **Rates** — name, lower bound, upper bound, in declaration order (this *is*
  `name_modeled_rates`, the layout of the rate vector you must supply)
- **Derivatives** — per state: unit, the biological expression verbatim from
  `reaction_ode.derivatives`, and separately the `+ feed(...)`,
  `− dilution(...)`, and `+ retention(...)` terms hybrax.format adds on top
- **Volume** — additions from feeds, removals from samples

Accepts a `BioProcess` or a `BioProcessCollection`. For a container it first
runs
[`validate_reaction_ode_equivalence`](04_validation.md#validate_reaction_ode_equivalencecontainer)
and raises `ValueError` if the processes do not share the same
`reaction_ode` — printing one process's ODE as if it described all of them
would be misleading. The title names the container, not the process that was
rendered.

Splitting biological from physical terms is the point: it shows exactly which
part of `dc/dt` you wrote and which part came from the volume machinery.

## Plotting

### `plot_process(process, figsize_per_panel=(5, 3), save_path=None, show=True)`

One panel per variable — reactor components, process variables, total volume —
with discrete samples as markers and any fitted spline drawn through them.
Pseudobatch-transformed species are shown backtransformed into real space, so
the curve is directly comparable to the measurements.

Returns the matplotlib figure.

### `plot_collection(collection, figsize_per_panel=(5, 3), save_path=None, show=True)`

A grid: one column per process, one row per variable. The fastest way to spot a
run whose units, scale, or sampling schedule differ from the rest.

Returns the matplotlib figure.

Both take `figsize_per_panel` as `(width, height)` in inches per panel, and
write the figure to `save_path` if given. With `show=True`, they return the
matplotlib figure. With `show=False`, they save first when requested, close the
figure immediately, and return `None`.

## Examples

```python
import hybrax.format as bp

collection = bp.serialization.load_process_collection("data.json")
process = collection.processes["run_1"]

bp.print_process_structure(process, verbosity=1)
bp.print_collection_structure(collection)

# what will actually be integrated
bp.print_rhs_ode(collection)

fig = bp.plot_process(process)
fig.savefig("run_1.png", dpi=150, bbox_inches="tight")

bp.plot_collection(collection, figsize_per_panel=(4, 2.5),
                   save_path="overview.png")
```

`print_rhs_ode` output for a two-process study with an intracellular product, a
continuous glucose feed, and discrete sampling:

```
+---------------------- RhsOde Structure: demo_2026 (2 processes) ----------------------+
| Algebraic                                                                             |
+---------------------------------------------------------------------------------------+
| Name     | Expression                                                                 |
| X_active | biomass - product                                                          |
+---------------------------------------------------------------------------------------+
| Rates (declaration order — this is `name_modeled_rates`)                              |
+---------------------------------------------------------------------------------------+
| Name      | Lower |                                                             Upper |
| q_growth  |     0 |                                                                 — |
| q_product |     — |                                                                 — |
| q_glucose |     — |                                                                 0 |
+---------------------------------------------------------------------------------------+
| Derivatives                                                                           |
+---------------------------------------------------------------------------------------+
| State   | Unit  | Reaction                          | Feed         | Dilution         |
| biomass | [g/L] | (q_growth + q_product) * X_active |              | − dilution(feed) |
| glucose | [g/L] | q_glucose * X_active              | + feed(feed) | − dilution(feed) |
| product | [g/L] | q_product * X_active              |              | − dilution(feed) |
+---------------------------------------------------------------------------------------+
| Volume                                                                                |
+---------------------------------------------------------------------------------------+
| State | Unit | Additions | Removals                                                   |
| V     | [L]  | feed      | − sample(sampling)                                         |
+---------------------------------------------------------------------------------------+
```

Read it as: only `glucose` gets a `+ feed(...)` term because it is the only
species with a non-zero concentration in the feed medium; every reactor species
is diluted by the continuous feed; the discrete sampling events appear under
Volume rather than Dilution because they are applied as state jumps, not as a
continuous flow. For a continuous `Outflow` with component retention,
`print_rhs_ode` also shows the corresponding `+ retention(...)` term separately.

## See also

- [Data Model](02_data_model.md) — what is being inspected
- [Mechanistic](08_mechanistic.md) — the ODE `print_rhs_ode` renders
- [Splines](07_splines.md) — the fits `plot_process` draws
