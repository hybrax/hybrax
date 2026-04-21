# Inspection and Visualization

Source: `bp_format/inspect.py`

## Purpose

Provides human-readable text output and matplotlib plotting for quick data exploration. These functions are designed for interactive use in notebooks and scripts, not for production pipelines.

## Public API

### Text Output

#### `print_process_structure(process, verbosity=3)`

Prints a hierarchical tree view of a BioProcess. The `verbosity` parameter controls detail level:

| Level | Shows |
|-------|-------|
| 1 | Process name, type, time range, component count |
| 2 | Level 1 + component names and units |
| 3 | Level 2 + data point counts, value ranges, spline status |

#### `print_dataset_structure(dataset, verbosity=3)`

Prints an overview of a full BenchmarkDataset: metadata, case studies, organisms, process counts, and per-process details (controlled by verbosity).

### Plotting

#### `plot_process(process, figsize_per_panel=(5, 3), save_path=None)`

Creates a matplotlib figure with one subplot per variable:
- Reactor medium components (concentrations over time)
- Volume profile (initial + cumulative changes)
- Process variables (pH, temperature, etc.)
- Spline fits (if interpolators are present, plotted alongside discrete data)

Returns the matplotlib figure object.

#### `plot_case_study(case_study, figsize_per_panel=(5, 3), save_path=None)`

Creates a grid plot comparing all processes in a case study. Each column is a process, each row is a variable. Useful for visually checking consistency across runs.

Returns the matplotlib figure object.

**Parameters for both plot functions:**
- `figsize_per_panel`: Tuple `(width, height)` in inches per subplot panel.
- `save_path`: If provided, saves the figure to this path (PNG, PDF, etc.).

## Examples

### Printing Process Structure

```python
import bp_format as bp

dataset = bp.serialization.load_dataset("data.json")
process = dataset.case_studies["kittler_2022"].processes["batch_001"]

# Quick overview
bp.print_process_structure(process, verbosity=1)
# Output:
#   batch_001 (batch) | 0.0 - 24.0 h | 2 components, 0 process variables

# Detailed view
bp.print_process_structure(process, verbosity=3)
# Output:
#   batch_001 (batch) | 0.0 - 24.0 h
#   Reactor Medium: minimal_medium (1.0 kg/L)
#     biomass [g/L]: 5 data points, range [0.5, 12.0]
#     glucose [g/L]: 5 data points, range [0.1, 20.0]
#   Volume: 1.0 L, 0 changes
#   ...
```

### Printing Dataset Structure

```python
import bp_format as bp

dataset = bp.serialization.load_dataset("data.json")
bp.print_dataset_structure(dataset, verbosity=1)
# Output:
#   bp-format Dataset: 3 case studies
#     kittler_2022 (Escherichia coli): 4 processes
#     gotsmy_2023 (Escherichia coli): 3 processes
#     bayer_2020_a (Escherichia coli HMS174(DE3)): 5 processes
```

### Plotting a Single Process

```python
import bp_format as bp

dataset = bp.serialization.load_dataset("data.json")
process = dataset.case_studies["kittler_2022"].processes["batch_001"]

fig = bp.plot_process(process)
fig.savefig("batch_001.png", dpi=150, bbox_inches="tight")
```

### Plotting a Case Study for Comparison

```python
import bp_format as bp

dataset = bp.serialization.load_dataset("data.json")
case_study = dataset.case_studies["kittler_2022"]

fig = bp.plot_case_study(case_study, figsize_per_panel=(4, 2.5))
fig.savefig("kittler_overview.png", dpi=150, bbox_inches="tight")
```

## See Also

- [Data Model](02_data_model.md) -- the structures being inspected
- [Splines](07_splines.md) -- spline fits visualized by `plot_process`
