# BPbench Documentation

## Getting Started

**New to BPbench?** Read in this order:
1. [Design Rationale](01_design_rationale.md) -- understand the "why" behind the framework
2. [Data Model](02_data_model.md) -- learn the hierarchical data structures
3. [Serialization](03_serialization.md) -- load and save datasets
4. [Validation](04_validation.md) -- check data integrity
5. [Inspection](05_inspection.md) -- explore and plot your data

**Building models?** Continue with:
6. [TimeSeries](06_time_series.md) -- measurement container with optional spline state
7. [Splines](07_splines.md) -- pseudobatch transform and spline fitting
8. [Mechanistic](08_mechanistic.md) -- ODE RHS generation and integration
9. [Utilities](09_utilities.md) -- cross-validation helpers

## Module Reference

| Module | Source | Documentation | Description |
|--------|--------|---------------|-------------|
| Data Model | `bpbench/dataclasses.py` | [02_data_model.md](02_data_model.md) | 19 hierarchical dataclasses for bioprocess data |
| TimeSeries | `bpbench/time_series/` | [06_time_series.md](06_time_series.md) | Time-series container with optional fitted spline coefficients (eqx.Module) |
| Splines | `bpbench/splines.py` | [07_splines.md](07_splines.md) | Pseudobatch transformation and segmented spline fitting |
| Mechanistic | `bpbench/mechanistic.py` | [08_mechanistic.md](08_mechanistic.md) | JAX/Equinox ODE RHS generation and integration |
| Serialization | `bpbench/serialization.py` | [03_serialization.md](03_serialization.md) | JSON save/load for the full data hierarchy |
| Validation | `bpbench/validate.py` | [04_validation.md](04_validation.md) | Data integrity checks (9 validators) |
| Inspection | `bpbench/inspect.py` | [05_inspection.md](05_inspection.md) | Text printing and matplotlib visualization |
| Utilities | `bpbench/utils.py` | [09_utilities.md](09_utilities.md) | Cross-validation helpers (LOO-CV) |

## Cross-Cutting Design

- [Design Rationale](01_design_rationale.md) -- JAX-first architecture, hierarchical model, pseudobatch normalization, ecosystem vision

## Data Structure

### Hierarchy

```
BenchmarkDataset
 ├─ case_studies: Dict[str, CaseStudy]
 └─ metadata: Dict[str, str]

CaseStudy
 ├─ case_id: str
 ├─ organism: str
 ├─ citation: str
 └─ processes: Dict[str, BioProcess]

BioProcess
 ├─ metadata: BioProcessMetadata
 │    ├─ name: str
 │    ├─ process_type: str (batch, fed_batch, continuous)
 │    └─ notes: Optional[str]
 ├─ time_axis: TimeAxis
 │    ├─ unit: str
 │    ├─ start: float
 │    ├─ end: float
 │    └─ time_reference: str
 ├─ reactor_medium: ReactorMedium
 │    ├─ name: str
 │    ├─ density: float
 │    ├─ density_unit: str
 │    └─ components: Dict[str, ReactorMediumComponent]
 │         ├─ name: str
 │         ├─ unit: str
 │         ├─ concentration: TimeSeries | StaticVariable
 │         └─ is_intracellular: bool
 ├─ process_variables: Dict[str, ProcessVariable]
 │    ├─ name: str
 │    ├─ unit: str
 │    ├─ is_controlled: bool  # True for controls, False for states
 │    ├─ values: TimeSeries | StaticVariable
 │    └─ interpolator: Optional[Interpolator]
 └─ volume: Volume
      ├─ initial_volume: float
      ├─ unit: str
      └─ volume_changes: Dict[str, VolumeChange]
           ├─ name: str
           ├─ unit: str
           ├─ is_controlled: bool
           ├─ is_continuous: bool
           ├─ feed_medium: FeedMedium
           └─ values: TimeSeries

TimeSeries
 ├─ times: jnp.ndarray
 ├─ values: jnp.ndarray
 ├─ breaks / coeffs / segment_start_piece_idx (optional fitted spline state)
 └─ canonical API only (legacy `timepoints` removed)

StaticVariable
 └─ value: float

FeedMedium
 ├─ name: str
 ├─ density: float
 ├─ density_unit: str
 └─ components: Dict[str, FeedMediumComponent]
      ├─ name: str
      ├─ unit: str
      ├─ concentration: TimeSeries | StaticVariable
      └─ is_controlled: bool
```

## Examples

The `examples/` directory contains 11 case studies with complete data preprocessing workflows:

| Directory | Organism | Citation |
|-----------|----------|----------|
| `01_kittler_2022/` | E. coli | Kittler, S., Ebner, J., et al. (2022). Recombinant protein L: production, purification and characterization of a universal binding ligand. *Journal of Biotechnology*, 359, 108-115. |
| `02_gotsmy_2023/` | E. coli | Gotsmy, M., Strobl, F., et al. (2023). Sulfate limitation increases specific plasmid DNA yield and productivity in E. coli fed-batch processes. *Microbial Cell Factories*, 22(1), 242. |
| `03_bayer_2020_a/` | E. coli HMS174(DE3) | Bayer, B., Striedner, G., & Duerkop, M. (2020). Hybrid modeling and intensified DoE: an approach to accelerate upstream process characterization. *Biotechnology Journal*, 15(9), 2000121. |
| `04_bayer_2020_b/` | E. coli HMS174(DE3) | Bayer, B., Striedner, G., & Duerkop, M. (2020). (same as above, variant B) |
| `05_martens_2025_a/` -- `10_martens_2025_f/` | CHO (simulated) | Martens, A., Neufang, M., et al. (2025). Holistic Bioprocess Development Across Scales Using Multi-Fidelity Batch Bayesian Optimization. *arXiv preprint arXiv:2508.10970*. |
| `11_tub_2025/` | V. natriegens | Unpublished data, TU Berlin, 2025. |

Each case study follows the same structure:
```
<case_study>/
  00_original_data/          # raw CSV files + README
  00_data_preprocessing/     # Jupyter notebooks for data cleaning
  01_bpbench_data_single/    # single process in BPbench format (data.json)
  02_bpbench_data_all/       # all processes aggregated (data.json)
```

The `00_combined/` directory contains cross-study workflows:
- `01_combined_dataset/` -- merging all case studies into one BenchmarkDataset
- `02_validation/` -- running validation across the full dataset
- `03_pseudobatch_splines/` -- fitting splines to all processes
- `04_spline_serialization/` -- saving spline-augmented data
- `05_mechanistic/` -- ODE RHS construction and integration

## Internal / Developer Notes

The following documents are developer-facing and not part of the user documentation:

| File | Content |
|------|---------|
| [PRD.md](PRD.md) | Product roadmap and 6-package ecosystem design |
| [TODO.md](TODO.md) | Active development priorities and known issues |
| [TIMESERIES_MIGRATION.md](TIMESERIES_MIGRATION.md) | Breaking API change: `timepoints` to `times` migration |
| [fix_discrete_jumps.md](fix_discrete_jumps.md) | Technical guide: step vs. linear interpolation for ADF |
