# bp-format Documentation

## Getting Started

**New to bp-format?** Read in this order:
1. [Design Rationale](01_design_rationale.md) -- understand the "why" behind the framework
2. [Data Model](02_data_model.md) -- learn the hierarchical data structures
3. [Serialization](03_serialization.md) -- load and save datasets
4. [Validation](04_validation.md) -- check data integrity
5. [Inspection](05_inspection.md) -- explore and plot your data

**Building models?** Continue with:
6. [TimeSeries](06_time_series.md) -- measurement container with optional spline state
7. [Splines](07_splines.md) -- pseudobatch transform and spline fitting
8. [Mechanistic](08_mechanistic.md) -- ODE RHS generation and integration
9. [Simulation](10_simulation.md) -- deterministic ground-truth simulation helpers

## Module Reference

| Module | Source | Documentation | Description |
|--------|--------|---------------|-------------|
| Data Model | `bp_format/dataclasses.py` | [02_data_model.md](02_data_model.md) | 20 hierarchical dataclasses for bioprocess data (incl. `AugmentedBioProcess` placeholder) |
| TimeSeries | `bp_format/time_series/` | [06_time_series.md](06_time_series.md) | Time-series container with optional fitted spline coefficients (eqx.Module) |
| Splines | `bp_format/splines.py` | [07_splines.md](07_splines.md) | Pseudobatch transformation and segmented spline fitting |
| Mechanistic | `bp_format/mechanistic.py` | [08_mechanistic.md](08_mechanistic.md) | JAX/Equinox ODE RHS generation and integration |
| Simulation | `bp_format/simulation.py` | [10_simulation.md](10_simulation.md) | Deterministic simulation helpers for dense truth and event CSVs |
| Serialization | `bp_format/serialization.py` | [03_serialization.md](03_serialization.md) | JSON save/load for the full data hierarchy |
| Validation | `bp_format/validate.py` | [04_validation.md](04_validation.md) | Data integrity checks (9 validators) |
| Inspection | `bp_format/inspect.py` | [05_inspection.md](05_inspection.md) | Text printing and matplotlib visualization |

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
 └─ processes: Dict[str, BioProcess]   # may include AugmentedBioProcess subclasses

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
 │         └─ concentration: TimeSeries | StaticVariable
 ├─ process_variables: Dict[str, ProcessVariable]
 │    ├─ name: str
 │    ├─ unit: str
 │    ├─ is_controlled: bool  # True for controls, False for states
 │    └─ values: TimeSeries | StaticVariable
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

AugmentedBioProcess(BioProcess)        # placeholder for synthetic variants
 └─ parent_process: str                # key of the real BioProcess in the same container

TimeSeries
 ├─ times: jnp.ndarray
 ├─ values: jnp.ndarray
 ├─ breaks / coeffs / segment_start_piece_idx (optional fitted spline state)
 ├─ metadata (optional transform / fit metadata)
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

The `examples/` directory contains case-study workflows plus focused
simulation/verification examples:

| Directory | Type | Organism | Citation / purpose |
|-----------|------|----------|--------------------|
| `01_kittler_2022/` | case study | E. coli | Kittler, S., Ebner, J., et al. (2022). Recombinant protein L: production, purification and characterization of a universal binding ligand. *Journal of Biotechnology*, 359, 108-115. |
| `02_gotsmy_2023/` | case study | E. coli | Gotsmy, M., Strobl, F., et al. (2023). Sulfate limitation increases specific plasmid DNA yield and productivity in E. coli fed-batch processes. *Microbial Cell Factories*, 22(1), 242. |
| `03_bayer_2020_a/` | case study | E. coli HMS174(DE3) | Bayer, B., Striedner, G., & Duerkop, M. (2020). Hybrid modeling and intensified DoE: an approach to accelerate upstream process characterization. *Biotechnology Journal*, 15(9), 2000121. |
| `04_bayer_2020_b/` | case study | E. coli HMS174(DE3) | Bayer, B., Striedner, G., & Duerkop, M. (2020). (same as above, variant B) |
| `05_martens_2025_a/` -- `10_martens_2025_f/` | case study | CHO (simulated) | Martens, A., Neufang, M., et al. (2025). Holistic Bioprocess Development Across Scales Using Multi-Fidelity Batch Bayesian Optimization. *arXiv preprint arXiv:2508.10970*. |
| `11_tub_2025/` | case study | V. natriegens | Unpublished data, TU Berlin, 2025. |
| `12_martens_expanded/` | case study | CHO (simulated) | Expanded Martens-inspired fed-batch case study with bolus, sampling, and mechanistic verification. |
| `13_volume_integration/` | verification | Synthetic | Pure dilution and volume-integration verification example. |
| `14_simulation_intracellular/` | simulation | Synthetic CHO | Deterministic Simulation example with intracellular product and dense event-output CSVs. |

Most literature case-study directories follow this structure:
```
<case_study>/
  00_original_data/          # raw CSV files + README
  00_data_preprocessing/     # Jupyter notebooks for data cleaning
  01_bp_format_data_single/    # single process in bp-format format (data.json)
  02_bp_format_data_all/       # all processes aggregated (data.json)
```

Focused simulation/verification examples, such as `13_volume_integration/` and
`14_simulation_intracellular/`, use smaller task-specific stage directories
instead of the full case-study layout.

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
