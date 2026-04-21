# Design Rationale

This document explains the cross-cutting design decisions behind bp-format. Individual module docs reference these sections for context.

## 1. JAX-First Architecture

bp-format is built on [JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox) to support:

- **Automatic differentiation (AD):** Gradient-based optimization of hybrid bioprocess models requires differentiating through ODE integration, spline evaluation, and loss computation. JAX's `jax.grad` / `jax.value_and_grad` enable this end-to-end.
- **JIT compilation:** `jax.jit` (or `eqx.filter_jit`) compiles Python functions to optimized XLA code, which is critical for the repeated forward solves in training loops.
- **Vectorization:** `jax.vmap` allows batching over processes or parameter sets without manual loop code.

**Why Equinox?** Equinox provides `eqx.Module`, a frozen dataclass that registers as a JAX pytree. This means that objects like `TimeSeries`, `ControlSplines`, and `RhsOde` can be passed into and out of JIT-compiled functions without manual pytree registration. Equinox also provides `eqx.filter_jit`, which automatically separates static (non-differentiable) fields from dynamic (array) leaves.

**Constraints this introduces:**
- Array fields must be JAX arrays (`jnp.ndarray`), not plain NumPy.
- Python dicts with string keys are not natural JAX pytree leaves. The outer container classes (`BioProcess`, `CaseStudy`, etc.) use standard Python `@dataclass` rather than `eqx.Module` because they hold `Dict[str, ...]` fields that are manipulated outside the JIT boundary. Only the inner numerical objects (e.g., `TimeSeries`) need to be `eqx.Module`.
- Mutation is not allowed on `eqx.Module` instances (they are frozen). Use `eqx.tree_at` for functional updates.

## 2. Hierarchical Data Model

Bioprocess experiments are organized in a four-level hierarchy:

```
BenchmarkDataset
  -> CaseStudy        (one publication / experimental campaign)
    -> BioProcess     (one experimental run)
      -> Components   (reactor medium, volume, process variables)
```

**Why this hierarchy?**

- **BenchmarkDataset** groups multiple case studies for cross-study benchmarking. Metadata (name, version) tracks dataset identity.
- **CaseStudy** corresponds to one publication or experimental campaign. It carries `organism`, `citation`, and a `case_id`, which is the natural grouping for leave-one-process-out cross-validation.
- **BioProcess** is a single fermentation run. It contains everything needed to simulate or analyze that run: time axis, volume operations, reactor medium concentrations, and process variables.
- **Components** within a process are organized by their physical role:
  - `ReactorMedium` holds concentration time series (biomass, substrates, products).
  - `Volume` holds all volume-change operations (feeds, sampling).
  - `ProcessVariable` holds non-concentration signals (pH, temperature, dissolved oxygen, off-gas).

**Why `Dict[str, ...]` keyed by name?** String-keyed dicts provide O(1) lookup by name, produce readable JSON, and make it easy to iterate over components. Lists would require linear search and lose the semantic naming.

## 3. Volume as a First-Class Concept

Volume is not a state variable, not a control input, and not a process variable. It is its own category because:

- **Multiple operations affect it:** A single process can have continuous feeds, bolus feeds, and sampling events, each with different media compositions and schedules.
- **It interacts with the ODE differently:** In the mass balance equation, each feed stream `k` contributes a dilution term `(f_k / V) * (C_in[k,i] - c_i)` for every species `i`. Volume itself evolves as `dV/dt = sum(f_k)`.
- **It carries composition metadata:** Each `FeedVolumeChange` references a `FeedMedium` that defines what enters the reactor. Sampling (`SampleVolumeChange`) removes reactor contents at current concentrations.

The `Volume` dataclass aggregates `initial_volume` and a dict of `VolumeChange` entries (either `FeedVolumeChange` or `SampleVolumeChange`). Sign conventions are enforced: feeds are non-negative, samples are non-positive.

## 4. TimeSeries Structure

The `TimeSeries` class (an `eqx.Module`) stores measured data as `times` and `values` arrays, and optionally carries fitted spline coefficients (`breaks`, `coeffs`, `segment_start_piece_idx`) that provide a continuous-time interpolation of that data.

**Important:** `TimeSeries` itself is agnostic about whether the data it holds represents continuous or discrete (discontinuous) quantities. That semantic is determined by the parent object. For example, a `VolumeChange` with `is_continuous=True` uses its `TimeSeries` to represent a continuous flow profile, while `is_continuous=False` means the same `TimeSeries` holds discrete event data (bolus feeds, sampling). When the data represents discrete events, fitting splines to it would not be meaningful.

**Why optional spline coefficients?**

- The raw `times`/`values` are the ground truth from experiments. They are needed for loss computation and data validation.
- Spline coefficients enable continuous-time evaluation during ODE integration. Instead of interpolating at each solver step, the solver can evaluate the spline directly.
- Spline fitting (in `bp_format.splines`) populates the spline fields from discrete samples when appropriate (i.e., for continuous quantities like concentrations or continuous feed profiles).
- A `TimeSeries` can be spline-only (no discrete samples) in pseudobatch workflows where the original samples are no longer meaningful.

**Why power-basis storage (not B-spline basis)?** Power-basis polynomials `c[0]*h^3 + c[1]*h^2 + c[2]*h + c[3]` (where `h = t - t_break`) can be evaluated with Horner's method in a few multiply-adds. This is simple, fast, and maps cleanly to JAX operations. B-spline basis evaluation requires recursive knot-vector lookups that are harder to vectorize.

## 5. Pseudobatch Normalization

Fed-batch processes change volume over time, which means observed concentrations are affected by dilution from feeds in addition to biological activity. This makes it difficult to compare fed-batch and batch processes or to fit smooth splines to fed-batch concentration data.

**The pseudobatch transform** (Hesselberg-Thomsen et al., 2024) converts measured concentrations `c(t)` to pseudo-concentrations `c*(t)` that represent what the concentrations *would have been* in a batch process with the same biological activity:

```
c*(t) = c(t) * ADF(t) - feed_correction(t)
```

where:
- `ADF(t)` is the accumulative dilution factor (ratio of current to initial volume)
- `feed_correction(t)` accounts for mass added by feed streams

**Why is this central to bp-format?**

- **Smoother curves:** Pseudo-concentrations remove dilution artifacts, producing smoother time courses that are better approximated by cubic splines.
- **Fair comparison:** Models trained on pseudobatch data can be compared across batch and fed-batch processes.
- **Spline segmentation:** Even after pseudobatch transformation, bolus feed events create discontinuities. The spline fitting pipeline segments the time axis at event boundaries and fits each segment independently.

**Step interpolation for ADF/feed-term:** ADF and feed correction are piecewise-constant (they jump at discrete events). They must be evaluated with step (nearest-neighbor) interpolation, not linear interpolation, to preserve correct discontinuity behavior in the backtransform.

## 6. Validation-First Approach

Bioprocess data comes from diverse sources (different labs, instruments, conventions). Common errors include:
- Negative feed volumes (sign convention confusion)
- Missing biomass component in reactor medium
- Mismatched array lengths between times and values
- Feed media that do not define all reactor species
- Measurement times coinciding with sampling events (corrupted by volume change)

bp-format validates data early and explicitly. All validation functions return `(bool, str)` tuples, making them composable and easy to aggregate. `validate_process()` runs all checks on a single process; `validate_case_study()` adds cross-process consistency checks (e.g., all processes in a case study should have the same reactor medium components).

**Why not raise exceptions?** Returning `(bool, str)` allows callers to collect all issues in one pass and present a comprehensive report, rather than failing on the first error.

## 7. Ecosystem Vision

bp-format is currently a single package, but it is designed as the data foundation for a planned ecosystem of 6 packages:

| Package | Purpose |
|---------|---------|
| **bp-form** (current bp-format) | Data classes, I/O, validation, basic simulation |
| **bp-bench** | Pre-processed case study database |
| **bp-prep** | Web app for preprocessing raw experimental data |
| **bp-train** | Training utilities (LOO-CV, augmentation, checkpointing) |
| **bp-sim** | Data generation with design-of-experiments support |
| **bp-opt** | Post-training model optimization |

Currently, bp-format combines the functionality of both bp-form and bp-bench. The split will happen as the ecosystem matures. All downstream packages will depend on bp-form for data structures and I/O.
