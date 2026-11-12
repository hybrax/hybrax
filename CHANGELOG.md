# Changelog

All notable changes to hybrax are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, minor releases can break the
public API; such entries are marked **Breaking**.

## [Unreleased]

### Added

- `train.workers` spreads the per-process solves of each training update over
  worker processes; the default of 1 keeps training in one process. More workers
  give faster updates for more memory and a few seconds of start-up. On the
  12-process OptFed example with `Kvaerno5`, 50 updates took 87 s with one worker,
  67 s with two and 59 s with four, at a peak of 1.0, 1.6 and 2.8 GiB. Workers are
  CPU-only. With an implicit solver, results differ from `workers: 1` within the
  solver tolerances.
- `loo.workers_per_fold` sets the worker count of each fold's training.

### Changed

- The ODE right-hand side is traced once per solve instead of once per place the
  solver calls it. This cut the first-update compile time by 16% on the OptFed
  gallery example and by 40% on the HEK293 LSTM example. Results were
  bit-identical in our tests with explicit solvers. With implicit solvers, they
  change within the solver tolerance and updates can be about 1% slower: on OptFed (Kvaerno5, rtol 1e-4)
  the first loss moved by 4e-5 relative, and a run breaks even after about 700
  updates.
- **Breaking:** jax is now supported up to 0.12 (`jax>=0.9,<0.13`). jax 0.12 wheels
  need a CPU with AVX2 and FMA; on older CPUs install an earlier jax.
- **Breaking:** LOO folds now train in one process each unless `loo.workers_per_fold`
  says otherwise. Before, a fold used `n_cpu // parallel_folds` CPU devices by
  default, so LOO runs with few parallel folds can be slower than before until you
  set `workers_per_fold`. Each extra worker costs memory in every concurrent fold.
- cloudpickle is a new dependency. It sends the prepared model to the workers.

### Removed

- **Breaking:** `train.devices`, `loo.devices_per_fold`, the `HYBRAX_TRAIN_DEVICES`
  and `HYBRAX_GSPMD` environment variables, and the `"max"` device count. Training
  no longer shards the batch over several JAX CPU devices. Use `train.workers` and
  `loo.workers_per_fold` instead. Saved run directories that record `devices` still
  load.

### Fixed

- Training with an implicit solver such as `Kvaerno5` on more than one core no
  longer fails. This also fixes `hybrax loo` runs that failed with such models
  because folds used several CPU devices by default.

## [0.2.0] - 2026-11-12

### Added

- `metrics.csv` and `metrics.jsonl` now break the holdout loss down by target
  (`holdout_per_target_loss`) and by holdout process (`holdout_per_process_loss`,
  `holdout_process_names`) at every checkpoint boundary, mirroring the training
  columns.
- `hybrax plot RUN_DIR` draws `loss_curve.png` and `grad_norm_curve.png` from a
  run's `metrics.csv`, also while training is still running. `--kind loss` or
  `--kind grad-norm` draws only one of them.
- `BioProcessMetadata.extra` is a free-form dict for recording anything about a
  process, such as induction time or operator. It is saved with the process and has
  no effect on solving or training.
- `loo.cpus_per_fold` gives each parallel LOO worker its own set of CPUs, preferring
  distinct physical cores, so concurrent folds stop competing for the same cores.
  It is off by default. On a 12-fold, 192-process benchmark with 12 parallel folds,
  `cpus_per_fold: 2` cut the run from 166 to 140 s. Results are unchanged.

### Changed

- pandas and sympy are no longer capped at the next minor version, so hybrax installs
  alongside newer releases of both.
- jax now needs at least 0.9, the oldest version hybrax is tested with. The upper
  bound (`<0.10.1`) stays.
- jaxlib is no longer a direct dependency; jax installs the matching jaxlib itself.
- Training updates spend far less time outside the compiled step: the batch is
  gathered inside the jitted step, step outputs are fetched in one transfer, and
  `metrics.csv` rows are written to an open file instead of through pandas. On a
  165-process benchmark this cut the host overhead per update from about 40 ms to
  under 1 ms (1.44x faster updates). Results are unchanged.
- `loss_curve.png` and `grad_norm_curve.png` are drawn once, when training
  finishes, instead of being redrawn at checkpoint boundaries. Training waited
  for every redraw, which took up to 15 s with many loss terms and about a fifth
  of a long run's time.
- The ODE right-hand side and save points interpolate the controls once per call
  instead of once per flow, rate and process-variable group, and each sample's
  controls are selected before its solve. On a 165-process benchmark this made
  updates 1.26x faster. Results are equal up to floating-point rounding: they
  were bit-identical in our tests except when spline and linearly interpolated
  controls are mixed and several processes are solved together. There, rounding
  differences grew to about 1e-5 relative in the training loss after 20 updates
  and about 1e-4 relative in the final holdout loss.
- The continuous-flow preflight now integrates all of a process's controls in one
  solve (flow rates, and process-variable values as the training equations use
  them), then checks each flow's integral separately. A feed-only solve could take
  much larger steps than training ever does and reject feeds that training
  integrates accurately.
- The default `train.flow_integral_relative_tolerance` is now `1e-2` (1%), up from
  `1e-3`. Training also adapts its steps to the state dynamics, so it usually takes
  shorter steps and integrates the flows more accurately than the preflight does.
- Checkpoint boundaries are cheaper. `train` and `loo` compress prepared data once
  and every checkpoint copies that file, and holdout evaluation and prediction
  exports no longer compile small kernels for each new batch size. On a 165-process
  benchmark with 47.8 MB of prepared data, each checkpoint write took 0.02 s instead
  of 2.9 s, the first holdout evaluation compiled 2 kernels instead of 29, and a
  4-epoch run took 8.6% less time. Results are unchanged, except that with a
  non-zero `AffineScaler` offset exported states can differ by floating-point
  rounding (about 2e-13 relative in our test).
- **Breaking:** a `loo` run directory now always holds `prepared.json.gz`. Before,
  it held a plain `prepared.json` when the input was plain JSON, which is what
  `hybrax prepare` writes. Only code that opens `prepared.json` in a LOO run
  directory directly breaks; hybrax itself reads either name, and existing LOO
  run directories keep their plain file and still resume.
- LOO setup is faster: the runtime producer computes control spline derivatives with
  NumPy and checks flow spline signs in batches, instead of compiling small JAX
  kernels for each control grid. On a 192-process benchmark the producer took 52 s
  instead of 169 s (837 compiles instead of 2412), and a 12-fold LOO run took 166 s
  instead of 298 s. Runtime artifacts and fold results are byte-identical.

### Removed

- **Breaking:** the `rebuild_count` column of `metrics.csv` and `metrics.jsonl`,
  and the `train_step_rebuild_count` and `train_step_input_signature` fields of
  `TrainHarnessResult`. The train step's input shapes are fixed, so it was never
  rebuilt and these were always 0 or constant.

## [0.1.0] - 2026-10-29

First release on PyPI.

[Unreleased]: https://github.com/hybrax/hybrax/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/hybrax/hybrax/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/hybrax/hybrax/releases/tag/v0.1.0
