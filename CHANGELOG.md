# Changelog

All notable changes to hybrax are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, minor releases can break the
public API; such entries are marked **Breaking**.

## [Unreleased]

### Added

- `metrics.csv` and `metrics.jsonl` now break the holdout loss down by target
  (`holdout_per_target_loss`) and by holdout process (`holdout_per_process_loss`,
  `holdout_process_names`) at every checkpoint boundary, mirroring the training
  columns.
- `hybrax plot RUN_DIR` draws `loss_curve.png` and `grad_norm_curve.png` from a
  run's `metrics.csv`, also while training is still running. `--kind loss` or
  `--kind grad-norm` draws only one of them.

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

### Removed

- **Breaking:** the `rebuild_count` column of `metrics.csv` and `metrics.jsonl`,
  and the `train_step_rebuild_count` and `train_step_input_signature` fields of
  `TrainHarnessResult`. The train step's input shapes are fixed, so it was never
  rebuilt and these were always 0 or constant.

## [0.1.0] - 2026-10-29

First release on PyPI.

[Unreleased]: https://github.com/hybrax/hybrax/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hybrax/hybrax/releases/tag/v0.1.0
