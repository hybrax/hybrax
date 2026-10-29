# Changelog

All notable changes to hybrax are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, minor releases can break the
public API; such entries are marked **Breaking**.

## [Unreleased]

### Changed

- pandas and sympy are no longer capped at the next minor version, so hybrax installs
  alongside newer releases of both.
- jax now needs at least 0.9, the oldest version hybrax is tested with. The upper
  bound (`<0.10.1`) stays.
- jaxlib is no longer a direct dependency; jax installs the matching jaxlib itself.

## [0.1.0] - 2026-10-29

First release on PyPI.

[Unreleased]: https://github.com/hybrax/hybrax/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hybrax/hybrax/releases/tag/v0.1.0
