# Gallery: PLS-dFBA

Extends FBA-Hyb's surrogate with a PLS-inspired low-rank linear map that reads a
controlled process variable (`media_blend_fraction`) alongside state. Its weights
are trained against concentration trajectories, rather than selected using the
covariance criterion of classical PLS regression.

Narrated version: `docs/source/gallery/pls_dfba.md`.

## Run

```bash
python run.py
```

Prints R² for all four targets (biomass, glucose, acetate, succinate).
Output lands in `run/`.
