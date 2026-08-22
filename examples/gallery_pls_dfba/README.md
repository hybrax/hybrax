# Gallery: PLS-dFBA

Extends FBA-Hyb's surrogate with a real PLS-shaped component (linear,
low-rank latent-variable regression, no nonlinearity) that reads a
controlled process variable (`media_blend_fraction`) alongside state.

Narrated version: `docs/source/gallery/pls_dfba.md`.

## Run

```bash
python run.py
```

Prints R² for all four targets (biomass, glucose, acetate, succinate).
Output lands in `run/`.
