# Gallery: OptFed

A published non-competitive-inhibition Michaelis-Menten rate law with an
Eyring-equation temperature dependence, as the reaction module. Temperature
(a controlled process variable) feeds directly into the kinetics.

Narrated version: `docs/source/gallery/optfed.md`.

## Run

```bash
python run.py
```

4000 epochs. Prints pooled R² and fitted vs. true thermal optima (uptake and
production recover well; maintenance's true optimum sits outside the
training temperature range and doesn't). Output lands in `run/`.
