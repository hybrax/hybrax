# Gallery: FBA-Hyb

A hybrid dynamic-FBA reaction module: two small MLPs predict a glucose-uptake
rate and an FBA objective from the current state; a frozen, pole-free
surrogate converts that into real metabolic rates. No LP solve ever happens
during training.

Narrated version: `docs/source/gallery/fba_hyb.md`.

## Run

```bash
python run.py
```

Prints R² and writes `out/objective_weights.png` (the model's inferred FBA
objective weights over the batch). Output lands in `run/`.

## Optional: regenerating the surrogate

`01_generate_fba_data.py` and `02_fit_surrogate.py` are the reference chain
that fit `custom.py`'s frozen `surrogate_fba` coefficients, offline, against
10,000 real pFBA solves on the bundled `e_coli_core.xml` (Orth, Fleming &
Palsson 2010). They are **not run** by `run.py` or needed to use this
example: solving 10,000 LPs takes a couple of minutes and needs `cobra`,
which this example otherwise has no dependency on. Run them yourself, from
this directory, only if you want to refit the surrogate:

```bash
python 01_generate_fba_data.py
python 02_fit_surrogate.py
```
