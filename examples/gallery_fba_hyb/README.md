# Gallery: FBA-Hyb

A hybrid dynamic-FBA reaction module: two small MLPs predict a glucose-uptake
rate and an FBA objective from the current state; a frozen, pole-free
surrogate converts that into real metabolic rates. No LP solve ever happens
during training.

Narrated version: `docs/source/gallery/fba_hyb.md`.

## Run

```bash
uv run --extra dev python examples/gallery_fba_hyb/run.py
```

Run from the repository root. Requires uv. First regenerates
`01_obj_fba_data.csv.gz` using an isolated environment with Cobra and GLPK.
Then trains with the frozen surrogate, prints R², and writes
`out/objective_weights.png` (the model's inferred FBA objective weights over
the batch). Outputs land beside this script in `run/` and `out/`.

## Optional: regenerating the surrogate

`01_generate_fba_data.py` and `02_fit_surrogate.py` are the reference chain
that fit `custom.py`'s frozen `surrogate_fba` coefficients, offline, against
10,000 real pFBA solves on the bundled `e_coli_core.xml` (Orth, Fleming &
Palsson 2010). The generated CSV is checked in for use without running the
generator. `run.py` regenerates it, but does not refit the surrogate.
To regenerate the CSV alone, run from the repository root:

```bash
uv run examples/gallery_fba_hyb/01_generate_fba_data.py
```

The CSV is gzip-compressed; pandas reads it directly without manual extraction.
The generator declares its dependencies at the top of the script. uv manages
that environment separately from Hybrax, since Cobra requires pandas < 3.
Solving 10,000 LPs takes a couple of minutes.

To refit the surrogate against the CSV, run from this directory:

```bash
uv run --extra dev python 02_fit_surrogate.py
```

Copy the resulting `02_surrogate_body.txt` into `surrogate_fba` in both
`gallery_fba_hyb/custom.py` and `gallery_pls_dfba/custom.py` to use the new fit.
