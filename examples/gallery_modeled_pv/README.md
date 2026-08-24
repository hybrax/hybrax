# Gallery: A modeled process variable

`glyco_frac` is declared as a modeled (uncontrolled) process variable, with
its own trained rate `r_glyco_frac`, alongside the ordinary modeled reactor
component `biomass`. One large dilution bolus lands midway through each run:
`biomass` concentration steps down, `glyco_frac` does not, because
`hybrax.format` never applies a feed/dilution term to process-variable
states.

Narrated version: `docs/source/gallery/modeled_pv.md`.

## Run

```bash
python run.py
```

Prints the assembled right-hand side, R², the forward plot path, and the
fitted vs. true rates. Output lands in `run/`.
