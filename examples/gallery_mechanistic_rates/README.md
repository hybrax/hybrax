# Gallery: Mechanistic models

Monod growth + Luedeking-Piret product formation as the reaction module,
instead of an MLP: named, trainable, log-parameterized (for positivity)
kinetic constants.

Narrated version: `docs/source/gallery/mechanistic_rates.md`.

## Run

```bash
python run.py
```

Prints fitted vs. true kinetic constants (some recover well, some trade off
against each other — see the narrated version for why). Output lands in
`run/`.
