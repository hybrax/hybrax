# Gallery: Glutamine degradation

One physical rate, `r_Gln`, declared once in `biological_ode.rates`, feeds
two derivatives at once: a sink in Gln, a source in NH4. The value is
Ulonska et al. 2018's own cited decomposition rate.

Narrated version: `docs/source/gallery/glutamine_decay.md`.

## Run

```bash
python run.py
```

Prints R² and the fitted vs. true rates, `r_Gln` included: the same number
that explains Gln's decline correctly predicts NH4's rise. Output lands in
`run/`.
