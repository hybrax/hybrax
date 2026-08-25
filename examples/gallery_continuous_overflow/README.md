# Gallery: Continuous culture with controlled overflow

One process moves through batch, a one-hour pause, fed-batch filling, and
continuous culture. Equal prescribed feed and overflow rates hold the vessel at
1 L during the continuous phase.

The example fits the same noiseless biomass and glucose measurements with:

- a two-parameter Monod reaction module; and
- a 33-parameter neural reaction module (`1 → 4 → 4 → 1`).

Narrated version: `docs/source/gallery/continuous_overflow.md`.

## Run

```bash
python run.py
```

The script retains checkpoints at epochs 1, 50, and 200 and writes
`process.png`, `training.png`, and `results.json` alongside the run artifacts.
