# Gallery: Freezing parameters

`trainable_field()` / `frozen_field()`: splitting a reaction module into a
fixed random-projection encoder and a trainable readout head. Compares
against the same architecture with the encoder also trainable.

Narrated version: `docs/source/gallery/freezing.md`.

## Run

```bash
python run.py
```

Prints the trainable-structure report and the ~100x final-loss gap between
the frozen and unfrozen encoder. Output lands in `run/` and `run_unfrozen/`.
