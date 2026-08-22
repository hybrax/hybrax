# Gallery: Custom losses on the dense grid

A loss module (`custom.py`) that constrains the trajectory *between*
measurements: state and rate bounds hinges plus a smoothness penalty, all
evaluated on a dense grid. Compares against `base.py` (Tutorial 4's
unconstrained loss, same data/seed/epochs).

Narrated version: `docs/source/gallery/dense_loss.md`.

## Run

```bash
python run.py
```

Prints R², worst glucose excursion, and rate curvature for both runs. Output
lands in `run_full/` and `run_base/`.
