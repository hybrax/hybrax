# Gallery: Cross-validation

A cheap holdout check with no fold loop (Python API, `holdout_processes`),
then a full leave-one-out run via the CLI: one fold per process, the
`per_fold_holdout_sets` config schema.

Narrated version: `docs/source/gallery/loo.md`.

## Run

```bash
python run.py
```

Writes `out/holdout_check.png`, then runs full LOO and prints per-fold
holdout/train losses. Output lands in `loo_run/`.
