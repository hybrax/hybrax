# Gallery: Pseudobatch splines through a jump

Recovers a smooth concentration curve from 5 noisy measurements straddling a
discrete feed jump, checked against a known closed-form ground truth.
`hybrax.format` only: no reaction module, no training.

Narrated version: `docs/source/gallery/pseudobatch_splines.md`.

## Run

```bash
python run.py
```

No JAX training involved — runs in well under a second. Writes
`out/recovery.png` and prints the pre-/post-jump relative recovery error.
