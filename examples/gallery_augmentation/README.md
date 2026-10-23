# Gallery: Augmentation

Generates synthetic sibling processes from a single fed-batch run via
`prepare.augmentation`. `custom.py` fits a spline before augmenting (needed
for resampling) and imposes a nondecreasing `product` trajectory after default noise.
This is a hook demonstration; feeding can dilute product concentration.

Narrated version: `docs/source/gallery/augmentation.md`.

## Run

```bash
python run.py
```

Writes `prepared/augmented-data.png` and trains on the enlarged dataset.
