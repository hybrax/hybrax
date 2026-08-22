# Gallery: Knowledge transfer

Pools data from several products to help a data-poor new one: a one-hot
product-identity controlled PV, and an ensemble of GPs anchored to real
training data. Compares "local" (trained on the new product's 2 runs alone)
against "pooled" (plus 24 historical runs), evaluated on 2 held-out runs
outside the new product's own training range.

Narrated version: `docs/source/gallery/knowledge_transfer.md`.

## Run

```bash
python run.py
```

Prints held-out R² for both variants (local should fail badly on biomass;
pooled should recover) and writes `out/local_vs_pooled.png`.
