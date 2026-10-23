# Bundled OptFed checkpoint

Selected step 200 from the completed 200-update fit with linear learning-rate
decay from 0.05 to 0.0001. Training concentration NRMSE was 0.0817740066.
The prepared data and model implementation are bundled for portable loading.

The fit used the current `data.json` and `custom.py` in this example and is the
run shown on the OptFed gallery page.

From `examples/gallery_optfed/`, run:

```bash
hybrax forward --config forward-config.json --output-dir run/forward
```

`checkpoints/latest` points to the selected checkpoint. Running training with
`--overwrite` replaces this bundled run with a new fit.
