# Bundled neural OptFed checkpoint

Selected step 470 from the completed 500-update fit at a constant learning rate
of 0.03, with Kvaerno5 at `rtol=1e-4`, `atol=1e-6`. Training concentration
NRMSE was 0.0732469115.
The prepared data and model implementation are bundled for portable loading.

The fit used the current `custom.py` in this example with
`../gallery_optfed/data.json` and is the run shown on the neural OptFed gallery
page.

From `examples/gallery_optfed_ffnn/`, run:

```bash
hybrax forward --config forward-config.json --output-dir run/forward
```

`checkpoints/latest` points to the selected checkpoint. Running training with
`--overwrite` replaces this bundled run with a new fit.
