# Bundled OptFed checkpoint

Selected step 200 from the completed 200-update fit with linear learning-rate
decay from 0.05 to 0.0001. Training concentration NRMSE was 0.0811754143.
The prepared data and model implementation are bundled for portable loading.

The checkpoint predates the solver hook. Its bundled code adds `build_solver`
and its configuration removes `solver.method` to retain Kvaerno5 with the
current API. The fitted parameters and training data are unchanged.

From `examples/gallery_optfed/`, run:

```bash
hybrax forward --config forward-config.json --output-dir run/forward
```

`checkpoints/latest` points to the selected checkpoint. Running training with
`--overwrite` replaces this bundled run with a new fit.
