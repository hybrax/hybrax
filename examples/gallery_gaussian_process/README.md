# Gallery: A Gaussian-process model

A sparse GP (mean and variance via a closed-form Cholesky solve over
trainable inducing points) as the reaction module, trained end to end by
gradient descent through the ODE solve. Predictive uncertainty is exposed
through `ReactionOutputs.auxiliary`.

Narrated version: `docs/source/gallery/gaussian_process.md`.

## Run

```bash
python run.py
```

Prints R² and the fitted kernel hyperparameters, writes
`out/uncertainty.png` (predicted rate ± 2·std). Output lands in `run/`.
