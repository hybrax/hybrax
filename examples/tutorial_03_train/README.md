# Tutorial 3: Train a model

Fits a hybrid ODE to `demo_batch` using every default: a 2-layer MLP reaction
module, per-target MSE loss, no scaling, Adam.

Narrated version: `docs/source/tutorials/03_train.md`.

## Run

```bash
python run.py
```

Runs `prepare` -> `train` -> `forward` via the `hybrax` CLI and prints the
per-epoch metrics. Output lands in `prepared/` and `run/` (self-contained:
`cp -r` either anywhere).
