# Gallery: Stateful models

A reaction module with its own memory: a continuous-time LSTM whose hidden
and cell state are integrated as extra ODE dimensions alongside the physical
state. Requires the explicit `allow_stateful_models: true` opt-in.

Narrated version: `docs/source/gallery/stateful.md`.

## Run

```bash
python run.py
```

First shows the deliberate `ValueError` from training without the opt-in,
then trains for real. Output lands in `run/`.
