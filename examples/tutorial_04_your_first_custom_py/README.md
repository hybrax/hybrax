# Tutorial 4: Custom models

`custom.py` replaces the two defaults that matter most: the reaction module
(an MLP over the modeled state) and scale estimation. Trains `demo_batch`
twice, once with each, and compares them by R² in physical space.

Narrated version: `docs/source/tutorials/04_your_first_custom_py.md`.

## Run

```bash
python run.py
```

Output lands in `run_default/` and `run_custom/` (each self-contained:
`cp -r` anywhere).
