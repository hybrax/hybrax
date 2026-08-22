# Gallery: A KAN model

A Kolmogorov-Arnold Network as the reaction module: every edge carries its
own learnable univariate function (SiLU base + Gaussian radial-basis
expansion), summed at each node, in place of an MLP's fixed activation.

Narrated version: `docs/source/gallery/kan.md`.

## Run

```bash
python run.py
```

Prints R² and writes `out/edges.png` (the most informative learned edge
curve per input species). Output lands in `run/`.
