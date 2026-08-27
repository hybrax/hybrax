# Gallery: A KAN model

A Kolmogorov-Arnold Network as the reaction module: every edge carries its
own learnable univariate function (SiLU base + Gaussian radial-basis
expansion). Most nodes sum their incoming edges; each of the three output
rates also combines two of them multiplicatively, in place of an MLP's fixed
activation.

Narrated version: `docs/source/gallery/kan.md`.

## Run

```bash
python run.py
```

Prints R² and writes:

- `out/edges.png`: the most informative learned edge curve per input
  species, labeled with its best match against a small shape library.
- `out/edge_shapes.csv`: the same shape match for all 24 of `l1`'s edges.
- `out/equation_recovery.csv`: whether the trained model's own behavior
  (swept over made-up inputs) matches the real process that generated its
  training data.

Output lands in `run/`.
