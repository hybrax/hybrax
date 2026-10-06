# Gallery: OptFed mechanistic model

Fit eleven kinetic parameters directly to synthetic biomass, glycerol, and
product concentrations. Temperature and carbon feed are controlled inputs.
Substrate-limited processes use feed-implied uptake. The remaining processes
use kinetic uptake. One latent state tracks accumulated generations.

From the repository root:

```bash
uv run python examples/gallery_optfed/run.py
```

The run uses 500 Adam epochs, gradient clipping, and a warmup/cosine schedule.
Results and forward plots are written to `examples/gallery_optfed/run/`.
The narrated example is `docs/source/gallery/optfed.md`.
`ground_truth.json` records the synthetic generator's parameters for reference.

Regenerate the synthetic dataset with
`uv run python docs/source/_data/generate_optfed.py`. Copy `data.json` and
`ground_truth.json` from `docs/source/_data/out/demo_optfed/` into this directory,
or run `docs/docs_rebuild.sh`, which performs that synchronization.
