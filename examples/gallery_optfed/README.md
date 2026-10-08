# Gallery: OptFed mechanistic model

Fit eleven kinetic parameters directly to published biomass, glycerol, and
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
The bundled `data.json` contains twelve experimental processes from OptFed,
with 83 retained measurement rows. It uses kg for reactor mass and feeds,
g/kg for concentrations, and Kelvin for temperature. The authors' excluded
measurement at row 40 of `sampling_points.csv` is omitted from the targets.
Withdrawals are individual reactor-mass drops, not cumulative sample amounts.

Regenerate the dataset with:

```bash
uv run python examples/gallery_optfed/import_data.py
```

The importer downloads four CSVs from OptFed commit
`7811f7b9ef76d624f53b852c03d06cb1b5cc96a3`. Training uses the bundled JSON and
does not require network access. Nine processes use feed-implied uptake, and
`DoE1_R1`, `DoE2_R1`, and `DoE3_R2` use kinetic uptake.

The gallery also overlays the original global reduced-model predictions from
OptFed's rate fit at `alpha=0.2`. `original_predictions.csv` contains 1,000
trajectory points per process; `original_sampling_predictions.csv` contains
the predictions at the 83 retained measurement times. These reference snapshots
were exported from the authors' `dfs_CV_plot.pickle[0.2][7]` and
`dfs_CV.pickle[0.2][7]` outputs from the same pinned analysis code. They use
hours and g/kg, matching the published mass-based measurements. They are
reference predictions, not concentration targets or Hybrax-generated fits.
