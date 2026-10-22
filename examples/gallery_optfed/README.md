# Gallery: OptFed mechanistic model

Fit eleven kinetic parameters directly to published biomass, glycerol, and
product concentrations. Temperature and carbon feed are controlled inputs.
All twelve processes use kinetic uptake. One latent state tracks accumulated
generations.

From this directory:

```bash
python run.py
```

The run uses 200 full-batch Adam epochs, gradient clipping at `0.05`, and linear
learning-rate decay from `0.05` to `0.0001`. It saves every 10 epochs and selects
the checkpoint with lowest concentration NRMSE on the training measurements.
The `build_solver` hook selects Kvaerno5 for training and evaluation, with
`rtol=1e-7`, `atol=1e-9`. The importer fits
monotone cubic splines to cumulative carbon and base feeds with an RMS error
budget of `1e-4 kg`, preserving both endpoints and the total feed. Temperature
uses cubic interpolation with `s=0`, matching OptFed. The temperature curve is fitted
to the full online trace, then restricted to each process interval without
refitting. Hybrax holds control values and rates at the solve-window endpoints
when solver stages query outside that window. Feed traces
are clipped to the process interval before fitting.
Concentration targets remain raw measurements.
Results and forward plots are written to `examples/gallery_optfed/run/`.
`run/selection.json` records the selected checkpoint; its trajectories are in
`run/selected-predictions.csv`. `run/predictions.csv` refers to the final epoch.
The narrated example is `docs/source/gallery/optfed.md`.
The bundled `data.json` contains twelve experimental processes from OptFed,
with 83 retained measurement rows. It uses kg for reactor mass and feeds,
g/kg for concentrations, and Kelvin for temperature. The authors' excluded
measurement at row 40 of `sampling_points.csv` is omitted from the targets.
Withdrawals are individual reactor-mass drops, not cumulative sample amounts.

Regenerate the dataset with:

```bash
python import_data.py
```

The importer downloads four CSVs from OptFed commit
`7811f7b9ef76d624f53b852c03d06cb1b5cc96a3`. Training uses the bundled JSON and
does not require network access.

The gallery also overlays the original global reduced-model predictions from
OptFed's rate fit at `alpha=0.2`. `original_predictions.csv` contains 1,000
trajectory points per process; `original_sampling_predictions.csv` contains
the predictions at the 83 retained measurement times. The dense curves match
the published
`data/all_process_predictions.csv` at the pinned OptFed commit. The
measurement-time snapshot was exported from locally archived OptFed
analysis output `dfs_CV.pickle[0.2][7]`; the corresponding dense archive
was checked against the published curves. These pickle archives are
not included in the pinned GitHub repository. They use
hours and g/kg, matching the published mass-based measurements. They are
reference predictions, not concentration targets or Hybrax-generated fits.
