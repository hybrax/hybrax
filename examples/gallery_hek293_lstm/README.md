# Gallery: HEK293, PCA reaction modes with an LSTM

This example implements the hybrid structure of Ramos et al. (2024),
[doi:10.1002/bit.28668](https://doi.org/10.1002/bit.28668), on the authors'
public synthetic data. It uses Hybrax's built-in LSTM reaction module instead
of the MATLAB implementation's custom recurrent cell.

Four fixed PCA reaction modes drive 25 concentration balances. The LSTM reads
25 concentrations, volume and time, all z-scored with the means and sample SDs
of all nine cultures, as in MATLAB. Training uses Br1-4 and Br9, Br7 is
validation, and Br5, Br6 and Br8 are test cultures. PCA and loss maxima also
use all nine cultures, following the public benchmark.

From this directory:

```bash
python run.py
```

The script regenerates data and config, plots the source trajectories,
prepares the collection, and trains 20,000 epochs with constant learning rate
`5e-5` and seed 0. It evaluates Br7 every 10 epochs and keeps the best 20
checkpoints plus the final one. Prepared data are not bundled into checkpoints.
Existing `prepared/` and `run/` outputs are replaced. Training takes several hours on
CPU, depending on the machine.

Checkpoint selection minimizes Br7 NMSE. Each physical concentration residual
is divided by that state's maximum across all nine cultures, then squared and
averaged. The prescribed initial condition is excluded; later zeros stay in.
There is no negative-concentration penalty. Test data and visual inspection do
not guide epoch selection. The selected epoch 19,970 has an all-25 NRMSE of
0.05207 on Br7 and 0.03492 on the test cultures, against 0.05761 and 0.03757 for
MATLAB's saved model. The recurrent architectures differ, so this shows
comparable accuracy, not a reproduction. Both models can predict negative
concentrations.

`results.py` reloads the Br7-selected checkpoint and simulates the training and
test cultures to write all three splits' Hybrax and MATLAB NRMSE to
`comparison-scores.csv`. It reuses saved Br7 predictions. It also writes loss
curves, the selected epoch, per-state scores, and a 25-panel Br7 comparison.
Grey panels indicate lower Hybrax NRMSE; white panels indicate lower or equal
MATLAB NRMSE. All states keep equal
weight in training and selection.

`import_from_mat.py` converts the data, computes the 25×4 PCA matrix and the
all-nine statistics, and writes `train-config.json`. It writes the config
because `estimate_all_scales` only sees the training cultures. Only `Xv` is
renamed to `biomass`; other source identifiers are kept. `custom.py` defines
the 27-input, four-unit LSTM, a hidden-only 4-to-4 linear rate head, z-score
scaling, and MATLAB-normalized loss. It uses the built-in LSTM initialization
and overrides the forward pass to send only hidden state to the head. All
three weight matrices start at 0.1 times their initialized values; biases are
unchanged. The model has 532 trainable parameters. `run.py` runs the full pipeline.
To run the steps by hand:

```bash
python import_from_mat.py
hybrax prepare --config prepare-config.json
hybrax train --config train-config.json
python results.py
```

The bundled `source-data.mat` and feed composition come from the
[pinned source repository](https://github.com/jrcramos/Hybrid-modeling-of-bioreactor-with-LSTM/tree/d4197ed84ba46bfb53c3e4f42cfe00f44916edb5).
`matlab-reference.npz` contains saved reference predictions, not weights used by
Hybrax. `provenance.json` records their derivation. No MATLAB installation or
network access is required to run the example.

`plot_data.py` plots all nine cultures before fitting, writing
`synthetic-overview.png` and `synthetic-all-states.png`. Run it independently
to inspect the bundled source data without starting training.
The narrated example is `docs/source/gallery/hek293_lstm.md`.
