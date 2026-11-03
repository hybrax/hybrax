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

The script regenerates data and configs, plots the source trajectories,
prepares the collection, and trains 2,000 epochs with constant learning rate
`1e-3` and seed 0. It evaluates Br7 every 10 epochs and keeps the best 20
checkpoints plus the final one.
It then fine-tunes the selected checkpoint with a penalty on negative
concentrations into `run-nonneg/`, as described below. Existing `prepared/`,
`run/` and `run-nonneg/` outputs are replaced.

Checkpoint selection minimizes the Br7 loss. In the main run this is the NMSE:
each physical concentration residual is divided by that state's maximum across
all nine cultures, then squared and averaged. The prescribed initial condition
is excluded; later zeros stay in. Test data and visual inspection do not guide
epoch selection. The recurrent architectures differ, so comparable accuracy to
MATLAB's saved model does not mean a reproduction. Both models can predict
negative concentrations.

`custom_nonneg.py` is a copy of `custom.py` whose loss adds one soft-penalty
term per concentration, built by Hybrax's `BoundsViolationLossModule` from the
zero lower bounds in the data. The penalty uses the same max normalization as
the fit, and `custom.nonnegativity_weight` in `train-config-nonneg.json` (10)
sets its weight relative to the fit residuals. `run.py` starts this run with
`--init-from` from the main run's selected checkpoint, which resets the
optimizer state, and trains 1,000 epochs at learning rate `1e-4`. Its selection
minimizes the Br7 loss including the penalty.

`results.py` reads the loss curves from `metrics.csv`, selects the checkpoints,
and simulates all nine cultures with each selected checkpoint. It writes
Hybrax and MATLAB NRMSE for the training, Br7 and test cultures to
`comparison-scores.csv`, and adds the fine-tuned model and the mean undershoot
below zero in `nonneg-scores.csv`. `test-cultures.png` and
`nonneg-test-cultures.png` plot biomass, mAb and glucose on the test cultures.
Before each test plot, `results.py` saves the test-culture data and
predictions to `test-predictions.npz` and then plots from that file, so the
bundled file holds exactly what `nonneg-test-cultures.png` shows: MATLAB and
both selected checkpoints. `plot_cultures.py` draws the plots and needs only
NumPy and Matplotlib, so the file can be replotted without Hybrax or the
trained models:

```python
import plot_cultures

predictions = plot_cultures.load_predictions("test-predictions.npz")
plot_cultures.plot_test_cultures(*predictions, "test-cultures.pdf")
```
Loss curves and selected epochs are written per run, with a `run-nonneg-`
prefix for the fine-tuned run.

`import_from_mat.py` converts the data, computes the 25×4 PCA matrix and the
all-nine statistics, and writes `train-config.json` and `train-config-nonneg.json`. It writes the
configs because `estimate_all_scales` only sees the training cultures. Only `Xv` is
renamed to `biomass`; other source identifiers are kept. `custom.py` defines
the 27-input, four-unit LSTM, a hidden-only 4-to-4 linear rate head, z-score
scaling, and MATLAB-normalized loss. It uses the built-in LSTM initialization
and overrides the forward pass to send only hidden state to the head. All
three weight matrices start at 0.1 times their initialized values; biases are
unchanged. The model has 532 trainable parameters.

`run.py` runs the full pipeline. To run the steps by hand, replace `<epoch>`
with the main run's selected epoch, zero-padded to five digits;
`python -c "from results import select_checkpoint; print(select_checkpoint('.'))"`
prints it after the first training run:

```bash
python import_from_mat.py
hybrax prepare --config prepare-config.json
hybrax train --config train-config.json
hybrax train --config train-config-nonneg.json --init-from run/checkpoints/step_<epoch>
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

## Data license

Files from the source repository, and files derived from them, keep its GPL-3.0
license rather than Hybrax's MIT license. A copy of the license is in
`LICENSE-GPL-3.0`. `source-data.mat` is an unmodified copy of `data/data.mat` from
the pinned commit. The Hybrax authors derived `feed.json` (from `data/data.xlsx`)
and `matlab-reference.npz` (from `hybrid_LSTM_1.mat`) as recorded in
`provenance.json`, and `import_from_mat.py` builds `data.json` from
`source-data.mat` and `feed.json`. Both training configs contain normalization
constants that `import_from_mat.py` computes from the data, so they count as
derived as well. The code in this directory is Hybrax's own and MIT-licensed.
