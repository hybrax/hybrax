# Gallery: OptFed with neural kinetics

Replace the kinetic rate laws of the OptFed mechanistic example with a small
neural network: two hidden layers of four `tanh` units, 59 parameters. The network
predicts a glycerol uptake capacity and two allocation fractions (maintenance,
then product). OptFed's fixed glycerol half-saturation constant, allocation order
and yields turn these into the three rates. A latent state tracks OptFed's
generation count, which is also a network input.

The data, mass balances, scales, solver and loss are those of
`examples/gallery_optfed/`; `prepare-config.json` reads
`../gallery_optfed/data.json`.

From this directory:

```bash
python run.py
```

The run uses 500 full-batch Adam epochs at a constant learning rate of `0.03`.
It saves every 10 epochs and selects the checkpoint with lowest concentration
NRMSE on the training measurements, using the scoring code in
`../gallery_optfed/run.py`. Kvaerno5 with `rtol=1e-4`, `atol=1e-6` is used for
training and evaluation. Training uses four worker processes. With four free CPU
cores, allow about 5 minutes.
Results and forward plots are written to `run/`. `run/selection.json` records the
selected checkpoint; its trajectories are in `run/selected-predictions.csv`.
The narrated example is `docs/source/gallery/optfed_neural.md`.

## Data license

This example trains on the OptFed data in `../gallery_optfed/data.json`. The OptFed
repository does not specify a license, so the files under `run/` other than
`run/README.md` and the copies of `custom.py` and `config.json` are computed from
that data and are not covered by Hybrax's MIT license; all rights remain with the
OptFed authors. See `THIRD_PARTY.md` at the repository root. The code in this
directory and those exceptions are Hybrax's own and MIT-licensed.
