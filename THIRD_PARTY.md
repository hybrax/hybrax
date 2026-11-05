# Third-party material

Hybrax is MIT-licensed (see `LICENSE`), except for the third-party data listed here.
These files are not covered by the MIT license; their status is given per dataset.
None of them ship in the PyPI package.

The FBA examples also use a third-party model that is not in this repository; see
the last section.

## HEK293 synthetic cultures (Ramos et al.)

- **Source:** <https://github.com/jrcramos/Hybrid-modeling-of-bioreactor-with-LSTM>,
  commit `d4197ed84ba46bfb53c3e4f42cfe00f44916edb5`. Paper:
  [doi:10.1002/bit.28668](https://doi.org/10.1002/bit.28668).
- **License:** GPL-3.0, copy in `examples/gallery_hek293_lstm/LICENSE-GPL-3.0`.
- **Files:** in `examples/gallery_hek293_lstm/`:
  - `source-data.mat`, an unmodified copy of `data/data.mat`;
  - `feed.json` and `matlab-reference.npz`, derived from `data/data.xlsx` and
    `hybrid_LSTM_1.mat` as recorded in `provenance.json`;
  - `data.json`, which `import_from_mat.py` converts from `source-data.mat` and
    `feed.json`;
  - `train-config.json` and `train-config-nonneg.json`, whose normalization
    constants `import_from_mat.py` computes from the data.

  The built docs (`docs/html/`) serve copies of `source-data.mat`, `feed.json` and
  `matlab-reference.npz` as downloads and show plots and tables computed from the data.

## OptFed fed-batch data (Kittler et al., Schlögel et al.)

- **Source:** <https://github.com/gschloegel/OptFed>, commit
  `7811f7b9ef76d624f53b852c03d06cb1b5cc96a3`, `data/`. Papers:
  [doi:10.1016/j.csbj.2024.09.024](https://doi.org/10.1016/j.csbj.2024.09.024)
  (OptFed) and
  [doi:10.1016/j.jbiotec.2022.10.002](https://doi.org/10.1016/j.jbiotec.2022.10.002)
  (the original fermentations).
- **License:** the source repository does not specify a license, so all rights
  remain with its authors.
- **Files:** in `examples/gallery_optfed/`:
  - `data.json`, which `import_data.py` converts from the source CSVs;
  - `original_predictions.csv` and `original_sampling_predictions.csv`, the original
    OptFed model's predictions, extracted from the output of the repository's
    `code/01_fit.py` (`dfs_CV_plot.pickle` and `dfs_CV.pickle`). The repository does
    not store this output itself.

  `run/` holds a training run on this data. Its files are computed from the data and
  fall under the same terms, except `run/README.md` and the copies of `custom.py` and
  `config.json`, which are Hybrax's own. The built docs (`docs/html/`) serve copies
  of `data.json` and both prediction files as downloads and show plots and tables
  computed from them.

  `examples/gallery_optfed_ffnn/` trains a neural variant of the model on the same
  `data.json`. Its `run/` falls under the same terms, with the same exceptions, and
  the built docs show plots and tables computed from the data.

## E. coli core metabolic model (BiGG Models)

`examples/gallery_fba_hyb/01_generate_fba_data.py` downloads `e_coli_core.xml` from
<https://bigg.ucsd.edu/models/e_coli_core> when it runs; the file is not part of this
repository. BiGG Models permits use for educational, research and non-profit purposes
only, while Hybrax uses the MIT license; see BiGG's
[license](https://bigg.ucsd.edu/license). Model paper:
[doi:10.1128/ecosalplus.10.2.1](https://doi.org/10.1128/ecosalplus.10.2.1).
